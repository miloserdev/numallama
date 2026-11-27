/* Continuous batching for dense/MoE models: several sessions decode one token each in one
 * weight-reading pass (forward_sessions). Hybrid/recurrent models, GLM and -J 1 use server_serial.
 *
 * v1 scope: blocking prefill, no speculative decoding, client-side tools and server-side tools
 * (the latter are served by the serial loop once the batch drains). Each session owns a full -c
 * KV cache (SrvSession.kv); the model's global cache stays for the serial fallback. */

#include <poll.h>
#include <fcntl.h>

static void server_batch(Transformer* tr, Tokenizer* tk, Sampler* sp);
static Transformer* g_cb_tr = NULL; // the model of the batched loop (helpers run outside the dispatch)

static void server(Transformer* tr, Tokenizer* tk, Sampler* sp) {
    if (SRV_J <= 1 || tr->config.hybrid || tr->config.glm) {
        server_serial(tr, tk, sp);
        return;
    }
    server_batch(tr, tk, sp);
}

typedef struct {
    int used, fd;
    SrvReq r;
    SrvSession* S;
    ChatHistory h;
    int* P;
    JNode* req;
    Tools tools;        // request tools (kept for the whole reply)
    int tools_owned;
    int np, sys_len, reuse, max_gen, gen, ended, include_usage;
    int n_stops;
    char* stops[8];
    size_t max_stop;
    float* logits;      // (vocab) current distribution
    ReplyState rs;
    JNode* calls[MAX_CALLS_PER_REPLY];
    Sampler samp;
    int next_tok, next_pos, feed;
    int think;          // per-request reasoning setting (reply_token uses the global THINK)
    int own_tools;      // server-side tools from -f tools.json
    int rounds;         // tool-call rounds in this reply
    ReplySink sink;
    char* body;         // journal
    long t_req, t_pre, t_gen;
    volatile int hb_stop, hb_failed;
    SrvHb hb;
    pthread_t hbt;
    int hb_on;
} CbConn;

static CbConn g_cb[SPEC_MAX_TOK];

typedef struct { int fd; char shead[80]; char* body; } CbPending;
static CbPending g_pend[16];
static int g_npend;

static int cb_active(void) {
    int n = 0;
    for (int i = 0; i < SPEC_MAX_TOK; i++) n += g_cb[i].used;
    return n;
}

// the model's own cache is the one the serial fallback uses; batched sessions attach their own
static void cb_restore_global(Transformer* tr) {
    if (tr->state.kv_owned) kv_attach(&tr->state, &tr->state.own);
    else kv_detach(&tr->state);
}

static void cb_keepalive(CbConn* c) {
    if (!c->used || !c->r.stream || c->r.failed) return;
    if (send(c->fd, ": keep-alive\n\n", 14, MSG_NOSIGNAL) != 14) {
        c->r.failed = 1;
        c->sink.stop = 1;
    }
}

/* Resident per-session KV caches. An agent turn usually reuses almost the whole previous prompt,
 * so the cache should stay in memory: reallocating a full -c cache and dumping/restoring hundreds
 * of MB on every turn was the dominant server overhead. Caches are counted by the positions they
 * actually hold and evicted LRU when the RAM budget is exceeded; dumps are periodic (crash
 * recovery) and on eviction, not after every reply. */
static size_t SRV_KV_BUDGET = (size_t)8 << 30;
static long SRV_DUMP_SECS = 300;

static size_t srv_mem_available(void) {
#if defined(__linux__)
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long v;
    size_t out = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %llu", &v) == 1) { out = (size_t)v * 1024; break; }
    fclose(f);
    return out;
#else
    return 0;
#endif
}

static void cb_kv_budget_init(void) {
    const char* e = getenv("RUNQ_KV_RESIDENT_MB");
    if (e && atoi(e) > 0) {
        SRV_KV_BUDGET = (size_t)atoi(e) << 20;
    } else {
        size_t avail = srv_mem_available();
        size_t b = avail ? avail / 4 : (size_t)8 << 30;
        if (b > (size_t)32 << 30) b = (size_t)32 << 30;
        SRV_KV_BUDGET = b;
    }
    e = getenv("RUNQ_DUMP_SECS");
    if (e && atoi(e) >= 0) SRV_DUMP_SECS = atoi(e);
    fprintf(stderr, "[server] resident session KV budget: %.1f GB, dump every %ld s\n", SRV_KV_BUDGET / 1e9,
            SRV_DUMP_SECS);
}

static size_t cb_kv_bytes(const Transformer* tr, const SrvSession* S) {
    return S->kv_ready ? sess_region_bytes(&tr->config, S->kv_n, S->kv.type) : 0;
}

// write the session state to its dump; 1 = written. Skipped when the disk is nearly full.
static int cb_kv_dump(Transformer* tr, SrvSession* S) {
    if (!S->kv_ready || S->n <= 0 || S->kv_n < S->n || S == &g_fallback || !SRV_SESS_DIR) return 0;
    struct statvfs vfs;
    if (statvfs(SRV_SESS_DIR, &vfs) == 0 &&
        (uint64_t)vfs.f_bavail * vfs.f_frsize < sess_dump_bytes(tr, S->n) * 5 / 4)
        return 0;
    char path[1200];
    snprintf(path, sizeof path, "%s/%s.bin", SRV_SESS_DIR, S->id);
    kv_attach(&tr->state, &S->kv);
    long t0 = time_in_ms();
    int ok = sess_write(tr, S->tok, S->n, path);
    long ms = time_in_ms() - t0;
    S->last_dump = time(NULL);
    S->dumped_n = ok ? S->n : 0;
    fprintf(stderr, "[server] session %s: state %d tok%s in %ld ms\n", S->id, S->n, ok ? " saved" : " NOT saved", ms);
    cb_restore_global(tr);
    return ok;
}

static void cb_kv_evict(Transformer* tr, SrvSession* S) {
    if (!S->kv_ready) return;
    size_t bytes = cb_kv_bytes(tr, S);
    int dumped = cb_kv_dump(tr, S);
    kv_cache_free(&S->kv);
    S->kv_ready = 0;
    S->kv_n = 0;
    S->kv_disk_checked = dumped ? 0 : 1; // no stale dump when the save did not happen
    fprintf(stderr, "[server] session %s: resident KV evicted (%.1f MB, %s)\n", S->id, bytes / 1e6,
            dumped ? "dumped" : "not dumped");
}

// keep the total resident KV under budget; never evict `keep` or a session that is decoding now
static void cb_kv_budget(Transformer* tr, SrvSession* keep) {
    size_t keep_bytes = keep ? cb_kv_bytes(tr, keep) : 0;
    for (;;) {
        size_t total = 0;
        SrvSession* old = NULL;
        for (int i = 0; i < g_nsessions; i++) {
            SrvSession* s = &g_sessions[i];
            if (!s->kv_ready || s->busy || s == keep) continue;
            total += cb_kv_bytes(tr, s);
            if (!old || s->last_use < old->last_use) old = s;
        }
        if (total + keep_bytes <= SRV_KV_BUDGET || !old) return;
        cb_kv_evict(tr, old);
    }
}

// start streaming headers + the role chunk; called once per request before prefill
static void cb_stream_begin(CbConn* c) {
    SrvReq* r = &c->r;
    snprintf(r->id, sizeof r->id, "chatcmpl-%lx%04x", (unsigned long)time(NULL), (unsigned)(rand() & 0xffff));
    r->created = time(NULL);
    sb_puts(&r->content, "");
    sb_puts(&r->reasoning, "");
    c->sink = (ReplySink){.text = srv_text, .ctx = r, .max_gen = c->max_gen, .stop = 0};
    g_sink = &c->sink;
    if (!r->stream) return;
    const char* hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                      "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    srv_write(r, hdr, strlen(hdr));
    SBuf b = {0};
    srv_chunk_head(r, &b);
    sb_puts(&b, ", \"choices\": [{\"index\": 0, \"delta\": {\"role\": \"assistant\", \"content\": \"\"}, \"finish_reason\": null}]}\n\n");
    srv_write(r, b.buf, b.len);
    free(b.buf);
}

static void cb_free(CbConn* c, int close_fd) {
    if (c->hb_on) {
        c->hb_stop = 1;
        pthread_join(c->hbt, NULL);
        c->hb_on = 0;
    }
    if (c->S) c->S->busy = 0;
    if (c->S) c->S->live = 0;
    if (close_fd && c->fd >= 0) {
        shutdown(c->fd, SHUT_WR);
        close(c->fd);
    }
    if (c->tools_owned) json_free(c->tools.root);
    free(c->samp.seen);
    free(c->samp.freq);
    json_free(c->req);
    free(c->P);
    free(c->logits);
    free(c->body);
    free(c->r.content.buf);
    free(c->r.reasoning.buf);
    free(c->rs.buf.buf);
    g_sink = NULL;
    g_tools = NULL;
    memset(c, 0, sizeof *c);
    c->fd = -1;
}

/* the reply is over: build the OpenAI response, journal/dump the session and free the connection */
static void cb_finish(CbConn* c, Transformer* tr, Tokenizer* tk) {
    SrvReq* r = &c->r;
    g_sink = &c->sink; // send failures must stop this connection, not another session
    if (ARCH < ARCH_QWEN2 && c->rs.mode == 2) { // Llama: the buffered message may be a call
        g_sink = &c->sink;
        JNode* call = normalize_call(json_parse(c->rs.buf.buf, NULL));
        if (call) c->calls[c->rs.n_calls++] = call;
        else out_text(c->rs.buf.buf, 0);
        c->rs.mode = 1;
    }
    int n_calls = c->rs.n_calls;
    if (r->stop_hit) { // the reply was cut inside the text: no calls
        for (int i = 0; i < n_calls; i++) json_free(c->calls[i]);
        n_calls = 0;
    }
    const char* finish = n_calls ? "tool_calls" : (c->ended || r->stop_hit) ? "stop" : "length";
    if (n_calls)
        while (r->content.len > 0 && isspace((unsigned char)r->content.buf[r->content.len - 1])) r->content.buf[--r->content.len] = '\0';

    SBuf tcj = {0};
    sb_puts(&tcj, "[");
    for (int i = 0; i < n_calls; i++) {
        char cid[40];
        snprintf(cid, sizeof cid, "call_%08x%08x", (unsigned)rand(), (unsigned)rand());
        JNode* params = json_get(c->calls[i], "parameters");
        SBuf args = {0};
        if (params) json_write(&args, params, -1, 0); else sb_puts(&args, "{}");
        char hd[160];
        snprintf(hd, sizeof hd, "%s{\"index\": %d, \"id\": \"%s\", \"type\": \"function\", \"function\": {\"name\": ", i ? ", " : "", i, cid);
        sb_puts(&tcj, hd);
        json_write_string(&tcj, json_get(c->calls[i], "name")->str);
        sb_puts(&tcj, ", \"arguments\": ");
        json_write_string(&tcj, args.buf);
        sb_puts(&tcj, "}}");
        free(args.buf);
        json_free(c->calls[i]);
    }
    sb_puts(&tcj, "]");
    char usage[256];
    snprintf(usage, sizeof usage, "{\"prompt_tokens\": %d, \"completion_tokens\": %d, \"total_tokens\": %d, "
             "\"prompt_tokens_details\": {\"cached_tokens\": %d}}", c->np, c->gen, c->np + c->gen, c->reuse);
    if (!r->stream) { utf8_clean(r->content.buf); utf8_clean(r->reasoning.buf); }
    if (r->stream) {
        srv_flush(r, 1, 1);
        srv_flush(r, 0, 1);
        SBuf b = {0};
        if (n_calls) {
            srv_chunk_head(r, &b);
            sb_puts(&b, ", \"choices\": [{\"index\": 0, \"delta\": {\"tool_calls\": ");
            sb_puts(&b, tcj.buf);
            sb_puts(&b, "}, \"finish_reason\": null}]}\n\n");
        }
        srv_chunk_head(r, &b);
        sb_puts(&b, ", \"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"");
        sb_puts(&b, finish);
        sb_puts(&b, "\"}]}\n\n");
        if (c->include_usage) {
            srv_chunk_head(r, &b);
            sb_puts(&b, ", \"choices\": [], \"usage\": ");
            sb_puts(&b, usage);
            sb_puts(&b, "}\n\n");
        }
        sb_puts(&b, "data: [DONE]\n\n");
        srv_write(r, b.buf, b.len);
        free(b.buf);
    } else {
        SBuf b = {0};
        sb_puts(&b, "{\"id\": \"");
        sb_puts(&b, r->id);
        char hd[128];
        snprintf(hd, sizeof hd, "\", \"object\": \"chat.completion\", \"created\": %ld, \"model\": ", r->created);
        sb_puts(&b, hd);
        json_write_string(&b, SRV_MODEL);
        sb_puts(&b, ", \"choices\": [{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": ");
        if (n_calls && r->content.len == 0) sb_puts(&b, "null");
        else json_write_string(&b, r->content.buf);
        if (r->reasoning.len) { sb_puts(&b, ", \"reasoning_content\": "); json_write_string(&b, r->reasoning.buf); }
        if (n_calls) { sb_puts(&b, ", \"tool_calls\": "); sb_puts(&b, tcj.buf); }
        sb_puts(&b, "}, \"finish_reason\": \"");
        sb_puts(&b, finish);
        sb_puts(&b, "\"}], \"usage\": ");
        sb_puts(&b, usage);
        sb_puts(&b, "}");
        srv_respond(r->fd, 200, "application/json", b.buf);
        free(b.buf);
    }
    SrvSession* S = c->S;
    if (S) {
        S->n = c->h.n;
        S->kv_n = c->h.n;
        S->live = 1;
        long now = time(NULL);
        SRV_REQUESTS++; SRV_PROMPT_TOKENS += c->np; SRV_COMPLETION_TOKENS += c->gen; SRV_CACHE_TOKENS += c->reuse;
        fprintf(stderr, "[server] session %s: prompt %d tok (cached %d) | reply %d tok | %s%s%s"
                        " | prefill %ld ms | ttft %ld ms\n",
                S->id, c->np, c->reuse, c->gen, finish, r->stream ? ", stream" : "",
                r->failed ? ", client disconnected" : "", c->t_pre,
                r->t_first ? r->t_first - c->t_req : -1);
        if (!r->failed) srv_journal(S, c->body, c->np, c->gen, c->reuse, finish);
        if (now - S->last_dump >= SRV_DUMP_SECS) cb_kv_dump(tr, S); // crash recovery; not every turn
        cb_restore_global(g_cb_tr);
        S->kv_disk_checked = 1;
        cb_kv_budget(g_cb_tr, S); // keep the cache resident for the next turn
    }
    free(tcj.buf);
    int fd = c->fd;
    cb_free(c, 0);
    if (fd >= 0) { shutdown(fd, SHUT_WR); close(fd); }
}

// feed one generated token: returns 1 to continue (it becomes the next batched step's input)
static int cb_token(CbConn* c, Tokenizer* tk, int tok) {
    if (is_end_token(tok)) {
        if (ARCH == ARCH_GLM5) c->ended = 1; // GLM is serial; kept for completeness
        else if (c->h.n < c->h.cap) {
            kv_attach(&g_cb_tr->state, &c->S->kv);
            chat_feed(g_cb_tr, &c->h, &tok, 1); // keep the end token in the history
            c->S->kv_n = c->h.n;
            c->ended = 1;
        }
        return 0;
    }
    g_sink = &c->sink;
    g_tools = c->own_tools ? SRV_TOOLS : (c->tools.root ? &c->tools : NULL);
    reply_token(&c->rs, tk, tok);
    c->gen++;
    if (sink_done(c->gen)) return 0;
    if (c->h.n >= c->h.cap) {
        fprintf(stderr, "\n[context window full, reply truncated]");
        return 0;
    }
    c->h.tok[c->h.n++] = tok; // positions are written by the next batched step
    c->S->n = c->h.n;
    c->S->kv_n = c->h.n;
    c->next_tok = tok;
    c->next_pos = c->h.n - 1;
    c->feed = 1;
    return 1;
}

/* a reply turn ended: run server-side tools and continue, or finalize the request.
 * returns 1: the session continues (a new tool round is feeding), 0: finished. */
static int cb_after_end(CbConn* c, Transformer* tr, Tokenizer* tk) {
    if (c->own_tools && c->rs.n_calls > 0 && !c->r.stop_hit && !c->r.failed && c->rounds < 16) {
        c->rounds++;
        SBuf prefix = {0};
        sb_puts(&prefix, "");
        if (!c->ended) sb_puts(&prefix, end_text());
        sb_puts(&prefix, after_end());
        int nc = c->rs.n_calls;
        char* outs[MAX_CALLS_PER_REPLY];
        for (int i = 0; i < nc; i++) {
            char* shown = NULL;
            outs[i] = tools_run(SRV_TOOLS, c->calls[i], &shown, 1);
            fprintf(stderr, "[server] tool %s -> %zu bytes\n", json_get(c->calls[i], "name")->str, strlen(outs[i]));
            free(shown);
            json_free(c->calls[i]);
        }
        c->rs.n_calls = 0;
        int m = build_tool_results(tk, prefix.buf, outs, nc, c->P, 0);
        for (int i = 0; i < nc; i++) free(outs[i]);
        free(prefix.buf);
        if (c->r.failed || c->h.n + m + 512 >= c->h.cap) {
            fprintf(stderr, "[server] tool output does not fit the context window (%d tokens)\n", c->h.cap);
            cb_finish(c, tr, tk);
            return 0;
        }
        kv_attach(&g_cb_tr->state, &c->S->kv);
        float* logits = chat_feed(g_cb_tr, &c->h, c->P, m);
        if (!logits) { cb_finish(c, tr, tk); return 0; } // client gone mid-feed
        c->S->n = c->h.n;
        c->S->kv_n = c->h.n;
        memcpy(c->logits, logits, (size_t)tr->config.vocab_size * sizeof(float));
        free(c->rs.buf.buf);
        c->rs = (ReplyState){.mode = 0, .n_calls = 0, .token = -1, .calls = c->calls};
        sb_puts(&c->rs.buf, "");
        c->ended = 0;
        g_sink = &c->sink;
        g_tools = SRV_TOOLS;
        THINK = c->think;
        int tok = sample(&c->samp, c->logits);
        if (cb_token(c, tk, tok)) return 1;
        return cb_after_end(c, tr, tk);
    }
    if (c->own_tools) {
        for (int i = 0; i < c->rs.n_calls; i++) json_free(c->calls[i]);
        c->rs.n_calls = 0;
    }
    cb_finish(c, tr, tk);
    return 0;
}

/* parse + render + (blocking) prefill one request into a connection slot.
 * returns 1: active; 0: request answered/closed; -1: not suitable for the batch (queue serial). */
static int cb_prepare(CbConn* c, Transformer* tr, Tokenizer* tk, Sampler* sp, int fd, const char* shead, const char* body) {
    const int ctx = tr->config.seq_len;
    memset(c, 0, sizeof *c);
    c->t_req = time_in_ms();
    c->fd = fd;
    c->r.fd = fd;
    c->body = strdup(body);
    c->req = json_parse(body, NULL);
    JNode* messages = c->req ? json_get(c->req, "messages") : NULL;
    if (!c->req || c->req->type != J_OBJ || !messages || messages->type != J_ARR || !messages->child) {
        srv_error(fd, 400, "invalid_request_error", "the body must be a JSON object with a non-empty messages array");
        cb_free(c, 1);
        return 0;
    }
    JNode* v;
    if ((v = json_get(c->req, "n")) && v->type == J_NUM && atoi(v->str) != 1) {
        srv_error(fd, 400, "invalid_request_error", "only n = 1 is supported");
        cb_free(c, 1);
        return 0;
    }
    char sid[72] = "default";
    JNode* sj = json_get(c->req, "session_id");
    if (!sj) sj = json_get(c->req, "conversation_id");
    if (sj && sj->type == J_STR && sj->str[0]) snprintf(sid, sizeof sid, "%.71s", sj->str);
    else if (shead && shead[0]) snprintf(sid, sizeof sid, "%.71s", shead);
    srv_sanitize_id(sid);
    SrvSession* S = srv_session(sid, ctx);
    if (S->busy) { cb_free(c, 0); return -1; } // the same session is already decoding: queue it
    c->S = S;
    S->last_use = time(NULL);
    S->busy = 1;
    // per-request sampling / generation settings (own sampler copy and RNG state)
    c->samp = *sp;
    c->samp.seen = NULL; // the shared buffers belong to the serial sampler
    c->samp.freq = NULL;
    if ((v = json_get(c->req, "temperature")) && v->type == J_NUM) c->samp.temperature = atof(v->str);
    if ((v = json_get(c->req, "top_p")) && v->type == J_NUM) c->samp.topp = atof(v->str);
    if ((v = json_get(c->req, "min_p")) && v->type == J_NUM) c->samp.min_p = atof(v->str);
    if ((v = json_get(c->req, "presence_penalty")) && v->type == J_NUM) c->samp.presence_penalty = atof(v->str);
    if ((v = json_get(c->req, "frequency_penalty")) && v->type == J_NUM) c->samp.frequency_penalty = atof(v->str);
    if (c->samp.topp <= 0.0f || c->samp.topp > 1.0f) c->samp.topp = 1.0f;
    if (c->samp.min_p < 0.0f || c->samp.min_p > 1.0f) c->samp.min_p = 0.0f;
    if (c->samp.presence_penalty < -2.0f || c->samp.presence_penalty > 2.0f) c->samp.presence_penalty = 0.0f;
    if (c->samp.frequency_penalty < -2.0f || c->samp.frequency_penalty > 2.0f) c->samp.frequency_penalty = 0.0f;
    if (c->samp.presence_penalty != 0.0f || c->samp.frequency_penalty != 0.0f) {
        size_t V = (size_t)tr->config.vocab_size;
        c->samp.seen = calloc(V, 1);
        c->samp.freq = calloc(V, sizeof(uint16_t));
        if (!c->samp.seen || !c->samp.freq) { c->samp.presence_penalty = c->samp.frequency_penalty = 0.0f; }
    }
    if (c->samp.temperature < 0.0f) c->samp.temperature = 0.0f;
    if ((v = json_get(c->req, "seed")) && v->type == J_NUM) c->samp.rng_state = (unsigned long long)atoll(v->str) + 1;
    const int think0 = THINK;
    if ((v = json_get(c->req, "reasoning_effort")) && v->type == J_STR) THINK = strcmp(v->str, "none") && strcmp(v->str, "minimal");
    JNode* kw = json_get(c->req, "chat_template_kwargs");
    if (kw && (v = json_get(kw, "enable_thinking")) && v->type == J_BOOL) THINK = v->boolean;
    c->think = THINK;
    if ((v = json_get(c->req, "max_completion_tokens")) && v->type == J_NUM) c->max_gen = atoi(v->str);
    else if ((v = json_get(c->req, "max_tokens")) && v->type == J_NUM) c->max_gen = atoi(v->str);
    JNode* so = json_get(c->req, "stream_options");
    c->include_usage = so && (v = json_get(so, "include_usage")) && v->type == J_BOOL && v->boolean;
    c->r.stream = (v = json_get(c->req, "stream")) && v->type == J_BOOL && v->boolean;
    JNode* stop = json_get(c->req, "stop");
    if (stop && stop->type == J_STR && stop->str[0]) c->stops[c->n_stops++] = stop->str;
    else if (stop && stop->type == J_ARR)
        for (JNode* s2 = stop->child; s2 && c->n_stops < 8; s2 = s2->next)
            if (s2->type == J_STR && s2->str[0]) c->stops[c->n_stops++] = s2->str;
    c->r.stops = c->stops;
    for (int i = 0; i < c->n_stops; i++) if (strlen(c->stops[i]) > c->max_stop) c->max_stop = strlen(c->stops[i]);
    c->r.max_stop = c->max_stop;
    // tools: this path returns client tools; server-side tools (-f) go through the serial loop
    JNode* tc = json_get(c->req, "tool_choice");
    int use_tools = !(tc && tc->type == J_STR && !strcmp(tc->str, "none"));
    if (use_tools && request_tools(json_get(c->req, "tools"), &c->tools)) {
        c->tools_owned = 1;
        g_tools = &c->tools;
    } else if (use_tools && SRV_TOOLS) { // server-side tools: run them here, the client sees only text
        c->own_tools = 1;
        g_tools = SRV_TOOLS;
    } else {
        g_tools = NULL;
    }
    c->P = malloc(((size_t)strlen(body) + 65536) * sizeof(int));
    char err[256];
    c->np = render_chat(tk, messages, c->P, err, sizeof err, &c->sys_len);
    if (c->np < 0) {
        srv_error(fd, 400, "invalid_request_error", err);
        THINK = think0;
        cb_free(c, 1);
        return 0;
    }
    if (c->np + 1 >= ctx) { // drop the oldest turns, keep the system prefix and the newest tail
        int keep = ctx - c->sys_len - 2048;
        if (keep < 256) keep = 256;
        if (c->sys_len + keep < c->np) {
            memmove(c->P + c->sys_len, c->P + c->np - keep, (size_t)keep * sizeof(int));
            fprintf(stderr, "[server] context window full: dropped %d old tokens (kept the system prefix and the last %d)\n",
                    c->np - c->sys_len - keep, keep);
            c->np = c->sys_len + keep;
        }
    }
    if (c->np + 1 >= ctx) {
        snprintf(err, sizeof err, "the prompt is %d tokens, the context window is %d (-c)", c->np, ctx);
        srv_error(fd, 400, "context_length_exceeded", err);
        THINK = think0;
        cb_free(c, 1);
        return 0;
    }
    // session state: this session's own KV cache, kept resident between turns
    if (!S->kv_ready) {
        cb_kv_budget(tr, S); // make room before allocating another full -c cache
        kv_cache_alloc(&tr->config, tr->state.kv_type, &S->kv);
        S->kv_ready = 1;
        S->kv_n = 0;
    }
    kv_attach(&tr->state, &S->kv);
    if (!S->kv_disk_checked && S->kv_n == 0 && SRV_SESS_DIR && S != &g_fallback) { // a dump from an earlier process
        char path[1200];
        snprintf(path, sizeof path, "%s/%s.bin", SRV_SESS_DIR, S->id);
        long r0 = time_in_ms();
        int loaded = sess_read(tr, S->tok, S->cap, path);
        long rms = time_in_ms() - r0;
        S->kv_disk_checked = 1;
        if (loaded > 0) {
            S->n = loaded;
            S->kv_n = loaded;
            fprintf(stderr, "[server] session %s: restored %d tokens from disk in %ld ms\n", S->id, loaded, rms);
        } else {
            S->n = 0;
        }
    }
    int lcp = 0;
    while (lcp < S->kv_n && lcp < c->np && S->tok[lcp] == c->P[lcp]) lcp++;
    // the shared system prefix of agent clients: restore its state instead of re-processing it
    SrvSys* sys_slot = c->sys_len >= SRV_SYS_MIN && c->sys_len < c->np ? srv_sys_find(c->P, c->sys_len) : NULL;
    if (sys_slot && lcp < c->sys_len) {
        srv_sys_restore(tr, sys_slot, S); // the state stands at sys_len, in this session's cache
        memcpy(S->tok, c->P, (size_t)c->sys_len * sizeof(int));
        S->n = S->kv_n = c->sys_len;
        lcp = c->sys_len;
    }
    c->reuse = lcp < c->np ? lcp : c->np - 1; // at least one token is fed to get the logits
    if (c->reuse < S->n) S->n = c->reuse;     // a diverged branch overwrites the tail
    // a new distinct system prefix: capture its state while the prefill passes through it
    SrvSys* cap_slot = NULL;
    if (!sys_slot && c->sys_len >= SRV_SYS_MIN && c->sys_len < c->np && c->reuse < c->sys_len) {
        cap_slot = srv_sys_slot();
        if (cap_slot->n < c->sys_len) {
            free(cap_slot->tok);
            cap_slot->tok = malloc((size_t)c->sys_len * sizeof(int));
        }
        if (cap_slot->tok) {
            memcpy(cap_slot->tok, c->P, (size_t)c->sys_len * sizeof(int));
            cap_slot->n = c->sys_len;
            cap_slot->snap.n = 0;
        } else {
            cap_slot = NULL;
        }
    }
    c->h.tok = S->tok;
    c->h.n = c->reuse;
    c->h.cap = ctx;
    c->h.sys_len = c->sys_len;
    c->h.turn = NULL;
    c->h.n_turns = 0;
    c->logits = malloc((size_t)tr->config.vocab_size * sizeof(float));
    c->rs = (ReplyState){.mode = 0, .n_calls = 0, .token = -1, .calls = c->calls};
    sb_puts(&c->rs.buf, "");
    cb_stream_begin(c);
    if (c->r.failed) { THINK = think0; cb_finish(c, tr, tk); return 0; }
    // blocking prefill of the tail (the session cache is attached); keep-alives for the other streams
    for (int i = 0; i < SPEC_MAX_TOK; i++) if (g_cb[i].used) cb_keepalive(&g_cb[i]);
    for (int i = 0; i < SPEC_MAX_TOK; i++) { // active streams must not time out during this prefill
        CbConn* a = &g_cb[i];
        if (a->used && a->r.stream && !a->r.failed && a->hb_on == 0) {
            a->hb_stop = 0;
            a->hb_failed = 0;
            a->hb = (SrvHb){.fd = a->fd, .stop = &a->hb_stop, .failed = &a->hb_failed, .abort = NULL};
            if (pthread_create(&a->hbt, NULL, srv_heartbeat, &a->hb) == 0) a->hb_on = 2;
        }
    }
    g_sink = &c->sink;
    g_srv_abort = 0;
    long t0 = time_in_ms();
    if (c->r.stream) {
        c->hb_failed = 0;
        c->hb_stop = 0;
        c->hb_on = 1;
        c->hb = (SrvHb){.fd = fd, .stop = &c->hb_stop, .failed = &c->hb_failed, .abort = &g_srv_abort};
        pthread_create(&c->hbt, NULL, srv_heartbeat, &c->hb);
    }
    float* logits = NULL;
    int at = c->reuse;
    if (cap_slot) { // feed through the end of the system prefix and keep its state
        logits = chat_feed(tr, &c->h, c->P + at, c->sys_len - at);
        if (logits && srv_snap_take(tr, &cap_slot->snap, c->sys_len) && SRV_SESS_DIR) {
            char path[1200];
            snprintf(path, sizeof path, "%s/.sys-%08x.bin", SRV_SESS_DIR, srv_tok_hash(c->P, c->sys_len));
            sess_write(tr, c->P, c->sys_len, path); // best effort: survives a restart
        }
        at = c->sys_len;
        if (!logits) c->r.failed = 1;
    }
    if (!c->r.failed) logits = chat_feed(tr, &c->h, c->P + at, c->np - at);
    if (c->hb_on) {
        c->hb_stop = 1;
        pthread_join(c->hbt, NULL);
        c->hb_on = 0;
        if (c->hb_failed) c->r.failed = 1;
    }
    for (int i = 0; i < SPEC_MAX_TOK; i++) { // stop the keep-alives of the other streams
        CbConn* a = &g_cb[i];
        if (a->hb_on == 2) {
            a->hb_stop = 1;
            pthread_join(a->hbt, NULL);
            a->hb_on = 0;
            if (a->hb_failed) { a->r.failed = 1; a->sink.stop = 1; }
        }
    }
    c->t_pre = time_in_ms() - t0;
    if (!logits) { // the client went away mid-prefill
        S->n = c->h.n;
        S->kv_n = c->h.n;
        cb_finish(c, tr, tk);
        THINK = think0;
        return 0;
    }
    S->n = c->h.n;
    S->kv_n = c->h.n;
    memcpy(c->logits, logits, (size_t)tr->config.vocab_size * sizeof(float));
    c->t_gen = time_in_ms();
    g_tools = c->own_tools ? SRV_TOOLS : (c->tools.root ? &c->tools : NULL);
    int tok = sample(&c->samp, c->logits);
    if (!cb_token(c, tk, tok)) {
        int cont = cb_after_end(c, tr, tk);
        THINK = think0;
        return cont;
    }
    THINK = think0;
    return 1;
}

static int cb_decode_step(Transformer* tr, Tokenizer* tk) {
    SrvTok ts[SPEC_MAX_TOK];
    int idx[SPEC_MAX_TOK], n = 0;
    for (int i = 0; i < SPEC_MAX_TOK; i++) {
        CbConn* c = &g_cb[i];
        if (!c->used || !c->feed) continue;
        c->feed = 0;
        ts[n].kv = &c->S->kv;
        ts[n].tok = c->next_tok;
        ts[n].pos = c->next_pos;
        ts[n].logits = c->logits;
        idx[n++] = i;
    }
    if (n == 0) return 0;
    if (!forward_sessions(tr, ts, n)) {
        for (int i = 0; i < n; i++) cb_finish(&g_cb[idx[i]], tr, tk);
        return n;
    }
    for (int i = 0; i < n; i++) {
        CbConn* c = &g_cb[idx[i]];
        if (c->r.failed) { cb_finish(c, tr, tk); continue; }
        int tok = sample(&c->samp, c->logits);
        if (!cb_token(c, tk, tok)) cb_after_end(c, tr, tk);
    }
    return n;
}

static int cb_dispatch(Transformer* tr, Tokenizer* tk, Sampler* sp, int fd, const char* method, const char* path,
                       int authorized, const char* shead, const char* body) {
    // returns 1 when the fd is owned by a decode slot or the serial queue (or already closed),
    // 0 when the caller should close it after the response
    if (!strcmp(method, "OPTIONS")) {
        const char* h = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                        "Access-Control-Allow-Headers: Authorization, Content-Type\r\nConnection: close\r\n\r\n";
        SrvReq r = {.fd = fd};
        srv_write(&r, h, strlen(h));
        return 0;
    }
    if (!strcmp(path, "/health")) { srv_respond(fd, 200, "application/json", "{\"status\": \"ok\"}"); return 0; }
    if (!authorized) { srv_error(fd, 401, "invalid_request_error", "invalid API key"); return 0; }
    if (!strcmp(method, "GET") && !strcmp(path, "/metrics")) {
        SBuf b = {0};
        srv_metrics_json(&b, cb_active(), g_npend);
        srv_respond(fd, 200, "application/json", b.buf);
        free(b.buf);
        return 0;
    }
    if (!strcmp(method, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/models"))) {
        SBuf b = {0};
        sb_puts(&b, "{\"object\": \"list\", \"data\": [{\"id\": ");
        json_write_string(&b, SRV_MODEL);
        char t[160];
        snprintf(t, sizeof t, ", \"object\": \"model\", \"created\": %ld, \"owned_by\": \"numallama\", \"context_length\": %d}]}",
                 (long)time(NULL), tr->config.seq_len);
        sb_puts(&b, t);
        srv_respond(fd, 200, "application/json", b.buf);
        free(b.buf);
        return 0;
    }
    if (!strcmp(method, "GET") && !strcmp(path, "/v1/sessions")) {
        SBuf b = {0};
        sb_puts(&b, "{\"object\": \"list\", \"data\": [");
        int first = 1;
        for (int i = 0; i < g_nsessions; i++) {
            if (!first) sb_puts(&b, ", ");
            sb_puts(&b, "{\"id\": ");
            json_write_string(&b, g_sessions[i].id);
            char it[64];
            snprintf(it, sizeof it, ", \"tokens\": %d, \"live\": %s}", g_sessions[i].n, g_sessions[i].live ? "true" : "false");
            sb_puts(&b, it);
            first = 0;
        }
        if (SRV_SESS_DIR) { // sessions dumped by an earlier process
            DIR* d = opendir(SRV_SESS_DIR);
            if (d) {
                struct dirent* e;
                while ((e = readdir(d))) {
                    if (strncmp(e->d_name, ".sys-", 5) == 0) continue;
                    char* dot = strstr(e->d_name, ".bin");
                    if (!dot || dot[4] != '\0') continue;
                    char id[72];
                    snprintf(id, sizeof id, "%.*s", (int)(dot - e->d_name), e->d_name);
                    int known = 0;
                    for (int i = 0; i < g_nsessions; i++) known |= !strcmp(g_sessions[i].id, id);
                    if (known) continue;
                    if (!first) sb_puts(&b, ", ");
                    sb_puts(&b, "{\"id\": ");
                    json_write_string(&b, id);
                    sb_puts(&b, ", \"tokens\": 0, \"live\": false, \"on_disk\": true}");
                    first = 0;
                }
                closedir(d);
            }
        }
        sb_puts(&b, "]}");
        srv_respond(fd, 200, "application/json", b.buf);
        free(b.buf);
        return 0;
    }
    if (!strcmp(method, "POST") && (!strcmp(path, "/v1/chat/completions") || !strcmp(path, "/chat/completions"))) {
        int slot = -1;
        for (int i = 0; i < SPEC_MAX_TOK; i++) if (!g_cb[i].used) { slot = i; break; }
        if (slot < 0) { srv_error(fd, 503, "server_error", "all decode slots are busy"); return 0; }
        CbConn* c = &g_cb[slot];
        int rc = cb_prepare(c, tr, tk, sp, fd, shead, body);
        if (rc < 0) { // queue for the serial loop (server-side tools / busy session)
            if (g_npend >= 16) { srv_error(fd, 503, "server_error", "the serial queue is full"); return 0; }
            CbPending* p = &g_pend[g_npend++];
            p->fd = fd;
            snprintf(p->shead, sizeof p->shead, "%s", shead ? shead : "");
            p->body = strdup(body);
            return 1; // the fd is owned by the queue
        }
        if (rc == 0) return 1; // answered and closed inside
        c->used = 1;
        return 1;
    }
    srv_error(fd, 404, "invalid_request_error", "unknown endpoint");
    return 0;
}

// system-prefix slot files of an earlier process: read them into the in-memory slots
static void cb_load_sys(Transformer* tr) {
    if (!SRV_SESS_DIR) return;
    DIR* d = opendir(SRV_SESS_DIR);
    if (!d) return;
    int* buf = malloc((size_t)tr->config.seq_len * sizeof(int));
    int cnt = 0;
    struct dirent* e;
    while (buf && (e = readdir(d))) {
        if (strncmp(e->d_name, ".sys-", 5) != 0 || !strstr(e->d_name, ".bin")) continue;
        char path[1200];
        snprintf(path, sizeof path, "%s/%s", SRV_SESS_DIR, e->d_name);
        cb_restore_global(tr);
        int n = sess_read(tr, buf, tr->config.seq_len, path);
        if (n < SRV_SYS_MIN) continue;
        SrvSys* s = srv_sys_slot();
        free(s->tok);
        s->tok = malloc((size_t)n * sizeof(int));
        if (!s->tok) { s->n = 0; continue; }
        memcpy(s->tok, buf, (size_t)n * sizeof(int));
        s->n = n;
        if (!srv_snap_take(tr, &s->snap, n)) {
            s->n = 0;
            free(s->tok);
            s->tok = NULL;
        } else {
            cnt++;
        }
    }
    if (d) closedir(d);
    free(buf);
    cb_restore_global(tr);
    if (cnt) fprintf(stderr, "[server] %d system-prefix slot%s loaded\n", cnt, cnt == 1 ? "" : "s");
}

static void cb_run_pending_serial(Transformer* tr, Tokenizer* tk, Sampler* sp) {
    if (!g_npend || cb_active() > 0) return;
    CbPending p = g_pend[0];
    memmove(g_pend, g_pend + 1, (size_t)(--g_npend) * sizeof *g_pend);
    cb_restore_global(tr);
    srv_chat(tr, tk, sp, p.shead, p.fd, p.body);
    free(p.body);
    shutdown(p.fd, SHUT_WR);
    close(p.fd);
}

// incremental HTTP request reader of one connection (headers, then the body)
#define CB_MAX_READERS 8
typedef struct {
    int fd;
    char *buf;
    size_t len, cap, bstart;
    long clen;
    int head, auth;
    char shead[80], method[16], path[512];
    long t0;
} CbReader;
static CbReader g_rd[CB_MAX_READERS];

static void cb_rd_drop(int i) {
    if (g_rd[i].fd < 0) return;
    close(g_rd[i].fd);
    free(g_rd[i].buf);
    memset(&g_rd[i], 0, sizeof g_rd[i]);
    g_rd[i].fd = -1;
}

static int cb_rd_add(int fd) {
    for (int i = 0; i < CB_MAX_READERS; i++)
        if (g_rd[i].fd < 0) {
            memset(&g_rd[i], 0, sizeof g_rd[i]);
            g_rd[i].fd = fd;
            g_rd[i].cap = 65536;
            g_rd[i].buf = malloc(g_rd[i].cap);
            g_rd[i].t0 = (long)time(NULL);
            if (!g_rd[i].buf) { cb_rd_drop(i); return -1; }
            return i;
        }
    return -1;
}

// 1: headers parsed; 0: need more; -1: answered/closed
static int cb_rd_head(CbReader* r) {
    char* he = strstr(r->buf, "\r\n\r\n");
    if (!he) {
        if (r->len > (1 << 16)) { srv_error(r->fd, 413, "invalid_request_error", "headers too large"); return -1; }
        return 0;
    }
    size_t hlen = (size_t)(he - r->buf) + 4;
    *he = '\0';
    r->clen = 0;
    r->auth = SRV_KEY == NULL;
    r->shead[0] = '\0';
    if (sscanf(r->buf, "%15s %511s", r->method, r->path) != 2) { srv_error(r->fd, 400, "invalid_request_error", "bad request"); return -1; }
    int expect = 0;
    for (char* line = strstr(r->buf, "\r\n"); line; line = strstr(line, "\r\n")) {
        line += 2;
        if (!strncasecmp(line, "Content-Length:", 15)) r->clen = atol(line + 15);
        else if (!strncasecmp(line, "Expect:", 7) && strstr(line, "100-continue")) expect = 1;
        else if (SRV_KEY && !strncasecmp(line, "Authorization:", 14)) {
            const char* v = line + 14;
            while (*v == ' ') v++;
            if (!strncasecmp(v, "Bearer ", 7)) v += 7;
            size_t kl = strlen(SRV_KEY);
            if (!strncmp(v, SRV_KEY, kl) && (v[kl] == '\r' || v[kl] == '\0')) r->auth = 1;
        } else if (r->shead[0] == '\0') {
            const char* sh = NULL;
            if (!strncasecmp(line, "x-opencode-session-id:", 22)) sh = line + 22;
            else if (!strncasecmp(line, "x-session-id:", 13)) sh = line + 13;
            else if (!strncasecmp(line, "x-conversation-id:", 18)) sh = line + 18;
            if (sh) {
                while (*sh == ' ') sh++;
                size_t n = strcspn(sh, "\r\n");
                if (n >= sizeof r->shead) n = sizeof r->shead - 1;
                memcpy(r->shead, sh, n);
                r->shead[n] = '\0';
            }
        }
    }
    if (r->clen < 0 || r->clen > SRV_MAX_BODY) { srv_error(r->fd, 413, "invalid_request_error", "request body too large"); return -1; }
    if (expect) { SrvReq w = {.fd = r->fd}; srv_write(&w, "HTTP/1.1 100 Continue\r\n\r\n", 25); }
    r->bstart = hlen;
    size_t need = hlen + (size_t)r->clen + 1;
    if (need > r->cap) {
        char* nb = realloc(r->buf, need);
        if (!nb) { srv_error(r->fd, 500, "server_error", "out of memory"); return -1; }
        r->buf = nb;
        r->cap = need;
    }
    r->head = 1;
    return 1;
}

// read what is available; 1: a complete request is ready (method/path/body in r->buf + bstart)
static int cb_rd_pump(CbReader* r) {
    for (;;) {
        if (!r->head && r->len + 4096 > r->cap) {
            size_t nc = r->cap * 2;
            char* nb = realloc(r->buf, nc);
            if (!nb) { srv_error(r->fd, 500, "server_error", "out of memory"); return -1; }
            r->buf = nb;
            r->cap = nc;
        }
        ssize_t k = recv(r->fd, r->buf + r->len, r->cap - r->len - 1, MSG_DONTWAIT);
        if (k < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            return -1;
        }
        if (k == 0) return -1; // client closed
        r->len += (size_t)k;
        r->buf[r->len] = '\0';
        if (!r->head) {
            int h = cb_rd_head(r);
            if (h < 0) return -1;
        }
        if (r->head && r->len >= r->bstart + (size_t)r->clen) {
            r->buf[r->bstart + (size_t)r->clen] = '\0';
            return 1;
        }
    }
    return 0;
}

static void cb_rd_service(Transformer* tr, Tokenizer* tk, Sampler* sp, int i) {
    CbReader* r = &g_rd[i];
    if (r->fd < 0) return;
    int st = cb_rd_pump(r);
    if (st < 0) { cb_rd_drop(i); return; }
    if (st == 0) return;
    int fd = r->fd;
    char method[16], path[512], shead[80];
    snprintf(method, sizeof method, "%s", r->method);
    snprintf(path, sizeof path, "%s", r->path);
    snprintf(shead, sizeof shead, "%s", r->shead);
    int auth = r->auth;
    char* body = r->buf + r->bstart;
    char* q = strchr(path, '?');
    if (q) *q = '\0';
    int owned = cb_dispatch(tr, tk, sp, fd, method, path, auth, shead, body);
    r->fd = -1;
    free(r->buf);
    memset(r, 0, sizeof *r);
    r->fd = -1;
    if (!owned) { shutdown(fd, SHUT_WR); close(fd); }
}

static void server_batch(Transformer* tr, Tokenizer* tk, Sampler* sp) {
    g_cb_tr = tr;
    cb_kv_budget_init();
    signal(SIGPIPE, SIG_IGN);
    SRV_STARTED = time(NULL); SRV_STOP = 0; SRV_LS = -1;
    srv_install_stop();
    init_special_tokens(tk);
    for (int i = 0; i < SPEC_MAX_TOK; i++) g_cb[i].fd = -1;
    g_fallback.id[0] = '\0';
    g_fallback.cap = tr->config.seq_len;
    g_fallback.tok = malloc((size_t)tr->config.seq_len * sizeof(int));
    if (SRV_SESS_DIR && SRV_SESS_DIR[0]) {
        if (mkdir(SRV_SESS_DIR, 0755) != 0 && errno != EEXIST)
            fprintf(stderr, "[server] warning: cannot create %s: %s\n", SRV_SESS_DIR, strerror(errno));
        cb_load_sys(tr);
    }
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(SRV_PORT)};
    if (inet_pton(AF_INET, SRV_HOST, &a.sin_addr) != 1) { fprintf(stderr, "bad host %s (IPv4 address expected)\n", SRV_HOST); exit(EXIT_FAILURE); }
    if (bind(ls, (struct sockaddr*)&a, sizeof a) < 0 || listen(ls, 128) < 0) {
        fprintf(stderr, "cannot listen on %s:%d: %s\n", SRV_HOST, SRV_PORT, strerror(errno));
        exit(EXIT_FAILURE);
    }
    int fl = fcntl(ls, F_GETFL, 0);
    fcntl(ls, F_SETFL, fl | O_NONBLOCK);
    size_t per = sess_region_bytes(&tr->config, tr->config.seq_len, tr->state.kv_type);
    fprintf(stderr, "[server] continuous batching: %d sessions, KV %.1f MB each (%.1f GB total)\n", SRV_J,
            per / 1e6, (double)per * SRV_J / 1e9);
    fprintf(stderr, "[server] model %s, context %d, listening on http://%s:%d/v1%s\n", SRV_MODEL, tr->config.seq_len, SRV_HOST,
            SRV_PORT, SRV_KEY ? " (API key required)" : "");
    for (int i = 0; i < CB_MAX_READERS; i++) g_rd[i].fd = -1;
    SRV_LS = ls;
    while (!SRV_STOP) {
        cb_run_pending_serial(tr, tk, sp);
        if (cb_active() > 0) cb_decode_step(tr, tk);
        int active = cb_active();
        struct pollfd pf[1 + SPEC_MAX_TOK + CB_MAX_READERS];
        int rpos[CB_MAX_READERS];
        int nf = 1, nrd = 0;
        for (int i = 0; i < CB_MAX_READERS; i++) rpos[i] = -1;
        pf[0] = (struct pollfd){.fd = ls, .events = POLLIN};
        for (int i = 0; i < SPEC_MAX_TOK; i++)
            if (g_cb[i].used) pf[nf++] = (struct pollfd){.fd = g_cb[i].fd, .events = POLLIN};
        for (int i = 0; i < CB_MAX_READERS; i++)
            if (g_rd[i].fd >= 0) {
                rpos[i] = nf;
                pf[nf++] = (struct pollfd){.fd = g_rd[i].fd, .events = POLLIN};
                nrd++;
            }
        int timeout = active > 0 ? 0 : (g_npend ? 0 : (nrd ? 1000 : -1));
        int pr = poll(pf, nf, timeout);
        if (SRV_STOP) break;
        if (pr < 0 && errno == EINTR) continue;
        // a decode connection went away: stop its generation (the others continue)
        for (int k = 1; k < nf; k++) {
            if (pf[k].revents & (POLLERR | POLLHUP | POLLRDHUP | POLLNVAL)) {
                for (int i = 0; i < SPEC_MAX_TOK; i++)
                    if (g_cb[i].used && g_cb[i].fd == pf[k].fd) { g_cb[i].r.failed = 1; g_cb[i].sink.stop = 1; }
            }
        }
        // incremental request readers
        for (int i = 0; i < CB_MAX_READERS; i++) {
            if (rpos[i] < 0) continue;
            short re = pf[rpos[i]].revents;
            if (re & (POLLERR | POLLHUP | POLLRDHUP | POLLNVAL)) { cb_rd_drop(i); continue; }
            if (re & POLLIN) cb_rd_service(tr, tk, sp, i);
            else if (time(NULL) - g_rd[i].t0 > 60) cb_rd_drop(i); // a silent client
        }
        // accept when a decode slot and a reader slot are free
        int free_rd = 0;
        for (int i = 0; i < CB_MAX_READERS; i++) free_rd += g_rd[i].fd < 0;
        if ((pf[0].revents & POLLIN) && cb_active() < SRV_J && free_rd > 0) {
            for (;;) {
                int fd = accept(ls, NULL, NULL);
                if (fd < 0) break;
                struct timeval stv = {.tv_sec = 10};
                setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof stv);
                setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                int ri = cb_rd_add(fd);
                if (ri < 0) { shutdown(fd, SHUT_WR); close(fd); break; }
                cb_rd_service(tr, tk, sp, ri); // the request may already be in the socket
                int slots = 0, rd = 0;
                for (int i = 0; i < CB_MAX_READERS; i++) rd += g_rd[i].fd < 0;
                slots = cb_active() < SRV_J;
                if (!slots || !rd) break;
            }
        }
    }
}
