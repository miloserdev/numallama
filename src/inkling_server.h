// Minimal OpenAI-compatible HTTP face for the Inkling text forward.
// One request is decoded at a time, but a single conversation keeps its state and logits across
// turns. The NUMA replica launcher/router provides multi-client concurrency.
static long g_ink_req;
static volatile int g_ink_abort;
static volatile sig_atomic_t g_ink_signal;
static int g_ink_ls = -1;
static struct {
  long requests, prompt_tokens, completion_tokens, cache_hits, cache_misses, errors;
  long started;
} g_ink_metrics;
static struct {
  int valid, n, cap;
  char sid[80];
  int *tok;
  float *logits;
} g_ink_cache;

static void inkling_sid(const char *shead, JNode *req, char *out, size_t cap) {
  JNode *v = req ? json_get(req, "session_id") : NULL;
  if (!v) v = req ? json_get(req, "conversation_id") : NULL;
  const char *s = v && v->type == J_STR ? v->str : shead;
  if (!s || !s[0]) s = "default";
  snprintf(out, cap, "%.79s", s);
  for (char *p = out; *p; p++)
    if (!isalnum((unsigned char)*p) && *p != '-' && *p != '_' && *p != '.') *p = '_';
}

static void inkling_on_signal(int sig) {
  (void)sig;
  g_ink_signal = 1;
  if (g_ink_ls >= 0) shutdown(g_ink_ls, SHUT_RDWR);
  if (g_ink_ls >= 0) close(g_ink_ls);
  g_ink_ls = -1;
}

static void ink_sse(SrvReq *r, long rid, int reasoning, const char *piece, const char *reason) {
  SBuf b = {0};
  char pre[200];
  snprintf(pre, sizeof pre, "data: {\"id\":\"chatcmpl-ink%d\",\"object\":\"chat.completion.chunk\","
           "\"created\":%ld,\"model\":\"inkling\",\"choices\":[{\"index\":0,\"delta\":{", (int)rid, (long)time(NULL));
  sb_puts(&b, pre);
  if (piece) {
    sb_puts(&b, reasoning ? "\"reasoning_content\": " : "\"content\": ");
    json_write_string(&b, piece);
  }
  sb_puts(&b, "},\"finish_reason\":");
  if (reason) { sb_puts(&b, "\""); sb_puts(&b, reason); sb_puts(&b, "\""); }
  else sb_puts(&b, "null");
  sb_puts(&b, "}]}\n\n");
  srv_write(r, b.buf, b.len);
  free(b.buf);
}

static void inkling_http_chat(InklingModel *m, InklingState *st, InklingRuntime *rt, Tokenizer *tk,
                              int fd, const char *body, const char *shead) {
  JNode *req = json_parse(body, NULL);
  JNode *msgs = req ? json_get(req, "messages") : NULL;
  if (!req || req->type != J_OBJ || !msgs || msgs->type != J_ARR || !msgs->child) {
    g_ink_metrics.errors++;
    srv_error(fd, 400, "invalid_request_error", "the body must contain a non-empty messages array");
    json_free(req);
    return;
  }
  JNode *v;
  if ((v = json_get(req, "n")) && v->type == J_NUM && atoi(v->str) != 1) {
    g_ink_metrics.errors++;
    srv_error(fd, 400, "invalid_request_error", "only n = 1 is supported");
    json_free(req);
    return;
  }
  int stream = 0, max_gen = 256, include_usage = 0;
  float temp = 0.7f, topp = 0.9f, minp = 0.0f, pp = 0.0f, fp = 0.0f;
  unsigned long long seed = (unsigned long long)time(NULL);
  const char *stops[8];
  int n_stops = 0;
  if ((v = json_get(req, "stream")) && v->type == J_BOOL) stream = v->boolean;
  if ((v = json_get(req, "max_tokens")) && v->type == J_NUM) max_gen = atoi(v->str);
  else if ((v = json_get(req, "max_completion_tokens")) && v->type == J_NUM) max_gen = atoi(v->str);
  if ((v = json_get(req, "temperature")) && v->type == J_NUM) temp = (float)atof(v->str);
  if ((v = json_get(req, "top_p")) && v->type == J_NUM) topp = (float)atof(v->str);
  if ((v = json_get(req, "min_p")) && v->type == J_NUM) minp = (float)atof(v->str);
  if ((v = json_get(req, "presence_penalty")) && v->type == J_NUM) pp = (float)atof(v->str);
  if ((v = json_get(req, "frequency_penalty")) && v->type == J_NUM) fp = (float)atof(v->str);
  if ((v = json_get(req, "seed")) && v->type == J_NUM) seed = (unsigned long long)atoll(v->str) + 1;
  if ((v = json_get(req, "stream_options")) && (v = json_get(v, "include_usage")) && v->type == J_BOOL)
    include_usage = v->boolean;
  JNode *stop = json_get(req, "stop");
  if (stop && stop->type == J_STR && stop->str[0]) stops[n_stops++] = stop->str;
  else if (stop && stop->type == J_ARR)
    for (JNode *s = stop->child; s && n_stops < 8; s = s->next)
      if (s->type == J_STR && s->str[0]) stops[n_stops++] = s->str;
  if (max_gen <= 0) max_gen = 256;
  if (temp < 0) temp = 0;
  if (topp <= 0 || topp > 1) topp = 0.9f;
  if (minp < 0 || minp > 1) minp = 0;
  if (pp < -2 || pp > 2) pp = 0;
  if (fp < -2 || fp > 2) fp = 0;

  int *ids = malloc((size_t)st->ctx * sizeof(int));
  int n = 0;
  n = push_tpl(tk, "<|message_system|><|content_text|>Thinking effort level: 0.0<|end_message|>", ids, n);
  for (JNode *msg = msgs->child; msg && n < st->ctx; msg = msg->next) {
    JNode *role = json_get(msg, "role");
    if (!role || role->type != J_STR) continue;
    const char *r = role->str;
    if (!strcmp(r, "tool")) continue;
    if (!strcmp(r, "assistant")) {
      JNode *rc = json_get(msg, "reasoning_content");
      if (rc && rc->type == J_STR && rc->str[0]) {
        n = push_tpl(tk, "<|message_model|><|content_thinking|>", ids, n);
        n = push_text(tk, rc->str, ids, n);
        n = push_tpl(tk, "<|end_message|>", ids, n);
      }
    }
    const char *hdr = !strcmp(r, "system") ? "<|message_system|><|content_text|>"
                    : !strcmp(r, "assistant") ? "<|message_model|><|content_text|>"
                    : "<|message_user|><|content_text|>";
    SBuf text = {0};
    int has_text = msg_text(json_get(msg, "content"), &text) && text.buf && text.buf[0];
    if (has_text) {
      n = push_tpl(tk, hdr, ids, n);
      if (strlen(text.buf) < (size_t)st->ctx * 4) n = push_text(tk, text.buf, ids, n);
      n = push_tpl(tk, "<|end_message|>", ids, n);
    }
    free(text.buf);
    if (!strcmp(r, "assistant")) n = push_tpl(tk, "<|content_model_end_sampling|>", ids, n);
  }
  const char *lim = getenv("INKLING_MAX_PROMPT");
  int maxp = lim ? atoi(lim) : 4096;
  if (n < 1 || n >= st->ctx || (maxp > 0 && n > maxp)) {
    g_ink_metrics.errors++;
    char msg[180];
    if (n >= st->ctx) snprintf(msg, sizeof msg, "prompt does not fit the context (%d tokens)", n);
    else if (maxp > 0 && n > maxp)
      snprintf(msg, sizeof msg, "Inkling prefill is single-threaded: %d tokens > INKLING_MAX_PROMPT=%d", n, maxp);
    else snprintf(msg, sizeof msg, "empty prompt");
    srv_error(fd, 400, "invalid_request_error", msg);
    free(ids); json_free(req); return;
  }

  char sid[80];
  inkling_sid(shead, req, sid, sizeof sid);
  int reuse = g_ink_cache.valid && !strcmp(g_ink_cache.sid, sid) && g_ink_cache.n <= n &&
              !memcmp(ids, g_ink_cache.tok, (size_t)g_ink_cache.n * sizeof(int));
  int start = reuse ? g_ink_cache.n : 0;
  if (reuse) g_ink_metrics.cache_hits++;
  else {
    g_ink_metrics.cache_misses++;
    g_ink_cache.valid = 0;
    g_ink_cache.n = 0;
    inkling_state_clear(m, st);
  }
  memcpy(g_ink_cache.tok, ids, (size_t)n * sizeof(int));
  g_ink_cache.n = n;
  g_ink_cache.valid = 1;
  snprintf(g_ink_cache.sid, sizeof g_ink_cache.sid, "%s", sid);

  float *logits = malloc((size_t)m->unpadded * sizeof(float));
  int have_logits = reuse && start == n && g_ink_cache.logits;
  if (have_logits) memcpy(logits, g_ink_cache.logits, (size_t)m->unpadded * sizeof(float));

  SrvReq out = {.fd = fd};
  long rid = ++g_ink_req;
  g_ink_metrics.requests++;
  g_ink_metrics.prompt_tokens += n - start;
  volatile int hb_stop = 0, hb_failed = 0;
  pthread_t hbt;
  int hb_on = 0;
  if (stream) {
    const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                    "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    srv_write(&out, h, strlen(h));
    ink_sse(&out, rid, 0, "", NULL);
    if (pthread_create(&hbt, NULL, srv_heartbeat,
                       &(SrvHb){.fd = fd, .stop = &hb_stop, .failed = &hb_failed, .abort = &g_ink_abort}) == 0)
      hb_on = 1;
  }
  g_ink_abort = 0;
  if (!have_logits) {
    if (n - start > 64)
      fprintf(stderr, "[inkling] prefill start: %d tok (%d cached), ctx %d\n", n - start, start, st->ctx);
    for (int i = start; i < n; i++) {
      if (g_ink_abort || out.failed) break;
      inkling_forward_token(m, st, rt, ids[i], i, logits);
      if (n - start > 512 && (i - start) % 512 == 0)
        fprintf(stderr, "[inkling] prefill %d/%d\n", i - start, n - start);
    }
  }
  if (hb_on) { hb_stop = 1; pthread_join(hbt, NULL); }
  if (g_ink_abort || hb_failed || out.failed) {
    g_ink_metrics.errors++;
    g_ink_cache.valid = 0;
    free(logits); free(ids); json_free(req); return;
  }

  Sampler sp;
  build_sampler(&sp, m->unpadded, temp, topp, seed);
  sp.min_p = minp; sp.presence_penalty = pp; sp.frequency_penalty = fp;
  const int id_model = find_special(tk, "<|message_model|>");
  const int id_text = find_special(tk, "<|content_text|>");
  const int id_end = find_special(tk, "<|end_message|>");
  const int id_think = find_special(tk, "<|content_thinking|>");
  const int id_begin = find_special(tk, "<|begin_of_text|>");
  SBuf content = {0}, reasoning = {0};
  int pos = n, prev = ids[n - 1], gen = 0, stopped = 0, thinking = 0;
  for (; gen < max_gen && pos < st->ctx && !out.failed; gen++) {
    int tok = sample(&sp, logits);
    if (tok == EOS_ID || tok == id_end) {
      if (pos < st->ctx) { g_ink_cache.tok[pos] = tok; inkling_forward_token(m, st, rt, tok, pos, logits); pos++; }
      if (tok == id_end && pos < st->ctx) { g_ink_cache.tok[pos] = EOS_ID; inkling_forward_token(m, st, rt, EOS_ID, pos, logits); pos++; }
      stopped = 1;
      break;
    }
    if (tok == id_think) thinking = 1;
    else if (tok == id_text) thinking = 0;
    else if (!(tok == id_model || tok == id_begin)) {
      char *piece = decode(tk, prev, tok);
      SBuf *dst = thinking ? &reasoning : &content;
      if (piece) sb_puts(dst, piece);
      if (stream) ink_sse(&out, rid, thinking, piece, NULL);
      for (int si = 0; si < n_stops; si++) {
        size_t l = strlen(stops[si]);
        if (dst->len >= l && !strcmp(dst->buf + dst->len - l, stops[si])) {
          dst->len -= l;
          dst->buf[dst->len] = '\0';
          stopped = 1;
          break;
        }
      }
    }
    if (pos < st->ctx) { g_ink_cache.tok[pos] = tok; inkling_forward_token(m, st, rt, tok, pos, logits); pos++; }
    prev = tok;
    if (stopped) break;
  }
  g_ink_cache.n = pos;
  memcpy(g_ink_cache.logits, logits, (size_t)m->unpadded * sizeof(float));
  const char *reason = stopped || gen < max_gen ? "stop" : "length";
  if (stream) {
    ink_sse(&out, rid, 0, NULL, reason);
    if (include_usage) {
      char u[256];
      snprintf(u, sizeof u, "data: {\"id\":\"chatcmpl-ink%d\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
               "\"model\":\"inkling\",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
               "\"total_tokens\":%d}}\n\n", (int)rid, (long)time(NULL), n, gen, n + gen);
      srv_write(&out, u, strlen(u));
    }
    srv_write(&out, "data: [DONE]\n\n", 14);
  } else {
    SBuf b = {0};
    char pre[200];
    snprintf(pre, sizeof pre, "{\"id\":\"chatcmpl-ink%d\",\"object\":\"chat.completion\",\"created\":%ld,"
             "\"model\":\"inkling\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":",
             (int)rid, (long)time(NULL));
    sb_puts(&b, pre);
    json_write_string(&b, content.buf ? content.buf : "");
    if (reasoning.len) {
      sb_puts(&b, ", \"reasoning_content\": ");
      json_write_string(&b, reasoning.buf);
    }
    char tail[256];
    snprintf(tail, sizeof tail, "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
             "\"total_tokens\":%d}}", reason, n, gen, n + gen);
    sb_puts(&b, tail);
    srv_respond(fd, 200, "application/json", b.buf);
    free(b.buf);
  }
  g_ink_metrics.completion_tokens += gen;
  free(content.buf);
  free(reasoning.buf);
  free_sampler(&sp);
  free(logits);
  free(ids);
  json_free(req);
}

static int inkling_server(const char *model_path, const char *tok_path, int ctx) {
  ARCH = ARCH_INKLING; SPECIAL_START = 199998; EOS_ID = 200006; GS = 64;
  SRV_MODEL = "inkling-mini";
  g_ink_metrics.started = time(NULL);
  InklingModel m;
  if (!inkling_load(model_path, &m)) { fprintf(stderr, "cannot load %s\n", model_path); return 1; }
  InklingState st;
  if (!inkling_state_alloc(&m, ctx, &st)) { fprintf(stderr, "state alloc failed\n"); return 1; }
  InklingRuntime rt;
  if (!inkling_runtime_init(&m, &rt)) { fprintf(stderr, "runtime init failed\n"); return 1; }
  Tokenizer tk;
  build_tokenizer(&tk, (char *)tok_path, m.vocab);
  g_ink_cache.cap = ctx;
  g_ink_cache.tok = malloc((size_t)ctx * sizeof(int));
  g_ink_cache.logits = malloc((size_t)m.unpadded * sizeof(float));
  if (!g_ink_cache.tok || !g_ink_cache.logits) { fprintf(stderr, "inkling cache alloc failed\n"); return 1; }

  signal(SIGPIPE, SIG_IGN);
  struct sigaction sa = {0};
  sa.sa_handler = inkling_on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGTERM, &sa, NULL);
  int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(SRV_PORT)};
  if (inet_pton(AF_INET, SRV_HOST, &a.sin_addr) != 1 ||
      bind(ls, (struct sockaddr *)&a, sizeof a) < 0 || listen(ls, 128) < 0) {
    fprintf(stderr, "cannot listen on %s:%d: %s\n", SRV_HOST, SRV_PORT, strerror(errno));
    return 1;
  }
  g_ink_ls = ls;
  fprintf(stderr, "[inkling] model %s, context %d, listening on http://%s:%d/v1%s\n",
          model_path, ctx, SRV_HOST, SRV_PORT, SRV_KEY ? " (API key required)" : "");
  while (!g_ink_signal) {
    struct pollfd pfd = {.fd = ls, .events = POLLIN};
    int pr = poll(&pfd, 1, 1000);
    if (g_ink_signal) break;
    if (pr < 0) { if (errno == EINTR) continue; break; }
    if (pr == 0 || !(pfd.revents & POLLIN)) continue;
    int fd = accept(ls, NULL, NULL);
    if (fd < 0) { if (g_ink_signal) break; continue; }
    struct timeval tv = {.tv_sec = 60};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    char method[16], path[512], *body = NULL, shead[80];
    int authorized = 0;
    if (srv_read(fd, method, path, &body, &authorized, shead, sizeof shead)) {
      char *q = strchr(path, '?');
      if (q) *q = '\0';
      if (!strcmp(method, "OPTIONS")) {
        const char *h = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                        "Access-Control-Allow-Headers: Authorization, Content-Type\r\nConnection: close\r\n\r\n";
        SrvReq r = {.fd = fd};
        srv_write(&r, h, strlen(h));
      } else if (!strcmp(path, "/health")) {
        srv_respond(fd, 200, "application/json", "{\"status\": \"ok\"}");
      } else if (!strcmp(path, "/metrics")) {
        char b[512];
        snprintf(b, sizeof b, "{\"uptime_s\":%ld,\"requests\":%ld,\"prompt_tokens\":%ld,\"completion_tokens\":%ld,"
                 "\"cache_hits\":%ld,\"cache_misses\":%ld,\"errors\":%ld}",
                 (long)(time(NULL) - g_ink_metrics.started), g_ink_metrics.requests, g_ink_metrics.prompt_tokens,
                 g_ink_metrics.completion_tokens, g_ink_metrics.cache_hits, g_ink_metrics.cache_misses,
                 g_ink_metrics.errors);
        srv_respond(fd, 200, "application/json", b);
      } else if (!authorized) {
        srv_error(fd, 401, "invalid_request_error", "invalid API key");
      } else if (!strcmp(method, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/models"))) {
        char b[256];
        snprintf(b, sizeof b, "{\"object\":\"list\",\"data\":[{\"id\":\"inkling-mini\",\"object\":\"model\","
                 "\"owned_by\":\"numallama\",\"context_length\":%d}]}", ctx);
        srv_respond(fd, 200, "application/json", b);
      } else if (!strcmp(method, "POST") && (!strcmp(path, "/v1/chat/completions") || !strcmp(path, "/chat/completions"))) {
        inkling_http_chat(&m, &st, &rt, &tk, fd, body, shead);
      } else {
        srv_error(fd, 404, "invalid_request_error", "unknown endpoint");
      }
    }
    free(body);
    shutdown(fd, SHUT_WR);
    close(fd);
  }
  if (g_ink_ls >= 0) close(g_ink_ls);
  fprintf(stderr, "[inkling] shutting down\n");
  free(g_ink_cache.tok);
  free(g_ink_cache.logits);
  free_tokenizer(&tk);
  inkling_runtime_free(&rt);
  inkling_state_free(&st);
  inkling_free(&m);
  return 0;
}
