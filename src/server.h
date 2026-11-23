/* OpenAI-compatible HTTP server: `runq model.bin -m server [-H host] [-P port] [-A api_key]`.
 *
 *   GET  /health, /v1/models
 *   POST /v1/chat/completions   messages (system / developer / user / assistant with tool_calls /
 *                               tool), tools, tool_choice "none", temperature, top_p, seed,
 *                               max_tokens / max_completion_tokens, stop, stream (SSE),
 *                               stream_options.include_usage, reasoning_effort /
 *                               chat_template_kwargs.enable_thinking (Qwen3.5 / 3.8)
 *
 * One request at a time: a request already uses every core, so the others wait in the listen queue.
 * The KV cache is kept between requests and the longest common token prefix is reused, so a
 * multi-turn conversation only pays for the new messages. A hybrid model (recurrent state that can
 * not be rewound) reuses the state saved at the end of the previous prompt.
 * Tool calls are returned to the client, which runs them (the -f sandbox is a CLI chat feature). */

#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#define SRV_MAX_BODY (64 << 20)

typedef struct {
  int fd, failed;
  int stream;
  SBuf content, reasoning;
  size_t sent_c, sent_r;     // bytes already streamed
  char **stops;
  int n_stops;
  size_t max_stop;
  int stop_hit;
  char id[48];
  long created;
  volatile int *hb_stop;     // SSE keep-alive during prefill: cleared on the first content chunk
} SrvReq;

static void srv_write(SrvReq *r, const char *s, size_t n) {
  while (!r->failed && n > 0) {
    ssize_t w = send(r->fd, s, n, MSG_NOSIGNAL);
    if (w <= 0) {
      if (w < 0 && errno == EINTR) continue;
      r->failed = 1; // client gone: stop generating
      if (g_sink) g_sink->stop = 1;
      return;
    }
    s += w;
    n -= (size_t)w;
  }
}

static void srv_respond(int fd, int code, const char *ctype, const char *body) {
  const char *st = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 401 ? "Unauthorized" : code == 404 ? "Not Found"
                 : code == 413 ? "Payload Too Large" : "Internal Server Error";
  char hdr[512];
  int hl = snprintf(hdr, sizeof hdr,
                    "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nAccess-Control-Allow-Origin: *\r\n"
                    "Connection: close\r\n\r\n", code, st, ctype, strlen(body));
  SrvReq r = {.fd = fd};
  srv_write(&r, hdr, hl);
  srv_write(&r, body, strlen(body));
}

static void srv_error(int fd, int code, const char *type, const char *msg) {
  SBuf b = {0};
  sb_puts(&b, "{\"error\": {\"message\": ");
  json_write_string(&b, msg);
  sb_puts(&b, ", \"type\": ");
  json_write_string(&b, type);
  sb_puts(&b, ", \"code\": null}}");
  srv_respond(fd, code, "application/json", b.buf);
  free(b.buf);
}

// length of the longest prefix of s[0..n) that does not end inside a UTF-8 sequence
static size_t utf8_whole(const char *s, size_t n) {
  size_t i = n;
  int back = 0;
  while (i > 0 && back < 4 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) { i--; back++; }
  if (i == 0) return n;
  unsigned char c = (unsigned char)s[i - 1];
  size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
  return (i - 1) + need <= n ? n : i - 1;
}

// replace bytes that are not part of a valid UTF-8 sequence with '?' (JSON must be valid UTF-8)
static void utf8_clean(char *s) {
  unsigned char *p = (unsigned char *)s;
  while (*p) {
    int n = *p < 0x80 ? 1 : (*p & 0xE0) == 0xC0 && *p >= 0xC2 ? 2 : (*p & 0xF0) == 0xE0 ? 3 : (*p & 0xF8) == 0xF0 && *p <= 0xF4 ? 4 : 0;
    int k = 1;
    while (k < n && (p[k] & 0xC0) == 0x80) k++;
    if (n == 0 || k < n) { *p++ = '?'; continue; }
    p += n;
  }
}

static void srv_chunk_head(SrvReq *r, SBuf *b) {
  sb_puts(b, "data: {\"id\": \"");
  sb_puts(b, r->id);
  char t[160];
  snprintf(t, sizeof t, "\", \"object\": \"chat.completion.chunk\", \"created\": %ld, \"model\": ", r->created);
  sb_puts(b, t);
  json_write_string(b, SRV_MODEL);
}

// stream the not yet sent part of the content (which = 0) or the reasoning (1)
static void srv_flush(SrvReq *r, int which, int final) {
  SBuf *src = which ? &r->reasoning : &r->content;
  size_t *sent = which ? &r->sent_r : &r->sent_c;
  if (!src->buf) return;
  size_t lim = src->len;
  if (!final) {
    if (!which && r->n_stops) lim = lim > r->max_stop - 1 ? lim - (r->max_stop - 1) : 0; // may still become a stop string
    lim = utf8_whole(src->buf, lim);
  }
  if (lim <= *sent) return;
  char *part = strndup(src->buf + *sent, lim - *sent);
  utf8_clean(part);
  SBuf b = {0};
  srv_chunk_head(r, &b);
  sb_puts(&b, ", \"choices\": [{\"index\": 0, \"delta\": {");
  sb_puts(&b, which ? "\"reasoning_content\": " : "\"content\": ");
  json_write_string(&b, part);
  sb_puts(&b, "}, \"finish_reason\": null}]}\n\n");
  srv_write(r, b.buf, b.len);
  free(b.buf);
  free(part);
  *sent = lim;
  if (r->hb_stop) { *r->hb_stop = 1; r->hb_stop = NULL; } // the stream flows: stop the keep-alive
}

// SSE comment lines while the prompt is being processed, so a client does not time out on a
// silent connection (prefill of a long prompt can take a minute); harmless per the SSE spec.
typedef struct {
  int fd;
  volatile int *stop, *failed;
} SrvHb;
static void *srv_heartbeat(void *arg) {
  SrvHb *hb = arg;
  while (!*hb->stop) {
    struct timespec ts = {2, 0};
    nanosleep(&ts, NULL);
    if (*hb->stop) break;
    if (send(hb->fd, ": keep-alive\n\n", 14, MSG_NOSIGNAL) != 14) {
      *hb->failed = 1;
      g_srv_abort = 1; // stop the prefill: the client is gone
      if (g_sink) g_sink->stop = 1;
      break;
    }
  }
  return NULL;
}

static void srv_text(void *ctx, const char *s, int reasoning) {
  SrvReq *r = ctx;
  if (reasoning) {
    sb_puts(&r->reasoning, s);
    if (r->stream) srv_flush(r, 1, 0);
    return;
  }
  size_t from = r->content.len;
  sb_puts(&r->content, s);
  for (int i = 0; i < r->n_stops; i++) { // a stop string ends the reply and is not part of it
    size_t ls = strlen(r->stops[i]);
    size_t st = from >= ls ? from - ls + 1 : 0;
    char *hit = strstr(r->content.buf + st, r->stops[i]);
    if (hit) {
      r->content.len = hit - r->content.buf;
      r->content.buf[r->content.len] = '\0';
      r->stop_hit = 1;
      g_sink->stop = 1;
      break;
    }
  }
  if (r->stream) srv_flush(r, 0, 0);
}

// text of a message "content": a string, null or an array of {"type": "text", "text": ...} parts
static int msg_text(JNode *c, SBuf *out) {
  sb_puts(out, "");
  if (!c || c->type == J_NULL) return 1;
  if (c->type == J_STR) { sb_puts(out, c->str); return 1; }
  if (c->type != J_ARR) return 0;
  for (JNode *p = c->child; p; p = p->next) {
    JNode *ty = json_get(p, "type"), *tx = json_get(p, "text");
    if (!ty || ty->type != J_STR || strcmp(ty->str, "text") || !tx || tx->type != J_STR) return 0;
    sb_puts(out, tx->str);
  }
  return 1;
}

static JNode *jdup(JNode *n) { // deep copy (key not kept)
  SBuf b = {0};
  json_write(&b, n, -1, 0);
  JNode *c = json_parse(b.buf, NULL);
  free(b.buf);
  return c;
}

// arguments of an OpenAI tool call (a JSON string, sometimes an object) as an object
static JNode *call_args(JNode *fn) {
  JNode *a = json_get(fn, "arguments");
  JNode *o = NULL;
  if (a && a->type == J_STR) o = json_parse(a->str, NULL);
  else if (a && a->type == J_OBJ) o = jdup(a);
  if (!o || o->type != J_OBJ) { json_free(o); o = calloc(1, sizeof(JNode)); o->type = J_OBJ; }
  return o;
}

// an assistant message's tool calls in the model's own format
static void render_calls(SBuf *b, JNode *calls, int had_text) {
  int i = 0;
  for (JNode *c = calls->child; c; c = c->next, i++) {
    JNode *fn = json_get(c, "function"), *name = fn ? json_get(fn, "name") : NULL;
    if (!name || name->type != J_STR) continue;
    JNode *args = call_args(fn);
    if (ARCH == ARCH_GLM5) { // calls follow the text directly, one after the other
      sb_puts(b, "<tool_call>");
      sb_puts(b, name->str);
      for (JNode *p = args->child; p; p = p->next) {
        sb_puts(b, "<arg_key>");
        sb_puts(b, p->key);
        sb_puts(b, "</arg_key><arg_value>");
        char *v = arg_to_string(p);
        sb_puts(b, v);
        free(v);
        sb_puts(b, "</arg_value>");
      }
      sb_puts(b, "</tool_call>");
    } else if (ARCH == ARCH_QWEN35) {
      sb_puts(b, i == 0 ? (had_text ? "\n\n" : "") : "\n");
      sb_puts(b, "<tool_call>\n<function=");
      sb_puts(b, name->str);
      sb_puts(b, ">\n");
      for (JNode *p = args->child; p; p = p->next) {
        sb_puts(b, "<parameter=");
        sb_puts(b, p->key);
        sb_puts(b, ">\n");
        char *v = arg_to_string(p);
        sb_puts(b, v);
        free(v);
        sb_puts(b, "\n</parameter>\n");
      }
      sb_puts(b, "</function>\n</tool_call>");
    } else if (ARCH >= ARCH_QWEN2) {
      sb_puts(b, i == 0 ? (had_text ? "\n" : "") : "\n");
      sb_puts(b, "<tool_call>\n{\"name\": ");
      json_write_string(b, name->str);
      sb_puts(b, ", \"arguments\": ");
      json_write(b, args, -1, 0);
      sb_puts(b, "}\n</tool_call>");
    } else { // Llama 3.1: the message is the JSON call
      sb_puts(b, "{\"name\": ");
      json_write_string(b, name->str);
      sb_puts(b, ", \"parameters\": ");
      json_write(b, args, -1, 0);
      sb_puts(b, "}");
    }
    json_free(args);
  }
}

// the whole conversation as prompt tokens, ending with the assistant header. -1 + err on bad input.
// *sys_out (optional) receives the length of the system prefix (tools + system text): identical
// across the sessions of agent clients, so the server reuses its KV / state (see srv_chat).
static int render_chat(Tokenizer *tk, JNode *messages, int *out, char *err, size_t errlen, int *sys_out) {
  SBuf sys = {0}, t = {0};
  sb_puts(&sys, "");
  for (JNode *m = messages->child; m; m = m->next) { // every system / developer message goes first
    JNode *role = json_get(m, "role");
    if (!role || role->type != J_STR) { snprintf(err, errlen, "every message needs a role"); free(sys.buf); return -1; }
    if (strcmp(role->str, "system") && strcmp(role->str, "developer")) continue;
    if (!msg_text(json_get(m, "content"), &t)) { snprintf(err, errlen, "only text content is supported"); free(sys.buf); free(t.buf); return -1; }
    if (sys.len) sb_puts(&sys, "\n\n");
    sb_puts(&sys, t.buf);
    t.len = 0;
  }
  char *st = sys.buf; // the templates trim the system message
  while (*st && isspace((unsigned char)*st)) st++;
  size_t sl = strlen(st);
  while (sl > 0 && isspace((unsigned char)st[sl - 1])) st[--sl] = '\0';
  int n = build_system(tk, st, out, 0);
  if (sys_out) *sys_out = n;
  free(sys.buf);
  const int glm = ARCH == ARCH_GLM5, qwen = ARCH_IS_QWEN;
  int in_tools = 0; // Qwen: consecutive tool messages share one user message (GLM: one <|observation|>)
  for (JNode *m = messages->child; m; m = m->next) {
    const char *role = json_get(m, "role")->str;
    if (!strcmp(role, "system") || !strcmp(role, "developer")) continue;
    int is_tool = !strcmp(role, "tool") || !strcmp(role, "function");
    if (in_tools && !is_tool) { if (qwen) n = push_tpl(tk, "<|im_end|>\n", out, n); in_tools = 0; }
    t.len = 0;
    if (!msg_text(json_get(m, "content"), &t)) { snprintf(err, errlen, "only text content is supported"); free(t.buf); return -1; }
    if (!strcmp(role, "user")) {
      n = push_tpl(tk, glm ? "<|user|>" : qwen ? "<|im_start|>user\n" : "<|start_header_id|>user<|end_header_id|>\n\n", out, n);
      n = push_text(tk, t.buf, out, n);
      n = push_tpl(tk, glm ? "" : qwen ? "<|im_end|>\n" : "<|eot_id|>", out, n);
    } else if (is_tool) {
      if (glm) {
        if (!in_tools) n = push_tpl(tk, "<|observation|>", out, n);
        in_tools = 1;
        n = push_tpl(tk, "<tool_response>", out, n);
        n = push_text(tk, t.buf, out, n);
        n = push_tpl(tk, "</tool_response>", out, n);
      } else if (qwen) {
        if (!in_tools) n = push_tpl(tk, "<|im_start|>user", out, n);
        in_tools = 1;
        n = push_tpl(tk, "\n<tool_response>\n", out, n);
        n = push_text(tk, t.buf, out, n);
        n = push_tpl(tk, "\n</tool_response>", out, n);
      } else {
        n = push_tpl(tk, "<|start_header_id|>ipython<|end_header_id|>\n\n", out, n);
        n = push_text(tk, t.buf, out, n);
        n = push_tpl(tk, "<|eot_id|>", out, n);
      }
    } else if (!strcmp(role, "assistant")) {
      JNode *calls = json_get(m, "tool_calls");
      int has_calls = calls && calls->type == J_ARR && calls->child;
      if (!qwen && !glm && has_calls) { // Llama 3.1: one call per message
        for (JNode *c = calls->child; c; c = c->next) {
          JNode one = {.type = J_ARR, .child = c};
          JNode *nx = c->next;
          c->next = NULL;
          SBuf cb = {0};
          sb_puts(&cb, "");
          render_calls(&cb, &one, 0);
          c->next = nx;
          n = push_tpl(tk, "<|start_header_id|>assistant<|end_header_id|>\n\n", out, n);
          n = push_text(tk, cb.buf, out, n);
          n = push_tpl(tk, "<|eot_id|>", out, n);
          free(cb.buf);
        }
        continue;
      }
      n = push_tpl(tk, glm ? "<|assistant|>" : qwen ? "<|im_start|>assistant\n" : "<|start_header_id|>assistant<|end_header_id|>\n\n", out, n);
      if (glm) { // reasoning is kept in the history (clear_thinking = false)
        JNode *rc = json_get(m, "reasoning_content");
        n = push_tpl(tk, "<think>", out, n);
        if (rc && rc->type == J_STR) n = push_text(tk, rc->str, out, n);
        n = push_tpl(tk, "</think>", out, n);
        char *c = t.buf; // the template strips the content
        while (*c && isspace((unsigned char)*c)) c++;
        size_t cl = strlen(c);
        while (cl > 0 && isspace((unsigned char)c[cl - 1])) c[--cl] = '\0';
        memmove(t.buf, c, cl + 1);
        t.len = cl;
      } else if (ARCH == ARCH_QWEN35) { // the same think block the generation prompt opens
        JNode *rc = json_get(m, "reasoning_content");
        if (THINK && rc && rc->type == J_STR) {
          n = push_tpl(tk, "<think>\n", out, n);
          n = push_text(tk, rc->str, out, n);
          n = push_tpl(tk, "</think>\n\n", out, n);
        } else {
          n = push_tpl(tk, "<think>\n\n</think>\n\n", out, n);
        }
      }
      n = push_text(tk, t.buf, out, n);
      if (has_calls) { // template text: <tool_call> is a special token
        SBuf cb = {0};
        sb_puts(&cb, "");
        render_calls(&cb, calls, t.len > 0);
        n = push_tpl(tk, cb.buf, out, n);
        free(cb.buf);
      }
      n = push_tpl(tk, end_text(), out, n);
      n = push_tpl(tk, after_end(), out, n);
    } else {
      snprintf(err, errlen, "unknown role: %s", role);
      free(t.buf);
      return -1;
    }
  }
  if (in_tools && qwen) n = push_tpl(tk, "<|im_end|>\n", out, n);
  free(t.buf);
  if (glm) {
    n = push_tpl(tk, "<|assistant|>", out, n);
    n = push_tpl(tk, think_open(), out, n);
  } else if (qwen) {
    n = push_tpl(tk, "<|im_start|>assistant\n", out, n);
    if (ARCH == ARCH_QWEN35) n = push_tpl(tk, think_open(), out, n);
  } else {
    n = push_tpl(tk, "<|start_header_id|>assistant<|end_header_id|>\n\n", out, n);
  }
  return n;
}

// {"tools": [{name, description, parameters}, ...]} from the request's OpenAI tool list
static int request_tools(JNode *tools, Tools *out) {
  memset(out, 0, sizeof(*out));
  if (!tools || tools->type != J_ARR || !tools->child) return 0;
  JNode *root = calloc(1, sizeof(JNode)), *arr = calloc(1, sizeof(JNode));
  root->type = J_OBJ;
  arr->type = J_ARR;
  arr->key = strdup("tools");
  root->child = arr;
  JNode **tail = &arr->child;
  for (JNode *t = tools->child; t; t = t->next) {
    JNode *fn = json_get(t, "function");
    if (!fn) fn = t; // also accept a bare {name, description, parameters}
    JNode *name = json_get(fn, "name");
    if (fn->type != J_OBJ || !name || name->type != J_STR) continue;
    JNode *c = jdup(fn);
    *tail = c;
    tail = &c->next;
  }
  out->root = root;
  return arr->child != NULL;
}

// an in-memory model-state snapshot for tokens [0..n): the position-indexed cache region plus, for
// recurrent models, the recurrent state. Sessions keep one (so swapping the model between sessions
// is a memcpy, and the throttled on-disk dump stays just persistence); system-prefix slots own one
// each (so several prefixes coexist no matter what the physical caches currently hold).
typedef struct {
  int n;           // tokens covered (0: none)
  float *region;   // sess_region_floats(model, n)
  float *S, *conv; // recurrent models: lin_S / conv_state at n
} SrvSnap;

// fill the snapshot from the model state (it must stand at n); allocates/grows the buffers
static int srv_snap_take(Transformer *tr, SrvSnap *sp, int n) {
  const Config *p = &tr->config;
  size_t rf = sess_region_floats(p, n);
  if (sp->n < n || !sp->region) {
    free(sp->region);
    sp->region = malloc(rf * sizeof(float));
  }
  if (!sp->region) { sp->n = 0; return 0; }
  sess_region_save(tr, n, sp->region);
  if (RECURRENT(p)) {
    size_t sf = hyb_state_floats(p) * sizeof(float), cf = hyb_conv_floats(p) * sizeof(float);
    if (!sp->S) {
      sp->S = malloc(sf);
      sp->conv = malloc(cf);
    }
    if (!sp->S) { sp->n = 0; return 0; }
    memcpy(sp->S, tr->state.lin_S, sf);
    memcpy(sp->conv, tr->state.conv_state, cf);
  }
  sp->n = n;
  return 1;
}
// restore the model state from the snapshot: afterwards the state stands at sp->n
static void srv_snap_load(Transformer *tr, const SrvSnap *sp) {
  const Config *p = &tr->config;
  sess_region_load(tr, sp->n, sp->region);
  if (RECURRENT(p)) {
    memcpy(tr->state.lin_S, sp->S, hyb_state_floats(p) * sizeof(float));
    memcpy(tr->state.conv_state, sp->conv, hyb_conv_floats(p) * sizeof(float));
    tr->state.next_pos = sp->n;
  }
}
static void srv_snap_free(SrvSnap *sp) {
  free(sp->region);
  free(sp->S);
  free(sp->conv);
  memset(sp, 0, sizeof *sp);
}

// A named conversation: the rendered token history (tok[0..n)) whose KV / state is valid in the
// model. The session whose state currently sits in the model is "live"; any other session is
// swapped in from its snapshot (memory, else the on-disk dump) on demand. "default" is the
// anonymous session.
typedef struct {
  char id[72];
  int *tok;
  int n, cap;
  int live; // the model state matches tok[0..n)
  long last_dump; // throttling: the on-disk state dump is expensive on a big session
  int dumped_n;
  long last_use;
  SrvSnap snap; // in-memory state snapshot for a fast swap-back (n == this->n when fresh)
} SrvSession;

#define SRV_MAX_SESS 64
static SrvSession g_sessions[SRV_MAX_SESS];
static int g_nsessions;
static SrvSession *g_live;
static SrvSession *g_pending_dump; // set by srv_chat, written to disk after the connection closes
static SrvSession g_fallback; // registry full / persistence off: one anonymous in-memory session

// cap the total in-memory snapshot bytes: free the least recently used snapshots (never keep1/2)
#define SRV_SNAP_BUDGET ((size_t)24 << 30)
static void srv_snap_budget(Transformer *tr, SrvSession *keep1, SrvSession *keep2) {
  const Config *p = &tr->config;
  for (;;) {
    size_t total = 0;
    SrvSession *old = NULL;
    for (int i = 0; i < g_nsessions; i++) {
      SrvSession *s = &g_sessions[i];
      if (s->snap.n > 0) {
        total += sess_region_floats(p, s->snap.n) * sizeof(float);
        if (RECURRENT(p)) total += (hyb_state_floats(p) + hyb_conv_floats(p)) * sizeof(float);
        if (s != keep1 && s != keep2 && (!old || s->last_use < old->last_use)) old = s;
      }
    }
    if (total <= SRV_SNAP_BUDGET || !old) return;
    srv_snap_free(&old->snap);
  }
}

// Agent clients (opencode and the like) send the same fat system prompt (tools + instructions) in
// every session. Its tokens are the prefix of every rendered prompt, so the full model state at its
// end is captured once and restored for any session that shares it: the first request of a new
// session then only processes its own tail instead of the whole system prompt. Slots own their
// state copies, so they coexist regardless of what the physical caches currently hold.
#define SRV_MAX_SYS 8
typedef struct {
  int n;        // prefix length in tokens (0: empty slot)
  int *tok;     // the rendered prefix tokens
  SrvSnap snap; // the full state at n (valid when snap.n == n)
} SrvSys;
static SrvSys g_sys[SRV_MAX_SYS];
static int g_sys_next; // circular hand for eviction

// the slot whose tokens match P[0..sys_len) with a complete state, or NULL
static SrvSys *srv_sys_find(const int *P, int sys_len) {
  for (int i = 0; i < SRV_MAX_SYS; i++)
    if (g_sys[i].n == sys_len && g_sys[i].snap.n == sys_len && g_sys[i].tok &&
        !memcmp(P, g_sys[i].tok, (size_t)sys_len * sizeof(int)))
      return &g_sys[i];
  return NULL;
}
// a slot for a new prefix: an empty one, else the circular victim
static SrvSys *srv_sys_slot(void) {
  for (int i = 0; i < SRV_MAX_SYS; i++)
    if (g_sys[i].n == 0) return &g_sys[i];
  SrvSys *v = &g_sys[g_sys_next];
  g_sys_next = (g_sys_next + 1) % SRV_MAX_SYS;
  return v;
}
// file name identity of a prefix: FNV-1a over its tokens
static unsigned srv_tok_hash(const int *tok, int n) {
  unsigned h = 2166136261u;
  for (int i = 0; i < n; i++) {
    h ^= (unsigned)tok[i];
    h *= 16777619u;
  }
  return h;
}

static SrvSession *srv_session(const char *id, int cap) {
  for (int i = 0; i < g_nsessions; i++)
    if (!strcmp(g_sessions[i].id, id)) return &g_sessions[i];
  if (g_nsessions >= SRV_MAX_SESS) return &g_fallback;
  SrvSession *S = &g_sessions[g_nsessions++];
  memset(S, 0, sizeof *S);
  snprintf(S->id, sizeof S->id, "%s", id);
  S->cap = cap;
  S->tok = malloc((size_t)cap * sizeof(int));
  return S;
}

// keep only [A-Za-z0-9._-] so the id is a safe file name
static void srv_sanitize_id(char *id) {
  for (char *c = id; *c; c++)
    if (!isalnum((unsigned char)*c) && *c != '-' && *c != '_' && *c != '.') *c = '_';
  id[63] = '\0';
}

static void srv_journal(const SrvSession *S, const char *body, int np, int gen, int reuse, const char *finish) {
  if (!SRV_SESS_DIR || S == &g_fallback) return;
  char path[1200];
  snprintf(path, sizeof path, "%s/%s.jsonl", SRV_SESS_DIR, S->id);
  struct stat st;
  if (!stat(path, &st) && st.st_size > (200 << 20)) { // keep one backup of an overgrown journal
    char old[1208];
    snprintf(old, sizeof old, "%s.1", path);
    rename(path, old);
  }
  FILE *f = fopen(path, "a");
  if (!f) return;
  fprintf(f, "{\"ts\": %ld, \"session\": \"%s\", \"prompt_tokens\": %d, \"completion_tokens\": %d, \"cached_tokens\": %d, "
             "\"finish\": \"%s\", \"body\": ",
          (long)time(NULL), S->id, np, gen, reuse, finish);
  fputs(body, f); // the raw request JSON, embedded as an object
  fputs("}\n", f);
  fclose(f);
}

static void srv_chat(Transformer *tr, Tokenizer *tk, Sampler *sp, const char *sess_hdr, int fd, const char *body) {
  const int ctx = tr->config.seq_len;
  JNode *req = json_parse(body, NULL);
  JNode *messages = req ? json_get(req, "messages") : NULL;
  if (!req || req->type != J_OBJ || !messages || messages->type != J_ARR || !messages->child) {
    srv_error(fd, 400, "invalid_request_error", "the body must be a JSON object with a non-empty messages array");
    json_free(req);
    return;
  }
  JNode *v;
  if ((v = json_get(req, "n")) && v->type == J_NUM && atoi(v->str) != 1) {
    srv_error(fd, 400, "invalid_request_error", "only n = 1 is supported");
    json_free(req);
    return;
  }
  // session identity: a body field wins, then the client's session headers; "default" if none
  char sid[72] = "default";
  JNode *sj = json_get(req, "session_id");
  if (!sj) sj = json_get(req, "conversation_id");
  if (sj && sj->type == J_STR && sj->str[0]) snprintf(sid, sizeof sid, "%s", sj->str);
  else if (sess_hdr && sess_hdr[0]) snprintf(sid, sizeof sid, "%s", sess_hdr);
  srv_sanitize_id(sid);
  SrvSession *S = srv_session(sid, tr->config.seq_len);
  S->last_use = time(NULL);
  if (S != g_live) { // the model holds another session's state: swap ours in
    if (g_live && g_live != &g_fallback && g_live->n > 0) // keep the outgoing state for a fast swap-back
      srv_snap_take(tr, &g_live->snap, g_live->n);
    int loaded = 0, from_mem = 0;
    if (S->snap.n == S->n && S->n > 0) { // fresh in-memory snapshot: a memcpy swap
      srv_snap_load(tr, &S->snap);
      loaded = S->n;
      from_mem = 1;
    } else if (SRV_SESS_DIR && S != &g_fallback) {
      char path[1200];
      snprintf(path, sizeof path, "%s/%s.bin", SRV_SESS_DIR, S->id);
      loaded = sess_read(tr, S->tok, S->cap, path);
    }
    if (loaded > 0) {
      fprintf(stderr, "[server] session %s: restored %d tokens from %s\n", S->id, loaded, from_mem ? "memory" : "disk");
      S->n = loaded;
      S->live = 1;
    } else {
      if (S->n > 0) fprintf(stderr, "[server] session %s: state gone, re-processing from scratch\n", S->id);
      S->n = 0; // no dump (or it does not match this model): the prompt below re-processes everything
    }
    if (g_live) g_live->live = 0;
    g_live = S;
  }
  // per-request settings (restored at the end)
  const float temp0 = sp->temperature, topp0 = sp->topp;
  const int think0 = THINK;
  if ((v = json_get(req, "temperature")) && v->type == J_NUM) sp->temperature = atof(v->str);
  if ((v = json_get(req, "top_p")) && v->type == J_NUM) sp->topp = atof(v->str);
  if (sp->topp <= 0.0f || sp->topp > 1.0f) sp->topp = 1.0f;
  if ((v = json_get(req, "seed")) && v->type == J_NUM) sp->rng_state = (unsigned long long)atoll(v->str) + 1;
  if ((v = json_get(req, "reasoning_effort")) && v->type == J_STR) THINK = strcmp(v->str, "none") && strcmp(v->str, "minimal");
  JNode *kw = json_get(req, "chat_template_kwargs");
  if (kw && (v = json_get(kw, "enable_thinking")) && v->type == J_BOOL) THINK = v->boolean;
  int max_gen = 0;
  if ((v = json_get(req, "max_completion_tokens")) && v->type == J_NUM) max_gen = atoi(v->str);
  else if ((v = json_get(req, "max_tokens")) && v->type == J_NUM) max_gen = atoi(v->str);
  JNode *so = json_get(req, "stream_options");
  int include_usage = so && (v = json_get(so, "include_usage")) && v->type == J_BOOL && v->boolean;

  SrvReq r = {.fd = fd};
  r.stream = (v = json_get(req, "stream")) && v->type == J_BOOL && v->boolean;
  JNode *stop = json_get(req, "stop");
  char *stops[8];
  if (stop && stop->type == J_STR && stop->str[0]) stops[r.n_stops++] = stop->str;
  else if (stop && stop->type == J_ARR)
    for (JNode *s = stop->child; s && r.n_stops < 8; s = s->next)
      if (s->type == J_STR && s->str[0]) stops[r.n_stops++] = s->str;
  r.stops = stops;
  for (int i = 0; i < r.n_stops; i++) if (strlen(stops[i]) > r.max_stop) r.max_stop = strlen(stops[i]);

  Tools rt;
  JNode *tc = json_get(req, "tool_choice");
  int use_tools = !(tc && tc->type == J_STR && !strcmp(tc->str, "none"));
  g_tools = use_tools && request_tools(json_get(req, "tools"), &rt) ? &rt : NULL;

  char err[256];
  int *P = malloc(((size_t)strlen(body) + 65536) * sizeof(int));
  int sys_len = 0;
  int np = render_chat(tk, messages, P, err, sizeof err, &sys_len);
  int status = 0;
  if (np < 0) { srv_error(fd, 400, "invalid_request_error", err); status = 1; }
  else if (np + 1 >= ctx) {
    snprintf(err, sizeof err, "the prompt is %d tokens, the context window is %d (-c)", np, ctx);
    srv_error(fd, 400, "context_length_exceeded", err);
    status = 1;
  }
  if (status) goto done;

  // for a stream: answer with the headers immediately and keep the connection warm with SSE
  // comment lines while the prompt is being processed (prefill can take a minute)
  snprintf(r.id, sizeof r.id, "chatcmpl-%lx%04x", (unsigned long)time(NULL), (unsigned)(rand() & 0xffff));
  r.created = time(NULL);
  sb_puts(&r.content, "");
  sb_puts(&r.reasoning, "");
  ReplySink sink = {.text = srv_text, .ctx = &r, .max_gen = max_gen, .stop = r.failed};
  volatile int hb_stop = 0, hb_failed = 0;
  SrvHb hb = {.fd = fd, .stop = &hb_stop, .failed = &hb_failed};
  pthread_t hbt;
  int hb_on = 0;
  if (r.stream) {
    const char *hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                      "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    srv_write(&r, hdr, strlen(hdr));
    SBuf b = {0};
    srv_chunk_head(&r, &b);
    sb_puts(&b, ", \"choices\": [{\"index\": 0, \"delta\": {\"role\": \"assistant\", \"content\": \"\"}, \"finish_reason\": null}]}\n\n");
    srv_write(&r, b.buf, b.len);
    free(b.buf);
    if (!r.failed) {
      hb_on = 1;
      r.hb_stop = &hb_stop;
      pthread_create(&hbt, NULL, srv_heartbeat, &hb);
    }
  }

  // reuse the longest common prefix of the session's token history (GLM and other recurrent
  // models can only resume where the state stands: the end of the history, the snapshot, or 0);
  // the shared system prefix of agent clients is reused across sessions (state restore / KV)
  long t0 = time_in_ms();
  int lcp = 0;
  while (lcp < S->n && lcp < np && S->tok[lcp] == P[lcp]) lcp++;
  const int rec = RECURRENT(&tr->config);
  SrvSys *sys_slot = sys_len > 1 && sys_len < np ? srv_sys_find(P, sys_len) : NULL;
  int sys_cap = sys_len > 1 && sys_len < np && !sys_slot; // a new distinct system prefix
  int reuse;
  if (!rec) {
    reuse = lcp < np ? lcp : np - 1; // at least one token is fed to get the logits
    if (sys_slot && reuse < sys_len) reuse = sys_len; // the slot restores the shared prefix's caches
  } else if (sys_slot && S->n < sys_len) {
    srv_snap_load(tr, &sys_slot->snap); // state stands at sys_len now
    reuse = sys_len; // a fresh session with the shared system prefix resumes right after it
  } else if (lcp == S->n && S->n < np) {
    reuse = S->n; // the state is exactly at the end of the shared prefix
  } else if (tr->state.snap_S && tr->state.snap_pos > 0 && tr->state.snap_pos <= lcp &&
             tr->state.snap_pos < np && hybrid_restore(tr, tr->state.snap_pos)) {
    reuse = tr->state.snap_pos; // back to the end of the previous prompt
  } else if (sys_slot) {
    srv_snap_load(tr, &sys_slot->snap);
    reuse = sys_len; // a diverged session with the same system prefix still beats a full re-feed
  } else {
    reuse = 0;
  }
  if (S->n > 0 && reuse < S->n) {
    // the request does not continue this session's history (a stray client reused the id, or the
    // history was edited): the session restarts from this request (the shared KV prefix is reused,
    // the new branch overwrites the tail). Keep the previous state on disk under ".fork" — but only
    // when a meaningful part of the history matched: opencode sends its title request and its agent
    // request under one session id with different system prompts (they share ~10 tokens), and that
    // throwaway state is not worth a fork file on every session start
    int keep_fork = lcp >= 256 || lcp * 4 >= S->n;
    if (keep_fork && SRV_SESS_DIR && S != &g_fallback) {
      char old[1208], fork[1216];
      snprintf(old, sizeof old, "%s/%s.bin", SRV_SESS_DIR, S->id);
      snprintf(fork, sizeof fork, "%s/%s.fork.bin", SRV_SESS_DIR, S->id);
      if (rename(old, fork) == 0)
        fprintf(stderr, "[server] session %s: diverging request (shared %d of %d tok), the session restarts; "
                "previous state kept at %s.fork.bin\n", S->id, lcp, S->n, S->id);
    } else if (!keep_fork) {
      fprintf(stderr, "[server] session %s: diverging request (shared %d of %d tok), the session restarts\n",
              S->id, lcp, S->n);
    }
  }
  if (reuse == sys_len && S->n < sys_len)
    memcpy(S->tok, P, (size_t)sys_len * sizeof(int)); // the session inherits the shared system prefix
  if (DRAFT_POS > reuse) DRAFT_POS = reuse;
  ChatHistory h = {.tok = S->tok, .n = reuse, .cap = ctx, .sys_len = 0, .turn = NULL, .n_turns = 0};
  SrvSys *cap_slot = NULL;
  if (sys_cap) { // remember the new system prefix (its full state is captured during the feed below)
    cap_slot = srv_sys_slot();
    if (cap_slot->n < sys_len) {
      free(cap_slot->tok);
      cap_slot->tok = malloc((size_t)sys_len * sizeof(int));
    }
    if (cap_slot->tok) {
      memcpy(cap_slot->tok, P, (size_t)sys_len * sizeof(int));
      cap_slot->n = sys_len;
      cap_slot->snap.n = 0;
    } else {
      cap_slot = NULL;
    }
  }
  // recurrent: save the state one token before the end of the prompt, so that the same prompt again
  // (a retry, a regenerate) or any continuation of it can start from there; a new system prefix is
  // captured at its end on the way (a copy of the full state, so feeding on cannot corrupt it).
  // g_srv_abort (set by the keep-alive thread when the client goes away) stops the prefill between
  // blocks: the session keeps the processed prefix, a retry feeds only the rest.
  g_srv_abort = 0;
  if (rec && np - 1 > reuse) {
    if (cap_slot && reuse < sys_len) {
      if (chat_feed(tr, &h, P + reuse, sys_len - reuse)) {
        if (srv_snap_take(tr, &cap_slot->snap, sys_len) && SRV_SESS_DIR) { // keep the prefix warm across restarts
          char path[1200];
          snprintf(path, sizeof path, "%s/.sys-%08x.bin", SRV_SESS_DIR, srv_tok_hash(P, sys_len));
          sess_write(tr, P, sys_len, path); // best effort
        }
      }
      chat_feed(tr, &h, P + sys_len, np - 1 - sys_len);
    } else {
      chat_feed(tr, &h, P + reuse, np - 1 - reuse);
    }
  }
  if (rec && !g_srv_abort && h.n == np - 1) hybrid_snapshot(tr);
  float *logits = g_srv_abort ? NULL : chat_feed(tr, &h, P + h.n, np - h.n);
  long t1 = time_in_ms();
  if (!logits) { // client gone mid-prefill: keep the processed prefix (the snapshot below), skip the reply
    fprintf(stderr, "[server] prompt %d tok: client gone at %d tok, prefill aborted\n", np, h.n);
    S->n = h.n;
    r.failed = 1;
  }

  g_sink = &sink;
  JNode *calls[MAX_CALLS_PER_REPLY];
  int ended = 0, gen = 0, n_calls = 0;
  if (!r.failed) n_calls = chat_reply(tr, tk, sp, &h, logits, &ended, &gen, calls);
  g_sink = NULL;
  if (hb_on) {
    hb_stop = 1;
    pthread_join(hbt, NULL);
    r.hb_stop = NULL;
    if (hb_failed) r.failed = 1;
  }
  S->n = h.n;
  long t2 = time_in_ms();
  if (r.stop_hit) { // the reply was cut inside the text: no calls
    for (int i = 0; i < n_calls; i++) json_free(calls[i]);
    n_calls = 0;
  }
  const char *finish = n_calls ? "tool_calls" : (ended || r.stop_hit) ? "stop" : "length";
  if (n_calls) { // text around tool calls is mostly whitespace
    while (r.content.len > 0 && isspace((unsigned char)r.content.buf[r.content.len - 1])) r.content.buf[--r.content.len] = '\0';
  }
  // the tool calls as OpenAI JSON
  SBuf tcj = {0};
  sb_puts(&tcj, "[");
  for (int i = 0; i < n_calls; i++) {
    char cid[40];
    snprintf(cid, sizeof cid, "call_%08x%08x", (unsigned)rand(), (unsigned)rand());
    JNode *params = json_get(calls[i], "parameters");
    SBuf args = {0};
    if (params) json_write(&args, params, -1, 0); else sb_puts(&args, "{}");
    char hd[160];
    snprintf(hd, sizeof hd, "%s{\"index\": %d, \"id\": \"%s\", \"type\": \"function\", \"function\": {\"name\": ", i ? ", " : "", i, cid);
    sb_puts(&tcj, hd);
    json_write_string(&tcj, json_get(calls[i], "name")->str);
    sb_puts(&tcj, ", \"arguments\": ");
    json_write_string(&tcj, args.buf);
    sb_puts(&tcj, "}}");
    free(args.buf);
    json_free(calls[i]);
  }
  sb_puts(&tcj, "]");
  char usage[256];
  snprintf(usage, sizeof usage, "{\"prompt_tokens\": %d, \"completion_tokens\": %d, \"total_tokens\": %d, "
           "\"prompt_tokens_details\": {\"cached_tokens\": %d}}", np, gen, np + gen, reuse);

  if (!r.stream) { utf8_clean(r.content.buf); utf8_clean(r.reasoning.buf); }
  if (r.stream) {
    srv_flush(&r, 1, 1);
    srv_flush(&r, 0, 1);
    SBuf b = {0};
    if (n_calls) {
      srv_chunk_head(&r, &b);
      sb_puts(&b, ", \"choices\": [{\"index\": 0, \"delta\": {\"tool_calls\": ");
      sb_puts(&b, tcj.buf);
      sb_puts(&b, "}, \"finish_reason\": null}]}\n\n");
    }
    srv_chunk_head(&r, &b);
    sb_puts(&b, ", \"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"");
    sb_puts(&b, finish);
    sb_puts(&b, "\"}]}\n\n");
    if (include_usage) {
      srv_chunk_head(&r, &b);
      sb_puts(&b, ", \"choices\": [], \"usage\": ");
      sb_puts(&b, usage);
      sb_puts(&b, "}\n\n");
    }
    sb_puts(&b, "data: [DONE]\n\n");
    srv_write(&r, b.buf, b.len);
    free(b.buf);
  } else {
    SBuf b = {0};
    sb_puts(&b, "{\"id\": \"");
    sb_puts(&b, r.id);
    char hd[128];
    snprintf(hd, sizeof hd, "\", \"object\": \"chat.completion\", \"created\": %ld, \"model\": ", r.created);
    sb_puts(&b, hd);
    json_write_string(&b, SRV_MODEL);
    sb_puts(&b, ", \"choices\": [{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": ");
    if (n_calls && r.content.len == 0) sb_puts(&b, "null");
    else json_write_string(&b, r.content.buf);
    if (r.reasoning.len) { sb_puts(&b, ", \"reasoning_content\": "); json_write_string(&b, r.reasoning.buf); }
    if (n_calls) { sb_puts(&b, ", \"tool_calls\": "); sb_puts(&b, tcj.buf); }
    sb_puts(&b, "}, \"finish_reason\": \"");
    sb_puts(&b, finish);
    sb_puts(&b, "\"}], \"usage\": ");
    sb_puts(&b, usage);
    sb_puts(&b, "}");
    srv_respond(fd, 200, "application/json", b.buf);
    free(b.buf);
  }
  fprintf(stderr, "[server] prompt %d tok (cached %d) %.0f tok/s | reply %d tok %.1f tok/s | %s%s%s\n", np, reuse,
          (np - reuse) * 1000.0 / (t1 - t0 > 0 ? t1 - t0 : 1), gen, gen * 1000.0 / (t2 - t1 > 0 ? t2 - t1 : 1), finish,
          r.stream ? ", stream" : "", r.failed ? ", client disconnected" : "");
  if (S != &g_fallback && r.content.buf) { // durable record + restorable state (best effort)
    if (!r.failed) srv_journal(S, body, np, gen, reuse, finish);
    // a snapshot costs seconds on a big session: throttle it; the memory snapshot makes the next
    // session swap a memcpy, the on-disk dump (written after the connection closes) survives restarts
    long now = time(NULL);
    if (S->n > 0 && (S->last_dump == 0 || S->n - S->dumped_n >= 2048 || now - S->last_dump >= 30)) {
      srv_snap_take(tr, &S->snap, S->n);
      srv_snap_budget(tr, S, g_live);
      if (SRV_SESS_DIR) g_pending_dump = S;
    }
  }
  free(tcj.buf);
done:
  free(r.content.buf);
  free(r.reasoning.buf);
  free(P);
  if (g_tools == &rt) json_free(rt.root);
  g_tools = NULL;
  sp->temperature = temp0;
  sp->topp = topp0;
  THINK = think0;
  json_free(req);
}

// read one HTTP request; returns 0 on a malformed / oversized request (already answered);
// *sess (optional) receives the client's session header
static int srv_read(int fd, char *method, char *path, char **body, int *authorized, char *sess, size_t sesslen) {
  SBuf b = {0};
  char chunk[65536];
  char *hend = NULL;
  while (!hend) {
    ssize_t k = recv(fd, chunk, sizeof chunk, 0);
    if (k <= 0) { free(b.buf); return 0; }
    sb_putn(&b, chunk, k);
    hend = strstr(b.buf, "\r\n\r\n");
    if (!hend && b.len > (1 << 16)) { srv_error(fd, 413, "invalid_request_error", "headers too large"); free(b.buf); return 0; }
  }
  *hend = '\0';
  size_t hlen = hend - b.buf + 4;
  if (sscanf(b.buf, "%15s %511s", method, path) != 2) { free(b.buf); return 0; }
  long clen = 0;
  int expect = 0;
  *authorized = SRV_KEY == NULL;
  if (sess) sess[0] = '\0';
  for (char *line = strstr(b.buf, "\r\n"); line; line = strstr(line, "\r\n")) {
    line += 2;
    if (!strncasecmp(line, "Content-Length:", 15)) clen = atol(line + 15);
    else if (!strncasecmp(line, "Expect:", 7) && strstr(line, "100-continue")) expect = 1;
    else if (SRV_KEY && !strncasecmp(line, "Authorization:", 14)) {
      const char *v = line + 14;
      while (*v == ' ') v++;
      if (!strncasecmp(v, "Bearer ", 7)) v += 7;
      size_t kl = strlen(SRV_KEY);
      if (!strncmp(v, SRV_KEY, kl) && (v[kl] == '\r' || v[kl] == '\0')) *authorized = 1;
    } else if (sess && sess[0] == '\0') {
      const char *sh = NULL;
      if (!strncasecmp(line, "x-opencode-session-id:", 22)) sh = line + 22;
      else if (!strncasecmp(line, "x-session-id:", 13)) sh = line + 13;
      else if (!strncasecmp(line, "x-conversation-id:", 18)) sh = line + 18;
      if (sh) {
        while (*sh == ' ') sh++;
        size_t n = strcspn(sh, "\r\n");
        if (n >= sesslen) n = sesslen - 1;
        memcpy(sess, sh, n);
        sess[n] = '\0';
      }
    }
  }
  if (clen < 0 || clen > SRV_MAX_BODY) { srv_error(fd, 413, "invalid_request_error", "request body too large"); free(b.buf); return 0; }
  if (expect) { SrvReq r = {.fd = fd}; srv_write(&r, "HTTP/1.1 100 Continue\r\n\r\n", 25); }
  char *bd = malloc(clen + 1);
  size_t have = b.len - hlen;
  if (have > (size_t)clen) have = clen;
  memcpy(bd, b.buf + hlen, have);
  free(b.buf);
  while (have < (size_t)clen) {
    ssize_t k = recv(fd, bd + have, clen - have, 0);
    if (k <= 0) { free(bd); return 0; }
    have += k;
  }
  bd[clen] = '\0';
  *body = bd;
  return 1;
}

static void server(Transformer *tr, Tokenizer *tk, Sampler *sp) {
  signal(SIGPIPE, SIG_IGN);
  init_special_tokens(tk);
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(SRV_PORT)};
  if (inet_pton(AF_INET, SRV_HOST, &a.sin_addr) != 1) { fprintf(stderr, "bad host %s (IPv4 address expected)\n", SRV_HOST); exit(EXIT_FAILURE); }
  if (bind(ls, (struct sockaddr *)&a, sizeof a) < 0 || listen(ls, 128) < 0) {
    fprintf(stderr, "cannot listen on %s:%d: %s\n", SRV_HOST, SRV_PORT, strerror(errno));
    exit(EXIT_FAILURE);
  }
  g_fallback.id[0] = '\0';
  g_fallback.cap = tr->config.seq_len;
  g_fallback.tok = malloc((size_t)tr->config.seq_len * sizeof(int));
  if (SRV_SESS_DIR && SRV_SESS_DIR[0]) { // make the sessions directory, list what is on disk
    if (mkdir(SRV_SESS_DIR, 0755) != 0 && errno != EEXIST)
      fprintf(stderr, "[server] warning: cannot create %s: %s\n", SRV_SESS_DIR, strerror(errno));
    DIR *d = opendir(SRV_SESS_DIR);
    int nfiles = 0;
    if (d) {
      struct dirent *e;
      while ((e = readdir(d))) {
        if (strstr(e->d_name, ".bin") && strncmp(e->d_name, ".sys-", 5)) nfiles++;
        if (!strncmp(e->d_name, ".sys-", 5) && g_nsessions < SRV_MAX_SESS + SRV_MAX_SYS) {
          char path[1200];
          snprintf(path, sizeof path, "%s/%s", SRV_SESS_DIR, e->d_name);
          int *tok = malloc((size_t)tr->config.seq_len * sizeof(int));
          int n = sess_read(tr, tok, tr->config.seq_len, path); // a persisted system prefix
          if (n > 1) {
            SrvSys *slot = srv_sys_slot();
            srv_snap_free(&slot->snap);
            free(slot->tok);
            slot->tok = tok;
            slot->n = n;
            srv_snap_take(tr, &slot->snap, n);
            fprintf(stderr, "[server] system prefix: %d tokens loaded from %s\n", n, e->d_name);
            tok = NULL;
          }
          free(tok);
        }
      }
      closedir(d);
    }
    fprintf(stderr, "[server] sessions in %s: %d state file%s on disk\n", SRV_SESS_DIR, nfiles, nfiles == 1 ? "" : "s");
  }
  fprintf(stderr, "[server] model %s, context %d, listening on http://%s:%d/v1%s\n", SRV_MODEL, tr->config.seq_len, SRV_HOST,
          SRV_PORT, SRV_KEY ? " (API key required)" : "");
  for (;;) {
    int fd = accept(ls, NULL, NULL);
    if (fd < 0) continue;
    struct timeval tv = {.tv_sec = 60};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    char method[16], path[512], *body = NULL, shead[80];
    int authorized = 0;
    if (srv_read(fd, method, path, &body, &authorized, shead, sizeof shead)) {
      char *q = strchr(path, '?');
      if (q) *q = '\0';
      if (!strcmp(method, "OPTIONS")) {
        const char *h = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                        "Access-Control-Allow-Headers: Authorization, Content-Type\r\nConnection: close\r\n\r\n";
        SrvReq r = {.fd = fd};
        srv_write(&r, h, strlen(h));
      } else if (!strcmp(path, "/health")) {
        srv_respond(fd, 200, "application/json", "{\"status\": \"ok\"}");
      } else if (!authorized) {
        srv_error(fd, 401, "invalid_request_error", "invalid API key");
      } else if (!strcmp(method, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/models"))) {
        SBuf b = {0};
        sb_puts(&b, "{\"object\": \"list\", \"data\": [{\"id\": ");
        json_write_string(&b, SRV_MODEL);
        char t[160];
        snprintf(t, sizeof t, ", \"object\": \"model\", \"created\": %ld, \"owned_by\": \"numallama\", \"context_length\": %d}]}",
                 (long)time(NULL), tr->config.seq_len);
        sb_puts(&b, t);
        srv_respond(fd, 200, "application/json", b.buf);
        free(b.buf);
      } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/sessions")) { // known sessions + what is on disk
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
        if (SRV_SESS_DIR) {
          DIR *d = opendir(SRV_SESS_DIR);
          if (d) {
            struct dirent *e;
            while ((e = readdir(d))) {
              if (strncmp(e->d_name, ".sys-", 5) == 0) continue; // system-prefix slots are not sessions
              char *dot = strstr(e->d_name, ".bin");
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
      } else if (!strcmp(method, "POST") && (!strcmp(path, "/v1/chat/completions") || !strcmp(path, "/chat/completions"))) {
        srv_chat(tr, tk, sp, shead, fd, body);
      } else {
        srv_error(fd, 404, "invalid_request_error", "unknown endpoint");
      }
    }
    free(body);
    shutdown(fd, SHUT_WR);
    close(fd);
    if (g_pending_dump) { // the client is gone: now spend the seconds a state dump takes
      SrvSession *S = g_pending_dump;
      g_pending_dump = NULL;
      char path[1200];
      snprintf(path, sizeof path, "%s/%s.bin", SRV_SESS_DIR, S->id);
      struct statvfs vfs;
      int low = statvfs(SRV_SESS_DIR, &vfs) == 0 &&
                (uint64_t)vfs.f_bavail * vfs.f_frsize < sess_dump_bytes(tr, S->n) * 5 / 4;
      if (low) {
        fprintf(stderr, "[server] session %s: %.1f GB dump skipped (disk full)\n", S->id, sess_dump_bytes(tr, S->n) / 1e9);
      } else {
        long d0 = time_in_ms();
        int okw = sess_write(tr, S->tok, S->n, path);
        S->last_dump = time(NULL);
        S->dumped_n = okw ? S->n : 0;
        fprintf(stderr, "[server] session %s: state %d tok%s in %.2fs\n", S->id, S->n, okw ? " saved" : " NOT saved",
                (time_in_ms() - d0) / 1000.0);
      }
    }
  }
}
