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
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <strings.h>
#include <sys/socket.h>

static const char *SRV_HOST = "127.0.0.1";
static int SRV_PORT = 8080;
static const char *SRV_KEY = NULL; // -A: required "Authorization: Bearer <key>"
static const char *SRV_MODEL = "numallama";
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

// the whole conversation as prompt tokens, ending with the assistant header. -1 + err on bad input
static int render_chat(Tokenizer *tk, JNode *messages, int *out, char *err, size_t errlen) {
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

typedef struct {
  int *tok; // tokens whose KV entries are valid; hybrid: the state stands at n
  int n, cap;
} SrvCache;

static void srv_chat(Transformer *tr, Tokenizer *tk, Sampler *sp, SrvCache *C, int fd, const char *body) {
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
  int np = render_chat(tk, messages, P, err, sizeof err);
  int status = 0;
  if (np < 0) { srv_error(fd, 400, "invalid_request_error", err); status = 1; }
  else if (np + 1 >= ctx) {
    snprintf(err, sizeof err, "the prompt is %d tokens, the context window is %d (-c)", np, ctx);
    srv_error(fd, 400, "context_length_exceeded", err);
    status = 1;
  }
  if (status) goto done;

  // reuse the longest common prefix of the KV cache
  long t0 = time_in_ms();
  int lcp = 0;
  while (lcp < C->n && lcp < np && C->tok[lcp] == P[lcp]) lcp++;
  int reuse;
  RunState *s = &tr->state;
  if (!tr->config.hybrid) {
    reuse = lcp < np ? lcp : np - 1; // at least one token is fed to get the logits
  } else if (lcp == C->n && C->n < np) {
    reuse = C->n; // the state is exactly at the end of the shared prefix
  } else if (s->snap_S && s->snap_pos > 0 && s->snap_pos <= lcp && s->snap_pos < np && hybrid_restore(tr, s->snap_pos)) {
    reuse = s->snap_pos; // back to the end of the previous prompt
  } else {
    reuse = 0;
  }
  if (DRAFT_POS > reuse) DRAFT_POS = reuse;
  ChatHistory h = {.tok = C->tok, .n = reuse, .cap = ctx, .sys_len = 0, .turn = NULL, .n_turns = 0};
  // hybrid: save the state one token before the end of the prompt, so that the same prompt again
  // (a retry, a regenerate) or any continuation of it can start from there
  if (tr->config.hybrid && np - 1 > reuse) chat_feed(tr, &h, P + reuse, np - 1 - reuse);
  if (tr->config.hybrid && h.n == np - 1) hybrid_snapshot(tr);
  float *logits = chat_feed(tr, &h, P + h.n, np - h.n);
  long t1 = time_in_ms();

  snprintf(r.id, sizeof r.id, "chatcmpl-%lx%04x", (unsigned long)time(NULL), (unsigned)(rand() & 0xffff));
  r.created = time(NULL);
  sb_puts(&r.content, "");
  sb_puts(&r.reasoning, "");
  if (r.stream) {
    const char *hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                      "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    srv_write(&r, hdr, strlen(hdr));
    SBuf b = {0};
    srv_chunk_head(&r, &b);
    sb_puts(&b, ", \"choices\": [{\"index\": 0, \"delta\": {\"role\": \"assistant\", \"content\": \"\"}, \"finish_reason\": null}]}\n\n");
    srv_write(&r, b.buf, b.len);
    free(b.buf);
  }
  ReplySink sink = {.text = srv_text, .ctx = &r, .max_gen = max_gen, .stop = r.failed};
  g_sink = &sink;
  JNode *calls[MAX_CALLS_PER_REPLY];
  int ended = 0, gen = 0, n_calls = 0;
  if (!r.failed) n_calls = chat_reply(tr, tk, sp, &h, logits, &ended, &gen, calls);
  g_sink = NULL;
  C->n = h.n;
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

// read one HTTP request; returns 0 on a malformed / oversized request (already answered)
static int srv_read(int fd, char *method, char *path, char **body, int *authorized) {
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
  SrvCache C = {.tok = malloc((size_t)tr->config.seq_len * sizeof(int)), .n = 0, .cap = tr->config.seq_len};
  fprintf(stderr, "[server] model %s, context %d, listening on http://%s:%d/v1%s\n", SRV_MODEL, tr->config.seq_len, SRV_HOST,
          SRV_PORT, SRV_KEY ? " (API key required)" : "");
  for (;;) {
    int fd = accept(ls, NULL, NULL);
    if (fd < 0) continue;
    struct timeval tv = {.tv_sec = 60};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    char method[16], path[512], *body = NULL;
    int authorized = 0;
    if (srv_read(fd, method, path, &body, &authorized)) {
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
      } else if (!strcmp(method, "POST") && (!strcmp(path, "/v1/chat/completions") || !strcmp(path, "/chat/completions"))) {
        srv_chat(tr, tk, sp, &C, fd, body);
      } else {
        srv_error(fd, 404, "invalid_request_error", "unknown endpoint");
      }
    }
    free(body);
    shutdown(fd, SHUT_WR);
    close(fd);
  }
}
