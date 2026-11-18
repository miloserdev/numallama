/* Tool calling support for llama3.c chat: a tiny JSON parser, tools.json loading and
 * safe execution of whitelisted commands (fork/execvp, no shell).
 *
 * tools.json:
 * {
 *   "root": ".",              working directory for every tool; path arguments must stay inside
 *   "timeout_sec": 10,        kill the command after this many seconds
 *   "max_output": 3000,       bytes of output handed back to the model
 *   "tools": [
 *     { "name": "ls", "description": "...",
 *       "parameters": { "type": "object", "properties": { "path": {"type": "string", "default": "."} },
 *                       "required": [] },
 *       "command": ["ls", "-la", "--", "{path}"],     argv; an element that is exactly "{param}" is substituted
 *       "path_params": ["path"],                       values resolved inside root (no escaping it)
 *       "allow_dash": ["pattern"],                     values that may start with '-' (rejected by default)
 *       "confirm": false                               ask the user before running
 *     }, ...
 *   ]
 * }
 */
#ifndef LLAMA3C_TOOLS_H
#define LLAMA3C_TOOLS_H

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

// ----------------------------------------------------------------------------
// minimal JSON

enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ };

typedef struct JNode {
  int type;
  char *key;           // member name when inside an object
  char *str;           // J_STR value, or the raw text of a J_NUM
  int boolean;
  struct JNode *child; // first element / member
  struct JNode *next;  // next sibling
} JNode;

typedef struct {
  char *buf;
  size_t len, cap;
} SBuf;

static void sb_putn(SBuf *b, const char *s, size_t n) {
  if (b->len + n + 1 > b->cap) {
    while (b->len + n + 1 > b->cap) b->cap = b->cap ? b->cap * 2 : 256;
    b->buf = realloc(b->buf, b->cap);
  }
  memcpy(b->buf + b->len, s, n);
  b->len += n;
  b->buf[b->len] = '\0';
}
static void sb_puts(SBuf *b, const char *s) { sb_putn(b, s, strlen(s)); }
static void sb_putc(SBuf *b, char c) { sb_putn(b, &c, 1); }

static void json_free(JNode *n) {
  while (n) {
    JNode *next = n->next;
    json_free(n->child);
    free(n->key);
    free(n->str);
    free(n);
    n = next;
  }
}

static const char *json_ws(const char *p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  return p;
}

static void utf8_put(SBuf *b, unsigned cp) {
  char o[4];
  if (cp < 0x80) { o[0] = cp; sb_putn(b, o, 1); }
  else if (cp < 0x800) { o[0] = 0xC0 | (cp >> 6); o[1] = 0x80 | (cp & 0x3F); sb_putn(b, o, 2); }
  else if (cp < 0x10000) { o[0] = 0xE0 | (cp >> 12); o[1] = 0x80 | ((cp >> 6) & 0x3F); o[2] = 0x80 | (cp & 0x3F); sb_putn(b, o, 3); }
  else { o[0] = 0xF0 | (cp >> 18); o[1] = 0x80 | ((cp >> 12) & 0x3F); o[2] = 0x80 | ((cp >> 6) & 0x3F); o[3] = 0x80 | (cp & 0x3F); sb_putn(b, o, 4); }
}

static const char *json_parse_string(const char *p, char **out) {
  if (*p != '"') return NULL;
  p++;
  SBuf b = {0};
  sb_puts(&b, "");
  while (*p && *p != '"') {
    if (*p == '\\') {
      p++;
      switch (*p) {
      case '"': sb_putc(&b, '"'); break;
      case '\\': sb_putc(&b, '\\'); break;
      case '/': sb_putc(&b, '/'); break;
      case 'b': sb_putc(&b, '\b'); break;
      case 'f': sb_putc(&b, '\f'); break;
      case 'n': sb_putc(&b, '\n'); break;
      case 'r': sb_putc(&b, '\r'); break;
      case 't': sb_putc(&b, '\t'); break;
      case 'u': {
        unsigned cp = 0;
        for (int i = 1; i <= 4; i++) {
          char c = p[i];
          if (!isxdigit((unsigned char)c)) { free(b.buf); return NULL; }
          cp = cp * 16 + (isdigit((unsigned char)c) ? c - '0' : (tolower(c) - 'a' + 10));
        }
        p += 4;
        if (cp >= 0xD800 && cp < 0xDC00 && p[1] == '\\' && p[2] == 'u') { // surrogate pair
          unsigned lo = 0;
          int ok = 1;
          for (int i = 3; i <= 6; i++) {
            char c = p[i];
            if (!isxdigit((unsigned char)c)) { ok = 0; break; }
            lo = lo * 16 + (isdigit((unsigned char)c) ? c - '0' : (tolower(c) - 'a' + 10));
          }
          if (ok && lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); p += 6; }
        }
        utf8_put(&b, cp);
        break;
      }
      default: free(b.buf); return NULL;
      }
      p++;
    } else {
      sb_putc(&b, *p++);
    }
  }
  if (*p != '"') { free(b.buf); return NULL; }
  *out = b.buf;
  return p + 1;
}

static const char *json_parse_value(const char *p, JNode **out, int depth);

static const char *json_parse_value(const char *p, JNode **out, int depth) {
  if (depth > 64) return NULL;
  p = json_ws(p);
  JNode *n = calloc(1, sizeof(JNode));
  if (*p == '{' || *p == '[') {
    int obj = *p == '{';
    n->type = obj ? J_OBJ : J_ARR;
    p = json_ws(p + 1);
    JNode **tail = &n->child;
    if (*p == (obj ? '}' : ']')) { *out = n; return p + 1; }
    for (;;) {
      char *key = NULL;
      if (obj) {
        p = json_parse_string(json_ws(p), &key);
        if (!p) goto fail;
        p = json_ws(p);
        if (*p != ':') { free(key); goto fail; }
        p++;
      }
      JNode *c = NULL;
      p = json_parse_value(p, &c, depth + 1);
      if (!p) { free(key); goto fail; }
      c->key = key;
      *tail = c;
      tail = &c->next;
      p = json_ws(p);
      if (*p == ',') { p++; continue; }
      if (*p == (obj ? '}' : ']')) { *out = n; return p + 1; }
      goto fail;
    }
  } else if (*p == '"') {
    n->type = J_STR;
    p = json_parse_string(p, &n->str);
    if (!p) goto fail;
  } else if (!strncmp(p, "true", 4) || !strncmp(p, "false", 5)) {
    n->type = J_BOOL;
    n->boolean = *p == 't';
    p += n->boolean ? 4 : 5;
  } else if (!strncmp(p, "null", 4)) {
    n->type = J_NULL;
    p += 4;
  } else if (*p == '-' || isdigit((unsigned char)*p)) {
    const char *s = p;
    p++;
    while (isdigit((unsigned char)*p) || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-') p++;
    n->type = J_NUM;
    n->str = strndup(s, p - s);
  } else {
    goto fail;
  }
  *out = n;
  return p;
fail:
  json_free(n);
  return NULL;
}

static JNode *json_parse(const char *text, const char **end) {
  JNode *n = NULL;
  const char *p = json_parse_value(text, &n, 0);
  if (end) *end = p;
  return p ? n : NULL;
}

static JNode *json_get(JNode *obj, const char *key) {
  if (!obj || obj->type != J_OBJ) return NULL;
  for (JNode *c = obj->child; c; c = c->next)
    if (c->key && !strcmp(c->key, key)) return c;
  return NULL;
}

static void json_write_string(SBuf *b, const char *s) {
  sb_putc(b, '"');
  for (; *s; s++) {
    unsigned char c = *s;
    if (c == '"') sb_puts(b, "\\\"");
    else if (c == '\\') sb_puts(b, "\\\\");
    else if (c == '\n') sb_puts(b, "\\n");
    else if (c == '\r') sb_puts(b, "\\r");
    else if (c == '\t') sb_puts(b, "\\t");
    else if (c < 0x20) { char t[8]; snprintf(t, sizeof t, "\\u%04x", c); sb_puts(b, t); }
    else sb_putc(b, c);
  }
  sb_putc(b, '"');
}

// indent < 0: compact with ", " / ": " separators (like python json.dumps)
static void json_write(SBuf *b, JNode *n, int indent, int level) {
  switch (n->type) {
  case J_NULL: sb_puts(b, "null"); break;
  case J_BOOL: sb_puts(b, n->boolean ? "true" : "false"); break;
  case J_NUM: sb_puts(b, n->str); break;
  case J_STR: json_write_string(b, n->str); break;
  default: {
    int obj = n->type == J_OBJ;
    sb_putc(b, obj ? '{' : '[');
    if (!n->child) { sb_putc(b, obj ? '}' : ']'); break; }
    for (JNode *c = n->child; c; c = c->next) {
      if (indent >= 0) {
        sb_putc(b, '\n');
        for (int i = 0; i < indent * (level + 1); i++) sb_putc(b, ' ');
      }
      if (obj) { json_write_string(b, c->key); sb_puts(b, ": "); }
      json_write(b, c, indent, level + 1);
      if (c->next) sb_puts(b, indent >= 0 ? "," : ", ");
    }
    if (indent >= 0) {
      sb_putc(b, '\n');
      for (int i = 0; i < indent * level; i++) sb_putc(b, ' ');
    }
    sb_putc(b, obj ? '}' : ']');
  }
  }
}

// ----------------------------------------------------------------------------
// tools

typedef struct {
  JNode *root;          // parsed tools.json
  char workdir[PATH_MAX];
  int timeout_sec;
  int max_output;
} Tools;

__attribute__((unused)) static int tools_load(Tools *t, const char *path) {
  memset(t, 0, sizeof(*t));
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
  SBuf b = {0};
  char chunk[4096];
  size_t n;
  while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) sb_putn(&b, chunk, n);
  fclose(f);
  const char *end;
  t->root = b.buf ? json_parse(b.buf, &end) : NULL;
  free(b.buf);
  if (!t->root || t->root->type != J_OBJ || !json_get(t->root, "tools")) {
    fprintf(stderr, "%s: invalid tools file\n", path);
    return 0;
  }
  JNode *r = json_get(t->root, "root");
  // a relative root is relative to the directory of tools.json
  char base[PATH_MAX];
  snprintf(base, sizeof base, "%s", path);
  char *slash = strrchr(base, '/');
  if (slash) *slash = '\0'; else snprintf(base, sizeof base, ".");
  char want[PATH_MAX * 2];
  const char *rv = (r && r->type == J_STR) ? r->str : ".";
  if (rv[0] == '/') snprintf(want, sizeof want, "%s", rv);
  else snprintf(want, sizeof want, "%s/%s", base, rv);
  if (!realpath(want, t->workdir)) { fprintf(stderr, "tools root %s: %s\n", want, strerror(errno)); return 0; }
  JNode *to = json_get(t->root, "timeout_sec"), *mo = json_get(t->root, "max_output");
  t->timeout_sec = to && to->type == J_NUM ? atoi(to->str) : 10;
  t->max_output = mo && mo->type == J_NUM ? atoi(mo->str) : 3000;
  return 1;
}

static JNode *tools_find(Tools *t, const char *name) {
  for (JNode *c = json_get(t->root, "tools")->child; c; c = c->next) {
    JNode *nm = json_get(c, "name");
    if (nm && nm->type == J_STR && !strcmp(nm->str, name)) return c;
  }
  return NULL;
}

// the function list as shown to the model: one {"type": "function", "function": {...}} object per
// tool, each followed by `sep`; indent 4 (Llama 3 template) or -1 = one line (Qwen2.5 template)
static char *tools_prompt(Tools *t, int indent, const char *before, const char *after) {
  SBuf b = {0};
  sb_puts(&b, "");
  for (JNode *c = json_get(t->root, "tools")->child; c; c = c->next) {
    JNode fn = {.type = J_OBJ}, *name = json_get(c, "name"), *desc = json_get(c, "description"), *params = json_get(c, "parameters");
    // {"type": "function", "function": {"name", "description", "parameters"}}
    JNode k_name = *name, k_desc, k_params;
    k_name.key = "name"; k_name.next = NULL;
    JNode *last = &k_name;
    if (desc) { k_desc = *desc; k_desc.key = "description"; k_desc.next = NULL; last->next = &k_desc; last = &k_desc; }
    if (params) { k_params = *params; k_params.key = "parameters"; k_params.next = NULL; last->next = &k_params; }
    fn.child = &k_name;
    fn.key = "function";
    JNode type = {.type = J_STR, .key = "type", .str = "function", .next = &fn};
    JNode wrap = {.type = J_OBJ, .child = &type};
    sb_puts(&b, before);
    json_write(&b, &wrap, indent, 0);
    sb_puts(&b, after);
  }
  return b.buf;
}

// value of a call argument as a string (strings as-is, other JSON compactly)
static char *arg_to_string(JNode *v) {
  if (v->type == J_STR) return strdup(v->str);
  SBuf b = {0};
  json_write(&b, v, -1, 0);
  return b.buf;
}

static int in_list(JNode *tool, const char *field, const char *name) {
  JNode *pp = json_get(tool, field);
  if (!pp || pp->type != J_ARR) return 0;
  for (JNode *c = pp->child; c; c = c->next)
    if (c->type == J_STR && !strcmp(c->str, name)) return 1;
  return 0;
}

// resolve `value` inside workdir; returns a path relative to workdir or NULL if it escapes
static char *resolve_inside(Tools *t, const char *value, char *err, size_t errlen) {
  char joined[PATH_MAX * 2], real[PATH_MAX];
  if (value[0] == '/') snprintf(joined, sizeof joined, "%s", value);
  else snprintf(joined, sizeof joined, "%s/%s", t->workdir, value);
  if (!realpath(joined, real)) {
    snprintf(err, errlen, "%s: %s", value, strerror(errno));
    return NULL;
  }
  size_t wl = strlen(t->workdir);
  if (strncmp(real, t->workdir, wl) != 0 || (real[wl] != '\0' && real[wl] != '/')) {
    snprintf(err, errlen, "%s: access outside the workspace is not allowed", value);
    return NULL;
  }
  return strdup(real[wl] ? real + wl + 1 : ".");
}

/* Run a tool call {"name": ..., "parameters": {...}}. Returns malloc'd text for the model.
 * *shown receives a short human readable description of the call. */
static char *tools_run(Tools *t, JNode *call, char **shown, int ask_confirm) {
  SBuf out = {0};
  sb_puts(&out, "");
  *shown = NULL;
  JNode *name = json_get(call, "name");
  JNode *params = json_get(call, "parameters");
  if (!params) params = json_get(call, "arguments");
  if (!name || name->type != J_STR) { sb_puts(&out, "Error: the call has no function name."); return out.buf; }
  JNode *tool = tools_find(t, name->str);
  if (!tool) {
    sb_puts(&out, "Error: unknown function \"");
    sb_puts(&out, name->str);
    sb_puts(&out, "\".");
    return out.buf;
  }
  JNode *cmd = json_get(tool, "command");
  JNode *props = json_get(json_get(tool, "parameters"), "properties");
  if (!cmd || cmd->type != J_ARR || !cmd->child) { sb_puts(&out, "Error: tool has no command."); return out.buf; }

  // build argv
  int argc = 0;
  for (JNode *c = cmd->child; c; c = c->next) argc++;
  char **argv = calloc(argc + 1, sizeof(char *));
  char err[PATH_MAX + 128] = {0};
  int i = 0;
  for (JNode *c = cmd->child; c; c = c->next, i++) {
    const char *s = c->type == J_STR ? c->str : "";
    size_t l = strlen(s);
    if (l > 2 && s[0] == '{' && s[l - 1] == '}') {
      char pname[256];
      snprintf(pname, sizeof pname, "%.*s", (int)(l - 2), s + 1);
      JNode *v = json_get(params, pname);
      char *val = NULL;
      if (v && v->type != J_NULL) {
        val = arg_to_string(v);
      } else {
        JNode *def = json_get(json_get(props, pname), "default");
        if (def) val = arg_to_string(def);
      }
      if (!val) { snprintf(err, sizeof err, "Error: missing required parameter \"%s\".", pname); break; }
      if (val[0] == '-' && !in_list(tool, "allow_dash", pname)) { snprintf(err, sizeof err, "Error: parameter \"%s\" must not start with '-'.", pname); free(val); break; }
      if (in_list(tool, "path_params", pname)) {
        char *r = resolve_inside(t, val, err, sizeof err);
        free(val);
        if (!r) break;
        if (r[0] == '-') { char *r2 = malloc(strlen(r) + 3); sprintf(r2, "./%s", r); free(r); r = r2; }
        val = r;
      }
      argv[i] = val;
    } else {
      argv[i] = strdup(s);
    }
  }
  // what we show to the user
  SBuf sh = {0};
  sb_puts(&sh, "");
  for (int k = 0; k < argc && argv[k]; k++) { if (k) sb_putc(&sh, ' '); sb_puts(&sh, argv[k]); }
  *shown = sh.buf;
  if (err[0]) {
    sb_puts(&out, err);
    goto done;
  }
  JNode *conf = json_get(tool, "confirm");
  if (ask_confirm && conf && conf->type == J_BOOL && conf->boolean) {
    fprintf(stderr, "Run `%s`? [y/N] ", *shown);
    char ans[16] = {0};
    if (!fgets(ans, sizeof ans, stdin) || (ans[0] != 'y' && ans[0] != 'Y')) {
      sb_puts(&out, "Error: the user declined to run this command.");
      goto done;
    }
  }

  // run: fork/execvp (no shell), stdout+stderr captured, stdin from /dev/null, timeout
  int pfd[2];
  if (pipe(pfd) != 0) { sb_puts(&out, "Error: pipe failed."); goto done; }
  fflush(stdout);
  fflush(stderr);
  pid_t pid = fork();
  if (pid == 0) {
    setpgid(0, 0);
    int devnull = open("/dev/null", O_RDONLY);
    if (devnull >= 0) dup2(devnull, 0);
    dup2(pfd[1], 1);
    dup2(pfd[1], 2);
    close(pfd[0]);
    close(pfd[1]);
    if (chdir(t->workdir) != 0) _exit(126);
    execvp(argv[0], argv);
    _exit(127);
  }
  close(pfd[1]);
  if (pid < 0) { close(pfd[0]); sb_puts(&out, "Error: fork failed."); goto done; }
  size_t total = 0;
  int timed_out = 0;
  time_t start = time(NULL);
  char chunk[4096];
  for (;;) {
    struct pollfd p = {.fd = pfd[0], .events = POLLIN};
    int left = t->timeout_sec - (int)(time(NULL) - start);
    if (left <= 0) { timed_out = 1; break; }
    int r = poll(&p, 1, left * 1000);
    if (r == 0) { timed_out = 1; break; }
    if (r < 0) { if (errno == EINTR) continue; break; }
    ssize_t n = read(pfd[0], chunk, sizeof chunk);
    if (n <= 0) break;
    if (total < (size_t)t->max_output) {
      size_t take = (size_t)n;
      if (total + take > (size_t)t->max_output) take = t->max_output - total;
      sb_putn(&out, chunk, take);
    }
    total += n;
  }
  if (timed_out) kill(-pid, SIGKILL);
  close(pfd[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  if (total == 0) sb_puts(&out, "(no output)");
  if (total > (size_t)t->max_output) {
    char note[128];
    snprintf(note, sizeof note, "\n...[output truncated: showing %d of %zu bytes]", t->max_output, total);
    sb_puts(&out, note);
  }
  if (timed_out) {
    char note[96];
    snprintf(note, sizeof note, "\n[killed after %d s timeout]", t->timeout_sec);
    sb_puts(&out, note);
  } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
    char note[64];
    snprintf(note, sizeof note, "\n[exit code %d]", WEXITSTATUS(status));
    sb_puts(&out, note);
  }
done:
  for (int k = 0; k < argc; k++) free(argv[k]);
  free(argv);
  return out.buf;
}

#endif
