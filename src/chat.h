// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

static int is_end_token(int t);
static void init_special_tokens(Tokenizer *tk);
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, ARCH == ARCH_LLAMA3, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  init_special_tokens(tokenizer);
  // start the main loop
  long start = 0;               // used to time our code, only initialized after first iteration
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  while (pos < steps) {

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if (is_end_token(next) && pos > num_prompt_tokens)
      break;
    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms();
    }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop (Llama 3 instruct template). The whole dialog lives in the KV cache:
// every turn is appended at the current position, so the model sees the full history.
// When the context window fills up, the oldest turns are dropped (the system prompt is
// kept) and the remaining history is re-prefilled.

#define LLAMA3_BOS 128000
#define LLAMA3_EOS 128001
#define LLAMA3_START_HEADER 128006
#define LLAMA3_END_HEADER 128007
#define LLAMA3_EOT 128009
#define LLAMA3_EOM 128008
#define LLAMA3_PYTHON_TAG 128010

static int chat_read_line(const char *guide, char *buffer, size_t bufsize) {
  printf("%s", guide);
  fflush(stdout);
  if (fgets(buffer, bufsize, stdin) == NULL) {
    return 0; // EOF
  }
  size_t len = strlen(buffer);
  while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
    buffer[--len] = '\0';
  }
  return 1;
}

static int DRAFT_POS = 0; // tokens of the dialog history already in the draft model's KV cache (speculation)

typedef struct {
  int *tok;    // all tokens currently in the KV cache, tok[i] sits at position i
  int n;       // == current position
  int cap;     // context size
  int sys_len; // length of the BOS + system prompt prefix (never dropped)
  int *turn;   // start positions of the user turns
  int n_turns;
} ChatHistory;

// push tokens through the model, appending them to the history; returns the last logits
/* Hybrid models: the recurrent state cannot be rewound, so the chat saves it right after the system
 * prompt and restores it when old turns are dropped (the KV cache of the prefix stays valid). */
static size_t hyb_state_floats(const Config *p) { return (size_t)(p->n_layers - p->n_attn) * p->lin_nv * p->lin_dv * p->lin_dk; }
static size_t hyb_conv_floats(const Config *p) { return (size_t)(p->n_layers - p->n_attn) * p->conv_dim * (p->conv_k - 1); }
static void hybrid_snapshot(Transformer *t) {
  RunState *s = &t->state;
  if (!RECURRENT(&t->config)) return;
  if (!s->snap_S) {
    s->snap_S = malloc(hyb_state_floats(&t->config) * sizeof(float));
    s->snap_conv = malloc(hyb_conv_floats(&t->config) * sizeof(float));
  }
  memcpy(s->snap_S, s->lin_S, hyb_state_floats(&t->config) * sizeof(float));
  memcpy(s->snap_conv, s->conv_state, hyb_conv_floats(&t->config) * sizeof(float));
  s->snap_pos = s->next_pos;
}
static int hybrid_restore(Transformer *t, int pos) { // 1 if the state now stands at pos
  RunState *s = &t->state;
  if (!RECURRENT(&t->config)) return 1;
  if (s->snap_pos != pos) return 0;
  memcpy(s->lin_S, s->snap_S, hyb_state_floats(&t->config) * sizeof(float));
  memcpy(s->conv_state, s->snap_conv, hyb_conv_floats(&t->config) * sizeof(float));
  s->next_pos = pos;
  return 1;
}

/* Speculative verification on hybrid models: the batched pass advances the recurrent state past
 * rejected drafts, so the state is saved before the pass and restored (then the accepted prefix
 * is replayed) on a partial accept. A separate slot from the system-prompt anchor above. */
static void hybrid_spec_snapshot(Transformer *t) {
  RunState *s = &t->state;
  if (!RECURRENT(&t->config)) return;
  if (!s->spec_S) {
    s->spec_S = malloc(hyb_state_floats(&t->config) * sizeof(float));
    s->spec_conv = malloc(hyb_conv_floats(&t->config) * sizeof(float));
  }
  parallel_copy(s->spec_S, s->lin_S, hyb_state_floats(&t->config) * sizeof(float));
  parallel_copy(s->spec_conv, s->conv_state, hyb_conv_floats(&t->config) * sizeof(float));
  s->spec_pos = s->next_pos;
}
static void hybrid_spec_restore(Transformer *t) { // back to the state saved before the verify pass
  RunState *s = &t->state;
  if (!RECURRENT(&t->config) || !s->spec_S) return;
  parallel_copy(s->lin_S, s->spec_S, hyb_state_floats(&t->config) * sizeof(float));
  parallel_copy(s->conv_state, s->spec_conv, hyb_conv_floats(&t->config) * sizeof(float));
  s->next_pos = s->spec_pos;
}

static float *chat_feed(Transformer *t, ChatHistory *h, const int *toks, int n) {
  if (h->n + n > h->cap) { // never past the window (history buffer and KV cache are sized for it)
    fprintf(stderr, "[context window full: %d tokens not processed]\n", h->n + n - h->cap);
    n = h->cap - h->n;
  }
  if (n <= 0) return t->state.logits;
  int fed = 0;
  float *logits = forward_batch(t, toks, n, h->n, &fed);
  memcpy(h->tok + h->n, toks, (size_t)fed * sizeof(int));
  h->n += fed;
  return fed < n ? NULL : logits; // NULL: aborted mid-feed (g_srv_abort, the client went away)
}

// make room for `need` more tokens by dropping the oldest turns, then re-prefill
// keep_current: never drop the last (current) turn, e.g. while tool outputs are being added to it
static int chat_make_room(Transformer *t, ChatHistory *h, int need, int keep_current) { // 1 if `need` tokens fit now
  if (h->n + need <= h->cap) {
    return 1;
  }
  // keep the system prefix; drop whole turns from the front until we fit, keeping at least
  // half the window free for the new turn and the reply
  int target = h->cap / 2 < h->cap - need ? h->cap / 2 : h->cap - need;
  int drop_to = -1;
  const int last = keep_current && h->n_turns > 0 ? h->n_turns - 1 : h->n_turns;
  for (int i = 0; i < last; i++) {
    int keep = h->sys_len + (h->n - h->turn[i]);
    if (keep <= target) {
      drop_to = i;
      break;
    }
  }
  int start = drop_to >= 0 ? h->turn[drop_to] : (last < h->n_turns ? h->turn[last] : h->n); // first token kept after the prefix
  if (start <= h->sys_len) return h->n + need <= h->cap; // nothing to drop
  int kept = h->n - start;
  int dropped = start - h->sys_len;
  memmove(h->tok + h->sys_len, h->tok + start, kept * sizeof(int));
  int *replay = malloc((kept > 0 ? kept : 1) * sizeof(int));
  memcpy(replay, h->tok + h->sys_len, kept * sizeof(int));
  // turn starts shift down
  int nt = 0;
  for (int i = 0; i < h->n_turns; i++) {
    if (h->turn[i] >= start) {
      h->turn[nt++] = h->turn[i] - dropped;
    }
  }
  h->n_turns = nt;
  h->n = h->sys_len; // the KV cache for the system prefix is still valid
  if (DRAFT_POS > h->sys_len) DRAFT_POS = h->sys_len;
  long t0 = time_in_ms();
  if (!hybrid_restore(t, h->sys_len)) { // no saved state: re-process the system prompt too (from 0)
    int *all = malloc(((size_t)h->sys_len + kept + 1) * sizeof(int));
    memcpy(all, h->tok, h->sys_len * sizeof(int));
    memcpy(all + h->sys_len, replay, kept * sizeof(int));
    h->n = 0;
    chat_feed(t, h, all, h->sys_len + kept);
    free(all);
    kept = 0;
  }
  chat_feed(t, h, replay, kept);
  fprintf(stderr, "[context full: dropped %d old tokens, re-processed %d in %.1fs]\n", dropped, kept,
          (time_in_ms() - t0) / 1000.0);
  free(replay);
  return h->n + need <= h->cap;
}

static Tools *g_tools = NULL; // set by -f tools.json

// ---- chat formats -----------------------------------------------------------
// Llama 3:  <|start_header_id|>role<|end_header_id|>\n\n ... <|eot_id|>, tool calls as a bare JSON
//           message, tool output as an "ipython" message.
// ChatML (Qwen2.5): <|im_start|>role\n ... <|im_end|>\n, tool calls inside <tool_call></tool_call>,
//           tool output as <tool_response></tool_response> inside a user message.

// ChatML special token ids (Qwen2.5 / Qwen3 values; looked up in the vocabulary by init_special_tokens)
static int QWEN_ENDOFTEXT = 151643, QWEN_IM_END = 151645, QWEN_TOOL_CALL = 151657, QWEN_TOOL_CALL_END = 151658;
static int QWEN_THINK = -1, QWEN_THINK_END = -1;
static int GLM_USER = -1, GLM_OBSERVATION = -1; // GLM: a reply ends where the next turn would start
static int THINK = 0; // -r 1: Qwen3.5 / 3.8 reasoning (the reply starts inside <think>); 0: an empty think block
static int find_special(Tokenizer *tk, const char *s) {
  for (int i = SPECIAL_START; i < tk->vocab_size; i++)
    if (tk->vocab[i] && !strcmp(tk->vocab[i], s)) return i;
  return -1;
}
static void init_special_tokens(Tokenizer *tk) {
  if (ARCH < ARCH_QWEN2) return;
  int v;
  if ((v = find_special(tk, "<|endoftext|>")) >= 0) QWEN_ENDOFTEXT = v;
  if ((v = find_special(tk, "<|im_end|>")) >= 0) QWEN_IM_END = v;
  if ((v = find_special(tk, "<tool_call>")) >= 0) QWEN_TOOL_CALL = v;
  if ((v = find_special(tk, "</tool_call>")) >= 0) QWEN_TOOL_CALL_END = v;
  QWEN_THINK = find_special(tk, "<think>");
  QWEN_THINK_END = find_special(tk, "</think>");
  if (ARCH == ARCH_GLM5) {
    QWEN_IM_END = -1;
    GLM_USER = find_special(tk, "<|user|>");
    GLM_OBSERVATION = find_special(tk, "<|observation|>");
  }
}

static int is_end_token(int t) {
  if (ARCH == ARCH_GLM5) return t == QWEN_ENDOFTEXT || t == GLM_USER || t == GLM_OBSERVATION;
  if (ARCH >= ARCH_QWEN2) return t == QWEN_IM_END || t == QWEN_ENDOFTEXT;
  return t == LLAMA3_EOT || t == LLAMA3_EOS || t == LLAMA3_EOM;
}

// text that follows the end token of an assistant message in the template
// (GLM: none, the next turn's role token ends the message; the model's end token is not kept)
static const char *after_end(void) { return ARCH_IS_QWEN ? "\n" : ""; }
static const char *end_text(void) { return ARCH == ARCH_GLM5 ? "" : ARCH_IS_QWEN ? "<|im_end|>" : "<|eot_id|>"; }
// the generation prompt opens a think block (Qwen3.5 / 3.8, GLM): -r 1 leaves it open, otherwise it is empty
#define THINK_IN_PROMPT (ARCH == ARCH_QWEN35 || ARCH == ARCH_GLM5)
static const char *think_open(void) {
  if (ARCH == ARCH_GLM5) return THINK ? "<think>" : "<think></think>";
  return THINK ? "<think>\n" : "<think>\n\n</think>\n\n";
}

// append the tokens of plain text (special tokens are NOT recognized: user content can't inject them)
static int push_text(Tokenizer *tk, const char *s, int *msg, int m) {
  int n = 0;
  encode(tk, (char *)s, 0, 0, msg + m, &n);
  return m + n;
}

// append template text: special tokens such as <|im_start|> or <tool_call> are mapped to their ids
static int push_tpl(Tokenizer *tk, const char *s, int *msg, int m) {
  static int n_special = -1;
  static int *special_ids;
  static size_t *special_len;
  if (n_special < 0) { // special tokens look like <...> and live at ids >= SPECIAL_START
    n_special = 0;
    special_ids = malloc(tk->vocab_size * sizeof(int));
    special_len = malloc(tk->vocab_size * sizeof(size_t));
    for (int i = SPECIAL_START; i < tk->vocab_size; i++) {
      const char *v = tk->vocab[i];
      size_t l = strlen(v);
      if (l > 2 && ((v[0] == '<' && v[l - 1] == '>') || (v[0] == '[' && v[l - 1] == ']')) && strncmp(v, "<|unused_", 9) && strncmp(v, "<|dup_", 6)) {
        special_ids[n_special] = i;
        special_len[n_special++] = l;
      }
    }
  }
  SBuf seg = {0};
  sb_puts(&seg, "");
  while (*s) {
    int found = -1;
    if (*s == '<' || *s == '[') {
      for (int k = 0; k < n_special; k++) {
        if (!strncmp(s, tk->vocab[special_ids[k]], special_len[k])) { found = k; break; }
      }
    }
    if (found >= 0) {
      if (seg.len) { m = push_text(tk, seg.buf, msg, m); seg.len = 0; seg.buf[0] = '\0'; }
      msg[m++] = special_ids[found];
      s += special_len[found];
    } else {
      sb_putc(&seg, *s++);
    }
  }
  if (seg.len) m = push_text(tk, seg.buf, msg, m);
  free(seg.buf);
  return m;
}

// the system message (with the tool list when tools are enabled)
static int build_system(Tokenizer *tk, const char *sys_text, int *msg, int m) {
  SBuf b = {0};
  sb_puts(&b, "");
  if (ARCH == ARCH_GLM5) { // GLM-5.3: no newlines between turns, reasoning effort in the first system message
    sb_puts(&b, "[gMASK]<sop><|system|>Reasoning Effort: Max");
    if (g_tools) {
      sb_puts(&b, "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                  "You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
      char *tp = tools_prompt(g_tools, -1, "\n", "");
      sb_puts(&b, tp);
      free(tp);
      sb_puts(&b, "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
                  "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                  "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
    }
    if (sys_text[0]) { sb_puts(&b, "<|system|>"); sb_puts(&b, sys_text); }
  } else if (ARCH == ARCH_QWEN35) { // Qwen3.5 / 3.8 template (XML tool calls, optional reasoning instructions)
    const char *reason = THINK ? "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                                 "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                                 "clarity in the final answer." : "";
    if (g_tools) {
      sb_puts(&b, "<|im_start|>system\n");
      if (reason[0]) { sb_puts(&b, reason); sb_puts(&b, "\n\n"); }
      sb_puts(&b, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
      char *tp = tools_prompt(g_tools, -1, "\n", "");
      sb_puts(&b, tp);
      free(tp);
      sb_puts(&b, "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                  "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                  "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
                  "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
                  "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
                  "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
                  "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
                  "the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
      if (sys_text[0]) { sb_puts(&b, "\n\n"); sb_puts(&b, sys_text); }
      sb_puts(&b, "<|im_end|>\n");
    } else if (sys_text[0] || reason[0]) {
      sb_puts(&b, "<|im_start|>system\n");
      if (reason[0]) { sb_puts(&b, reason); if (sys_text[0]) sb_puts(&b, "\n\n"); }
      sb_puts(&b, sys_text);
      sb_puts(&b, "<|im_end|>\n");
    }
  } else if (ARCH >= ARCH_QWEN2) {
    if (ARCH == ARCH_QWEN3 && !sys_text[0] && !g_tools) { // Qwen3: no default system message
      free(b.buf);
      return m;
    }
    sb_puts(&b, "<|im_start|>system\n");
    if (ARCH == ARCH_QWEN3) {
      sb_puts(&b, sys_text);
      if (sys_text[0] && g_tools) sb_puts(&b, "\n\n");
    } else {
      sb_puts(&b, sys_text[0] ? sys_text : "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
      if (g_tools) sb_puts(&b, "\n\n");
    }
    if (g_tools) {
      sb_puts(&b, "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                  "You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
      char *tp = tools_prompt(g_tools, -1, "\n", "");
      sb_puts(&b, tp);
      free(tp);
      sb_puts(&b, "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
                  "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
                  "<args-json-object>}\n</tool_call>");
    }
    sb_puts(&b, "<|im_end|>\n");
  } else {
    sb_puts(&b, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n");
    if (g_tools) sb_puts(&b, "Environment: ipython\n");
    char date[64];
    time_t now = time(NULL);
    strftime(date, sizeof date, "%d %b %Y", localtime(&now));
    sb_puts(&b, "Cutting Knowledge Date: December 2023\nToday Date: ");
    sb_puts(&b, date);
    sb_puts(&b, "\n\n");
    if (g_tools) {
      sb_puts(&b, "You have access to the following functions. To call a function, please respond with JSON for a function call.");
      sb_puts(&b, "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.");
      sb_puts(&b, "Do not use variables.\n\n");
      char *tp = tools_prompt(g_tools, 4, "", "\n\n");
      sb_puts(&b, tp);
      free(tp);
    }
    sb_puts(&b, sys_text);
    sb_puts(&b, "<|eot_id|>");
  }
  m = push_tpl(tk, b.buf, msg, m);
  free(b.buf);
  return m;
}

static int build_user(Tokenizer *tk, const char *prefix, const char *text, int *msg, int m) {
  m = push_tpl(tk, prefix, msg, m);
  if (ARCH == ARCH_GLM5) {
    m = push_tpl(tk, "<|user|>", msg, m);
    m = push_text(tk, text, msg, m);
    m = push_tpl(tk, "<|assistant|>", msg, m);
    m = push_tpl(tk, think_open(), msg, m);
  } else if (ARCH >= ARCH_QWEN2) {
    m = push_tpl(tk, "<|im_start|>user\n", msg, m);
    m = push_text(tk, text, msg, m);
    m = push_tpl(tk, "<|im_end|>\n<|im_start|>assistant\n", msg, m);
    if (ARCH == ARCH_QWEN35) m = push_tpl(tk, think_open(), msg, m);
  } else {
    m = push_tpl(tk, "<|start_header_id|>user<|end_header_id|>\n\n", msg, m);
    m = push_text(tk, text, msg, m);
    m = push_tpl(tk, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", msg, m);
  }
  return m;
}

static int build_tool_results(Tokenizer *tk, const char *prefix, char **outs, int n, int *msg, int m) {
  m = push_tpl(tk, prefix, msg, m);
  if (ARCH == ARCH_GLM5) {
    m = push_tpl(tk, "<|observation|>", msg, m);
    for (int i = 0; i < n; i++) {
      m = push_tpl(tk, "<tool_response>", msg, m);
      m = push_text(tk, outs[i], msg, m);
      m = push_tpl(tk, "</tool_response>", msg, m);
    }
    m = push_tpl(tk, "<|assistant|>", msg, m);
    m = push_tpl(tk, think_open(), msg, m);
  } else if (ARCH >= ARCH_QWEN2) {
    m = push_tpl(tk, "<|im_start|>user", msg, m);
    for (int i = 0; i < n; i++) {
      m = push_tpl(tk, "\n<tool_response>\n", msg, m);
      m = push_text(tk, outs[i], msg, m);
      m = push_tpl(tk, "\n</tool_response>", msg, m);
    }
    m = push_tpl(tk, "<|im_end|>\n<|im_start|>assistant\n", msg, m);
    if (ARCH == ARCH_QWEN35) m = push_tpl(tk, think_open(), msg, m);
  } else {
    for (int i = 0; i < n; i++) {
      m = push_tpl(tk, "<|start_header_id|>ipython<|end_header_id|>\n\n", msg, m);
      m = push_text(tk, outs[i], msg, m);
      m = push_tpl(tk, "<|eot_id|>", msg, m);
    }
    m = push_tpl(tk, "<|start_header_id|>assistant<|end_header_id|>\n\n", msg, m);
  }
  return m;
}

/* Small models do not always follow the call format exactly; accept
 *   {"name": ..., "parameters"|"arguments": {...} or "<json string>"}
 *   {"type": "function", "function": {"name": ..., ...}}  /  {"function": "name", "parameters": ...}
 * and rewrite them to {"name": ..., "parameters": {...}}. Returns NULL if it is not a call. */
static JNode *normalize_call(JNode *j) {
  if (!j || j->type != J_OBJ) { json_free(j); return NULL; }
  JNode *fn = json_get(j, "function");
  JNode *src = (fn && fn->type == J_OBJ) ? fn : j;
  JNode *name = json_get(src, "name");
  if (!name && fn && fn->type == J_STR) name = fn;
  JNode *params = json_get(src, "parameters");
  if (!params) params = json_get(src, "arguments");
  if (!params && src != j) { params = json_get(j, "parameters"); if (!params) params = json_get(j, "arguments"); }
  if (!name || name->type != J_STR) { json_free(j); return NULL; }
  JNode *out = calloc(1, sizeof(JNode));
  out->type = J_OBJ;
  JNode *n = calloc(1, sizeof(JNode));
  n->type = J_STR; n->key = strdup("name"); n->str = strdup(name->str);
  out->child = n;
  JNode *p = NULL;
  if (params && params->type == J_STR) { // arguments given as a JSON string
    p = json_parse(params->str, NULL);
    if (p && p->type != J_OBJ) { json_free(p); p = NULL; }
  } else if (params && params->type == J_OBJ) {
    SBuf b = {0};
    json_write(&b, params, -1, 0);
    p = json_parse(b.buf, NULL); // deep copy
    free(b.buf);
  }
  if (!p) { p = calloc(1, sizeof(JNode)); p->type = J_OBJ; }
  p->key = strdup("parameters");
  n->next = p;
  json_free(j);
  return out;
}

/* Qwen3.5 / 3.8 call body: <function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>
 * -> {"name": NAME, "parameters": {P: VALUE, ...}} (a VALUE that parses as a JSON number / object /
 * array / literal keeps its type, anything else is a string). NULL if it is not a call. */
/* GLM: name<arg_key>k</arg_key><arg_value>v</arg_value>... (v is raw text, or JSON for non-strings) */
static JNode *parse_glm_call(const char *s) {
  while (*s && isspace((unsigned char)*s)) s++;
  const char *ne = strstr(s, "<arg_key>");
  if (!ne) ne = s + strlen(s);
  const char *nend = ne;
  while (nend > s && isspace((unsigned char)nend[-1])) nend--;
  if (nend == s) return NULL;
  JNode *out = calloc(1, sizeof(JNode));
  out->type = J_OBJ;
  JNode *n = calloc(1, sizeof(JNode));
  n->type = J_STR; n->key = strdup("name"); n->str = strndup(s, nend - s);
  out->child = n;
  JNode *params = calloc(1, sizeof(JNode));
  params->type = J_OBJ; params->key = strdup("parameters");
  n->next = params;
  JNode **tail = &params->child;
  const char *p = ne;
  while ((p = strstr(p, "<arg_key>")) != NULL) {
    p += 9;
    const char *ke = strstr(p, "</arg_key>");
    if (!ke) break;
    const char *v = strstr(ke, "<arg_value>");
    if (!v) break;
    v += 11;
    const char *ve = strstr(v, "</arg_value>");
    if (!ve) break;
    char *val = strndup(v, ve - v);
    JNode *j = NULL;
    if (val[0] && strchr("-0123456789{[tfn", val[0])) {
      const char *endp = NULL;
      j = json_parse(val, &endp);
      if (j && (j->type == J_STR || (endp && *endp))) { json_free(j); j = NULL; }
    }
    if (!j) { j = calloc(1, sizeof(JNode)); j->type = J_STR; j->str = strdup(val); }
    free(val);
    j->key = strndup(p, ke - p);
    *tail = j;
    tail = &j->next;
    p = ve + 12;
  }
  return out;
}
static JNode *parse_xml_call(const char *s) {
  const char *f = strstr(s, "<function=");
  if (!f) return NULL;
  f += 10;
  const char *fe = strchr(f, '>');
  if (!fe) return NULL;
  JNode *out = calloc(1, sizeof(JNode));
  out->type = J_OBJ;
  JNode *n = calloc(1, sizeof(JNode));
  n->type = J_STR; n->key = strdup("name"); n->str = strndup(f, fe - f);
  out->child = n;
  JNode *params = calloc(1, sizeof(JNode));
  params->type = J_OBJ; params->key = strdup("parameters");
  n->next = params;
  JNode **tail = &params->child;
  const char *p = fe + 1;
  while ((p = strstr(p, "<parameter=")) != NULL) {
    p += 11;
    const char *pe = strchr(p, '>');
    if (!pe) break;
    const char *v = pe + 1, *ve = strstr(v, "</parameter>");
    if (!ve) break;
    if (*v == '\n') v++;
    const char *vend = ve;
    if (vend > v && vend[-1] == '\n') vend--;
    char *val = strndup(v, vend - v);
    JNode *j = NULL;
    if (val[0] && strchr("-0123456789{[tfn", val[0])) {
      const char *endp = NULL;
      j = json_parse(val, &endp);
      if (j && (j->type == J_STR || (endp && *endp))) { json_free(j); j = NULL; }
    }
    if (!j) { j = calloc(1, sizeof(JNode)); j->type = J_STR; j->str = strdup(val); }
    free(val);
    j->key = strndup(p, pe - p);
    *tail = j;
    tail = &j->next;
    p = ve + 12;
  }
  return out;
}

#define MAX_CALLS_PER_REPLY 8

/* Generate one assistant message. Plain text is streamed to stdout. Tool calls are collected:
 *  - Llama 3: a message that starts with '{' (or <|python_tag|>) is one JSON call
 *  - ChatML:  every <tool_call> ... </tool_call> block is one call
 * Returns the number of calls stored in calls[]. *ended tells whether the end token was fed. */
static int SPEC_K = 8;              // -k: max draft length for prompt-lookup speculation, 0 = off
static Transformer *g_draft = NULL; // -d: small draft model with the same tokenizer
static int DRAFT_K = 6;             // max tokens drafted by the draft model per step

/* Draft up to kmax tokens with the draft model, continuing hist[0..n) + next. The draft model
 * first catches up with the history (batched), then decodes greedily and stops early when it is
 * unsure. Returns the number of drafted tokens; its KV cache then holds next, out[0..k-2]. */
static int draft_model_tokens(const int *hist, int n, int next, int *out, int kmax) {
  Transformer *d = g_draft;
  if (DRAFT_POS > n) DRAFT_POS = n;
  if (DRAFT_POS < n) {
    int fed = 0;
    forward_batch(d, hist + DRAFT_POS, n - DRAFT_POS, DRAFT_POS, &fed);
    DRAFT_POS = n;
  }
  const int V = d->config.vocab_size;
  int tok = next, pos = n, k = 0;
  while (k < kmax && pos < d->config.seq_len) {
    float *l = forward(d, tok, pos++);
    int a = 0;
    for (int i = 1; i < V; i++) if (l[i] > l[a]) a = i;
    float z = 0.0f;
    for (int i = 0; i < V; i++) z += expf(l[i] - l[a]);
    out[k++] = a;
    tok = a;
    if (1.0f / z < 0.4f) break; // the draft model is unsure: further tokens would mostly be rejected
  }
  return k;
}
static long SPEC_DRAFTED = 0, SPEC_ACCEPTED = 0, SPEC_STEPS = 0;

typedef struct {   // streaming state of one assistant message
  SBuf buf;        // Llama: the undecided/buffered message; ChatML: the current <tool_call> body
  int mode;        // Llama: 0 undecided, 1 streaming text, 2 buffering a call; ChatML: 0/1 text, 2 inside <tool_call>
  int n_calls;
  int token;       // previous token (for decode)
  int think;       // inside <think> ... </think>
  JNode **calls;
} ReplyState;

/* Where the reply text goes: the CLI prints it, the HTTP server (src/server.h) collects and streams
 * it. The server also limits the reply length and can stop it (stop string, client gone). */
typedef struct {
  void (*text)(void *ctx, const char *s, int reasoning);
  void *ctx;
  int max_gen; // > 0: stop after this many generated tokens
  int stop;    // set by text(): end the reply now
} ReplySink;
static ReplySink *g_sink = NULL; // NULL: print to stdout
static volatile int g_srv_abort = 0; // server: the client went away (heartbeat): stop the prefill
static void out_text(const char *s, int reasoning) {
  if (g_sink) { g_sink->text(g_sink->ctx, s, reasoning); return; }
  printf("%s", s);
  fflush(stdout);
}
static void out_ctl(const char *s) { // terminal-only decoration
  if (!g_sink) { printf("%s", s); fflush(stdout); }
}
static int sink_done(int gen) { return g_sink && (g_sink->stop || (g_sink->max_gen > 0 && gen >= g_sink->max_gen)); }

// handle one generated (non-end) token: print it or collect it into a tool call
static void reply_token(ReplyState *rs, Tokenizer *tk, int next) {
  SBuf *buf = &rs->buf;
  if (ARCH >= ARCH_QWEN2 && next == QWEN_TOOL_CALL) {
    rs->mode = 2;
    buf->len = 0;
    buf->buf[0] = '\0';
  } else if (ARCH >= ARCH_QWEN2 && next == QWEN_TOOL_CALL_END) {
    if (rs->mode == 2 && rs->n_calls < MAX_CALLS_PER_REPLY) {
      JNode *c = ARCH == ARCH_GLM5 ? parse_glm_call(buf->buf)
                 : ARCH == ARCH_QWEN35 && strstr(buf->buf, "<function=") ? parse_xml_call(buf->buf) : normalize_call(json_parse(buf->buf, NULL));
      if (c) rs->calls[rs->n_calls++] = c;
      else { out_text("<tool_call>", 0); out_text(buf->buf, 0); out_text("</tool_call>", 0); }
    }
    rs->mode = 0; // skip the whitespace between calls
  } else if (ARCH < ARCH_QWEN2 && next == LLAMA3_PYTHON_TAG) {
    rs->mode = 2;
  } else if (next >= 0 && next == QWEN_THINK) { // reasoning is shown dimmed
    rs->think = 1;
    if (THINK) out_ctl("\033[2m");
  } else if (next >= 0 && next == QWEN_THINK_END) {
    rs->think = 0;
    if (THINK) out_ctl("\033[0m\n");
    rs->mode = 0; // skip the blank lines before the answer
  } else {
    char *piece = decode(tk, rs->token, next);
    // drop lone control bytes; bytes >= 0x80 are parts of multi-byte UTF-8 characters and must stay
    int ok = piece && piece[0] && !(piece[1] == '\0' && (unsigned char)piece[0] < 0x20 && !isspace((unsigned char)piece[0]));
    if (ok) {
      if (rs->mode == 2) {
        sb_puts(buf, piece);
      } else if (ARCH >= ARCH_QWEN2 || rs->mode == 1) {
        if (rs->mode == 0) { // skip leading whitespace of the message
          const char *p = piece;
          while (*p && isspace((unsigned char)*p)) p++;
          if (*p) { rs->mode = 1; out_text(p, rs->think); }
        } else {
          out_text(piece, rs->think);
        }
      } else { // Llama, undecided
        sb_puts(buf, piece);
        const char *p = buf->buf;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '{' && g_tools) {
          rs->mode = 2;
        } else if (*p) {
          rs->mode = 1;
          out_text(p, 0);
        }
      }
    }
  }
  rs->token = next;
}

/* Generate one assistant message. Plain text is streamed to stdout; tool calls are collected
 * (Llama 3: a message starting with '{' or <|python_tag|>; ChatML: each <tool_call>...</tool_call>).
 * Decoding is speculative: a draft from prompt lookup is verified in one batched pass, the result
 * is distributed exactly as without speculation. Returns the number of calls in calls[]. */
static int chat_reply(Transformer *tr, Tokenizer *tk, Sampler *sp, ChatHistory *h, float *logits, int *ended, int *gen,
                      JNode **calls) {
  ReplyState rs = {.mode = 0, .n_calls = 0, .token = -1, .calls = calls};
  if (THINK_IN_PROMPT && THINK) { out_ctl("\033[2m"); rs.think = 1; } // the prompt already opened <think>
  sb_puts(&rs.buf, "");
  const int V = tr->config.vocab_size;
  float *own = malloc((size_t)V * sizeof(float)); // logits we own (the shared buffers get overwritten)
  memcpy(own, logits, (size_t)V * sizeof(float));
  int pending = -1; // a token already sampled (after a rejected draft) but not yet fed
  const int moe = tr->config.n_experts > 0;
  int spec_len = moe && SPEC_K > 4 ? 4 : SPEC_K; // current prompt-lookup draft length (adaptive for MoE)
  int spec_cool = 0, spec_wait = 0; // MoE: after rejected drafts, skip lookups for a growing number of tokens
  *ended = 0;
  for (;;) {
    int next = pending >= 0 ? pending : sample(sp, own);
    pending = -1;
    if (is_end_token(next)) {
      if (ARCH == ARCH_GLM5) *ended = 1; // the next turn starts with its own role token
      else if (h->n < h->cap) {
        chat_feed(tr, h, &next, 1); // keep the end token in the history
        *ended = 1;
      }
      break;
    }
    reply_token(&rs, tk, next);
    (*gen)++;
    if (sink_done(*gen)) break; // server: length limit, stop string or client gone
    if (h->n >= h->cap) {
      fprintf(stderr, "\n[context window full, reply truncated]");
      break;
    }
    // draft the continuation by prompt lookup over the whole dialog (+ the new token)
    int seq[SPEC_MAX + 1], k = 0;
    seq[0] = next;
    int from_model = 0;
    int room = h->cap - h->n - 1;
    if (SPEC_K > 0 && spec_wait > 0) spec_wait--;
    else if (SPEC_K > 0) {
      h->tok[h->n] = next; // temporarily, for the lookup
      int kmax = spec_len < room ? spec_len : room;
      if (kmax > SPEC_MAX - 1) kmax = SPEC_MAX - 1;
      k = kmax > 0 ? ngram_draft(h->tok, h->n + 1, seq + 1, kmax) : 0;
    }
    if (g_draft && k < 3 && room > 1) { // no good n-gram match: ask the draft model
      int kmax = DRAFT_K < room ? DRAFT_K : room;
      if (kmax > SPEC_MAX - 1) kmax = SPEC_MAX - 1;
      k = draft_model_tokens(h->tok, h->n, next, seq + 1, kmax);
      from_model = 1;
    }
    if (k == 0) {
      float *l = chat_feed(tr, h, &next, 1);
      memcpy(own, l, (size_t)V * sizeof(float));
      continue;
    }
    // verify [next, d1..dk] in one pass; row i = distribution after seq[i]
    int base = h->n;
    const int rec = RECURRENT(&tr->config);
    if (rec) hybrid_spec_snapshot(tr); // the pass advances the recurrent state past rejected drafts
    float *L = forward_all(tr, seq, k + 1, base);
    memcpy(h->tok + base, seq, (k + 1) * sizeof(int));
    int acc = 0, stop = 0;
    SPEC_DRAFTED += k;
    SPEC_STEPS++;
    for (int i = 0; i < k; i++) {
      int y = sample_draft(sp, L + (size_t)i * V, seq[i + 1]);
      if (y != seq[i + 1]) { pending = y; break; }
      acc++;
      if (is_end_token(y)) { *ended = 1; stop = 1; break; } // the end token is already in the cache
      reply_token(&rs, tk, y);
      (*gen)++;
      if (sink_done(*gen)) { stop = 1; break; }
    }
    SPEC_ACCEPTED += acc;
    if (!from_model && rec && !moe) { // dense hybrid: a verify pass costs the same at any k (the weights are read once per pass)
      // and a rejected draft only costs a snapshot + a state-only commit: keep the draft length, cool down after misses
      if (acc == 0) { spec_cool = spec_cool ? (spec_cool * 2 < 32 ? spec_cool * 2 : 32) : 2; spec_wait = spec_cool; }
      else spec_cool = 0;
    } else if (!from_model && moe) { // MoE: a verification step costs ~1.4x (2 tokens) .. 2.5x (8): adapt the draft length
      if (acc == k) spec_len = spec_len * 2 < SPEC_K ? spec_len * 2 : SPEC_K;
      else spec_len = acc + 1 > 2 ? acc + 1 : 2;
      if (acc == 0) { spec_cool = spec_cool ? (spec_cool * 2 < 32 ? spec_cool * 2 : 32) : 2; spec_wait = spec_cool; }
      else spec_cool = 0;
    }
    h->n = base + 1 + acc; // positions after this hold rejected drafts: they get overwritten
    if (rec && acc < k) { // rewind the recurrent state to the snapshot and replay the accepted prefix
      hybrid_spec_restore(tr);
      if (tr->config.hybrid) hybrid_spec_commit(tr, acc + 1); // state-only replay from the saved fused inputs
      else forward_batch(tr, seq, acc + 1, base, NULL);
    }
    if (from_model) DRAFT_POS = base + 1 + (acc < k - 1 ? acc : k - 1); // draft cache: next, accepted drafts
    if (stop) break;
    if (pending < 0) memcpy(own, L + (size_t)k * V, (size_t)V * sizeof(float)); // all accepted
  }
  if (ARCH < ARCH_QWEN2 && rs.mode == 2) {
    JNode *c = normalize_call(json_parse(rs.buf.buf, NULL));
    if (c) calls[rs.n_calls++] = c;
    else out_text(rs.buf.buf, 0); // looked like JSON but is not a call: show it as text
  }
  free(rs.buf.buf);
  free(own);
  return rs.n_calls;
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *cli_system_prompt, int steps) {
  const int ctx = transformer->config.seq_len;
  const int BUF = 1 << 16;
  const int MAX_ROUNDS = 16; // tool-call rounds per user message
  char *text = malloc(BUF);
  int *msg = malloc((4 * BUF + 64) * sizeof(int));
  ChatHistory h = {.tok = malloc(ctx * sizeof(int)), .n = 0, .cap = ctx, .sys_len = 0, .turn = malloc(ctx * sizeof(int)), .n_turns = 0};
  init_special_tokens(tokenizer);
  (void)steps;
  SBuf prefix = {0}; // template text owed before the next message (end of the previous reply)
  sb_puts(&prefix, "");

  if (cli_system_prompt != NULL) {
    snprintf(text, BUF, "%s", cli_system_prompt);
  } else if (!chat_read_line("Enter system prompt (optional): ", text, BUF)) {
    text[0] = '\0';
  }
  char *st = text; // the templates trim the system message
  while (*st && isspace((unsigned char)*st)) st++;
  size_t sl = strlen(st);
  while (sl > 0 && isspace((unsigned char)st[sl - 1])) st[--sl] = '\0';
  int m = build_system(tokenizer, st, msg, 0);
  if (m >= ctx / 2) {
    fprintf(stderr, "system prompt is too long (%d tokens) for the context window (%d)\n", m, ctx);
    exit(EXIT_FAILURE);
  }
  long ts = time_in_ms();
  chat_feed(transformer, &h, msg, m);
  h.sys_len = h.n;
  hybrid_snapshot(transformer); // hybrid models: the state after the system prompt, for context trimming
  fprintf(stderr, "[system prompt: %d tokens, %.1fs]\n", m, (time_in_ms() - ts) / 1000.0);

  for (int first = 1;; first = 0) {
    if (first && cli_user_prompt != NULL) {
      snprintf(text, BUF, "%s", cli_user_prompt);
      printf("User: %s\n", text);
    } else if (!chat_read_line("User (or exit): ", text, BUF) || strcmp(text, "exit") == 0) {
      break;
    }
    // the owed prefix belongs to the previous turn: feed it first
    m = push_tpl(tokenizer, prefix.buf, msg, 0);
    prefix.len = 0;
    prefix.buf[0] = '\0';
    int turn_off = m;
    m = build_user(tokenizer, "", text, msg, m);
    if (m - turn_off + h.sys_len >= ctx) {
      fprintf(stderr, "[message is too long for the context window (%d tokens), skipped]\n", ctx);
      continue;
    }
    int reserve = ctx / 8 < 512 ? ctx / 8 : 512;
    if (!chat_make_room(transformer, &h, m + reserve, 0))
      fprintf(stderr, "[message is too long for the context window (%d tokens)]\n", ctx);
    if (turn_off > 0) chat_feed(transformer, &h, msg, turn_off);
    h.turn[h.n_turns++] = h.n;
    long t0 = time_in_ms();
    float *logits = chat_feed(transformer, &h, msg + turn_off, m - turn_off);
    int prompt_tok = m - turn_off;
    long prompt_ms = time_in_ms() - t0;

    printf("Assistant: ");
    fflush(stdout);
    int gen = 0;
    long gen_ms = 0;
    for (int round = 0;; round++) {
      JNode *calls[MAX_CALLS_PER_REPLY];
      int ended = 0;
      long g0 = time_in_ms();
      int n_calls = chat_reply(transformer, tokenizer, sampler, &h, logits, &ended, &gen, calls);
      gen_ms += time_in_ms() - g0;
      // what the template owes after this reply
      if (!ended) sb_puts(&prefix, end_text());
      sb_puts(&prefix, after_end());
      if (n_calls == 0) break;
      if (round > MAX_ROUNDS) { // it ignored the limit: end the turn here
        for (int i = 0; i < n_calls; i++) json_free(calls[i]);
        fprintf(stderr, "[tool-call limit reached]\n");
        break;
      }
      // run the tools and hand their output back
      char *outs[MAX_CALLS_PER_REPLY];
      for (int i = 0; i < n_calls; i++) {
        char *shown = NULL;
        outs[i] = round < MAX_ROUNDS ? tools_run(g_tools, calls[i], &shown, 1) : strdup("Error: the limit of function calls for this message is reached. Do not call functions again: answer the user now with what you already know.");
        SBuf args = {0};
        JNode *params = json_get(calls[i], "parameters");
        if (params) json_write(&args, params, -1, 0);
        printf("\n  [tool %s %s] $ %s  (%zu bytes)", json_get(calls[i], "name")->str, args.buf ? args.buf : "{}", shown ? shown : "",
               strlen(outs[i]));
        free(args.buf);
        free(shown);
        json_free(calls[i]);
      }
      printf("\n");
      // the outputs must fit the window: drop old turns, then halve the longest outputs
      int fits = 0, cut = 0;
      for (int attempt = 0; attempt < 24; attempt++) {
        m = build_tool_results(tokenizer, prefix.buf, outs, n_calls, msg, 0);
        if (chat_make_room(transformer, &h, m + reserve, 1)) { fits = 1; break; }
        int lg = 0;
        for (int i = 1; i < n_calls; i++) if (strlen(outs[i]) > strlen(outs[lg])) lg = i;
        size_t len = strlen(outs[lg]);
        if (len < 64) break;
        const char *note = "\n[... output truncated to fit the context window]";
        char *o = malloc(len / 2 + strlen(note) + 1);
        memcpy(o, outs[lg], len / 2);
        strcpy(o + len / 2, note);
        free(outs[lg]);
        outs[lg] = o;
        cut = 1;
      }
      for (int i = 0; i < n_calls; i++) free(outs[i]);
      if (cut && fits) fprintf(stderr, "[tool output shortened to fit the context window]\n");
      if (!fits) { // keep the owed prefix for the next turn and stop calling tools
        fprintf(stderr, "[tool output does not fit the context window (%d tokens)]\n", ctx);
        break;
      }
      prefix.len = 0;
      prefix.buf[0] = '\0';
      long p0 = time_in_ms();
      logits = chat_feed(transformer, &h, msg, m);
      prompt_ms += time_in_ms() - p0;
      prompt_tok += m;
      printf("Assistant: ");
      fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
    fprintf(stderr, "[prompt %d tok %.1f tok/s | reply %d tok %.1f tok/s | context %d/%d | draft accepted %ld/%ld in %ld steps]\n", prompt_tok,
            prompt_tok * 1000.0 / (prompt_ms > 0 ? prompt_ms : 1), gen, gen * 1000.0 / (gen_ms > 0 ? gen_ms : 1), h.n, ctx,
            SPEC_ACCEPTED, SPEC_DRAFTED, SPEC_STEPS);
    SPEC_ACCEPTED = SPEC_DRAFTED = SPEC_STEPS = 0;
  }
  free(prefix.buf);
  free(text);
  free(msg);
  free(h.tok);
  free(h.turn);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
/* -m imatrix: run a text through the model (chunks of up to 512 tokens, each from position 0) and
 * write the importance statistics: "NMX1", int L, dim, q_dim, hidden, E, moe_hidden, long long ntok,
 * then float32 means: att (L,dim), wo (L,q_dim), ffn (L,[E,]dim), w2 (L,[E,]hidden or moe_hidden),
 * and for MoE the int64 token counts (L,E). Collect on a model exported without --smooth. */
__attribute__((unused)) static void imatrix(Transformer* t, Tokenizer* tk, const char* text_path, const char* out_path) {
    Config* p = &t->config;
    FILE* f = fopen(text_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", text_path); exit(EXIT_FAILURE); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* txt = malloc(n + 1);
    if (fread(txt, 1, n, f) != (size_t)n) { fprintf(stderr, "read failed\n"); exit(EXIT_FAILURE); }
    txt[n] = 0; fclose(f);
    int* ids = malloc((n + 8) * sizeof(int)), nt = 0;
    encode(tk, txt, 0, 0, ids, &nt);
    const int L = p->n_layers, E = p->n_experts, H = E > 0 ? p->moe_hidden : p->hidden_dim, EE = E > 0 ? E : 1;
    IMX.att = calloc((size_t)L * p->dim, sizeof(double));
    IMX_WO_W = p->q_dim;
    if (p->hybrid && p->lin_nv * p->lin_dv > IMX_WO_W) IMX_WO_W = p->lin_nv * p->lin_dv;
    IMX.wo = calloc((size_t)L * IMX_WO_W, sizeof(double));
    IMX.ffn = calloc((size_t)L * EE * p->dim, sizeof(double));
    IMX.w2 = calloc((size_t)L * EE * H, sizeof(double));
    IMX.nexp = calloc((size_t)L * EE, sizeof(long long));
    IMX.on = 1;
    const int CH = 512;
    long t0 = time_in_ms();
    for (int c = 0; c + 16 <= nt; c += CH - 1) {
        int len = nt - c < CH - 1 ? nt - c : CH - 1;
        int chunk[CH];
        chunk[0] = BOS_ID;
        memcpy(chunk + 1, ids + c, len * sizeof(int));
        forward_batch(t, chunk, len + 1, 0, &(int){0});
        IMX.ntok += len + 1;
        fprintf(stderr, "\r[imatrix] %lld / %d tokens", IMX.ntok, nt);
    }
    IMX.on = 0;
    fprintf(stderr, " (%.1f s)\n", (time_in_ms() - t0) / 1000.0);
    FILE* o = fopen(out_path, "wb");
    if (!o) { fprintf(stderr, "cannot write %s\n", out_path); exit(EXIT_FAILURE); }
    int hdr[7] = {0x31584D4E, L, p->dim, IMX_WO_W, H, E, E > 0 ? p->moe_hidden : 0};
    fwrite(hdr, sizeof(int), 7, o);
    fwrite(&IMX.ntok, sizeof(long long), 1, o);
    struct { double* a; size_t rows, n; } parts[4] = {{IMX.att, L, p->dim}, {IMX.wo, L, IMX_WO_W}, {IMX.ffn, (size_t)L * EE, p->dim}, {IMX.w2, (size_t)L * EE, H}};
    for (int k = 0; k < 4; k++)
        for (size_t r = 0; r < parts[k].rows; r++) {
            double cnt = (E > 0 && k >= 2) ? (double)IMX.nexp[r] : (double)IMX.ntok;
            for (size_t i = 0; i < parts[k].n; i++) {
                float v = cnt > 0 ? (float)(parts[k].a[r * parts[k].n + i] / cnt) : 0.0f;
                fwrite(&v, sizeof(float), 1, o);
            }
        }
    if (E > 0) fwrite(IMX.nexp, sizeof(long long), (size_t)L * E, o);
    fclose(o);
    fprintf(stderr, "[imatrix] %lld tokens -> %s\n", IMX.ntok, out_path);
}
