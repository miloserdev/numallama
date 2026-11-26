// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) { return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str); }

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];

  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

// Llama 3 (tiktoken) encoder:
//  1) split the text with the Llama 3 pre-tokenizer regex
//     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//  2) byte-level BPE inside every piece: repeatedly merge the adjacent pair whose
//     concatenation has the lowest rank (for tiktoken the rank is the token id)
// Unicode classes are approximated by code point ranges (no ICU): this is exact for ASCII,
// Latin, Cyrillic, Greek, CJK etc., and only differs on exotic letters/digits/spaces.

static int utf8_decode(const unsigned char *s, int *cp) {
  if (s[0] < 0x80) { *cp = s[0]; return 1; }
  if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) { *cp = ((s[0] & 0x1F) << 6) | (s[1] & 0x3F); return 2; }
  if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
    *cp = ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3;
  }
  if ((s[0] & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
    *cp = ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4;
  }
  *cp = -1; // invalid byte: its own class (punctuation-like)
  return 1;
}

static int cp_is_space(int c) {
  return c == ' ' || (c >= 0x09 && c <= 0x0D) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static int cp_is_number(int c) {
  if (c >= '0' && c <= '9') return 1;
  if (c < 0x80) return 0;
  return c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) || (c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) ||
         (c >= 0x966 && c <= 0x96F) || (c >= 0x2070 && c <= 0x2079) || (c >= 0x2080 && c <= 0x2089) || (c >= 0x2150 && c <= 0x2189) ||
         (c >= 0x2460 && c <= 0x249B) || (c >= 0xFF10 && c <= 0xFF19);
}

static int cp_is_letter(int c) {
  if (c < 0) return 0;
  if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  if (cp_is_space(c) || cp_is_number(c)) return 0;
  if (c < 0xC0) return c == 0xAA || c == 0xB5 || c == 0xBA; // Latin-1 punctuation/symbols
  if (c == 0xD7 || c == 0xF7) return 0;                       // multiplication / division signs
  if (ARCH == ARCH_QWEN35 && c >= 0x300 && c <= 0x36F) return 1; // Qwen3.5 regex: [\p{L}\p{M}]+
  if (c >= 0x2B0 && c <= 0x36F) return c < 0x2C2 || (c >= 0x2C6 && c <= 0x2D1) || (c >= 0x2E0 && c <= 0x2E4); // modifiers, combining marks
  if (c >= 0x2000 && c <= 0x2BFF) return c == 0x2071 || c == 0x207F || (c >= 0x2090 && c <= 0x209C); // punctuation, symbols, arrows, math
  if (c >= 0x3000 && c <= 0x303F) return c == 0x3005 || c == 0x3006; // CJK punctuation
  if (c >= 0xFE30 && c <= 0xFE6F) return 0;                          // CJK compatibility forms
  if (c >= 0xFF00 && c <= 0xFF65) return (c >= 0xFF21 && c <= 0xFF3A) || (c >= 0xFF41 && c <= 0xFF5A); // fullwidth ASCII
  if (c >= 0x1F000 && c <= 0x1FAFF) return 0; // emoji and pictographs
  if (c >= 0xE000 && c <= 0xF8FF) return 0;   // private use
  if (c == 0x37E || c == 0x387 || c == 0x55A || (c >= 0x55A && c <= 0x55F) || c == 0x589 || c == 0x5BE || c == 0x5C0 || c == 0x5C3 ||
      c == 0x5F3 || c == 0x5F4 || (c >= 0x60C && c <= 0x60D) || c == 0x61B || c == 0x61F || c == 0x6D4 || c == 0x964 || c == 0x965 ||
      c == 0x482)
    return 0; // common script punctuation
  return 1;
}

// length in bytes of the next pre-token starting at s (s points inside a NUL-terminated string)
static int pretoken_len(const unsigned char *s) {
  int c, n = utf8_decode(s, &c);
  // contractions (case-insensitive): 's 't 're 've 'm 'll 'd
  if (c == '\'') {
    int a = tolower(s[1]), b = a ? tolower(s[2]) : 0;
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
  }
  // [^\r\n\p{L}\p{N}]?\p{L}+
  {
    int i = 0, c2;
    if (!cp_is_letter(c) && c != '\r' && c != '\n' && !cp_is_number(c)) {
      int n2 = utf8_decode(s + n, &c2);
      (void)n2;
      if (s[n] && cp_is_letter(c2)) i = n;
    }
    int d, m = utf8_decode(s + i, &d);
    if (s[i] && cp_is_letter(d)) {
      i += m;
      while (s[i] && (m = utf8_decode(s + i, &d), cp_is_letter(d))) i += m;
      return i;
    }
  }
  // \p{N}{1,3}
  if (cp_is_number(c)) {
    int i = n, d, m, k = 1;
    int maxd = ARCH_IS_QWEN ? 1 : 3; // \p{N} (Qwen2) vs \p{N}{1,3} (Llama 3, GLM)
    while (k < maxd && s[i] && (m = utf8_decode(s + i, &d), cp_is_number(d))) { i += m; k++; }
    return i;
  }
  // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
  {
    int i = (c == ' ') ? 1 : 0, d, m, start;
    start = i;
    while (s[i] && (m = utf8_decode(s + i, &d), !cp_is_space(d) && !cp_is_letter(d) && !cp_is_number(d))) i += m;
    if (i > start) {
      while (s[i] == '\r' || s[i] == '\n') i++;
      return i;
    }
  }
  // whitespace alternatives
  if (cp_is_space(c)) {
    int i = 0, d, m, last_nl = -1, prev = 0;
    while (s[i] && (m = utf8_decode(s + i, &d), cp_is_space(d))) {
      if (d == '\r' || d == '\n') last_nl = i;
      prev = i;
      i += m;
    }
    if (last_nl >= 0) return last_nl + 1; // \s*[\r\n]+
    if (!s[i]) return i;                  // \s+(?!\S) at the end of the text
    if (prev > 0) return prev;            // \s+(?!\S): leave one space for the next word
    return i;                             // \s+
  }
  return n; // anything else: a single code point
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }
  if (t->sorted_vocab == NULL) {
    // lazily sort the vocabulary for binary search
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }
  *n_tokens = 0;
  if (bos) tokens[(*n_tokens)++] = BOS_ID;

  size_t cap = 64;
  char *buf = malloc(cap);
  int *ids = malloc(cap * sizeof(int)), *beg = malloc(cap * sizeof(int)), *len = malloc(cap * sizeof(int));
  const unsigned char *s = (const unsigned char *)text;
  while (*s) {
    int plen = pretoken_len(s);
    if ((size_t)plen + 1 > cap) {
      while ((size_t)plen + 1 > cap) cap *= 2;
      buf = realloc(buf, cap);
      ids = realloc(ids, cap * sizeof(int));
      beg = realloc(beg, cap * sizeof(int));
      len = realloc(len, cap * sizeof(int));
    }
    // start from single bytes
    int n = 0;
    for (int i = 0; i < plen; i++) {
      char one[2] = {(char)s[i], 0};
      ids[n] = str_lookup(one, t->sorted_vocab, t->vocab_size);
      beg[n] = i;
      len[n] = 1;
      n++;
    }
    // merge the lowest-ranked pair until no pair is in the vocabulary
    while (n > 1) {
      int best = -1, best_id = INT32_MAX;
      for (int i = 0; i < n - 1; i++) {
        int l = len[i] + len[i + 1];
        memcpy(buf, s + beg[i], l);
        buf[l] = '\0';
        int id = str_lookup(buf, t->sorted_vocab, t->vocab_size);
        if (id >= 0 && id < best_id) {
          best_id = id;
          best = i;
        }
      }
      if (best < 0) break;
      ids[best] = best_id;
      len[best] += len[best + 1];
      for (int i = best + 1; i < n - 1; i++) {
        ids[i] = ids[i + 1];
        beg[i] = beg[i + 1];
        len[i] = len[i + 1];
      }
      n--;
    }
    for (int i = 0; i < n; i++) {
      if (ids[i] >= 0) tokens[(*n_tokens)++] = ids[i];
    }
    s += plen;
  }
  free(buf);
  free(ids);
  free(beg);
  free(len);
  if (eos) tokens[(*n_tokens)++] = EOS_ID;
}

