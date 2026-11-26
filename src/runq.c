/* numallama: single translation unit. The code lives in topical headers, included here in
 * dependency order (each header uses only what the previous ones define):
 *   core.h          config, model structs, quantized tensors, file loading
 *   weights.h       NUMA placement and repacking of the weights
 *   pool.h          thread pool, barriers, per-node reduction, rmsnorm, profiling
 *   quant.h         int8/int4 quantization and the NEON matmul kernels (single + batched)
 *   kv.h            KV cache formats (f32 / int8 with per-group scales), store/dot/axpy
 *   model_hybrid.h  Qwen3.5/3.8 decode step (Gated DeltaNet + full attention)
 *   glm.h           GLM-5.3 decode step
 *   model_dense.h   dense Llama/Qwen decode step
 *   runtime.h       thread placement, pool workers, model runtime tables
 *   batch.h         batched (prefill) infrastructure, MoE FFN, hybrid batched prefill
 *   glm_batch.h     GLM batched prefill
 *   decode_batch.h  dense batched prefill, multi-token decode, forward/forward_all
 *   tokenizer.h     BPE tokenizer
 *   sampler.h       argmax / temperature / top-p sampling (pool-parallel)
 *   spec.h          pool-parallel memcpy and the weightless speculative commit
 *   chat.h          chat history, templates, tool calls, the reply generation loop
 *   server.h        the OpenAI-compatible HTTP server (-m server)
 */
#include "core.h"
#include "weights.h"
#include "pool.h"
#include "quant.h"
#include "kv.h"
#include "model_hybrid.h"
#include "glm.h"
#include "model_dense.h"
#include "runtime.h"
#include "batch.h"
#include "glm_batch.h"
#include "decode_batch.h"
#include "tokenizer.h"
#include "sampler.h"
#include "spec.h"
#include "chat.h"

#ifndef TESTING

static const char *SRV_HOST = "127.0.0.1";
static int SRV_PORT = 8080;
static const char *SRV_KEY = NULL;     // -A: required "Authorization: Bearer <key>"
static const char *SRV_MODEL = "numallama";
static const char *SRV_SESS_DIR = NULL; // -W: session journal / state dumps (default: workspace/sessions)
static Tools *SRV_TOOLS = NULL;         // -f tools.json: tools the server executes itself when the request has none

// Server session persistence (enabled by default in server mode, -W <dir>): per session an
// append-only JSONL journal of the requests plus a binary state dump (prompt tokens + KV cache /
// recurrent state + GLM caches). After a server restart a reconnecting client resumes from the
// dump: only the tokens it added since the dump are processed. Writes go to "<path>.tmp" + rename.

#define SESS_MAGIC "NASESS01"
#define SESS_VER 2 // 2: head-major rows, kv_type + group size in the header (1: legacy f32 dump)

static int w_blk(FILE *f, const void *p, size_t bytes) { return bytes == 0 || fwrite(p, 1, bytes, f) == bytes; }
static int r_blk(FILE *f, void *p, size_t bytes) { return bytes == 0 || fread(p, 1, bytes, f) == bytes; }

// contiguous staging buffer for the strided per-layer cache rows (the server is single-threaded)
static void *sess_stage;
static size_t sess_stage_bytes;

static void *sess_stage_for(size_t bytes) {
  if (bytes > sess_stage_bytes) {
    free(sess_stage);
    sess_stage_bytes = 0;
    sess_stage = malloc(bytes);
    if (!sess_stage) return NULL;
    sess_stage_bytes = bytes;
  }
  return sess_stage;
}

// the position-indexed caches for tokens [0..n) as one contiguous blob; the same layout backs the
// on-disk dumps, the in-memory session snapshots and the system-prefix slots. Attention / hybrid:
// per layer the K rows of every kv head [0..n), then V (head-major, the same order as in memory),
// then the scales; GLM (always f32): latent, indexer keys, indexer gates, pooled indexer keys.
static size_t sess_region_bytes(const Config *p, int n, int kv_type) {
  if (p->glm)
    return ((size_t)p->n_attn * n * (p->g_kvl + 2 * p->g_ih) +
            (size_t)p->n_attn * (n / p->g_kpool + 1) * p->g_ih) * sizeof(float);
  size_t vals = (size_t)p->n_attn * n * p->kv_dim * 2;
  if (kv_type == KV_I8) return vals + vals / GS * sizeof(float);
  return vals * sizeof(float);
}

static void sess_region_save(const Transformer *t, int n, void *dst) {
  const Config *p = &t->config;
  const RunState *s = &t->state;
  uint8_t *d = dst;
  if (!p->glm) {
    const int hs = p->head_dim, nk = p->n_kv_heads;
    const size_t rb = (size_t)n * hs * (s->kv_type == KV_I8 ? 1 : 4);
    const size_t sb = s->kv_type == KV_I8 ? (size_t)n * hs / GS * sizeof(float) : 0;
    for (int la = 0; la < p->n_attn; la++) {
      if (s->kv_type == KV_I8) {
        for (int g = 0; g < nk; g++) { memcpy(d, s->key_cache_q + kv_row_off(p, la, g, 0), rb); d += rb; }
        for (int g = 0; g < nk; g++) { memcpy(d, s->value_cache_q + kv_row_off(p, la, g, 0), rb); d += rb; }
        for (int g = 0; g < nk; g++) { memcpy(d, s->key_cache_s + kv_sc_off(p, la, g, 0), sb); d += sb; }
        for (int g = 0; g < nk; g++) { memcpy(d, s->value_cache_s + kv_sc_off(p, la, g, 0), sb); d += sb; }
      } else {
        for (int g = 0; g < nk; g++) { memcpy(d, s->key_cache + kv_row_off(p, la, g, 0), rb); d += rb; }
        for (int g = 0; g < nk; g++) { memcpy(d, s->value_cache + kv_row_off(p, la, g, 0), rb); d += rb; }
      }
    }
  } else {
    const GlmState *g = s->glm;
    const size_t kvl = p->g_kvl, ih = p->g_ih;
    for (int la = 0; la < p->n_attn; la++) {
      const float *base = g->lat[0] + (size_t)la * p->seq_len * kvl;
      memcpy(d, base, (size_t)n * kvl * sizeof(float));
      d += (size_t)n * kvl * sizeof(float);
    }
    for (int la = 0; la < p->n_attn; la++) {
      const float *base = g->ikey + (size_t)la * p->seq_len * ih;
      memcpy(d, base, (size_t)n * ih * sizeof(float));
      d += (size_t)n * ih * sizeof(float);
    }
    for (int la = 0; la < p->n_attn; la++) {
      const float *base = g->igate + (size_t)la * p->seq_len * ih;
      memcpy(d, base, (size_t)n * ih * sizeof(float));
      d += (size_t)n * ih * sizeof(float);
    }
    int npool = n / p->g_kpool + 1;
    for (int la = 0; la < p->n_attn; la++) {
      const float *base = g->ipool + (size_t)la * (p->seq_len / p->g_kpool + 1) * ih;
      memcpy(d, base, (size_t)npool * ih * sizeof(float)); // contiguous: pool rows are never rewritten
      d += (size_t)npool * ih * sizeof(float);
    }
  }
}

static void sess_region_load(Transformer *t, int n, const void *src) {
  const Config *p = &t->config;
  RunState *s = &t->state;
  const uint8_t *d = src;
  if (!p->glm) {
    const int hs = p->head_dim, nk = p->n_kv_heads;
    const size_t rb = (size_t)n * hs * (s->kv_type == KV_I8 ? 1 : 4);
    const size_t sb = s->kv_type == KV_I8 ? (size_t)n * hs / GS * sizeof(float) : 0;
    for (int la = 0; la < p->n_attn; la++) {
      if (s->kv_type == KV_I8) {
        for (int g = 0; g < nk; g++) { memcpy(s->key_cache_q + kv_row_off(p, la, g, 0), d, rb); d += rb; }
        for (int g = 0; g < nk; g++) { memcpy(s->value_cache_q + kv_row_off(p, la, g, 0), d, rb); d += rb; }
        for (int g = 0; g < nk; g++) { memcpy(s->key_cache_s + kv_sc_off(p, la, g, 0), d, sb); d += sb; }
        for (int g = 0; g < nk; g++) { memcpy(s->value_cache_s + kv_sc_off(p, la, g, 0), d, sb); d += sb; }
      } else {
        for (int g = 0; g < nk; g++) { memcpy(s->key_cache + kv_row_off(p, la, g, 0), d, rb); d += rb; }
        for (int g = 0; g < nk; g++) { memcpy(s->value_cache + kv_row_off(p, la, g, 0), d, rb); d += rb; }
      }
    }
  } else {
    GlmState *g = s->glm;
    const size_t kvl = p->g_kvl, ih = p->g_ih;
    for (int la = 0; la < p->n_attn; la++) {
      float *base = g->lat[0] + (size_t)la * p->seq_len * kvl;
      memcpy(base, d, (size_t)n * kvl * sizeof(float));
      d += (size_t)n * kvl * sizeof(float);
    }
    for (int la = 0; la < p->n_attn; la++) {
      float *base = g->ikey + (size_t)la * p->seq_len * ih;
      memcpy(base, d, (size_t)n * ih * sizeof(float));
      d += (size_t)n * ih * sizeof(float);
    }
    for (int la = 0; la < p->n_attn; la++) {
      float *base = g->igate + (size_t)la * p->seq_len * ih;
      memcpy(base, d, (size_t)n * ih * sizeof(float));
      d += (size_t)n * ih * sizeof(float);
    }
    int npool = n / p->g_kpool + 1;
    for (int la = 0; la < p->n_attn; la++) {
      float *base = g->ipool + (size_t)la * (p->seq_len / p->g_kpool + 1) * ih;
      memcpy(base, d, (size_t)npool * ih * sizeof(float));
      d += (size_t)npool * ih * sizeof(float);
    }
    for (int k2 = 1; k2 < NN; k2++) // the per-node latent copies are identical: rebuild them from lat[0]
      for (int la = 0; la < p->n_attn; la++)
        memcpy(g->lat[k2] + (size_t)la * p->seq_len * kvl, g->lat[0] + (size_t)la * p->seq_len * kvl, (size_t)n * kvl * sizeof(float));
  }
}

// upper bound of the dump for n tokens (for the free-space check)
static size_t sess_dump_bytes(const Transformer *t, int n) {
  const Config *p = &t->config;
  size_t b = 8 + 9 * 4 + 64 + 4 + (size_t)n * sizeof(int);
  if (RECURRENT(p)) b += 4 + (hyb_state_floats(p) + hyb_conv_floats(p)) * sizeof(float);
  b += sess_region_bytes(p, n, t->state.kv_type);
  if (RECURRENT(p) && t->state.snap_S && t->state.snap_pos > 0) b += 4 + (hyb_state_floats(p) + hyb_conv_floats(p)) * sizeof(float);
  return b;
}

// write the state that corresponds to tok[0..n); 1 = ok (dump complete and renamed)
static int sess_write(const Transformer *t, const int *tok, int n, const char *path) {
  const Config *p = &t->config;
  const RunState *s = &t->state;
  size_t linf = RECURRENT(p) ? hyb_state_floats(p) : 0;
  size_t convf = RECURRENT(p) ? hyb_conv_floats(p) : 0;
  int snap = RECURRENT(p) && s->snap_S && s->snap_pos > 0;
  char tmp[1210];
  snprintf(tmp, sizeof tmp, "%s.tmp", path);
  FILE *f = fopen(tmp, "wb");
  if (!f) return 0;
  int ok = 1;
  ok &= w_blk(f, SESS_MAGIC, 8);
  uint32_t hdr[9] = {SESS_VER, (RECURRENT(p) ? 1u : 0u) | (snap ? 2u : 0u), (uint32_t)p->vocab_size,
                     (uint32_t)p->dim, (uint32_t)p->n_layers, (uint32_t)p->n_attn, (uint32_t)ARCH,
                     (uint32_t)s->kv_type, (uint32_t)GS};
  ok &= w_blk(f, hdr, sizeof hdr);
  char model[64] = {0};
  snprintf(model, sizeof model, "%s", SRV_MODEL);
  ok &= w_blk(f, model, sizeof model);
  uint32_t nn = (uint32_t)n;
  ok &= w_blk(f, &nn, 4);
  ok &= w_blk(f, tok, (size_t)n * sizeof(int));
  if (RECURRENT(p)) {
    uint32_t np = (uint32_t)s->next_pos;
    ok &= w_blk(f, &np, 4);
    ok &= w_blk(f, s->lin_S, linf * sizeof(float));
    ok &= w_blk(f, s->conv_state, convf * sizeof(float));
  }
  size_t rf = sess_region_bytes(p, n, s->kv_type);
  void *rb = sess_stage_for(rf);
  ok &= rb != NULL;
  if (ok) sess_region_save(t, n, rb);
  ok &= w_blk(f, rb, rf);
  if (snap) {
    uint32_t sp = (uint32_t)s->snap_pos;
    ok &= w_blk(f, &sp, 4);
    ok &= w_blk(f, s->snap_S, linf * sizeof(float));
    ok &= w_blk(f, s->snap_conv, convf * sizeof(float));
  }
  ok &= fclose(f) == 0;
  if (!ok) { remove(tmp); return 0; }
  if (rename(tmp, path) != 0) { remove(tmp); return 0; }
  return 1;
}

// load a dump written by sess_write: validates it against the loaded model, fills tok[0..n) (cap
// entries max) and restores the model state to it. Returns the token count, 0 = no usable dump.
static int sess_read(Transformer *t, int *tok, int cap, const char *path) {
  const Config *p = &t->config;
  RunState *s = &t->state;
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  int n = 0, ok = 1;
  char magic[8];
  ok &= r_blk(f, magic, 8) && !memcmp(magic, SESS_MAGIC, 8);
  uint32_t hdr[9] = {0};
  ok &= r_blk(f, hdr, sizeof hdr);
  ok &= ok && hdr[0] == SESS_VER && (hdr[1] & 1) == (RECURRENT(p) ? 1u : 0u) && hdr[2] == (uint32_t)p->vocab_size &&
         hdr[3] == (uint32_t)p->dim && hdr[4] == (uint32_t)p->n_layers && hdr[5] == (uint32_t)p->n_attn &&
         hdr[6] == (uint32_t)ARCH && hdr[7] == (uint32_t)s->kv_type && hdr[8] == (uint32_t)GS;
  char model[64] = {0};
  ok &= r_blk(f, model, sizeof model) && !strcmp(model, SRV_MODEL);
  uint32_t nn = 0;
  ok &= r_blk(f, &nn, 4);
  n = (int)nn;
  ok &= ok && n > 0 && n <= cap && n <= p->seq_len;
  if (ok) {
    int *tt = malloc((size_t)n * sizeof(int));
    if (!tt || !r_blk(f, tt, (size_t)n * sizeof(int))) ok = 0;
    else memcpy(tok, tt, (size_t)n * sizeof(int));
    free(tt);
  }
  if (ok && RECURRENT(p)) {
    uint32_t np = 0;
    ok &= r_blk(f, &np, 4);
    ok &= r_blk(f, s->lin_S, hyb_state_floats(p) * sizeof(float));
    ok &= r_blk(f, s->conv_state, hyb_conv_floats(p) * sizeof(float));
    s->next_pos = (int)np;
  }
  size_t rf = sess_region_bytes(p, n, s->kv_type);
  void *rb = sess_stage_for(rf);
  ok &= rb != NULL;
  ok &= r_blk(f, rb, rf);
  if (ok) sess_region_load(t, n, rb);
  if (ok && RECURRENT(p) && (hdr[1] & 2)) {
    if (!s->snap_S) {
      s->snap_S = malloc(hyb_state_floats(p) * sizeof(float));
      s->snap_conv = malloc(hyb_conv_floats(p) * sizeof(float));
    }
    uint32_t sp = 0;
    ok &= s->snap_S && s->snap_conv;
    ok &= r_blk(f, &sp, 4);
    ok &= r_blk(f, s->snap_S, hyb_state_floats(p) * sizeof(float));
    ok &= r_blk(f, s->snap_conv, hyb_conv_floats(p) * sizeof(float));
    s->snap_pos = (int)sp;
  }
  fclose(f);
  return ok ? n : 0;
}

#include "server.h"

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 4096 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 4096. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat|server|imatrix, default: generate\n");
  fprintf(stderr, "              server: OpenAI-compatible API (/v1/chat/completions, /v1/models)\n");
  fprintf(stderr, "  -H <string> server: address to listen on, default 127.0.0.1 (0.0.0.0 = all interfaces)\n");
  fprintf(stderr, "  -P <int>    server: port, default 8080\n");
  fprintf(stderr, "  -A <string> server: API key the clients must send (Authorization: Bearer ...)\n");
  fprintf(stderr, "  -W <dir>    server: sessions directory (journal + state dumps; \"none\" disables, default workspace/sessions)\n");
  fprintf(stderr, "              imatrix: -i <text file> -o <out.imx> collects Q4 importance statistics\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr, "  -T <int>    number of threads, default 0 = auto (placed on the most idle CPUs; RUNQ_PIN=1 keeps them pinned, RUNQ_PIN=0 disables placement)\n");
  fprintf(stderr, "  -c <int>    context window in tokens (KV cache size), default 8192\n");
  fprintf(stderr, "  -K <string> KV cache type: i8|f32, default i8 (f32 = full precision, 4x more memory)\n");
  fprintf(stderr, "  -f <string> tools.json: enables tool calling in chat mode\n");
  fprintf(stderr, "  -d <string> draft model (same tokenizer, smaller) for speculative decoding in chat\n");
  fprintf(stderr, "  -k <int>    max draft length of prompt-lookup speculative decoding in chat, default 8, 0 = off\n");
  fprintf(stderr, "  -N <int>    NUMA nodes to use, default 0 = all (weights are split between the nodes)\n");
  fprintf(stderr, "  -r <int>    Qwen3.5 / 3.8: 1 = reasoning (thinking shown dimmed), default 0 = answer directly\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  // default parameters
  char *checkpoint_path = NULL; // e.g. out/model.bin
  char *tokenizer_path = "data/tokenizer.bin";
  float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 4096;                // number of steps to run for
  char *prompt = NULL;             // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  char *mode = "generate";         // generate|chat
  char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode
  int nthreads = 0;                // threads for the forward pass, 0 = auto
  int ctx = 8192;                  // context window (KV cache size) in tokens
  char *tools_path = NULL;         // tools.json for tool calling in chat mode
  char *out_path = NULL;           // -o: output file (imatrix mode)
  char *draft_path = NULL;         // draft model for speculative decoding
  int spec_k_set = 0;              // -k was given explicitly

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage();
    } // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else if (argv[i][1] == 'm') {
      mode = argv[i + 1];
    } else if (argv[i][1] == 'y') {
      system_prompt = argv[i + 1];
    } else if (argv[i][1] == 'T') {
      nthreads = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'c') {
      ctx = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'K') {
      KV_REQ = kv_type_parse(argv[i + 1]);
      if (KV_REQ < 0) error_usage();
    } else if (argv[i][1] == 'd') {
      draft_path = argv[i + 1];
    } else if (argv[i][1] == 'k') {
      SPEC_K = atoi(argv[i + 1]);
      spec_k_set = 1;
    } else if (argv[i][1] == 'N') {
      NODES_REQ = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'f') {
      tools_path = argv[i + 1];
    } else if (argv[i][1] == 'o') {
      out_path = argv[i + 1];
    } else if (argv[i][1] == 'r') {
      THINK = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'H') {
      SRV_HOST = argv[i + 1];
    } else if (argv[i][1] == 'P') {
      SRV_PORT = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'A') {
      SRV_KEY = argv[i + 1];
    } else if (argv[i][1] == 'W') {
      SRV_SESS_DIR = argv[i + 1];
    } else {
      error_usage();
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
  if (steps < 0)
    steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
#if defined(__linux__)
  // choose the CPUs first and load the weights from the first of them (NUMA-local memory)
  // choose the CPUs (one or more NUMA nodes) first: the weights are then placed on those nodes
  nthreads = choose_cpus(nthreads);
  if (P_cpus) {
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(P_cpus[0], &one);
    pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
  }
  fprintf(stderr, "[threads: %d on %d NUMA node%s:", nthreads, NN, NN > 1 ? "s" : "");
  for (int k = 0; k < NN; k++) fprintf(stderr, " node%d x%d", NODE_ID[k], NODE_T0[k + 1] - NODE_T0[k]);
  fprintf(stderr, "]\n");
#endif
  build_transformer(&transformer, checkpoint_path, ctx);
  init_pool(&transformer, nthreads);
  fprintf(stderr, "[kv cache: %s]\n", kv_type_name(transformer.state.kv_type));
  if (transformer.config.glm && !spec_k_set) // GLM is MoE: the verify pass reads the union of the routed experts, measured no gain
    SPEC_K = 0;
  if (RECURRENT(&transformer.config) && draft_path) { // draft-model cache rollback is not implemented for recurrent targets
    draft_path = NULL;
  }
  Transformer draft;
  if (draft_path) {
    build_transformer(&draft, draft_path, transformer.config.seq_len);
    if (draft.config.dim > transformer.config.dim || draft.config.hidden_dim > transformer.config.hidden_dim ||
        draft.config.n_heads / draft.config.n_kv_heads > transformer.config.n_heads / transformer.config.n_kv_heads) {
      fprintf(stderr, "the draft model must not be larger than the main model\n");
      exit(EXIT_FAILURE);
    }
    init_model_runtime(&draft);
    use_model(&transformer);
    g_draft = &draft;
    fprintf(stderr, "[draft model %s: %d layers, dim %d]\n", draft_path, draft.config.n_layers, draft.config.dim);
  }
  Tools tools;
  if (tools_path) {
    if (!tools_load(&tools, tools_path)) exit(EXIT_FAILURE);
    g_tools = &tools;
    SRV_TOOLS = &tools; // server mode: the fallback tool set for requests without tools
    fprintf(stderr, "[tools from %s, workspace %s]\n", tools_path, tools.workdir);
  }
  if (steps == 0 || steps > transformer.config.seq_len)
    steps = transformer.config.seq_len; // override to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // run!
  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  } else if (strcmp(mode, "chat") == 0) {
    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  } else if (strcmp(mode, "server") == 0) {
    const char *base = strrchr(checkpoint_path, '/');
    base = base ? base + 1 : checkpoint_path;
    char *name = strdup(base), *dot = strrchr(name, '.');
    if (dot && !strcmp(dot, ".bin")) *dot = '\0';
    SRV_MODEL = name;
    if (!SRV_SESS_DIR) SRV_SESS_DIR = "workspace/sessions"; // sessions survive a restart: "-W none" disables
    if (!strcmp(SRV_SESS_DIR, "none")) SRV_SESS_DIR = NULL;
    server(&transformer, &tokenizer, &sampler);
  } else if (strcmp(mode, "imatrix") == 0) {
    if (!prompt || !out_path) error_usage();
    imatrix(&transformer, &tokenizer, prompt, out_path);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_pool();
  free_transformer(&transformer);
  return 0;
}
#endif
