/* Inference for Llama-3 Transformer model in pure C, int8 quantized forward pass. */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // pthread_setaffinity_np, CPU_SET, M_PI
#endif

#include <ctype.h>
#include <fcntl.h>
#include <float.h>
#include <pthread.h>
#include <stdatomic.h>
#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tools.h"
#include <sys/mman.h>
#include <unistd.h>
// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights
// RoPE parameters, read from the extended v2 header (see export_hf_q8.py)
double ROPE_THETA = 500000.0; // Llama 3 default
double ROPE_FACTOR = 0.0;     // > 0 enables llama3 rope scaling (Llama 3.1 / 3.2)
double ROPE_LOW_FREQ = 1.0, ROPE_HIGH_FREQ = 4.0, ROPE_ORIG_CTX = 8192.0;
// architecture info from the extended header
enum { ARCH_LLAMA3 = 0, ARCH_QWEN2 = 1, ARCH_QWEN3 = 2 }; // Qwen2/3: ChatML template, Qwen tokenizer
int ARCH = ARCH_LLAMA3;
float NORM_EPS = 1e-5f;
int QKV_BIAS = 0;
int SPECIAL_START = 128000; // first special token id
int BOS_ID = 128000;
int EOS_ID = 128009;
int Q4_MASK = 0;
// NUMA: nodes in use, their OS ids; threads of node k are [NODE_T0[k], NODE_T0[k+1])
#define MAX_NODES 8
int NN = 1;
int NODE_ID[MAX_NODES] = {0};
int NODE_T0[MAX_NODES + 1] = {0, 1};
// rows [lo, hi) of a d-row matrix that live on (and are computed by) node k; multiples of 4
static inline void node_rows(int d, int k, int* lo, int* hi) {
  int bl = d / 4;
  *lo = 4 * (int)((long)bl * k / NN);
  *hi = (k == NN - 1) ? d : 4 * (int)((long)bl * (k + 1) / NN);
}
// kv heads [lo, hi) owned by node k (their KV cache lives there; attention for them runs there)
static inline void node_groups(int n_kv, int k, int* lo, int* hi) {
  *lo = n_kv * k / NN;
  *hi = n_kv * (k + 1) / NN;
} // matrix kinds stored as Q4 (bit 0 wq ... bit 6 w3)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  int dim;        // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 4096 (byte-level)
  int seq_len;    // max sequence length
  // ---- not part of the 7-int block in the file: derived / extended header
  int head_dim;   // per-head size (Qwen3: 128 even when dim / n_heads differs)
  int q_dim;      // n_heads * head_dim
  int kv_dim;     // n_kv_heads * head_dim
  int n_experts;  // MoE: experts per layer (0 = dense FFN)
  int n_active;   // MoE: experts per token
  int moe_hidden; // MoE: hidden size of one expert
  int qk_norm;    // RMSNorm on every q / k head before RoPE (Qwen3)
  int norm_topk;  // MoE: renormalize the top-k router weights
} Config;

typedef struct {
  int8_t *q; // quantized values (Q4: packed nibbles)
  float *s;  // scaling factors (Q8: per GS values, Q4: per 32 values)
  int bits;  // 8 or 4
} QuantizedTensor;

typedef struct {
  // token embedding table
  QuantizedTensor *q_tokens;    // (vocab_size, dim)
  float *token_embedding_table; // same, but dequantized

  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
  QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  QuantizedTensor *w1; // (layer, hidden_dim, dim)
  QuantizedTensor *w2; // (layer, dim, hidden_dim)
  QuantizedTensor *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  float *bq, *bk, *bv;     // (layer, q_dim) / (layer, kv_dim) qkv biases (Qwen2), or NULL
  float *q_norm, *k_norm;  // (layer, head_dim) per-head RMSNorm weights (Qwen3), or NULL
  QuantizedTensor *wg;     // MoE router (layer, n_experts, dim), or NULL; w1/w3 are then (layer, n_experts * moe_hidden, dim)
                           // and w2 (layer, n_experts * dim, moe_hidden): expert e owns a contiguous block of rows
  // (optional) classifier weights for the logits, on the last layer
  QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;           // activation at current time stamp (dim,)
  float *xb;          // same, but inside a residual branch (dim,)
  float *xb2;         // an additional buffer just for convenience (dim,)
  float *hb;          // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;         // buffer for hidden dimension in the ffn (hidden_dim,)
  QuantizedTensor xq; // quantized x (dim,)
  QuantizedTensor hq; // quantized hb (hidden_dim,)
  float *q;           // query (dim,)
  float *k;           // key (dim,)
  float *v;           // value (dim,)
  float *att;         // buffer for scores/attention values (n_heads, seq_len)
  float *logits;      // output logits
  float *router;      // MoE router logits (n_experts)
  // kv cache
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state;             // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;            // file descriptor for memory mapping
  float *data;       // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
  // per-model settings read from the header (the engine works on one model at a time; use_model() switches)
  struct {
    int gs, arch, qkv_bias, special_start, bos, eos;
    double rope_theta, rope_factor, rope_low, rope_high, rope_orig;
    float norm_eps;
  } g;
  // per-model runtime tables (set by init_model_runtime)
  double *rope_freq;
  float *part;
  int C;
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = p->kv_dim;
  int hid = p->hidden_dim;
  if (p->n_experts > 0 && p->n_active * p->moe_hidden > hid) hid = p->n_active * p->moe_hidden;
  int qd = p->q_dim > p->dim ? p->q_dim : p->dim;
  s->x = calloc(p->dim, sizeof(float));
  s->xb = calloc(qd, sizeof(float));
  s->xb2 = calloc(p->dim, sizeof(float));
  s->hb = calloc(hid, sizeof(float));
  s->hb2 = calloc(hid, sizeof(float));
  s->router = calloc(p->n_experts > 0 ? p->n_experts : 1, sizeof(float));
  s->xq = (QuantizedTensor){.q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float))};
  s->hq = (QuantizedTensor){.q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim, sizeof(float))};
  s->q = calloc(p->q_dim, sizeof(float));
  s->k = calloc(kv_dim, sizeof(float));
  s->v = calloc(kv_dim, sizeof(float));
  s->att = calloc((size_t)p->n_heads * p->seq_len, sizeof(float));
  s->logits = calloc(p->vocab_size, sizeof(float));
  s->key_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->xq.q);
  free(s->xq.s);
  free(s->hq.q);
  free(s->hq.s);
  free(s->q);
  free(s->k);
  free(s->v);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float *x, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = qx->q[i] * qx->s[i / GS];
  }
}

void quantize(QuantizedTensor *qx, float *x, int n) {
  int num_groups = n / GS;
  float Q_MAX = 127.0f;

  for (int group = 0; group < num_groups; group++) {

    // find the max absolute value in the current group
    float wmax = 0.0;
    for (int i = 0; i < GS; i++) {
      float val = fabs(x[group * GS + i]);
      if (val > wmax) {
        wmax = val;
      }
    }

    // calculate and write the scaling factor
    float scale = wmax / Q_MAX;
    qx->s[group] = scale;

    // calculate and write the quantized values
    for (int i = 0; i < GS; i++) {
      float quant_value = x[group * GS + i] / scale; // scale
      int8_t quantized = (int8_t)round(quant_value); // round and clamp
      qx->q[group * GS + i] = quantized;
    }
  }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each, int bits) {
  void *p = *ptr;
  QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
  for (int i = 0; i < n; i++) {
    res[i].bits = bits;
    res[i].q = (int8_t *)p;
    p = (int8_t *)p + (bits == 4 ? size_each / 2 : size_each);
    res[i].s = (float *)p;
    p = (float *)p + (bits == 4 ? size_each / 32 : size_each / GS);
  }
  *ptr = p; // advance ptr to current position
  return res;
}

void memory_map_weights(TransformerWeights *w, Config *p, void *ptr, uint8_t shared_classifier) {
  const int q_dim = p->q_dim, kv_dim = p->kv_dim;
  // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
  float *fptr = (float *)ptr; // cast our pointer to float*
  w->rms_att_weight = fptr;
  fptr += p->n_layers * p->dim;
  w->rms_ffn_weight = fptr;
  fptr += p->n_layers * p->dim;
  w->rms_final_weight = fptr;
  fptr += p->dim;
  w->bq = w->bk = w->bv = NULL;
  if (QKV_BIAS) {
    w->bq = fptr;
    fptr += p->n_layers * q_dim;
    w->bk = fptr;
    fptr += p->n_layers * kv_dim;
    w->bv = fptr;
    fptr += p->n_layers * kv_dim;
  }
  w->q_norm = w->k_norm = NULL;
  if (p->qk_norm) {
    w->q_norm = fptr;
    fptr += p->n_layers * p->head_dim;
    w->k_norm = fptr;
    fptr += p->n_layers * p->head_dim;
  }

  // now read all the quantized weights
  ptr = (void *)fptr; // now cast the pointer back to void*
  w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim, 8);
  // dequantize token embedding table
  w->token_embedding_table = malloc((size_t)p->vocab_size * p->dim * sizeof(float));
  // (dequantized by read_checkpoint once the data is loaded)

  w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * q_dim, Q4_MASK & 1 ? 4 : 8);
  w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * kv_dim, Q4_MASK & 2 ? 4 : 8);
  w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * kv_dim, Q4_MASK & 4 ? 4 : 8);
  w->wo = init_quantized_tensors(&ptr, p->n_layers, q_dim * p->dim, Q4_MASK & 8 ? 4 : 8);

  w->wg = NULL;
  int ffn = p->hidden_dim; // rows of w1/w3 (dense) or experts * moe_hidden
  if (p->n_experts > 0) {
    w->wg = init_quantized_tensors(&ptr, p->n_layers, p->n_experts * p->dim, 8);
    ffn = p->n_experts * p->moe_hidden;
  }
  w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * ffn, Q4_MASK & 16 ? 4 : 8);
  w->w2 = init_quantized_tensors(&ptr, p->n_layers, ffn * p->dim, Q4_MASK & 32 ? 4 : 8);
  w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * ffn, Q4_MASK & 64 ? 4 : 8);

  w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size, 8);
}

#if defined(__linux__)
#include <sys/syscall.h>
// bind the pages fully inside [a, a+len) to a NUMA node (before they are touched)
static void bind_range(const void *a, size_t len, int node) {
  uintptr_t s0 = ((uintptr_t)a + 4095) & ~(uintptr_t)4095, e0 = ((uintptr_t)a + len) & ~(uintptr_t)4095;
  if (e0 <= s0) return;
  unsigned long mask[4] = {0};
  mask[node / 64] = 1UL << (node % 64);
  syscall(SYS_mbind, s0, e0 - s0, 2 /* MPOL_BIND */, mask, 256, 0);
}
// split every row-partitioned tensor across the nodes, matching node_rows()
static void place_tensor(QuantizedTensor *t, int n, int rows, int cols) {
  size_t bpr = t->bits == 4 ? cols / 2 : cols, spr = t->bits == 4 ? cols / 32 : cols / GS;
  for (int i = 0; i < n; i++)
    for (int k = 0; k < NN; k++) {
      int lo, hi;
      node_rows(rows, k, &lo, &hi);
      bind_range(t[i].q + lo * bpr, (hi - lo) * bpr, NODE_ID[k]);
      bind_range(t[i].s + lo * spr, (hi - lo) * spr * sizeof(float), NODE_ID[k]);
    }
}
// MoE: every expert block (rows [e*rows_per_expert, (e+1)*rows_per_expert)) is split between the
// nodes like a dense matrix, so the 8 experts of a token always put the same load on every node
static void place_experts(QuantizedTensor *t, int n, int experts, int rows_per_expert, int cols) {
  size_t bpr = t->bits == 4 ? cols / 2 : cols, spr = t->bits == 4 ? cols / 32 : cols / GS;
  for (int i = 0; i < n; i++)
    for (int e = 0; e < experts; e++)
      for (int k = 0; k < NN; k++) {
        int lo, hi;
        node_rows(rows_per_expert, k, &lo, &hi);
        size_t r0 = (size_t)e * rows_per_expert + lo;
        bind_range(t[i].q + r0 * bpr, (hi - lo) * bpr, NODE_ID[k]);
        bind_range(t[i].s + r0 * spr, (hi - lo) * spr * sizeof(float), NODE_ID[k]);
      }
}
static void place_weights(TransformerWeights *w, Config *p) {
  place_tensor(w->wq, p->n_layers, p->q_dim, p->dim);
  place_tensor(w->wk, p->n_layers, p->kv_dim, p->dim);
  place_tensor(w->wv, p->n_layers, p->kv_dim, p->dim);
  place_tensor(w->wo, p->n_layers, p->dim, p->q_dim);
  if (p->n_experts > 0) {
    place_tensor(w->wg, p->n_layers, p->n_experts, p->dim);
    place_experts(w->w1, p->n_layers, p->n_experts, p->moe_hidden, p->dim);
    place_experts(w->w3, p->n_layers, p->n_experts, p->moe_hidden, p->dim);
    place_experts(w->w2, p->n_layers, p->n_experts, p->dim, p->moe_hidden);
  } else {
    place_tensor(w->w1, p->n_layers, p->hidden_dim, p->dim);
    place_tensor(w->w2, p->n_layers, p->dim, p->hidden_dim);
    place_tensor(w->w3, p->n_layers, p->hidden_dim, p->dim);
  }
  place_tensor(w->wcls, 1, p->vocab_size, p->dim);
}
#endif

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights, int *fd, float **data, ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  if (magic_number != 0x616b3432) {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }
  // read in the version number (uint32), has to be 2
  int version;
  if (fread(&version, sizeof(int), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  if (version != 2) {
    fprintf(stderr, "Bad version %d, need version 2\n", version);
    exit(EXIT_FAILURE);
  }
  int header_size = 256; // the header size for version 2 in bytes
  // read in the Config
  memset(config, 0, sizeof(Config));
  if (fread(config, 7 * sizeof(int), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // read in flags
  uint8_t shared_classifier; // a byte to indicate if the classifier is shared
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  int group_size; // the group size used in quantization
  if (fread(&group_size, sizeof(int), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  GS = group_size; // set as global, as it will be used in many places
  // optional rope parameters (zeros in files written by the original export.py)
  struct {
    float theta, factor, low, high;
    int orig;
  } rope = {0};
  if (fread(&rope, sizeof(rope), 1, file) == 1 && rope.theta > 0.0f) {
    ROPE_THETA = rope.theta;
    ROPE_FACTOR = rope.factor;
    ROPE_LOW_FREQ = rope.low;
    ROPE_HIGH_FREQ = rope.high;
    ROPE_ORIG_CTX = rope.orig;
  }
  struct {
    int arch;
    float eps;
    int qkv_bias, special_start, bos, eos;
  } ext = {0};
  if (fread(&ext, sizeof(ext), 1, file) == 1 && ext.eps > 0.0f) {
    ARCH = ext.arch;
    NORM_EPS = ext.eps;
    QKV_BIAS = ext.qkv_bias;
    SPECIAL_START = ext.special_start;
    BOS_ID = ext.bos;
    EOS_ID = ext.eos;
  }
  int q4mask = 0;
  if (fread(&q4mask, sizeof(int), 1, file) == 1) Q4_MASK = q4mask;
  int ext2[6] = {0}; // head_dim, n_experts, n_active, moe_hidden, qk_norm, norm_topk
  if (fread(ext2, sizeof(ext2), 1, file) != 1) memset(ext2, 0, sizeof(ext2));
  config->head_dim = ext2[0] > 0 ? ext2[0] : config->dim / config->n_heads;
  config->n_experts = ext2[1];
  config->n_active = ext2[2];
  config->moe_hidden = ext2[3];
  config->qk_norm = ext2[4];
  config->norm_topk = ext2[5];
  config->q_dim = config->n_heads * config->head_dim;
  config->kv_dim = config->n_kv_heads * config->head_dim;
  // figure out the file size
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
  // read the whole checkpoint into anonymous memory (NUMA-local, huge pages) instead of mmap
  *fd = -1;
  if (posix_memalign((void **)data, 2 * 1024 * 1024, *file_size) != 0) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
#if defined(MADV_HUGEPAGE)
  madvise(*data, *file_size, MADV_HUGEPAGE);
#endif
  void *weights_ptr = ((char *)*data) + header_size; // skip header bytes. char is 1 byte
  memory_map_weights(weights, config, weights_ptr, shared_classifier); // pointers only, nothing touched yet
#if defined(__linux__)
  if (NN > 1) place_weights(weights, config); // each node gets its rows of every matrix
#endif
  fseek(file, 0, SEEK_SET);
  if (fread(*data, 1, *file_size, file) != (size_t)*file_size) {
    fprintf(stderr, "read failed!\n");
    exit(EXIT_FAILURE);
  }
  fclose(file);
  dequantize(weights->q_tokens, weights->token_embedding_table, config->vocab_size * config->dim);
}

void build_transformer(Transformer *t, char *checkpoint_path, int ctx) {
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  t->g.gs = GS; t->g.arch = ARCH; t->g.qkv_bias = QKV_BIAS; t->g.special_start = SPECIAL_START;
  t->g.bos = BOS_ID; t->g.eos = EOS_ID; t->g.norm_eps = NORM_EPS;
  t->g.rope_theta = ROPE_THETA; t->g.rope_factor = ROPE_FACTOR; t->g.rope_low = ROPE_LOW_FREQ;
  t->g.rope_high = ROPE_HIGH_FREQ; t->g.rope_orig = ROPE_ORIG_CTX;
  t->rope_freq = NULL; t->part = NULL; t->C = 0;
  // limit the context window (the KV cache is allocated for seq_len positions)
  if (ctx > 0 && ctx < t->config.seq_len) {
    t->config.seq_len = ctx;
  }
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
#if defined(__linux__)
  if (NN > 1) { // KV cache of kv head g lives on the node that runs its attention
    Config *p = &t->config;
    int hs = p->head_dim;
    size_t head_bytes = (size_t)p->seq_len * hs * sizeof(float);
    for (int l = 0; l < p->n_layers; l++)
      for (int k = 0; k < NN; k++) {
        int g0, g1;
        node_groups(p->n_kv_heads, k, &g0, &g1);
        size_t off = ((size_t)l * p->n_kv_heads + g0) * head_bytes;
        bind_range((char *)t->state.key_cache + off, (g1 - g0) * head_bytes, NODE_ID[k]);
        bind_range((char *)t->state.value_cache + off, (g1 - g0) * head_bytes, NODE_ID[k]);
      }
  }
#endif
}

void free_transformer(Transformer *t) {
  // free QuantizedTensors
  free(t->weights.q_tokens);
  free(t->weights.token_embedding_table);
  free(t->weights.wq);
  free(t->weights.wk);
  free(t->weights.wv);
  free(t->weights.wo);
  free(t->weights.w1);
  free(t->weights.w2);
  free(t->weights.w3);
  if (t->weights.wcls != t->weights.q_tokens) {
    free(t->weights.wcls);
  }
  // close the memory mapping
  free(t->data);
  // free the RunState buffers
  free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += NORM_EPS;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

// ----------------------------------------------------------------------------
// Parallel runtime: a persistent pool of pinned threads that run the whole
// forward pass together, synchronizing with a spin barrier between phases.
// Small serial steps (rmsnorm, quantization of activations, attention merge)
// are computed redundantly by every thread into private buffers, which is
// cheaper than an extra barrier for models of this size.

typedef struct {
    float* xb;     // (dim,) private normalized input / attention output
    int8_t* q;     // (max(dim, hidden_dim),) private quantized activations
    float* s;      // scale factors for q
    float* rope_c; // (head_size/2,) cos for the current position
    float* rope_s; // (head_size/2,) sin for the current position
    int lsense;    // barrier local sense
    int node;      // NUMA node index (0..NN-1) of this thread
    int ltid, nthn; // index within the node, threads of the node
    int lsense_node; // node barrier local sense
} ThreadBuf;

typedef struct {
    int nth;            // number of threads, including the caller
    int C;              // attention context chunks per head
    Transformer* t;
    int token, pos;
    int mode;           // 0: one token (forward_thread), 1: a block of tokens (forward_batch_thread)
    struct WorkCounter { _Alignas(64) atomic_int v; }* work[MAX_NODES][2]; // per node, per token parity: next free row of each phase
    int seq;            // forward call counter (selects the counter set)
    pthread_t* tids;
    ThreadBuf* tb;
    float* part;        // (n_heads * C, head_size + 2) attention partials: max, sum, acc
    double* rope_freq;  // (head_size/2,)
    _Alignas(64) atomic_int bar_count;   // top level of the barrier tree (one arrival per group)
    _Alignas(64) atomic_int bar_sense;
    struct BarGroup { _Alignas(64) atomic_int c; }* bar_group; // first level: groups of BAR_GROUP threads
    _Alignas(64) atomic_int go;
    _Alignas(64) atomic_int sleepers;   // workers blocked in futex_wait on `go`
    atomic_int quit;
} Pool;

static Pool P;

/* Dynamic row distribution for the decode matmuls: threads grab blocks of rows from a shared
 * counter, so a core that gets less memory bandwidth (mesh position) simply does fewer blocks
 * instead of making everybody wait at the next barrier. */
// phase k of a token, d-row matrix: the threads of a node take blocks of that node's rows only
// (the counters of a node live in that node's memory: a remote atomic per block was a bottleneck)
static inline int grab_rows(int k, int d, const ThreadBuf* tb, int* r0, int* r1) {
    int lo, hi;
    node_rows(d, tb->node, &lo, &hi);
    int n = hi - lo;
    int chunk = (n / (tb->nthn * 4) + 3) & ~3;
    if (chunk < 16) chunk = 16;
    int b = atomic_fetch_add_explicit(&P.work[tb->node][P.seq & 1][k].v, chunk, memory_order_relaxed);
    if (b >= n) return 0;
    *r0 = lo + b;
    *r1 = lo + (b + chunk < n ? b + chunk : n);
    return 1;
}
// like grab_rows, over a node-local virtual range [0, n) (MoE: the node's halves of the selected experts)
static inline int grab_virtual(int k, int n, const ThreadBuf* tb, int* r0, int* r1) {
    int chunk = (n / (tb->nthn * 4) + 15) & ~15;
    if (chunk < 16) chunk = 16;
    int b = atomic_fetch_add_explicit(&P.work[tb->node][P.seq & 1][k].v, chunk, memory_order_relaxed);
    if (b >= n) return 0;
    *r0 = b;
    *r1 = b + chunk < n ? b + chunk : n;
    return 1;
}
// rows [e*rows, (e+1)*rows) of a tensor as a tensor of its own (an expert of a MoE layer)
static inline QuantizedTensor expert_view(const QuantizedTensor* t, int e, int rows, int cols) {
    size_t bpr = t->bits == 4 ? cols / 2 : cols, spr = t->bits == 4 ? cols / 32 : cols / GS;
    QuantizedTensor v = {t->q + (size_t)e * rows * bpr, t->s + (size_t)e * rows * spr, t->bits};
    return v;
}
#define MOE_MAX_ACTIVE 16
// MoE routing (Qwen3-MoE): softmax over the experts, top-k, optional renormalization
static void moe_route(const float* logits, int E, int k, int renorm, int* sel, float* wt) {
    float m = logits[0], z = 0.0f;
    for (int e = 1; e < E; e++) m = logits[e] > m ? logits[e] : m;
    for (int e = 0; e < E; e++) z += expf(logits[e] - m);
    for (int j = 0; j < k; j++) { sel[j] = -1; wt[j] = -1.0f; }
    for (int e = 0; e < E; e++) { // insertion into the sorted top-k
        float pr = expf(logits[e] - m) / z;
        int j = k - 1;
        if (pr <= wt[j]) continue;
        while (j > 0 && wt[j - 1] < pr) { wt[j] = wt[j - 1]; sel[j] = sel[j - 1]; j--; }
        wt[j] = pr;
        sel[j] = e;
    }
    if (renorm) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++) sum += wt[j];
        for (int j = 0; j < k; j++) wt[j] /= sum;
    }
}
// per-head RMSNorm (Qwen3 q_norm / k_norm) followed by RoPE on adjacent pairs
static inline void head_norm_rope(float* v, const float* w, int hs, const float* rc, const float* rs) {
    float ss = 0.0f;
    for (int i = 0; i < hs; i++) ss += v[i] * v[i];
    float inv = 1.0f / sqrtf(ss / hs + NORM_EPS);
    for (int i = 0; i < hs; i++) v[i] = w[i] * (v[i] * inv);
    for (int i = 0; i < hs; i += 2) {
        float v0 = v[i], v1 = v[i + 1];
        v[i] = v0 * rc[i / 2] - v1 * rs[i / 2];
        v[i + 1] = v0 * rs[i / 2] + v1 * rc[i / 2];
    }
}
#define WORK_PHASES 7 // per layer: q, k, v, wo, w1/w3 (or experts), w2, MoE router (+1 classifier)

// ---------------------------------------------------------------------------- profiling (-DPROFILE)
enum { PH_ROPE, PH_NORM, PH_QKV, PH_ATT, PH_MERGE, PH_WO, PH_FNORM, PH_W13, PH_QH, PH_W2, PH_CLS, PH_BARRIER, PH_IDLE, PH_N };
#ifdef PROFILE
static const char* PH_NAME[PH_N] = {"rope/setup", "att rmsnorm+quant", "qkv matmul+rope", "attention", "att merge+quant",
                                    "wo matmul", "ffn rmsnorm+quant", "w1/w3 matmul+swiglu", "quant hidden", "w2 matmul",
                                    "final norm+classifier", "barrier wait", "idle (outside forward)"};
#endif
#ifdef PROFILE
static inline uint64_t prof_ticks(void) { uint64_t v; __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(v)); return v; }
static inline uint64_t prof_freq(void) { uint64_t v; __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(v)); return v; }
typedef struct { _Alignas(64) uint64_t t[PH_N]; uint64_t last; int cur; } ProfThread;
static ProfThread PROF_T[1024];
static long PROF_TOKENS = 0;
#define PROF(tid, ph) do { ProfThread* pt_ = &PROF_T[tid]; uint64_t n_ = prof_ticks(); pt_->t[pt_->cur] += n_ - pt_->last; pt_->last = n_; pt_->cur = (ph); } while (0)
static void prof_init(int nth) {
    uint64_t n = prof_ticks();
    for (int i = 0; i < nth; i++) { PROF_T[i].last = n; PROF_T[i].cur = PH_IDLE; }
}
static void prof_report(int nth) {
    if (PROF_TOKENS == 0) return;
    double f = (double)prof_freq();
    double tot0 = 0, totm = 0;
    fprintf(stderr, "\n[profile] %ld forward calls, %d threads, ms per call: thread0 | mean over threads | max thread\n", PROF_TOKENS, nth);
    for (int ph = 0; ph < PH_N; ph++) {
        double t0 = PROF_T[0].t[ph] / f * 1e3 / PROF_TOKENS, mean = 0, mx = 0;
        for (int i = 0; i < nth; i++) { double v = PROF_T[i].t[ph] / f * 1e3 / PROF_TOKENS; mean += v; if (v > mx) mx = v; }
        mean /= nth;
        tot0 += t0; totm += mean;
        fprintf(stderr, "  %-24s %8.3f | %8.3f | %8.3f", PH_NAME[ph], t0, mean, mx);
        for (int k = 0; k < NN; k++) {
            double m = 0;
            for (int i = NODE_T0[k]; i < NODE_T0[k + 1]; i++) m += PROF_T[i].t[ph] / f * 1e3 / PROF_TOKENS;
            fprintf(stderr, " | node%d %8.3f", NODE_ID[k], m / (NODE_T0[k + 1] - NODE_T0[k]));
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "  %-24s %8.3f | %8.3f\n", "total", tot0, totm);
}
#else
#define PROF(tid, ph) do { } while (0)
static void prof_report(int nth) { (void)nth; }
static void prof_init(int nth) { (void)nth; }
#endif

#define KV_MUL_MAX 16 // max query heads per kv head (GQA group size)

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

static void barrier_impl(ThreadBuf* tb);
static void barrier(ThreadBuf* tb) {
#ifdef PROFILE
    int tid = (int)(tb - P.tb), saved = PROF_T[tid].cur;
    PROF(tid, PH_BARRIER);
    barrier_impl(tb);
    PROF(tid, saved);
#else
    barrier_impl(tb);
#endif
}
/* Two-level barrier: threads arrive at the counter of their group of BAR_GROUP, the last one of
 * each group arrives at the top counter, the very last releases everybody through one sense flag.
 * 96 threads: ~2.3 us instead of ~8.7 us for a single counter (fewer serialized atomics on one line). */
#define BAR_GROUP 8
static void barrier_impl(ThreadBuf* tb) {
    const int nth = P.nth;
    if (nth == 1) return;
    const int tid = (int)(tb - P.tb);
    const int g = tid / BAR_GROUP, ngroups = (nth + BAR_GROUP - 1) / BAR_GROUP;
    const int gsize = (g == ngroups - 1) ? nth - g * BAR_GROUP : BAR_GROUP;
    int sense = (tb->lsense ^= 1);
    if (atomic_fetch_add_explicit(&P.bar_group[g].c, 1, memory_order_acq_rel) == gsize - 1) {
        atomic_store_explicit(&P.bar_group[g].c, 0, memory_order_relaxed);
        if (atomic_fetch_add_explicit(&P.bar_count, 1, memory_order_acq_rel) == ngroups - 1) {
            atomic_store_explicit(&P.bar_count, 0, memory_order_relaxed);
            atomic_store_explicit(&P.bar_sense, sense, memory_order_release);
            return;
        }
    }
    while (atomic_load_explicit(&P.bar_sense, memory_order_acquire) != sense) { cpu_relax(); }
}

/* Per-node cooperation. The input vector of every matmul (normalized + quantized) is produced
 * once per NUMA node by all threads of that node, each doing a slice, into a node-local buffer.
 * Every cache line of the shared fp32 activations crosses the socket link once per node instead
 * of once per thread, and nobody repeats the whole rmsnorm/quantization. */
typedef struct {
    int8_t* q;     // quantized input vector (max(dim, hidden_dim))
    float* s;      // its scales
    float* ss;     // per-thread partial sums of squares (one per cache line)
    struct BarGroup* grp;
    _Alignas(64) atomic_int top;
    _Alignas(64) atomic_int sense;
} NodeBuf;
static NodeBuf NB[MAX_NODES];
static int P_max_layers = 0; // most layers of any loaded model (work counter reset range)
#define SPEC_MAX_TOK 16 // tokens per multi-token decode step (speculative verification)
static void quantize_vec(int8_t* q, float* s, const float* x, int n);

static void node_barrier(ThreadBuf* tb) {
    if (tb->nthn == 1) return;
    NodeBuf* nb = &NB[tb->node];
    const int g = tb->ltid / BAR_GROUP, ngroups = (tb->nthn + BAR_GROUP - 1) / BAR_GROUP;
    const int gsize = (g == ngroups - 1) ? tb->nthn - g * BAR_GROUP : BAR_GROUP;
    int sense = (tb->lsense_node ^= 1);
    if (atomic_fetch_add_explicit(&nb->grp[g].c, 1, memory_order_acq_rel) == gsize - 1) {
        atomic_store_explicit(&nb->grp[g].c, 0, memory_order_relaxed);
        if (atomic_fetch_add_explicit(&nb->top, 1, memory_order_acq_rel) == ngroups - 1) {
            atomic_store_explicit(&nb->top, 0, memory_order_relaxed);
            atomic_store_explicit(&nb->sense, sense, memory_order_release);
            return;
        }
    }
    while (atomic_load_explicit(&nb->sense, memory_order_acquire) != sense) { cpu_relax(); }
}

// groups [g0, g1) of an n-vector handled by this thread inside its node
static inline void node_groups_slice(int n, const ThreadBuf* tb, int* g0, int* g1) {
    int G = n / GS;
    *g0 = (int)((long)G * tb->ltid / tb->nthn);
    *g1 = (int)((long)G * (tb->ltid + 1) / tb->nthn);
}

// node buffer <- quantize(rmsnorm(x) * weight), cooperatively
static void node_norm_quant(ThreadBuf* tb, const float* x, const float* weight, int n) {
    NodeBuf* nb = &NB[tb->node];
    int g0, g1;
    node_groups_slice(n, tb, &g0, &g1);
    float ss = 0.0f;
    for (int i = g0 * GS; i < g1 * GS; i++) ss += x[i] * x[i];
    nb->ss[tb->ltid * 16] = ss;
    node_barrier(tb);
    float tot = 0.0f;
    for (int t = 0; t < tb->nthn; t++) tot += nb->ss[t * 16];
    const float inv = 1.0f / sqrtf(tot / n + NORM_EPS);
    float tmp[256];
    for (int g = g0; g < g1; g++) {
        for (int i = 0; i < GS; i++) tmp[i] = weight[g * GS + i] * (inv * x[g * GS + i]);
        quantize_vec(nb->q + (size_t)g * GS, nb->s + g, tmp, GS);
    }
    node_barrier(tb);
}

// node buffer <- quantize(v), cooperatively
static void node_quant(ThreadBuf* tb, const float* v, int n) {
    NodeBuf* nb = &NB[tb->node];
    int g0, g1;
    node_groups_slice(n, tb, &g0, &g1);
    if (g1 > g0) quantize_vec(nb->q + (size_t)g0 * GS, nb->s + g0, v + (size_t)g0 * GS, (g1 - g0) * GS);
    node_barrier(tb);
}

// multi-token versions: B vectors of length n at X + b*n -> node buffer (token b at q + b*n, s + b*n/GS)
static void node_norm_quant_multi(ThreadBuf* tb, const float* X, const float* weight, int n, int B) {
    NodeBuf* nb = &NB[tb->node];
    int g0, g1;
    node_groups_slice(n, tb, &g0, &g1);
    for (int b = 0; b < B; b++) {
        const float* x = X + (size_t)b * n;
        float ss = 0.0f;
        for (int i = g0 * GS; i < g1 * GS; i++) ss += x[i] * x[i];
        nb->ss[tb->ltid * 16 + b] = ss;
    }
    node_barrier(tb);
    float tmp[256];
    for (int b = 0; b < B; b++) {
        float tot = 0.0f;
        for (int t = 0; t < tb->nthn; t++) tot += nb->ss[t * 16 + b];
        const float inv = 1.0f / sqrtf(tot / n + NORM_EPS);
        const float* x = X + (size_t)b * n;
        for (int g = g0; g < g1; g++) {
            for (int i = 0; i < GS; i++) tmp[i] = weight[g * GS + i] * (inv * x[g * GS + i]);
            quantize_vec(nb->q + (size_t)b * n + (size_t)g * GS, nb->s + (size_t)b * (n / GS) + g, tmp, GS);
        }
    }
    node_barrier(tb);
}
static void node_quant_multi(ThreadBuf* tb, const float* V, int n, int B) {
    NodeBuf* nb = &NB[tb->node];
    int g0, g1;
    node_groups_slice(n, tb, &g0, &g1);
    if (g1 > g0)
        for (int b = 0; b < B; b++)
            quantize_vec(nb->q + (size_t)b * n + (size_t)g0 * GS, nb->s + (size_t)b * (n / GS) + g0, V + (size_t)b * n + (size_t)g0 * GS, (g1 - g0) * GS);
    node_barrier(tb);
}

// split the rows of the thread's NUMA node among the threads of that node, in blocks of 4 rows
static inline void part_range_node(int d, const ThreadBuf* tb, int* r0, int* r1) {
    int lo, hi;
    node_rows(d, tb->node, &lo, &hi);
    int blocks = (hi - lo) / 4;
    *r0 = lo + 4 * (int)((long)blocks * tb->ltid / tb->nthn);
    *r1 = (tb->ltid == tb->nthn - 1) ? hi : lo + 4 * (int)((long)blocks * (tb->ltid + 1) / tb->nthn);
}
// split d rows among threads in blocks of 4 rows
static inline void part_range(int d, int tid, int nth, int* r0, int* r1) {
    int blocks = d / 4;
    *r0 = 4 * (int)((long)blocks * tid / nth);
    *r1 = (tid == nth - 1) ? d : 4 * (int)((long)blocks * (tid + 1) / nth);
}

static void quantize_vec(int8_t* q, float* s, const float* x, int n) {
    int num_groups = n / GS;
    for (int g = 0; g < num_groups; g++) {
        const float* xg = x + g * GS;
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) { float a = fabsf(xg[i]); wmax = a > wmax ? a : wmax; }
        float scale = wmax / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        s[g] = scale;
        for (int i = 0; i < GS; i++) { q[g * GS + i] = (int8_t) roundf(xg[i] * inv); }
    }
}

// out[i] = W[i,:] . x  for rows i in [r0, r1); x is quantized (xq, xs), n % GS == 0
static void matmul_rows_q8(float* out, const int8_t* xq, const float* xs,
                        const QuantizedTensor* w, int n, int r0, int r1) {
    const int ng = n / GS;
    int i = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (GS % 16 == 0) {
        for (; i + 4 <= r1; i += 4) {
            const int8_t* w0 = w->q + (size_t)i * n;
            const int8_t* w1 = w0 + n;
            const int8_t* w2 = w1 + n;
            const int8_t* w3 = w2 + n;
            const float* s0 = w->s + (size_t)i * ng;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int g = 0; g < ng; g++) {
                int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
                const int off = g * GS;
                for (int k = 0; k < GS; k += 16) {
                    int8x16_t xv = vld1q_s8(xq + off + k);
                    a0 = vdotq_s32(a0, vld1q_s8(w0 + off + k), xv);
                    a1 = vdotq_s32(a1, vld1q_s8(w1 + off + k), xv);
                    a2 = vdotq_s32(a2, vld1q_s8(w2 + off + k), xv);
                    a3 = vdotq_s32(a3, vld1q_s8(w3 + off + k), xv);
                }
                int32x4_t r = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
                float32x4_t ws = { s0[g], s0[g + ng], s0[g + 2 * ng], s0[g + 3 * ng] };
                acc = vfmaq_f32(acc, vcvtq_f32_s32(r), vmulq_n_f32(ws, xs[g]));
            }
            vst1q_f32(out + i, acc);
        }
    }
#endif
    for (; i < r1; i++) {
        const int8_t* wr = w->q + (size_t)i * n;
        const float* sr = w->s + (size_t)i * ng;
        float val = 0.0f;
        for (int g = 0; g < ng; g++) {
            int32_t ival = 0;
            for (int k = 0; k < GS; k++) { ival += (int32_t)xq[g * GS + k] * (int32_t)wr[g * GS + k]; }
            val += (float)ival * sr[g] * xs[g];
        }
        out[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Q4 weights: groups of 32 values in 16 bytes (low nibble = value k, high nibble = value k+16,
// stored as q+8) with one fp32 scale per group. Activations stay int8 with scales per GS.

#if defined(__ARM_FEATURE_DOTPROD)
// unpack 16 bytes into the two signed int8x16 halves of a 32-value group
static inline void q4_unpack(const uint8_t* p, int8x16_t* lo, int8x16_t* hi) {
    const uint8x16_t v = vld1q_u8(p);
    *lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
    *hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), vdupq_n_s8(8));
}
#endif

static void matmul_rows_q4(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    const int nb = n / 32, bpr = n / 2, xsh = GS / 32;
    const uint8_t* wq = (const uint8_t*)w->q;
    int i = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    for (; i + 4 <= r1; i += 4) {
        const uint8_t* w0 = wq + (size_t)i * bpr;
        const float* s0 = w->s + (size_t)i * nb;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            const int8x16_t x0 = vld1q_s8(xq + b * 32), x1 = vld1q_s8(xq + b * 32 + 16);
            int32x4_t a[4];
            for (int r = 0; r < 4; r++) {
                int8x16_t lo, hi;
                q4_unpack(w0 + (size_t)r * bpr + b * 16, &lo, &hi);
                a[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, x0), hi, x1);
            }
            int32x4_t rr = vpaddq_s32(vpaddq_s32(a[0], a[1]), vpaddq_s32(a[2], a[3]));
            float32x4_t ws = {s0[b], s0[b + nb], s0[b + 2 * nb], s0[b + 3 * nb]};
            acc = vfmaq_f32(acc, vcvtq_f32_s32(rr), vmulq_n_f32(ws, xs[b / xsh]));
        }
        vst1q_f32(out + i, acc);
    }
#endif
    for (; i < r1; i++) {
        const uint8_t* wr = wq + (size_t)i * bpr;
        const float* sr = w->s + (size_t)i * nb;
        float val = 0.0f;
        for (int b = 0; b < nb; b++) {
            int32_t iv = 0;
            for (int k = 0; k < 16; k++) {
                iv += ((wr[b * 16 + k] & 15) - 8) * xq[b * 32 + k] + ((wr[b * 16 + k] >> 4) - 8) * xq[b * 32 + 16 + k];
            }
            val += iv * sr[b] * xs[b / xsh];
        }
        out[i] = val;
    }
}

/* batched kernels: 4 rows x 4 tokens. Every 16-byte weight load (and, for Q4, every unpack) is
 * reused for 4 tokens; partial sums are reduced per group. */
static void matmul_rows_batch_q4(float* out, int ldo, const int8_t* xq, const float* xs, int B,
                                 const QuantizedTensor* w, int n, int r0, int r1) {
    const int nb = n / 32, bpr = n / 2, xsh = GS / 32, ngx = n / GS;
    const uint8_t* wq = (const uint8_t*)w->q;
    int i = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    for (; i + 4 <= r1; i += 4) {
        const uint8_t* w0 = wq + (size_t)i * bpr;
        const float* s0 = w->s + (size_t)i * nb;
        int t = 0;
        for (; t + 4 <= B; t += 4) {
            float32x4_t acc[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            for (int b = 0; b < nb; b++) {
                int8x16_t lo[4], hi[4];
                for (int r = 0; r < 4; r++) q4_unpack(w0 + (size_t)r * bpr + b * 16, &lo[r], &hi[r]);
                const float32x4_t ws = {s0[b], s0[b + nb], s0[b + 2 * nb], s0[b + 3 * nb]};
                for (int u = 0; u < 4; u++) {
                    const int8_t* x = xq + (size_t)(t + u) * n + b * 32;
                    const int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16);
                    int32x4_t a0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[0], x0), hi[0], x1);
                    int32x4_t a1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[1], x0), hi[1], x1);
                    int32x4_t a2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[2], x0), hi[2], x1);
                    int32x4_t a3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[3], x0), hi[3], x1);
                    int32x4_t rr = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
                    acc[u] = vfmaq_f32(acc[u], vcvtq_f32_s32(rr), vmulq_n_f32(ws, xs[(size_t)(t + u) * ngx + b / xsh]));
                }
            }
            for (int u = 0; u < 4; u++) vst1q_f32(out + (size_t)(t + u) * ldo + i, acc[u]);
        }
        for (; t < B; t++) matmul_rows_q4(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ngx, w, n, i, i + 4);
    }
#endif
    if (i < r1)
        for (int t = 0; t < B; t++) matmul_rows_q4(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ngx, w, n, i, r1);
}

static void matmul_rows_batch_q8(float* out, int ldo, const int8_t* xq, const float* xs, int B,
                                 const QuantizedTensor* w, int n, int r0, int r1) {
    const int ng = n / GS;
    int i = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (GS % 16 == 0) {
        for (; i + 4 <= r1; i += 4) {
            const int8_t* w0 = w->q + (size_t)i * n;
            const float* s0 = w->s + (size_t)i * ng;
            int t = 0;
            for (; t + 4 <= B; t += 4) {
                float32x4_t acc[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
                for (int g = 0; g < ng; g++) {
                    int32x4_t a[4][4];
                    for (int u = 0; u < 4; u++)
                        for (int r = 0; r < 4; r++) a[u][r] = vdupq_n_s32(0);
                    const int off = g * GS;
                    for (int k = 0; k < GS; k += 16) {
                        int8x16_t wv[4];
                        for (int r = 0; r < 4; r++) wv[r] = vld1q_s8(w0 + (size_t)r * n + off + k);
                        for (int u = 0; u < 4; u++) {
                            const int8x16_t xv = vld1q_s8(xq + (size_t)(t + u) * n + off + k);
                            for (int r = 0; r < 4; r++) a[u][r] = vdotq_s32(a[u][r], wv[r], xv);
                        }
                    }
                    const float32x4_t ws = {s0[g], s0[g + ng], s0[g + 2 * ng], s0[g + 3 * ng]};
                    for (int u = 0; u < 4; u++) {
                        int32x4_t rr = vpaddq_s32(vpaddq_s32(a[u][0], a[u][1]), vpaddq_s32(a[u][2], a[u][3]));
                        acc[u] = vfmaq_f32(acc[u], vcvtq_f32_s32(rr), vmulq_n_f32(ws, xs[(size_t)(t + u) * ng + g]));
                    }
                }
                for (int u = 0; u < 4; u++) vst1q_f32(out + (size_t)(t + u) * ldo + i, acc[u]);
            }
            for (; t < B; t++) matmul_rows_q8(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ng, w, n, i, i + 4);
        }
    }
#endif
    if (i < r1)
        for (int t = 0; t < B; t++) matmul_rows_q8(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ng, w, n, i, r1);
}

static void matmul_rows(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    if (w->bits == 4) matmul_rows_q4(out, xq, xs, w, n, r0, r1);
    else matmul_rows_q8(out, xq, xs, w, n, r0, r1);
}

static void forward_thread(int tid) {
    Transformer* transformer = P.t;
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    ThreadBuf* tb = &P.tb[tid];
    const int pos = P.pos;
    float* x = s->x;
    const int dim = p->dim;
    const int kv_dim = p->kv_dim;
    const int q_dim = p->q_dim;
    const int qkn = p->qk_norm;
    const int kv_mul = p->n_heads / p->n_kv_heads;
    const int hidden_dim = p->hidden_dim;
    const int head_size = p->head_dim;
    const int n_heads = p->n_heads;
    // split the context into chunks of at least 64 positions (enough parallelism, cheap merge)
    const int C = (pos + 1) / 64 < 1 ? 1 : ((pos + 1) / 64 < P.C ? (pos + 1) / 64 : P.C);
    const int T = pos + 1;
    const float att_scale = 1.0f / sqrtf((float)head_size);
    const float* emb = w->token_embedding_table + (size_t)P.token * dim;
    int r0, r1;
    const int8_t* nq = NB[tb->node].q; // the node's quantized matmul input
    const float* ns = NB[tb->node].s;

    if (tb->ltid == 0) { // node-local reset of the counter set of the next token (unused now)
        struct WorkCounter* wc = P.work[tb->node][(P.seq + 1) & 1];
        for (int k = 0; k < WORK_PHASES * P_max_layers + 1; k++) atomic_store_explicit(&wc[k].v, 0, memory_order_relaxed); // all models
    }
    PROF(tid, PH_ROPE);
    // RoPE rotation for this position (same for every layer)
    for (int j = 0; j < head_size / 2; j++) {
        double val = pos * P.rope_freq[j];
        tb->rope_c[j] = (float)cos(val);
        tb->rope_s[j] = (float)sin(val);
    }

    for (int l = 0; l < p->n_layers; l++) {
        const float* xin = (l == 0) ? emb : x; // residual stream input of this layer
        size_t loff = (size_t)l * p->seq_len * kv_dim;
        // kv cache layout: (layer, kv_head, seq_len, head_size), so every head is one contiguous stream
        float* kc = s->key_cache + loff + (size_t)pos * head_size;
        float* vc = s->value_cache + loff + (size_t)pos * head_size;

        PROF(tid, PH_NORM);
        // attention rmsnorm + quantize (cooperative, per node)
        node_norm_quant(tb, xin, w->rms_att_weight + l * dim, dim);

        PROF(tid, PH_QKV);
        // fused q|k|v matmul over rows [0, dim + 2*kv_dim), k and v go straight to the kv cache
        while (grab_rows(WORK_PHASES * l + 0, q_dim, tb, &r0, &r1)) {
            matmul_rows(s->q, nq, ns, w->wq + l, dim, r0, r1);
            if (w->bq) for (int i = r0; i < r1; i++) s->q[i] += w->bq[(size_t)l * q_dim + i];
            if (!qkn) for (int i = r0; i < r1; i += 2) {
                int j = (i % head_size) / 2;
                float v0 = s->q[i], v1 = s->q[i + 1];
                s->q[i]     = v0 * tb->rope_c[j] - v1 * tb->rope_s[j];
                s->q[i + 1] = v0 * tb->rope_s[j] + v1 * tb->rope_c[j];
            }
        }
        while (grab_rows(WORK_PHASES * l + 1, kv_dim, tb, &r0, &r1)) {
            matmul_rows(s->k, nq, ns, w->wk + l, dim, r0, r1);
            if (w->bk) for (int i = r0; i < r1; i++) s->k[i] += w->bk[(size_t)l * kv_dim + i];
            if (!qkn) for (int i = r0; i < r1; i += 2) {
                int j = (i % head_size) / 2;
                float v0 = s->k[i], v1 = s->k[i + 1];
                s->k[i]     = v0 * tb->rope_c[j] - v1 * tb->rope_s[j];
                s->k[i + 1] = v0 * tb->rope_s[j] + v1 * tb->rope_c[j];
            }
            if (!qkn) for (int i = r0; i < r1; i++) { kc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = s->k[i]; }
        }
        while (grab_rows(WORK_PHASES * l + 2, kv_dim, tb, &r0, &r1)) {
            matmul_rows(s->v, nq, ns, w->wv + l, dim, r0, r1);
            if (w->bv) for (int i = r0; i < r1; i++) s->v[i] += w->bv[(size_t)l * kv_dim + i];
            for (int i = r0; i < r1; i++) { vc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = s->v[i]; }
        }
        barrier(tb);
        if (qkn) { // per-head q/k RMSNorm + RoPE (needs whole heads), k into the cache
            for (int task = tid; task < n_heads + p->n_kv_heads; task += P.nth) {
                if (task < n_heads) {
                    head_norm_rope(s->q + (size_t)task * head_size, w->q_norm + (size_t)l * head_size, head_size, tb->rope_c, tb->rope_s);
                } else {
                    int g = task - n_heads;
                    float* kv = s->k + (size_t)g * head_size;
                    head_norm_rope(kv, w->k_norm + (size_t)l * head_size, head_size, tb->rope_c, tb->rope_s);
                    memcpy(kc + (size_t)g * p->seq_len * head_size, kv, head_size * sizeof(float));
                }
            }
            barrier(tb);
        }

        PROF(tid, PH_ATT);
        // attention: work items are (kv head group, chunk of the context); the kv_mul query heads
        // sharing a kv head are processed together so K and V are read once. Merged with online softmax.
        int g0, g1;
        node_groups(p->n_kv_heads, tb->node, &g0, &g1);
        // few kv heads (GQA): also split the query heads of a group into HS subsets so that
        // every thread of the node gets an item; K/V are then read once per subset
        int HS = 1;
        while (HS < kv_mul && (g1 - g0) * C * HS * 2 <= tb->nthn && kv_mul % (HS * 2) == 0) HS *= 2;
        const int hper = kv_mul / HS;
        for (int item = tb->ltid; item < (g1 - g0) * C * HS; item += tb->nthn) {
            int g = g0 + item / (C * HS), c = (item / HS) % C, hsub = item % HS;
            int t0 = (int)((long)T * c / C), t1 = (int)((long)T * (c + 1) / C);
            const float* kb = s->key_cache + loff + (size_t)g * p->seq_len * head_size;
            const float* vb = s->value_cache + loff + (size_t)g * p->seq_len * head_size;
            float m[KV_MUL_MAX], sum[KV_MUL_MAX];
            for (int r = 0; r < hper; r++) m[r] = -FLT_MAX;
            for (int t = t0; t < t1; t++) {
                const float* k = kb + (size_t)t * head_size;
                for (int r = 0; r < hper; r++) {
                    int h = g * kv_mul + hsub * hper + r;
                    const float* q = s->q + h * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) { score += q[i] * k[i]; }
                    score *= att_scale;
                    s->att[(size_t)h * p->seq_len + t] = score;
                    m[r] = score > m[r] ? score : m[r];
                }
            }
            for (int r = 0; r < hper; r++) {
                int h = g * kv_mul + hsub * hper + r;
                float* att = s->att + (size_t)h * p->seq_len;
                float sm = 0.0f;
                for (int t = t0; t < t1; t++) { float e = expf(att[t] - m[r]); att[t] = e; sm += e; }
                sum[r] = sm;
                memset(P.part + ((size_t)h * C + c) * (head_size + 2) + 2, 0, head_size * sizeof(float));
            }
            for (int t = t0; t < t1; t++) {
                const float* v = vb + (size_t)t * head_size;
                for (int r = 0; r < hper; r++) {
                    int h = g * kv_mul + hsub * hper + r;
                    float e = s->att[(size_t)h * p->seq_len + t];
                    float* acc = P.part + ((size_t)h * C + c) * (head_size + 2) + 2;
                    for (int i = 0; i < head_size; i++) { acc[i] += e * v[i]; }
                }
            }
            for (int r = 0; r < hper; r++) {
                float* pr = P.part + ((size_t)(g * kv_mul + hsub * hper + r) * C + c) * (head_size + 2);
                pr[0] = m[r];
                pr[1] = sum[r];
            }
        }
        barrier(tb);

        PROF(tid, PH_MERGE);
        // merge the partials of this thread's slice of heads and quantize them into the node buffer
        // (head_size is a multiple of GS, so a head is a whole number of quantization groups)
        int h0 = (int)((long)n_heads * tb->ltid / tb->nthn), h1 = (int)((long)n_heads * (tb->ltid + 1) / tb->nthn);
        for (int h = h0; h < h1; h++) {
            float* xb = tb->xb + h * head_size;
            const float* ph = P.part + (size_t)h * C * (head_size + 2);
            float M = -FLT_MAX;
            for (int c = 0; c < C; c++) {
                const float* pr = ph + (size_t)c * (head_size + 2);
                if (pr[1] > 0.0f && pr[0] > M) M = pr[0];
            }
            float den = 0.0f;
            memset(xb, 0, head_size * sizeof(float));
            for (int c = 0; c < C; c++) {
                const float* pr = ph + (size_t)c * (head_size + 2);
                if (pr[1] <= 0.0f) continue;
                float f = expf(pr[0] - M);
                den += f * pr[1];
                for (int i = 0; i < head_size; i++) { xb[i] += f * pr[2 + i]; }
            }
            float inv = 1.0f / den;
            for (int i = 0; i < head_size; i++) { xb[i] *= inv; }
            quantize_vec(NB[tb->node].q + (size_t)h * head_size, NB[tb->node].s + h * head_size / GS, xb, head_size);
        }
        node_barrier(tb);

        PROF(tid, PH_WO);
        // output projection + residual
        while (grab_rows(WORK_PHASES * l + 3, dim, tb, &r0, &r1)) {
            matmul_rows(s->xb2, nq, ns, w->wo + l, q_dim, r0, r1);
            for (int i = r0; i < r1; i++) { x[i] = xin[i] + s->xb2[i]; }
        }
        barrier(tb);

        PROF(tid, PH_FNORM);
        // ffn rmsnorm + quantize (cooperative, per node)
        node_norm_quant(tb, x, w->rms_ffn_weight + l * dim, dim);

        PROF(tid, PH_W13);
        if (p->n_experts > 0) { // ---- MoE FFN
            const int E = p->n_experts, A = p->n_active, mh = p->moe_hidden;
            while (grab_rows(WORK_PHASES * l + 6, E, tb, &r0, &r1)) matmul_rows(s->router, nq, ns, w->wg + l, dim, r0, r1);
            barrier(tb);
            int sel[MOE_MAX_ACTIVE];
            float wt[MOE_MAX_ACTIVE];
            moe_route(s->router, E, A, p->norm_topk, sel, wt); // same result in every thread
            int lo, hi;
            node_rows(mh, tb->node, &lo, &hi);
            const int half = hi - lo;
            while (grab_virtual(WORK_PHASES * l + 4, A * half, tb, &r0, &r1)) {
                for (int v = r0; v < r1;) {
                    int j = v / half, vend = (j + 1) * half < r1 ? (j + 1) * half : r1;
                    int a0 = lo + (v - j * half), a1 = lo + (vend - j * half);
                    QuantizedTensor e1 = expert_view(w->w1 + l, sel[j], mh, dim), e3 = expert_view(w->w3 + l, sel[j], mh, dim);
                    float* h1 = s->hb + (size_t)j * mh;
                    float* h3 = s->hb2 + (size_t)j * mh;
                    matmul_rows(h1, nq, ns, &e1, dim, a0, a1);
                    matmul_rows(h3, nq, ns, &e3, dim, a0, a1);
                    for (int i = a0; i < a1; i++) {
                        float val = h1[i];
                        val *= (1.0f / (1.0f + expf(-val)));
                        h1[i] = val * h3[i] * wt[j];
                    }
                    v = vend;
                }
            }
            barrier(tb);
            PROF(tid, PH_QH);
            node_quant(tb, s->hb, A * mh);
            PROF(tid, PH_W2);
            while (grab_rows(WORK_PHASES * l + 5, dim, tb, &r0, &r1)) {
                for (int j = 0; j < A; j++) {
                    QuantizedTensor e2 = expert_view(w->w2 + l, sel[j], dim, mh);
                    matmul_rows(tb->xb, nq + (size_t)j * mh, ns + (size_t)j * mh / GS, &e2, mh, r0, r1);
                    for (int i = r0; i < r1; i++) x[i] += tb->xb[i];
                }
            }
        } else { // ---- dense FFN
            // fused w1|w3 + SwiGLU
            while (grab_rows(WORK_PHASES * l + 4, hidden_dim, tb, &r0, &r1)) {
                matmul_rows(s->hb, nq, ns, w->w1 + l, dim, r0, r1);
                matmul_rows(s->hb2, nq, ns, w->w3 + l, dim, r0, r1);
                for (int i = r0; i < r1; i++) {
                    float val = s->hb[i];
                    val *= (1.0f / (1.0f + expf(-val)));
                    s->hb[i] = val * s->hb2[i];
                }
            }
            barrier(tb);

            PROF(tid, PH_QH);
            // w2 + residual
            node_quant(tb, s->hb, hidden_dim);
            PROF(tid, PH_W2);
            while (grab_rows(WORK_PHASES * l + 5, dim, tb, &r0, &r1)) {
                matmul_rows(s->xb2, nq, ns, w->w2 + l, hidden_dim, r0, r1);
                for (int i = r0; i < r1; i++) { x[i] += s->xb2[i]; }
            }
        }
        barrier(tb);
    }

PROF(tid, PH_CLS);
    // final rmsnorm (private) and classifier
    node_norm_quant(tb, x, w->rms_final_weight, dim);
    while (grab_rows(WORK_PHASES * p->n_layers, p->vocab_size, tb, &r0, &r1)) {
        matmul_rows(s->logits, nq, ns, w->wcls, dim, r0, r1);
    }
    barrier(tb);
    PROF(tid, PH_IDLE);
}

/* Thread placement. On a shared machine the scheduler initially stacks spin-waiting threads on
 * busy cores and needs seconds to spread them, which stalls every barrier. So at start-up we
 * pick the most idle CPUs (sampled from /proc/stat) among those allowed by taskset/numactl and
 * move thread i onto the i-th of them.
 *   RUNQ_PIN unset: place, then release the pin to the CPUs of that NUMA node
 *   RUNQ_PIN=1:     keep threads pinned
 *   RUNQ_PIN=0:     leave placement entirely to the scheduler */
static int* P_cpus = NULL; // chosen CPUs, one per thread

#if defined(__linux__)
static int read_cpu_idle(long long* idle, int max) {
    FILE* f = fopen("/proc/stat", "r");
    if (!f) return 0;
    char line[512];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        int cpu;
        long long v[8] = {0};
        if (sscanf(line, "cpu%d %lld %lld %lld %lld %lld %lld %lld %lld", &cpu, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) >= 5 &&
            cpu >= 0 && cpu < max) {
            idle[cpu] = v[3] + v[4]; // idle + iowait
            if (cpu + 1 > n) n = cpu + 1;
        }
    }
    fclose(f);
    return n;
}

static int NODES_REQ = 0; // -N: NUMA nodes to use, 0 = all usable
#if defined(__linux__)
static cpu_set_t P_node_mask[MAX_NODES]; // released threads may move only inside their node
#endif
/* Choose CPUs on one or more NUMA nodes (most idle first). nth <= 0: automatic thread count
 * (idle CPUs of each node minus a few, at most 96 per node). Sets NN, NODE_ID, NODE_T0, P_cpus.
 * Returns the number of threads. */
static int choose_cpus(int nth) {
    const char* mode = getenv("RUNQ_PIN");
    if (mode && !strcmp(mode, "0")) { NN = 1; return nth > 0 ? nth : 1; }
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return nth > 0 ? nth : 1;
    static long long i0[CPU_SETSIZE], i1[CPU_SETSIZE];
    read_cpu_idle(i0, CPU_SETSIZE);
    struct timespec ts = {0, 100 * 1000 * 1000};
    nanosleep(&ts, NULL);
    int n = read_cpu_idle(i1, CPU_SETSIZE);
    static int node_of[CPU_SETSIZE];
    for (int c = 0; c < CPU_SETSIZE; c++) node_of[c] = 0;
    int n_nodes = 1;
    for (int node = 0; node < 64; node++) {
        char path[96];
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        char list[4096];
        if (fgets(list, sizeof list, f)) {
            for (char* tok = strtok(list, ",\n"); tok; tok = strtok(NULL, ",\n")) {
                int lo, hi;
                if (sscanf(tok, "%d-%d", &lo, &hi) == 2) { for (int c = lo; c <= hi && c < CPU_SETSIZE; c++) node_of[c] = node; }
                else if (sscanf(tok, "%d", &lo) == 1 && lo < CPU_SETSIZE) node_of[lo] = node;
            }
        }
        fclose(f);
        if (node + 1 > n_nodes) n_nodes = node + 1;
    }
    if (n_nodes > 64) n_nodes = 64;
    long long period = 1;
    for (int c = 0; c < n; c++) if (i1[c] - i0[c] > period) period = i1[c] - i0[c]; // ticks of a fully idle CPU
    int idle_cnt[64] = {0};
    for (int c = 0; c < n; c++)
        if (CPU_ISSET(c, &allowed) && (i1[c] - i0[c]) * 10 >= period * 9) idle_cnt[node_of[c]]++;
    // nodes by idle CPUs, most idle first
    int order[64];
    for (int k = 0; k < n_nodes; k++) order[k] = k;
    for (int i = 1; i < n_nodes; i++) {
        int v = order[i], j = i - 1;
        while (j >= 0 && idle_cnt[order[j]] < idle_cnt[v]) { order[j + 1] = order[j]; j--; }
        order[j + 1] = v;
    }
    int want = NODES_REQ > 0 ? NODES_REQ : n_nodes;
    if (want > MAX_NODES) want = MAX_NODES;
    if (nth == 1) want = 1;
    NN = 0;
    for (int k = 0; k < n_nodes && NN < want; k++)
        if (idle_cnt[order[k]] >= (NODES_REQ > 0 ? 1 : 16) || NN == 0) NODE_ID[NN++] = order[k];
    int threads[MAX_NODES], total = 0;
    for (int k = 0; k < NN; k++) {
        if (nth <= 0) {
            threads[k] = idle_cnt[NODE_ID[k]] - 4;
            if (threads[k] > 96) threads[k] = 96;
            if (threads[k] < 1) threads[k] = 1;
        } else {
            threads[k] = nth / NN + (k < nth % NN);
        }
        if (threads[k] > idle_cnt[NODE_ID[k]])
            fprintf(stderr, "[warning: only %d idle CPUs on NUMA node %d for %d threads; expect slowdowns]\n", idle_cnt[NODE_ID[k]], NODE_ID[k], threads[k]);
        NODE_T0[k] = total;
        total += threads[k];
    }
    NODE_T0[NN] = total;
    P_cpus = malloc(total * sizeof(int));
    for (int k = 0; k < NN; k++) {
        int cand[CPU_SETSIZE], nc = 0;
        for (int c = 0; c < n; c++)
            if (CPU_ISSET(c, &allowed) && node_of[c] == NODE_ID[k]) cand[nc++] = c;
        // most idle first; ties keep CPU order so threads land on neighbouring cores
        for (int i = 1; i < nc; i++) {
            int c = cand[i], j = i - 1;
            while (j >= 0 && (i1[cand[j]] - i0[cand[j]]) < (i1[c] - i0[c])) { cand[j + 1] = cand[j]; j--; }
            cand[j + 1] = c;
        }
        CPU_ZERO(&P_node_mask[k]);
        for (int i = 0; i < nc; i++) CPU_SET(cand[i], &P_node_mask[k]);
        for (int t = NODE_T0[k]; t < NODE_T0[k + 1]; t++) P_cpus[t] = nc ? cand[(t - NODE_T0[k]) % nc] : 0;
    }
    return total;
}
#endif

static void pin_thread(int tid) {
#if defined(__linux__)
    if (!P_cpus) return;
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(P_cpus[tid], &one);
    pthread_setaffinity_np(pthread_self(), sizeof(one), &one); // migrates the thread there now
    const char* mode = getenv("RUNQ_PIN");
    if (!(mode && !strcmp(mode, "1"))) {
        int k = 0;
        while (k < NN - 1 && tid >= NODE_T0[k + 1]) k++;
        pthread_setaffinity_np(pthread_self(), sizeof(P_node_mask[k]), &P_node_mask[k]); // soft: release the pin within the node
    }
#else
    (void)tid;
#endif
}

static void forward_batch_thread(int tid);
static void forward_multi_thread(int tid);
static void init_batch(Transformer* t);

/* Idle workers spin for a short while (tokens of one reply follow each other within ~1 ms) and
 * then sleep on a futex, so an idle chat does not keep a hundred cores busy. */
#include <linux/futex.h>
#include <sys/syscall.h>
#define IDLE_SPIN_NS 2000000LL
static inline long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static int wait_for_work(int seen) {
    int g;
    long long t0 = 0;
    for (unsigned spins = 0;; spins++) {
        if ((g = atomic_load_explicit(&P.go, memory_order_acquire)) != seen) return g;
        cpu_relax();
        if ((spins & 1023) == 1023) {
            long long t = now_ns();
            if (t0 == 0) t0 = t;
            else if (t - t0 > IDLE_SPIN_NS) {
                atomic_fetch_add(&P.sleepers, 1); // seq_cst: ordered against the waker's increment of `go`
                if (atomic_load(&P.go) == seen) syscall(SYS_futex, &P.go, FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
                atomic_fetch_sub(&P.sleepers, 1);
                t0 = 0;
            }
        }
    }
}
static inline void wake_workers(void) {
    atomic_fetch_add(&P.go, 1); // seq_cst
    if (atomic_load(&P.sleepers) > 0) syscall(SYS_futex, &P.go, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL, NULL, 0);
}

static void* pool_worker(void* arg) {
    int tid = (int)(intptr_t)arg;
    pin_thread(tid);
    int seen = 0;
    for (;;) {
        seen = wait_for_work(seen);
        if (atomic_load_explicit(&P.quit, memory_order_acquire)) break;
        if (P.mode == 1) forward_batch_thread(tid);
        else if (P.mode == 2) forward_multi_thread(tid);
        else forward_thread(tid);
    }
    return NULL;
}

/* Model-specific runtime tables (RoPE frequencies, attention partials). Must be called right after
 * build_transformer() of that model (the header globals still hold its settings) and after the
 * pool layout (NN, NODE_T0, P.nth) is known. */
static void init_model_runtime(Transformer* t) {
    Config* p = &t->config;
    if (p->n_layers > P_max_layers) P_max_layers = p->n_layers; // counters are reset for the deepest model
    int head_size = p->head_dim;
    ROPE_THETA = t->g.rope_theta; ROPE_FACTOR = t->g.rope_factor; ROPE_LOW_FREQ = t->g.rope_low;
    ROPE_HIGH_FREQ = t->g.rope_high; ROPE_ORIG_CTX = t->g.rope_orig;
    {   // context chunks per kv head: enough items for the threads of the smallest node
        int min_t = P.nth, min_g = p->n_kv_heads;
        for (int k = 0; k < NN; k++) {
            int g0, g1;
            node_groups(p->n_kv_heads, k, &g0, &g1);
            if (NODE_T0[k + 1] - NODE_T0[k] < min_t) min_t = NODE_T0[k + 1] - NODE_T0[k];
            if (g1 - g0 < min_g) min_g = g1 - g0;
        }
        if (min_g < 1) min_g = 1;
        t->C = min_t / min_g > 1 ? min_t / min_g : 1;
    }
    t->part = calloc((size_t)p->n_heads * t->C * (head_size + 2), sizeof(float));
    t->rope_freq = malloc((head_size / 2) * sizeof(double));
    for (int j = 0; j < head_size / 2; j++) {
        double freq = 1.0 / pow(ROPE_THETA, (2.0 * j) / head_size);
        if (ROPE_FACTOR > 0.0f) { // llama3 rope scaling (Llama 3.1 / 3.2)
            double low_wavelen = ROPE_ORIG_CTX / ROPE_LOW_FREQ;
            double high_wavelen = ROPE_ORIG_CTX / ROPE_HIGH_FREQ;
            double wavelen = 2.0 * M_PI / freq;
            if (wavelen > low_wavelen) {
                freq = freq / ROPE_FACTOR;
            } else if (wavelen >= high_wavelen) {
                double smooth = (ROPE_ORIG_CTX / wavelen - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ);
                freq = (1.0 - smooth) * freq / ROPE_FACTOR + smooth * freq;
            }
        }
        t->rope_freq[j] = freq;
    }
}

// make t the current model: restore its header settings and runtime tables
static Transformer* P_cur = NULL;
static void use_model(Transformer* t) {
    if (P_cur == t) return;
    GS = t->g.gs; ARCH = t->g.arch; QKV_BIAS = t->g.qkv_bias; SPECIAL_START = t->g.special_start;
    BOS_ID = t->g.bos; EOS_ID = t->g.eos; NORM_EPS = t->g.norm_eps;
    ROPE_THETA = t->g.rope_theta; ROPE_FACTOR = t->g.rope_factor; ROPE_LOW_FREQ = t->g.rope_low;
    ROPE_HIGH_FREQ = t->g.rope_high; ROPE_ORIG_CTX = t->g.rope_orig;
    P.rope_freq = t->rope_freq;
    P.part = t->part;
    P.C = t->C;
    P_cur = t;
}

void init_pool(Transformer* t, int nth) {
    Config* p = &t->config;
    int head_size = p->head_dim;
    if (nth < 1) nth = 1;
    if (p->dim % GS != 0 || p->hidden_dim % GS != 0) {
        fprintf(stderr, "group size %d must divide dim %d and hidden_dim %d; re-export the model\n",
                GS, p->dim, p->hidden_dim);
        exit(EXIT_FAILURE);
    }
    P.nth = nth;
    P.t = t;
    if (NN == 1 || NODE_T0[NN] != nth) { NN = 1; NODE_T0[0] = 0; NODE_T0[1] = nth; } // no (matching) placement
    if (p->n_heads / p->n_kv_heads > KV_MUL_MAX) {
        fprintf(stderr, "n_heads / n_kv_heads > %d is not supported\n", KV_MUL_MAX);
        exit(EXIT_FAILURE);
    }
    init_model_runtime(t);
    use_model(t);
    int qmax = p->dim > p->hidden_dim ? p->dim : p->hidden_dim;
    if (p->q_dim > qmax) qmax = p->q_dim;
    if (p->n_experts > 0 && p->n_active * p->moe_hidden > qmax) qmax = p->n_active * p->moe_hidden;
    if (p->n_active > MOE_MAX_ACTIVE) { fprintf(stderr, "at most %d active experts are supported\n", MOE_MAX_ACTIVE); exit(EXIT_FAILURE); }
    P.tb = calloc(nth, sizeof(ThreadBuf));
    for (int i = 0; i < nth; i++) {
        P.tb[i].xb = calloc(p->q_dim > p->dim ? p->q_dim : p->dim, sizeof(float));
        P.tb[i].q = calloc(qmax, sizeof(int8_t));
        P.tb[i].s = calloc(qmax / GS + 1, sizeof(float));
        P.tb[i].rope_c = calloc(head_size / 2, sizeof(float));
        P.tb[i].rope_s = calloc(head_size / 2, sizeof(float));
        int k = 0;
        while (k < NN - 1 && i >= NODE_T0[k + 1]) k++;
        P.tb[i].node = k;
        P.tb[i].ltid = i - NODE_T0[k];
        P.tb[i].nthn = NODE_T0[k + 1] - NODE_T0[k];
    }
    if (head_size % GS != 0) {
        fprintf(stderr, "head_size %d must be a multiple of the group size %d\n", head_size, GS);
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < NN; k++) { // node-local buffers (bound to the node before first touch)
        int nthn = NODE_T0[k + 1] - NODE_T0[k];
        size_t qb = ((size_t)qmax * SPEC_MAX_TOK + 4095) & ~(size_t)4095;
        size_t sb = (((size_t)(qmax / GS + 1) * SPEC_MAX_TOK * sizeof(float)) + 4095) & ~(size_t)4095;
        size_t ssb = (((size_t)nthn * 16 * sizeof(float)) + 4095) & ~(size_t)4095;
        NB[k].q = aligned_alloc(4096, qb);
        NB[k].s = aligned_alloc(4096, sb);
        NB[k].ss = aligned_alloc(4096, ssb);
#if defined(__linux__)
        if (NN > 1) {
            bind_range(NB[k].q, qb, NODE_ID[k]);
            bind_range(NB[k].s, sb, NODE_ID[k]);
            bind_range(NB[k].ss, ssb, NODE_ID[k]);
        }
#endif
        memset(NB[k].q, 0, qb);
        memset(NB[k].s, 0, sb);
        memset(NB[k].ss, 0, ssb);
        NB[k].grp = aligned_alloc(64, (size_t)((nthn + BAR_GROUP - 1) / BAR_GROUP) * sizeof(*NB[k].grp));
        for (int g = 0; g < (nthn + BAR_GROUP - 1) / BAR_GROUP; g++) atomic_store(&NB[k].grp[g].c, 0);
        atomic_store(&NB[k].top, 0);
        atomic_store(&NB[k].sense, 0);
    }
    atomic_store(&P.bar_count, 0);
    atomic_store(&P.bar_sense, 0);
    P.bar_group = aligned_alloc(64, (size_t)((nth + BAR_GROUP - 1) / BAR_GROUP) * sizeof(*P.bar_group));
    for (int g = 0; g < (nth + BAR_GROUP - 1) / BAR_GROUP; g++) atomic_store(&P.bar_group[g].c, 0);
    atomic_store(&P.go, 0);
    atomic_store(&P.sleepers, 0);
    atomic_store(&P.quit, 0);
#if defined(__linux__)
#endif
    pin_thread(0);
    init_batch(t);
    for (int k = 0; k < NN; k++)
        for (int par = 0; par < 2; par++) {
            size_t bytes = ((size_t)(WORK_PHASES * 256 + 1) * sizeof(*P.work[k][par]) + 4095) & ~(size_t)4095; // any model up to 256 layers
            P.work[k][par] = aligned_alloc(4096, bytes);
#if defined(__linux__)
            if (NN > 1) bind_range(P.work[k][par], bytes, NODE_ID[k]);
#endif
            memset(P.work[k][par], 0, bytes);
        }
    P.seq = 0;
    prof_init(nth);
    P.tids = calloc(nth, sizeof(pthread_t));
    for (int i = 1; i < nth; i++) {
        if (pthread_create(&P.tids[i], NULL, pool_worker, (void*)(intptr_t)i) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            exit(EXIT_FAILURE);
        }
    }
}

void free_pool(void) {
    prof_report(P.nth);
    atomic_store_explicit(&P.quit, 1, memory_order_release);
    wake_workers();
    for (int i = 1; i < P.nth; i++) { pthread_join(P.tids[i], NULL); }
    for (int i = 0; i < P.nth; i++) {
        free(P.tb[i].xb); free(P.tb[i].q); free(P.tb[i].s);
        free(P.tb[i].rope_c); free(P.tb[i].rope_s);
    }
    free(P.tb); free(P.tids); free(P.part); free(P.rope_freq);
}

float* forward(Transformer* transformer, int token, int pos) {
    use_model(transformer);
    P.t = transformer;
    P.token = token;
    P.pos = pos;
    P.seq++;
    wake_workers();
    forward_thread(0);
#ifdef PROFILE
    PROF_TOKENS++;
#endif
    return transformer->state.logits;
}

// ----------------------------------------------------------------------------
// Batched prefill: push a block of B prompt tokens through the model at once.
// Every weight row is loaded once per block and reused for all B tokens (matrix x matrix),
// which turns the memory-bound decode loop into a compute-bound one.

#define BATCH_MAX 128

typedef struct {
    int B;            // tokens in the current block
    int pos0;         // position of the first token
    const int* tok;   // the tokens
    float* x;         // (B, dim) residual stream
    float* xb;        // (B, dim) normalized input / attention output
    float* q;         // (B, dim)
    float* k;         // (B, kv_dim)
    float* v;         // (B, kv_dim)
    float* hb;        // (B, hidden_dim)
    float* hb2;       // (B, hidden_dim)
    float* xb2;       // (B, dim)
    int8_t* xq;       // (B, max(dim, hidden_dim)) quantized activations
    float* xs;        // scales for xq
    float* rope_c;    // (B, head_size/2)
    float* rope_s;
    float* att;       // (nth, kv_mul, seq_len) per-thread attention scratch
    float* logits_all; // when set: (B, vocab) logits of every position (speculative verification)
    // MoE prefill (sized for BATCH_MAX tokens x n_active slots)
    float* router;     // (B, n_experts)
    int *moe_sel, *moe_cnt, *moe_fill, *moe_tok;
    float *moe_wt, *moe_w;
    int8_t* gxq; float* gxs; // gathered quantized ffn inputs per slot
    float *gh, *gh3;   // (slots, moe_hidden)
    int8_t* gq; float* gqs;
    float* gout;       // (slots, dim)
} Batch;

static Batch BT;

// out[b*ldo + i] = W[i,:] . x_b for rows i in [r0, r1) and tokens b in [0, B)
static void matmul_rows_batch(float* out, int ldo, const int8_t* xq, const float* xs, int B,
                              const QuantizedTensor* w, int n, int r0, int r1) {
    if (w->bits == 4) matmul_rows_batch_q4(out, ldo, xq, xs, B, w, n, r0, r1);
    else matmul_rows_batch_q8(out, ldo, xq, xs, B, w, n, r0, r1);
}

// split B tokens among threads (for per-token work such as rmsnorm/quantize)
static inline void tok_range(int B, int tid, int nth, int* b0, int* b1) {
    *b0 = (int)((long)B * tid / nth);
    *b1 = (int)((long)B * (tid + 1) / nth);
}

/* MoE FFN for a block of B tokens (prefill). Input: BT.xq/BT.xs (normalized + quantized ffn input).
 * Tokens are grouped by expert so every expert's weights are read once per block:
 * router (batched) -> top-k per token -> per-expert token lists with gathered inputs ->
 * w1/w3 + weighted SwiGLU per expert (this node's half of the expert's rows) -> quantize ->
 * w2 per expert on this node's rows of dim, summed into the residual stream. */
static void moe_batch(ThreadBuf* tb, int l, int B) {
    Transformer* transformer = P.t;
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    const int tid = (int)(tb - P.tb), nth = P.nth;
    const int dim = p->dim, E = p->n_experts, A = p->n_active, mh = p->moe_hidden;
    const int ngd = dim / GS, ngm = mh / GS;
    int r0, r1;
    // router logits for all tokens
    part_range_node(E, tb, &r0, &r1);
    matmul_rows_batch(BT.router, E, BT.xq, BT.xs, B, w->wg + l, dim, r0, r1);
    barrier(tb);
    // routing and grouping (one thread; B*A entries)
    if (tid == 0) {
        int* cnt = BT.moe_cnt;
        memset(cnt, 0, (E + 1) * sizeof(int));
        for (int b = 0; b < B; b++) {
            moe_route(BT.router + (size_t)b * E, E, A, p->norm_topk, BT.moe_sel + b * A, BT.moe_wt + b * A);
            for (int j = 0; j < A; j++) cnt[BT.moe_sel[b * A + j] + 1]++;
        }
        for (int e = 0; e < E; e++) cnt[e + 1] += cnt[e]; // prefix sums: expert e owns slots [cnt[e], cnt[e+1])
        int* fill = BT.moe_fill;
        memcpy(fill, cnt, E * sizeof(int));
        for (int b = 0; b < B; b++)
            for (int j = 0; j < A; j++) {
                int e = BT.moe_sel[b * A + j], slot = fill[e]++;
                BT.moe_tok[slot] = b;
                BT.moe_w[slot] = BT.moe_wt[b * A + j];
                memcpy(BT.gxq + (size_t)slot * dim, BT.xq + (size_t)b * dim, dim);
                memcpy(BT.gxs + (size_t)slot * ngd, BT.xs + (size_t)b * ngd, ngd * sizeof(float));
            }
    }
    barrier(tb);
    const int* off = BT.moe_cnt;
    // w1/w3 + SwiGLU: this node's half of every used expert, split among the node's threads
    int lo, hi;
    node_rows(mh, tb->node, &lo, &hi);
    const int half = hi - lo, tot = E * half;
    int v0 = (int)((long)tot * tb->ltid / tb->nthn), v1 = (int)((long)tot * (tb->ltid + 1) / tb->nthn);
    v0 &= ~3; if (tb->ltid == tb->nthn - 1) v1 = tot; else v1 &= ~3;
    for (int v = v0; v < v1;) {
        int e = v / half, vend = (e + 1) * half < v1 ? (e + 1) * half : v1;
        int n = off[e + 1] - off[e];
        if (n > 0) {
            int a0 = lo + (v - e * half), a1 = lo + (vend - e * half);
            QuantizedTensor e1 = expert_view(w->w1 + l, e, mh, dim), e3 = expert_view(w->w3 + l, e, mh, dim);
            float* h1 = BT.gh + (size_t)off[e] * mh;
            float* h3 = BT.gh3 + (size_t)off[e] * mh;
            matmul_rows_batch(h1, mh, BT.gxq + (size_t)off[e] * dim, BT.gxs + (size_t)off[e] * ngd, n, &e1, dim, a0, a1);
            matmul_rows_batch(h3, mh, BT.gxq + (size_t)off[e] * dim, BT.gxs + (size_t)off[e] * ngd, n, &e3, dim, a0, a1);
            for (int t = 0; t < n; t++) {
                float wgt = BT.moe_w[off[e] + t];
                float* x1 = h1 + (size_t)t * mh;
                const float* x3 = h3 + (size_t)t * mh;
                for (int i = a0; i < a1; i++) {
                    float val = x1[i];
                    val *= (1.0f / (1.0f + expf(-val)));
                    x1[i] = val * x3[i] * wgt;
                }
            }
        }
        v = vend;
    }
    barrier(tb);
    // quantize the hidden activations of every slot
    const int S = off[E];
    int s0, s1;
    tok_range(S, tid, nth, &s0, &s1);
    for (int sl = s0; sl < s1; sl++)
        quantize_vec(BT.gq + (size_t)sl * mh, BT.gqs + (size_t)sl * ngm, BT.gh + (size_t)sl * mh, mh);
    barrier(tb);
    // w2: this node's rows of dim, for every used expert, then add every slot into its token
    part_range_node(dim, tb, &r0, &r1);
    for (int e = 0; e < E; e++) {
        int n = off[e + 1] - off[e];
        if (n == 0 || r0 >= r1) continue;
        QuantizedTensor e2 = expert_view(w->w2 + l, e, dim, mh);
        matmul_rows_batch(BT.gout + (size_t)off[e] * dim, dim, BT.gq + (size_t)off[e] * mh, BT.gqs + (size_t)off[e] * ngm, n, &e2, mh, r0, r1);
    }
    for (int sl = 0; sl < S && r0 < r1; sl++) {
        float* x = BT.x + (size_t)BT.moe_tok[sl] * dim;
        const float* o = BT.gout + (size_t)sl * dim;
        for (int i = r0; i < r1; i++) x[i] += o[i];
    }
}

static void forward_batch_thread(int tid) {
    Transformer* transformer = P.t;
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    ThreadBuf* tb = &P.tb[tid];
    const int nth = P.nth;
    const int B = BT.B, pos0 = BT.pos0;
    const int dim = p->dim;
    const int kv_dim = p->kv_dim;
    const int q_dim = p->q_dim;
    const int qkn = p->qk_norm;
    const int kv_mul = p->n_heads / p->n_kv_heads;
    const int hidden_dim = p->hidden_dim;
    const int head_size = p->head_dim;
    const int hs2 = head_size / 2;
    const int ngd = dim / GS, ngh = hidden_dim / GS, ngq = q_dim / GS;
    const float att_scale = 1.0f / sqrtf((float)head_size);
    int r0, r1, b0, b1;

    // embeddings and RoPE tables for this thread's share of the tokens
    tok_range(B, tid, nth, &b0, &b1);
    for (int b = b0; b < b1; b++) {
        memcpy(BT.x + (size_t)b * dim, w->token_embedding_table + (size_t)BT.tok[b] * dim, dim * sizeof(float));
        for (int j = 0; j < hs2; j++) {
            double val = (double)(pos0 + b) * P.rope_freq[j];
            BT.rope_c[b * hs2 + j] = (float)cos(val);
            BT.rope_s[b * hs2 + j] = (float)sin(val);
        }
    }
    (void)tb;

    for (int l = 0; l < p->n_layers; l++) {
        size_t loff = (size_t)l * p->seq_len * kv_dim;

        // attention rmsnorm + quantize, split over tokens
        for (int b = b0; b < b1; b++) {
            rmsnorm(BT.xb + (size_t)b * dim, BT.x + (size_t)b * dim, w->rms_att_weight + l * dim, dim);
            quantize_vec(BT.xq + (size_t)b * dim, BT.xs + (size_t)b * ngd, BT.xb + (size_t)b * dim, dim);
        }
        barrier(tb);

        // fused q|k|v for all tokens
        {
            int a, e;
            part_range_node(q_dim, tb, &a, &e);
            if (a < e) {
                matmul_rows_batch(BT.q, q_dim, BT.xq, BT.xs, B, w->wq + l, dim, a, e);
                for (int b = 0; b < B; b++) {
                    float* q = BT.q + (size_t)b * q_dim;
                    if (w->bq) for (int i = a; i < e; i++) q[i] += w->bq[(size_t)l * q_dim + i];
                    if (qkn) continue; // per-head norm + RoPE after the barrier
                    const float* rc = BT.rope_c + b * hs2;
                    const float* rs = BT.rope_s + b * hs2;
                    for (int i = a; i < e; i += 2) {
                        int j = (i % head_size) / 2;
                        float v0 = q[i], v1 = q[i + 1];
                        q[i] = v0 * rc[j] - v1 * rs[j];
                        q[i + 1] = v0 * rs[j] + v1 * rc[j];
                    }
                }
            }
            part_range_node(kv_dim, tb, &a, &e);
            if (a < e) {
                matmul_rows_batch(BT.k, kv_dim, BT.xq, BT.xs, B, w->wk + l, dim, a, e);
                for (int b = 0; b < B; b++) {
                    float* k = BT.k + (size_t)b * kv_dim;
                    if (w->bk) for (int i = a; i < e; i++) k[i] += w->bk[(size_t)l * kv_dim + i];
                    if (qkn) continue; // per-head norm + RoPE + cache after the barrier
                    const float* rc = BT.rope_c + b * hs2;
                    const float* rs = BT.rope_s + b * hs2;
                    float* kc = s->key_cache + loff + (size_t)(pos0 + b) * head_size;
                    for (int i = a; i < e; i += 2) {
                        int j = (i % head_size) / 2;
                        float v0 = k[i], v1 = k[i + 1];
                        k[i] = v0 * rc[j] - v1 * rs[j];
                        k[i + 1] = v0 * rs[j] + v1 * rc[j];
                    }
                    for (int i = a; i < e; i++) { kc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = k[i]; }
                }
            }
            part_range_node(kv_dim, tb, &a, &e);
            if (a < e) {
                matmul_rows_batch(BT.v, kv_dim, BT.xq, BT.xs, B, w->wv + l, dim, a, e);
                for (int b = 0; b < B; b++) {
                    float* v = BT.v + (size_t)b * kv_dim;
                    if (w->bv) for (int i = a; i < e; i++) v[i] += w->bv[(size_t)l * kv_dim + i];
                    float* vc = s->value_cache + loff + (size_t)(pos0 + b) * head_size;
                    for (int i = a; i < e; i++) { vc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = v[i]; }
                }
            }
        }
        barrier(tb);
        if (qkn) { // per-head RMSNorm + RoPE of q and k of every token, k into the cache
            const int nh = p->n_heads, nk = p->n_kv_heads;
            for (int task = tid; task < B * (nh + nk); task += nth) {
                int b = task / (nh + nk), h = task % (nh + nk);
                const float* rc = BT.rope_c + b * hs2;
                const float* rs = BT.rope_s + b * hs2;
                if (h < nh) {
                    head_norm_rope(BT.q + (size_t)b * q_dim + (size_t)h * head_size, w->q_norm + (size_t)l * head_size, head_size, rc, rs);
                } else {
                    int g = h - nh;
                    float* kv = BT.k + (size_t)b * kv_dim + (size_t)g * head_size;
                    head_norm_rope(kv, w->k_norm + (size_t)l * head_size, head_size, rc, rs);
                    memcpy(s->key_cache + loff + ((size_t)g * p->seq_len + pos0 + b) * head_size, kv, head_size * sizeof(float));
                }
            }
            barrier(tb);
        }

        // causal attention: work items are (kv group, token); each item handles the kv_mul query
        // heads of its group, so K and V rows are loaded once per item
        float* att = BT.att + (size_t)tid * kv_mul * p->seq_len;
        int g0, g1;
        node_groups(p->n_kv_heads, tb->node, &g0, &g1);
        for (int item = tb->ltid; item < (g1 - g0) * B; item += tb->nthn) {
            int g = g0 + item % (g1 - g0), b = item / (g1 - g0);
            int T = pos0 + b + 1;
            const float* kb = s->key_cache + loff + (size_t)g * p->seq_len * head_size;
            const float* vb = s->value_cache + loff + (size_t)g * p->seq_len * head_size;
            const float* qb = BT.q + (size_t)b * q_dim + (size_t)g * kv_mul * head_size;
            float m[KV_MUL_MAX], sum[KV_MUL_MAX];
            for (int r = 0; r < kv_mul; r++) m[r] = -FLT_MAX;
            for (int t = 0; t < T; t++) {
                const float* k = kb + (size_t)t * head_size;
                for (int r = 0; r < kv_mul; r++) {
                    const float* q = qb + r * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) { score += q[i] * k[i]; }
                    score *= att_scale;
                    att[(size_t)r * p->seq_len + t] = score;
                    m[r] = score > m[r] ? score : m[r];
                }
            }
            for (int r = 0; r < kv_mul; r++) {
                float* ar = att + (size_t)r * p->seq_len;
                float sm = 0.0f;
                for (int t = 0; t < T; t++) { float ev = expf(ar[t] - m[r]); ar[t] = ev; sm += ev; }
                sum[r] = 1.0f / sm;
            }
            float* ob = BT.xb + (size_t)b * q_dim + (size_t)g * kv_mul * head_size;
            memset(ob, 0, kv_mul * head_size * sizeof(float));
            for (int t = 0; t < T; t++) {
                const float* v = vb + (size_t)t * head_size;
                for (int r = 0; r < kv_mul; r++) {
                    float a = att[(size_t)r * p->seq_len + t];
                    float* o = ob + r * head_size;
                    for (int i = 0; i < head_size; i++) { o[i] += a * v[i]; }
                }
            }
            for (int r = 0; r < kv_mul; r++) {
                float* o = ob + r * head_size;
                for (int i = 0; i < head_size; i++) { o[i] *= sum[r]; }
            }
        }
        barrier(tb);

        // quantize the attention output, split over tokens
        for (int b = b0; b < b1; b++) {
            quantize_vec(BT.xq + (size_t)b * q_dim, BT.xs + (size_t)b * ngq, BT.xb + (size_t)b * q_dim, q_dim);
        }
        barrier(tb);

        // output projection + residual
        part_range_node(dim, tb, &r0, &r1);
        matmul_rows_batch(BT.xb2, dim, BT.xq, BT.xs, B, w->wo + l, q_dim, r0, r1);
        for (int b = 0; b < B; b++) {
            float* x = BT.x + (size_t)b * dim;
            const float* o = BT.xb2 + (size_t)b * dim;
            for (int i = r0; i < r1; i++) { x[i] += o[i]; }
        }
        barrier(tb);

        // ffn rmsnorm + quantize, split over tokens
        for (int b = b0; b < b1; b++) {
            rmsnorm(BT.xb + (size_t)b * dim, BT.x + (size_t)b * dim, w->rms_ffn_weight + l * dim, dim);
            quantize_vec(BT.xq + (size_t)b * dim, BT.xs + (size_t)b * ngd, BT.xb + (size_t)b * dim, dim);
        }
        barrier(tb);

        if (p->n_experts > 0) {
            moe_batch(tb, l, B);
        } else {
            // fused w1|w3 + SwiGLU
            part_range_node(hidden_dim, tb, &r0, &r1);
            matmul_rows_batch(BT.hb, hidden_dim, BT.xq, BT.xs, B, w->w1 + l, dim, r0, r1);
            matmul_rows_batch(BT.hb2, hidden_dim, BT.xq, BT.xs, B, w->w3 + l, dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* h1 = BT.hb + (size_t)b * hidden_dim;
                const float* h3 = BT.hb2 + (size_t)b * hidden_dim;
                for (int i = r0; i < r1; i++) {
                    float val = h1[i];
                    val *= (1.0f / (1.0f + expf(-val)));
                    h1[i] = val * h3[i];
                }
            }
            barrier(tb);

            // quantize the hidden activations, split over tokens
            for (int b = b0; b < b1; b++) {
                quantize_vec(BT.xq + (size_t)b * hidden_dim, BT.xs + (size_t)b * ngh, BT.hb + (size_t)b * hidden_dim, hidden_dim);
            }
            barrier(tb);

            // w2 + residual
            part_range_node(dim, tb, &r0, &r1);
            matmul_rows_batch(BT.xb2, dim, BT.xq, BT.xs, B, w->w2 + l, hidden_dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* x = BT.x + (size_t)b * dim;
                const float* o = BT.xb2 + (size_t)b * dim;
                for (int i = r0; i < r1; i++) { x[i] += o[i]; }
            }
        }
        barrier(tb);
    }

    if (BT.logits_all) { // logits of every position
        for (int b = b0; b < b1; b++) {
            rmsnorm(BT.xb + (size_t)b * dim, BT.x + (size_t)b * dim, w->rms_final_weight, dim);
            quantize_vec(BT.xq + (size_t)b * dim, BT.xs + (size_t)b * ngd, BT.xb + (size_t)b * dim, dim);
        }
        barrier(tb);
        part_range_node(p->vocab_size, tb, &r0, &r1);
        matmul_rows_batch(BT.logits_all, p->vocab_size, BT.xq, BT.xs, B, w->wcls, dim, r0, r1);
        barrier(tb);
        return;
    }
    // logits for the last token only (private rmsnorm + quantize, split classifier rows)
    rmsnorm(tb->xb, BT.x + (size_t)(B - 1) * dim, w->rms_final_weight, dim);
    quantize_vec(tb->q, tb->s, tb->xb, dim);
    part_range_node(p->vocab_size, tb, &r0, &r1);
    matmul_rows(s->logits, tb->q, tb->s, w->wcls, dim, r0, r1);
    barrier(tb);
}

/* Multi-token decode step (B = 2..SPEC_MAX_TOK tokens at positions pos0..pos0+B-1), used to verify
 * speculative drafts. Same structure as forward_thread (dynamic node-local rows, cooperative
 * per-node quantization, tree barriers), but every weight row is applied to all B tokens at once,
 * so the step costs about one weight read. Writes the logits of every position to BT.logits_all. */
static void forward_multi_thread(int tid) {
    Transformer* transformer = P.t;
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    ThreadBuf* tb = &P.tb[tid];
    const int B = BT.B, pos0 = BT.pos0;
    const int dim = p->dim;
    const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    const int kv_mul = p->n_heads / p->n_kv_heads;
    const int hidden_dim = p->hidden_dim;
    const int head_size = dim / p->n_heads;
    const int hs2 = head_size / 2;
    const float att_scale = 1.0f / sqrtf((float)head_size);
    const int8_t* nq = NB[tb->node].q;
    const float* ns = NB[tb->node].s;
    float* X = BT.x; // (B, dim) residual stream, initialized with the embeddings by the caller
    int r0, r1;

    if (tb->ltid == 0) { // node-local reset of the counter set of the next step (unused now)
        struct WorkCounter* wc = P.work[tb->node][(P.seq + 1) & 1];
        for (int k = 0; k < WORK_PHASES * P_max_layers + 1; k++) atomic_store_explicit(&wc[k].v, 0, memory_order_relaxed); // all models
    }

    for (int l = 0; l < p->n_layers; l++) {
        size_t loff = (size_t)l * p->seq_len * kv_dim;

        node_norm_quant_multi(tb, X, w->rms_att_weight + l * dim, dim, B);

        while (grab_rows(WORK_PHASES * l + 0, dim, tb, &r0, &r1)) {
            matmul_rows_batch(BT.q, dim, nq, ns, B, w->wq + l, dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* q = BT.q + (size_t)b * dim;
                const float* rc = BT.rope_c + b * hs2;
                const float* rs = BT.rope_s + b * hs2;
                if (w->bq) for (int i = r0; i < r1; i++) q[i] += w->bq[(size_t)l * dim + i];
                for (int i = r0; i < r1; i += 2) {
                    int j = (i % head_size) / 2;
                    float v0 = q[i], v1 = q[i + 1];
                    q[i] = v0 * rc[j] - v1 * rs[j];
                    q[i + 1] = v0 * rs[j] + v1 * rc[j];
                }
            }
        }
        while (grab_rows(WORK_PHASES * l + 1, kv_dim, tb, &r0, &r1)) {
            matmul_rows_batch(BT.k, kv_dim, nq, ns, B, w->wk + l, dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* k = BT.k + (size_t)b * kv_dim;
                const float* rc = BT.rope_c + b * hs2;
                const float* rs = BT.rope_s + b * hs2;
                float* kc = s->key_cache + loff + (size_t)(pos0 + b) * head_size;
                if (w->bk) for (int i = r0; i < r1; i++) k[i] += w->bk[(size_t)l * kv_dim + i];
                for (int i = r0; i < r1; i += 2) {
                    int j = (i % head_size) / 2;
                    float v0 = k[i], v1 = k[i + 1];
                    k[i] = v0 * rc[j] - v1 * rs[j];
                    k[i + 1] = v0 * rs[j] + v1 * rc[j];
                }
                for (int i = r0; i < r1; i++) kc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = k[i];
            }
        }
        while (grab_rows(WORK_PHASES * l + 2, kv_dim, tb, &r0, &r1)) {
            matmul_rows_batch(BT.v, kv_dim, nq, ns, B, w->wv + l, dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* v = BT.v + (size_t)b * kv_dim;
                float* vc = s->value_cache + loff + (size_t)(pos0 + b) * head_size;
                if (w->bv) for (int i = r0; i < r1; i++) v[i] += w->bv[(size_t)l * kv_dim + i];
                for (int i = r0; i < r1; i++) vc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = v[i];
            }
        }
        barrier(tb);

        // causal attention, items = (kv group of this node, token); output straight into BT.xb
        float* att = BT.att + (size_t)tid * kv_mul * p->seq_len;
        int g0, g1;
        node_groups(p->n_kv_heads, tb->node, &g0, &g1);
        for (int item = tb->ltid; item < (g1 - g0) * B; item += tb->nthn) {
            int g = g0 + item % (g1 - g0), b = item / (g1 - g0);
            int T = pos0 + b + 1;
            const float* kb = s->key_cache + loff + (size_t)g * p->seq_len * head_size;
            const float* vb = s->value_cache + loff + (size_t)g * p->seq_len * head_size;
            const float* qb = BT.q + (size_t)b * dim + (size_t)g * kv_mul * head_size;
            float m[KV_MUL_MAX], sum[KV_MUL_MAX];
            for (int r = 0; r < kv_mul; r++) m[r] = -FLT_MAX;
            for (int t = 0; t < T; t++) {
                const float* k = kb + (size_t)t * head_size;
                for (int r = 0; r < kv_mul; r++) {
                    const float* q = qb + r * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                    score *= att_scale;
                    att[(size_t)r * p->seq_len + t] = score;
                    m[r] = score > m[r] ? score : m[r];
                }
            }
            for (int r = 0; r < kv_mul; r++) {
                float* ar = att + (size_t)r * p->seq_len;
                float sm = 0.0f;
                for (int t = 0; t < T; t++) { float e = expf(ar[t] - m[r]); ar[t] = e; sm += e; }
                sum[r] = 1.0f / sm;
            }
            float* ob = BT.xb + (size_t)b * dim + (size_t)g * kv_mul * head_size;
            memset(ob, 0, kv_mul * head_size * sizeof(float));
            for (int t = 0; t < T; t++) {
                const float* v = vb + (size_t)t * head_size;
                for (int r = 0; r < kv_mul; r++) {
                    float a = att[(size_t)r * p->seq_len + t];
                    float* o = ob + r * head_size;
                    for (int i = 0; i < head_size; i++) o[i] += a * v[i];
                }
            }
            for (int r = 0; r < kv_mul; r++) {
                float* o = ob + r * head_size;
                for (int i = 0; i < head_size; i++) o[i] *= sum[r];
            }
        }
        barrier(tb);
        node_quant_multi(tb, BT.xb, dim, B);

        while (grab_rows(WORK_PHASES * l + 3, dim, tb, &r0, &r1)) {
            matmul_rows_batch(BT.xb2, dim, nq, ns, B, w->wo + l, dim, r0, r1);
            for (int b = 0; b < B; b++)
                for (int i = r0; i < r1; i++) X[(size_t)b * dim + i] += BT.xb2[(size_t)b * dim + i];
        }
        barrier(tb);

        node_norm_quant_multi(tb, X, w->rms_ffn_weight + l * dim, dim, B);
        while (grab_rows(WORK_PHASES * l + 4, hidden_dim, tb, &r0, &r1)) {
            matmul_rows_batch(BT.hb, hidden_dim, nq, ns, B, w->w1 + l, dim, r0, r1);
            matmul_rows_batch(BT.hb2, hidden_dim, nq, ns, B, w->w3 + l, dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* h1 = BT.hb + (size_t)b * hidden_dim;
                const float* h3 = BT.hb2 + (size_t)b * hidden_dim;
                for (int i = r0; i < r1; i++) {
                    float val = h1[i];
                    val *= (1.0f / (1.0f + expf(-val)));
                    h1[i] = val * h3[i];
                }
            }
        }
        barrier(tb);

        node_quant_multi(tb, BT.hb, hidden_dim, B);
        while (grab_rows(WORK_PHASES * l + 5, dim, tb, &r0, &r1)) {
            matmul_rows_batch(BT.xb2, dim, nq, ns, B, w->w2 + l, hidden_dim, r0, r1);
            for (int b = 0; b < B; b++)
                for (int i = r0; i < r1; i++) X[(size_t)b * dim + i] += BT.xb2[(size_t)b * dim + i];
        }
        barrier(tb);
    }

    node_norm_quant_multi(tb, X, w->rms_final_weight, dim, B);
    while (grab_rows(WORK_PHASES * p->n_layers, p->vocab_size, tb, &r0, &r1)) {
        matmul_rows_batch(BT.logits_all, p->vocab_size, nq, ns, B, w->wcls, dim, r0, r1);
    }
    barrier(tb);
}

static void init_batch(Transformer* t) {
    Config* p = &t->config;
    int kv_dim = p->kv_dim, q_dim = p->q_dim, head_size = p->head_dim;
    int qmax = p->dim > p->hidden_dim ? p->dim : p->hidden_dim;
    if (q_dim > qmax) qmax = q_dim;
    int wide = q_dim > p->dim ? q_dim : p->dim;
    size_t B = BATCH_MAX;
    BT.x = calloc(B * p->dim, sizeof(float));
    BT.xb = calloc(B * wide, sizeof(float));
    BT.xb2 = calloc(B * p->dim, sizeof(float));
    BT.q = calloc(B * q_dim, sizeof(float));
    BT.k = calloc(B * kv_dim, sizeof(float));
    BT.v = calloc(B * kv_dim, sizeof(float));
    BT.hb = calloc(B * p->hidden_dim, sizeof(float));
    BT.hb2 = calloc(B * p->hidden_dim, sizeof(float));
    BT.xq = calloc(B * qmax, sizeof(int8_t));
    BT.xs = calloc(B * (qmax / GS + 1), sizeof(float));
    BT.rope_c = calloc(B * (head_size / 2), sizeof(float));
    BT.rope_s = calloc(B * (head_size / 2), sizeof(float));
    BT.att = calloc((size_t)P.nth * (p->n_heads / p->n_kv_heads) * p->seq_len, sizeof(float));
    if (p->n_experts > 0) {
        size_t S = B * p->n_active, E = p->n_experts, mh = p->moe_hidden;
        BT.router = calloc(B * E, sizeof(float));
        BT.moe_sel = calloc(B * p->n_active, sizeof(int));
        BT.moe_wt = calloc(B * p->n_active, sizeof(float));
        BT.moe_cnt = calloc(E + 1, sizeof(int));
        BT.moe_fill = calloc(E + 1, sizeof(int));
        BT.moe_tok = calloc(S, sizeof(int));
        BT.moe_w = calloc(S, sizeof(float));
        BT.gxq = calloc(S * p->dim, 1);
        BT.gxs = calloc(S * (p->dim / GS), sizeof(float));
        BT.gh = calloc(S * mh, sizeof(float));
        BT.gh3 = calloc(S * mh, sizeof(float));
        BT.gq = calloc(S * mh, 1);
        BT.gqs = calloc(S * (mh / GS), sizeof(float));
        BT.gout = calloc(S * p->dim, sizeof(float));
    }
}

/* Process n tokens starting at position pos0 (prompt prefill). Fills the KV cache and returns
 * the logits of the last token. */
/* Speculative verification: run n (<= SPEC_MAX) tokens at positions pos0.. and return the
 * logits of every position as an (n, vocab) array (row i = distribution after token i). */
#define SPEC_MAX SPEC_MAX_TOK
// the batched / multi-token paths support dense models with q_dim == dim and no qk-norm
static int needs_token_path(const Config* p) { return p->n_experts > 0 || p->qk_norm || p->q_dim != p->dim; }
static float* SPEC_LOGITS = NULL;
float* forward_all(Transformer* transformer, const int* tokens, int n, int pos0) {
    use_model(transformer);
    int V = transformer->config.vocab_size;
    if (!SPEC_LOGITS) SPEC_LOGITS = malloc((size_t)SPEC_MAX * V * sizeof(float));
    if (n == 1 || needs_token_path(&transformer->config)) {
        for (int i = 0; i < n; i++) memcpy(SPEC_LOGITS + (size_t)i * V, forward(transformer, tokens[i], pos0 + i), V * sizeof(float));
        return SPEC_LOGITS;
    }
    Config* p = &transformer->config;
    int dim = p->dim, hs2 = p->dim / p->n_heads / 2;
    P.t = transformer;
    BT.B = n;
    BT.pos0 = pos0;
    BT.tok = tokens;
    BT.logits_all = SPEC_LOGITS;
    for (int b = 0; b < n; b++) { // embeddings and RoPE tables of the step
        memcpy(BT.x + (size_t)b * dim, transformer->weights.token_embedding_table + (size_t)tokens[b] * dim, dim * sizeof(float));
        for (int j = 0; j < hs2; j++) {
            double val = (double)(pos0 + b) * P.rope_freq[j];
            BT.rope_c[b * hs2 + j] = (float)cos(val);
            BT.rope_s[b * hs2 + j] = (float)sin(val);
        }
    }
    P.mode = 2;
    P.seq++;
    wake_workers();
    forward_multi_thread(0);
    P.mode = 0;
    BT.logits_all = NULL;
    return SPEC_LOGITS;
}

float* forward_batch(Transformer* transformer, const int* tokens, int n, int pos0) {
    use_model(transformer);

    float* logits = NULL;
    for (int done = 0; done < n;) {
        int B = n - done < BATCH_MAX ? n - done : BATCH_MAX;
        if (B == 1) {
            logits = forward(transformer, tokens[done], pos0 + done);
        } else {
            P.t = transformer;
            P.mode = 1;
            BT.B = B;
            BT.pos0 = pos0 + done;
            BT.tok = tokens + done;
            wake_workers();
            forward_batch_thread(0);
            P.mode = 0;
            logits = transformer->state.logits;
        }
        done += B;
    }
    return logits;
}


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
    int maxd = ARCH >= ARCH_QWEN2 ? 1 : 3; // \p{N} (Qwen2) vs \p{N}{1,3} (Llama 3)
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

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

/* Speculative sampling with a deterministic draft token d (prompt lookup): returns d with
 * probability p(d) under the sampler's distribution (temperature + top-p), otherwise a sample from
 * that distribution with d removed. The output is distributed exactly like sample(). Greedy: argmax. */
int sample_draft(Sampler *sampler, float *logits, int d) {
  if (sampler->temperature == 0.0f) return sample_argmax(logits, sampler->vocab_size);
  int n = sampler->vocab_size;
  for (int q = 0; q < n; q++) logits[q] /= sampler->temperature;
  softmax(logits, n);
  ProbIndex *pi = sampler->probindex;
  int n0 = 0, last;
  float cum = 0.0f;
  if (sampler->topp <= 0 || sampler->topp >= 1) {
    for (int i = 0; i < n; i++) { pi[n0].index = i; pi[n0].prob = logits[i]; n0++; }
    last = n0 - 1;
    cum = 1.0f;
  } else {
    const float cutoff = (1.0f - sampler->topp) / (n - 1);
    for (int i = 0; i < n; i++)
      if (logits[i] >= cutoff) { pi[n0].index = i; pi[n0].prob = logits[i]; n0++; }
    qsort(pi, n0, sizeof(ProbIndex), compare);
    last = n0 - 1;
    for (int i = 0; i < n0; i++) {
      cum += pi[i].prob;
      if (cum > sampler->topp) { last = i; break; }
    }
  }
  float pd = 0.0f; // probability of d in the (truncated, renormalized) distribution
  for (int i = 0; i <= last; i++) if (pi[i].index == d) { pd = pi[i].prob / cum; break; }
  if (random_f32(&sampler->rng_state) < pd) return d;
  // rejected: sample from the distribution without d
  float rest = cum - pd * cum;
  float r = random_f32(&sampler->rng_state) * rest, cdf = 0.0f;
  int fallback = -1;
  for (int i = 0; i <= last; i++) {
    if (pi[i].index == d) continue;
    fallback = pi[i].index;
    cdf += pi[i].prob;
    if (r < cdf) return pi[i].index;
  }
  return fallback >= 0 ? fallback : d;
}

/* Prompt lookup: find the most recent earlier occurrence of the last N tokens of hist (N = 4..2)
 * and propose the tokens that followed it. Returns the draft length. */
static int ngram_draft(const int *hist, int n, int *draft, int max_k) {
  for (int N = 4; N >= 2; N--) {
    if (n <= N) continue;
    const int *tail = hist + n - N;
    for (int s = n - N - 1; s >= 0; s--) { // most recent match first
      if (hist[s] != tail[0] || memcmp(hist + s, tail, N * sizeof(int)) != 0) continue;
      int k = 0;
      while (k < max_k && s + N + k < n) { draft[k] = hist[s + N + k]; k++; }
      if (k > 0) return k;
    }
  }
  return 0;
}

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
    if ((next == 128001 || next == 128009 || next == 151643 || next == 151645) && pos > num_prompt_tokens)
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
static float *chat_feed(Transformer *t, ChatHistory *h, const int *toks, int n) {
  if (n <= 0) return NULL;
  float *logits = forward_batch(t, toks, n, h->n);
  memcpy(h->tok + h->n, toks, n * sizeof(int));
  h->n += n;
  return logits;
}

// make room for `need` more tokens by dropping the oldest turns, then re-prefill
static void chat_make_room(Transformer *t, ChatHistory *h, int need) {
  if (h->n + need <= h->cap) {
    return;
  }
  // keep the system prefix; drop whole turns from the front until we fit, keeping at least
  // half the window free for the new turn and the reply
  int target = h->cap / 2 < h->cap - need ? h->cap / 2 : h->cap - need;
  int drop_to = -1;
  for (int i = 0; i < h->n_turns; i++) {
    int keep = h->sys_len + (h->n - h->turn[i]);
    if (keep <= target) {
      drop_to = i;
      break;
    }
  }
  int start = drop_to >= 0 ? h->turn[drop_to] : h->n; // first token to keep after the prefix
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
  chat_feed(t, h, replay, kept);
  fprintf(stderr, "[context full: dropped %d old tokens, re-processed %d in %.1fs]\n", dropped, kept,
          (time_in_ms() - t0) / 1000.0);
  free(replay);
  if (h->n + need > h->cap) {
    fprintf(stderr, "[message is too long for the context window (%d tokens)]\n", h->cap);
  }
}

static Tools *g_tools = NULL; // set by -f tools.json

// ---- chat formats -----------------------------------------------------------
// Llama 3:  <|start_header_id|>role<|end_header_id|>\n\n ... <|eot_id|>, tool calls as a bare JSON
//           message, tool output as an "ipython" message.
// ChatML (Qwen2.5): <|im_start|>role\n ... <|im_end|>\n, tool calls inside <tool_call></tool_call>,
//           tool output as <tool_response></tool_response> inside a user message.

#define QWEN_ENDOFTEXT 151643
#define QWEN_IM_END 151645
#define QWEN_TOOL_CALL 151657
#define QWEN_TOOL_CALL_END 151658

static int is_end_token(int t) {
  if (ARCH >= ARCH_QWEN2) return t == QWEN_IM_END || t == QWEN_ENDOFTEXT;
  return t == LLAMA3_EOT || t == LLAMA3_EOS || t == LLAMA3_EOM;
}

// text that follows the end token of an assistant message in the template
static const char *after_end(void) { return ARCH >= ARCH_QWEN2 ? "\n" : ""; }
static const char *end_text(void) { return ARCH >= ARCH_QWEN2 ? "<|im_end|>" : "<|eot_id|>"; }

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
      if (l > 2 && v[0] == '<' && v[l - 1] == '>' && strncmp(v, "<|unused_", 9) && strncmp(v, "<|dup_", 6)) {
        special_ids[n_special] = i;
        special_len[n_special++] = l;
      }
    }
  }
  SBuf seg = {0};
  sb_puts(&seg, "");
  while (*s) {
    int found = -1;
    if (*s == '<') {
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
  if (ARCH >= ARCH_QWEN2) {
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
  if (ARCH >= ARCH_QWEN2) {
    m = push_tpl(tk, "<|im_start|>user\n", msg, m);
    m = push_text(tk, text, msg, m);
    m = push_tpl(tk, "<|im_end|>\n<|im_start|>assistant\n", msg, m);
  } else {
    m = push_tpl(tk, "<|start_header_id|>user<|end_header_id|>\n\n", msg, m);
    m = push_text(tk, text, msg, m);
    m = push_tpl(tk, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", msg, m);
  }
  return m;
}

static int build_tool_results(Tokenizer *tk, const char *prefix, char **outs, int n, int *msg, int m) {
  m = push_tpl(tk, prefix, msg, m);
  if (ARCH >= ARCH_QWEN2) {
    m = push_tpl(tk, "<|im_start|>user", msg, m);
    for (int i = 0; i < n; i++) {
      m = push_tpl(tk, "\n<tool_response>\n", msg, m);
      m = push_text(tk, outs[i], msg, m);
      m = push_tpl(tk, "\n</tool_response>", msg, m);
    }
    m = push_tpl(tk, "<|im_end|>\n<|im_start|>assistant\n", msg, m);
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
    forward_batch(d, hist + DRAFT_POS, n - DRAFT_POS, DRAFT_POS);
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
static long SPEC_DRAFTED = 0, SPEC_ACCEPTED = 0;

typedef struct {   // streaming state of one assistant message
  SBuf buf;        // Llama: the undecided/buffered message; ChatML: the current <tool_call> body
  int mode;        // Llama: 0 undecided, 1 streaming text, 2 buffering a call; ChatML: 0/1 text, 2 inside <tool_call>
  int n_calls;
  int token;       // previous token (for decode)
  JNode **calls;
} ReplyState;

// handle one generated (non-end) token: print it or collect it into a tool call
static void reply_token(ReplyState *rs, Tokenizer *tk, int next) {
  SBuf *buf = &rs->buf;
  if (ARCH >= ARCH_QWEN2 && next == QWEN_TOOL_CALL) {
    rs->mode = 2;
    buf->len = 0;
    buf->buf[0] = '\0';
  } else if (ARCH >= ARCH_QWEN2 && next == QWEN_TOOL_CALL_END) {
    if (rs->mode == 2 && rs->n_calls < MAX_CALLS_PER_REPLY) {
      JNode *c = normalize_call(json_parse(buf->buf, NULL));
      if (c) rs->calls[rs->n_calls++] = c;
      else { printf("<tool_call>%s</tool_call>", buf->buf); fflush(stdout); }
    }
    rs->mode = 1;
  } else if (ARCH < ARCH_QWEN2 && next == LLAMA3_PYTHON_TAG) {
    rs->mode = 2;
  } else {
    char *piece = decode(tk, rs->token, next);
    int ok = piece && piece[0] && !(piece[1] == '\0' && !(isprint((unsigned char)piece[0]) || isspace((unsigned char)piece[0])));
    if (ok) {
      if (rs->mode == 2) {
        sb_puts(buf, piece);
      } else if (ARCH >= ARCH_QWEN2 || rs->mode == 1) {
        if (rs->mode == 0) { // skip leading whitespace of the message
          const char *p = piece;
          while (*p && isspace((unsigned char)*p)) p++;
          if (*p) { rs->mode = 1; printf("%s", p); }
        } else {
          printf("%s", piece);
        }
        fflush(stdout);
      } else { // Llama, undecided
        sb_puts(buf, piece);
        const char *p = buf->buf;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '{' && g_tools) {
          rs->mode = 2;
        } else if (*p) {
          rs->mode = 1;
          printf("%s", p);
          fflush(stdout);
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
  sb_puts(&rs.buf, "");
  const int V = tr->config.vocab_size;
  float *own = malloc((size_t)V * sizeof(float)); // logits we own (the shared buffers get overwritten)
  memcpy(own, logits, (size_t)V * sizeof(float));
  int pending = -1; // a token already sampled (after a rejected draft) but not yet fed
  *ended = 0;
  for (;;) {
    int next = pending >= 0 ? pending : sample(sp, own);
    pending = -1;
    if (is_end_token(next)) {
      if (h->n < h->cap) {
        chat_feed(tr, h, &next, 1); // keep the end token in the history
        *ended = 1;
      }
      break;
    }
    reply_token(&rs, tk, next);
    (*gen)++;
    if (h->n >= h->cap) {
      fprintf(stderr, "\n[context window full, reply truncated]");
      break;
    }
    // draft the continuation by prompt lookup over the whole dialog (+ the new token)
    int seq[SPEC_MAX + 1], k = 0;
    seq[0] = next;
    int from_model = 0;
    int room = h->cap - h->n - 1;
    if (SPEC_K > 0) {
      h->tok[h->n] = next; // temporarily, for the lookup
      int kmax = SPEC_K < room ? SPEC_K : room;
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
    float *L = forward_all(tr, seq, k + 1, base);
    memcpy(h->tok + base, seq, (k + 1) * sizeof(int));
    int acc = 0, stop = 0;
    SPEC_DRAFTED += k;
    for (int i = 0; i < k; i++) {
      int y = sample_draft(sp, L + (size_t)i * V, seq[i + 1]);
      if (y != seq[i + 1]) { pending = y; break; }
      acc++;
      if (is_end_token(y)) { *ended = 1; stop = 1; break; } // the end token is already in the cache
      reply_token(&rs, tk, y);
      (*gen)++;
    }
    SPEC_ACCEPTED += acc;
    h->n = base + 1 + acc; // positions after this hold rejected drafts: they get overwritten
    if (from_model) DRAFT_POS = base + 1 + (acc < k - 1 ? acc : k - 1); // draft cache: next, accepted drafts
    if (stop) break;
    if (pending < 0) memcpy(own, L + (size_t)k * V, (size_t)V * sizeof(float)); // all accepted
  }
  if (ARCH < ARCH_QWEN2 && rs.mode == 2) {
    JNode *c = normalize_call(json_parse(rs.buf.buf, NULL));
    if (c) calls[rs.n_calls++] = c;
    else printf("%s", rs.buf.buf); // looked like JSON but is not a call: show it as text
  }
  free(rs.buf.buf);
  free(own);
  return rs.n_calls;
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *cli_system_prompt, int steps) {
  const int ctx = transformer->config.seq_len;
  const int BUF = 1 << 16;
  const int MAX_ROUNDS = 8; // tool-call rounds per user message
  char *text = malloc(BUF);
  int *msg = malloc((4 * BUF + 64) * sizeof(int));
  ChatHistory h = {.tok = malloc(ctx * sizeof(int)), .n = 0, .cap = ctx, .sys_len = 0, .turn = malloc(ctx * sizeof(int)), .n_turns = 0};
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
    chat_make_room(transformer, &h, m + reserve);
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
      // run the tools and hand their output back
      char *outs[MAX_CALLS_PER_REPLY];
      for (int i = 0; i < n_calls; i++) {
        char *shown = NULL;
        outs[i] = round < MAX_ROUNDS ? tools_run(g_tools, calls[i], &shown, 1) : strdup("Error: too many function calls for one message.");
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
      m = build_tool_results(tokenizer, prefix.buf, outs, n_calls, msg, 0);
      prefix.len = 0;
      prefix.buf[0] = '\0';
      for (int i = 0; i < n_calls; i++) free(outs[i]);
      chat_make_room(transformer, &h, m + reserve);
      long p0 = time_in_ms();
      logits = chat_feed(transformer, &h, msg, m);
      prompt_ms += time_in_ms() - p0;
      prompt_tok += m;
      printf("Assistant: ");
      fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
    fprintf(stderr, "[prompt %d tok %.1f tok/s | reply %d tok %.1f tok/s | context %d/%d | draft accepted %ld/%ld]\n", prompt_tok,
            prompt_tok * 1000.0 / (prompt_ms > 0 ? prompt_ms : 1), gen, gen * 1000.0 / (gen_ms > 0 ? gen_ms : 1), h.n, ctx,
            SPEC_ACCEPTED, SPEC_DRAFTED);
    SPEC_ACCEPTED = SPEC_DRAFTED = 0;
  }
  free(prefix.buf);
  free(text);
  free(msg);
  free(h.tok);
  free(h.turn);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

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
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr, "  -T <int>    number of threads, default 0 = auto (placed on the most idle CPUs; RUNQ_PIN=1 keeps them pinned, RUNQ_PIN=0 disables placement)\n");
  fprintf(stderr, "  -c <int>    context window in tokens (KV cache size), default 8192\n");
  fprintf(stderr, "  -f <string> tools.json: enables tool calling in chat mode\n");
  fprintf(stderr, "  -d <string> draft model (same tokenizer, smaller) for speculative decoding in chat\n");
  fprintf(stderr, "  -k <int>    max draft length of prompt-lookup speculative decoding in chat, default 8, 0 = off\n");
  fprintf(stderr, "  -N <int>    NUMA nodes to use, default 0 = all (weights are split between the nodes)\n");
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
  char *draft_path = NULL;         // draft model for speculative decoding

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
    } else if (argv[i][1] == 'd') {
      draft_path = argv[i + 1];
    } else if (argv[i][1] == 'k') {
      SPEC_K = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'N') {
      NODES_REQ = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'f') {
      tools_path = argv[i + 1];
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
  if (needs_token_path(&transformer.config) && (SPEC_K > 0 || draft_path)) {
    // no multi-token path for this model yet: speculation would only cost time
    SPEC_K = 0;
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
