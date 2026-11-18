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
enum { ARCH_LLAMA3 = 0, ARCH_QWEN2 = 1, ARCH_QWEN3 = 2, ARCH_QWEN35 = 3 }; // Qwen*: ChatML template, Qwen tokenizer
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
  // hybrid models (Qwen3.5 / Qwen3.8): every full_interval-th layer full attention, the others Gated DeltaNet
  int hybrid, full_interval, n_attn; // n_attn: layers with full attention (all of them if not hybrid)
  int lin_nk, lin_nv, lin_dk, lin_dv, conv_k; // linear attention: k / v heads, their dims, conv kernel
  int attn_gate, rot_dim;  // full attention: sigmoid output gate, rotary dims (partial RoPE, rotate-half)
  int conv_dim, lin_in;    // 2 nk dk + nv dv; rows of the fused input projection (conv_dim + nv dv + 2 nv)
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
  // hybrid: full-attention tensors are indexed by full layer (n_attn), these by linear layer
  QuantizedTensor *wgate;  // (n_attn, q_dim, dim) attention output gate
  QuantizedTensor *w_in;   // (n_lin, lin_in, dim) [q k v (conv channels) | z | b | a]
  QuantizedTensor *w_out;  // (n_lin, dim, nv dv)
  float *conv_w, *dt_bias, *A, *lnorm; // (n_lin, conv_dim, K), (n_lin, nv), (n_lin, nv) = -exp(A_log), (n_lin, dv)
  QuantizedTensor *wg;     // MoE router (layer, n_experts, dim), or NULL; w1/w3 are then (layer, n_experts * moe_hidden, dim)
  QuantizedTensor *wg_node[8]; // per NUMA node: a node-local copy of the router (every node routes on its own)
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
  float *gate;        // (q_dim,) hybrid: attention output gate
  float *lin;         // (lin_in,) hybrid: fused input projection (after conv / activations)
  float *lin_o;       // (nv dv,) hybrid: recurrence output
  float *lin_S;       // (n_lin, nv, dv, dk) recurrent states (row c of head h = column c of S)
  float *conv_state;  // (n_lin, conv_dim, K - 1) last inputs of the causal convolutions
  int next_pos;       // hybrid: the position the recurrent state is at (it can only advance)
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
  s->key_cache = calloc((size_t)p->n_attn * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = calloc((size_t)p->n_attn * p->seq_len * kv_dim, sizeof(float));
  s->gate = s->lin = s->lin_o = s->lin_S = s->conv_state = NULL;
  s->next_pos = 0;
  if (p->hybrid) {
    int nlin = p->n_layers - p->n_attn;
    s->gate = calloc(p->q_dim, sizeof(float));
    s->lin = calloc(p->lin_in, sizeof(float));
    s->lin_o = calloc((size_t)p->lin_nv * p->lin_dv, sizeof(float));
    s->lin_S = malloc((size_t)nlin * p->lin_nv * p->lin_dv * p->lin_dk * sizeof(float)); // first touched by its threads
    s->conv_state = calloc((size_t)nlin * p->conv_dim * (p->conv_k - 1), sizeof(float));
  }
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
  free(s->gate); free(s->lin); free(s->lin_o); free(s->lin_S); free(s->conv_state);
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
    fptr += p->n_attn * p->head_dim;
    w->k_norm = fptr;
    fptr += p->n_attn * p->head_dim;
  }
  const int nlin = p->n_layers - p->n_attn;
  w->conv_w = w->dt_bias = w->A = w->lnorm = NULL;
  w->wgate = w->w_in = w->w_out = NULL;
  if (p->hybrid) {
    w->conv_w = fptr;
    fptr += (size_t)nlin * p->conv_dim * p->conv_k;
    w->dt_bias = fptr;
    fptr += nlin * p->lin_nv;
    w->A = fptr;
    fptr += nlin * p->lin_nv;
    w->lnorm = fptr;
    fptr += nlin * p->lin_dv;
  }

  // now read all the quantized weights
  ptr = (void *)fptr; // now cast the pointer back to void*
  w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim, 8);
  // dequantize token embedding table
  w->token_embedding_table = malloc((size_t)p->vocab_size * p->dim * sizeof(float));
  // (dequantized by read_checkpoint once the data is loaded)

  w->wq = init_quantized_tensors(&ptr, p->n_attn, p->dim * q_dim, Q4_MASK & 1 ? 4 : 8);
  if (p->hybrid && p->attn_gate) w->wgate = init_quantized_tensors(&ptr, p->n_attn, p->dim * q_dim, Q4_MASK & 1 ? 4 : 8);
  w->wk = init_quantized_tensors(&ptr, p->n_attn, p->dim * kv_dim, Q4_MASK & 2 ? 4 : 8);
  w->wv = init_quantized_tensors(&ptr, p->n_attn, p->dim * kv_dim, Q4_MASK & 4 ? 4 : 8);
  w->wo = init_quantized_tensors(&ptr, p->n_attn, q_dim * p->dim, Q4_MASK & 8 ? 4 : 8);
  if (p->hybrid) {
    w->w_in = init_quantized_tensors(&ptr, nlin, p->lin_in * p->dim, Q4_MASK & 128 ? 4 : 8);
    w->w_out = init_quantized_tensors(&ptr, nlin, p->dim * p->lin_nv * p->lin_dv, Q4_MASK & 256 ? 4 : 8);
  }

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
  place_tensor(w->wq, p->n_attn, p->q_dim, p->dim);
  place_tensor(w->wk, p->n_attn, p->kv_dim, p->dim);
  place_tensor(w->wv, p->n_attn, p->kv_dim, p->dim);
  place_tensor(w->wo, p->n_attn, p->dim, p->q_dim);
  if (w->wgate) place_tensor(w->wgate, p->n_attn, p->q_dim, p->dim);
  if (p->hybrid) {
    place_tensor(w->w_in, p->n_layers - p->n_attn, p->lin_in, p->dim);
    place_tensor(w->w_out, p->n_layers - p->n_attn, p->dim, p->lin_nv * p->lin_dv);
  }
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

/* Weight layout (Q4: see matmul_rows_q4). After loading, every int8 matrix is repacked in place into blocks of 4 rows:
 * per block and quantization group, GS/4 chunks of 16 bytes [row0 k..k+3 | row1 | row2 | row3],
 * and the scales of a block as [group][4 rows]. One `sdot` by lane (4 weight rows x 4 activations)
 * then yields the 4 rows in the 4 lanes, with no horizontal reductions; the batched kernel reuses
 * every weight vector for 8 tokens. Row offsets of multiples of 4 (node splits, experts) keep their
 * meaning, only the order inside a block changes. */
typedef struct { QuantizedTensor *t; int rows, cols; } PackJob;
static void *pack_job(void *arg) {
  PackJob *j = arg;
  if (j->t->bits == 4) { // Q4: chunk j of a group = bytes 4j..4j+3 of the 4 rows; scales [s0..s3 m0..m3]
    const int n = j->cols, ng = n / 32, bpr = n / 2;
    uint8_t *tq = malloc((size_t)4 * bpr);
    uint16_t *ts = malloc((size_t)4 * ng * 2 * sizeof(uint16_t));
    for (int b = 0; b < j->rows / 4; b++) {
      uint8_t *q = (uint8_t *)j->t->q + (size_t)b * 4 * bpr;
      uint16_t *sc = (uint16_t *)(j->t->s + (size_t)b * 4 * ng);
      memcpy(tq, q, (size_t)4 * bpr);
      memcpy(ts, sc, (size_t)4 * ng * 2 * sizeof(uint16_t));
      for (int g = 0; g < ng; g++) {
        for (int c = 0; c < 4; c++)
          for (int r = 0; r < 4; r++) memcpy(q + (size_t)g * 64 + c * 16 + r * 4, tq + (size_t)r * bpr + g * 16 + c * 4, 4);
        for (int r = 0; r < 4; r++) {
          sc[g * 8 + r] = ts[((size_t)r * ng + g) * 2];         // s
          sc[g * 8 + 4 + r] = ts[((size_t)r * ng + g) * 2 + 1]; // m
        }
      }
    }
    free(tq);
    free(ts);
    return NULL;
  }
  const int n = j->cols, ng = n / GS;
  int8_t *tq = malloc((size_t)4 * n);
  float *ts = malloc((size_t)4 * ng * sizeof(float));
  for (int b = 0; b < j->rows / 4; b++) {
    int8_t *q = j->t->q + (size_t)b * 4 * n;
    float *sc = j->t->s + (size_t)b * 4 * ng;
    memcpy(tq, q, (size_t)4 * n);
    memcpy(ts, sc, (size_t)4 * ng * sizeof(float));
    for (int g = 0; g < ng; g++) {
      for (int kk = 0; kk < GS / 4; kk++)
        for (int r = 0; r < 4; r++) memcpy(q + (size_t)g * GS * 4 + kk * 16 + r * 4, tq + (size_t)r * n + g * GS + kk * 4, 4);
      for (int r = 0; r < 4; r++) sc[g * 4 + r] = ts[(size_t)r * ng + g];
    }
  }
  free(tq);
  free(ts);
  return NULL;
}
static void pack_q8(TransformerWeights *w, Config *p) {
  struct { QuantizedTensor *t; int n, rows, cols; } list[] = {
    {w->wq, p->n_attn, p->q_dim, p->dim}, {w->wk, p->n_attn, p->kv_dim, p->dim}, {w->wv, p->n_attn, p->kv_dim, p->dim},
    {w->wo, p->n_attn, p->dim, p->q_dim}, {w->wgate, w->wgate ? p->n_attn : 0, p->q_dim, p->dim},
    {w->w_in, w->w_in ? p->n_layers - p->n_attn : 0, p->lin_in, p->dim},
    {w->w_out, w->w_out ? p->n_layers - p->n_attn : 0, p->dim, p->lin_nv * p->lin_dv},
    {w->w1, p->n_layers, p->n_experts > 0 ? p->n_experts * p->moe_hidden : p->hidden_dim, p->dim},
    {w->w3, p->n_layers, p->n_experts > 0 ? p->n_experts * p->moe_hidden : p->hidden_dim, p->dim},
    {w->w2, p->n_layers, p->n_experts > 0 ? p->n_experts * p->dim : p->dim, p->n_experts > 0 ? p->moe_hidden : p->hidden_dim},
    {w->wg, w->wg ? p->n_layers : 0, p->n_experts, p->dim}, {w->wcls, 1, p->vocab_size, p->dim}};
  int nj = 0;
  for (size_t k = 0; k < sizeof(list) / sizeof(list[0]); k++) nj += list[k].n;
  PackJob *jobs = malloc(nj * sizeof(PackJob));
  pthread_t *th = malloc(nj * sizeof(pthread_t));
  int m = 0;
  for (size_t k = 0; k < sizeof(list) / sizeof(list[0]); k++)
    for (int l = 0; l < list[k].n; l++) {
      if (list[k].rows % 4 || list[k].cols % GS) { fprintf(stderr, "Q8 matrix %d x %d: rows must be a multiple of 4\n", list[k].rows, list[k].cols); exit(EXIT_FAILURE); }
      jobs[m++] = (PackJob){list[k].t + l, list[k].rows, list[k].cols};
    }
  for (int a = 0; a < m; a += 128) { // at most 128 threads at a time
    int e = a + 128 < m ? a + 128 : m;
    for (int k = a; k < e; k++) pthread_create(&th[k], NULL, pack_job, &jobs[k]);
    for (int k = a; k < e; k++) pthread_join(th[k], NULL);
  }
  free(jobs);
  free(th);
}

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
  int q4_format = 0;
  if (fread(&q4_format, sizeof(int), 1, file) != 1) q4_format = 0;
  int hyb[8] = {0}; // full_interval, linear k heads, v heads, k dim, v dim, conv kernel, attention gate, rotary dims
  if (fread(hyb, sizeof(hyb), 1, file) != 1) memset(hyb, 0, sizeof(hyb));
  config->hybrid = ARCH == ARCH_QWEN35 && hyb[0] > 0;
  config->full_interval = hyb[0];
  config->n_attn = config->hybrid ? config->n_layers / hyb[0] : config->n_layers;
  config->lin_nk = hyb[1]; config->lin_nv = hyb[2]; config->lin_dk = hyb[3]; config->lin_dv = hyb[4];
  config->conv_k = hyb[5]; config->attn_gate = hyb[6];
  config->rot_dim = hyb[7] > 0 ? hyb[7] : 0;
  config->conv_dim = 2 * hyb[1] * hyb[3] + hyb[2] * hyb[4];
  config->lin_in = config->conv_dim + hyb[2] * hyb[4] + 2 * hyb[2];
  if (Q4_MASK && q4_format != 1) {
    fprintf(stderr, "%s: old symmetric Q4 format, please re-export it with scripts/export_hf_q8.py --q4\n", checkpoint);
    exit(EXIT_FAILURE);
  }
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
  pack_q8(weights, config); // after the embeddings were dequantized (the classifier may share them)
  for (int k = 0; k < NN; k++) weights->wg_node[k] = weights->wg;
#if defined(__linux__)
  if (NN > 1 && config->n_experts > 0) { // the router is small (n_experts x dim): replicate it on every node
    size_t qb = (size_t)config->n_experts * config->dim, sb = qb / GS * sizeof(float);
    size_t per = ((qb + sb) + 4095) & ~(size_t)4095;
    for (int k = 0; k < NN; k++) {
      QuantizedTensor *c = malloc(config->n_layers * sizeof(QuantizedTensor));
      char *mem = aligned_alloc(4096, per * config->n_layers);
      bind_range(mem, per * config->n_layers, NODE_ID[k]);
      for (int l = 0; l < config->n_layers; l++) {
        c[l].q = (int8_t *)(mem + per * l);
        c[l].s = (float *)(mem + per * l + qb);
        c[l].bits = 8;
        memcpy(c[l].q, weights->wg[l].q, qb);
        memcpy(c[l].s, weights->wg[l].s, sb);
      }
      weights->wg_node[k] = c;
    }
  }
#endif
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
    for (int l = 0; l < p->n_attn; l++)
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

#define MOE_MAX_ACTIVE 16
struct BarGroup { _Alignas(64) atomic_int c; };
typedef struct {
    _Alignas(128) float* xb; // (dim,) private normalized input / attention output (padded: no false sharing)
    int8_t* q;     // (max(dim, hidden_dim),) private quantized activations
    float* s;      // scale factors for q
    float* rope_c; // (head_size/2,) cos for the current position
    float* rope_s; // (head_size/2,) sin for the current position
    int node;      // NUMA node index (0..NN-1) of this thread
    int ltid, nthn; // index within the node, threads of the node
    int ep;          // barrier epoch of this thread (node-wide: every barrier of the node increments it)
    long long phase; // work phase of the last grab (the first grab of a phase takes the static ticket ltid)
    int npf;         // weight regions to prefetch while waiting at the next barrier
    const char* pf[MOE_MAX_ACTIVE * 2];
    int pflen[MOE_MAX_ACTIVE * 2];
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
    struct BarGroup xflag[8];            // cross-node level: epoch published by each node leader
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
// Guided schedule over [0, n): tickets 0..nbig-1 are big chunks, the last quarter of the range goes
// in 4-row tickets, so the phase ends with short pieces instead of a long tail of one big chunk
// (under full memory load a thread streams only ~2 GB/s, a 16-row chunk would take ~15 us).
static inline int ticket_range(int t, int n, int nthn, int* b0, int* b1) {
    int chunk = (n / (nthn * 4) + 3) & ~3;
    if (chunk < 16) chunk = 16; // shorter row runs stream worse (prefetcher ramp-up per row)
    int nbig = (n - n / 4) / chunk;
    int b, e;
    if (t < nbig) { b = t * chunk; e = b + chunk; }
    else { b = nbig * chunk + (t - nbig) * 4; e = b + 4; }
    if (b >= n) return 0;
    *b0 = b;
    *b1 = e < n ? e : n;
    return 1;
}
// The first ticket of every thread in a phase is static (ticket ltid), so its rows are known before
// the phase starts and can be prefetched while the thread waits at the barrier (pf_first).
static inline int grab_ticket(atomic_int* ctr, int n, ThreadBuf* tb, int k, int* b0, int* b1) {
    long long key = ((long long)P.seq << 20) | k;
    int t;
    if (tb->phase != key) { tb->phase = key; t = tb->ltid; }
    else t = tb->nthn + atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
    return ticket_range(t, n, tb->nthn, b0, b1);
}
static inline int grab_rows(int k, int d, ThreadBuf* tb, int* r0, int* r1) {
    int lo, hi;
    node_rows(d, tb->node, &lo, &hi);
    if (!grab_ticket(&P.work[tb->node][P.seq & 1][k].v, hi - lo, tb, k, r0, r1)) return 0;
    *r0 += lo;
    *r1 += lo;
    return 1;
}
// like grab_rows, over a node-local virtual range [0, n) (MoE: the node's halves of the selected experts)
static inline int grab_virtual(int k, int n, ThreadBuf* tb, int* r0, int* r1) {
    return grab_ticket(&P.work[tb->node][P.seq & 1][k].v, n, tb, k, r0, r1);
}
// queue rows [r0, r1) of w (cols columns) for prefetching during the next barrier wait
static inline void pf_add(ThreadBuf* tb, const QuantizedTensor* w, int cols, int r0, int r1) {
    if (tb->npf >= MOE_MAX_ACTIVE * 2 || r1 <= r0) return;
    size_t bpr = w->bits == 4 ? cols / 2 : cols;
    tb->pf[tb->npf] = (const char*)(w->q + (size_t)r0 * bpr);
    tb->pflen[tb->npf] = (int)((r1 - r0) * bpr);
    tb->npf++;
}
// the rows of this thread's static first ticket of a node-split d-row matrix (virtual: n rows from 0)
static inline void pf_first(ThreadBuf* tb, const QuantizedTensor* w, int cols, int d, int virt) {
    int lo = 0, hi = d, b0, b1;
    if (!virt) node_rows(d, tb->node, &lo, &hi);
    if (ticket_range(tb->ltid, hi - lo, tb->nthn, &b0, &b1)) pf_add(tb, w, cols, lo + b0, lo + b1);
}
// one step of the prefetch queue (called from the barrier spin loops): a few lines into L2
static inline void pf_step(ThreadBuf* tb) {
    if (tb->npf == 0) return;
    int i = tb->npf - 1;
    for (int j = 0; j < 4 && tb->pflen[i] > 0; j++) {
#if defined(__aarch64__)
        __asm__ __volatile__("prfm pldl2keep, [%0]" : : "r"(tb->pf[i]));
#else
        __builtin_prefetch(tb->pf[i], 0, 2);
#endif
        tb->pf[i] += 64;
        tb->pflen[i] -= 64;
    }
    if (tb->pflen[i] <= 0) tb->npf--;
}
// rows [e*rows, (e+1)*rows) of a tensor as a tensor of its own (an expert of a MoE layer)
static inline QuantizedTensor expert_view(const QuantizedTensor* t, int e, int rows, int cols) {
    size_t bpr = t->bits == 4 ? cols / 2 : cols, spr = t->bits == 4 ? cols / 32 : cols / GS;
    QuantizedTensor v = {t->q + (size_t)e * rows * bpr, t->s + (size_t)e * rows * spr, t->bits};
    return v;
}
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
enum { PH_ROPE, PH_NORM, PH_QKV, PH_ATT, PH_MERGE, PH_WO, PH_FNORM, PH_W13, PH_QH, PH_W2, PH_CLS, PH_BARRIER, PH_NBARRIER, PH_IDLE, PH_N };
#ifdef PROFILE
static const char* PH_NAME[PH_N] = {"rope/setup", "att rmsnorm+quant", "qkv matmul+rope", "attention", "att merge+quant",
                                    "wo matmul", "ffn rmsnorm+quant", "w1/w3 matmul+swiglu", "quant hidden", "w2 matmul",
                                    "final norm+classifier", "barrier wait", "node barrier wait", "idle (outside forward)"};
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
/* Barriers are trees: threads arrive at the counter of their group of BAR_GROUP, the last one of
 * each group arrives at the node counter, the last one of the node either releases the node
 * (node barrier) or first meets the representatives of the other nodes (global barrier). All
 * spinning happens on a node-local line; only NN threads touch the cross-socket counter. */
#define BAR_GROUP 16
static void node_arrive_release(ThreadBuf* tb, int global);
static void barrier_impl(ThreadBuf* tb) {
    if (P.nth == 1) return;
    node_arrive_release(tb, NN > 1);
}

/* Per-node cooperation. The input vector of every matmul (normalized + quantized) is produced
 * once per NUMA node by all threads of that node, each doing a slice, into a node-local buffer.
 * Every cache line of the shared fp32 activations crosses the socket link once per node instead
 * of once per thread, and nobody repeats the whole rmsnorm/quantization. */
typedef struct {
    int8_t* q;     // quantized input vector (max(dim, hidden_dim))
    float* s;      // its scales
    float* ss;     // per-thread partial sums of squares (one per cache line)
    struct NodeSync* sy; // barrier state of the node, in a page bound to the node
    float* router;       // MoE router logits computed by the node (every node routes on its own)
    int8_t* gq;          // MoE multi-token step: quantized ffn input of every (expert, token) pair
    float* gs;
} NodeBuf;
struct NodeSync {
    _Alignas(64) atomic_int rel;     // release epoch the node's threads spin on
    _Alignas(64) int xep;            // cross-node epoch (node leader only)
    struct BarGroup arr[512];        // per-thread arrival epochs (one line each)
    struct BarGroup garr[64];        // per-group arrival epochs
    struct BarGroup grel[64];        // per-group release epochs (set by the group leader)
};
static NodeBuf NB[MAX_NODES];
static int P_qmax = 0; // longest quantized node vector (NB slot size)
static int P_max_layers = 0; // most layers of any loaded model (work counter reset range)
#define SPEC_MAX_TOK 16 // tokens per multi-token decode step (speculative verification)
static void quantize_vec(int8_t* q, float* s, const float* x, int n);

/* Flag tree, no atomic read-modify-write: a thread publishes its epoch in its own cache line; the
 * fixed leader of each group of BAR_GROUP polls its members, the node leader (ltid 0) polls the group
 * leaders, meets the other node leaders through one flag line each (global barrier) and releases the
 * node with one store. Polling reads in parallel instead of serializing RMWs on a shared line. */
static void node_arrive_release(ThreadBuf* tb, int global) {
    struct NodeSync* sy = NB[tb->node].sy;
    const int lt = tb->ltid, e = ++tb->ep;
    pf_step(tb);
    if (lt % BAR_GROUP != 0) {
        atomic_store_explicit(&sy->arr[lt].c, e, memory_order_release);
    } else {
        const int end = lt + BAR_GROUP < tb->nthn ? lt + BAR_GROUP : tb->nthn;
        for (int j = lt + 1; j < end; j++)
            while (atomic_load_explicit(&sy->arr[j].c, memory_order_acquire) != e) { pf_step(tb); cpu_relax(); }
        if (lt != 0) {
            atomic_store_explicit(&sy->garr[lt / BAR_GROUP].c, e, memory_order_release);
            while (atomic_load_explicit(&sy->rel, memory_order_acquire) != e) { pf_step(tb); cpu_relax(); }
            atomic_store_explicit(&sy->grel[lt / BAR_GROUP].c, e, memory_order_release);
            tb->npf = 0;
            return;
        } else {
            const int ng = (tb->nthn + BAR_GROUP - 1) / BAR_GROUP;
            for (int g = 1; g < ng; g++)
                while (atomic_load_explicit(&sy->garr[g].c, memory_order_acquire) != e) { pf_step(tb); cpu_relax(); }
            if (global) {
                int xe = ++sy->xep;
                atomic_store_explicit(&P.xflag[tb->node].c, xe, memory_order_release);
                for (int k = 0; k < NN; k++)
                    if (k != tb->node)
                        while (atomic_load_explicit(&P.xflag[k].c, memory_order_acquire) - xe < 0) { pf_step(tb); cpu_relax(); }
            }
            atomic_store_explicit(&sy->rel, e, memory_order_release);
            atomic_store_explicit(&sy->grel[0].c, e, memory_order_release);
            tb->npf = 0;
            return;
        }
    }
    while (atomic_load_explicit(&sy->grel[lt / BAR_GROUP].c, memory_order_acquire) != e) { pf_step(tb); cpu_relax(); }
    tb->npf = 0;
}
static void node_barrier(ThreadBuf* tb) {
    if (tb->nthn == 1) return;
#ifdef PROFILE
    int tid = (int)(tb - P.tb), saved = PROF_T[tid].cur;
    PROF(tid, PH_NBARRIER);
    node_arrive_release(tb, 0);
    PROF(tid, saved);
#else
    node_arrive_release(tb, 0);
#endif
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
// scalar access to the packed Q8 layout (row i, column k)
static inline int8_t q8_at(const QuantizedTensor* w, int n, int i, int k) {
    return w->q[(size_t)(i / 4) * 4 * n + (size_t)(k / GS) * GS * 4 + ((k % GS) / 4) * 16 + (i % 4) * 4 + k % 4];
}
static inline float q8_scale(const QuantizedTensor* w, int n, int i, int g) {
    return w->s[(size_t)(i / 4) * 4 * (n / GS) + g * 4 + i % 4];
}
static void matmul_rows_q8_scalar(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    for (int i = r0; i < r1; i++) {
        float val = 0.0f;
        for (int g = 0; g < n / GS; g++) {
            int32_t iv = 0;
            for (int k = g * GS; k < (g + 1) * GS; k++) iv += (int32_t)xq[k] * q8_at(w, n, i, k);
            val += (float)iv * q8_scale(w, n, i, g) * xs[g];
        }
        out[i] = val;
    }
}
static void matmul_rows_q8(float* out, const int8_t* xq, const float* xs,
                        const QuantizedTensor* w, int n, int r0, int r1) {
    const int ng = n / GS;
    int i = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (GS % 16 == 0 && r0 % 4 == 0) {
        const int8_t* qend = w->q + (size_t)r1 * n;
        for (; i + 4 <= r1; i += 4) {
            const int8_t* wb = w->q + (size_t)i * n;
            const float* sb = w->s + (size_t)i * ng;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int g = 0; g < ng; g++) {
                const int8_t* wg = wb + (size_t)g * GS * 4;
                if (wg + 4096 + 256 <= qend) { // one block is a single stream: keep it ahead, within the range
                    __builtin_prefetch(wg + 4096, 0, 0);
                    __builtin_prefetch(wg + 4096 + 64, 0, 0);
                    __builtin_prefetch(wg + 4096 + 128, 0, 0);
                    __builtin_prefetch(wg + 4096 + 192, 0, 0);
                }
                int32x4_t a = vdupq_n_s32(0), b = a;
                for (int k = 0; k < GS; k += 16) {
                    int8x16_t xv = vld1q_s8(xq + g * GS + k);
                    a = vdotq_laneq_s32(a, vld1q_s8(wg + k * 4), xv, 0);
                    b = vdotq_laneq_s32(b, vld1q_s8(wg + k * 4 + 16), xv, 1);
                    a = vdotq_laneq_s32(a, vld1q_s8(wg + k * 4 + 32), xv, 2);
                    b = vdotq_laneq_s32(b, vld1q_s8(wg + k * 4 + 48), xv, 3);
                }
                acc = vfmaq_f32(acc, vcvtq_f32_s32(vaddq_s32(a, b)), vmulq_n_f32(vld1q_f32(sb + g * 4), xs[g]));
            }
            vst1q_f32(out + i, acc);
        }
    }
#endif
    if (i < r1) matmul_rows_q8_scalar(out, xq, xs, w, n, i, r1);
}

// ----------------------------------------------------------------------------
// Q4 weights (q4_format 1): value = s * u + m per group of 32, u in 0..15, s and m in fp16.
// File: per row 16 bytes per group (byte k = u[k] | u[k+16] << 4), then per group [s, m].
// Memory (packed at load like Q8): blocks of 4 rows; per group 4 chunks of 16 bytes, chunk j =
// [row0 bytes 4j..4j+3 | row1 | row2 | row3]: its low nibbles are the 4 rows x k = 4j..4j+3, its high
// nibbles the same for k = 16+4j..; the scales of a block per group as [s0 s1 s2 s3 m0 m1 m2 m3].
// sum_k (s u_k + m) x_k = s * sum_k u_k x_k + m * sum_k x_k: the min only needs the group sums of the
// activations (once per token), so the inner loop is AND / shift / sdot by lane.
#define Q4_MAXG 2048 // groups of 32 per row (n <= 65536)
#if defined(__aarch64__)
typedef __fp16 half_t;
#else
typedef _Float16 half_t;
#endif
static inline const half_t* q4_scales(const QuantizedTensor* w, int n, int i) { // block of row i
    return (const half_t*)(w->s + (size_t)(i / 4) * 4 * (n / 32));
}
static void matmul_rows_q4_scalar(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    const uint8_t* q = (const uint8_t*)w->q;
    for (int i = r0; i < r1; i++) {
        const uint8_t* bq = q + (size_t)(i / 4) * 4 * (n / 2);
        const half_t* sc = q4_scales(w, n, i);
        float val = 0.0f;
        for (int g = 0; g < n / 32; g++) {
            int32_t su = 0, sx = 0;
            for (int kk = 0; kk < 32; kk++) {
                int j = (kk % 16) / 4, b = kk % 4;
                uint8_t byte = bq[(size_t)g * 64 + j * 16 + (i % 4) * 4 + b];
                int u = kk < 16 ? (byte & 15) : (byte >> 4);
                su += u * xq[g * 32 + kk];
                sx += xq[g * 32 + kk];
            }
            float xsc = xs[g * 32 / GS];
            val += (float)sc[g * 8 + i % 4] * xsc * su + (float)sc[g * 8 + 4 + i % 4] * xsc * sx;
        }
        out[i] = val;
    }
}
#if defined(__ARM_FEATURE_DOTPROD)
// group sums of the activations of one token, times their scales: the min terms
static inline void q4_xsums(float* xsum, const int8_t* xq, const float* xs, int n) {
    for (int g = 0; g < n / 32; g++)
        xsum[g] = (float)(vaddlvq_s8(vld1q_s8(xq + g * 32)) + vaddlvq_s8(vld1q_s8(xq + g * 32 + 16))) * xs[g * 32 / GS];
}
#endif
static void matmul_rows_q4(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    int i = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (r0 % 4 == 0 && GS % 32 == 0 && n / 32 <= Q4_MAXG) {
        const int ng = n / 32;
        float xsum[Q4_MAXG];
        q4_xsums(xsum, xq, xs, n);
        const uint8_t* qend = (const uint8_t*)w->q + (size_t)r1 * (n / 2);
        const uint8x16_t m15 = vdupq_n_u8(0x0F);
        for (; i + 4 <= r1; i += 4) {
            const uint8_t* wb = (const uint8_t*)w->q + (size_t)i * (n / 2);
            const half_t* sb = q4_scales(w, n, i);
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int g = 0; g < ng; g++) {
                const uint8_t* wg = wb + (size_t)g * 64;
                if (wg + 2048 + 64 <= qend) __builtin_prefetch(wg + 2048, 0, 0);
                const int8x16_t xa = vld1q_s8(xq + g * 32), xb = vld1q_s8(xq + g * 32 + 16);
                int32x4_t a = vdupq_n_s32(0), b = a;
                uint8x16_t v;
#define Q4_CHUNK(J) v = vld1q_u8(wg + J * 16); \
                a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(vandq_u8(v, m15)), xa, J); \
                b = vdotq_laneq_s32(b, vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), xb, J);
                Q4_CHUNK(0) Q4_CHUNK(1) Q4_CHUNK(2) Q4_CHUNK(3)
#undef Q4_CHUNK
                const float16x8_t sm = vld1q_f16((const float16_t*)(sb + g * 8));
                acc = vfmaq_f32(acc, vcvtq_f32_s32(vaddq_s32(a, b)), vmulq_n_f32(vcvt_f32_f16(vget_low_f16(sm)), xs[g * 32 / GS]));
                acc = vfmaq_n_f32(acc, vcvt_high_f32_f16(sm), xsum[g]);
            }
            vst1q_f32(out + i, acc);
        }
    }
#endif
    if (i < r1) matmul_rows_q4_scalar(out, xq, xs, w, n, i, r1);
}
#if defined(__ARM_FEATURE_DOTPROD)
// tile of 4 rows x TT tokens: the 8 unpacked weight vectors of a group serve TT tokens
#define Q4_TILE(TT)                                                                                         \
    do {                                                                                                    \
        float32x4_t acc[TT];                                                                                \
        for (int u = 0; u < TT; u++) acc[u] = vdupq_n_f32(0.0f);                                            \
        for (int g = 0; g < ng; g++) {                                                                      \
            const uint8_t* wg = wb + (size_t)g * 64;                                                        \
            const uint8x16_t v0 = vld1q_u8(wg), v1 = vld1q_u8(wg + 16), v2 = vld1q_u8(wg + 32), v3 = vld1q_u8(wg + 48); \
            const int8x16_t l0 = vreinterpretq_s8_u8(vandq_u8(v0, m15)), h0 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)); \
            const int8x16_t l1 = vreinterpretq_s8_u8(vandq_u8(v1, m15)), h1 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)); \
            const int8x16_t l2 = vreinterpretq_s8_u8(vandq_u8(v2, m15)), h2 = vreinterpretq_s8_u8(vshrq_n_u8(v2, 4)); \
            const int8x16_t l3 = vreinterpretq_s8_u8(vandq_u8(v3, m15)), h3 = vreinterpretq_s8_u8(vshrq_n_u8(v3, 4)); \
            const float16x8_t sm = vld1q_f16((const float16_t*)(sb + g * 8));                               \
            const float32x4_t s4 = vcvt_f32_f16(vget_low_f16(sm)), m4 = vcvt_high_f32_f16(sm);              \
            for (int u = 0; u < TT; u++) {                                                                  \
                const int8_t* xp = xq + (size_t)(t + u) * n + g * 32;                                       \
                const int8x16_t xa = vld1q_s8(xp), xb = vld1q_s8(xp + 16);                                   \
                int32x4_t a = vdupq_n_s32(0);                                                               \
                a = vdotq_laneq_s32(a, l0, xa, 0); a = vdotq_laneq_s32(a, h0, xb, 0);                       \
                a = vdotq_laneq_s32(a, l1, xa, 1); a = vdotq_laneq_s32(a, h1, xb, 1);                       \
                a = vdotq_laneq_s32(a, l2, xa, 2); a = vdotq_laneq_s32(a, h2, xb, 2);                       \
                a = vdotq_laneq_s32(a, l3, xa, 3); a = vdotq_laneq_s32(a, h3, xb, 3);                       \
                acc[u] = vfmaq_f32(acc[u], vcvtq_f32_s32(a), vmulq_n_f32(s4, xs[(size_t)(t + u) * ngx + g * 32 / GS])); \
                acc[u] = vfmaq_n_f32(acc[u], m4, xsum[u][g]);                                               \
            }                                                                                               \
        }                                                                                                   \
        for (int u = 0; u < TT; u++) vst1q_f32(out + (size_t)(t + u) * ldo + i, acc[u]);                     \
    } while (0)
#endif
static void matmul_rows_batch_q4(float* out, int ldo, const int8_t* xq, const float* xs, int B,
                                 const QuantizedTensor* w, int n, int r0, int r1) {
    const int ngx = n / GS;
    int iend = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (r0 % 4 == 0 && GS % 32 == 0 && n / 32 <= Q4_MAXG) {
        const int ng = n / 32;
        const uint8x16_t m15 = vdupq_n_u8(0x0F);
        static _Thread_local float xsum[8][Q4_MAXG];
        iend = r0 + (r1 - r0) / 4 * 4;
        int t = 0;
        for (; t + 8 <= B; t += 8) { // token tiles outside: the thread's weights stay in L2
            for (int u = 0; u < 8; u++) q4_xsums(xsum[u], xq + (size_t)(t + u) * n, xs + (size_t)(t + u) * ngx, n);
            for (int i = r0; i < iend; i += 4) {
                const uint8_t* wb = (const uint8_t*)w->q + (size_t)i * (n / 2);
                const half_t* sb = q4_scales(w, n, i);
                Q4_TILE(8);
            }
        }
        for (; t + 4 <= B; t += 4) {
            for (int u = 0; u < 4; u++) q4_xsums(xsum[u], xq + (size_t)(t + u) * n, xs + (size_t)(t + u) * ngx, n);
            for (int i = r0; i < iend; i += 4) {
                const uint8_t* wb = (const uint8_t*)w->q + (size_t)i * (n / 2);
                const half_t* sb = q4_scales(w, n, i);
                Q4_TILE(4);
            }
        }
        for (; t < B; t++) matmul_rows_q4(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ngx, w, n, r0, iend);
    }
#endif
    if (iend < r1)
        for (int t = 0; t < B; t++) matmul_rows_q4_scalar(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ngx, w, n, iend, r1);
}

// tile of 4 rows x TT tokens on the packed layout
#define Q8_TILE(TT)                                                                                        \
    do {                                                                                                   \
        float32x4_t acc[TT];                                                                               \
        for (int u = 0; u < TT; u++) acc[u] = vdupq_n_f32(0.0f);                                           \
        for (int g = 0; g < ng; g++) {                                                                     \
            int32x4_t a[TT];                                                                               \
            for (int u = 0; u < TT; u++) a[u] = vdupq_n_s32(0);                                            \
            const int8_t* wg = wb + (size_t)g * GS * 4;                                                    \
            for (int k = 0; k < GS; k += 16) {                                                             \
                int8x16_t w0 = vld1q_s8(wg + k * 4), w1 = vld1q_s8(wg + k * 4 + 16);                       \
                int8x16_t w2 = vld1q_s8(wg + k * 4 + 32), w3 = vld1q_s8(wg + k * 4 + 48);                  \
                for (int u = 0; u < TT; u++) {                                                             \
                    int8x16_t xv = vld1q_s8(xq + (size_t)(t + u) * n + g * GS + k);                         \
                    a[u] = vdotq_laneq_s32(a[u], w0, xv, 0);                                               \
                    a[u] = vdotq_laneq_s32(a[u], w1, xv, 1);                                               \
                    a[u] = vdotq_laneq_s32(a[u], w2, xv, 2);                                               \
                    a[u] = vdotq_laneq_s32(a[u], w3, xv, 3);                                               \
                }                                                                                          \
            }                                                                                              \
            const float32x4_t ws = vld1q_f32(sb + g * 4);                                                  \
            for (int u = 0; u < TT; u++)                                                                   \
                acc[u] = vfmaq_f32(acc[u], vcvtq_f32_s32(a[u]), vmulq_n_f32(ws, xs[(size_t)(t + u) * ng + g])); \
        }                                                                                                  \
        for (int u = 0; u < TT; u++) vst1q_f32(out + (size_t)(t + u) * ldo + i, acc[u]);                    \
    } while (0)
static void matmul_rows_batch_q8(float* out, int ldo, const int8_t* xq, const float* xs, int B,
                                 const QuantizedTensor* w, int n, int r0, int r1) {
    const int ng = n / GS;
    int iend = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (GS % 16 == 0 && r0 % 4 == 0) {
        iend = r0 + (r1 - r0) / 4 * 4;
        // token tiles outside, the thread's rows inside: its weights stay in L2, the tile's inputs in L1
        int t = 0;
        for (; t + 8 <= B; t += 8)
            for (int i = r0; i < iend; i += 4) {
                const int8_t* wb = w->q + (size_t)i * n;
                const float* sb = w->s + (size_t)i * ng;
                Q8_TILE(8);
            }
        for (; t + 4 <= B; t += 4)
            for (int i = r0; i < iend; i += 4) {
                const int8_t* wb = w->q + (size_t)i * n;
                const float* sb = w->s + (size_t)i * ng;
                Q8_TILE(4);
            }
        for (; t < B; t++) matmul_rows_q8(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ng, w, n, r0, iend);
    }
#endif
    if (iend < r1)
        for (int t = 0; t < B; t++) matmul_rows_q8_scalar(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ng, w, n, iend, r1);
}

static void matmul_rows(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    if (w->bits == 4) matmul_rows_q4(out, xq, xs, w, n, r0, r1);
    else matmul_rows_q8(out, xq, xs, w, n, r0, r1);
}

// ----------------------------------------------------------------------------
// Hybrid models (Qwen3.5 / Qwen3.8): Gated DeltaNet linear attention + gated full attention
static inline int hyb_full(const Config* p, int l) { return (l + 1) % p->full_interval == 0; }
static inline int hyb_fidx(const Config* p, int l) { return (l + 1) / p->full_interval - 1; } // full layer index
static inline int hyb_lidx(const Config* p, int l) { return l - (l + 1) / p->full_interval; }  // linear layer index
static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }
// per-head RMSNorm (weights stored as 1 + w) and rotate-half RoPE on the first rot dims
static inline void hyb_norm_rope(float* v, const float* w, int hs, int rot, const float* rc, const float* rs) {
    float ss = 0.0f;
    for (int i = 0; i < hs; i++) ss += v[i] * v[i];
    const float inv = 1.0f / sqrtf(ss / hs + NORM_EPS);
    for (int i = 0; i < hs; i++) v[i] = w[i] * (v[i] * inv);
    const int h2 = rot / 2;
    for (int i = 0; i < h2; i++) {
        float x1 = v[i], x2 = v[i + h2];
        v[i] = x1 * rc[i] - x2 * rs[i];
        v[i + h2] = x2 * rc[i] + x1 * rs[i];
    }
}
// conv / activations of fused input row r of linear layer li for one token (in place)
static inline void hyb_in_row(const Config* p, const TransformerWeights* w, float* conv_state, int li, int r, float* val) {
    const int K = p->conv_k, vd = p->lin_nv * p->lin_dv;
    if (r < p->conv_dim) { // causal depthwise conv over the last K inputs, then SiLU
        float* cs = conv_state + ((size_t)li * p->conv_dim + r) * (K - 1);
        const float* cw = w->conv_w + ((size_t)li * p->conv_dim + r) * K;
        float y = cw[K - 1] * *val;
        for (int j = 0; j < K - 1; j++) y += cw[j] * cs[j];
        for (int j = 0; j < K - 2; j++) cs[j] = cs[j + 1];
        cs[K - 2] = *val;
        *val = silu_f(y);
    } else if (r >= p->conv_dim + vd && r < p->conv_dim + vd + p->lin_nv) { // b -> beta
        *val = 1.0f / (1.0f + expf(-*val));
    } else if (r >= p->conv_dim + vd + p->lin_nv) { // a -> decay = exp(-exp(A_log) * softplus(a + dt_bias))
        int h = r - p->conv_dim - vd - p->lin_nv;
        float a = *val + w->dt_bias[(size_t)li * p->lin_nv + h];
        float sp = a > 20.0f ? a : log1pf(expf(a));
        *val = expf(w->A[(size_t)li * p->lin_nv + h] * sp);
    }
}
#define HYB_ROWS 32
static int HYB_BATCH = 0; // hybrid models: batched prefill available // state rows (value dims) per recurrence item
/* One token of the gated delta rule for item (head h, state rows c0..c0+HYB_ROWS): the state of a
 * head is stored transposed, row c = column c of S (dk values), so every update is contiguous.
 * lin: the fused input of the token (q, k, v after conv, z, beta, decay). */
static inline void hyb_delta(const Config* p, float* S, const float* lin, int h, int c0, float* out) {
    const int nk = p->lin_nk, nv = p->lin_nv, dk = p->lin_dk, dv = p->lin_dv, vd = nv * dv;
    const int kh = h / (nv / nk);
    const float* q = lin + (size_t)kh * dk;
    const float* k = lin + (size_t)nk * dk + (size_t)kh * dk;
    const float* v = lin + (size_t)2 * nk * dk + (size_t)h * dv;
    const float beta = lin[p->conv_dim + vd + h], decay = lin[p->conv_dim + vd + nv + h];
    float qq[256], kk[256], sq = 0.0f, sk = 0.0f;
    for (int i = 0; i < dk; i++) { sq += q[i] * q[i]; sk += k[i] * k[i]; }
    const float qn = 1.0f / sqrtf(sq + 1e-6f) / sqrtf((float)dk), kn = 1.0f / sqrtf(sk + 1e-6f);
    for (int i = 0; i < dk; i++) { qq[i] = q[i] * qn; kk[i] = k[i] * kn; }
    for (int c = c0; c < c0 + HYB_ROWS && c < dv; c++) {
        float* row = S + (size_t)(c - c0) * dk;
        float kvm = 0.0f;
        for (int i = 0; i < dk; i++) { row[i] *= decay; kvm += row[i] * kk[i]; }
        const float d = (v[c] - kvm) * beta;
        float o = 0.0f;
        for (int i = 0; i < dk; i++) { row[i] += kk[i] * d; o += row[i] * qq[i]; }
        out[(size_t)h * dv + c] = o;
    }
}
static inline float* hyb_state(const Config* p, const RunState* s, int li, int h, int c0) {
    return s->lin_S + (((size_t)li * p->lin_nv + h) * p->lin_dv + c0) * p->lin_dk;
}
// gated RMSNorm of head h of the recurrence output (weight * norm(o) * silu(z)) into y
static inline void hyb_gated_norm(const Config* p, const TransformerWeights* w, int li, const float* o, const float* z, float* y) {
    const int dv = p->lin_dv;
    float ss = 0.0f;
    for (int c = 0; c < dv; c++) ss += o[c] * o[c];
    const float inv = 1.0f / sqrtf(ss / dv + NORM_EPS);
    const float* nw = w->lnorm + (size_t)li * dv;
    for (int c = 0; c < dv; c++) y[c] = nw[c] * (o[c] * inv) * silu_f(z[c]);
}

/* Decode step of a hybrid model (one token). Synchronization is kept simple (global barriers between
 * phases); the recurrence items are a fixed function of the thread, so every thread keeps its slice of
 * the recurrent state (~16 KB per layer) in its own L2. */
static void forward_hybrid_thread(int tid) {
    Transformer* transformer = P.t;
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    ThreadBuf* tb = &P.tb[tid];
    const int pos = P.pos, dim = p->dim, kv_dim = p->kv_dim, q_dim = p->q_dim, hidden_dim = p->hidden_dim;
    const int hs = p->head_dim, n_heads = p->n_heads, kv_mul = n_heads / p->n_kv_heads, rot = p->rot_dim > 0 ? p->rot_dim : hs;
    const int nv = p->lin_nv, dv = p->lin_dv, vd = nv * dv, nitems = nv * ((dv + HYB_ROWS - 1) / HYB_ROWS);
    const int C = (pos + 1) / 64 < 1 ? 1 : ((pos + 1) / 64 < P.C ? (pos + 1) / 64 : P.C);
    const int T = pos + 1;
    const float att_scale = 1.0f / sqrtf((float)hs);
    float* x = s->x;
    const float* emb = w->token_embedding_table + (size_t)P.token * dim;
    const int8_t* nq = NB[tb->node].q;
    const float* ns = NB[tb->node].s;
    int r0, r1;

    if (tb->ltid == 0) {
        struct WorkCounter* wc = P.work[tb->node][(P.seq + 1) & 1];
        for (int k = 0; k < WORK_PHASES * P_max_layers + 1; k++) atomic_store_explicit(&wc[k].v, 0, memory_order_relaxed);
    }
    if (pos == 0) { // new sequence: zero the recurrent and conv states (each thread its own items)
        const int nlin = p->n_layers - p->n_attn;
        for (int li = 0; li < nlin; li++)
            for (int it = tid; it < nitems; it += P.nth) {
                int h = it / ((dv + HYB_ROWS - 1) / HYB_ROWS), c0 = (it % ((dv + HYB_ROWS - 1) / HYB_ROWS)) * HYB_ROWS;
                int rows = c0 + HYB_ROWS < dv ? HYB_ROWS : dv - c0;
                memset(hyb_state(p, s, li, h, c0), 0, (size_t)rows * p->lin_dk * sizeof(float));
            }
        size_t cn = (size_t)nlin * p->conv_dim * (p->conv_k - 1);
        memset(s->conv_state + cn * tid / P.nth, 0, (cn * (tid + 1) / P.nth - cn * tid / P.nth) * sizeof(float));
        barrier(tb);
    }
    for (int j = 0; j < rot / 2; j++) {
        double val = pos * P.rope_freq[j];
        tb->rope_c[j] = (float)cos(val);
        tb->rope_s[j] = (float)sin(val);
    }

    for (int l = 0; l < p->n_layers; l++) {
        const float* xin = (l == 0) ? emb : x;
        PROF(tid, PH_NORM);
        node_norm_quant(tb, xin, w->rms_att_weight + (size_t)l * dim, dim);
        if (hyb_full(p, l)) {
            const int fi = hyb_fidx(p, l);
            const size_t loff = (size_t)fi * p->seq_len * kv_dim;
            float* vc = s->value_cache + loff + (size_t)pos * hs;
            PROF(tid, PH_QKV);
            while (grab_rows(WORK_PHASES * l + 0, q_dim, tb, &r0, &r1)) matmul_rows(s->q, nq, ns, w->wq + fi, dim, r0, r1);
            while (grab_rows(WORK_PHASES * l + 6, q_dim, tb, &r0, &r1)) matmul_rows(s->gate, nq, ns, w->wgate + fi, dim, r0, r1);
            while (grab_rows(WORK_PHASES * l + 1, kv_dim, tb, &r0, &r1)) matmul_rows(s->k, nq, ns, w->wk + fi, dim, r0, r1);
            while (grab_rows(WORK_PHASES * l + 2, kv_dim, tb, &r0, &r1)) {
                matmul_rows(s->v, nq, ns, w->wv + fi, dim, r0, r1);
                for (int i = r0; i < r1; i++) vc[(size_t)(i / hs) * p->seq_len * hs + i % hs] = s->v[i];
            }
            barrier(tb);
            for (int task = tid; task < n_heads + p->n_kv_heads; task += P.nth) {
                if (task < n_heads) {
                    hyb_norm_rope(s->q + (size_t)task * hs, w->q_norm + (size_t)fi * hs, hs, rot, tb->rope_c, tb->rope_s);
                } else {
                    int g = task - n_heads;
                    float* kv = s->k + (size_t)g * hs;
                    hyb_norm_rope(kv, w->k_norm + (size_t)fi * hs, hs, rot, tb->rope_c, tb->rope_s);
                    memcpy(s->key_cache + loff + (size_t)g * p->seq_len * hs + (size_t)pos * hs, kv, hs * sizeof(float));
                }
            }
            barrier(tb);
            PROF(tid, PH_ATT);
            // attention items (kv group, context chunk, query-head subset) over all threads
            int HS = 1;
            while (HS < kv_mul && p->n_kv_heads * C * HS * 2 <= P.nth && kv_mul % (HS * 2) == 0) HS *= 2;
            const int hper = kv_mul / HS;
            for (int item = tid; item < p->n_kv_heads * C * HS; item += P.nth) {
                int g = item / (C * HS), c = (item / HS) % C, hsub = item % HS;
                int t0 = (int)((long)T * c / C), t1 = (int)((long)T * (c + 1) / C);
                const float* kb = s->key_cache + loff + (size_t)g * p->seq_len * hs;
                const float* vb = s->value_cache + loff + (size_t)g * p->seq_len * hs;
                for (int r = 0; r < hper; r++) {
                    int h = g * kv_mul + hsub * hper + r;
                    const float* q = s->q + (size_t)h * hs;
                    float* att = s->att + (size_t)h * p->seq_len;
                    float m = -FLT_MAX;
                    for (int t = t0; t < t1; t++) {
                        const float* k = kb + (size_t)t * hs;
                        float score = 0.0f;
                        for (int i = 0; i < hs; i++) score += q[i] * k[i];
                        score *= att_scale;
                        att[t] = score;
                        m = score > m ? score : m;
                    }
                    float* pr = P.part + ((size_t)h * C + c) * (hs + 2);
                    float sm = 0.0f;
                    float* acc = pr + 2;
                    memset(acc, 0, hs * sizeof(float));
                    for (int t = t0; t < t1; t++) {
                        float e = expf(att[t] - m);
                        sm += e;
                        const float* v = vb + (size_t)t * hs;
                        for (int i = 0; i < hs; i++) acc[i] += e * v[i];
                    }
                    pr[0] = m;
                    pr[1] = sm;
                }
            }
            barrier(tb);
            PROF(tid, PH_MERGE);
            // merge every head (each node for itself), apply the sigmoid gate, quantize into the node buffer
            int h0 = (int)((long)n_heads * tb->ltid / tb->nthn), h1 = (int)((long)n_heads * (tb->ltid + 1) / tb->nthn);
            for (int h = h0; h < h1; h++) {
                float* xb = tb->xb + (size_t)h * hs;
                const float* ph = P.part + (size_t)h * C * (hs + 2);
                float M = -FLT_MAX, den = 0.0f;
                for (int c = 0; c < C; c++) if (ph[(size_t)c * (hs + 2) + 1] > 0.0f && ph[(size_t)c * (hs + 2)] > M) M = ph[(size_t)c * (hs + 2)];
                memset(xb, 0, hs * sizeof(float));
                for (int c = 0; c < C; c++) {
                    const float* pr = ph + (size_t)c * (hs + 2);
                    if (pr[1] <= 0.0f) continue;
                    float f = expf(pr[0] - M);
                    den += f * pr[1];
                    for (int i = 0; i < hs; i++) xb[i] += f * pr[2 + i];
                }
                const float inv = 1.0f / den;
                const float* gt = s->gate + (size_t)h * hs;
                for (int i = 0; i < hs; i++) xb[i] *= inv * (p->attn_gate ? 1.0f / (1.0f + expf(-gt[i])) : 1.0f);
                quantize_vec(NB[tb->node].q + (size_t)h * hs, NB[tb->node].s + h * hs / GS, xb, hs);
            }
            node_barrier(tb);
            PROF(tid, PH_WO);
            while (grab_rows(WORK_PHASES * l + 3, dim, tb, &r0, &r1)) {
                matmul_rows(s->xb2, nq, ns, w->wo + fi, q_dim, r0, r1);
                for (int i = r0; i < r1; i++) x[i] = xin[i] + s->xb2[i];
            }
            barrier(tb);
        } else {
            const int li = hyb_lidx(p, l);
            PROF(tid, PH_QKV);
            while (grab_rows(WORK_PHASES * l + 0, p->lin_in, tb, &r0, &r1)) {
                matmul_rows(s->lin, nq, ns, w->w_in + li, dim, r0, r1);
                for (int r = r0; r < r1; r++) hyb_in_row(p, w, s->conv_state, li, r, s->lin + r);
            }
            barrier(tb);
            PROF(tid, PH_ATT);
            for (int it = tid; it < nitems; it += P.nth) {
                int h = it / ((dv + HYB_ROWS - 1) / HYB_ROWS), c0 = (it % ((dv + HYB_ROWS - 1) / HYB_ROWS)) * HYB_ROWS;
                hyb_delta(p, hyb_state(p, s, li, h, c0), s->lin, h, c0, s->lin_o);
            }
            barrier(tb);
            PROF(tid, PH_MERGE);
            // gated norm of every head (each node for itself), quantized into the node buffer
            for (int h = tb->ltid; h < nv; h += tb->nthn) {
                float* y = tb->xb;
                hyb_gated_norm(p, w, li, s->lin_o + (size_t)h * dv, s->lin + p->conv_dim + (size_t)h * dv, y);
                quantize_vec(NB[tb->node].q + (size_t)h * dv, NB[tb->node].s + h * dv / GS, y, dv);
            }
            node_barrier(tb);
            PROF(tid, PH_WO);
            while (grab_rows(WORK_PHASES * l + 3, dim, tb, &r0, &r1)) {
                matmul_rows(s->xb2, nq, ns, w->w_out + li, vd, r0, r1);
                for (int i = r0; i < r1; i++) x[i] = xin[i] + s->xb2[i];
            }
            barrier(tb);
        }
        PROF(tid, PH_FNORM);
        node_norm_quant(tb, x, w->rms_ffn_weight + (size_t)l * dim, dim);
        PROF(tid, PH_W13);
        while (grab_rows(WORK_PHASES * l + 4, hidden_dim, tb, &r0, &r1)) {
            matmul_rows(s->hb, nq, ns, w->w1 + l, dim, r0, r1);
            matmul_rows(s->hb2, nq, ns, w->w3 + l, dim, r0, r1);
            for (int i = r0; i < r1; i++) s->hb[i] = silu_f(s->hb[i]) * s->hb2[i];
        }
        barrier(tb);
        PROF(tid, PH_QH);
        node_quant(tb, s->hb, hidden_dim);
        PROF(tid, PH_W2);
        while (grab_rows(WORK_PHASES * l + 5, dim, tb, &r0, &r1)) {
            matmul_rows(s->xb2, nq, ns, w->w2 + l, hidden_dim, r0, r1);
            for (int i = r0; i < r1; i++) x[i] += s->xb2[i];
        }
        barrier(tb);
    }
    PROF(tid, PH_CLS);
    node_norm_quant(tb, x, w->rms_final_weight, dim);
    while (grab_rows(WORK_PHASES * p->n_layers, p->vocab_size, tb, &r0, &r1)) matmul_rows(s->logits, nq, ns, w->wcls, dim, r0, r1);
    barrier(tb);
    PROF(tid, PH_IDLE);
}

static void forward_thread(int tid) {
    if (P.t->config.hybrid) { forward_hybrid_thread(tid); return; }
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
    // head locality: the q/k/v rows a node computes are exactly the heads of its kv groups (true when the
    // node split falls on kv group boundaries), so q/k/v, qk-norm and attention only need node barriers
    int g0, g1, hl = 1;
    for (int k = 0; k < NN; k++) {
        int a, b, lo, hi;
        node_groups(p->n_kv_heads, k, &a, &b);
        node_rows(q_dim, k, &lo, &hi);
        if (lo != a * kv_mul * head_size || hi != b * kv_mul * head_size) hl = 0;
        node_rows(kv_dim, k, &lo, &hi);
        if (lo != a * head_size || hi != b * head_size) hl = 0;
    }
    node_groups(p->n_kv_heads, tb->node, &g0, &g1);

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
        if (l == 0) pf_first(tb, w->wq, dim, q_dim, 0); // later layers: queued at the previous layer's last barrier
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
        if (hl) node_barrier(tb); else barrier(tb);
        if (qkn) { // per-head q/k RMSNorm + RoPE (needs whole heads), k into the cache
            // heads of the node's kv groups: q heads [g0*kv_mul, g1*kv_mul), k heads [g0, g1)
            int t0 = hl ? g0 * kv_mul : 0, tq = hl ? (g1 - g0) * kv_mul : n_heads, tk = hl ? g1 - g0 : p->n_kv_heads;
            int me = hl ? tb->ltid : tid, nme = hl ? tb->nthn : P.nth;
            for (int task = me; task < tq + tk; task += nme) {
                if (task < tq) {
                    head_norm_rope(s->q + (size_t)(t0 + task) * head_size, w->q_norm + (size_t)l * head_size, head_size, tb->rope_c, tb->rope_s);
                } else {
                    int g = (hl ? g0 : 0) + task - tq;
                    float* kv = s->k + (size_t)g * head_size;
                    head_norm_rope(kv, w->k_norm + (size_t)l * head_size, head_size, tb->rope_c, tb->rope_s);
                    memcpy(kc + (size_t)g * p->seq_len * head_size, kv, head_size * sizeof(float));
                }
            }
            if (hl) node_barrier(tb); else barrier(tb);
        }

        PROF(tid, PH_ATT);
        // attention: work items are (kv head group, chunk of the context); the kv_mul query heads
        // sharing a kv head are processed together so K and V are read once. Merged with online softmax.
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
        if (hl) node_barrier(tb); else barrier(tb);

        PROF(tid, PH_MERGE);
        // merge the partials of this thread's slice of heads and quantize them into the node buffer
        // (head_size is a multiple of GS, so a head is a whole number of quantization groups).
        // Head-local: a node merges only its own heads and writes them into the buffers of all nodes.
        int hb0 = hl ? g0 * kv_mul : 0, nhm = hl ? (g1 - g0) * kv_mul : n_heads;
        int h0 = hb0 + (int)((long)nhm * tb->ltid / tb->nthn), h1 = hb0 + (int)((long)nhm * (tb->ltid + 1) / tb->nthn);
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
            // into the second slot of the node buffers: the other node may still be reading the first
            // (its q/k/v input) when this node writes its heads there
            int8_t* hq = NB[tb->node].q + P_qmax + (size_t)h * head_size;
            float* hsc = NB[tb->node].s + P_qmax / GS + h * head_size / GS;
            quantize_vec(hq, hsc, xb, head_size);
            if (hl)
                for (int k = 0; k < NN; k++)
                    if (k != tb->node) {
                        memcpy(NB[k].q + P_qmax + (size_t)h * head_size, hq, head_size);
                        memcpy(NB[k].s + P_qmax / GS + h * head_size / GS, hsc, head_size / GS * sizeof(float));
                    }
        }
        pf_first(tb, w->wo + l, q_dim, dim, 0);
        if (hl && NN > 1) barrier(tb); else node_barrier(tb);

        PROF(tid, PH_WO);
        // output projection + residual
        while (grab_rows(WORK_PHASES * l + 3, dim, tb, &r0, &r1)) {
            matmul_rows(s->xb2, nq + P_qmax, ns + P_qmax / GS, w->wo + l, q_dim, r0, r1);
            for (int i = r0; i < r1; i++) { x[i] = xin[i] + s->xb2[i]; }
        }
        if (p->n_experts > 0) pf_first(tb, w->wg_node[tb->node] + l, dim, p->n_experts, 1);
        else { pf_first(tb, w->w1 + l, dim, hidden_dim, 0); pf_first(tb, w->w3 + l, dim, hidden_dim, 0); }
        barrier(tb);

        PROF(tid, PH_FNORM);
        // ffn rmsnorm + quantize (cooperative, per node)
        node_norm_quant(tb, x, w->rms_ffn_weight + l * dim, dim);

        PROF(tid, PH_W13);
        if (p->n_experts > 0) { // ---- MoE FFN
            const int E = p->n_experts, A = p->n_active, mh = p->moe_hidden;
            // every node computes all router logits from its own copy of the router (node barrier only)
            float* rl = NB[tb->node].router;
            while (grab_virtual(WORK_PHASES * l + 6, E, tb, &r0, &r1)) matmul_rows(rl, nq, ns, w->wg_node[tb->node] + l, dim, r0, r1);
            node_barrier(tb);
            int sel[MOE_MAX_ACTIVE];
            float wt[MOE_MAX_ACTIVE];
            moe_route(rl, E, A, p->norm_topk, sel, wt); // same result in every thread
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
            { // w2 rows of this thread's first ticket, for all selected experts
                int lo2, hi2, b0, b1;
                node_rows(dim, tb->node, &lo2, &hi2);
                if (ticket_range(tb->ltid, hi2 - lo2, tb->nthn, &b0, &b1))
                    for (int j = 0; j < A; j++) {
                        QuantizedTensor e2 = expert_view(w->w2 + l, sel[j], dim, mh);
                        pf_add(tb, &e2, mh, lo2 + b0, lo2 + b1);
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
            pf_first(tb, w->w2 + l, hidden_dim, dim, 0);
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
        if (l + 1 < p->n_layers) pf_first(tb, w->wq + l + 1, dim, q_dim, 0);
        else pf_first(tb, w->wcls, dim, p->vocab_size, 0);
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

static void sample_thread(int tid);
// RUNQ_BARBENCH=1: measure the synchronization primitives alone (us per barrier)
static double now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3; }
static void barrier_bench_thread(int tid) {
    ThreadBuf* tb = &P.tb[tid];
    const int R = 20000;
    for (int i = 0; i < 2000; i++) barrier(tb); // warm-up
    double t0 = now_us();
    for (int i = 0; i < R; i++) barrier(tb);
    double t1 = now_us();
    for (int i = 0; i < R; i++) node_barrier(tb);
    barrier(tb);
    double t2 = now_us();
    if (tid == 0) fprintf(stderr, "[barrier bench] %d threads: global %.2f us, node %.2f us\n", P.nth, (t1 - t0) / R, (t2 - t1) / R);
}
static void* pool_worker(void* arg) {
    int tid = (int)(intptr_t)arg;
    pin_thread(tid);
    int seen = 0;
    for (;;) {
        seen = wait_for_work(seen);
        if (atomic_load_explicit(&P.quit, memory_order_acquire)) break;
        if (P.mode == 4) sample_thread(tid);
        else if (P.mode == 3) barrier_bench_thread(tid);
        else if (P.mode == 1) forward_batch_thread(tid);
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
    const int rdim = p->rot_dim > 0 ? p->rot_dim : head_size; // partial RoPE: frequencies over the rotated dims
    for (int j = 0; j < head_size / 2; j++) {
        double freq = j < rdim / 2 ? 1.0 / pow(ROPE_THETA, (2.0 * j) / rdim) : 0.0;
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
    P_qmax = qmax;
    if (p->n_experts > 256 || (p->n_experts > 0 && SPEC_MAX_TOK * p->n_experts > 4096)) { fprintf(stderr, "at most 256 experts are supported\n"); exit(EXIT_FAILURE); }
    if (p->n_active > MOE_MAX_ACTIVE) { fprintf(stderr, "at most %d active experts are supported\n", MOE_MAX_ACTIVE); exit(EXIT_FAILURE); }
    P.tb = aligned_alloc(128, nth * sizeof(ThreadBuf));
    memset(P.tb, 0, nth * sizeof(ThreadBuf));
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
        if (nthn > 512) { fprintf(stderr, "too many threads per node\n"); exit(EXIT_FAILURE); }
        size_t syb = (sizeof(struct NodeSync) + 4095) & ~(size_t)4095;
        NB[k].sy = aligned_alloc(4096, syb);
#if defined(__linux__)
        if (NN > 1) bind_range(NB[k].sy, syb, NODE_ID[k]);
#endif
        memset(NB[k].sy, 0, syb);
        NB[k].router = aligned_alloc(4096, 4096 * sizeof(float));
#if defined(__linux__)
        if (NN > 1) bind_range(NB[k].router, 4096 * sizeof(float), NODE_ID[k]);
#endif
        memset(NB[k].router, 0, 4096 * sizeof(float));
        if (p->n_experts > 0) {
            size_t gb = ((size_t)SPEC_MAX_TOK * p->n_active * p->dim + 4095) & ~(size_t)4095;
            size_t gsb = ((size_t)SPEC_MAX_TOK * p->n_active * (p->dim / GS) * sizeof(float) + 4095) & ~(size_t)4095;
            NB[k].gq = aligned_alloc(4096, gb);
            NB[k].gs = aligned_alloc(4096, gsb);
#if defined(__linux__)
            if (NN > 1) { bind_range(NB[k].gq, gb, NODE_ID[k]); bind_range(NB[k].gs, gsb, NODE_ID[k]); }
#endif
            memset(NB[k].gq, 0, gb);
            memset(NB[k].gs, 0, gsb);
        }
    }
    for (int k = 0; k < 8; k++) atomic_store(&P.xflag[k].c, 0);
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
    if (getenv("RUNQ_BARBENCH")) {
        P.mode = 3;
        wake_workers();
        barrier_bench_thread(0);
        P.mode = 0;
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
    if (transformer->config.hybrid) { // the recurrent state can only advance, or restart at 0
        if (pos != 0 && pos != transformer->state.next_pos) {
            fprintf(stderr, "hybrid model: position %d out of order (state at %d)\n", pos, transformer->state.next_pos);
            exit(EXIT_FAILURE);
        }
        transformer->state.next_pos = pos + 1;
    }
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
    float *gate, *lin, *lino; // hybrid: (B, q_dim) attention gates, (B, lin_in) fused inputs, (B, nv dv) recurrence outputs
    float* mpart;     // multi-token step: attention partials (token, head, context chunk, head_size + 2)
    size_t mpart_n;   // its capacity in floats
    float* logits_all; // when set: (B, vocab) logits of every position (speculative verification)
    // MoE prefill (sized for BATCH_MAX tokens x n_active slots)
    float* router;     // (B, n_experts)
    int *moe_sel, *moe_cnt, *moe_fill, *moe_tok;
    int *moe_ue, moe_U;  // the experts used by the block (ascending), their number
    struct WorkCounter* moe_ctr; // per node: dynamic chunks of the w1/w3 phase
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

/* Per-node copies of the quantized inputs of the batched matmuls: every node normalizes / quantizes
 * all tokens itself into memory bound to it, so the 96 threads of a socket never stream their
 * inputs from the other socket (that made 2 sockets slower than 1 for prefill). */
static struct {
    int8_t* xq; float* xs;   // (B, max(dim, hidden_dim, q_dim)) block inputs
    int8_t* gxq; float* gxs; // MoE: gathered ffn inputs per slot
    int8_t* gq; float* gqs;  // MoE: quantized hidden activations per slot
} BN[MAX_NODES];

/* Importance statistics (imatrix, `-m imatrix`): the mean x^2 of every input channel of every matmul,
 * collected on the batched path. They weight the rounding error when a model is quantized to Q4
 * (scripts/export_hf_q8.py --imatrix). Dense: att (L, dim), wo (L, q_dim), ffn (L, dim), w2
 * (L, hidden); MoE: ffn (L, E, dim) and w2 (L, E, moe_hidden) per expert, over the tokens routed to it. */
static int IMX_WO_W = 0; // width of a wo row of the statistics (q_dim, or max(q_dim, nv dv) for hybrid models)
static struct {
    int on;
    double *att, *wo, *ffn, *w2;
    long long ntok;   // tokens seen
    long long* nexp;  // MoE: (L, E) tokens routed to each expert
} IMX;
// channels split among all threads; X (B, ld), optionally rmsnorm'ed with weight nw first
static void imx_dense(ThreadBuf* tb, double* acc, const float* X, int ld, int B, int n, const float* nw) {
    int tid = (int)(tb - P.tb), c0, c1;
    c0 = (int)((long)n * tid / P.nth);
    c1 = (int)((long)n * (tid + 1) / P.nth);
    for (int b = 0; b < B; b++) {
        const float* x = X + (size_t)b * ld;
        float inv = 1.0f;
        if (nw) {
            float ss = 0.0f;
            for (int i = 0; i < n; i++) ss += x[i] * x[i];
            inv = 1.0f / sqrtf(ss / n + NORM_EPS);
        }
        for (int c = c0; c < c1; c++) {
            float v = x[c] * (nw ? inv * nw[c] : 1.0f);
            acc[c] += (double)v * v;
        }
    }
    barrier(tb);
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
    const int tid = (int)(tb - P.tb);
    const int dim = p->dim, E = p->n_experts, A = p->n_active, mh = p->moe_hidden;
    const int ngd = dim / GS, ngm = mh / GS;
    int r0, r1;
    int8_t* XQ = BN[tb->node].xq; float* XS = BN[tb->node].xs;
    int8_t* GXQ = BN[tb->node].gxq; float* GXS = BN[tb->node].gxs;
    int8_t* GQ = BN[tb->node].gq; float* GQS = BN[tb->node].gqs;
    // router logits for all tokens
    part_range_node(E, tb, &r0, &r1);
    matmul_rows_batch(BT.router, E, XQ, XS, B, w->wg + l, dim, r0, r1);
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
            }
        BT.moe_U = 0;
        for (int e = 0; e < E; e++) if (cnt[e + 1] > cnt[e]) BT.moe_ue[BT.moe_U++] = e;
        for (int k = 0; k < NN; k++) atomic_store_explicit(&BT.moe_ctr[k].v, 0, memory_order_relaxed);
    }
    barrier(tb);
    const int* off = BT.moe_cnt;
    { // gather the quantized inputs of the slots (all threads)
        int s0, s1;
        tok_range(off[E], tb->ltid, tb->nthn, &s0, &s1); // every node gathers all slots into its buffer
        for (int sl = s0; sl < s1; sl++) {
            int b = BT.moe_tok[sl];
            memcpy(GXQ + (size_t)sl * dim, XQ + (size_t)b * dim, dim);
            memcpy(GXS + (size_t)sl * ngd, XS + (size_t)b * ngd, ngd * sizeof(float));
        }
    }
    barrier(tb);
    // w1/w3 + SwiGLU: this node's half of every used expert, in dynamic 16-row chunks (the cost of a
    // chunk is proportional to the expert's token count, so a static split would be badly unbalanced)
    int lo, hi;
    node_rows(mh, tb->node, &lo, &hi);
    const int half = hi - lo, U = BT.moe_U;
    const int cpe = (half + 15) / 16; // chunks per expert
    for (;;) {
        int tk = atomic_fetch_add_explicit(&BT.moe_ctr[tb->node].v, 1, memory_order_relaxed);
        if (tk >= U * cpe) break;
        int u = tk / cpe, e = BT.moe_ue[u], c = tk % cpe;
        int n = off[e + 1] - off[e];
        {
            int a0 = lo + c * 16, a1 = lo + (c * 16 + 16 < half ? c * 16 + 16 : half);
            QuantizedTensor e1 = expert_view(w->w1 + l, e, mh, dim), e3 = expert_view(w->w3 + l, e, mh, dim);
            float* h1 = BT.gh + (size_t)off[e] * mh;
            float* h3 = BT.gh3 + (size_t)off[e] * mh;
            matmul_rows_batch(h1, mh, GXQ + (size_t)off[e] * dim, GXS + (size_t)off[e] * ngd, n, &e1, dim, a0, a1);
            matmul_rows_batch(h3, mh, GXQ + (size_t)off[e] * dim, GXS + (size_t)off[e] * ngd, n, &e3, dim, a0, a1);
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
    }
    barrier(tb);
    if (IMX.on) { // per expert: its ffn inputs (dequantized) and its w2 inputs, experts split among threads
        for (int e = tid; e < E; e += P.nth) {
            double* af = IMX.ffn + ((size_t)l * E + e) * dim;
            double* a2 = IMX.w2 + ((size_t)l * E + e) * mh;
            for (int sl = off[e]; sl < off[e + 1]; sl++) {
                int b = BT.moe_tok[sl];
                for (int c = 0; c < dim; c++) { double v = (double)XQ[(size_t)b * dim + c] * XS[(size_t)b * ngd + c / GS]; af[c] += v * v; }
                for (int c = 0; c < mh; c++) { double v = BT.gh[(size_t)sl * mh + c] / (BT.moe_w[sl] > 0 ? BT.moe_w[sl] : 1.0f); a2[c] += v * v; }
            }
            IMX.nexp[(size_t)l * E + e] += off[e + 1] - off[e];
        }
        barrier(tb);
    }
    // quantize the hidden activations of every slot
    const int S = off[E];
    int s0, s1;
    tok_range(S, tb->ltid, tb->nthn, &s0, &s1); // every node quantizes all slots into its buffer
    for (int sl = s0; sl < s1; sl++)
        quantize_vec(GQ + (size_t)sl * mh, GQS + (size_t)sl * ngm, BT.gh + (size_t)sl * mh, mh);
    barrier(tb);
    // w2: this node's rows of dim, for every used expert, then add every slot into its token
    part_range_node(dim, tb, &r0, &r1);
    if (r0 < r1) {
        for (int u = 0; u < BT.moe_U; u++) {
            int e = BT.moe_ue[u], n = off[e + 1] - off[e];
            QuantizedTensor e2 = expert_view(w->w2 + l, e, dim, mh);
            matmul_rows_batch(BT.gout + (size_t)off[e] * dim, dim, GQ + (size_t)off[e] * mh, GQS + (size_t)off[e] * ngm, n, &e2, mh, r0, r1);
        }
        for (int sl = 0; sl < S; sl++) {
            float* x = BT.x + (size_t)BT.moe_tok[sl] * dim;
            const float* o = BT.gout + (size_t)sl * dim;
            for (int i = r0; i < r1; i++) x[i] += o[i];
        }
    }
}

/* Batched prefill of a hybrid model: the matmuls see all B tokens at once; the conv runs per channel
 * and the gated delta rule per state slice, each sequentially over the tokens of the block. */
static void forward_hybrid_batch_thread(int tid) {
    Transformer* transformer = P.t;
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    ThreadBuf* tb = &P.tb[tid];
    const int nth = P.nth, B = BT.B, pos0 = BT.pos0;
    const int dim = p->dim, kv_dim = p->kv_dim, q_dim = p->q_dim, hidden_dim = p->hidden_dim;
    const int hs = p->head_dim, hs2 = hs / 2, n_heads = p->n_heads, kv_mul = n_heads / p->n_kv_heads;
    const int rot = p->rot_dim > 0 ? p->rot_dim : hs;
    const int nv = p->lin_nv, dv = p->lin_dv, vd = nv * dv, LI = p->lin_in, ipb = (dv + HYB_ROWS - 1) / HYB_ROWS, nitems = nv * ipb;
    const int ngd = dim / GS, ngh = hidden_dim / GS, ngq = q_dim / GS, ngv = vd / GS;
    const float att_scale = 1.0f / sqrtf((float)hs);
    int8_t* XQ = BN[tb->node].xq;
    float* XS = BN[tb->node].xs;
    int r0, r1, b0, b1, n0, n1;
    tok_range(B, tid, nth, &b0, &b1);
    tok_range(B, tb->ltid, tb->nthn, &n0, &n1);
    for (int b = b0; b < b1; b++) {
        memcpy(BT.x + (size_t)b * dim, w->token_embedding_table + (size_t)BT.tok[b] * dim, dim * sizeof(float));
        for (int j = 0; j < rot / 2; j++) {
            double val = (double)(pos0 + b) * P.rope_freq[j];
            BT.rope_c[b * hs2 + j] = (float)cos(val);
            BT.rope_s[b * hs2 + j] = (float)sin(val);
        }
    }
    if (pos0 == 0) { // new sequence: zero the states (same item partition as the recurrence)
        const int nlin = p->n_layers - p->n_attn;
        for (int li = 0; li < nlin; li++)
            for (int it = tid; it < nitems; it += nth) {
                int h = it / ipb, c0 = (it % ipb) * HYB_ROWS, rows = c0 + HYB_ROWS < dv ? HYB_ROWS : dv - c0;
                memset(hyb_state(p, s, li, h, c0), 0, (size_t)rows * p->lin_dk * sizeof(float));
            }
        size_t cn = (size_t)nlin * p->conv_dim * (p->conv_k - 1);
        memset(s->conv_state + cn * tid / nth, 0, (cn * (tid + 1) / nth - cn * tid / nth) * sizeof(float));
    }
    barrier(tb);

    for (int l = 0; l < p->n_layers; l++) {
        if (IMX.on) imx_dense(tb, IMX.att + (size_t)l * dim, BT.x, dim, B, dim, w->rms_att_weight + (size_t)l * dim);
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_att_weight + (size_t)l * dim, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);
        if (hyb_full(p, l)) {
            const int fi = hyb_fidx(p, l);
            const size_t loff = (size_t)fi * p->seq_len * kv_dim;
            part_range_node(q_dim, tb, &r0, &r1);
            if (r0 < r1) {
                matmul_rows_batch(BT.q, q_dim, XQ, XS, B, w->wq + fi, dim, r0, r1);
                matmul_rows_batch(BT.gate, q_dim, XQ, XS, B, w->wgate + fi, dim, r0, r1);
            }
            part_range_node(kv_dim, tb, &r0, &r1);
            if (r0 < r1) {
                matmul_rows_batch(BT.k, kv_dim, XQ, XS, B, w->wk + fi, dim, r0, r1);
                matmul_rows_batch(BT.v, kv_dim, XQ, XS, B, w->wv + fi, dim, r0, r1);
                for (int b = 0; b < B; b++) {
                    float* vc = s->value_cache + loff + (size_t)(pos0 + b) * hs;
                    for (int i = r0; i < r1; i++) vc[(size_t)(i / hs) * p->seq_len * hs + i % hs] = BT.v[(size_t)b * kv_dim + i];
                }
            }
            barrier(tb);
            for (int task = tid; task < B * (n_heads + p->n_kv_heads); task += nth) {
                int b = task / (n_heads + p->n_kv_heads), h = task % (n_heads + p->n_kv_heads);
                const float* rc = BT.rope_c + b * hs2;
                const float* rs = BT.rope_s + b * hs2;
                if (h < n_heads) {
                    hyb_norm_rope(BT.q + (size_t)b * q_dim + (size_t)h * hs, w->q_norm + (size_t)fi * hs, hs, rot, rc, rs);
                } else {
                    int g = h - n_heads;
                    float* kv = BT.k + (size_t)b * kv_dim + (size_t)g * hs;
                    hyb_norm_rope(kv, w->k_norm + (size_t)fi * hs, hs, rot, rc, rs);
                    memcpy(s->key_cache + loff + ((size_t)g * p->seq_len + pos0 + b) * hs, kv, hs * sizeof(float));
                }
            }
            barrier(tb);
            // causal attention, items (kv group of this node, token); gate applied to the output
            float* att = BT.att + (size_t)tid * kv_mul * p->seq_len;
            int g0, g1;
            node_groups(p->n_kv_heads, tb->node, &g0, &g1);
            for (int item = tb->ltid; item < (g1 - g0) * B; item += tb->nthn) {
                int g = g0 + item % (g1 - g0), b = item / (g1 - g0);
                int T = pos0 + b + 1;
                const float* kb = s->key_cache + loff + (size_t)g * p->seq_len * hs;
                const float* vb = s->value_cache + loff + (size_t)g * p->seq_len * hs;
                const float* qb = BT.q + (size_t)b * q_dim + (size_t)g * kv_mul * hs;
                float m[KV_MUL_MAX], sum[KV_MUL_MAX];
                for (int r = 0; r < kv_mul; r++) m[r] = -FLT_MAX;
                for (int t = 0; t < T; t++) {
                    const float* k = kb + (size_t)t * hs;
                    for (int r = 0; r < kv_mul; r++) {
                        const float* q = qb + r * hs;
                        float score = 0.0f;
                        for (int i = 0; i < hs; i++) score += q[i] * k[i];
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
                float* ob = BT.xb + (size_t)b * q_dim + (size_t)g * kv_mul * hs;
                memset(ob, 0, kv_mul * hs * sizeof(float));
                for (int t = 0; t < T; t++) {
                    const float* v = vb + (size_t)t * hs;
                    for (int r = 0; r < kv_mul; r++) {
                        float a = att[(size_t)r * p->seq_len + t];
                        float* o = ob + r * hs;
                        for (int i = 0; i < hs; i++) o[i] += a * v[i];
                    }
                }
                const float* gt = BT.gate + (size_t)b * q_dim + (size_t)g * kv_mul * hs;
                for (int i = 0; i < kv_mul * hs; i++)
                    ob[i] *= sum[i / hs] * (p->attn_gate ? 1.0f / (1.0f + expf(-gt[i])) : 1.0f);
            }
            barrier(tb);
            if (IMX.on) imx_dense(tb, IMX.wo + (size_t)l * IMX_WO_W, BT.xb, q_dim, B, q_dim, NULL);
            for (int b = n0; b < n1; b++) quantize_vec(XQ + (size_t)b * q_dim, XS + (size_t)b * ngq, BT.xb + (size_t)b * q_dim, q_dim);
            barrier(tb);
            part_range_node(dim, tb, &r0, &r1);
            if (r0 < r1) {
                matmul_rows_batch(BT.xb2, dim, XQ, XS, B, w->wo + fi, q_dim, r0, r1);
                for (int b = 0; b < B; b++)
                    for (int i = r0; i < r1; i++) BT.x[(size_t)b * dim + i] += BT.xb2[(size_t)b * dim + i];
            }
            barrier(tb);
        } else {
            const int li = hyb_lidx(p, l);
            part_range_node(LI, tb, &r0, &r1);
            if (r0 < r1) matmul_rows_batch(BT.lin, LI, XQ, XS, B, w->w_in + li, dim, r0, r1);
            barrier(tb);
            { // conv / activations: channels split among all threads, tokens in order
                int c0, c1;
                tok_range(LI, tid, nth, &c0, &c1);
                for (int r = c0; r < c1; r++)
                    for (int b = 0; b < B; b++) hyb_in_row(p, w, s->conv_state, li, r, BT.lin + (size_t)b * LI + r);
            }
            barrier(tb);
            for (int it = tid; it < nitems; it += nth) {
                int h = it / ipb, c0 = (it % ipb) * HYB_ROWS;
                float* S = hyb_state(p, s, li, h, c0);
                for (int b = 0; b < B; b++) hyb_delta(p, S, BT.lin + (size_t)b * LI, h, c0, BT.lino + (size_t)b * vd);
            }
            barrier(tb);
            for (int task = tid; task < B * nv; task += nth) {
                int b = task / nv, h = task % nv;
                hyb_gated_norm(p, w, li, BT.lino + (size_t)b * vd + (size_t)h * dv, BT.lin + (size_t)b * LI + p->conv_dim + (size_t)h * dv,
                               BT.xb + (size_t)b * vd + (size_t)h * dv);
            }
            barrier(tb);
            if (IMX.on) imx_dense(tb, IMX.wo + (size_t)l * IMX_WO_W, BT.xb, vd, B, vd, NULL);
            for (int b = n0; b < n1; b++) quantize_vec(XQ + (size_t)b * vd, XS + (size_t)b * ngv, BT.xb + (size_t)b * vd, vd);
            barrier(tb);
            part_range_node(dim, tb, &r0, &r1);
            if (r0 < r1) {
                matmul_rows_batch(BT.xb2, dim, XQ, XS, B, w->w_out + li, vd, r0, r1);
                for (int b = 0; b < B; b++)
                    for (int i = r0; i < r1; i++) BT.x[(size_t)b * dim + i] += BT.xb2[(size_t)b * dim + i];
            }
            barrier(tb);
        }
        if (IMX.on) imx_dense(tb, IMX.ffn + (size_t)l * dim, BT.x, dim, B, dim, w->rms_ffn_weight + (size_t)l * dim);
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_ffn_weight + (size_t)l * dim, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);
        part_range_node(hidden_dim, tb, &r0, &r1);
        if (r0 < r1) {
            matmul_rows_batch(BT.hb, hidden_dim, XQ, XS, B, w->w1 + l, dim, r0, r1);
            matmul_rows_batch(BT.hb2, hidden_dim, XQ, XS, B, w->w3 + l, dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* h1 = BT.hb + (size_t)b * hidden_dim;
                const float* h3 = BT.hb2 + (size_t)b * hidden_dim;
                for (int i = r0; i < r1; i++) h1[i] = silu_f(h1[i]) * h3[i];
            }
        }
        barrier(tb);
        if (IMX.on) imx_dense(tb, IMX.w2 + (size_t)l * hidden_dim, BT.hb, hidden_dim, B, hidden_dim, NULL);
        for (int b = n0; b < n1; b++) quantize_vec(XQ + (size_t)b * hidden_dim, XS + (size_t)b * ngh, BT.hb + (size_t)b * hidden_dim, hidden_dim);
        barrier(tb);
        part_range_node(dim, tb, &r0, &r1);
        if (r0 < r1) {
            matmul_rows_batch(BT.xb2, dim, XQ, XS, B, w->w2 + l, hidden_dim, r0, r1);
            for (int b = 0; b < B; b++)
                for (int i = r0; i < r1; i++) BT.x[(size_t)b * dim + i] += BT.xb2[(size_t)b * dim + i];
        }
        barrier(tb);
    }
    if (BT.logits_all) {
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_final_weight, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);
        part_range_node(p->vocab_size, tb, &r0, &r1);
        matmul_rows_batch(BT.logits_all, p->vocab_size, XQ, XS, B, w->wcls, dim, r0, r1);
        barrier(tb);
        return;
    }
    rmsnorm(tb->xb, BT.x + (size_t)(B - 1) * dim, w->rms_final_weight, dim);
    quantize_vec(tb->q, tb->s, tb->xb, dim);
    part_range_node(p->vocab_size, tb, &r0, &r1);
    matmul_rows(s->logits, tb->q, tb->s, w->wcls, dim, r0, r1);
    barrier(tb);
}

static void forward_batch_thread(int tid) {
    if (P.t->config.hybrid) { forward_hybrid_batch_thread(tid); return; }
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
    int r0, r1, b0, b1, n0, n1;
    int8_t* XQ = BN[tb->node].xq; // the node's quantized block inputs
    float* XS = BN[tb->node].xs;
    tok_range(B, tb->ltid, tb->nthn, &n0, &n1); // this thread's tokens for the node-local quantizations

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
    barrier(tb); // the node-local normalizations below read tokens written by other threads
    // static row ranges: a thread keeps a tile of token inputs in L1 over its whole range (dynamic
    // 16-row chunks re-read all B inputs per chunk and were slower). q, k and v are split as one
    // range of q_dim + 2 kv_dim rows so that every thread gets the same number of rows
    int qa, qe, ka, ke, va, ve;
    {
        int qlo, qhi, klo, khi, x0, x1;
        node_rows(q_dim, tb->node, &qlo, &qhi);
        node_rows(kv_dim, tb->node, &klo, &khi);
        const int nq = qhi - qlo, nk = khi - klo, tot = (nq + 2 * nk) / 4;
        x0 = 4 * (int)((long)tot * tb->ltid / tb->nthn);
        x1 = 4 * (int)((long)tot * (tb->ltid + 1) / tb->nthn);
        if (tb->ltid == tb->nthn - 1) x1 = nq + 2 * nk;
        // intersect [x0, x1) with the three sub-ranges
        qa = qlo + (x0 < nq ? x0 : nq); qe = qlo + (x1 < nq ? x1 : nq);
        ka = klo + (x0 - nq < 0 ? 0 : (x0 - nq > nk ? nk : x0 - nq)); ke = klo + (x1 - nq < 0 ? 0 : (x1 - nq > nk ? nk : x1 - nq));
        va = klo + (x0 - nq - nk < 0 ? 0 : (x0 - nq - nk > nk ? nk : x0 - nq - nk)); ve = klo + (x1 - nq - nk < 0 ? 0 : (x1 - nq - nk > nk ? nk : x1 - nq - nk));
    }

    for (int l = 0; l < p->n_layers; l++) {
        size_t loff = (size_t)l * p->seq_len * kv_dim;

        if (IMX.on) imx_dense(tb, IMX.att + (size_t)l * dim, BT.x, dim, B, dim, w->rms_att_weight + l * dim);
        // attention rmsnorm + quantize, split over tokens
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_att_weight + l * dim, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);

        // fused q|k|v for all tokens
        {
            int a = qa, e = qe;
            if (a < e) {
                matmul_rows_batch(BT.q, q_dim, XQ, XS, B, w->wq + l, dim, a, e);
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
            a = ka; e = ke;
            if (a < e) {
                matmul_rows_batch(BT.k, kv_dim, XQ, XS, B, w->wk + l, dim, a, e);
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
            a = va; e = ve;
            if (a < e) {
                matmul_rows_batch(BT.v, kv_dim, XQ, XS, B, w->wv + l, dim, a, e);
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

        if (IMX.on) imx_dense(tb, IMX.wo + (size_t)l * IMX_WO_W, BT.xb, q_dim, B, q_dim, NULL);
        // quantize the attention output, split over tokens
        for (int b = n0; b < n1; b++) {
            quantize_vec(XQ + (size_t)b * q_dim, XS + (size_t)b * ngq, BT.xb + (size_t)b * q_dim, q_dim);
        }
        barrier(tb);

        // output projection + residual
        part_range_node(dim, tb, &r0, &r1);
        {
            matmul_rows_batch(BT.xb2, dim, XQ, XS, B, w->wo + l, q_dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* x = BT.x + (size_t)b * dim;
                const float* o = BT.xb2 + (size_t)b * dim;
                for (int i = r0; i < r1; i++) { x[i] += o[i]; }
            }
        }
        barrier(tb);

        if (IMX.on && p->n_experts == 0) imx_dense(tb, IMX.ffn + (size_t)l * dim, BT.x, dim, B, dim, w->rms_ffn_weight + l * dim);
        // ffn rmsnorm + quantize, split over tokens
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_ffn_weight + l * dim, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);

        if (p->n_experts > 0) {
            moe_batch(tb, l, B);
        } else {
            // fused w1|w3 + SwiGLU
            part_range_node(hidden_dim, tb, &r0, &r1);
            {
                matmul_rows_batch(BT.hb, hidden_dim, XQ, XS, B, w->w1 + l, dim, r0, r1);
                matmul_rows_batch(BT.hb2, hidden_dim, XQ, XS, B, w->w3 + l, dim, r0, r1);
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

            if (IMX.on) imx_dense(tb, IMX.w2 + (size_t)l * hidden_dim, BT.hb, hidden_dim, B, hidden_dim, NULL);
            // quantize the hidden activations, split over tokens
            for (int b = n0; b < n1; b++) {
                quantize_vec(XQ + (size_t)b * hidden_dim, XS + (size_t)b * ngh, BT.hb + (size_t)b * hidden_dim, hidden_dim);
            }
            barrier(tb);

            // w2 + residual
            part_range_node(dim, tb, &r0, &r1);
            {
                matmul_rows_batch(BT.xb2, dim, XQ, XS, B, w->w2 + l, hidden_dim, r0, r1);
                for (int b = 0; b < B; b++) {
                    float* x = BT.x + (size_t)b * dim;
                    const float* o = BT.xb2 + (size_t)b * dim;
                    for (int i = r0; i < r1; i++) { x[i] += o[i]; }
                }
            }
        }
        barrier(tb);
    }

    if (BT.logits_all) { // logits of every position
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_final_weight, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);
        part_range_node(p->vocab_size, tb, &r0, &r1);
        matmul_rows_batch(BT.logits_all, p->vocab_size, XQ, XS, B, w->wcls, dim, r0, r1);
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
    const int dim = p->dim, q_dim = p->q_dim, kv_dim = p->kv_dim;
    const int kv_mul = p->n_heads / p->n_kv_heads;
    const int hidden_dim = p->hidden_dim;
    const int head_size = p->head_dim, n_heads = p->n_heads;
    const int hs2 = head_size / 2;
    const int qkn = p->qk_norm;
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

        while (grab_rows(WORK_PHASES * l + 0, q_dim, tb, &r0, &r1)) {
            matmul_rows_batch(BT.q, q_dim, nq, ns, B, w->wq + l, dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* q = BT.q + (size_t)b * q_dim;
                const float* rc = BT.rope_c + b * hs2;
                const float* rs = BT.rope_s + b * hs2;
                if (w->bq) for (int i = r0; i < r1; i++) q[i] += w->bq[(size_t)l * q_dim + i];
                if (!qkn) for (int i = r0; i < r1; i += 2) {
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
                if (qkn) continue; // per-head norm + RoPE below, then into the cache
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
        if (qkn) { // per-head q/k RMSNorm + RoPE of every token, k into the cache
            const int per = n_heads + p->n_kv_heads;
            for (int task = tid; task < B * per; task += P.nth) {
                int b = task / per, h = task % per;
                const float* rc = BT.rope_c + b * hs2;
                const float* rs = BT.rope_s + b * hs2;
                if (h < n_heads) {
                    head_norm_rope(BT.q + (size_t)b * q_dim + (size_t)h * head_size, w->q_norm + (size_t)l * head_size, head_size, rc, rs);
                } else {
                    int g = h - n_heads;
                    float* kv = BT.k + (size_t)b * kv_dim + (size_t)g * head_size;
                    head_norm_rope(kv, w->k_norm + (size_t)l * head_size, head_size, rc, rs);
                    memcpy(s->key_cache + loff + (size_t)g * p->seq_len * head_size + (size_t)(pos0 + b) * head_size, kv, head_size * sizeof(float));
                }
            }
            barrier(tb);
        }

        // causal attention: items = (kv group of this node, chunk of the context, subset of the tokens),
        // each handling the kv_mul heads of the group, so every K/V chunk is read once; online-softmax
        // partials are merged per (token, head) by the node afterwards
        float* att = BT.att + (size_t)tid * kv_mul * p->seq_len;
        int g0, g1;
        node_groups(p->n_kv_heads, tb->node, &g0, &g1);
        const int Tt = pos0 + B; // positions seen by the last token
        int C = Tt / 32 < 1 ? 1 : Tt / 32;
        int cmax = tb->nthn / (g1 - g0 > 0 ? g1 - g0 : 1);
        if (C > cmax) C = cmax > 0 ? cmax : 1;
        while (C > 1 && (size_t)B * n_heads * C * (head_size + 2) > BT.mpart_n) C--;
        const int hp = head_size + 2;
        int TS = 1; // short context: also split the tokens so that every thread of the node gets an item
        while (TS < B && (g1 - g0) * C * TS * 2 <= tb->nthn) TS *= 2;
        if (TS > B) TS = B;
        for (int item = tb->ltid; item < (g1 - g0) * C * TS; item += tb->nthn) {
            int g = g0 + item / (C * TS), c = (item / TS) % C, ts = item % TS;
            int t0 = (int)((long)Tt * c / C), t1 = (int)((long)Tt * (c + 1) / C);
            const float* kb = s->key_cache + loff + (size_t)g * p->seq_len * head_size;
            const float* vb = s->value_cache + loff + (size_t)g * p->seq_len * head_size;
            for (int b = B * ts / TS; b < B * (ts + 1) / TS; b++) {
                int te = pos0 + b + 1 < t1 ? pos0 + b + 1 : t1; // causal end for token b
                for (int r = 0; r < kv_mul; r++) {
                    int h = g * kv_mul + r;
                    float* pr = BT.mpart + (((size_t)b * n_heads + h) * C + c) * hp;
                    if (te <= t0) { pr[1] = 0.0f; continue; }
                    const float* q = BT.q + (size_t)b * q_dim + (size_t)h * head_size;
                    float* ar = att + (size_t)r * p->seq_len;
                    float m = -FLT_MAX;
                    for (int t = t0; t < te; t++) {
                        const float* k = kb + (size_t)t * head_size;
                        float score = 0.0f;
                        for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                        score *= att_scale;
                        ar[t - t0] = score;
                        m = score > m ? score : m;
                    }
                    float sm = 0.0f;
                    float* o = pr + 2;
                    memset(o, 0, head_size * sizeof(float));
                    for (int t = t0; t < te; t++) {
                        float e = expf(ar[t - t0] - m);
                        sm += e;
                        const float* v = vb + (size_t)t * head_size;
                        for (int i = 0; i < head_size; i++) o[i] += e * v[i];
                    }
                    pr[0] = m;
                    pr[1] = sm;
                }
            }
        }
        node_barrier(tb);
        { // merge: (token, head of this node) pairs split among the node's threads
            const int h0 = g0 * kv_mul, nh = (g1 - g0) * kv_mul;
            for (int it = tb->ltid; it < B * nh; it += tb->nthn) {
                int b = it / nh, h = h0 + it % nh;
                const float* ph = BT.mpart + ((size_t)b * n_heads + h) * C * hp;
                float M = -FLT_MAX, den = 0.0f;
                for (int c = 0; c < C; c++) if (ph[(size_t)c * hp + 1] > 0.0f && ph[(size_t)c * hp] > M) M = ph[(size_t)c * hp];
                float* ob = BT.xb + (size_t)b * q_dim + (size_t)h * head_size;
                memset(ob, 0, head_size * sizeof(float));
                for (int c = 0; c < C; c++) {
                    const float* pr = ph + (size_t)c * hp;
                    if (pr[1] <= 0.0f) continue;
                    float f = expf(pr[0] - M);
                    den += f * pr[1];
                    for (int i = 0; i < head_size; i++) ob[i] += f * pr[2 + i];
                }
                float inv = 1.0f / den;
                for (int i = 0; i < head_size; i++) ob[i] *= inv;
            }
        }
        barrier(tb);
        node_quant_multi(tb, BT.xb, q_dim, B);

        while (grab_rows(WORK_PHASES * l + 3, dim, tb, &r0, &r1)) {
            matmul_rows_batch(BT.xb2, dim, nq, ns, B, w->wo + l, q_dim, r0, r1);
            for (int b = 0; b < B; b++)
                for (int i = r0; i < r1; i++) X[(size_t)b * dim + i] += BT.xb2[(size_t)b * dim + i];
        }
        barrier(tb);

        node_norm_quant_multi(tb, X, w->rms_ffn_weight + l * dim, dim, B);
        if (p->n_experts > 0) {
            /* MoE: every node computes the router logits of all tokens; every thread then derives the
             * same union of selected experts (ascending id) with the (token, weight) pairs of each.
             * Each expert's rows are read once and applied to all its tokens (4x4 kernel). */
            const int E = p->n_experts, A = p->n_active, mh = p->moe_hidden;
            NodeBuf* nb = &NB[tb->node];
            float* rl = nb->router; // (B, E)
            while (grab_virtual(WORK_PHASES * l + 6, E, tb, &r0, &r1))
                matmul_rows_batch(rl, E, nq, ns, B, w->wg_node[tb->node] + l, dim, r0, r1);
            node_barrier(tb);
            int sel[SPEC_MAX_TOK * MOE_MAX_ACTIVE], cnt[257] = {0};
            float wt[SPEC_MAX_TOK * MOE_MAX_ACTIVE];
            for (int b = 0; b < B; b++) {
                moe_route(rl + (size_t)b * E, E, A, p->norm_topk, sel + b * A, wt + b * A);
                for (int j = 0; j < A; j++) cnt[sel[b * A + j]]++;
            }
            int ue[SPEC_MAX_TOK * MOE_MAX_ACTIVE], up0[SPEC_MAX_TOK * MOE_MAX_ACTIVE + 1], U = 0;
            int ptok[SPEC_MAX_TOK * MOE_MAX_ACTIVE];
            float pw[SPEC_MAX_TOK * MOE_MAX_ACTIVE];
            up0[0] = 0;
            for (int e = 0; e < E; e++) if (cnt[e]) { ue[U] = e; up0[U + 1] = up0[U] + cnt[e]; cnt[e] = U++; }
            int fill[SPEC_MAX_TOK * MOE_MAX_ACTIVE] = {0};
            for (int b = 0; b < B; b++) // token order inside every expert: ascending b
                for (int j = 0; j < A; j++) {
                    int u = cnt[sel[b * A + j]], pi = up0[u] + fill[u]++;
                    ptok[pi] = b;
                    pw[pi] = wt[b * A + j];
                }
            const int NP = up0[U];
            // gather the quantized ffn input of every pair into the node's pair buffer
            const int ngd = dim / GS;
            for (int pi = tb->ltid; pi < NP; pi += tb->nthn) {
                memcpy(nb->gq + (size_t)pi * dim, nq + (size_t)ptok[pi] * dim, dim);
                memcpy(nb->gs + (size_t)pi * ngd, ns + (size_t)ptok[pi] * ngd, ngd * sizeof(float));
            }
            node_barrier(tb);
            int lo, hi;
            node_rows(mh, tb->node, &lo, &hi);
            const int half = hi - lo;
            while (grab_virtual(WORK_PHASES * l + 4, U * half, tb, &r0, &r1)) {
                for (int v = r0; v < r1;) {
                    int u = v / half, vend = (u + 1) * half < r1 ? (u + 1) * half : r1;
                    int a0 = lo + (v - u * half), a1 = lo + (vend - u * half);
                    int p0 = up0[u], np = up0[u + 1] - p0;
                    QuantizedTensor e1 = expert_view(w->w1 + l, ue[u], mh, dim), e3 = expert_view(w->w3 + l, ue[u], mh, dim);
                    matmul_rows_batch(BT.gh + (size_t)p0 * mh, mh, nb->gq + (size_t)p0 * dim, nb->gs + (size_t)p0 * ngd, np, &e1, dim, a0, a1);
                    matmul_rows_batch(BT.gh3 + (size_t)p0 * mh, mh, nb->gq + (size_t)p0 * dim, nb->gs + (size_t)p0 * ngd, np, &e3, dim, a0, a1);
                    for (int pi = p0; pi < p0 + np; pi++) {
                        float* h1 = BT.gh + (size_t)pi * mh;
                        const float* h3 = BT.gh3 + (size_t)pi * mh;
                        for (int i = a0; i < a1; i++) {
                            float val = h1[i];
                            val *= (1.0f / (1.0f + expf(-val)));
                            h1[i] = val * h3[i] * pw[pi];
                        }
                    }
                    v = vend;
                }
            }
            barrier(tb);
            node_quant_multi(tb, BT.gh, mh, NP); // pair pi at nq + pi * mh
            while (grab_rows(WORK_PHASES * l + 5, dim, tb, &r0, &r1)) {
                for (int u = 0; u < U; u++) {
                    int p0 = up0[u], np = up0[u + 1] - p0;
                    QuantizedTensor e2 = expert_view(w->w2 + l, ue[u], dim, mh);
                    matmul_rows_batch(BT.gout + (size_t)p0 * dim, dim, nq + (size_t)p0 * mh, ns + (size_t)p0 * (mh / GS), np, &e2, mh, r0, r1);
                }
                for (int b = 0; b < B; b++) { // add the experts of token b in rank order (as the decode path)
                    float* x = X + (size_t)b * dim;
                    for (int j = 0; j < A; j++) {
                        int u = cnt[sel[b * A + j]], pi = up0[u];
                        while (ptok[pi] != b) pi++;
                        const float* o = BT.gout + (size_t)pi * dim;
                        for (int i = r0; i < r1; i++) x[i] += o[i];
                    }
                }
            }
            barrier(tb);
        } else {
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
    if (p->hybrid && p->lin_nv * p->lin_dv > wide) wide = p->lin_nv * p->lin_dv;
    if (p->hybrid && p->lin_nv * p->lin_dv > qmax) qmax = p->lin_nv * p->lin_dv;
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
    for (int k = 0; k < NN; k++) {
        size_t S = p->n_experts > 0 ? B * p->n_active : 0, mh = p->moe_hidden;
        size_t sz[6] = {B * qmax, B * (qmax / GS + 1) * sizeof(float), S * p->dim, S * (p->dim / GS) * sizeof(float), S * mh, S * (mh / GS) * sizeof(float)};
        void* ptr[6];
        for (int j = 0; j < 6; j++) {
            size_t b = (sz[j] + 4095) & ~(size_t)4095;
            ptr[j] = b ? aligned_alloc(4096, b) : NULL;
#if defined(__linux__)
            if (b && NN > 1) bind_range(ptr[j], b, NODE_ID[k]);
#endif
            if (b) memset(ptr[j], 0, b);
        }
        BN[k].xq = ptr[0]; BN[k].xs = ptr[1]; BN[k].gxq = ptr[2]; BN[k].gxs = ptr[3]; BN[k].gq = ptr[4]; BN[k].gqs = ptr[5];
    }
    BT.rope_c = calloc(B * (head_size / 2), sizeof(float));
    BT.rope_s = calloc(B * (head_size / 2), sizeof(float));
    BT.att = calloc((size_t)P.nth * (p->n_heads / p->n_kv_heads) * p->seq_len, sizeof(float));
    BT.gate = BT.lin = BT.lino = NULL;
    if (p->hybrid) {
        BT.gate = calloc(B * q_dim, sizeof(float));
        BT.lin = calloc(B * p->lin_in, sizeof(float));
        BT.lino = calloc(B * p->lin_nv * p->lin_dv, sizeof(float));
        HYB_BATCH = 1;
    }
    BT.mpart_n = (size_t)SPEC_MAX_TOK * p->n_heads * (P.nth > 1 ? P.nth : 1) * (head_size + 2);
    BT.mpart = calloc(BT.mpart_n, sizeof(float));
    if (p->n_experts > 0) {
        size_t S = B * p->n_active, E = p->n_experts, mh = p->moe_hidden;
        BT.router = calloc(B * E, sizeof(float));
        BT.moe_sel = calloc(B * p->n_active, sizeof(int));
        BT.moe_wt = calloc(B * p->n_active, sizeof(float));
        BT.moe_cnt = calloc(E + 1, sizeof(int));
        BT.moe_fill = calloc(E + 1, sizeof(int));
        BT.moe_ue = calloc(E, sizeof(int));
        BT.moe_ctr = aligned_alloc(64, MAX_NODES * sizeof(struct WorkCounter));
        memset(BT.moe_ctr, 0, MAX_NODES * sizeof(struct WorkCounter));
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
static float* SPEC_LOGITS = NULL;
float* forward_all(Transformer* transformer, const int* tokens, int n, int pos0) {
    use_model(transformer);
    int V = transformer->config.vocab_size;
    if (!SPEC_LOGITS) SPEC_LOGITS = malloc((size_t)SPEC_MAX * V * sizeof(float));
    if (n == 1 || transformer->config.hybrid) { // hybrid: no multi-token step (the state cannot roll back)
        for (int i = 0; i < n; i++) memcpy(SPEC_LOGITS + (size_t)i * V, forward(transformer, tokens[i], pos0 + i), V * sizeof(float));
        return SPEC_LOGITS;
    }
    Config* p = &transformer->config;
    int dim = p->dim, hs2 = p->head_dim / 2;
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
    if (transformer->config.hybrid) {
        if (!HYB_BATCH) { // token by token
            float* l = NULL;
            for (int i = 0; i < n; i++) l = forward(transformer, tokens[i], pos0 + i);
            return l;
        }
        if (pos0 != 0 && pos0 != transformer->state.next_pos) {
            fprintf(stderr, "hybrid model: position %d out of order (state at %d)\n", pos0, transformer->state.next_pos);
            exit(EXIT_FAILURE);
        }
        transformer->state.next_pos = pos0; // advanced block by block below (a 1-token block goes through forward())
    }

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
            transformer->state.next_pos = pos0 + done + B;
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

/* Parallel sampling on the thread pool (the vocabulary is 128-152k entries: a serial softmax +
 * nucleus scan cost ~1.3 ms per token, more than 5% of a 30B-A3B decode step). Every thread takes a
 * static slice of the logits: (1) its max and sum of exp((l - max) / T) (or its argmax), barrier,
 * (2) the global max M and normalizer Z (online-softmax merge), and the slice's nucleus candidates
 * l >= M + T * ln(cutoff * Z), i.e. probability >= (1 - topp) / (n - 1) exactly as in sample_topp.
 * Thread 0 then sorts the few candidates. Same distribution as the serial code. */
static struct {
  const float *l;
  int n;
  float T, lnc; // temperature (0: argmax), ln of the nucleus cutoff probability
  float *m, *z; // per-thread max / partial normalizer (stride 16: one cache line per thread)
  int *arg, *cnt;
  int *cand;    // candidate indices, thread t writes from the start of its slice
  float M, Z;
} SP;
static void sample_thread(int tid) {
  ThreadBuf *tb = &P.tb[tid];
  const int n = SP.n, nth = P.nth;
  const int i0 = (int)((long)n * tid / nth), i1 = (int)((long)n * (tid + 1) / nth);
  const float *l = SP.l;
  float m = -FLT_MAX;
  int am = i0;
  for (int i = i0; i < i1; i++) if (l[i] > m) { m = l[i]; am = i; }
  SP.m[tid * 16] = m;
  SP.arg[tid * 16] = am;
  if (SP.T > 0.0f) {
    const float it = 1.0f / SP.T;
    float z = 0.0f;
    for (int i = i0; i < i1; i++) z += expf((l[i] - m) * it);
    SP.z[tid * 16] = z;
  }
  barrier(tb);
  if (SP.T <= 0.0f) return;
  float M = -FLT_MAX, Z = 0.0f;
  for (int t = 0; t < nth; t++) M = SP.m[t * 16] > M ? SP.m[t * 16] : M;
  for (int t = 0; t < nth; t++) Z += SP.z[t * 16] * expf((SP.m[t * 16] - M) / SP.T);
  const float thr = M + SP.T * (SP.lnc + logf(Z));
  int c = 0;
  for (int i = i0; i < i1; i++) if (l[i] >= thr) SP.cand[i0 + c++] = i;
  SP.cnt[tid * 16] = c;
  if (tid == 0) { SP.M = M; SP.Z = Z; }
  barrier(tb);
}
// builds the sorted nucleus of logits / T (0 < topp < 1) in pi; returns its size (all candidates),
// sets *last (last index inside the nucleus) and *cum (its probability mass)
static int par_nucleus(Sampler *sp, const float *logits, ProbIndex *pi, int *last, float *cum) {
  int n = sp->vocab_size;
  if (!SP.m) {
    SP.m = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(float));
    SP.z = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(float));
    SP.arg = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(int));
    SP.cnt = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(int));
  }
  static int cand_n = 0;
  if (cand_n < n) { free(SP.cand); SP.cand = malloc((size_t)n * sizeof(int)); cand_n = n; }
  SP.l = logits; SP.n = n; SP.T = sp->temperature;
  SP.lnc = logf((1.0f - sp->topp) / (n - 1));
  P.mode = 4;
  wake_workers();
  sample_thread(0);
  P.mode = 0;
  int n0 = 0;
  const float iz = 1.0f / SP.Z;
  for (int t = 0; t < P.nth; t++) {
    int i0 = (int)((long)n * t / P.nth);
    for (int c = 0; c < SP.cnt[t * 16]; c++) {
      int i = SP.cand[i0 + c];
      pi[n0].index = i;
      pi[n0].prob = expf((logits[i] - SP.M) / SP.T) * iz;
      n0++;
    }
  }
  qsort(pi, n0, sizeof(ProbIndex), compare);
  float cp = 0.0f;
  int li = n0 - 1;
  for (int i = 0; i < n0; i++) {
    cp += pi[i].prob;
    if (cp > sp->topp) { li = i; break; }
  }
  *last = li;
  *cum = cp;
  return n0;
}
static int par_argmax(const float *logits, int n) {
  if (!SP.m) {
    SP.m = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(float));
    SP.z = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(float));
    SP.arg = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(int));
    SP.cnt = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(int));
  }
  SP.l = logits; SP.n = n; SP.T = 0.0f;
  P.mode = 4;
  wake_workers();
  sample_thread(0);
  P.mode = 0;
  int best = SP.arg[0];
  float bm = SP.m[0];
  for (int t = 1; t < P.nth; t++) if (SP.m[t * 16] > bm) { bm = SP.m[t * 16]; best = SP.arg[t * 16]; } // first max, like sample_argmax
  return best;
}
static inline int pool_ready(void) { return P.nth > 1 && P.tb != NULL; }

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f && pool_ready()) return par_argmax(logits, sampler->vocab_size);
  if (sampler->temperature > 0.0f && sampler->topp > 0 && sampler->topp < 1 && pool_ready()) {
    int last;
    float cum;
    par_nucleus(sampler, logits, sampler->probindex, &last, &cum);
    float r = random_f32(&sampler->rng_state) * cum, cdf = 0.0f;
    for (int i = 0; i <= last; i++) {
      cdf += sampler->probindex[i].prob;
      if (r < cdf) return sampler->probindex[i].index;
    }
    return sampler->probindex[last].index; // in case of rounding errors
  }
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
  if (sampler->temperature == 0.0f) return pool_ready() ? par_argmax(logits, sampler->vocab_size) : sample_argmax(logits, sampler->vocab_size);
  int n = sampler->vocab_size;
  ProbIndex *pi = sampler->probindex;
  int n0 = 0, last;
  float cum = 0.0f;
  if (sampler->topp > 0 && sampler->topp < 1 && pool_ready()) {
    par_nucleus(sampler, logits, pi, &last, &cum);
  } else if (sampler->topp <= 0 || sampler->topp >= 1) {
    for (int q = 0; q < n; q++) logits[q] /= sampler->temperature;
    softmax(logits, n);
    for (int i = 0; i < n; i++) { pi[n0].index = i; pi[n0].prob = logits[i]; n0++; }
    last = n0 - 1;
    cum = 1.0f;
  } else {
    for (int q = 0; q < n; q++) logits[q] /= sampler->temperature;
    softmax(logits, n);
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

// ChatML special token ids (Qwen2.5 / Qwen3 values; looked up in the vocabulary by init_special_tokens)
static int QWEN_ENDOFTEXT = 151643, QWEN_IM_END = 151645, QWEN_TOOL_CALL = 151657, QWEN_TOOL_CALL_END = 151658;
static int QWEN_THINK = -1, QWEN_THINK_END = -1;
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
}

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
  if (ARCH == ARCH_QWEN35) { // Qwen3.5 / 3.8 template (XML tool calls, optional reasoning instructions)
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
  if (ARCH >= ARCH_QWEN2) {
    m = push_tpl(tk, "<|im_start|>user\n", msg, m);
    m = push_text(tk, text, msg, m);
    m = push_tpl(tk, "<|im_end|>\n<|im_start|>assistant\n", msg, m);
    if (ARCH == ARCH_QWEN35) m = push_tpl(tk, THINK ? "<think>\n" : "<think>\n\n</think>\n\n", msg, m);
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
    if (ARCH == ARCH_QWEN35) m = push_tpl(tk, THINK ? "<think>\n" : "<think>\n\n</think>\n\n", msg, m);
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
      JNode *c = ARCH == ARCH_QWEN35 && strstr(buf->buf, "<function=") ? parse_xml_call(buf->buf) : normalize_call(json_parse(buf->buf, NULL));
      if (c) rs->calls[rs->n_calls++] = c;
      else { printf("<tool_call>%s</tool_call>", buf->buf); fflush(stdout); }
    }
    rs->mode = 1;
  } else if (ARCH < ARCH_QWEN2 && next == LLAMA3_PYTHON_TAG) {
    rs->mode = 2;
  } else if (next >= 0 && next == QWEN_THINK) { // reasoning is shown dimmed
    if (THINK) { printf("\033[2m"); fflush(stdout); }
  } else if (next >= 0 && next == QWEN_THINK_END) {
    if (THINK) { printf("\033[0m\n"); fflush(stdout); }
    rs->mode = 0; // skip the blank lines before the answer
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
  if (ARCH == ARCH_QWEN35 && THINK) { printf("\033[2m"); fflush(stdout); } // the prompt already opened <think>
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
    if (!from_model && moe) { // MoE: a verification step costs ~1.4x (2 tokens) .. 2.5x (8): adapt the draft length
      if (acc == k) spec_len = spec_len * 2 < SPEC_K ? spec_len * 2 : SPEC_K;
      else spec_len = acc + 1 > 2 ? acc + 1 : 2;
      if (acc == 0) { spec_cool = spec_cool ? (spec_cool * 2 < 32 ? spec_cool * 2 : 32) : 2; spec_wait = spec_cool; }
      else spec_cool = 0;
    }
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
/* -m imatrix: run a text through the model (chunks of up to 512 tokens, each from position 0) and
 * write the importance statistics: "NMX1", int L, dim, q_dim, hidden, E, moe_hidden, long long ntok,
 * then float32 means: att (L,dim), wo (L,q_dim), ffn (L,[E,]dim), w2 (L,[E,]hidden or moe_hidden),
 * and for MoE the int64 token counts (L,E). Collect on a model exported without --smooth. */
static void imatrix(Transformer* t, Tokenizer* tk, const char* text_path, const char* out_path) {
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
        forward_batch(t, chunk, len + 1, 0);
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
  fprintf(stderr, "  -m <string> mode: generate|chat|imatrix, default: generate\n");
  fprintf(stderr, "              imatrix: -i <text file> -o <out.imx> collects Q4 importance statistics\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr, "  -T <int>    number of threads, default 0 = auto (placed on the most idle CPUs; RUNQ_PIN=1 keeps them pinned, RUNQ_PIN=0 disables placement)\n");
  fprintf(stderr, "  -c <int>    context window in tokens (KV cache size), default 8192\n");
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
    } else if (argv[i][1] == 'o') {
      out_path = argv[i + 1];
    } else if (argv[i][1] == 'r') {
      THINK = atoi(argv[i + 1]);
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
  if (transformer.config.hybrid && (SPEC_K > 0 || draft_path)) { // a recurrent state cannot drop rejected drafts
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
