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
enum { ARCH_LLAMA3 = 0, ARCH_QWEN2 = 1, ARCH_QWEN3 = 2, ARCH_QWEN35 = 3, ARCH_GLM5 = 4, ARCH_INKLING = 5 }; // Qwen*: ChatML template, Qwen tokenizer
#define ARCH_IS_QWEN (ARCH >= ARCH_QWEN2 && ARCH <= ARCH_QWEN35)
int ARCH = ARCH_LLAMA3;
float NORM_EPS = 1e-5f;
int QKV_BIAS = 0;
int SPECIAL_START = 128000; // first special token id
int BOS_ID = 128000;
int EOS_ID = 128009;
int Q4_MASK = 0;
int EMB_PACKED = 0; // the classifier shares the embedding tensor, so pack_q8 repacked it
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
  // GLM-5.3-Flash (arch 4, see glm.h): n_attn MLA layers (the others KDA, which reuse lin_* / conv_* above),
  // n_kv_heads 1 and head_dim kv_lora_rank (the KV cache holds the MLA latent)
  int glm, g_dense, g_ql, g_kvl, g_qk, g_vh, g_H, g_dh, g_K, g_hc, g_sk, g_ihh, g_ih, g_itopk, g_kpool, g_tail, g_q4;
  float g_hc_eps, g_rscale, g_limit, g_lower;
  uint32_t g_amask[2];
} Config;
#define RECURRENT(p) ((p)->hybrid || (p)->glm) // a recurrent state that can only advance

typedef struct {
  int8_t *q; // quantized values (Q4: packed nibbles)
  float *s;  // scaling factors (Q8: per GS values, Q4: per 32 values)
  int bits;  // 8 or 4
} QuantizedTensor;

typedef struct {
  // token embedding table
  QuantizedTensor *q_tokens;    // (vocab_size, dim)
  float *token_embedding_table; // unused: rows are dequantized on demand (embed_row); kept NULL

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
  struct GlmWeights *glm;  // GLM-5.3-Flash weights (arch 4), or NULL
} TransformerWeights;

typedef struct {
  int8_t *kq, *vq;  // int8 K/V values
  float *ks, *vs;   // per-GS-group scales
  float *kf, *vf;   // f32 K/V (kf only for GLM, vf is a dummy there)
  int type;         // KV_F32 / KV_I8
} KVCache;

typedef struct {
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
  float *snap_S, *snap_conv; // hybrid: a saved copy of the recurrent / conv states ...
  int snap_pos;              // ... at this position (-1: none), e.g. right after the system prompt
  float *spec_S, *spec_conv; // hybrid: a second snapshot slot, taken right before a speculative verify pass
  int spec_pos;              // ... at this position
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
  // int8 cache (kv_type == KV_I8): the same index space as the f32 cache, values int8,
  // scales per GS group; key_cache/value_cache stay NULL in this mode
  int kv_type;
  int8_t *key_cache_q, *value_cache_q;
  float *key_cache_s, *value_cache_s;
  KVCache own;   // the cache allocated for this model (kv_alloc) ...
  int kv_owned;  // ... and whether the pointers above are that allocation (0 = attached session)
  struct GlmState *glm; // GLM-5.3-Flash buffers, or NULL
} RunState;
// defined in kv.h (included after this header): allocate/free the cache in the chosen format
static void kv_alloc(RunState *s, const Config *p);
static void kv_free(RunState *s);
static void glm_alloc_state(RunState *s, const Config *p, int nth);
struct GlmWeights;
static void glm_map_weights(TransformerWeights *tw, Config *p, void *ptr);
static void glm_place(const Config *p, struct GlmWeights *w);
static void glm_pack(const Config *p, struct GlmWeights *w);

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
  s->key_cache = s->value_cache = NULL;
  s->key_cache_q = s->value_cache_q = NULL;
  s->key_cache_s = s->value_cache_s = NULL;
  s->glm = NULL;
  s->gate = s->lin = s->lin_o = s->lin_S = s->conv_state = NULL;
  s->next_pos = 0;
  s->snap_S = s->snap_conv = NULL;
  s->snap_pos = -1;
  s->spec_S = s->spec_conv = NULL;
  s->spec_pos = -1;
  s->kv_owned = 0;
  kv_alloc(s, p);
  if (p->glm) {
    int nlin = p->n_layers - p->n_attn;
    s->lin_S = malloc((size_t)nlin * p->lin_nv * p->lin_dv * p->lin_dk * sizeof(float)); // first touched by its threads
    s->conv_state = calloc((size_t)nlin * p->conv_dim * (p->conv_k - 1), sizeof(float));
    glm_alloc_state(s, p, 1024);
  }
  if (p->hybrid) {
    int nlin = p->n_layers - p->n_attn;
    s->gate = calloc(p->q_dim, sizeof(float));
    s->lin = calloc(p->lin_in, sizeof(float));
    s->lin_o = calloc((size_t)p->lin_nv * p->lin_dv, sizeof(float));
    s->lin_S = malloc((size_t)nlin * p->lin_nv * p->lin_dv * p->lin_dk * sizeof(float)); // first touched by its threads
    s->conv_state = calloc((size_t)nlin * p->conv_dim * (p->conv_k - 1), sizeof(float));
  }
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits ||
      !(s->key_cache || s->key_cache_q)) {
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
  kv_free(s);
  free(s->gate); free(s->lin); free(s->lin_o); free(s->lin_S); free(s->conv_state);
  free(s->snap_S); free(s->snap_conv);
  free(s->spec_S); free(s->spec_conv);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float *x, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = qx->q[i] * qx->s[i / GS];
  }
}

// one row of a Q8 tensor (the embedding of one token): dequantized on demand, never materialized.
// Defined in quant.h: the layout depends on whether the classifier shares the embedding tensor
// (then pack_q8 repacks that shared tensor into 4-row blocks).
static inline void embed_row(const QuantizedTensor *qx, int row, float *dst, int n);
static inline void embed_row_slice(const QuantizedTensor *qx, int row, int dim, float *dst, int from, int n);

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
  w->token_embedding_table = NULL; // rows are dequantized on demand (embed_row), never materialized

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
