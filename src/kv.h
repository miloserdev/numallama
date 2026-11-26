// KV cache formats: f32 (default) and int8 with per-group scales (groups of GS values).
// The layout stays (layer, kv_head, pos, head_dim), so a head is one contiguous stream; the scales
// of a row are (head_dim / GS) floats, laid out the same way after the int8 blobs of a layer.
// Attention quantizes the query per head once and keeps the K/V rows quantized: scores are int8
// dot products (SDOT), V is accumulated with a folded scale, no dequantized copy is materialized.

enum { KV_F32 = 0, KV_I8 = 1 };

static int KV_REQ = -1; // -1: take RUNQ_KV from the environment, else int8 (fallbacks below)

static int kv_type_parse(const char* s) {
  if (!s) return -1;
  if (!strcmp(s, "f32") || !strcmp(s, "fp32") || !strcmp(s, "0")) return KV_F32;
  if (!strcmp(s, "i8") || !strcmp(s, "q8") || !strcmp(s, "1")) return KV_I8;
  return -1;
}

static inline const char* kv_type_name(int t) { return t == KV_I8 ? "i8" : "f32"; }

// pick the cache format for one model; int8 is the default, f32 is the precision escape hatch.
// i8 needs head_dim % GS == 0 and is not used for GLM (MLA latent cache).
static int kv_resolve(const Config* p) {
  if (KV_REQ < 0) {
    const char* e = getenv("RUNQ_KV");
    int t = kv_type_parse(e);
    KV_REQ = t < 0 ? KV_I8 : t;
  }
  if (KV_REQ == KV_I8 && p->glm) {
    fprintf(stderr, "[kv] i8 is not supported for GLM (MLA latent cache), using f32\n");
    return KV_F32;
  }
  if (KV_REQ == KV_I8 && p->head_dim % GS != 0) {
    fprintf(stderr, "[kv] i8 needs head_dim %% %d == 0 (head_dim %d), using f32\n", GS, p->head_dim);
    return KV_F32;
  }
  return KV_REQ;
}

static void kv_alloc(RunState* s, const Config* p) {
  s->kv_type = kv_resolve(p);
  if (s->kv_type == KV_I8) {
    size_t nq = (size_t)p->n_attn * p->seq_len * p->kv_dim;
    size_t ns = nq / GS;
    s->key_cache_q = calloc(nq, 1);
    s->value_cache_q = calloc(nq, 1);
    s->key_cache_s = calloc(ns, sizeof(float));
    s->value_cache_s = calloc(ns, sizeof(float));
    if (!s->key_cache_q || !s->value_cache_q || !s->key_cache_s || !s->value_cache_s) {
      fprintf(stderr, "malloc failed (kv i8 cache)!\n");
      exit(EXIT_FAILURE);
    }
  } else {
    s->key_cache = calloc((size_t)p->n_attn * p->seq_len * p->kv_dim, sizeof(float));
    // GLM: the attention cache holds the MLA latent only, no separate V
    s->value_cache = p->glm ? calloc(1, sizeof(float)) : calloc((size_t)p->n_attn * p->seq_len * p->kv_dim, sizeof(float));
  }
}

static void kv_free(RunState* s) {
  free(s->key_cache);
  free(s->value_cache);
  free(s->key_cache_q);
  free(s->value_cache_q);
  free(s->key_cache_s);
  free(s->value_cache_s);
  s->key_cache = s->value_cache = NULL;
  s->key_cache_q = s->value_cache_q = NULL;
  s->key_cache_s = s->value_cache_s = NULL;
}

static inline size_t kv_row_off(const Config* p, int l, int g, int pos) {  return (size_t)l * p->seq_len * p->kv_dim + (size_t)g * p->seq_len * p->head_dim +
         (size_t)pos * p->head_dim;
}
static inline size_t kv_sc_off(const Config* p, int l, int g, int pos) {
  return kv_row_off(p, l, g, pos) / GS;
}

static inline int8_t* kv_k_i8(RunState* s, const Config* p, int l, int g, int pos) {
  return s->key_cache_q + kv_row_off(p, l, g, pos);
}
static inline int8_t* kv_v_i8(RunState* s, const Config* p, int l, int g, int pos) {
  return s->value_cache_q + kv_row_off(p, l, g, pos);
}
static inline float* kv_ks(RunState* s, const Config* p, int l, int g, int pos) {
  return s->key_cache_s + kv_sc_off(p, l, g, pos);
}
static inline float* kv_vs(RunState* s, const Config* p, int l, int g, int pos) {
  return s->value_cache_s + kv_sc_off(p, l, g, pos);
}

// quantize one head row (n = head_dim, n % GS == 0)
static inline void kv_store_row(int8_t* dst, float* sc, const float* src, int n) { quantize_vec(dst, sc, src, n); }

// score = q . k with both sides quantized per GS group
static inline float kv_dot_q8(const int8_t* q, const float* qs, const int8_t* k, const float* ks, int n) {
  const int ng = n / GS;
  float acc = 0.0f;
  for (int g = 0; g < ng; g++) {
    int32_t iv = 0;
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t a = vdupq_n_s32(0);
    for (int i = 0; i < GS; i += 16) a = vdotq_s32(a, vld1q_s8(q + g * GS + i), vld1q_s8(k + g * GS + i));
    iv = vaddvq_s32(a);
#else
    for (int i = 0; i < GS; i++) iv += (int32_t)q[g * GS + i] * k[g * GS + i];
#endif
    acc += (float)iv * qs[g] * ks[g];
  }
  return acc;
}

// dequantize one head row into f32 (once per token, then FMA it into every query head of the group)
static inline void kv_dequant_row(float* dst, const int8_t* v, const float* vs, int n) {
  for (int g = 0; g < n / GS; g++) {
    float f = vs[g];
    const int8_t* p = v + g * GS;
    float* d = dst + g * GS;
    for (int i = 0; i < GS; i++) d[i] = f * (float)p[i];
  }
}

// out += a * v, v quantized per GS group (a includes the softmax weight)
static inline void kv_axpy_i8(float* out, float a, const int8_t* v, const float* vs, int n) {
  const int ng = n / GS;
  for (int g = 0; g < ng; g++) {
    float f = a * vs[g];
    float* o = out + g * GS;
    const int8_t* p = v + g * GS;
#if defined(__ARM_NEON)
    float32x4_t fv = vdupq_n_f32(f);
    for (int i = 0; i < GS; i += 16) {
      int8x16_t qv = vld1q_s8(p + i);
      int16x8_t lo = vmovl_s8(vget_low_s8(qv)), hi = vmovl_s8(vget_high_s8(qv));
      vst1q_f32(o + i, vfmaq_f32(vld1q_f32(o + i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), fv));
      vst1q_f32(o + i + 4, vfmaq_f32(vld1q_f32(o + i + 4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), fv));
      vst1q_f32(o + i + 8, vfmaq_f32(vld1q_f32(o + i + 8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), fv));
      vst1q_f32(o + i + 12, vfmaq_f32(vld1q_f32(o + i + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), fv));
    }
#else
    for (int i = 0; i < GS; i++) o[i] += f * (float)p[i];
#endif
  }
}
