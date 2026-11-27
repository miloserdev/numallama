// Inkling-2-Mini forward, built in milestones. M1: embedding row, RMSNorm and the Q8 LM head.
// The rest of the layer math is added here as it is validated against the llama.cpp oracle.
static inline void ink_cache_free(void);
typedef struct {
  const InklingModel *m;
  QuantizedTensor unembed;
  int unembed_ready;
} InklingRuntime;

static inline int inkling_runtime_init(const InklingModel *m, InklingRuntime *r) {
  memset(r, 0, sizeof *r);
  r->m = m;
  const InkTensor *u = inkling_find(m, "unembed.weight");
  if (!u || !inkling_pack_2d(u, &r->unembed)) return 0;
  r->unembed_ready = 1;
  return 1;
}

static inline void inkling_runtime_free(InklingRuntime *r) {
  if (r->unembed_ready) inkling_pack_free(&r->unembed);
  ink_cache_free();
  memset(r, 0, sizeof *r);
}

// Q8 token embedding row (row-major q then f32 scales per 64 values)
static inline void inkling_embed_row(const InklingModel *m, int tok, float *dst) {
  const InkTensor *e = inkling_find(m, "embed.weight");
  const int dim = m->dim, ng = dim / GS;
  const int8_t *q = (const int8_t *)e->data;
  const float *s = (const float *)(q + (size_t)m->vocab * dim);
  for (int i = 0; i < dim; i++) dst[i] = q[(size_t)tok * dim + i] * s[(size_t)tok * ng + i / GS];
}

// weightless RMSNorm used by the depth mixer (eps 1e-6, as in the reference)
static inline void inkling_rms0(float *dst, const float *x, int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float sc = 1.0f / sqrtf(ss / n + 1e-6f);
  for (int i = 0; i < n; i++) dst[i] = x[i] * sc;
}

// depth mix: states = [norm(blocks[nb-1]), ..., norm(blocks[0]), norm(cur)]; learned queries
// score all states, softmax over the depth, then mix the normalized states back.
static inline void inkling_mix(const float *const *blocks, int nb, const float *cur, const float *query,
                               int nq, float *out, int dim) {
  int ns = nb + 1;
  float *states = malloc((size_t)ns * dim * sizeof(float));
  float *sc = malloc((size_t)nq * ns * sizeof(float));
  for (int s = 0; s < nb; s++) inkling_rms0(states + (size_t)s * dim, blocks[nb - 1 - s], dim);
  inkling_rms0(states + (size_t)nb * dim, cur, dim);
  const float scale = 1.0f / sqrtf((float)dim);
  for (int q = 0; q < nq; q++) {
    const float *qv = query + (size_t)q * dim;
    for (int s = 0; s < ns; s++) {
      float v = 0.0f;
      for (int i = 0; i < dim; i++) v += states[(size_t)s * dim + i] * qv[i];
      sc[(size_t)q * ns + s] = v * scale;
    }
    float m = -FLT_MAX;
    for (int s = 0; s < ns; s++) m = sc[(size_t)q * ns + s] > m ? sc[(size_t)q * ns + s] : m;
    float z = 0.0f;
    for (int s = 0; s < ns; s++) { float e = expf(sc[(size_t)q * ns + s] - m); sc[(size_t)q * ns + s] = e; z += e; }
    for (int i = 0; i < dim; i++) out[(size_t)q * dim + i] = 0;
    for (int s = 0; s < ns; s++) {
      float w = sc[(size_t)q * ns + s] / z;
      for (int i = 0; i < dim; i++) out[(size_t)q * dim + i] += w * states[(size_t)s * dim + i];
    }
  }
  free(states); free(sc);
}

// One causal 4-tap short convolution step. hist is [channels][k-1] (oldest first), weight is
// [channels][k]; out[c] = sum_j x[j]*w[c][j] over the window past + current.
static inline void inkling_sconv(float *out, const float *cur, int ch, const float *w, float *hist, int k) {
  for (int c = 0; c < ch; c++) {
    float *h = hist + (size_t)c * (k - 1);
    const float *wc = w + (size_t)c * k;
    float v = 0.0f;
    for (int j = 0; j < k - 1; j++) v += h[j] * wc[j];
    v += cur[c] * wc[k - 1];
    out[c] = v;
    for (int j = 0; j < k - 2; j++) h[j] = h[j + 1];
    h[k - 2] = cur[c];
  }
}

// Full-attention short conv: static_w is [k][channels] (tap-major, as stored in HF) and gets the
// reversed-tap order of the upstream conv1d, plus one input-dependent tap per lag and KV head.
static inline void inkling_sconv_full(float *out, const float *cur, int ch, const float *sw,
                                      const float *dyn, int nheads, float *hist, int k) {
  const int hd = ch / nheads;
  for (int c = 0; c < ch; c++) {
    float *h = hist + (size_t)c * (k - 1);
    float v = 0.0f;
    for (int j = 0; j < k - 1; j++) v += h[j] * sw[(size_t)(k - 1 - j) * ch + c];
    v += cur[c] * sw[c];
    for (int lag = 0; lag < k; lag++) {
      int xs = k - 1 - lag;
      float x = xs < k - 1 ? h[xs] : cur[c];
      v += x * dyn[(size_t)lag * nheads + c / hd];
    }
    out[c] = v;
    for (int j = 0; j < k - 2; j++) h[j] = h[j + 1];
    h[k - 2] = cur[c];
  }
}

// x * min(1, threshold/|x|), optionally after a per-element scale (L2 clip of a head)
static inline void inkling_clip(float *x, const float *scale, int n, float threshold) {
  float ss = 0.0f;
  for (int i = 0; i < n; i++) {
    if (scale) x[i] *= scale[i];
    ss += x[i] * x[i];
  }
  float norm = sqrtf(ss);
  if (norm > threshold) {
    float f = threshold / norm;
    for (int i = 0; i < n; i++) x[i] *= f;
  }
}

static inline void inkling_normw(float *o, const float *x, const float *w, int n, float eps) {
  float ss = 0.0f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float sc = 1.0f / sqrtf(ss / n + eps);
  for (int i = 0; i < n; i++) o[i] = x[i] * sc * w[i];
}

static inline const InkTensor *ink_t(const InklingModel *m, int layer, const char *name) {
  char b[256];
  snprintf(b, sizeof b, "layers.%d.%s", layer, name);
  const InkTensor *t = inkling_find(m, b);
  if (!t) { fprintf(stderr, "missing tensor %s\n", b); exit(1); }
  return t;
}
static inline const float *ink_f32(const InklingModel *m, int layer, const char *name) {
  return (const float *)ink_t(m, layer, name)->data;
}
static inline QuantizedTensor ink_pack(const InklingModel *m, int layer, const char *name) {
  QuantizedTensor t;
  if (!inkling_pack_2d(ink_t(m, layer, name), &t)) { fprintf(stderr, "pack failed %s\n", name); exit(1); }
  return t;
}

static inline int inkling_pack_expert(const InkTensor *t, int e, QuantizedTensor *out);
typedef struct { char name[160]; QuantizedTensor t; } InkPackEntry;
static InkPackEntry *g_ink_packs;
static int g_ink_npacks, g_ink_pcap;

static inline QuantizedTensor *ink_pack_cached(const InklingModel *m, int layer, const char *name) {
  char full[160];
  snprintf(full, sizeof full, "layers.%d.%s", layer, name);
  for (int i = 0; i < g_ink_npacks; i++) if (!strcmp(g_ink_packs[i].name, full)) return &g_ink_packs[i].t;
  if (g_ink_npacks == g_ink_pcap) {
    g_ink_pcap = g_ink_pcap ? g_ink_pcap * 2 : 64;
    g_ink_packs = realloc(g_ink_packs, (size_t)g_ink_pcap * sizeof(*g_ink_packs));
  }
  InkPackEntry *e = &g_ink_packs[g_ink_npacks++];
  snprintf(e->name, sizeof e->name, "%s", full);
  if (!inkling_pack_2d(ink_t(m, layer, name), &e->t)) { fprintf(stderr, "pack failed %s\n", full); exit(1); }
  return &e->t;
}
static inline QuantizedTensor *ink_pack_expert_cached(const InklingModel *m, int layer, const char *kind, int ex) {
  char full[160];
  snprintf(full, sizeof full, "layers.%d.mlp.experts.%s#%d", layer, kind, ex);
  for (int i = 0; i < g_ink_npacks; i++) if (!strcmp(g_ink_packs[i].name, full)) return &g_ink_packs[i].t;
  if (g_ink_npacks == g_ink_pcap) {
    g_ink_pcap = g_ink_pcap ? g_ink_pcap * 2 : 64;
    g_ink_packs = realloc(g_ink_packs, (size_t)g_ink_pcap * sizeof(*g_ink_packs));
  }
  InkPackEntry *e = &g_ink_packs[g_ink_npacks++];
  snprintf(e->name, sizeof e->name, "%s", full);
  char nm[96]; snprintf(nm, sizeof nm, "mlp.experts.%s", kind);
  if (!inkling_pack_expert(ink_t(m, layer, nm), ex, &e->t)) { fprintf(stderr, "pack failed %s\n", full); exit(1); }
  return &e->t;
}
static inline void ink_cache_free(void) {
  for (int i = 0; i < g_ink_npacks; i++) inkling_pack_free(&g_ink_packs[i].t);
  free(g_ink_packs);
  g_ink_packs = NULL; g_ink_npacks = g_ink_pcap = 0;
}

static inline void inkling_attn(float *out, const float *q, const float *k, const float *v,
                                int hd, int pos, int window, const float *rel_bias, const float *r,
                                const float *sink, int d_rel, int extent, float *scores);
static inline void inkling_moe(const InklingModel *m, int layer, const float *x, float *out);

// Loop control: gate = rms_norm([cur; msg]); out = cur - tanh(Wf gate)*cur + tanh(Wg gate)*msg.
static inline void inkling_control(const InklingModel *m, int layer, int ctrl, const float *cur,
                                   const float *msg, float *out) {
  static const char *cn[4] = {"attn", "k", "v", "mlp"};
  const int dim = m->dim, rank = m->gate_rank;
  char nm[128];
  float *cat = malloc((size_t)2 * dim * sizeof(float));
  memcpy(cat, cur, (size_t)dim * sizeof(float));
  memcpy(cat + dim, msg, (size_t)dim * sizeof(float));
  inkling_rms0(cat, cat, 2 * dim);
  int8_t *cq = malloc(2 * dim); float *cs = malloc((size_t)(2 * dim / GS) * sizeof(float));
  quantize_vec(cq, cs, cat, 2 * dim);
  snprintf(nm, sizeof nm, "controllers.%s.down_weight", cn[ctrl]);
  QuantizedTensor *wd = ink_pack_cached(m, layer, nm);
  float *g = malloc((size_t)rank * sizeof(float));
  matmul_rows_q8(g, cq, cs, wd, 2 * dim, 0, rank);
  int8_t *gq = malloc(rank); float *gs = malloc((size_t)(rank / GS) * sizeof(float));
  quantize_vec(gq, gs, g, rank);
  snprintf(nm, sizeof nm, "controllers.%s.gain_weight", cn[ctrl]);
  QuantizedTensor *wg = ink_pack_cached(m, layer, nm);
  snprintf(nm, sizeof nm, "controllers.%s.forget_weight", cn[ctrl]);
  QuantizedTensor *wf = ink_pack_cached(m, layer, nm);
  float *gain = malloc((size_t)dim * sizeof(float)), *forget = malloc((size_t)dim * sizeof(float));
  matmul_rows_q8(gain, gq, gs, wg, rank, 0, dim);
  matmul_rows_q8(forget, gq, gs, wf, rank, 0, dim);
  for (int i = 0; i < dim; i++) {
    float gg = tanhf(gain[i]), ff = tanhf(forget[i]);
    out[i] = cur[i] - ff * cur[i] + gg * msg[i];
  }
  free(cat); free(cq); free(cs); free(g); free(gq); free(gs); free(gain); free(forget);
}

// One SWA layer (no loop messages). Query positions are stored in state->kf/vf at `slot,pos`;
// `ip` is the visit index used for the 6-layer attention-residual blocks.
static inline void inkling_layer_swa(const InklingModel *m, InklingState *s, int layer, int slot, int ip,
                                     int pos, const float *cur, const float *attn_query, const float *ffn_query,
                                     float **blocks, int *nblocks, int *balloc, float *blockmem, int full,
                                     const float *messages, float *out) {
  const int dim = m->dim, heads = m->heads, kvh_n = m->kv_heads, hd = m->head_dim;
  const int kvd = kvh_n * hd, qd = heads * hd, k = m->conv_k, drel = m->d_rel, ext = m->rel_extent;
  const float inv_post = 1.0f / sqrtf((float)k + 1.0f);
  const float inv_conv = full ? 1.0f / sqrtf((float)2 * k + 1.0f) : inv_post;
  const int nq = full ? 3 : 1;
  float *inpSA = malloc((size_t)dim * sizeof(float));
  memcpy(inpSA, cur, (size_t)dim * sizeof(float));
  float *in = malloc((size_t)3 * dim * sizeof(float));
  if (attn_query) {
    float *mix = malloc((size_t)nq * dim * sizeof(float));
    inkling_mix((const float *const *)blocks, *nblocks, cur, attn_query, nq, mix, dim);
    for (int j = 0; j < nq; j++) memcpy(in + (size_t)j * dim, mix + (size_t)j * dim, (size_t)dim * sizeof(float));
    for (int j = nq; j < 3; j++) memcpy(in + (size_t)j * dim, in, (size_t)dim * sizeof(float));
    free(mix);
  } else for (int j = 0; j < 3; j++) memcpy(in + (size_t)j * dim, cur, (size_t)dim * sizeof(float));
  if (messages) {
    float *c = malloc((size_t)dim * sizeof(float));
    for (int j = 0; j < 3; j++) {
      inkling_control(m, layer, j, in + (size_t)j * dim, messages + (size_t)j * dim, c);
      memcpy(in + (size_t)j * dim, c, (size_t)dim * sizeof(float));
    }
    free(c);
  }
  const int sep = full || messages;
  const int nin = sep ? 3 : 1;
  float *xn = malloc((size_t)3 * dim * sizeof(float));
  int8_t *xq[3]; float *xs[3];
  for (int j = 0; j < nin; j++) {
    const char *wn = full ? (j == 0 ? "q_input_norm.weight" : j == 1 ? "k_input_norm.weight" : "v_input_norm.weight")
                          : "attn_norm.weight";
    inkling_normw(xn + (size_t)j * dim, in + (size_t)j * dim, ink_f32(m, layer, wn), dim, m->eps);
    xq[j] = malloc(dim); xs[j] = malloc((size_t)(dim / GS) * sizeof(float));
    quantize_vec(xq[j], xs[j], xn + (size_t)j * dim, dim);
  }
  for (int j = nin; j < 3; j++) { xq[j] = xq[0]; xs[j] = xs[0]; }
  float *q = malloc((size_t)qd * sizeof(float)), *kr = malloc((size_t)kvd * sizeof(float));
  float *vr = malloc((size_t)kvd * sizeof(float)), *r = malloc((size_t)(heads * drel) * sizeof(float));
  float *gate = malloc((size_t)qd * sizeof(float)), *sink = malloc((size_t)heads * sizeof(float));
  QuantizedTensor *wq = ink_pack_cached(m, layer, "attn.wq_du.weight");
  QuantizedTensor *wk = ink_pack_cached(m, layer, "attn.wk_dv.weight");
  QuantizedTensor *wv = ink_pack_cached(m, layer, "attn.wv_dv.weight");
  QuantizedTensor *wr = ink_pack_cached(m, layer, "attn.wr_du.weight");
  QuantizedTensor *wg = ink_pack_cached(m, layer, "attn.gate_attn.weight");
  QuantizedTensor *ws = ink_pack_cached(m, layer, "attn.attn_sink_proj.weight");
  matmul_rows_q8(q, xq[0], xs[0], wq, dim, 0, qd);
  matmul_rows_q8(kr, xq[1], xs[1], wk, dim, 0, kvd);
  matmul_rows_q8(vr, xq[2], xs[2], wv, dim, 0, kvd);
  matmul_rows_q8(r, xq[0], xs[0], wr, dim, 0, heads * drel);
  matmul_rows_q8(gate, xq[0], xs[0], wg, dim, 0, qd);
  matmul_rows_q8(sink, xq[0], xs[0], ws, dim, 0, heads);

  float *hist = s->conv + (size_t)slot * (2 * kvd + 2 * dim) * (k - 1);
  float *kc = malloc((size_t)kvd * sizeof(float)), *vc = malloc((size_t)kvd * sizeof(float));
  if (full) {
    QuantizedTensor *kp = ink_pack_cached(m, layer, "attn.k_sconv.weight_proj");
    QuantizedTensor *vp = ink_pack_cached(m, layer, "attn.v_sconv.weight_proj");
    float *dk = malloc((size_t)k * kvh_n * sizeof(float)), *dv = malloc((size_t)k * kvh_n * sizeof(float));
    matmul_rows_q8(dk, xq[1], xs[1], kp, dim, 0, k * kvh_n);
    matmul_rows_q8(dv, xq[2], xs[2], vp, dim, 0, k * kvh_n);
    inkling_sconv_full(kc, kr, kvd, ink_f32(m, layer, "attn.k_sconv.static_w"), dk, kvh_n, hist, k);
    inkling_sconv_full(vc, vr, kvd, ink_f32(m, layer, "attn.v_sconv.static_w"), dv, kvh_n,
                       hist + (size_t)kvd * (k - 1), k);
    free(dk); free(dv);
  } else {
    inkling_sconv(kc, kr, kvd, ink_f32(m, layer, "attn.k_sconv.weight"), hist, k);
    inkling_sconv(vc, vr, kvd, ink_f32(m, layer, "attn.v_sconv.weight"), hist + (size_t)kvd * (k - 1), k);
  }
  for (int i = 0; i < kvd; i++) { kr[i] = (kr[i] + kc[i]) * inv_conv; vr[i] = (vr[i] + vc[i]) * inv_conv; }
  free(kc); free(vc);
  for (int j = 0; j < nin; j++) { free(xq[j]); free(xs[j]); }

  const float *qclip = ink_f32(m, layer, "attn.q_l2_clip.scale");
  const float *rclip = ink_f32(m, layer, "attn.rel_l2_clip.scale");
  for (int h = 0; h < heads; h++) inkling_clip(q + (size_t)h * hd, qclip + (size_t)h * hd, hd, m->qk_clip);
  for (int g = 0; g < kvh_n; g++) {
    inkling_clip(kr + (size_t)g * hd, NULL, hd, m->qk_clip);
    inkling_rms0(vr + (size_t)g * hd, vr + (size_t)g * hd, hd);
  }
  for (int h = 0; h < heads; h++) inkling_clip(r + (size_t)h * drel, rclip + (size_t)h * drel, drel, m->rel_clip);
  if (full) {
    float tau = 1.0f + m->log_alpha * logf(fmaxf(1.0f, (pos + 1) / m->log_floor));
    for (int i = 0; i < qd; i++) q[i] *= tau;
    for (int i = 0; i < heads * drel; i++) r[i] *= tau;
  }

  float *kbase = s->kf + (size_t)slot * kvd * s->ctx;
  float *vbase = s->vf + (size_t)slot * kvd * s->ctx;
  for (int g = 0; g < kvh_n; g++) {
    memcpy(kbase + ((size_t)g * s->ctx + pos) * hd, kr + (size_t)g * hd, (size_t)hd * sizeof(float));
    memcpy(vbase + ((size_t)g * s->ctx + pos) * hd, vr + (size_t)g * hd, (size_t)hd * sizeof(float));
  }
  const float *rb = ink_f32(m, layer, "attn.rel_logits_proj.proj");
  float *att = malloc((size_t)qd * sizeof(float));
  float *scores = malloc((size_t)s->ctx * sizeof(float));
  const int window = (m->local_mask >> layer) & 1 ? m->window : 0;
  const int kv_per = heads / kvh_n;
  for (int h = 0; h < heads; h++) {
    int g = h / kv_per;
    inkling_attn(att + (size_t)h * hd, q + (size_t)h * hd, kbase + (size_t)g * s->ctx * hd,
                 vbase + (size_t)g * s->ctx * hd, hd, pos, window, rb, r + (size_t)h * drel,
                 sink + h, drel, ext, scores);
  }
  free(scores);
  for (int i = 0; i < qd; i++) att[i] *= 1.0f / (1.0f + expf(-gate[i]));

  int8_t *aq = malloc(qd); float *as = malloc((size_t)(qd / GS) * sizeof(float));
  quantize_vec(aq, as, att, qd);
  QuantizedTensor *wo = ink_pack_cached(m, layer, "attn.wo_ud.weight");
  float *woo = malloc((size_t)dim * sizeof(float));
  matmul_rows_q8(woo, aq, as, wo, qd, 0, dim);
  free(aq); free(as); free(att); free(q); free(kr); free(vr); free(r); free(gate); free(sink);

  float *post = malloc((size_t)dim * sizeof(float));
  inkling_sconv(post, woo, dim, ink_f32(m, layer, "attn_sconv.weight"), hist + (size_t)2 * kvd * (k - 1), k);
  for (int i = 0; i < dim; i++) woo[i] = (woo[i] + post[i]) * inv_post;
  if (ip % m->res_block == 0) {
    int bi = *balloc;
    memcpy(blockmem + (size_t)bi * dim, inpSA, (size_t)dim * sizeof(float));
    blocks[*nblocks] = blockmem + (size_t)bi * dim;
    (*nblocks)++; (*balloc)++;
    memcpy(out, woo, (size_t)dim * sizeof(float));
  } else {
    for (int i = 0; i < dim; i++) out[i] = inpSA[i] + woo[i];
  }
  free(woo); free(inpSA); free(in);

  float *ffn_inp = malloc((size_t)dim * sizeof(float));
  memcpy(ffn_inp, out, (size_t)dim * sizeof(float));
  float *mix = malloc((size_t)dim * sizeof(float));
  inkling_mix((const float *const *)blocks, *nblocks, out, ffn_query, 1, mix, dim);
  float *fmix = mix, *ctrl = NULL;
  if (messages) {
    ctrl = malloc((size_t)dim * sizeof(float));
    inkling_control(m, layer, 3, mix, messages + (size_t)3 * dim, ctrl);
    fmix = ctrl;
  }
  inkling_normw(xn, fmix, ink_f32(m, layer, "mlp_norm.weight"), dim, m->eps);
  free(ctrl);
  float *moe = malloc((size_t)dim * sizeof(float));
  inkling_moe(m, layer, xn, moe);
  inkling_sconv(post, moe, dim, ink_f32(m, layer, "mlp_sconv.weight"), hist + (size_t)(2 * kvd + dim) * (k - 1), k);
  for (int i = 0; i < dim; i++) out[i] = ffn_inp[i] + (moe[i] + post[i]) * inv_post;
  free(ffn_inp); free(mix); free(moe); free(post); free(xn);
}

// Attention for one query token and one head. K/V rows are f32 [kv_head][pos][hd]; the relative
// bias is [d_rel][extent], r is [heads][d_rel], sink is [heads]. score = q.k/sqrt(hd) +
// bias[head][pos-t] - sink[head]; the denominator includes the zero sink: 1 + sum exp(score).
static inline void inkling_attn(float *out, const float *q, const float *k, const float *v,
                                int hd, int pos, int window, const float *rel_bias, const float *r,
                                const float *sink, int d_rel, int extent, float *scores) {
  const float scale = 1.0f / sqrtf((float)hd);
  int t0 = 0;
  if (window > 0 && pos - t0 + 1 > window) t0 = pos - window + 1;
  float m = 0.0f; // the zero sink logit participates in the max/normalizer
  for (int t = t0; t <= pos; t++) {
    const float *kt = k + (size_t)t * hd;
    float s = 0.0f;
    for (int i = 0; i < hd; i++) s += q[i] * kt[i];
    s *= scale;
    int d = pos - t;
    if (rel_bias && d < extent) {
      float b = 0.0f;
      for (int j = 0; j < d_rel; j++) b += r[j] * rel_bias[(size_t)j * extent + d];
      s += b;
    }
    if (sink) s -= sink[0];
    scores[t - t0] = s;
    m = s > m ? s : m;
  }
  float z = expf(-m); // the zero sink logit
  for (int t = t0; t <= pos; t++) { float e = expf(scores[t - t0] - m); scores[t - t0] = e; z += e; }
  for (int i = 0; i < hd; i++) out[i] = 0;
  for (int t = t0; t <= pos; t++) {
    float w = scores[t - t0] / z;
    const float *vt = v + (size_t)t * hd;
    for (int i = 0; i < hd; i++) out[i] += w * vt[i];
  }
}

static inline int inkling_pack_expert(const InkTensor *t, int e, QuantizedTensor *out) {
  size_t stride = t->bytes / t->dims[0];
  InkTensor v = *t;
  v.ndim = 2; v.dims[0] = t->dims[1]; v.dims[1] = t->dims[2];
  v.data = (char *)t->data + (size_t)e * stride;
  v.bytes = stride;
  return inkling_pack_2d(&v, out);
}

// MoE: sigmoid router, top-8 selection (with the learned bias used only for selection), weights
// normalized and scaled by route_scale, situ(gate,up) with the 4/25 softcaps, Q4 experts.
static inline void inkling_moe(const InklingModel *m, int layer, const float *x, float *out) {
  char nm[96];
  snprintf(nm, sizeof nm, "layers.%d.mlp.gate.bias", layer);
  const InkTensor *rb = inkling_find(m, nm);
  const int E = m->experts, A = m->active, dim = m->dim, inter = m->inter;
  float *logits = malloc((size_t)E * sizeof(float));
  float *sel = malloc((size_t)E * sizeof(float));
  int *top = malloc((size_t)E * sizeof(int));
  QuantizedTensor *rq = ink_pack_cached(m, layer, "mlp.gate.weight");
  int8_t *xq = malloc(dim); float *xs = malloc((size_t)(dim / GS) * sizeof(float));
  quantize_vec(xq, xs, x, dim);
  matmul_rows_q8(logits, xq, xs, rq, dim, 0, E);
  for (int e = 0; e < E; e++) {
    float p = 1.0f / (1.0f + expf(-logits[e]));
    logits[e] = p;
    sel[e] = p + (rb ? ((const float *)rb->data)[e] : 0.0f);
    top[e] = e;
  }
  for (int i = 0; i < A; i++) { // partial selection sort is enough (A = 8)
    int b = i;
    for (int j = i + 1; j < E; j++) if (sel[top[j]] > sel[top[b]]) b = j;
    int t = top[i]; top[i] = top[b]; top[b] = t;
  }
  float z = 0.0f;
  for (int i = 0; i < A; i++) z += logits[top[i]];
  if (z <= 0.0f) z = 1.0f;
  float *act = malloc((size_t)inter * sizeof(float));
  float *gh = malloc((size_t)(2 * inter) * sizeof(float));
  for (int i = 0; i < dim; i++) out[i] = 0;
  for (int i = 0; i < A; i++) {
    int e = top[i];
    float w = logits[e] / z * m->route_scale;
    QuantizedTensor *q13 = ink_pack_expert_cached(m, layer, "w13", e);
    QuantizedTensor *q2 = ink_pack_expert_cached(m, layer, "w2", e);
    matmul_rows(gh, xq, xs, q13, dim, 0, 2 * inter);
    for (int j = 0; j < inter; j++) {
      // the source w13 rows are interleaved [gate_j, up_j]; the upstream converter blocks them
      float g = gh[2 * j], u = gh[2 * j + 1];
      float a = m->gate_softcap * tanhf(g / m->gate_softcap) / (1.0f + expf(-g));
      float b = m->up_softcap > 0 ? m->up_softcap * tanhf(u / m->up_softcap) : u;
      act[j] = a * b * w;
    }
    int8_t *aq = malloc(inter); float *as = malloc((size_t)(inter / GS) * sizeof(float));
    quantize_vec(aq, as, act, inter);
    float *o = malloc((size_t)dim * sizeof(float));
    matmul_rows(o, aq, as, q2, inter, 0, dim);
    for (int j = 0; j < dim; j++) out[j] += o[j];
    free(o); free(aq); free(as);
  }
  free(logits); free(sel); free(top); free(act); free(gh); free(xq); free(xs);
}

// LM head over the Q8 classifier, then the final logit softcap and the padded-vocab trim
static inline void inkling_lm_head(InklingRuntime *r, const float *hidden, float *logits);
static inline const float *inkling_aq(const InklingModel *m, int layer) {
  if (layer == 0) return NULL;
  char b[80];
  snprintf(b, sizeof b, "attn_res.attn_queries.%d", layer - 1);
  const InkTensor *t = inkling_find(m, b);
  if (!t) { fprintf(stderr, "missing %s\n", b); exit(1); }
  return (const float *)t->data;
}
static inline const float *inkling_fq(const InklingModel *m, int layer) {
  char b[80];
  snprintf(b, sizeof b, "attn_res.mlp_queries.%d", layer);
  const InkTensor *t = inkling_find(m, b);
  if (!t) { fprintf(stderr, "missing %s\n", b); exit(1); }
  return (const float *)t->data;
}

static inline void inkling_forward_token(const InklingModel *m, InklingState *s, InklingRuntime *r,
                                         int token, int pos, float *logits) {
  const int dim = m->dim;
  float *cur = malloc((size_t)dim * sizeof(float)), *next = malloc((size_t)dim * sizeof(float));
  float *pre = malloc((size_t)dim * sizeof(float)), *bankout = malloc((size_t)dim * sizeof(float));
  float *mix = malloc((size_t)dim * sizeof(float)), *outn = malloc((size_t)dim * sizeof(float));
  float **blocks = malloc(64 * sizeof(float *));
  float *blockmem = malloc((size_t)64 * dim * sizeof(float));
  int nb = 0, balloc = 0;
  inkling_embed_row(m, token, cur);
  inkling_normw(cur, cur, (const float *)inkling_find(m, "embed_norm.weight")->data, dim, m->eps);
  for (int l = 0; l < 6; l++) {
    int full = !((m->local_mask >> l) & 1);
    inkling_layer_swa(m, s, l, l, l, pos, cur, inkling_aq(m, l), inkling_fq(m, l),
                      blocks, &nb, &balloc, blockmem, full, NULL, next);
    float *t = cur; cur = next; next = t;
  }
  memcpy(pre, cur, (size_t)dim * sizeof(float));
  const int pre_nb = nb;
  float *bank[64];
  float *msgs = malloc((size_t)4 * dim * sizeof(float));
  float *msgq = malloc((size_t)4 * dim * sizeof(float));
  const InkTensor *cq = inkling_find(m, "middle_cycle_cross_pass_queries");
  const InkTensor *ckv = inkling_find(m, "middle_cycle_cross_pass_kv_queries");
  for (int pass = 0; pass < m->passes; pass++) {
    int nbank = 0;
    if (pass > 0) {
      for (int i = pre_nb + 1; i < nb; i++) bank[nbank++] = blocks[i];
      memcpy(bankout, cur, (size_t)dim * sizeof(float));
      memcpy(cur, pre, (size_t)dim * sizeof(float));
      nb = pre_nb;
    }
    for (int ip = m->loop_start; ip < m->loop_end; ip++) {
      int slot = pass == 0 ? ip : m->phys_layers + (pass - 1) * (m->loop_end - m->loop_start) + (ip - m->loop_start);
      const float *msg = NULL;
      if (pass > 0) {
        const float *q = (const float *)((const char *)cq->data + (size_t)(2 * (ip - m->loop_start)) * dim * 4);
        const float *kv = (const float *)((const char *)ckv->data + (size_t)(2 * (ip - m->loop_start)) * dim * 4);
        memcpy(msgq, q, (size_t)dim * sizeof(float));
        memcpy(msgq + dim, kv, (size_t)dim * sizeof(float));
        memcpy(msgq + 2 * dim, kv + dim, (size_t)dim * sizeof(float));
        memcpy(msgq + 3 * dim, q + dim, (size_t)dim * sizeof(float));
        inkling_mix((const float *const *)bank, nbank, bankout, msgq, 4, msgs, dim);
        msg = msgs;
      }
      int full = !((m->local_mask >> ip) & 1);
      inkling_layer_swa(m, s, ip, slot, ip, pos, cur, inkling_aq(m, ip), inkling_fq(m, ip),
                        blocks, &nb, &balloc, blockmem, full, msg, next);
      float *t = cur; cur = next; next = t;
    }
  }
  for (int l = m->loop_end; l < m->phys_layers; l++) {
    int full = !((m->local_mask >> l) & 1);
    inkling_layer_swa(m, s, l, l, l, pos, cur, inkling_aq(m, l), inkling_fq(m, l),
                      blocks, &nb, &balloc, blockmem, full, NULL, next);
    float *t = cur; cur = next; next = t;
  }
  inkling_mix((const float *const *)blocks, nb, cur, (const float *)inkling_find(m, "attn_res.output_query")->data,
              1, mix, dim);
  inkling_normw(outn, mix, (const float *)inkling_find(m, "norm.weight")->data, dim, m->eps);
  inkling_lm_head(r, outn, logits);
  free(cur); free(next); free(pre); free(bankout); free(mix); free(outn); free(blocks); free(blockmem); free(msgs); free(msgq);
}

static inline void inkling_lm_head(InklingRuntime *r, const float *hidden, float *logits) {
  const InklingModel *m = r->m;
  int8_t *xq = malloc(m->dim);
  float *xs = malloc((size_t)(m->dim / GS) * sizeof(float));
  float *all = malloc((size_t)m->vocab * sizeof(float));
  quantize_vec(xq, xs, hidden, m->dim);
  matmul_rows_q8(all, xq, xs, &r->unembed, m->dim, 0, m->vocab);
  const float cap = m->final_softcap;
  for (int i = 0; i < m->unpadded; i++) logits[i] = cap * tanhf(all[i] / cap);
  free(all); free(xs); free(xq);
}
