/* GLM-5.3-Flash (arch 4, model_type glm5_next): the text model of Glm5NextForConditionalGeneration.
 *
 * 45 layers, each: [attention block, FFN block], both wrapped in manifold-constrained hyper-connections
 * (mHC): the residual stream is hc_mult (4) streams of dim; before a block they are collapsed into its
 * input with weights `pre`, after it its output is spread back with `post` and the old streams are
 * mixed with a doubly stochastic matrix `comb`; the three come from a small projection of the streams.
 *   attention: 34 layers Kimi Delta Attention (KDA: gated delta rule with a per-channel decay, short
 *              causal convs on q/k/v, gated RMSNorm on the output), every 4th layer (11) MLA without
 *              RoPE (latent KV cache of kv_lora_rank values per token, absorbed attention) with the
 *              DSA indexer that picks the top-k pools of 4 tokens once the context has more than top-k
 *              tokens (below that it keeps every token: plain causal attention).
 *   FFN: 3 dense layers, then MoE (288 experts, 8 active, sigmoid router with selection bias, one
 *        shared expert), SwiGLU with clamped inputs.
 * File layout: scripts/export_glm5next.py. Included by runq.c (uses its runtime). */

enum { GK_LIN_IN, GK_F_B, GK_G_B, GK_LIN_OUT, GK_MLA_IN, GK_Q_B, GK_KV_BK, GK_KV_BV, GK_O_PROJ, GK_W13, GK_W2,
       GK_SH13, GK_SH2, GK_EX1, GK_EX3, GK_EX2, GK_ROUTER, GK_EMBED, GK_LM_HEAD };

typedef struct GlmWeights {
    float *in_norm, *post_norm, *final_norm;   // (L, D), (L, D), (D)
    float *hc_fn, *hc_base, *hc_scale;         // (L, 2, 24, hc D), (L, 2, 24), (L, 2, 4)
    float *conv, *dt_bias, *A, *o_norm;        // linear: (LL, 3 H dh, K), (LL, H dh), (LL, H) exp(A_log), (LL, dh)
    float *qa_norm, *kva_norm, *ik_w, *ik_b, *ape; // attention: (LA, ql), (LA, kvl), (LA, ih) x 2, (LA, kpool, ih)
    float *rbias;                              // (LM, E) router selection bias
    QuantizedTensor *embed, *lin_in, *f_b, *g_b, *lin_out, *mla_in, *q_b, *kv_bk, *kv_bv, *o_proj;
    QuantizedTensor *w13, *w2, *router, *sh13, *sh2, *ex1, *ex3, *ex2, *lm_head;
    QuantizedTensor *router_node[MAX_NODES];   // node-local copies of the routers
} GlmWeights;

typedef struct GlmState {
    float* hs;      // (hc, D) residual streams
    float* xc;      // (D) collapsed block input
    float* y;       // (D) block output
    float* lin;     // (lin_in rows) linear: q k v (after conv + SiLU) | f_a | g_a | beta
    float* dec;     // (H dh) per-channel decay exp(g)
    float* gate;    // (H dh) output gate (pre-sigmoid)
    float* lo;      // (H dh) recurrence output
    float* mla;     // (mla_in rows) q_a | kv_a | indexer k | indexer head weights | k-pool gate
    float* qb;      // (nh qk + ihh ih) q heads | indexer q heads
    float* qabs;    // (nh kvl) absorbed queries W_k^T q
    float* ov;      // (nh vh) per-head values W_v ctx
    float* h1;      // ((A + 1) mh) or F: FFN hidden
    float* h3;
    float* hcp;     // (nth, 32) mHC partial dot products
    float* ikey;    // (LA, seq, ih) indexer keys
    float* igate;   // (LA, seq, ih) indexer k-pool gate scores
    float* ipool;   // (LA, seq / kpool, ih) pooled indexer keys of complete pools
    float* iscore;  // (seq / kpool) pool scores of the current token
    int* isel;      // (seq) selected tokens (sorted), -1 terminated
    int nsel;
} GlmState;

#define GLM_PH 12 // work-counter phases per layer
static int GLM_LAYERS = 0; // > 0: run only the first n layers (debug: compare with a truncated reference)
static int GLM_DEBUG_SEL = 0; // debug: print the tokens the indexer of the first attention layer selects

static inline int glm_is_attn(const Config* p, int l) { return (p->g_amask[l >> 5] >> (l & 31)) & 1; }

static QuantizedTensor* glm_qt(void** ptr, int n, size_t rows, size_t cols, int bits) {
    QuantizedTensor* res = malloc(n * sizeof(QuantizedTensor));
    char* c = *ptr;
    for (int i = 0; i < n; i++) {
        res[i].bits = bits;
        res[i].q = (int8_t*)c;
        c += bits == 4 ? rows * cols / 2 : rows * cols;
        res[i].s = (float*)c;
        c += (bits == 4 ? rows * cols / 32 : rows * cols / GS) * sizeof(float);
    }
    *ptr = c;
    return res;
}

static void glm_map_weights(TransformerWeights* tw, Config* p, void* ptr) {
    GlmWeights* w = calloc(1, sizeof(GlmWeights));
    tw->glm = w;
    const size_t D = p->dim, L = p->n_layers, LA = p->n_attn, LL = L - LA, LM = L - p->g_dense;
    const size_t H = p->g_H, dh = p->g_dh, K = p->g_K, nh = p->n_heads;
    float* f = ptr;
#define TAKE(dst, n) do { dst = f; f += (n); } while (0)
    TAKE(w->in_norm, L * D); TAKE(w->post_norm, L * D); TAKE(w->final_norm, D);
    TAKE(w->hc_fn, L * 2 * 24 * p->g_hc * D); TAKE(w->hc_base, L * 2 * 24); TAKE(w->hc_scale, L * 2 * 4);
    TAKE(w->conv, LL * 3 * H * dh * K); TAKE(w->dt_bias, LL * H * dh); TAKE(w->A, LL * H); TAKE(w->o_norm, LL * dh);
    TAKE(w->qa_norm, LA * p->g_ql); TAKE(w->kva_norm, LA * p->g_kvl); TAKE(w->ik_w, LA * p->g_ih);
    TAKE(w->ik_b, LA * p->g_ih); TAKE(w->ape, LA * p->g_kpool * p->g_ih); TAKE(w->rbias, LM * p->n_experts);
#undef TAKE
    void* q = f;
#define B(k) ((p->g_q4 >> (k)) & 1 ? 4 : 8)
    w->embed = glm_qt(&q, 1, p->vocab_size, D, B(GK_EMBED));
    w->lin_in = glm_qt(&q, LL, 3 * H * dh + 2 * dh + H, D, B(GK_LIN_IN));
    w->f_b = glm_qt(&q, LL, H * dh, dh, B(GK_F_B));
    w->g_b = glm_qt(&q, LL, H * dh, dh, B(GK_G_B));
    w->lin_out = glm_qt(&q, LL, D, H * dh, B(GK_LIN_OUT));
    w->mla_in = glm_qt(&q, LA, p->g_ql + p->g_kvl + 2 * p->g_ih + p->g_ihh, D, B(GK_MLA_IN));
    w->q_b = glm_qt(&q, LA, nh * p->g_qk + (size_t)p->g_ihh * p->g_ih, p->g_ql, B(GK_Q_B));
    w->kv_bk = glm_qt(&q, LA, nh * p->g_kvl, p->g_qk, B(GK_KV_BK));
    w->kv_bv = glm_qt(&q, LA, nh * p->g_vh, p->g_kvl, B(GK_KV_BV));
    w->o_proj = glm_qt(&q, LA, D, nh * p->g_vh, B(GK_O_PROJ));
    w->w13 = glm_qt(&q, p->g_dense, 2 * (size_t)p->hidden_dim, D, B(GK_W13));
    w->w2 = glm_qt(&q, p->g_dense, D, p->hidden_dim, B(GK_W2));
    w->router = glm_qt(&q, LM, p->n_experts, D, 8);
    w->sh13 = glm_qt(&q, LM, 2 * (size_t)p->moe_hidden, D, B(GK_SH13));
    w->sh2 = glm_qt(&q, LM, D, p->moe_hidden, B(GK_SH2));
    w->ex1 = glm_qt(&q, LM, (size_t)p->n_experts * p->moe_hidden, D, B(GK_EX1));
    w->ex3 = glm_qt(&q, LM, (size_t)p->n_experts * p->moe_hidden, D, B(GK_EX3));
    w->ex2 = glm_qt(&q, LM, (size_t)p->n_experts * D, p->moe_hidden, B(GK_EX2));
    w->lm_head = glm_qt(&q, 1, p->vocab_size, D, B(GK_LM_HEAD));
#undef B
    for (int k = 0; k < MAX_NODES; k++) w->router_node[k] = w->router;
    // the generic code uses these
    tw->q_tokens = w->embed;
    tw->wcls = w->lm_head;
    tw->token_embedding_table = malloc((size_t)p->vocab_size * D * sizeof(float));
}

/* every quantized tensor of the model: n tensors of rows x cols (for repacking and NUMA placement);
 * split: rows per block that is split between the nodes like a matrix of its own (experts, gate|up halves) */
typedef struct { QuantizedTensor* t; int n; size_t rows, cols, split; } GlmTensor;
static int glm_tensors(const Config* p, const GlmWeights* w, GlmTensor* o) {
    const size_t D = p->dim, LA = p->n_attn, LL = p->n_layers - LA, LM = p->n_layers - p->g_dense;
    const size_t H = p->g_H, dh = p->g_dh, nh = p->n_heads, mh = p->moe_hidden, E = p->n_experts;
    int n = 0;
#define T(t_, n_, r_, c_, s_) o[n++] = (GlmTensor){t_, (int)(n_), r_, c_, s_}
    T(w->lin_in, LL, 3 * H * dh + 2 * dh + H, D, 0);
    T(w->f_b, LL, H * dh, dh, 0); T(w->g_b, LL, H * dh, dh, 0); T(w->lin_out, LL, D, H * dh, 0);
    T(w->mla_in, LA, p->g_ql + p->g_kvl + 2 * p->g_ih + p->g_ihh, D, 0);
    T(w->q_b, LA, nh * p->g_qk + (size_t)p->g_ihh * p->g_ih, p->g_ql, 0);
    T(w->kv_bk, LA, nh * p->g_kvl, p->g_qk, 0); T(w->kv_bv, LA, nh * p->g_vh, p->g_kvl, 0);
    T(w->o_proj, LA, D, nh * p->g_vh, 0);
    T(w->w13, p->g_dense, 2 * (size_t)p->hidden_dim, D, p->hidden_dim); T(w->w2, p->g_dense, D, p->hidden_dim, 0);
    T(w->router, LM, E, D, 0);
    T(w->sh13, LM, 2 * mh, D, mh); T(w->sh2, LM, D, mh, 0);
    T(w->ex1, LM, E * mh, D, mh); T(w->ex3, LM, E * mh, D, mh); T(w->ex2, LM, E * D, mh, D);
    T(w->lm_head, 1, p->vocab_size, D, 0);
#undef T
    return n;
}

#if defined(__linux__)
static void glm_place(const Config* p, GlmWeights* w) {
    GlmTensor t[32];
    int n = glm_tensors(p, w, t);
    for (int i = 0; i < n; i++) {
        if (t[i].t == w->router) continue; // replicated on every node instead
        if (t[i].split) place_experts(t[i].t, t[i].n, (int)(t[i].rows / t[i].split), (int)t[i].split, (int)t[i].cols);
        else place_tensor(t[i].t, t[i].n, (int)t[i].rows, (int)t[i].cols);
    }
}
#endif

static void glm_pack(const Config* p, GlmWeights* w) {
    GlmTensor t[32];
    int n = glm_tensors(p, w, t), nj = 0;
    for (int i = 0; i < n; i++) nj += t[i].n;
    PackJob* jobs = malloc(nj * sizeof(PackJob));
    pthread_t* th = malloc(nj * sizeof(pthread_t));
    int m = 0;
    for (int i = 0; i < n; i++)
        for (int l = 0; l < t[i].n; l++) jobs[m++] = (PackJob){t[i].t + l, (int)t[i].rows, (int)t[i].cols};
    for (int a = 0; a < m; a += 128) {
        int e = a + 128 < m ? a + 128 : m;
        for (int k = a; k < e; k++) pthread_create(&th[k], NULL, pack_job, &jobs[k]);
        for (int k = a; k < e; k++) pthread_join(th[k], NULL);
    }
    free(jobs);
    free(th);
    // node-local router copies
    const int LM = p->n_layers - p->g_dense;
    size_t qb = (size_t)p->n_experts * p->dim, sb = qb / GS * sizeof(float), per = ((qb + sb) + 4095) & ~(size_t)4095;
    for (int k = 0; k < NN && NN > 1; k++) {
        QuantizedTensor* c = malloc(LM * sizeof(QuantizedTensor));
        char* mem = aligned_alloc(4096, per * LM);
#if defined(__linux__)
        bind_range(mem, per * LM, NODE_ID[k]);
#endif
        for (int l = 0; l < LM; l++) {
            c[l] = (QuantizedTensor){(int8_t*)(mem + per * l), (float*)(mem + per * l + qb), 8};
            memcpy(c[l].q, w->router[l].q, qb);
            memcpy(c[l].s, w->router[l].s, sb);
        }
        w->router_node[k] = c;
    }
}

static void glm_alloc_state(RunState* s, const Config* p, int nth) {
    GlmState* g = calloc(1, sizeof(GlmState));
    s->glm = g;
    const size_t D = p->dim, H = p->g_H, dh = p->g_dh, LA = p->n_attn, T = p->seq_len;
    size_t f = (size_t)(p->n_active + 1) * p->moe_hidden;
    if ((size_t)p->hidden_dim > f) f = p->hidden_dim;
    g->hs = calloc((size_t)p->g_hc * D, sizeof(float));
    g->xc = calloc(D, sizeof(float));
    g->y = calloc(D, sizeof(float));
    g->lin = calloc(3 * H * dh + 2 * dh + H, sizeof(float));
    g->dec = calloc(H * dh, sizeof(float));
    g->gate = calloc(H * dh, sizeof(float));
    g->lo = calloc(H * dh, sizeof(float));
    g->mla = calloc(p->g_ql + p->g_kvl + 2 * p->g_ih + p->g_ihh, sizeof(float));
    g->qb = calloc((size_t)p->n_heads * p->g_qk + (size_t)p->g_ihh * p->g_ih, sizeof(float));
    g->qabs = calloc((size_t)p->n_heads * p->g_kvl, sizeof(float));
    g->ov = calloc((size_t)p->n_heads * p->g_vh, sizeof(float));
    g->h1 = calloc(f, sizeof(float));
    g->h3 = calloc(f, sizeof(float));
    g->hcp = calloc((size_t)(nth > 0 ? nth : 1) * 32, sizeof(float));
    g->ikey = malloc(LA * T * p->g_ih * sizeof(float));
    g->igate = malloc(LA * T * p->g_ih * sizeof(float));
    g->ipool = malloc(LA * (T / p->g_kpool + 1) * p->g_ih * sizeof(float));
    g->iscore = malloc(2 * (T / p->g_kpool + 1) * sizeof(float));
    g->isel = malloc((T + 8) * sizeof(int));
    if (!g->hs || !g->h1 || !g->ikey || !g->igate || !g->ipool || !g->isel) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
}

// ---------------------------------------------------------------------------- mHC
/* Partial products of the mHC projection over this thread's slice of the hc*D stream values:
 * 24 dot products with fn and the sum of squares (the RMS norm is applied afterwards). */
static void glm_hc_partial(const Config* p, const float* fn, const float* hs, float* out, int tid, int nth) {
    const int N = p->g_hc * p->dim;
    int c0 = (int)((long)N * tid / nth) & ~3, c1 = tid == nth - 1 ? N : (int)((long)N * (tid + 1) / nth) & ~3;
    float acc[25] = {0};
    for (int j = c0; j < c1; j++) acc[24] += hs[j] * hs[j];
    for (int r = 0; r < 24; r++) {
        const float* row = fn + (size_t)r * N;
        float a = 0.0f;
        for (int j = c0; j < c1; j++) a += row[j] * hs[j];
        acc[r] = a;
    }
    memcpy(out, acc, sizeof(acc));
}
/* From the partials of all threads: pre (hc), post (hc), comb (hc x hc, comb[m][n]: weight of input
 * stream m in output stream n). Identical in every thread. */
static void glm_hc_weights(const Config* p, const float* hcp, int nth, const float* base, const float* scale,
                           float* pre, float* post, float* comb) {
    const int hc = p->g_hc, N = hc * p->dim;
    double acc[25] = {0};
    for (int t = 0; t < nth; t++)
        for (int r = 0; r < 25; r++) acc[r] += hcp[(size_t)t * 32 + r];
    const float rinv = 1.0f / sqrtf((float)(acc[24] / N) + NORM_EPS);
    float y[24];
    for (int r = 0; r < 24; r++) y[r] = (float)acc[r] * rinv;
    const float eps = p->g_hc_eps;
    for (int n = 0; n < hc; n++) {
        pre[n] = 1.0f / (1.0f + expf(-(y[n] * scale[0] + base[n]))) + eps;
        post[n] = 2.0f / (1.0f + expf(-(y[hc + n] * scale[1] + base[hc + n])));
    }
    for (int m = 0; m < hc; m++) { // softmax over the last dim, + eps
        float* row = comb + m * hc;
        float mx = -FLT_MAX;
        for (int n = 0; n < hc; n++) { row[n] = y[2 * hc + m * hc + n] * scale[2] + base[2 * hc + m * hc + n]; mx = row[n] > mx ? row[n] : mx; }
        float z = 0.0f;
        for (int n = 0; n < hc; n++) { row[n] = expf(row[n] - mx); z += row[n]; }
        for (int n = 0; n < hc; n++) row[n] = row[n] / z + eps;
    }
    // Sinkhorn: columns (sum over dim -2), then (iters - 1) x (rows, columns)
    for (int it = 0; it < p->g_sk; it++) {
        if (it > 0)
            for (int m = 0; m < hc; m++) {
                float s = 0.0f;
                for (int n = 0; n < hc; n++) s += comb[m * hc + n];
                for (int n = 0; n < hc; n++) comb[m * hc + n] /= s + eps;
            }
        for (int n = 0; n < hc; n++) {
            float s = 0.0f;
            for (int m = 0; m < hc; m++) s += comb[m * hc + n];
            for (int m = 0; m < hc; m++) comb[m * hc + n] /= s + eps;
        }
    }
}
// rows [r0, r1) of the streams after a block with output y: H'[n] = post[n] y + sum_m comb[m][n] H[m]
static inline void glm_hc_update(const Config* p, float* hs, const float* y, int r0, int r1, const float* post, const float* comb) {
    const int hc = p->g_hc, D = p->dim;
    for (int i = r0; i < r1; i++) {
        float old[8], yi = y[i];
        for (int m = 0; m < hc; m++) old[m] = hs[(size_t)m * D + i];
        for (int n = 0; n < hc; n++) {
            float v = post[n] * yi;
            for (int m = 0; m < hc; m++) v += comb[m * hc + n] * old[m];
            hs[(size_t)n * D + i] = v;
        }
    }
}

// ---------------------------------------------------------------------------- helpers
// rows [r0, r1) of a matrix made of per-head blocks of `rph` rows, head h reading its input at x + h * in_stride
static void glm_matmul_heads(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int rph, int cols,
                             int in_stride, int r0, int r1) {
    while (r0 < r1) {
        int h = r0 / rph, e = (h + 1) * rph < r1 ? (h + 1) * rph : r1;
        QuantizedTensor v = expert_view(w, h, rph, cols);
        matmul_rows(out + (size_t)h * rph, xq + (size_t)h * in_stride, xs + (size_t)h * in_stride / GS, &v, cols, r0 - h * rph, e - h * rph);
        r0 = e;
    }
}
static inline float glm_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* DSA indexer: with at most top-k / kpool complete pools every visible token is kept (nsel = -1),
 * otherwise the top-scoring pools of kpool tokens plus the incomplete tail pool. */
static int glm_cmp_desc(const void* a, const void* b) {
    float x = *(const float*)a, y = *(const float*)b;
    return x < y ? 1 : x > y ? -1 : 0;
}
// index scores of pools [j0, j1) of attention layer ai for the indexer queries qidx and head weights hw
static void glm_pool_scores(const Config* p, const GlmState* g, int ai, int j0, int j1, const float* qidx, const float* hw, float* score) {
    const int kp = p->g_kpool, ih = p->g_ih, ihh = p->g_ihh;
    const float* pk = g->ipool + (size_t)ai * (p->seq_len / kp + 1) * ih;
    const float sc = 1.0f / sqrtf((float)ih);
    const float hs = 1.0f / sqrtf((float)ihh); // head weights are weights_proj(x) / sqrt(heads)
    for (int j = j0; j < j1; j++) {
        const float* k = pk + (size_t)j * ih;
        float s = 0.0f;
        for (int h = 0; h < ihh; h++) {
            const float* q = qidx + (size_t)h * ih;
            float d = 0.0f;
            for (int i = 0; i < ih; i++) d += q[i] * k[i];
            d *= sc;
            if (d > 0.0f) s += hw[h] * hs * d;
        }
        score[j] = s;
    }
}
/* the tokens of the top-scoring maxpool pools plus the tail after the last complete pool, ascending;
 * tmp: npool floats of scratch. Ties at the threshold go to the lower pool index. Returns their number. */
static int glm_pick(const Config* p, const float* score, int pos, int* out, float* tmp) {
    const int kp = p->g_kpool, T = pos + 1, npool = T / kp, maxpool = p->g_itopk / kp;
    memcpy(tmp, score, npool * sizeof(float));
    qsort(tmp, npool, sizeof(float), glm_cmp_desc);
    const float thr = tmp[maxpool - 1];
    int above = 0, n = 0;
    for (int j = 0; j < npool; j++) above += score[j] > thr;
    int eq_left = maxpool - above;
    for (int j = 0; j < npool; j++)
        if (score[j] > thr || (score[j] == thr && eq_left-- > 0))
            for (int t = j * kp; t < j * kp + kp; t++) out[n++] = t;
    for (int t = npool * kp; t < T; t++) out[n++] = t;
    return n;
}
// does the indexer of a token at pos have to choose (more complete pools than top-k / kpool)?
static inline int glm_selects(const Config* p, int pos) { return (pos + 1) / p->g_kpool > p->g_itopk / p->g_kpool; }
// pooled indexer key of pool j of layer ai (once its kp tokens are cached): softmax over the pool's
// tokens (per channel) of gate score + position embedding, weighted sum of the keys
static void glm_pool_key(const Config* p, const GlmWeights* w, GlmState* g, int ai, int j) {
    const int kp = p->g_kpool, ih = p->g_ih;
    const float* ape = w->ape + (size_t)ai * kp * ih;
    const float* kk = g->ikey + ((size_t)ai * p->seq_len + (size_t)j * kp) * ih;
    const float* gg = g->igate + ((size_t)ai * p->seq_len + (size_t)j * kp) * ih;
    float* out = g->ipool + ((size_t)ai * (p->seq_len / kp + 1) + j) * ih;
    for (int c = 0; c < ih; c++) {
        float mx = -FLT_MAX, lg[16], z = 0.0f, v = 0.0f;
        for (int t = 0; t < kp; t++) { lg[t] = gg[(size_t)t * ih + c] + ape[(size_t)t * ih + c]; mx = lg[t] > mx ? lg[t] : mx; }
        for (int t = 0; t < kp; t++) { lg[t] = expf(lg[t] - mx); z += lg[t]; }
        for (int t = 0; t < kp; t++) v += lg[t] / z * kk[(size_t)t * ih + c];
        out[c] = v;
    }
}

// ---------------------------------------------------------------------------- one decode step
static void forward_glm_thread(int tid) {
    Transformer* tr = P.t;
    Config* p = &tr->config;
    GlmWeights* w = tr->weights.glm;
    RunState* s = &tr->state;
    GlmState* g = s->glm;
    ThreadBuf* tb = &P.tb[tid];
    const int pos = P.pos, D = p->dim, hc = p->g_hc, nth = P.nth;
    const int H = p->g_H, dh = p->g_dh, Kc = p->g_K, HD = H * dh, nh = p->n_heads;
    const int ql = p->g_ql, kvl = p->g_kvl, qk = p->g_qk, vh = p->g_vh, ih = p->g_ih, ihh = p->g_ihh, kp = p->g_kpool;
    const int E = p->n_experts, A = p->n_active, mh = p->moe_hidden, F = p->hidden_dim;
    const int L = GLM_LAYERS > 0 && GLM_LAYERS < p->n_layers ? GLM_LAYERS : p->n_layers;
    const int nitems = H * ((dh + HYB_ROWS - 1) / HYB_ROWS), ipr = (dh + HYB_ROWS - 1) / HYB_ROWS;
    const int8_t* nq = NB[tb->node].q;
    const float* ns = NB[tb->node].s;
    float pre[8], post[8], comb[64];
    int r0, r1;

    if (tb->ltid == 0) {
        struct WorkCounter* wc = P.work[tb->node][(P.seq + 1) & 1];
        for (int k = 0; k < GLM_PH * p->n_layers + 1; k++) atomic_store_explicit(&wc[k].v, 0, memory_order_relaxed);
    }
    if (pos == 0) { // new sequence: zero the recurrent and conv states (each thread its own items)
        const int nlin = p->n_layers - p->n_attn;
        for (int li = 0; li < nlin; li++)
            for (int it = tid; it < nitems; it += nth) {
                int h = it / ipr, c0 = (it % ipr) * HYB_ROWS, rows = c0 + HYB_ROWS < dh ? HYB_ROWS : dh - c0;
                memset(hyb_state(p, s, li, h, c0), 0, (size_t)rows * dh * sizeof(float));
            }
        size_t cn = (size_t)nlin * p->conv_dim * (Kc - 1);
        memset(s->conv_state + cn * tid / nth, 0, (cn * (tid + 1) / nth - cn * tid / nth) * sizeof(float));
    }
    { // streams <- embedding (every stream the same)
        const float* emb = tr->weights.token_embedding_table + (size_t)P.token * D;
        int a = (int)((long)D * tid / nth), b = (int)((long)D * (tid + 1) / nth);
        for (int n = 0; n < hc; n++) memcpy(g->hs + (size_t)n * D + a, emb + a, (b - a) * sizeof(float));
    }
    barrier(tb);

    int li = 0, ai = 0;
    for (int l = 0; l < L; l++) {
        const int attn = glm_is_attn(p, l), base = GLM_PH * l;
        for (int part = 0; part < 2; part++) {
            // ---- mHC: collapse the streams into the block input
            const size_t hi = (size_t)2 * l + part;
            glm_hc_partial(p, w->hc_fn + hi * 24 * hc * D, g->hs, g->hcp + (size_t)tid * 32, tid, nth);
            barrier(tb);
            glm_hc_weights(p, g->hcp, nth, w->hc_base + hi * 24, w->hc_scale + hi * 4, pre, post, comb);
            {
                int a = (int)((long)D * tid / nth), b = (int)((long)D * (tid + 1) / nth);
                for (int i = a; i < b; i++) {
                    float v = 0.0f;
                    for (int n = 0; n < hc; n++) v += pre[n] * g->hs[(size_t)n * D + i];
                    g->xc[i] = v;
                }
            }
            barrier(tb);
            node_norm_quant(tb, g->xc, (part == 0 ? w->in_norm : w->post_norm) + (size_t)l * D, D);

            if (part == 0 && !attn) { // ---------------------------------------------- KDA
                const QuantizedTensor* win = w->lin_in + li;
                const int rows = 3 * HD + 2 * dh + H;
                while (grab_rows(base + 0, rows, tb, &r0, &r1)) {
                    matmul_rows(g->lin, nq, ns, win, D, r0, r1);
                    for (int r = r0; r < r1; r++) {
                        if (r < 3 * HD) { // causal depthwise conv over the last K inputs, then SiLU
                            float* cs = s->conv_state + ((size_t)li * p->conv_dim + r) * (Kc - 1);
                            const float* cw = w->conv + ((size_t)li * p->conv_dim + r) * Kc;
                            float x = g->lin[r], yv = cw[Kc - 1] * x;
                            for (int j = 0; j < Kc - 1; j++) yv += cw[j] * cs[j];
                            for (int j = 0; j < Kc - 2; j++) cs[j] = cs[j + 1];
                            cs[Kc - 2] = x;
                            g->lin[r] = silu_f(yv);
                        } else if (r >= 3 * HD + 2 * dh) {
                            g->lin[r] = glm_sigmoid(g->lin[r]); // beta
                        }
                    }
                }
                barrier(tb);
                { // forget gate f_b(f_a x) -> decay, output gate g_b(g_a x): low-rank, inputs quantized privately
                    quantize_vec(tb->q, tb->s, g->lin + 3 * HD, dh);
                    quantize_vec(tb->q + dh, tb->s + dh / GS, g->lin + 3 * HD + dh, dh);
                    const float* dtb = w->dt_bias + (size_t)li * HD;
                    const float* Av = w->A + (size_t)li * H;
                    while (grab_rows(base + 1, HD, tb, &r0, &r1)) {
                        matmul_rows(g->dec, tb->q, tb->s, w->f_b + li, dh, r0, r1);
                        for (int r = r0; r < r1; r++)
                            g->dec[r] = expf(p->g_lower * glm_sigmoid(Av[r / dh] * (g->dec[r] + dtb[r])));
                    }
                    while (grab_rows(base + 2, HD, tb, &r0, &r1)) matmul_rows(g->gate, tb->q + dh, tb->s + dh / GS, w->g_b + li, dh, r0, r1);
                }
                barrier(tb);
                for (int it = tid; it < nitems; it += nth) { // gated delta rule, per (head, block of value rows)
                    const int h = it / ipr, c0 = (it % ipr) * HYB_ROWS;
                    float* S = hyb_state(p, s, li, h, c0);
                    const float* q = g->lin + (size_t)h * dh;
                    const float* k = g->lin + HD + (size_t)h * dh;
                    const float* v = g->lin + 2 * HD + (size_t)h * dh;
                    const float* dc = g->dec + (size_t)h * dh;
                    const float beta = g->lin[3 * HD + 2 * dh + h];
                    float qq[256], kk[256], sq = 0.0f, sk = 0.0f;
                    for (int i = 0; i < dh; i++) { sq += q[i] * q[i]; sk += k[i] * k[i]; }
                    const float qn = 1.0f / sqrtf(sq + 1e-6f) / sqrtf((float)dh), kn = 1.0f / sqrtf(sk + 1e-6f);
                    for (int i = 0; i < dh; i++) { qq[i] = q[i] * qn; kk[i] = k[i] * kn; }
                    for (int c = c0; c < c0 + HYB_ROWS && c < dh; c++) {
                        float* row = S + (size_t)(c - c0) * dh;
                        float kvm = 0.0f;
                        for (int i = 0; i < dh; i++) { row[i] *= dc[i]; kvm += row[i] * kk[i]; }
                        const float d = (v[c] - kvm) * beta;
                        float o = 0.0f;
                        for (int i = 0; i < dh; i++) { row[i] += kk[i] * d; o += row[i] * qq[i]; }
                        g->lo[(size_t)h * dh + c] = o;
                    }
                }
                barrier(tb);
                // gated RMSNorm per head (each node all heads), quantized into the node buffer
                for (int h = tb->ltid; h < H; h += tb->nthn) {
                    const float* o = g->lo + (size_t)h * dh;
                    const float* gt = g->gate + (size_t)h * dh;
                    const float* nw = w->o_norm + (size_t)li * dh;
                    float ss = 0.0f, yv[256];
                    for (int c = 0; c < dh; c++) ss += o[c] * o[c];
                    const float inv = 1.0f / sqrtf(ss / dh + NORM_EPS);
                    for (int c = 0; c < dh; c++) yv[c] = nw[c] * (o[c] * inv) * glm_sigmoid(gt[c]);
                    quantize_vec(NB[tb->node].q + (size_t)h * dh, NB[tb->node].s + (size_t)h * dh / GS, yv, dh);
                }
                node_barrier(tb);
                while (grab_rows(base + 3, D, tb, &r0, &r1)) {
                    matmul_rows(g->y, nq, ns, w->lin_out + li, HD, r0, r1);
                    glm_hc_update(p, g->hs, g->y, r0, r1, post, comb);
                }
                barrier(tb);
                li++;
            } else if (part == 0) { // --------------------------------------------------- MLA + DSA
                const int rows = ql + kvl + 2 * ih + ihh;
                while (grab_rows(base + 0, rows, tb, &r0, &r1)) matmul_rows(g->mla, nq, ns, w->mla_in + ai, D, r0, r1);
                barrier(tb);
                float* lat = s->key_cache + ((size_t)ai * p->seq_len + pos) * kvl;
                if (tid == nth - 1) { // latent kv: RMSNorm(kv_a) into the cache
                    const float* kv = g->mla + ql;
                    const float* nw = w->kva_norm + (size_t)ai * kvl;
                    float ss = 0.0f;
                    for (int i = 0; i < kvl; i++) ss += kv[i] * kv[i];
                    const float inv = 1.0f / sqrtf(ss / kvl + NORM_EPS);
                    for (int i = 0; i < kvl; i++) lat[i] = nw[i] * (kv[i] * inv);
                }
                if (tid == (nth >= 2 ? nth - 2 : 0)) { // indexer key (LayerNorm) and k-pool gate scores into the cache
                    const float* kr = g->mla + ql + kvl;
                    float* ik = g->ikey + ((size_t)ai * p->seq_len + pos) * ih;
                    float mu = 0.0f, var = 0.0f;
                    for (int i = 0; i < ih; i++) mu += kr[i];
                    mu /= ih;
                    for (int i = 0; i < ih; i++) var += (kr[i] - mu) * (kr[i] - mu);
                    var /= ih;
                    const float inv = 1.0f / sqrtf(var + 1e-6f);
                    for (int i = 0; i < ih; i++) ik[i] = (kr[i] - mu) * inv * w->ik_w[(size_t)ai * ih + i] + w->ik_b[(size_t)ai * ih + i];
                    memcpy(g->igate + ((size_t)ai * p->seq_len + pos) * ih, g->mla + ql + kvl + ih + ihh, ih * sizeof(float));
                    if ((pos + 1) % kp == 0) glm_pool_key(p, w, g, ai, pos / kp);
                }
                // q_resid = RMSNorm(q_a) -> node buffer, then q heads | indexer q heads
                node_norm_quant(tb, g->mla, w->qa_norm + (size_t)ai * ql, ql);
                while (grab_rows(base + 1, nh * qk + ihh * ih, tb, &r0, &r1)) matmul_rows(g->qb, nq, ns, w->q_b + ai, ql, r0, r1);
                barrier(tb);
                const int sel = glm_selects(p, pos);
                if (sel) { // indexer: score the complete pools (all threads)
                    const int np = (pos + 1) / kp, j0 = (int)((long)np * tid / nth), j1 = (int)((long)np * (tid + 1) / nth);
                    glm_pool_scores(p, g, ai, j0, j1, g->qb + (size_t)nh * qk, g->mla + ql + kvl + ih, g->iscore);
                }
                node_quant(tb, g->qb, nh * qk);
                // absorbed queries: q'_h = W_k,h^T q_h (kvl values per head)
                while (grab_rows(base + 2, nh * kvl, tb, &r0, &r1)) glm_matmul_heads(g->qabs, nq, ns, w->kv_bk + ai, kvl, qk, qk, r0, r1);
                barrier(tb);
                if (sel) { // tid 0 picks the tokens; without selection every token up to pos is attended
                    if (tid == 0) {
                        g->nsel = glm_pick(p, g->iscore, pos, g->isel, g->iscore + (pos + 1) / kp);
                        if (GLM_DEBUG_SEL && ai == 0) {
                            fprintf(stderr, "score %d:", pos);
                            for (int j = 0; j < (pos + 1) / kp; j++) fprintf(stderr, " %.5f", g->iscore[j]);
                            fprintf(stderr, "\n");
                        }
                    }
                    barrier(tb);
                }
                if (GLM_DEBUG_SEL && ai == 0 && tid == 0) {
                    fprintf(stderr, "sel %d:", pos);
                    for (int j = 0; j < (!sel ? pos + 1 : g->nsel); j++) fprintf(stderr, " %d", !sel ? j : g->isel[j]);
                    fprintf(stderr, "\n");
                }
                // attention over the selected tokens: items (head, chunk of the selection)
                const int all = !sel, T = all ? pos + 1 : g->nsel;
                const int C = T / 64 < 1 ? 1 : (T / 64 < P.C ? T / 64 : P.C);
                const float scale = 1.0f / sqrtf((float)qk);
                const float* lc = s->key_cache + (size_t)ai * p->seq_len * kvl;
                for (int item = tid; item < nh * C; item += nth) {
                    const int h = item / C, c = item % C;
                    const int t0 = (int)((long)T * c / C), t1 = (int)((long)T * (c + 1) / C);
                    const float* q = g->qabs + (size_t)h * kvl;
                    float* att = s->att + (size_t)h * p->seq_len;
                    float m = -FLT_MAX;
                    for (int j = t0; j < t1; j++) {
                        const float* kv = lc + (size_t)(all ? j : g->isel[j]) * kvl;
                        float sc = 0.0f;
                        for (int i = 0; i < kvl; i++) sc += q[i] * kv[i];
                        sc *= scale;
                        att[j] = sc;
                        m = sc > m ? sc : m;
                    }
                    float* pr = P.part + ((size_t)h * C + c) * (kvl + 2);
                    float* acc = pr + 2, sm = 0.0f;
                    memset(acc, 0, kvl * sizeof(float));
                    for (int j = t0; j < t1; j++) {
                        const float e = expf(att[j] - m);
                        sm += e;
                        const float* kv = lc + (size_t)(all ? j : g->isel[j]) * kvl;
                        for (int i = 0; i < kvl; i++) acc[i] += e * kv[i];
                    }
                    pr[0] = m;
                    pr[1] = sm;
                }
                barrier(tb);
                // merge (each node all heads) -> latent context per head, quantized into the node buffer
                for (int h = tb->ltid; h < nh; h += tb->nthn) {
                    const float* ph = P.part + (size_t)h * C * (kvl + 2);
                    float M = -FLT_MAX, den = 0.0f, ctx[1024];
                    for (int c = 0; c < C; c++) if (ph[(size_t)c * (kvl + 2) + 1] > 0.0f && ph[(size_t)c * (kvl + 2)] > M) M = ph[(size_t)c * (kvl + 2)];
                    memset(ctx, 0, kvl * sizeof(float));
                    for (int c = 0; c < C; c++) {
                        const float* pr = ph + (size_t)c * (kvl + 2);
                        if (pr[1] <= 0.0f) continue;
                        const float f = expf(pr[0] - M);
                        den += f * pr[1];
                        for (int i = 0; i < kvl; i++) ctx[i] += f * pr[2 + i];
                    }
                    for (int i = 0; i < kvl; i++) ctx[i] /= den;
                    quantize_vec(NB[tb->node].q + (size_t)h * kvl, NB[tb->node].s + (size_t)h * kvl / GS, ctx, kvl);
                }
                node_barrier(tb);
                // per-head values W_v,h ctx_h
                while (grab_rows(base + 3, nh * vh, tb, &r0, &r1)) glm_matmul_heads(g->ov, nq, ns, w->kv_bv + ai, vh, kvl, kvl, r0, r1);
                barrier(tb);
                node_quant(tb, g->ov, nh * vh);
                while (grab_rows(base + 4, D, tb, &r0, &r1)) {
                    matmul_rows(g->y, nq, ns, w->o_proj + ai, nh * vh, r0, r1);
                    glm_hc_update(p, g->hs, g->y, r0, r1, post, comb);
                }
                barrier(tb);
                ai++;
            } else { // ------------------------------------------------------------------ FFN
                const float lim = p->g_limit;
                int nx; // experts in this step (routed + shared, or the dense FFN as one)
                int sel[MOE_MAX_ACTIVE + 1];
                float wt[MOE_MAX_ACTIVE + 1];
                QuantizedTensor up1[MOE_MAX_ACTIVE + 1], up3[MOE_MAX_ACTIVE + 1], dn[MOE_MAX_ACTIVE + 1];
                int hid;
                if (l < p->g_dense) {
                    nx = 1; hid = F; wt[0] = 1.0f;
                    up1[0] = expert_view(w->w13 + l, 0, F, D); up3[0] = expert_view(w->w13 + l, 1, F, D); dn[0] = w->w2[l];
                } else {
                    const int mi = l - p->g_dense;
                    hid = mh;
                    float* rl = NB[tb->node].router;
                    while (grab_virtual(base + 5, E, tb, &r0, &r1)) matmul_rows(rl, nq, ns, w->router_node[tb->node] + mi, D, r0, r1);
                    node_barrier(tb);
                    // sigmoid scores; selection by score + bias, weights = normalized scores * scale
                    const float* bias = w->rbias + (size_t)mi * E;
                    float best[MOE_MAX_ACTIVE];
                    for (int j = 0; j < A; j++) { sel[j] = -1; best[j] = -FLT_MAX; }
                    for (int e = 0; e < E; e++) {
                        const float ch = glm_sigmoid(rl[e]) + bias[e];
                        int j = A - 1;
                        if (ch <= best[j]) continue;
                        while (j > 0 && best[j - 1] < ch) { best[j] = best[j - 1]; sel[j] = sel[j - 1]; j--; }
                        best[j] = ch;
                        sel[j] = e;
                    }
                    float sum = 0.0f;
                    for (int j = 0; j < A; j++) { wt[j] = glm_sigmoid(rl[sel[j]]); sum += wt[j]; }
                    for (int j = 0; j < A; j++) wt[j] = p->norm_topk ? wt[j] / (sum + 1e-20f) * p->g_rscale : wt[j] * p->g_rscale;
                    for (int j = 0; j < A; j++) {
                        up1[j] = expert_view(w->ex1 + mi, sel[j], mh, D);
                        up3[j] = expert_view(w->ex3 + mi, sel[j], mh, D);
                        dn[j] = expert_view(w->ex2 + mi, sel[j], D, mh);
                    }
                    up1[A] = expert_view(w->sh13 + mi, 0, mh, D); up3[A] = expert_view(w->sh13 + mi, 1, mh, D);
                    dn[A] = w->sh2[mi]; wt[A] = 1.0f;
                    nx = A + 1;
                }
                int lo, hi2;
                node_rows(hid, tb->node, &lo, &hi2);
                const int half = hi2 - lo;
                while (grab_virtual(base + 6, nx * half, tb, &r0, &r1)) {
                    for (int v = r0; v < r1;) {
                        const int j = v / half, vend = (j + 1) * half < r1 ? (j + 1) * half : r1;
                        const int a0 = lo + (v - j * half), a1 = lo + (vend - j * half);
                        float* h1 = g->h1 + (size_t)j * hid;
                        float* h3 = g->h3 + (size_t)j * hid;
                        matmul_rows(h1, nq, ns, &up1[j], D, a0, a1);
                        matmul_rows(h3, nq, ns, &up3[j], D, a0, a1);
                        for (int i = a0; i < a1; i++) {
                            float gt = h1[i] < lim ? h1[i] : lim, u = h3[i] < -lim ? -lim : (h3[i] > lim ? lim : h3[i]);
                            h1[i] = silu_f(gt) * u * wt[j];
                        }
                        v = vend;
                    }
                }
                barrier(tb);
                node_quant(tb, g->h1, nx * hid);
                while (grab_rows(base + 7, D, tb, &r0, &r1)) {
                    float* yv = g->y;
                    for (int i = r0; i < r1; i++) yv[i] = 0.0f;
                    for (int j = 0; j < nx; j++) {
                        matmul_rows(tb->xb, nq + (size_t)j * hid, ns + (size_t)j * hid / GS, &dn[j], hid, r0, r1);
                        for (int i = r0; i < r1; i++) yv[i] += tb->xb[i];
                    }
                    glm_hc_update(p, g->hs, yv, r0, r1, post, comb);
                }
                barrier(tb);
            }
        }
    }
    // final: mean of the streams, RMSNorm, classifier
    {
        int a = (int)((long)D * tid / nth), b = (int)((long)D * (tid + 1) / nth);
        for (int i = a; i < b; i++) {
            float v = 0.0f;
            for (int n = 0; n < hc; n++) v += g->hs[(size_t)n * D + i];
            s->x[i] = v / hc;
        }
    }
    barrier(tb);
    node_norm_quant(tb, s->x, w->final_norm, D);
    while (grab_rows(GLM_PH * p->n_layers, p->vocab_size, tb, &r0, &r1)) matmul_rows(s->logits, nq, ns, w->lm_head, D, r0, r1);
    barrier(tb);
}
