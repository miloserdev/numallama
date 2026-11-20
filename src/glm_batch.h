/* GLM-5.3-Flash batched prefill: a block of B prompt tokens at once (see glm.h for the model).
 * Every matrix is applied to all B tokens (weights read once per block); the mHC projection is an
 * fp32 dot over the 4 streams split by columns among the threads; the KDA conv and recurrence run
 * per channel / state slice sequentially over the tokens; the MLA attention of every (token, head)
 * runs over that token's indexer selection; the MoE groups the tokens by expert (the shared expert is
 * one more expert that every token uses). Included by runq.c after the batch helpers. */

typedef struct {
    float* hs;     // (B, hc, D) residual streams
    float* xc;     // (B, D) normalized block input
    float* hw;     // (B, 32) mHC weights per token: pre[8] post[8] comb[16]
    float* hcp;    // (nth, B, 32) mHC partial dot products
    float* y;      // (B, D) block output
    float* lin;    // (B, lin_in)
    float* dec;    // (B, H dh)
    float* gate;   // (B, H dh)
    float* lo;     // (B, H dh)
    float* mla;    // (B, mla_in)
    float* qb;     // (B, nh qk + ihh ih)
    float* qabs;   // (nh, B, kvl) head-major
    float* ctx;    // (nh, B, kvl) head-major
    float* ov;     // (B, nh vh)
    float *h1, *h3; // (B, max(F, mh))
    float* router; // (B, E)
    int* sel;      // (B, A)
    float* wt;     // (B, A)
    int *cnt, *fill, *stok, *ue, U; // slots of expert e: [cnt[e], cnt[e+1]) (e = E: the shared expert)
    float* sw;     // (slots) routing weight
    struct WorkCounter* ctr; // per node: dynamic chunks of the expert up-projections
    float *gh, *gh3, *gout; // (slots, mh), (slots, mh), (slots, D)
    int* isel;     // (B, itopk + kpool) selected tokens per token
    int* nsel;     // (B) their number (-1: every token up to the position)
    long* ccum;    // (B + 1) cumulative attention cost
    float* att;    // (nth, seq_len) scores
    int W;         // isel row width
} GlmBatch;
static GlmBatch GB;
static struct { int8_t *xq; float *xs; int8_t *gxq; float *gxs; int8_t *gq; float *gqs; } GBN[MAX_NODES];

static void* glm_node_alloc(size_t bytes, int k) {
    size_t b = (bytes + 4095) & ~(size_t)4095;
    void* p = aligned_alloc(4096, b);
#if defined(__linux__)
    if (NN > 1) bind_range(p, b, NODE_ID[k]);
#endif
    memset(p, 0, b);
    return p;
}

static void glm_init_batch(Transformer* t) {
    Config* p = &t->config;
    const size_t B = BATCH_MAX, D = p->dim, hc = p->g_hc, HD = (size_t)p->g_H * p->g_dh, nh = p->n_heads;
    const size_t LIN = 3 * HD + 2 * p->g_dh + p->g_H, MI = p->g_ql + p->g_kvl + 2 * p->g_ih + p->g_ihh;
    const size_t QB = nh * p->g_qk + (size_t)p->g_ihh * p->g_ih, E = p->n_experts, A = p->n_active, mh = p->moe_hidden;
    const size_t S = B * A + B; // routed slots + the shared expert's
    size_t hid = (size_t)p->hidden_dim > mh ? (size_t)p->hidden_dim : mh;
    GB.hs = calloc(B * hc * D, sizeof(float));
    GB.xc = calloc(B * D, sizeof(float));
    GB.hw = calloc(B * 32, sizeof(float));
    GB.hcp = calloc((size_t)P.nth * B * 32, sizeof(float));
    GB.y = calloc(B * D, sizeof(float));
    GB.lin = calloc(B * LIN, sizeof(float));
    GB.dec = calloc(B * HD, sizeof(float));
    GB.gate = calloc(B * HD, sizeof(float));
    GB.lo = calloc(B * HD, sizeof(float));
    GB.mla = calloc(B * MI, sizeof(float));
    GB.qb = calloc(B * QB, sizeof(float));
    GB.qabs = calloc(nh * B * p->g_kvl, sizeof(float));
    GB.ctx = calloc(nh * B * p->g_kvl, sizeof(float));
    GB.ov = calloc(B * nh * p->g_vh, sizeof(float));
    GB.h1 = calloc(B * hid, sizeof(float));
    GB.h3 = calloc(B * hid, sizeof(float));
    GB.router = calloc(B * E, sizeof(float));
    GB.sel = calloc(B * A, sizeof(int));
    GB.wt = calloc(B * A, sizeof(float));
    GB.cnt = calloc(E + 2, sizeof(int));
    GB.fill = calloc(E + 2, sizeof(int));
    GB.stok = calloc(S, sizeof(int));
    GB.ue = calloc(E + 1, sizeof(int));
    GB.sw = calloc(S, sizeof(float));
    GB.ctr = aligned_alloc(64, MAX_NODES * sizeof(struct WorkCounter));
    memset(GB.ctr, 0, MAX_NODES * sizeof(struct WorkCounter));
    GB.gh = calloc(S * mh, sizeof(float));
    GB.gh3 = calloc(S * mh, sizeof(float));
    GB.gout = calloc(S * D, sizeof(float));
    GB.W = p->g_itopk + p->g_kpool;
    GB.isel = calloc(B * GB.W, sizeof(int));
    GB.nsel = calloc(B, sizeof(int));
    GB.ccum = calloc(B + 1, sizeof(long));
    GB.att = calloc((size_t)P.nth * p->seq_len, sizeof(float));
    size_t xin = nh * p->g_kvl; // widest block input: head-major latent contexts
    if (nh * p->g_vh > xin) xin = nh * p->g_vh;
    if ((size_t)p->hidden_dim > xin) xin = p->hidden_dim;
    if (HD > xin) xin = HD;
    for (int k = 0; k < NN; k++) {
        GBN[k].xq = glm_node_alloc(B * xin, k);
        GBN[k].xs = glm_node_alloc(B * (xin / GS + 1) * sizeof(float), k);
        GBN[k].gxq = glm_node_alloc(S * D, k);
        GBN[k].gxs = glm_node_alloc(S * (D / GS) * sizeof(float), k);
        GBN[k].gq = glm_node_alloc(S * mh, k);
        GBN[k].gqs = glm_node_alloc(S * (mh / GS) * sizeof(float), k);
    }
    if (!GB.hs || !GB.lin || !GB.qabs || !GB.ctx || !GB.gout || !GB.att) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    HYB_BATCH = 1; // forward_batch: blocks through forward_glm_batch_thread
}

// post-callbacks of the block matmuls
typedef struct { const Config* p; float* hs; const float* y; const float* hw; } GlmHcCtx;
static void glm_post_hc(void* c, int b0, int b1, int r0, int r1) { // streams += block output (mHC), rows [r0, r1)
    const GlmHcCtx* x = c;
    const int hc = x->p->g_hc, D = x->p->dim;
    for (int b = b0; b < b1; b++) {
        const float* w = x->hw + (size_t)b * 32;
        glm_hc_update(x->p, x->hs + (size_t)b * hc * D, x->y + (size_t)b * D, r0, r1, w + 8, w + 16);
    }
}
typedef struct { float* dec; const float* dt; const float* A; float lower; int dh, ld; } GlmDecCtx;
static void glm_post_decay(void* c, int b0, int b1, int r0, int r1) {
    const GlmDecCtx* x = c;
    for (int b = b0; b < b1; b++) {
        float* d = x->dec + (size_t)b * x->ld;
        for (int r = r0; r < r1; r++) d[r] = expf(x->lower * glm_sigmoid(x->A[r / x->dh] * (d[r] + x->dt[r])));
    }
}
typedef struct { float* h1; const float* h3; int ld; float lim; } GlmSwiCtx;
static void glm_post_swiglu(void* c, int b0, int b1, int r0, int r1) {
    const GlmSwiCtx* x = c;
    for (int b = b0; b < b1; b++) {
        float* h1 = x->h1 + (size_t)b * x->ld;
        const float* h3 = x->h3 + (size_t)b * x->ld;
        for (int i = r0; i < r1; i++) {
            float g = h1[i] < x->lim ? h1[i] : x->lim, u = h3[i] < -x->lim ? -x->lim : (h3[i] > x->lim ? x->lim : h3[i]);
            h1[i] = silu_f(g) * u;
        }
    }
}

/* selected tokens of block token at pos (into out, count into *nout; -1: every token up to pos while there
 * are at most top-k / kpool complete pools); score: scratch of 2 * pools floats */
static void glm_select_into(const Config* p, const GlmState* g, int ai, int pos, const float* qidx, const float* hw,
                            float* score, int* out, int* nout) {
    const int kp = p->g_kpool, ih = p->g_ih, ihh = p->g_ihh, T = pos + 1, npool = T / kp, maxpool = p->g_itopk / kp;
    if (npool <= maxpool) { *nout = -1; return; }
    const float* pk = g->ipool + (size_t)ai * (p->seq_len / kp + 1) * ih;
    const float sc = 1.0f / sqrtf((float)ih), hs = 1.0f / sqrtf((float)ihh);
    for (int j = 0; j < npool; j++) {
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
    float* tmp = score + npool; // the maxpool-th largest score
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
    *nout = n;
}

static void forward_glm_batch_thread(int tid) {
    Transformer* tr = P.t;
    Config* p = &tr->config;
    GlmWeights* w = tr->weights.glm;
    RunState* s = &tr->state;
    GlmState* g = s->glm;
    ThreadBuf* tb = &P.tb[tid];
    const int nth = P.nth, B = BT.B, pos0 = BT.pos0;
    const int D = p->dim, hc = p->g_hc, N = hc * D, H = p->g_H, dh = p->g_dh, Kc = p->g_K, HD = H * dh, nh = p->n_heads;
    const int ql = p->g_ql, kvl = p->g_kvl, qk = p->g_qk, vh = p->g_vh, ih = p->g_ih, ihh = p->g_ihh, kp = p->g_kpool;
    const int E = p->n_experts, A = p->n_active, mh = p->moe_hidden, F = p->hidden_dim;
    const int LIN = 3 * HD + 2 * dh + H, MI = ql + kvl + 2 * ih + ihh, QB = nh * qk + ihh * ih;
    const int L = GLM_LAYERS > 0 && GLM_LAYERS < p->n_layers ? GLM_LAYERS : p->n_layers;
    const int ipr = (dh + HYB_ROWS - 1) / HYB_ROWS, nitems = H * ipr;
    int8_t* XQ = GBN[tb->node].xq;
    float* XS = GBN[tb->node].xs;
    int b0, b1, n0, n1, r0, r1;
    tok_range(B, tid, nth, &b0, &b1);          // tokens of this thread (global split)
    tok_range(B, tb->ltid, tb->nthn, &n0, &n1); // tokens of this thread within its node (node buffers)

    if (pos0 == 0) { // new sequence: zero the states (same item partition as the decode step)
        const int nlin = p->n_layers - p->n_attn;
        for (int li = 0; li < nlin; li++)
            for (int it = tid; it < nitems; it += nth) {
                int h = it / ipr, c0 = (it % ipr) * HYB_ROWS, rows = c0 + HYB_ROWS < dh ? HYB_ROWS : dh - c0;
                memset(hyb_state(p, s, li, h, c0), 0, (size_t)rows * dh * sizeof(float));
            }
        size_t cn = (size_t)nlin * p->conv_dim * (Kc - 1);
        memset(s->conv_state + cn * tid / nth, 0, (cn * (tid + 1) / nth - cn * tid / nth) * sizeof(float));
    }
    for (int b = b0; b < b1; b++) {
        const float* emb = tr->weights.token_embedding_table + (size_t)BT.tok[b] * D;
        for (int n = 0; n < hc; n++) memcpy(GB.hs + ((size_t)b * hc + n) * D, emb, D * sizeof(float));
    }
    barrier(tb);

    int li = 0, ai = 0;
    for (int l = 0; l < L; l++) {
        const int attn = glm_is_attn(p, l);
        for (int part = 0; part < 2; part++) {
            const size_t hi = (size_t)2 * l + part;
            { // mHC partial products: this thread's column slice of the streams, all tokens
                const float* fn = w->hc_fn + hi * 24 * N;
                const int c0 = (int)((long)N * tid / nth), c1 = (int)((long)N * (tid + 1) / nth);
                for (int b = 0; b < B; b++) {
                    const float* x = GB.hs + (size_t)b * N;
                    float* o = GB.hcp + ((size_t)tid * B + b) * 32;
                    float ss = 0.0f;
                    for (int j = c0; j < c1; j++) ss += x[j] * x[j];
                    for (int r = 0; r < 24; r++) {
                        const float* row = fn + (size_t)r * N;
                        float a = 0.0f;
                        for (int j = c0; j < c1; j++) a += row[j] * x[j];
                        o[r] = a;
                    }
                    o[24] = ss;
                }
            }
            barrier(tb);
            { // per token: weights, collapse, RMSNorm
                const float* base = w->hc_base + hi * 24;
                const float* sc = w->hc_scale + hi * 4;
                const float* nw = (part == 0 ? w->in_norm : w->post_norm) + (size_t)l * D;
                float* red = tb->xb; // 32 floats per thread, reduced over threads
                for (int b = b0; b < b1; b++) {
                    for (int r = 0; r < 25; r++) red[r] = 0.0f;
                    for (int t = 0; t < nth; t++) {
                        const float* o = GB.hcp + ((size_t)t * B + b) * 32;
                        for (int r = 0; r < 25; r++) red[r] += o[r];
                    }
                    // glm_hc_weights reduces nth partials of stride 32: hand it the single reduced row
                    float* hw = GB.hw + (size_t)b * 32;
                    glm_hc_weights(p, red, 1, base, sc, hw, hw + 8, hw + 16);
                    const float* x = GB.hs + (size_t)b * N;
                    float* xc = GB.xc + (size_t)b * D;
                    float ss = 0.0f;
                    for (int i = 0; i < D; i++) {
                        float v = 0.0f;
                        for (int n = 0; n < hc; n++) v += hw[n] * x[(size_t)n * D + i];
                        xc[i] = v;
                        ss += v * v;
                    }
                    const float inv = 1.0f / sqrtf(ss / D + NORM_EPS);
                    for (int i = 0; i < D; i++) xc[i] = nw[i] * (xc[i] * inv);
                }
            }
            barrier(tb);
            for (int b = n0; b < n1; b++) quantize_vec(XQ + (size_t)b * D, XS + (size_t)b * (D / GS), GB.xc + (size_t)b * D, D);
            node_barrier(tb);
            GlmHcCtx hcx = {p, GB.hs, GB.y, GB.hw};

            if (part == 0 && !attn) { // ---------------------------------------------- KDA
                matmul_batch_node(GB.lin, LIN, XQ, XS, B, w->lin_in + li, D, LIN, tb, NULL, NULL, NULL, NULL);
                barrier(tb);
                { // conv + SiLU per channel over the tokens; beta
                    int c0, c1;
                    tok_range(LIN, tid, nth, &c0, &c1);
                    for (int r = c0; r < c1; r++) {
                        if (r < 3 * HD) {
                            float* cs = s->conv_state + ((size_t)li * p->conv_dim + r) * (Kc - 1);
                            const float* cw = w->conv + ((size_t)li * p->conv_dim + r) * Kc;
                            for (int b = 0; b < B; b++) {
                                float* v = GB.lin + (size_t)b * LIN + r;
                                float x = *v, yv = cw[Kc - 1] * x;
                                for (int j = 0; j < Kc - 1; j++) yv += cw[j] * cs[j];
                                for (int j = 0; j < Kc - 2; j++) cs[j] = cs[j + 1];
                                cs[Kc - 2] = x;
                                *v = silu_f(yv);
                            }
                        } else if (r >= 3 * HD + 2 * dh) {
                            for (int b = 0; b < B; b++) GB.lin[(size_t)b * LIN + r] = glm_sigmoid(GB.lin[(size_t)b * LIN + r]);
                        }
                    }
                }
                barrier(tb);
                for (int b = n0; b < n1; b++) { // f_a | g_a (two input vectors per token)
                    quantize_vec(XQ + (size_t)b * dh, XS + (size_t)b * (dh / GS), GB.lin + (size_t)b * LIN + 3 * HD, dh);
                    quantize_vec(XQ + (size_t)(B + b) * dh, XS + (size_t)(B + b) * (dh / GS), GB.lin + (size_t)b * LIN + 3 * HD + dh, dh);
                }
                node_barrier(tb);
                {
                    GlmDecCtx dc = {GB.dec, w->dt_bias + (size_t)li * HD, w->A + (size_t)li * H, p->g_lower, dh, HD};
                    matmul_batch_node(GB.dec, HD, XQ, XS, B, w->f_b + li, dh, HD, tb, NULL, NULL, glm_post_decay, &dc);
                    matmul_batch_node(GB.gate, HD, XQ + (size_t)B * dh, XS + (size_t)B * (dh / GS), B, w->g_b + li, dh, HD, tb, NULL, NULL, NULL, NULL);
                }
                barrier(tb);
                for (int it = tid; it < nitems; it += nth) { // gated delta rule over the tokens
                    const int h = it / ipr, c0 = (it % ipr) * HYB_ROWS;
                    float* S = hyb_state(p, s, li, h, c0);
                    for (int b = 0; b < B; b++) {
                        const float* lin = GB.lin + (size_t)b * LIN;
                        const float* q = lin + (size_t)h * dh;
                        const float* k = lin + HD + (size_t)h * dh;
                        const float* v = lin + 2 * HD + (size_t)h * dh;
                        const float* dc = GB.dec + (size_t)b * HD + (size_t)h * dh;
                        const float beta = lin[3 * HD + 2 * dh + h];
                        float qq[256], kk[256], sq = 0.0f, sk = 0.0f;
                        for (int i = 0; i < dh; i++) { sq += q[i] * q[i]; sk += k[i] * k[i]; }
                        const float qn = 1.0f / sqrtf(sq + 1e-6f) / sqrtf((float)dh), kn = 1.0f / sqrtf(sk + 1e-6f);
                        for (int i = 0; i < dh; i++) { qq[i] = q[i] * qn; kk[i] = k[i] * kn; }
                        float* out = GB.lo + (size_t)b * HD + (size_t)h * dh;
                        for (int c = c0; c < c0 + HYB_ROWS && c < dh; c++) {
                            float* row = S + (size_t)(c - c0) * dh;
                            float kvm = 0.0f;
                            for (int i = 0; i < dh; i++) { row[i] *= dc[i]; kvm += row[i] * kk[i]; }
                            const float d = (v[c] - kvm) * beta;
                            float o = 0.0f;
                            for (int i = 0; i < dh; i++) { row[i] += kk[i] * d; o += row[i] * qq[i]; }
                            out[c] = o;
                        }
                    }
                }
                barrier(tb);
                for (int b = n0; b < n1; b++) // gated RMSNorm per head, quantized (each node all heads of its tokens)
                    for (int h = 0; h < H; h++) {
                        const float* o = GB.lo + (size_t)b * HD + (size_t)h * dh;
                        const float* gt = GB.gate + (size_t)b * HD + (size_t)h * dh;
                        const float* nw = w->o_norm + (size_t)li * dh;
                        float ss = 0.0f, yv[256];
                        for (int c = 0; c < dh; c++) ss += o[c] * o[c];
                        const float inv = 1.0f / sqrtf(ss / dh + NORM_EPS);
                        for (int c = 0; c < dh; c++) yv[c] = nw[c] * (o[c] * inv) * glm_sigmoid(gt[c]);
                        quantize_vec(XQ + (size_t)b * HD + (size_t)h * dh, XS + ((size_t)b * HD + (size_t)h * dh) / GS, yv, dh);
                    }
                node_barrier(tb);
                matmul_batch_node(GB.y, D, XQ, XS, B, w->lin_out + li, HD, D, tb, NULL, NULL, glm_post_hc, &hcx);
                barrier(tb);
                li++;
            } else if (part == 0) { // --------------------------------------------------- MLA + DSA
                matmul_batch_node(GB.mla, MI, XQ, XS, B, w->mla_in + ai, D, MI, tb, NULL, NULL, NULL, NULL);
                barrier(tb);
                for (int b = b0; b < b1; b++) { // caches (latent, indexer key / gate), q_a RMSNorm in place
                    float* m = GB.mla + (size_t)b * MI;
                    const int pos = pos0 + b;
                    float ss = 0.0f;
                    for (int i = 0; i < kvl; i++) ss += m[ql + i] * m[ql + i];
                    float inv = 1.0f / sqrtf(ss / kvl + NORM_EPS);
                    float* lat = s->key_cache + ((size_t)ai * p->seq_len + pos) * kvl;
                    for (int i = 0; i < kvl; i++) lat[i] = w->kva_norm[(size_t)ai * kvl + i] * (m[ql + i] * inv);
                    const float* kr = m + ql + kvl;
                    float* ikk = g->ikey + ((size_t)ai * p->seq_len + pos) * ih;
                    float mu = 0.0f, var = 0.0f;
                    for (int i = 0; i < ih; i++) mu += kr[i];
                    mu /= ih;
                    for (int i = 0; i < ih; i++) var += (kr[i] - mu) * (kr[i] - mu);
                    var /= ih;
                    inv = 1.0f / sqrtf(var + 1e-6f);
                    for (int i = 0; i < ih; i++) ikk[i] = (kr[i] - mu) * inv * w->ik_w[(size_t)ai * ih + i] + w->ik_b[(size_t)ai * ih + i];
                    memcpy(g->igate + ((size_t)ai * p->seq_len + pos) * ih, m + ql + kvl + ih + ihh, ih * sizeof(float));
                    ss = 0.0f;
                    for (int i = 0; i < ql; i++) ss += m[i] * m[i];
                    inv = 1.0f / sqrtf(ss / ql + NORM_EPS);
                    for (int i = 0; i < ql; i++) m[i] = w->qa_norm[(size_t)ai * ql + i] * (m[i] * inv);
                }
                barrier(tb);
                { // pooled indexer keys of the pools completed in this block
                    const int j0 = pos0 / kp, j1 = (pos0 + B) / kp; // pools [j0, j1) end inside [pos0, pos0 + B)
                    for (int j = j0 + tid; j < j1; j += nth)
                        if ((j + 1) * kp - 1 >= pos0) glm_pool_key(p, w, g, ai, j);
                }
                for (int b = n0; b < n1; b++) quantize_vec(XQ + (size_t)b * ql, XS + (size_t)b * (ql / GS), GB.mla + (size_t)b * MI, ql);
                node_barrier(tb);
                matmul_batch_node(GB.qb, QB, XQ, XS, B, w->q_b + ai, ql, QB, tb, NULL, NULL, NULL, NULL);
                barrier(tb);
                for (int b = n0; b < n1; b++) // q heads, head-major
                    for (int h = 0; h < nh; h++)
                        quantize_vec(XQ + ((size_t)h * B + b) * qk, XS + ((size_t)h * B + b) * (qk / GS), GB.qb + (size_t)b * QB + (size_t)h * qk, qk);
                for (int b = b0; b < b1; b++) // indexer selection per token
                    glm_select_into(p, g, ai, pos0 + b, GB.qb + (size_t)b * QB + (size_t)nh * qk, GB.mla + (size_t)b * MI + ql + kvl + ih,
                                    GB.att + (size_t)tid * p->seq_len, GB.isel + (size_t)b * GB.W, GB.nsel + b);
                node_barrier(tb);
                for (int h = 0; h < nh; h++) { // absorbed queries q'_h = W_k,h^T q_h
                    QuantizedTensor v = expert_view(w->kv_bk + ai, h, kvl, qk);
                    matmul_batch_node(GB.qabs + (size_t)h * B * kvl, kvl, XQ + (size_t)h * B * qk, XS + (size_t)h * B * (qk / GS), B, &v, qk, kvl, tb,
                                      NULL, NULL, NULL, NULL);
                }
                barrier(tb);
                { // attention, items (token, head) balanced by their cost (number of keys)
                    long tot = 0;
                    for (int b = 0; b < B; b++) {
                        GB.ccum[b] = tot;
                        tot += (long)nh * (GB.nsel[b] >= 0 ? GB.nsel[b] : pos0 + b + 1);
                    }
                    GB.ccum[B] = tot;
                    const long c0 = tot * tid / nth, c1 = tot * (tid + 1) / nth;
                    // first item whose cost range starts at or after c: token by binary search, then head
                    long it0, it1;
                    for (int side = 0; side < 2; side++) {
                        long c = side ? c1 : c0, item;
                        int lo = 0, hi2 = B;
                        while (lo < hi2) { int mid = (lo + hi2) / 2; if (GB.ccum[mid + 1] <= c) lo = mid + 1; else hi2 = mid; }
                        if (lo >= B) item = (long)B * nh;
                        else {
                            long T = GB.nsel[lo] >= 0 ? GB.nsel[lo] : pos0 + lo + 1;
                            long hh = (c - GB.ccum[lo] + T - 1) / T;
                            item = (long)lo * nh + (hh > nh ? nh : hh);
                        }
                        if (side) it1 = item; else it0 = item;
                    }
                    const float scale = 1.0f / sqrtf((float)qk);
                    const float* lc = s->key_cache + (size_t)ai * p->seq_len * kvl;
                    float* att = GB.att + (size_t)tid * p->seq_len;
                    for (long item = it0; item < it1; item++) {
                        const int b = (int)(item / nh), h = (int)(item % nh);
                        const int all = GB.nsel[b] < 0, T = all ? pos0 + b + 1 : GB.nsel[b];
                        const int* sl = GB.isel + (size_t)b * GB.W;
                        const float* q = GB.qabs + ((size_t)h * B + b) * kvl;
                        float m = -FLT_MAX;
                        for (int j = 0; j < T; j++) {
                            const float* kv = lc + (size_t)(all ? j : sl[j]) * kvl;
                            float sc = 0.0f;
                            for (int i = 0; i < kvl; i++) sc += q[i] * kv[i];
                            sc *= scale;
                            att[j] = sc;
                            m = sc > m ? sc : m;
                        }
                        float sm = 0.0f;
                        for (int j = 0; j < T; j++) { att[j] = expf(att[j] - m); sm += att[j]; }
                        const float inv = 1.0f / sm;
                        float* o = GB.ctx + ((size_t)h * B + b) * kvl;
                        memset(o, 0, kvl * sizeof(float));
                        for (int j = 0; j < T; j++) {
                            const float* kv = lc + (size_t)(all ? j : sl[j]) * kvl;
                            const float a = att[j] * inv;
                            for (int i = 0; i < kvl; i++) o[i] += a * kv[i];
                        }
                    }
                }
                barrier(tb);
                for (int b = n0; b < n1; b++)
                    for (int h = 0; h < nh; h++)
                        quantize_vec(XQ + ((size_t)h * B + b) * kvl, XS + ((size_t)h * B + b) * (kvl / GS), GB.ctx + ((size_t)h * B + b) * kvl, kvl);
                node_barrier(tb);
                for (int h = 0; h < nh; h++) { // per-head values W_v,h ctx_h
                    QuantizedTensor v = expert_view(w->kv_bv + ai, h, vh, kvl);
                    matmul_batch_node(GB.ov + (size_t)h * vh, nh * vh, XQ + (size_t)h * B * kvl, XS + (size_t)h * B * (kvl / GS), B, &v, kvl, vh, tb,
                                      NULL, NULL, NULL, NULL);
                }
                barrier(tb);
                for (int b = n0; b < n1; b++) quantize_vec(XQ + (size_t)b * nh * vh, XS + (size_t)b * (nh * vh / GS), GB.ov + (size_t)b * nh * vh, nh * vh);
                node_barrier(tb);
                matmul_batch_node(GB.y, D, XQ, XS, B, w->o_proj + ai, nh * vh, D, tb, NULL, NULL, glm_post_hc, &hcx);
                barrier(tb);
                ai++;
            } else if (l < p->g_dense) { // --------------------------------------------- dense FFN
                QuantizedTensor up1 = expert_view(w->w13 + l, 0, F, D), up3 = expert_view(w->w13 + l, 1, F, D);
                GlmSwiCtx sc = {GB.h1, GB.h3, F, p->g_limit};
                matmul_batch_node(GB.h1, F, XQ, XS, B, &up1, D, F, tb, &up3, GB.h3, glm_post_swiglu, &sc);
                barrier(tb);
                for (int b = n0; b < n1; b++) quantize_vec(XQ + (size_t)b * F, XS + (size_t)b * (F / GS), GB.h1 + (size_t)b * F, F);
                node_barrier(tb);
                matmul_batch_node(GB.y, D, XQ, XS, B, w->w2 + l, F, D, tb, NULL, NULL, glm_post_hc, &hcx);
                barrier(tb);
            } else { // ------------------------------------------------------------------- MoE
                const int mi = l - p->g_dense, ngd = D / GS, ngm = mh / GS;
                int8_t* GXQ = GBN[tb->node].gxq; float* GXS = GBN[tb->node].gxs;
                int8_t* GQ = GBN[tb->node].gq; float* GQS = GBN[tb->node].gqs;
                matmul_batch_node(GB.router, E, XQ, XS, B, w->router_node[tb->node] + mi, D, E, tb, NULL, NULL, NULL, NULL);
                barrier(tb);
                const float* bias = w->rbias + (size_t)mi * E;
                for (int b = b0; b < b1; b++) { // sigmoid scores; top-A by score + bias; normalized weights * scale
                    const float* rl = GB.router + (size_t)b * E;
                    int* sel = GB.sel + (size_t)b * A;
                    float* wt = GB.wt + (size_t)b * A, best[MOE_MAX_ACTIVE];
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
                }
                barrier(tb);
                if (tid == 0) { // slots grouped by expert; expert E = the shared expert (every token, weight 1)
                    int* cnt = GB.cnt;
                    memset(cnt, 0, (E + 2) * sizeof(int));
                    for (int b = 0; b < B; b++)
                        for (int j = 0; j < A; j++) cnt[GB.sel[b * A + j] + 1]++;
                    cnt[E + 1] = B;
                    for (int e = 0; e <= E; e++) cnt[e + 1] += cnt[e];
                    memcpy(GB.fill, cnt, (E + 1) * sizeof(int));
                    for (int b = 0; b < B; b++) {
                        for (int j = 0; j < A; j++) {
                            int e = GB.sel[b * A + j], slot = GB.fill[e]++;
                            GB.stok[slot] = b;
                            GB.sw[slot] = GB.wt[b * A + j];
                        }
                        int slot = GB.fill[E]++;
                        GB.stok[slot] = b;
                        GB.sw[slot] = 1.0f;
                    }
                    GB.U = 0;
                    for (int e = 0; e <= E; e++) if (cnt[e + 1] > cnt[e]) GB.ue[GB.U++] = e;
                    for (int k = 0; k < NN; k++) atomic_store_explicit(&GB.ctr[k].v, 0, memory_order_relaxed);
                }
                barrier(tb);
                const int* off = GB.cnt;
                const int S = off[E + 1];
                { // gather the quantized inputs of the slots (each node all slots)
                    int s0, s1;
                    tok_range(S, tb->ltid, tb->nthn, &s0, &s1);
                    for (int sl = s0; sl < s1; sl++) {
                        int b = GB.stok[sl];
                        memcpy(GXQ + (size_t)sl * D, XQ + (size_t)b * D, D);
                        memcpy(GXS + (size_t)sl * ngd, XS + (size_t)b * ngd, ngd * sizeof(float));
                    }
                }
                node_barrier(tb);
                { // up projections + clamped SwiGLU * weight: this node's half of every used expert, dynamic 16-row chunks
                    int lo, hi2;
                    node_rows(mh, tb->node, &lo, &hi2);
                    const int half = hi2 - lo, U = GB.U, cpe = (half + 15) / 16;
                    const float lim = p->g_limit;
                    for (;;) {
                        int tk = atomic_fetch_add_explicit(&GB.ctr[tb->node].v, 1, memory_order_relaxed);
                        if (tk >= U * cpe) break;
                        const int e = GB.ue[tk / cpe], c = tk % cpe, n = off[e + 1] - off[e];
                        const int a0 = lo + c * 16, a1 = lo + (c * 16 + 16 < half ? c * 16 + 16 : half);
                        QuantizedTensor e1, e3;
                        if (e < E) { e1 = expert_view(w->ex1 + mi, e, mh, D); e3 = expert_view(w->ex3 + mi, e, mh, D); }
                        else { e1 = expert_view(w->sh13 + mi, 0, mh, D); e3 = expert_view(w->sh13 + mi, 1, mh, D); }
                        float* h1 = GB.gh + (size_t)off[e] * mh;
                        float* h3 = GB.gh3 + (size_t)off[e] * mh;
                        matmul_rows_batch(h1, mh, GXQ + (size_t)off[e] * D, GXS + (size_t)off[e] * ngd, n, &e1, D, a0, a1);
                        matmul_rows_batch(h3, mh, GXQ + (size_t)off[e] * D, GXS + (size_t)off[e] * ngd, n, &e3, D, a0, a1);
                        for (int t = 0; t < n; t++) {
                            const float wgt = GB.sw[off[e] + t];
                            float* x1 = h1 + (size_t)t * mh;
                            const float* x3 = h3 + (size_t)t * mh;
                            for (int i = a0; i < a1; i++) {
                                float gg = x1[i] < lim ? x1[i] : lim, u = x3[i] < -lim ? -lim : (x3[i] > lim ? lim : x3[i]);
                                x1[i] = silu_f(gg) * u * wgt;
                            }
                        }
                    }
                }
                barrier(tb);
                {
                    int s0, s1;
                    tok_range(S, tb->ltid, tb->nthn, &s0, &s1);
                    for (int sl = s0; sl < s1; sl++) quantize_vec(GQ + (size_t)sl * mh, GQS + (size_t)sl * ngm, GB.gh + (size_t)sl * mh, mh);
                }
                node_barrier(tb);
                part_range_node(D, tb, &r0, &r1);
                if (r0 < r1) { // down projections on this thread's rows of D, summed per token, then the mHC update
                    for (int u = 0; u < GB.U; u++) {
                        const int e = GB.ue[u], n = off[e + 1] - off[e];
                        QuantizedTensor e2 = e < E ? expert_view(w->ex2 + mi, e, D, mh) : w->sh2[mi];
                        matmul_rows_batch(GB.gout + (size_t)off[e] * D, D, GQ + (size_t)off[e] * mh, GQS + (size_t)off[e] * ngm, n, &e2, mh, r0, r1);
                    }
                    for (int b = 0; b < B; b++)
                        for (int i = r0; i < r1; i++) GB.y[(size_t)b * D + i] = 0.0f;
                    for (int sl = 0; sl < S; sl++) {
                        float* yv = GB.y + (size_t)GB.stok[sl] * D;
                        const float* o = GB.gout + (size_t)sl * D;
                        for (int i = r0; i < r1; i++) yv[i] += o[i];
                    }
                    glm_post_hc(&hcx, 0, B, r0, r1);
                }
                barrier(tb);
            }
        }
    }
    // logits of the last token: mean of its streams, RMSNorm, classifier
    {
        const float* x = GB.hs + (size_t)(B - 1) * N;
        float* xm = tb->xb;
        float ss = 0.0f;
        for (int i = 0; i < D; i++) {
            float v = 0.0f;
            for (int n = 0; n < hc; n++) v += x[(size_t)n * D + i];
            xm[i] = v / hc;
            ss += xm[i] * xm[i];
        }
        const float inv = 1.0f / sqrtf(ss / D + NORM_EPS);
        for (int i = 0; i < D; i++) xm[i] = w->final_norm[i] * (xm[i] * inv);
        quantize_vec(tb->q, tb->s, xm, D);
        part_range_node(p->vocab_size, tb, &r0, &r1);
        matmul_rows(s->logits, tb->q, tb->s, w->lm_head, D, r0, r1);
    }
    barrier(tb);
}
