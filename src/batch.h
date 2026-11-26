// ----------------------------------------------------------------------------
// Batched prefill: push a block of B prompt tokens through the model at once.
// Every weight row is loaded once per block and reused for all B tokens (matrix x matrix),
// which turns the memory-bound decode loop into a compute-bound one.

/* Tokens per block. Every block streams all weights from memory once, so larger blocks read them
 * less often and pass fewer barriers: 512 instead of 128 gave +5..11% prefill (8B Q8: w2 134 -> 113
 * ms, barrier waits 168 -> 97 ms per 512 tokens). The buffers grow with it (8B: hb/hb2 29 MB each). */
#ifndef BATCH_MAX
#define BATCH_MAX 512
#endif

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
    float* save_in;    // when set: (n_lin, SPEC_MAX_TOK, lin_in) copy of every linear layer's fused input,
                       // so the accepted prefix of a speculative verify can be committed without a second weight read
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

static float* SPEC_IN = NULL; // (n_lin, SPEC_MAX_TOK, lin_in) saved fused inputs of a speculative verify pass

// out[b*ldo + i] = W[i,:] . x_b for rows i in [r0, r1) and tokens b in [0, B)
static void matmul_rows_batch(float* out, int ldo, const int8_t* xq, const float* xs, int B,
                              const QuantizedTensor* w, int n, int r0, int r1) {
    if (w->bits == 4) matmul_rows_batch_q4(out, ldo, xq, xs, B, w, n, r0, r1);
    else matmul_rows_batch_q8(out, ldo, xq, xs, B, w, n, r0, r1);
}

/* Batched matmul of the node's rows of a d-row matrix, balanced over the node's threads. The work
 * items are (block of 4 rows, tile of 8 tokens), numbered row-block major, and every thread takes an
 * equal contiguous share: with 4-row granularity alone the threads got e.g. 20 or 24 rows of w2 and
 * the others waited for the slowest at the barrier. Tiles stay the outer loop, so a thread's rows
 * stay in L2 across tiles. With w2 set the same rows of a second matrix go to out2 (w1 | w3).
 * post(ctx, b0, b1, r0, r1) runs on every piece the thread computed. */
typedef void (*BatchPost)(void* ctx, int b0, int b1, int r0, int r1);
static void matmul_batch_node(float* out, int ldo, const int8_t* xq, const float* xs, int B, const QuantizedTensor* w,
                              int n, int d, const ThreadBuf* tb, const QuantizedTensor* w2, float* out2,
                              BatchPost post, void* ctx) {
    int lo, hi;
    node_rows(d, tb->node, &lo, &hi);
    const int NR = (hi - lo) / 4, NT = (B + 7) / 8, ngx = n / GS;
    const long tot = (long)NR * NT, l0 = tot * tb->ltid / tb->nthn, l1 = tot * (tb->ltid + 1) / tb->nthn;
    for (int pass = 0; pass < (w2 ? 3 : 2); pass++) // one matrix at a time (w1 then w3), then post
        for (int t = 0; t < NT; t++) {
            const long s = l0 - t, e = l1 - t; // items rb * NT + t in [l0, l1): rb in [ceil(s / NT), ceil(e / NT))
            const int rbs = s <= 0 ? 0 : (int)((s + NT - 1) / NT);
            int rbe = e <= 0 ? 0 : (int)((e + NT - 1) / NT);
            if (rbe > NR) rbe = NR;
            if (rbs >= rbe) continue;
            const int b0 = 8 * t, nb = B - b0 < 8 ? B - b0 : 8, r0 = lo + 4 * rbs, r1 = lo + 4 * rbe;
            const int last = w2 ? 2 : 1;
            if (pass < last) {
                const QuantizedTensor* m = pass ? w2 : w;
                float* o = pass ? out2 : out;
                matmul_rows_batch(o + (size_t)b0 * ldo, ldo, xq + (size_t)b0 * n, xs + (size_t)b0 * ngx, nb, m, n, r0, r1);
            } else if (post) post(ctx, b0, b0 + nb, r0, r1);
        }
    if (tb->ltid == tb->nthn - 1 && lo + 4 * NR < hi) { // rows beyond the last full block
        matmul_rows_batch(out, ldo, xq, xs, B, w, n, lo + 4 * NR, hi);
        if (w2) matmul_rows_batch(out2, ldo, xq, xs, B, w2, n, lo + 4 * NR, hi);
        if (post) post(ctx, 0, B, lo + 4 * NR, hi);
    }
}
typedef struct { float* x; const float* o; int ld; } ResidCtx; // x[b][i] += o[b][i]
static void post_residual(void* c, int b0, int b1, int r0, int r1) {
    const ResidCtx* r = c;
    for (int b = b0; b < b1; b++)
        for (int i = r0; i < r1; i++) r->x[(size_t)b * r->ld + i] += r->o[(size_t)b * r->ld + i];
}
typedef struct { float* h1; const float* h3; int ld; } SwigluCtx; // h1 = silu(h1) * h3
static void post_swiglu(void* c, int b0, int b1, int r0, int r1) {
    const SwigluCtx* g = c;
    for (int b = b0; b < b1; b++) {
        float* h1 = g->h1 + (size_t)b * g->ld;
        const float* h3 = g->h3 + (size_t)b * g->ld;
        for (int i = r0; i < r1; i++) { float v = h1[i]; v *= (1.0f / (1.0f + expf(-v))); h1[i] = v * h3[i]; }
    }
}

static void post_swiglu_hyb(void* c, int b0, int b1, int r0, int r1) { // h1 = silu_f(h1) * h3
    const SwigluCtx* g = c;
    for (int b = b0; b < b1; b++)
        for (int i = r0; i < r1; i++) g->h1[(size_t)b * g->ld + i] = silu_f(g->h1[(size_t)b * g->ld + i]) * g->h3[(size_t)b * g->ld + i];
}
// first (token, head) pair whose cumulative cost reaches c; pair (b, h) costs pos0 + b + 1 (attention)
static int att_item_at(long c, int H, int B, int pos0) {
    long cum = 0;
    for (int b = 0; b < B; b++) {
        const long T = pos0 + b + 1, cb = (long)H * T;
        if (c < cum + cb) return b * H + (int)((c - cum + T - 1) / T);
        cum += cb;
    }
    return B * H;
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
    const int kvi8 = s->kv_type == KV_I8;
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
                if (!kvi8) for (int b = 0; b < B; b++) {
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
                    if (!kvi8) memcpy(s->key_cache + loff + ((size_t)g * p->seq_len + pos0 + b) * hs, kv, hs * sizeof(float));
                }
            }
            barrier(tb);
            int g0, g1;
            node_groups(p->n_kv_heads, tb->node, &g0, &g1);
            if (kvi8) { // quantize the K/V rows of this block (each node its own kv groups)
                for (int task = tb->ltid; task < (g1 - g0) * B; task += tb->nthn) {
                    int g = g0 + task % (g1 - g0), b = task / (g1 - g0);
                    kv_store_row(kv_k_i8(s, p, fi, g, pos0 + b), kv_ks(s, p, fi, g, pos0 + b), BT.k + (size_t)b * kv_dim + (size_t)g * hs, hs);
                    kv_store_row(kv_v_i8(s, p, fi, g, pos0 + b), kv_vs(s, p, fi, g, pos0 + b), BT.v + (size_t)b * kv_dim + (size_t)g * hs, hs);
                }
                node_barrier(tb);
            }
            // causal attention, items (kv group of this node, token); gate applied to the output
            float* att = BT.att + (size_t)tid * kv_mul * p->seq_len;
            const int ng = hs / GS;
            for (int item = tb->ltid; item < (g1 - g0) * B; item += tb->nthn) {
                int g = g0 + item % (g1 - g0), b = item / (g1 - g0);
                int T = pos0 + b + 1;
                const size_t kof = (size_t)g * p->seq_len * hs;
                const float* kb = kvi8 ? NULL : s->key_cache + loff + kof;
                const float* vb = kvi8 ? NULL : s->value_cache + loff + kof;
                const int8_t* kbq = kvi8 ? s->key_cache_q + loff + kof : NULL;
                const int8_t* vbq = kvi8 ? s->value_cache_q + loff + kof : NULL;
                const float* kbs = kvi8 ? s->key_cache_s + (loff + kof) / GS : NULL;
                const float* vbs = kvi8 ? s->value_cache_s + (loff + kof) / GS : NULL;
                const float* qb = BT.q + (size_t)b * q_dim + (size_t)g * kv_mul * hs;
                float m[KV_MUL_MAX], sum[KV_MUL_MAX];
                for (int r = 0; r < kv_mul; r++) m[r] = -FLT_MAX;
                if (kvi8) {
                    int8_t qq[KV_MUL_MAX * hs];
                    float qsc[KV_MUL_MAX * (hs / GS)];
                    for (int r = 0; r < kv_mul; r++) quantize_vec(qq + r * hs, qsc + r * ng, qb + r * hs, hs);
                    for (int t = 0; t < T; t++) {
                        const int8_t* k = kbq + (size_t)t * hs;
                        const float* ks = kbs + (size_t)t * ng;
                        for (int r = 0; r < kv_mul; r++) {
                            float score = kv_dot_q8(qq + r * hs, qsc + r * ng, k, ks, hs) * att_scale;
                            att[(size_t)r * p->seq_len + t] = score;
                            m[r] = score > m[r] ? score : m[r];
                        }
                    }
                } else {
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
                    if (kvi8) {
                        float vf[hs];
                        kv_dequant_row(vf, vbq + (size_t)t * hs, vbs + (size_t)t * ng, hs);
                        for (int r = 0; r < kv_mul; r++) {
                            float a = att[(size_t)r * p->seq_len + t];
                            float* o = ob + r * hs;
                            for (int i = 0; i < hs; i++) o[i] += a * vf[i];
                        }
                    } else {
                        const float* v = vb + (size_t)t * hs;
                        for (int r = 0; r < kv_mul; r++) {
                            float a = att[(size_t)r * p->seq_len + t];
                            float* o = ob + r * hs;
                            for (int i = 0; i < hs; i++) o[i] += a * v[i];
                        }
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
            {
                ResidCtx rc = {BT.x, BT.xb2, dim};
                matmul_batch_node(BT.xb2, dim, XQ, XS, B, w->wo + fi, q_dim, dim, tb, NULL, NULL, post_residual, &rc);
            }
            barrier(tb);
        } else {
            const int li = hyb_lidx(p, l);
            matmul_batch_node(BT.lin, LI, XQ, XS, B, w->w_in + li, dim, LI, tb, NULL, NULL, NULL, NULL);
            barrier(tb);
            if (BT.save_in) { // speculative verify: keep the fused inputs for a weightless commit of the accepted prefix
                size_t tot = (size_t)B * LI, i0 = tot * tid / nth, i1 = tot * (tid + 1) / nth;
                memcpy(BT.save_in + (size_t)li * SPEC_MAX_TOK * LI + i0, BT.lin + i0, (i1 - i0) * sizeof(float));
                barrier(tb);
            }
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
            {
                ResidCtx rc = {BT.x, BT.xb2, dim};
                matmul_batch_node(BT.xb2, dim, XQ, XS, B, w->w_out + li, vd, dim, tb, NULL, NULL, post_residual, &rc);
            }
            barrier(tb);
        }
        if (IMX.on) imx_dense(tb, IMX.ffn + (size_t)l * dim, BT.x, dim, B, dim, w->rms_ffn_weight + (size_t)l * dim);
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_ffn_weight + (size_t)l * dim, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);
        {
            SwigluCtx gc = {BT.hb, BT.hb2, hidden_dim};
            matmul_batch_node(BT.hb, hidden_dim, XQ, XS, B, w->w1 + l, dim, hidden_dim, tb, w->w3 + l, BT.hb2, post_swiglu_hyb, &gc);
        }
        barrier(tb);
        if (IMX.on) imx_dense(tb, IMX.w2 + (size_t)l * hidden_dim, BT.hb, hidden_dim, B, hidden_dim, NULL);
        for (int b = n0; b < n1; b++) quantize_vec(XQ + (size_t)b * hidden_dim, XS + (size_t)b * ngh, BT.hb + (size_t)b * hidden_dim, hidden_dim);
        barrier(tb);
        {
            ResidCtx rc = {BT.x, BT.xb2, dim};
            matmul_batch_node(BT.xb2, dim, XQ, XS, B, w->w2 + l, hidden_dim, dim, tb, NULL, NULL, post_residual, &rc);
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
