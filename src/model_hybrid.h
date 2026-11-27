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
    float emb_buf[dim];
    embed_row(w->q_tokens, P.token, emb_buf, dim);
    const float* emb = emb_buf;
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

    const int kvi8 = s->kv_type == KV_I8;
    for (int l = 0; l < p->n_layers; l++) {
        const float* xin = (l == 0) ? emb : x;
        PROF(tid, PH_NORM);
        node_norm_quant(tb, xin, w->rms_att_weight + (size_t)l * dim, dim);
        if (hyb_full(p, l)) {
            const int fi = hyb_fidx(p, l);
            const size_t loff = (size_t)fi * p->seq_len * kv_dim;
            float* vc = kvi8 ? NULL : s->value_cache + loff + (size_t)pos * hs;
            PROF(tid, PH_QKV);
            while (grab_rows(WORK_PHASES * l + 0, q_dim, tb, &r0, &r1)) matmul_rows(s->q, nq, ns, w->wq + fi, dim, r0, r1);
            while (grab_rows(WORK_PHASES * l + 6, q_dim, tb, &r0, &r1)) matmul_rows(s->gate, nq, ns, w->wgate + fi, dim, r0, r1);
            while (grab_rows(WORK_PHASES * l + 1, kv_dim, tb, &r0, &r1)) matmul_rows(s->k, nq, ns, w->wk + fi, dim, r0, r1);
            while (grab_rows(WORK_PHASES * l + 2, kv_dim, tb, &r0, &r1)) {
                matmul_rows(s->v, nq, ns, w->wv + fi, dim, r0, r1);
                if (!kvi8) for (int i = r0; i < r1; i++) vc[(size_t)(i / hs) * p->seq_len * hs + i % hs] = s->v[i];
            }
            barrier(tb);
            for (int task = tid; task < n_heads + p->n_kv_heads; task += P.nth) {
                if (task < n_heads) {
                    hyb_norm_rope(s->q + (size_t)task * hs, w->q_norm + (size_t)fi * hs, hs, rot, tb->rope_c, tb->rope_s);
                } else {
                    int g = task - n_heads;
                    float* kv = s->k + (size_t)g * hs;
                    hyb_norm_rope(kv, w->k_norm + (size_t)fi * hs, hs, rot, tb->rope_c, tb->rope_s);
                    if (!kvi8) memcpy(s->key_cache + loff + (size_t)g * p->seq_len * hs + (size_t)pos * hs, kv, hs * sizeof(float));
                }
            }
            barrier(tb);
            if (kvi8) { // quantize this position's K/V rows
                for (int g = tid; g < p->n_kv_heads; g += P.nth) {
                    kv_store_row(kv_k_i8(s, p, fi, g, pos), kv_ks(s, p, fi, g, pos), s->k + (size_t)g * hs, hs);
                    kv_store_row(kv_v_i8(s, p, fi, g, pos), kv_vs(s, p, fi, g, pos), s->v + (size_t)g * hs, hs);
                }
                barrier(tb);
            }
            PROF(tid, PH_ATT);
            // attention items (kv group, context chunk, query-head subset) over all threads
            int HS = 1;
            while (HS < kv_mul && p->n_kv_heads * C * HS * 2 <= P.nth && kv_mul % (HS * 2) == 0) HS *= 2;
            const int hper = kv_mul / HS;
            const int ng = hs / GS;
            for (int item = tid; item < p->n_kv_heads * C * HS; item += P.nth) {
                int g = item / (C * HS), c = (item / HS) % C, hsub = item % HS;
                int t0 = (int)((long)T * c / C), t1 = (int)((long)T * (c + 1) / C);
                const size_t kof = (size_t)g * p->seq_len * hs;
                const float* kb = kvi8 ? NULL : s->key_cache + loff + kof;
                const float* vb = kvi8 ? NULL : s->value_cache + loff + kof;
                const int8_t* kbq = kvi8 ? s->key_cache_q + loff + kof : NULL;
                const int8_t* vbq = kvi8 ? s->value_cache_q + loff + kof : NULL;
                const float* kbs = kvi8 ? s->key_cache_s + (loff + kof) / GS : NULL;
                const float* vbs = kvi8 ? s->value_cache_s + (loff + kof) / GS : NULL;
                for (int r = 0; r < hper; r++) {
                    int h = g * kv_mul + hsub * hper + r;
                    const float* q = s->q + (size_t)h * hs;
                    int8_t qq[hs];
                    float qsc[hs / GS];
                    if (kvi8) quantize_vec(qq, qsc, q, hs);
                    float* att = s->att + (size_t)h * p->seq_len;
                    float m = -FLT_MAX;
                    if (kvi8) {
                        for (int t = t0; t < t1; t++) {
                            float score = kv_dot_q8(qq, qsc, kbq + (size_t)t * hs, kbs + (size_t)t * ng, hs) * att_scale;
                            att[t] = score;
                            m = score > m ? score : m;
                        }
                    } else {
                        for (int t = t0; t < t1; t++) {
                            const float* k = kb + (size_t)t * hs;
                            float score = 0.0f;
                            for (int i = 0; i < hs; i++) score += q[i] * k[i];
                            score *= att_scale;
                            att[t] = score;
                            m = score > m ? score : m;
                        }
                    }
                    float* pr = P.part + ((size_t)h * C + c) * (hs + 2);
                    float sm = 0.0f;
                    float* acc = pr + 2;
                    memset(acc, 0, hs * sizeof(float));
                    for (int t = t0; t < t1; t++) {
                        float e = expf(att[t] - m);
                        sm += e;
                        if (kvi8) kv_axpy_i8(acc, e, vbq + (size_t)t * hs, vbs + (size_t)t * ng, hs);
                        else {
                            const float* v = vb + (size_t)t * hs;
                            for (int i = 0; i < hs; i++) acc[i] += e * v[i];
                        }
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

