
static void forward_batch_thread(int tid) {
    if (P.t->config.glm) { forward_glm_batch_thread(tid); return; }
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
    const int kvi8 = s->kv_type == KV_I8;
    const float att_scale = 1.0f / sqrtf((float)head_size);
    int r0, r1, b0, b1, n0, n1;
    int8_t* XQ = BN[tb->node].xq; // the node's quantized block inputs
    float* XS = BN[tb->node].xs;
    tok_range(B, tb->ltid, tb->nthn, &n0, &n1); // this thread's tokens for the node-local quantizations

    PROF(tid, PH_ROPE);
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
        PROF(tid, PH_NORM);
        // attention rmsnorm + quantize, split over tokens
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_att_weight + l * dim, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);

        PROF(tid, PH_QKV);
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
                    float* kc = kvi8 ? NULL : s->key_cache + loff + (size_t)(pos0 + b) * head_size;
                    for (int i = a; i < e; i += 2) {
                        int j = (i % head_size) / 2;
                        float v0 = k[i], v1 = k[i + 1];
                        k[i] = v0 * rc[j] - v1 * rs[j];
                        k[i + 1] = v0 * rs[j] + v1 * rc[j];
                    }
                    if (!kvi8) for (int i = a; i < e; i++) { kc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = k[i]; }
                }
            }
            a = va; e = ve;
            if (a < e) {
                matmul_rows_batch(BT.v, kv_dim, XQ, XS, B, w->wv + l, dim, a, e);
                for (int b = 0; b < B; b++) {
                    float* v = BT.v + (size_t)b * kv_dim;
                    if (w->bv) for (int i = a; i < e; i++) v[i] += w->bv[(size_t)l * kv_dim + i];
                    float* vc = kvi8 ? NULL : s->value_cache + loff + (size_t)(pos0 + b) * head_size;
                    if (!kvi8) for (int i = a; i < e; i++) { vc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = v[i]; }
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
                    if (!kvi8) memcpy(s->key_cache + loff + ((size_t)g * p->seq_len + pos0 + b) * head_size, kv, head_size * sizeof(float));
                }
            }
            barrier(tb);
        }
        if (kvi8) { // quantize the K/V rows of this block (each node its own kv groups)
            int qg0, qg1;
            node_groups(p->n_kv_heads, tb->node, &qg0, &qg1);
            for (int task = tb->ltid; task < (qg1 - qg0) * B; task += tb->nthn) {
                int g = qg0 + task % (qg1 - qg0), b = task / (qg1 - qg0);
                kv_store_row(kv_k_i8(s, p, l, g, pos0 + b), kv_ks(s, p, l, g, pos0 + b), BT.k + (size_t)b * kv_dim + (size_t)g * head_size, head_size);
                kv_store_row(kv_v_i8(s, p, l, g, pos0 + b), kv_vs(s, p, l, g, pos0 + b), BT.v + (size_t)b * kv_dim + (size_t)g * head_size, head_size);
            }
            node_barrier(tb);
        }

        PROF(tid, PH_ATT);
        // causal attention: the node's (token, query head) pairs, a pair costing the token's
        // position, are split among the node's threads by equal cost into contiguous ranges (a
        // round-robin split by (kv group, token) left the slowest thread at ~1.8x the mean). Heads of
        // the same kv group of a token are processed together, so their K and V rows load once.
        float* att = BT.att + (size_t)tid * kv_mul * p->seq_len;
        int g0, g1;
        node_groups(p->n_kv_heads, tb->node, &g0, &g1);
        const int H = (g1 - g0) * kv_mul; // query heads of the node per token
        const long cost = (long)H * ((long)B * pos0 + (long)B * (B + 1) / 2);
        int i0 = att_item_at(cost * tb->ltid / tb->nthn, H, B, pos0);
        int i1 = tb->ltid == tb->nthn - 1 ? B * H : att_item_at(cost * (tb->ltid + 1) / tb->nthn, H, B, pos0);
        for (int item = i0; item < i1;) {
            const int b = item / H, gl = (item % H) / kv_mul, ra = item % kv_mul;
            const int base = b * H + gl * kv_mul, rb = base + kv_mul <= i1 ? kv_mul : i1 - base;
            item = base + rb;
            const int g = g0 + gl, T = pos0 + b + 1;
            const size_t kof = (size_t)g * p->seq_len * head_size;
            const float* kb = kvi8 ? NULL : s->key_cache + loff + kof;
            const float* vb = kvi8 ? NULL : s->value_cache + loff + kof;
            const int8_t* kbq = kvi8 ? s->key_cache_q + loff + kof : NULL;
            const int8_t* vbq = kvi8 ? s->value_cache_q + loff + kof : NULL;
            const float* kbs = kvi8 ? s->key_cache_s + (loff + kof) / GS : NULL;
            const float* vbs = kvi8 ? s->value_cache_s + (loff + kof) / GS : NULL;
            const int ng = head_size / GS;
            const float* qb = BT.q + (size_t)b * q_dim + (size_t)g * kv_mul * head_size;
            float m[KV_MUL_MAX], sum[KV_MUL_MAX];
            for (int r = ra; r < rb; r++) m[r] = -FLT_MAX;
            if (kvi8) {
                int8_t qq[KV_MUL_MAX * head_size];
                float qsc[KV_MUL_MAX * (head_size / GS)];
                for (int r = ra; r < rb; r++) quantize_vec(qq + r * head_size, qsc + r * ng, qb + r * head_size, head_size);
                for (int t = 0; t < T; t++) {
                    const int8_t* k = kbq + (size_t)t * head_size;
                    const float* ks = kbs + (size_t)t * ng;
                    for (int r = ra; r < rb; r++) {
                        float score = kv_dot_q8(qq + r * head_size, qsc + r * ng, k, ks, head_size) * att_scale;
                        att[(size_t)r * p->seq_len + t] = score;
                        m[r] = score > m[r] ? score : m[r];
                    }
                }
            } else {
                for (int t = 0; t < T; t++) {
                    const float* k = kb + (size_t)t * head_size;
                    for (int r = ra; r < rb; r++) {
                        const float* q = qb + r * head_size;
                        float score = 0.0f;
                        for (int i = 0; i < head_size; i++) { score += q[i] * k[i]; }
                        score *= att_scale;
                        att[(size_t)r * p->seq_len + t] = score;
                        m[r] = score > m[r] ? score : m[r];
                    }
                }
            }
            for (int r = ra; r < rb; r++) {
                float* ar = att + (size_t)r * p->seq_len;
                float sm = 0.0f;
                for (int t = 0; t < T; t++) { float ev = expf(ar[t] - m[r]); ar[t] = ev; sm += ev; }
                sum[r] = 1.0f / sm;
            }
            float* ob = BT.xb + (size_t)b * q_dim + (size_t)g * kv_mul * head_size;
            memset(ob + (size_t)ra * head_size, 0, (size_t)(rb - ra) * head_size * sizeof(float));
            for (int t = 0; t < T; t++) {
                if (kvi8) {
                    float vf[head_size];
                    kv_dequant_row(vf, vbq + (size_t)t * head_size, vbs + (size_t)t * ng, head_size);
                    for (int r = ra; r < rb; r++) {
                        float a = att[(size_t)r * p->seq_len + t];
                        float* o = ob + r * head_size;
                        for (int i = 0; i < head_size; i++) { o[i] += a * vf[i]; }
                    }
                } else {
                    const float* v = vb + (size_t)t * head_size;
                    for (int r = ra; r < rb; r++) {
                        float a = att[(size_t)r * p->seq_len + t];
                        float* o = ob + r * head_size;
                        for (int i = 0; i < head_size; i++) { o[i] += a * v[i]; }
                    }
                }
            }
            for (int r = ra; r < rb; r++) {
                float* o = ob + r * head_size;
                for (int i = 0; i < head_size; i++) { o[i] *= sum[r]; }
            }
        }
        barrier(tb);

        if (IMX.on) imx_dense(tb, IMX.wo + (size_t)l * IMX_WO_W, BT.xb, q_dim, B, q_dim, NULL);
        PROF(tid, PH_MERGE);
        // quantize the attention output, split over tokens
        for (int b = n0; b < n1; b++) {
            quantize_vec(XQ + (size_t)b * q_dim, XS + (size_t)b * ngq, BT.xb + (size_t)b * q_dim, q_dim);
        }
        barrier(tb);

        PROF(tid, PH_WO);
        // output projection + residual
        {
            ResidCtx rc = {BT.x, BT.xb2, dim};
            matmul_batch_node(BT.xb2, dim, XQ, XS, B, w->wo + l, q_dim, dim, tb, NULL, NULL, post_residual, &rc);
        }
        barrier(tb);

        if (IMX.on && p->n_experts == 0) imx_dense(tb, IMX.ffn + (size_t)l * dim, BT.x, dim, B, dim, w->rms_ffn_weight + l * dim);
        PROF(tid, PH_FNORM);
        // ffn rmsnorm + quantize, split over tokens
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_ffn_weight + l * dim, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);

        PROF(tid, PH_W13);
        if (p->n_experts > 0) {
            moe_batch(tb, l, B);
        } else {
            // fused w1|w3 + SwiGLU
            {
                SwigluCtx gc = {BT.hb, BT.hb2, hidden_dim};
                matmul_batch_node(BT.hb, hidden_dim, XQ, XS, B, w->w1 + l, dim, hidden_dim, tb, w->w3 + l, BT.hb2, post_swiglu, &gc);
            }
            barrier(tb);

            if (IMX.on) imx_dense(tb, IMX.w2 + (size_t)l * hidden_dim, BT.hb, hidden_dim, B, hidden_dim, NULL);
            PROF(tid, PH_QH);
            // quantize the hidden activations, split over tokens
            for (int b = n0; b < n1; b++) {
                quantize_vec(XQ + (size_t)b * hidden_dim, XS + (size_t)b * ngh, BT.hb + (size_t)b * hidden_dim, hidden_dim);
            }
            barrier(tb);

            PROF(tid, PH_W2);
            // w2 + residual
            {
                ResidCtx rc = {BT.x, BT.xb2, dim};
                matmul_batch_node(BT.xb2, dim, XQ, XS, B, w->w2 + l, hidden_dim, dim, tb, NULL, NULL, post_residual, &rc);
            }
        }
        barrier(tb);
    }

    PROF(tid, PH_CLS);
    if (BT.logits_all) { // logits of every position
        for (int b = n0; b < n1; b++) {
            rmsnorm(tb->xb, BT.x + (size_t)b * dim, w->rms_final_weight, dim);
            quantize_vec(XQ + (size_t)b * dim, XS + (size_t)b * ngd, tb->xb, dim);
        }
        barrier(tb);
        part_range_node(p->vocab_size, tb, &r0, &r1);
        matmul_rows_batch(BT.logits_all, p->vocab_size, XQ, XS, B, w->wcls, dim, r0, r1);
        barrier(tb);
        PROF(tid, PH_IDLE);
        return;
    }
    // logits for the last token only (private rmsnorm + quantize, split classifier rows)
    rmsnorm(tb->xb, BT.x + (size_t)(B - 1) * dim, w->rms_final_weight, dim);
    quantize_vec(tb->q, tb->s, tb->xb, dim);
    part_range_node(p->vocab_size, tb, &r0, &r1);
    matmul_rows(s->logits, tb->q, tb->s, w->wcls, dim, r0, r1);
    barrier(tb);
    PROF(tid, PH_IDLE);
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
    const int kvi8 = s->kv_type == KV_I8;
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
                float* kc = kvi8 ? NULL : s->key_cache + loff + (size_t)(pos0 + b) * head_size;
                if (w->bk) for (int i = r0; i < r1; i++) k[i] += w->bk[(size_t)l * kv_dim + i];
                if (qkn) continue; // per-head norm + RoPE below, then into the cache
                for (int i = r0; i < r1; i += 2) {
                    int j = (i % head_size) / 2;
                    float v0 = k[i], v1 = k[i + 1];
                    k[i] = v0 * rc[j] - v1 * rs[j];
                    k[i + 1] = v0 * rs[j] + v1 * rc[j];
                }
                if (!kvi8) for (int i = r0; i < r1; i++) kc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = k[i];
            }
        }
        while (grab_rows(WORK_PHASES * l + 2, kv_dim, tb, &r0, &r1)) {
            matmul_rows_batch(BT.v, kv_dim, nq, ns, B, w->wv + l, dim, r0, r1);
            for (int b = 0; b < B; b++) {
                float* v = BT.v + (size_t)b * kv_dim;
                float* vc = kvi8 ? NULL : s->value_cache + loff + (size_t)(pos0 + b) * head_size;
                if (w->bv) for (int i = r0; i < r1; i++) v[i] += w->bv[(size_t)l * kv_dim + i];
                if (!kvi8) for (int i = r0; i < r1; i++) vc[(size_t)(i / head_size) * p->seq_len * head_size + i % head_size] = v[i];
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
                    if (!kvi8) memcpy(s->key_cache + loff + (size_t)g * p->seq_len * head_size + (size_t)(pos0 + b) * head_size, kv, head_size * sizeof(float));
                }
            }
            barrier(tb);
        }
        if (kvi8) { // quantize the K/V rows of this block (each node its own kv groups)
            int qg0, qg1;
            node_groups(p->n_kv_heads, tb->node, &qg0, &qg1);
            for (int task = tb->ltid; task < (qg1 - qg0) * B; task += tb->nthn) {
                int g = qg0 + task % (qg1 - qg0), b = task / (qg1 - qg0);
                kv_store_row(kv_k_i8(s, p, l, g, pos0 + b), kv_ks(s, p, l, g, pos0 + b), BT.k + (size_t)b * kv_dim + (size_t)g * head_size, head_size);
                kv_store_row(kv_v_i8(s, p, l, g, pos0 + b), kv_vs(s, p, l, g, pos0 + b), BT.v + (size_t)b * kv_dim + (size_t)g * head_size, head_size);
            }
            node_barrier(tb);
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
            const size_t kof = (size_t)g * p->seq_len * head_size;
            const float* kb = kvi8 ? NULL : s->key_cache + loff + kof;
            const float* vb = kvi8 ? NULL : s->value_cache + loff + kof;
            const int8_t* kbq = kvi8 ? s->key_cache_q + loff + kof : NULL;
            const int8_t* vbq = kvi8 ? s->value_cache_q + loff + kof : NULL;
            const float* kbs = kvi8 ? s->key_cache_s + (loff + kof) / GS : NULL;
            const float* vbs = kvi8 ? s->value_cache_s + (loff + kof) / GS : NULL;
            const int ng = head_size / GS;
            for (int b = B * ts / TS; b < B * (ts + 1) / TS; b++) {
                int te = pos0 + b + 1 < t1 ? pos0 + b + 1 : t1; // causal end for token b
                int8_t qq[KV_MUL_MAX * head_size];
                float qsc[KV_MUL_MAX * (head_size / GS)];
                if (kvi8)
                    for (int r = 0; r < kv_mul; r++)
                        quantize_vec(qq + r * head_size, qsc + r * ng, BT.q + (size_t)b * q_dim + (size_t)(g * kv_mul + r) * head_size, head_size);
                for (int r = 0; r < kv_mul; r++) {
                    int h = g * kv_mul + r;
                    float* pr = BT.mpart + (((size_t)b * n_heads + h) * C + c) * hp;
                    if (te <= t0) { pr[1] = 0.0f; continue; }
                    const float* q = BT.q + (size_t)b * q_dim + (size_t)h * head_size;
                    float* ar = att + (size_t)r * p->seq_len;
                    float m = -FLT_MAX;
                    if (kvi8) {
                        for (int t = t0; t < te; t++) {
                            float score = kv_dot_q8(qq + r * head_size, qsc + r * ng, kbq + (size_t)t * head_size, kbs + (size_t)t * ng, head_size) * att_scale;
                            ar[t - t0] = score;
                            m = score > m ? score : m;
                        }
                    } else {
                        for (int t = t0; t < te; t++) {
                            const float* k = kb + (size_t)t * head_size;
                            float score = 0.0f;
                            for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                            score *= att_scale;
                            ar[t - t0] = score;
                            m = score > m ? score : m;
                        }
                    }
                    float sm = 0.0f;
                    float* o = pr + 2;
                    memset(o, 0, head_size * sizeof(float));
                    for (int t = t0; t < te; t++) {
                        float e = expf(ar[t - t0] - m);
                        sm += e;
                        if (kvi8) kv_axpy_i8(o, e, vbq + (size_t)t * head_size, vbs + (size_t)t * ng, head_size);
                        else {
                            const float* v = vb + (size_t)t * head_size;
                            for (int i = 0; i < head_size; i++) o[i] += e * v[i];
                        }
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
    if (p->glm) { glm_init_batch(t); return; } // GLM: buffers of its own (glm_batch.h)
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
    if (n == 1 || (RECURRENT(&transformer->config) && !HYB_BATCH)) { // no per-position logits without the batch path
        for (int i = 0; i < n; i++) memcpy(SPEC_LOGITS + (size_t)i * V, forward(transformer, tokens[i], pos0 + i), V * sizeof(float));
        return SPEC_LOGITS;
    }
    if (RECURRENT(&transformer->config)) { // one batched pass: the weights are read once, the recurrent state advances n tokens
        if (pos0 != transformer->state.next_pos) {
            fprintf(stderr, "hybrid model: position %d out of order (state at %d)\n", pos0, transformer->state.next_pos);
            exit(EXIT_FAILURE);
        }
        Config* hp = &transformer->config;
        if (hp->hybrid && !SPEC_IN) SPEC_IN = malloc((size_t)(hp->n_layers - hp->n_attn) * SPEC_MAX * hp->lin_in * sizeof(float));
        P.t = transformer;
        P.mode = 1;
        BT.B = n;
        BT.pos0 = pos0;
        BT.tok = tokens;
        BT.logits_all = SPEC_LOGITS;
        BT.save_in = hp->hybrid ? SPEC_IN : NULL; // GLM: no saved inputs, a partial accept replays through forward_batch
        wake_workers();
        forward_batch_thread(0);
        P.mode = 0;
        BT.logits_all = NULL;
        BT.save_in = NULL;
        transformer->state.next_pos = pos0 + n;
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

static volatile int g_srv_abort; // defined below with the other server globals

float* forward_batch(Transformer* transformer, const int* tokens, int n, int pos0, int* done_out) {
    use_model(transformer);
    if (transformer->config.hybrid || transformer->config.glm) {
        if (!HYB_BATCH) { // token by token
            float* l = NULL;
            int i = 0;
            for (; i < n && !g_srv_abort; i++) l = forward(transformer, tokens[i], pos0 + i);
            if (done_out) *done_out = i;
            return l;
        }
        if (pos0 != 0 && pos0 != transformer->state.next_pos) {
            fprintf(stderr, "hybrid model: position %d out of order (state at %d)\n", pos0, transformer->state.next_pos);
            exit(EXIT_FAILURE);
        }
        transformer->state.next_pos = pos0; // advanced block by block below (a 1-token block goes through forward())
    }

    float* logits = NULL;
    int done = 0;
    for (done = 0; done < n && !g_srv_abort;) {
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
    if (done_out) *done_out = done;
    return logits;
}
