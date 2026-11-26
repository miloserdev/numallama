static void forward_thread(int tid) {
    if (P.t->config.glm) { forward_glm_thread(tid); return; }
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
