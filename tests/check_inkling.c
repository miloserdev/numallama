#define TESTING
#include "../src/runq.c"
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.ink2\n", argv[0]); return 2; }
    InklingModel m;
    if (!inkling_load(argv[1], &m)) { fprintf(stderr, "load failed\n"); return 1; }
    printf("layers %d slots %d dim %d inter %d heads %d/%d hd %d vocab %d experts %d/%d\n",
           m.phys_layers, m.slots, m.dim, m.inter, m.heads, m.kv_heads, m.head_dim, m.vocab,
           m.experts, m.active);
    printf("loop %d..%d x%d rank %d rel %d/%d window %d conv %d res %d local %08x\n",
           m.loop_start, m.loop_end, m.passes, m.gate_rank, m.d_rel, m.rel_extent, m.window,
           m.conv_k, m.res_block, m.local_mask);
    printf("tensors %d\n", m.n_tensors);
    const char *names[] = {"embed.weight", "layers.0.mlp.experts.w13", "layers.23.mlp.experts.w2",
                           "layers.6.controllers.attn.down_weight", "norm.weight"};
    for (int i = 0; i < 5; i++) {
        const InkTensor *t = inkling_find(&m, names[i]);
        if (!t) { fprintf(stderr, "missing %s\n", names[i]); return 1; }
        printf("%-42s dtype %d dims", t->name, t->dtype);
        for (int j = 0; j < t->ndim; j++) printf(" %d", t->dims[j]);
        printf(" bytes %zu\n", t->bytes);
    }
    InklingState st;
    if (!inkling_state_alloc(&m, 128, &st)) { fprintf(stderr, "state alloc failed\n"); return 1; }
    printf("state ctx %d slots %d kv %.1f MB conv %.1f KB slotmap %d %d %d %d %d\n", st.ctx, st.slots,
           2.0 * st.slots * m.kv_heads * st.ctx * m.head_dim / 1e6,
           st.slots * (2.0 * m.kv_heads * m.head_dim + 2.0 * m.dim) * (m.conv_k - 1) * 4 / 1e3,
           inkling_slot_phys(&m, 0), inkling_slot_phys(&m, 23), inkling_slot_phys(&m, 24),
           inkling_slot_phys(&m, 47), inkling_slot_phys(&m, 36));
    inkling_state_free(&st);

    InklingRuntime rt;
    if (!inkling_runtime_init(&m, &rt)) { fprintf(stderr, "runtime init failed\n"); return 1; }
    float *h = malloc((size_t)m.dim * sizeof(float));
    float *lg = malloc((size_t)m.unpadded * sizeof(float));
    for (int i = 0; i < m.dim; i++) h[i] = sinf(0.001f * (i + 1));
    inkling_lm_head(&rt, h, lg);
    int best = 0;
    for (int i = 1; i < m.unpadded; i++) if (lg[i] > lg[best]) best = i;
    printf("M1 head top-1 id %d logit %.6f\n", best, lg[best]);
    free(h); free(lg); inkling_runtime_free(&rt);

    // M2 primitives: sconv window, L2 clip and depth mix
    {
        float w[4] = {1, 1, 1, 1}, hist[3] = {0, 0, 0}, o[1];
        float cur = 1.0f;
        inkling_sconv(o, &cur, 1, w, hist, 4);
        if (fabsf(o[0] - 1.0f) > 1e-6f) { fprintf(stderr, "sconv1 %f\n", o[0]); return 1; }
        cur = 2.0f;
        inkling_sconv(o, &cur, 1, w, hist, 4);
        if (fabsf(o[0] - 3.0f) > 1e-6f) { fprintf(stderr, "sconv2 %f\n", o[0]); return 1; }
        float v[2] = {3, 4};
        inkling_clip(v, NULL, 2, 1.0f);
        if (fabsf(sqrtf(v[0] * v[0] + v[1] * v[1]) - 1.0f) > 1e-6f) { fprintf(stderr, "clip\n"); return 1; }
        float a[4] = {1, 2, 3, 4}, b[4] = {2, 4, 6, 8}, q[4] = {0, 0, 0, 0}, mo[4];
        const float *blk[2] = {a, b};
        inkling_mix(blk, 2, a, q, 1, mo, 4);
        printf("M2 primitives: sconv=3.0 clip=1.0 mix[0]=%.4f\n", mo[0]);
    }

    // M2 attention: relative bias + sink denominator
    {
        float q[2] = {1, 0}, k[2] = {1, 0}, v[2] = {3, 4}, r[1] = {1}, bias[1] = {2}, sink[1] = {0.5f};
        float sc[1], out[2];
        inkling_attn(out, q, k, v, 2, 0, 0, bias, r, sink, 1, 1, sc);
        float s = 1.0f / sqrtf(2.0f) + 2.0f - 0.5f;
        float w = expf(s) / (1.0f + expf(s));
        printf("M2 attention: score %.4f w %.4f out %.4f %.4f (expect %.4f %.4f)\n", s, w,
               out[0], out[1], w * 3, w * 4);
        if (fabsf(out[0] - w * 3) > 1e-5f || fabsf(out[1] - w * 4) > 1e-5f) {
            fprintf(stderr, "attention mismatch\n"); return 1;
        }
    }

    {
        float *mx = malloc((size_t)m.dim * sizeof(float)), *mo = malloc((size_t)m.dim * sizeof(float));
        for (int i = 0; i < m.dim; i++) mx[i] = 0.001f * sinf(0.01f * (i + 1));
        inkling_moe(&m, 0, mx, mo);
        double mn = 1e30, mxv = -1e30, sum = 0;
        int ok = 1;
        for (int i = 0; i < m.dim; i++) {
            if (!isfinite(mo[i])) ok = 0;
            if (mo[i] < mn) mn = mo[i];
            if (mo[i] > mxv) mxv = mo[i];
            sum += mo[i];
        }
        printf("M2 MoE layer0: finite=%d min %.4f max %.4f mean %.4f\n", ok, mn, mxv, sum / m.dim);
        free(mx); free(mo);
    }

    {
        InklingState s2;
        if (!inkling_state_alloc(&m, 16, &s2)) { fprintf(stderr, "state2\n"); return 1; }
        float *h = calloc(m.dim, sizeof(float)), *o = calloc(m.dim, sizeof(float));
        inkling_embed_row(&m, 371, h);
        const InkTensor *en = inkling_find(&m, "embed_norm.weight");
        inkling_normw(h, h, (const float *)en->data, m.dim, m.eps);
        float **blocks = calloc(32, sizeof(float *));
        float *bmem = calloc((size_t)32 * m.dim, sizeof(float));
        int nb = 0, ba = 0;
        const InkTensor *fq = inkling_find(&m, "attn_res.mlp_queries.0");
        inkling_layer_swa(&m, &s2, 0, 0, 0, 0, h, NULL, (const float *)fq->data, blocks, &nb, &ba, bmem, 0, NULL, o);
        double sum = 0; int ok = 1; float mn = 1e30f, mx = -1e30f;
        for (int i = 0; i < m.dim; i++) { if (!isfinite(o[i])) ok = 0; sum += o[i]; if (o[i] < mn) mn = o[i]; if (o[i] > mx) mx = o[i]; }
        printf("M2 layer0: finite=%d min %.4f max %.4f mean %.4f blocks %d\n", ok, mn, mx, sum / m.dim, nb);
        const InkTensor *aq5 = inkling_find(&m, "attn_res.attn_queries.4");
        const InkTensor *fq5 = inkling_find(&m, "attn_res.mlp_queries.5");
        inkling_layer_swa(&m, &s2, 5, 5, 5, 0, h, (const float *)aq5->data, (const float *)fq5->data,
                          blocks, &nb, &ba, bmem, 1, NULL, o);
        ok = 1; sum = 0; mn = 1e30f; mx = -1e30f;
        for (int i = 0; i < m.dim; i++) { if (!isfinite(o[i])) ok = 0; sum += o[i]; if (o[i] < mn) mn = o[i]; if (o[i] > mx) mx = o[i]; }
        printf("M3 layer5 full: finite=%d min %.4f max %.4f mean %.4f\n", ok, mn, mx, sum / m.dim);
        free(h); free(o); free(blocks); free(bmem); inkling_state_free(&s2);
    }

    {
        int toks[5] = {371, 151451, 149879, 151194, 149884};
        InklingState fs;
        if (!inkling_state_alloc(&m, 64, &fs)) { fprintf(stderr, "forward state\n"); return 1; }
        InklingRuntime fr;
        if (!inkling_runtime_init(&m, &fr)) return 1;
        float *fl = malloc((size_t)m.unpadded * sizeof(float));
        for (int i = 0; i < 5; i++) inkling_forward_token(&m, &fs, &fr, toks[i], i, fl);
        int bi = 0;
        for (int i = 1; i < m.unpadded; i++) if (fl[i] > fl[bi]) bi = i;
        printf("INKLING full top-1 id %d logit %.6f (oracle 152717 / 27.307991)\n", bi, fl[bi]);
        FILE *lf = fopen("/dev/shm/inkling2/logits_ours.bin", "wb");
        if (lf) { fwrite(fl, sizeof(float), m.unpadded, lf); fclose(lf); }
        int topi[10];
        for (int k = 0; k < 10; k++) {
            int b = -1;
            for (int i = 0; i < m.unpadded; i++) {
                int seen = 0;
                for (int j = 0; j < k; j++) if (topi[j] == i) seen = 1;
                if (!seen && (b < 0 || fl[i] > fl[b])) b = i;
            }
            topi[k] = b;
            printf("  top %d id %d logit %.6f\n", k, b, fl[b]);
        }
        free(fl); inkling_runtime_free(&fr); inkling_state_free(&fs);
    }

    // Q8 packing: engine 4-row layout must reproduce a direct row-major dot product
    const InkTensor *wq = inkling_find(&m, "layers.0.attn.wq_du.weight");
    if (!wq) return 1;
    int rows = wq->dims[0], n = wq->dims[1];
    QuantizedTensor q8;
    if (!inkling_pack_2d(wq, &q8)) { fprintf(stderr, "pack q8 failed\n"); return 1; }
    float *x = malloc((size_t)n * sizeof(float)), *xs = malloc((size_t)(n / GS) * sizeof(float));
    int8_t *xq = malloc(n);
    float *o = malloc((size_t)rows * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = sinf(0.017f * (i + 1));
    quantize_vec(xq, xs, x, n);
    matmul_rows_q8(o, xq, xs, &q8, n, 0, rows);
    const int8_t *src = (const int8_t *)wq->data;
    const float *ss = (const float *)((const char *)wq->data + (size_t)rows * n);
    double md = 0;
    for (int i = 0; i < rows; i++) {
        double e = 0;
        for (int k = 0; k < n; k++) e += (double)xq[k] * xs[k / GS] * src[(size_t)i * n + k] * ss[(size_t)i * (n / GS) + k / GS];
        double d = fabs((double)o[i] - e);
        if (d > md) md = d;
    }
    printf("Q8 pack %dx%d max|diff| %.5f\n", rows, n, md);
    free(x); free(xs); free(xq); free(o);
    inkling_pack_free(&q8);

    // Q4 packing: one expert block (w13) against a direct row-major decode
    const InkTensor *w13 = inkling_find(&m, "layers.0.mlp.experts.w13");
    if (!w13) return 1;
    if (w13->dtype != 2) {
        printf("Q4 expert pack skipped (experts dtype %d)\n", w13->dtype);
        inkling_free(&m);
        return 0;
    }
    int er = w13->dims[1], ec = w13->dims[2];
    size_t stride = w13->bytes / w13->dims[0];
    InkTensor one = *w13;
    one.ndim = 2; one.dims[0] = er; one.dims[1] = ec; one.data = w13->data; one.bytes = stride;
    QuantizedTensor q4;
    if (!inkling_pack_2d(&one, &q4)) { fprintf(stderr, "pack q4 failed\n"); return 1; }
    n = ec;
    x = malloc((size_t)n * sizeof(float)); xs = malloc((size_t)(n / GS) * sizeof(float));
    xq = malloc(n); o = malloc((size_t)er * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = cosf(0.013f * (i + 1));
    quantize_vec(xq, xs, x, n);
    matmul_rows_q4(o, xq, xs, &q4, n, 0, er);
    const uint8_t *q4src = (const uint8_t *)one.data;
    const __fp16 *q4s = (const __fp16 *)(q4src + (size_t)er * ec / 2);
    md = 0;
    for (int i = 0; i < er; i++) {
        double e = 0;
        for (int g = 0; g < ec / 32; g++) {
            const uint8_t *b = q4src + (size_t)i * (ec / 2) + g * 16;
            const __fp16 *sm = q4s + ((size_t)i * (ec / 32) + g) * 2;
            for (int kk = 0; kk < 32; kk++) {
                int u = kk < 16 ? (b[kk] & 15) : (b[kk - 16] >> 4);
                double wv = u * (double)sm[0] + (double)sm[1];
                e += wv * (double)xq[g * 32 + kk] * xs[(g * 32 + kk) / GS];
            }
        }
        double d = fabs((double)o[i] - e);
        if (d > md) md = d;
    }
    printf("Q4 pack %dx%d max|diff| %.5f\n", er, ec, md);
    free(x); free(xs); free(xq); free(o);
    inkling_pack_free(&q4);

    inkling_free(&m);
    return 0;
}
