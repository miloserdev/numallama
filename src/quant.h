static void quantize_vec(int8_t* q, float* s, const float* x, int n) {
    int num_groups = n / GS;
    for (int g = 0; g < num_groups; g++) {
        const float* xg = x + g * GS;
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) { float a = fabsf(xg[i]); wmax = a > wmax ? a : wmax; }
        float scale = wmax / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        s[g] = scale;
        for (int i = 0; i < GS; i++) { q[g * GS + i] = (int8_t) roundf(xg[i] * inv); }
    }
}

// out[i] = W[i,:] . x  for rows i in [r0, r1); x is quantized (xq, xs), n % GS == 0
// scalar access to the packed Q8 layout (row i, column k)
static inline int8_t q8_at(const QuantizedTensor* w, int n, int i, int k) {
    return w->q[(size_t)(i / 4) * 4 * n + (size_t)(k / GS) * GS * 4 + ((k % GS) / 4) * 16 + (i % 4) * 4 + k % 4];
}
static inline float q8_scale(const QuantizedTensor* w, int n, int i, int g) {
    return w->s[(size_t)(i / 4) * 4 * (n / GS) + g * 4 + i % 4];
}
static void matmul_rows_q8_scalar(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    for (int i = r0; i < r1; i++) {
        float val = 0.0f;
        for (int g = 0; g < n / GS; g++) {
            int32_t iv = 0;
            for (int k = g * GS; k < (g + 1) * GS; k++) iv += (int32_t)xq[k] * q8_at(w, n, i, k);
            val += (float)iv * q8_scale(w, n, i, g) * xs[g];
        }
        out[i] = val;
    }
}
static void matmul_rows_q8(float* out, const int8_t* xq, const float* xs,
                        const QuantizedTensor* w, int n, int r0, int r1) {
    const int ng = n / GS;
    int i = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (GS % 16 == 0 && r0 % 4 == 0) {
        const int8_t* qend = w->q + (size_t)r1 * n;
        for (; i + 4 <= r1; i += 4) {
            const int8_t* wb = w->q + (size_t)i * n;
            const float* sb = w->s + (size_t)i * ng;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int g = 0; g < ng; g++) {
                const int8_t* wg = wb + (size_t)g * GS * 4;
                if (wg + 4096 + 256 <= qend) { // one block is a single stream: keep it ahead, within the range
                    __builtin_prefetch(wg + 4096, 0, 0);
                    __builtin_prefetch(wg + 4096 + 64, 0, 0);
                    __builtin_prefetch(wg + 4096 + 128, 0, 0);
                    __builtin_prefetch(wg + 4096 + 192, 0, 0);
                }
                int32x4_t a = vdupq_n_s32(0), b = a;
                for (int k = 0; k < GS; k += 16) {
                    int8x16_t xv = vld1q_s8(xq + g * GS + k);
                    a = vdotq_laneq_s32(a, vld1q_s8(wg + k * 4), xv, 0);
                    b = vdotq_laneq_s32(b, vld1q_s8(wg + k * 4 + 16), xv, 1);
                    a = vdotq_laneq_s32(a, vld1q_s8(wg + k * 4 + 32), xv, 2);
                    b = vdotq_laneq_s32(b, vld1q_s8(wg + k * 4 + 48), xv, 3);
                }
                acc = vfmaq_f32(acc, vcvtq_f32_s32(vaddq_s32(a, b)), vmulq_n_f32(vld1q_f32(sb + g * 4), xs[g]));
            }
            vst1q_f32(out + i, acc);
        }
    }
#endif
    if (i < r1) matmul_rows_q8_scalar(out, xq, xs, w, n, i, r1);
}

// ----------------------------------------------------------------------------
// Q4 weights (q4_format 1): value = s * u + m per group of 32, u in 0..15, s and m in fp16.
// File: per row 16 bytes per group (byte k = u[k] | u[k+16] << 4), then per group [s, m].
// Memory (packed at load like Q8): blocks of 4 rows; per group 4 chunks of 16 bytes, chunk j =
// [row0 bytes 4j..4j+3 | row1 | row2 | row3]: its low nibbles are the 4 rows x k = 4j..4j+3, its high
// nibbles the same for k = 16+4j..; the scales of a block per group as [s0 s1 s2 s3 m0 m1 m2 m3].
// sum_k (s u_k + m) x_k = s * sum_k u_k x_k + m * sum_k x_k: the min only needs the group sums of the
// activations (once per token), so the inner loop is AND / shift / sdot by lane.
#define Q4_MAXG 2048 // groups of 32 per row (n <= 65536)
#if defined(__aarch64__)
typedef __fp16 half_t;
#else
typedef _Float16 half_t;
#endif
static inline const half_t* q4_scales(const QuantizedTensor* w, int n, int i) { // block of row i
    return (const half_t*)(w->s + (size_t)(i / 4) * 4 * (n / 32));
}
static void matmul_rows_q4_scalar(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    const uint8_t* q = (const uint8_t*)w->q;
    for (int i = r0; i < r1; i++) {
        const uint8_t* bq = q + (size_t)(i / 4) * 4 * (n / 2);
        const half_t* sc = q4_scales(w, n, i);
        float val = 0.0f;
        for (int g = 0; g < n / 32; g++) {
            int32_t su = 0, sx = 0;
            for (int kk = 0; kk < 32; kk++) {
                int j = (kk % 16) / 4, b = kk % 4;
                uint8_t byte = bq[(size_t)g * 64 + j * 16 + (i % 4) * 4 + b];
                int u = kk < 16 ? (byte & 15) : (byte >> 4);
                su += u * xq[g * 32 + kk];
                sx += xq[g * 32 + kk];
            }
            float xsc = xs[g * 32 / GS];
            val += (float)sc[g * 8 + i % 4] * xsc * su + (float)sc[g * 8 + 4 + i % 4] * xsc * sx;
        }
        out[i] = val;
    }
}
#if defined(__ARM_FEATURE_DOTPROD)
// group sums of the activations of one token, times their scales: the min terms
static inline void q4_xsums(float* xsum, const int8_t* xq, const float* xs, int n) {
    for (int g = 0; g < n / 32; g++)
        xsum[g] = (float)(vaddlvq_s8(vld1q_s8(xq + g * 32)) + vaddlvq_s8(vld1q_s8(xq + g * 32 + 16))) * xs[g * 32 / GS];
}
#endif
static void matmul_rows_q4(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    int i = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (r0 % 4 == 0 && GS % 32 == 0 && n / 32 <= Q4_MAXG) {
        const int ng = n / 32;
        float xsum[Q4_MAXG];
        q4_xsums(xsum, xq, xs, n);
        const uint8_t* qend = (const uint8_t*)w->q + (size_t)r1 * (n / 2);
        const uint8x16_t m15 = vdupq_n_u8(0x0F);
        for (; i + 4 <= r1; i += 4) {
            const uint8_t* wb = (const uint8_t*)w->q + (size_t)i * (n / 2);
            const half_t* sb = q4_scales(w, n, i);
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int g = 0; g < ng; g++) {
                const uint8_t* wg = wb + (size_t)g * 64;
                if (wg + 2048 + 64 <= qend) __builtin_prefetch(wg + 2048, 0, 0);
                const int8x16_t xa = vld1q_s8(xq + g * 32), xb = vld1q_s8(xq + g * 32 + 16);
                int32x4_t a = vdupq_n_s32(0), b = a;
                uint8x16_t v;
#define Q4_CHUNK(J) v = vld1q_u8(wg + J * 16); \
                a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(vandq_u8(v, m15)), xa, J); \
                b = vdotq_laneq_s32(b, vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), xb, J);
                Q4_CHUNK(0) Q4_CHUNK(1) Q4_CHUNK(2) Q4_CHUNK(3)
#undef Q4_CHUNK
                const float16x8_t sm = vld1q_f16((const float16_t*)(sb + g * 8));
                acc = vfmaq_f32(acc, vcvtq_f32_s32(vaddq_s32(a, b)), vmulq_n_f32(vcvt_f32_f16(vget_low_f16(sm)), xs[g * 32 / GS]));
                acc = vfmaq_n_f32(acc, vcvt_high_f32_f16(sm), xsum[g]);
            }
            vst1q_f32(out + i, acc);
        }
    }
#endif
    if (i < r1) matmul_rows_q4_scalar(out, xq, xs, w, n, i, r1);
}
#if defined(__ARM_FEATURE_DOTPROD)
/* Batched kernels on packed activations (the layout of Arm KleidiAI's qsi8d32p4x4 LHS): per quad of
 * 4 tokens and group of 32 values, 8 vectors of 16 bytes; vector j < 4 holds bytes 4j..4j+3 of the
 * 4 tokens (lane t = token t), vector 4 + j bytes 16+4j.. . One load then feeds 4 sdot by lane, one
 * per token, and the whole tile streams through one buffer instead of one row per token.
 * Per group the quad's 4 activation scales and 4 scaled group sums (for the Q4 min) are vectors. */
typedef struct { int8_t* q; float* s; float* m; } XPack;
static XPack xpack_buf(void) { // per thread, room for 8 tokens of the widest input, first touched by its owner
    static _Thread_local XPack p;
    if (!p.q) {
        p.q = aligned_alloc(64, (size_t)8 * Q4_MAXG * 32);
        p.s = aligned_alloc(64, sizeof(float) * 8 * Q4_MAXG);
        p.m = aligned_alloc(64, sizeof(float) * 8 * Q4_MAXG);
    }
    return p;
}
// pack tokens x[0..3] (rows of length n, scales per GS) into quad buffers at q / s / m
static void xpack_quad(int8_t* q, float* s, float* m, const int8_t* xq, const float* xs, int n) {
    const int ng = n / 32, ngx = n / GS;
    for (int g = 0; g < ng; g++) {
        int8x16_t a[4], b[4];
        float sc[4], sm[4];
        for (int t = 0; t < 4; t++) {
            a[t] = vld1q_s8(xq + (size_t)t * n + g * 32);
            b[t] = vld1q_s8(xq + (size_t)t * n + g * 32 + 16);
            sc[t] = xs[(size_t)t * ngx + g * 32 / GS];
            sm[t] = sc[t] * (float)(vaddlvq_s8(a[t]) + vaddlvq_s8(b[t]));
        }
        int8_t* d = q + (size_t)g * 128;
        for (int h = 0; h < 2; h++) { // 4 x 4 transpose of 32-bit words
            const int8x16_t* v = h ? b : a;
            const int32x4_t t0 = vtrn1q_s32(vreinterpretq_s32_s8(v[0]), vreinterpretq_s32_s8(v[1]));
            const int32x4_t t1 = vtrn2q_s32(vreinterpretq_s32_s8(v[0]), vreinterpretq_s32_s8(v[1]));
            const int32x4_t t2 = vtrn1q_s32(vreinterpretq_s32_s8(v[2]), vreinterpretq_s32_s8(v[3]));
            const int32x4_t t3 = vtrn2q_s32(vreinterpretq_s32_s8(v[2]), vreinterpretq_s32_s8(v[3]));
            vst1q_s64((int64_t*)(d + h * 64), vtrn1q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
            vst1q_s64((int64_t*)(d + h * 64 + 16), vtrn1q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
            vst1q_s64((int64_t*)(d + h * 64 + 32), vtrn2q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
            vst1q_s64((int64_t*)(d + h * 64 + 48), vtrn2q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
        }
        vst1q_f32(s + g * 4, vld1q_f32(sc));
        vst1q_f32(m + g * 4, vld1q_f32(sm));
    }
}
// 4 rows x 4 NQ tokens of Q4 on packed activations (quads at xp + q * ng * 128, scales at ps / pm + q * ng * 4)
static inline __attribute__((always_inline)) void q4_tile_packed(float* out, int ldo, const XPack* xp, int NQ,
                                                                 const uint8_t* wb, const half_t* sb, int ng) {
    const uint8x16_t m15 = vdupq_n_u8(0x0F);
    float32x4_t c[8];
    for (int u = 0; u < 4 * NQ; u++) c[u] = vdupq_n_f32(0.0f);
    for (int g = 0; g < ng; g++) {
        const uint8_t* wg = wb + (size_t)g * 64;
        const uint8x16_t v0 = vld1q_u8(wg), v1 = vld1q_u8(wg + 16), v2 = vld1q_u8(wg + 32), v3 = vld1q_u8(wg + 48);
        const int8x16_t W[8] = {
            vreinterpretq_s8_u8(vandq_u8(v0, m15)), vreinterpretq_s8_u8(vandq_u8(v1, m15)),
            vreinterpretq_s8_u8(vandq_u8(v2, m15)), vreinterpretq_s8_u8(vandq_u8(v3, m15)),
            vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)),
            vreinterpretq_s8_u8(vshrq_n_u8(v2, 4)), vreinterpretq_s8_u8(vshrq_n_u8(v3, 4))};
        const float16x8_t sm = vld1q_f16((const float16_t*)(sb + g * 8));
        const float32x4_t s4 = vcvt_f32_f16(vget_low_f16(sm)), m4 = vcvt_high_f32_f16(sm);
        for (int q = 0; q < NQ; q++) {
            const int8_t* x = xp->q + ((size_t)q * ng + g) * 128;
            int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
            for (int j = 0; j < 8; j++) {
                const int8x16_t X = vld1q_s8(x + j * 16);
                a0 = vdotq_laneq_s32(a0, W[j], X, 0);
                a1 = vdotq_laneq_s32(a1, W[j], X, 1);
                a2 = vdotq_laneq_s32(a2, W[j], X, 2);
                a3 = vdotq_laneq_s32(a3, W[j], X, 3);
            }
            const float32x4_t xs4 = vld1q_f32(xp->s + ((size_t)q * ng + g) * 4);
            const float32x4_t xm4 = vld1q_f32(xp->m + ((size_t)q * ng + g) * 4);
            float32x4_t* cq = c + 4 * q;
            cq[0] = vfmaq_laneq_f32(vfmaq_f32(cq[0], vcvtq_f32_s32(a0), vmulq_laneq_f32(s4, xs4, 0)), m4, xm4, 0);
            cq[1] = vfmaq_laneq_f32(vfmaq_f32(cq[1], vcvtq_f32_s32(a1), vmulq_laneq_f32(s4, xs4, 1)), m4, xm4, 1);
            cq[2] = vfmaq_laneq_f32(vfmaq_f32(cq[2], vcvtq_f32_s32(a2), vmulq_laneq_f32(s4, xs4, 2)), m4, xm4, 2);
            cq[3] = vfmaq_laneq_f32(vfmaq_f32(cq[3], vcvtq_f32_s32(a3), vmulq_laneq_f32(s4, xs4, 3)), m4, xm4, 3);
        }
    }
    for (int u = 0; u < 4 * NQ; u++) vst1q_f32(out + (size_t)u * ldo, c[u]);
}
#endif
static void matmul_rows_batch_q4(float* out, int ldo, const int8_t* xq, const float* xs, int B,
                                 const QuantizedTensor* w, int n, int r0, int r1) {
    const int ngx = n / GS;
    int iend = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (r0 % 4 == 0 && GS % 32 == 0 && n / 32 <= Q4_MAXG) {
        const int ng = n / 32;
        const XPack xp = xpack_buf();
        const XPack xp2 = {xp.q + (size_t)ng * 128, xp.s + (size_t)ng * 4, xp.m + (size_t)ng * 4};
        iend = r0 + (r1 - r0) / 4 * 4;
        int t = 0;
        for (; t + 8 <= B; t += 8) { // token tiles outside: the thread's weights stay in L2
            xpack_quad(xp.q, xp.s, xp.m, xq + (size_t)t * n, xs + (size_t)t * ngx, n);
            xpack_quad(xp2.q, xp2.s, xp2.m, xq + (size_t)(t + 4) * n, xs + (size_t)(t + 4) * ngx, n);
            for (int i = r0; i < iend; i += 4)
                q4_tile_packed(out + (size_t)t * ldo + i, ldo, &xp, 2, (const uint8_t*)w->q + (size_t)i * (n / 2), q4_scales(w, n, i), ng);
        }
        for (; t + 4 <= B; t += 4) {
            xpack_quad(xp.q, xp.s, xp.m, xq + (size_t)t * n, xs + (size_t)t * ngx, n);
            for (int i = r0; i < iend; i += 4)
                q4_tile_packed(out + (size_t)t * ldo + i, ldo, &xp, 1, (const uint8_t*)w->q + (size_t)i * (n / 2), q4_scales(w, n, i), ng);
        }
        for (; t < B; t++) matmul_rows_q4(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ngx, w, n, r0, iend);
    }
#endif
    if (iend < r1)
        for (int t = 0; t < B; t++) matmul_rows_q4_scalar(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ngx, w, n, iend, r1);
}

#if defined(__ARM_FEATURE_DOTPROD)
// Q8: pack tokens x[0..3] per 16 values (4 vectors: bytes 4j..4j+3 of the 4 tokens) and their GS scales
static void xpack_quad8(int8_t* q, float* s, const int8_t* xq, const float* xs, int n) {
    const int ngx = n / GS;
    for (int k = 0; k < n; k += 16) {
        int32x4_t v[4];
        for (int t = 0; t < 4; t++) v[t] = vreinterpretq_s32_s8(vld1q_s8(xq + (size_t)t * n + k));
        const int32x4_t t0 = vtrn1q_s32(v[0], v[1]), t1 = vtrn2q_s32(v[0], v[1]);
        const int32x4_t t2 = vtrn1q_s32(v[2], v[3]), t3 = vtrn2q_s32(v[2], v[3]);
        int8_t* d = q + (size_t)k * 4;
        vst1q_s64((int64_t*)d, vtrn1q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
        vst1q_s64((int64_t*)(d + 16), vtrn1q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
        vst1q_s64((int64_t*)(d + 32), vtrn2q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
        vst1q_s64((int64_t*)(d + 48), vtrn2q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
    }
    for (int g = 0; g < ngx; g++)
        for (int t = 0; t < 4; t++) s[g * 4 + t] = xs[(size_t)t * ngx + g];
}
// 4 rows x 4 NQ tokens of Q8 (packed weight layout) on packed activations (quad q at xp + q * n * 4)
static inline __attribute__((always_inline)) void q8_tile_packed(float* out, int ldo, const XPack* xp, int NQ,
                                                                 const int8_t* wb, const float* sb, int n) {
    const int ng = n / GS;
    float32x4_t c[8];
    for (int u = 0; u < 4 * NQ; u++) c[u] = vdupq_n_f32(0.0f);
    for (int g = 0; g < ng; g++) {
        int32x4_t a[8];
        for (int u = 0; u < 4 * NQ; u++) a[u] = vdupq_n_s32(0);
        for (int k = g * GS; k < (g + 1) * GS; k += 16) {
            const int8_t* wk = wb + (size_t)k * 4;
            const int8x16_t W0 = vld1q_s8(wk), W1 = vld1q_s8(wk + 16), W2 = vld1q_s8(wk + 32), W3 = vld1q_s8(wk + 48);
            for (int q = 0; q < NQ; q++) {
                const int8_t* x = xp->q + (size_t)q * n * 4 + (size_t)k * 4;
                int32x4_t* aq = a + 4 * q;
#define Q8_STEP(J, WJ) { const int8x16_t X = vld1q_s8(x + J * 16); \
                aq[0] = vdotq_laneq_s32(aq[0], WJ, X, 0); aq[1] = vdotq_laneq_s32(aq[1], WJ, X, 1); \
                aq[2] = vdotq_laneq_s32(aq[2], WJ, X, 2); aq[3] = vdotq_laneq_s32(aq[3], WJ, X, 3); }
                Q8_STEP(0, W0) Q8_STEP(1, W1) Q8_STEP(2, W2) Q8_STEP(3, W3)
#undef Q8_STEP
            }
        }
        const float32x4_t ws = vld1q_f32(sb + g * 4);
        for (int q = 0; q < NQ; q++) {
            const float32x4_t xs4 = vld1q_f32(xp->s + ((size_t)q * ng + g) * 4);
            float32x4_t* cq = c + 4 * q;
            cq[0] = vfmaq_f32(cq[0], vcvtq_f32_s32(a[4 * q + 0]), vmulq_laneq_f32(ws, xs4, 0));
            cq[1] = vfmaq_f32(cq[1], vcvtq_f32_s32(a[4 * q + 1]), vmulq_laneq_f32(ws, xs4, 1));
            cq[2] = vfmaq_f32(cq[2], vcvtq_f32_s32(a[4 * q + 2]), vmulq_laneq_f32(ws, xs4, 2));
            cq[3] = vfmaq_f32(cq[3], vcvtq_f32_s32(a[4 * q + 3]), vmulq_laneq_f32(ws, xs4, 3));
        }
    }
    for (int u = 0; u < 4 * NQ; u++) vst1q_f32(out + (size_t)u * ldo, c[u]);
}
#endif
static void matmul_rows_batch_q8(float* out, int ldo, const int8_t* xq, const float* xs, int B,
                                 const QuantizedTensor* w, int n, int r0, int r1) {
    const int ng = n / GS;
    int iend = r0;
#if defined(__ARM_FEATURE_DOTPROD)
    if (GS % 16 == 0 && r0 % 4 == 0 && n <= Q4_MAXG * 32 && ng <= Q4_MAXG) {
        const XPack xp = xpack_buf();
        const XPack xp2 = {xp.q + (size_t)n * 4, xp.s + (size_t)ng * 4, NULL};
        iend = r0 + (r1 - r0) / 4 * 4;
        // token tiles outside, the thread's rows inside: its weights stay in L2, the tile's inputs in L1
        int t = 0;
        for (; t + 8 <= B; t += 8) {
            xpack_quad8(xp.q, xp.s, xq + (size_t)t * n, xs + (size_t)t * ng, n);
            xpack_quad8(xp2.q, xp2.s, xq + (size_t)(t + 4) * n, xs + (size_t)(t + 4) * ng, n);
            for (int i = r0; i < iend; i += 4)
                q8_tile_packed(out + (size_t)t * ldo + i, ldo, &xp, 2, w->q + (size_t)i * n, w->s + (size_t)i * ng, n);
        }
        for (; t + 4 <= B; t += 4) {
            xpack_quad8(xp.q, xp.s, xq + (size_t)t * n, xs + (size_t)t * ng, n);
            for (int i = r0; i < iend; i += 4)
                q8_tile_packed(out + (size_t)t * ldo + i, ldo, &xp, 1, w->q + (size_t)i * n, w->s + (size_t)i * ng, n);
        }
        for (; t < B; t++) matmul_rows_q8(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ng, w, n, r0, iend);
    }
#endif
    if (iend < r1)
        for (int t = 0; t < B; t++) matmul_rows_q8_scalar(out + (size_t)t * ldo, xq + (size_t)t * n, xs + (size_t)t * ng, w, n, iend, r1);
}

static void matmul_rows(float* out, const int8_t* xq, const float* xs, const QuantizedTensor* w, int n, int r0, int r1) {
    if (w->bits == 4) matmul_rows_q4(out, xq, xs, w, n, r0, r1);
    else matmul_rows_q8(out, xq, xs, w, n, r0, r1);
}

