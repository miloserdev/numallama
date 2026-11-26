#define TESTING
#include "../src/runq.c"
// int8 KV helpers against a float reference: quantization error must stay tiny, and the axpy must
// match a float accumulation of the dequantized values.
static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }
int main(void) {
    GS = 64;
    int n = 256;
    static int8_t qi[256], ki[256], vi[256];
    static float qs[4], ks[4], vs[4], qf[256], kf[256], vf[256];
    srand(42);
    for (int i = 0; i < n; i++) { qf[i] = frand(); kf[i] = frand() * 3.0f; vf[i] = frand(); }
    kv_store_row(qi, qs, qf, n);
    kv_store_row(ki, ks, kf, n);
    kv_store_row(vi, vs, vf, n);

    double ref = 0, err = 0;
    for (int i = 0; i < n; i++) ref += qf[i] * kf[i];
    float got = kv_dot_q8(qi, qs, ki, ks, n);
    err = fabs(got - ref) / (fabs(ref) + 1e-6);
    printf("dot: ref %.6f got %.6f rel.err %.4f%%\n", ref, got, err * 100);

    static float o1[256], o2[256];
    memset(o1, 0, sizeof o1);
    memset(o2, 0, sizeof o2);
    float a = 0.37f;
    kv_axpy_i8(o1, a, vi, vs, n);
    for (int i = 0; i < n; i++) o2[i] += a * vf[i];
    double me = 0, mr = 0;
    for (int i = 0; i < n; i++) { double d = fabs(o1[i] - o2[i]); if (d > me) me = d; if (fabs(o2[i]) > mr) mr = fabs(o2[i]); }
    printf("axpy: max abs err %.6f (max |ref| %.4f)\n", me, mr);
    printf("%s\n", err < 0.05 && me < 0.02 ? "OK" : "FAIL");
    return err < 0.05 && me < 0.02 ? 0 : 1;
}
