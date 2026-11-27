#define TESTING
#include "../src/runq.c"
// continuous-batching core: one batched step over independent sessions must reproduce the logits
// of decoding each session on its own cache.
static int amax(float* l, int V) { int a = 0; for (int i = 1; i < V; i++) if (l[i] > l[a]) a = i; return a; }
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model [threads]\n", argv[0]); return 1; }
    int nth = argc > 2 ? atoi(argv[2]) : choose_cpus(0);
    Transformer t;
    build_transformer(&t, argv[1], 2048);
    init_pool(&t, nth);
    Config* p = &t.config;
    int V = p->vocab_size;
    enum { K = 4 };
    static const int lens[K] = {31, 47, 63, 89};
    KVCache kv[K];
    float* ref[K] = {0};
    float* out[K] = {0};
    int poss[K] = {0}, toks[K] = {0};
    for (int i = 0; i < K; i++) {
        kv_cache_alloc(p, t.state.kv_type, &kv[i]);
        kv_attach(&t.state, &kv[i]);
        int ids[128];
        for (int j = 0; j < lens[i]; j++) ids[j] = 1000 + j * 37 + i * 101;
        forward_batch(&t, ids, lens[i], 0, NULL);
        poss[i] = lens[i];
        toks[i] = 5000 + i * 17;
        kv_attach(&t.state, &kv[i]);
        ref[i] = malloc((size_t)V * sizeof(float));
        out[i] = malloc((size_t)V * sizeof(float));
        memcpy(ref[i], forward(&t, toks[i], poss[i]), (size_t)V * sizeof(float));
    }
    printf("kv %s\n", kv_type_name(t.state.kv_type));
    int fail = 0;
    for (int B = 2; B <= K; B++) {
        SrvTok ts[K];
        for (int b = 0; b < B; b++) { ts[b].kv = &kv[b]; ts[b].tok = toks[b]; ts[b].pos = poss[b]; ts[b].logits = out[b]; }
        if (!forward_sessions(&t, ts, B)) { printf("B=%d forward_sessions failed\n", B); fail = 1; break; }
        int ok = 0;
        double md = 0;
        for (int b = 0; b < B; b++) {
            double d = 0;
            for (int i = 0; i < V; i++) { double e = fabs((double)out[b][i] - ref[b][i]); if (e > d) d = e; }
            if (d > md) md = d;
            ok += amax(out[b], V) == amax(ref[b], V);
        }
        printf("B=%d argmax agree %d/%d, max|diff| %.4f\n", B, ok, B, md);
        if (ok != B) fail = 1;
    }
    for (int i = 0; i < K; i++) { free(ref[i]); free(out[i]); kv_cache_free(&kv[i]); }
    free_transformer(&t);
    printf("%s\n", fail ? "FAIL" : "OK");
    return fail;
}
