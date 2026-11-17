#define TESTING
#include "../src/runq.c"
static int amax(float* l, int V) { int a = 0; for (int i = 1; i < V; i++) if (l[i] > l[a]) a = i; return a; }
int main(int argc, char** argv) {
    int nth = choose_cpus(0);
    Transformer t; build_transformer(&t, argv[1], 2048); init_pool(&t, nth);
    int V = t.config.vocab_size; int toks[16]; for (int i = 0; i < 16; i++) toks[i] = 3000 + i * 101;
    for (int i = 0; i < 43; i++) forward(&t, 1000 + i, i);
    static float ref[16][128256];
    for (int B = 2; B <= 8; B++) {
        for (int i = 0; i < B; i++) memcpy(ref[i], forward(&t, toks[i], 43 + i), V * 4);
        float* L = forward_all(&t, toks, B, 43);
        int ok = 0, nan = 0; for (int i = 0; i < B; i++) { ok += amax(ref[i], V) == amax(L + (size_t)i * V, V); for (int j = 0; j < V; j++) if (L[(size_t)i*V+j] != L[(size_t)i*V+j]) { nan++; break; } }
        printf("B=%d argmax agree %d/%d, rows with NaN %d\n", B, ok, B, nan);
    }
    return 0;
}
