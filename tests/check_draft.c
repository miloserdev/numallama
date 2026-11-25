#define TESTING
#include "../src/runq.c"
static int amax(float* l, int V) { int a = 0; for (int i = 1; i < V; i++) if (l[i] > l[a]) a = i; return a; }
int main(int argc, char** argv) {
    int nth = choose_cpus(0);
    Transformer t; build_transformer(&t, argv[1], 2048); init_pool(&t, nth);
    static Transformer d; build_transformer(&d, argv[2], 2048); init_model_runtime(&d); use_model(&t);
    int V = t.config.vocab_size;
    int hist[2048], n = 0; for (int i = 0; i < 43; i++) hist[n++] = (i == 0) ? 128000 : 1000 + i * 37;
    // reference: pure greedy with single-token forward
    int ref[200]; { forward_batch(&t, hist, n, 0, NULL); int pos = n; float* l = forward(&t, hist[n - 1], n - 1); int tok = amax(l, V);
      for (int i = 0; i < 60; i++) { ref[i] = tok; l = forward(&t, tok, pos++); tok = amax(l, V); } }
    // speculative with draft (same loop shape as chat_reply, greedy)
    forward_batch(&t, hist, n - 1, 0, NULL); float own[128256]; memcpy(own, forward(&t, hist[n - 1], n - 1), V * 4);
    int out[200], no = 0, pending = -1; g_draft = &d; DRAFT_POS = 0;
    while (no < 60) {
        int next = pending >= 0 ? pending : amax(own, V); pending = -1;
        out[no++] = next;
        int seq[17]; seq[0] = next; int k = draft_model_tokens(hist, n, next, seq + 1, 6);
        int base = n; float* L = forward_all(&t, seq, k + 1, base); memcpy(hist + base, seq, (k + 1) * 4);
        int acc = 0; for (int i = 0; i < k; i++) { int y = amax(L + (size_t)i * V, V); if (y != seq[i + 1]) { pending = y; break; } acc++; out[no++] = y; }
        n = base + 1 + acc; DRAFT_POS = base + 1 + (acc < k - 1 ? acc : k - 1);
        if (pending < 0) memcpy(own, L + (size_t)k * V, V * 4);
        int bad = -1; for (int i = 0; i < no && i < 60; i++) if (out[i] != ref[i]) { bad = i; break; }
        printf("step base %d k %d acc %d -> %d tokens%s\n", base, k, acc, no, bad >= 0 ? " DIVERGED" : "");
        if (bad >= 0) { printf("first divergence at %d: got %d want %d\n", bad, out[bad], ref[bad]); break; }
    }
    return 0;
}
