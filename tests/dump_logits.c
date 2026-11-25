#define TESTING
#include "../src/runq.c"
// dump5 <model> <tokenizer> <threads> <out> <text> [batch]: plain-encodes text, dumps logits of every position
int main(int argc, char** argv) {
    int nth = atoi(argv[3]); choose_cpus(nth);
    Transformer t; build_transformer(&t, argv[1], 2048); init_pool(&t, nth);
    Tokenizer tk; build_tokenizer(&tk, argv[2], t.config.vocab_size);
    static int ids[8192]; int n = 0; encode(&tk, argv[5], 0, 0, ids, &n);
    FILE* o = fopen(argv[4], "wb");
    for (int i = 0; i < n; i++) { printf("%d ", ids[i]); float* l = forward(&t, ids[i], i); fwrite(l, 4, t.config.vocab_size, o); }
    printf("\n"); fclose(o);
    if (argc > 6) { float* l = forward_batch(&t, ids, n, 0, NULL); FILE* o2 = fopen(argv[6], "wb"); fwrite(l, 4, t.config.vocab_size, o2); fclose(o2); }
    return 0;
}
