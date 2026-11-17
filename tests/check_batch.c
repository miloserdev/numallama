#define TESTING
#include "../src/runq.c"
static double maxdiff(float* a, float* b, int n, int* am, int* bm){ double d=0; *am=*bm=0; for(int i=0;i<n;i++){ double e=fabs(a[i]-b[i]); if(e>d)d=e; if(a[i]>a[*am])*am=i; if(b[i]>b[*bm])*bm=i;} return d; }
int main(int argc, char** argv) {
    int nth = atoi(argv[2]), N = atoi(argv[3]); choose_cpus(nth);
    Transformer t; build_transformer(&t, argv[1], 4096); init_pool(&t, nth);
    int V = t.config.vocab_size; int* toks = malloc(N*4); for (int i=0;i<N;i++) toks[i] = 1000 + (i*7919)%60000;
    float* l1 = malloc(V*4), *l2 = malloc(V*4), *d1 = malloc(V*4), *d2 = malloc(V*4);
    long t0 = time_in_ms(); float* l = NULL; for (int i=0;i<N;i++) l = forward(&t, toks[i], i); long t1 = time_in_ms();
    memcpy(l1, l, V*4); memcpy(d1, forward(&t, 4242, N), V*4);
    long t2 = time_in_ms(); l = forward_batch(&t, toks, N, 0); long t3 = time_in_ms();
    memcpy(l2, l, V*4); memcpy(d2, forward(&t, 4242, N), V*4);
    int a, b; double m = maxdiff(l1, l2, V, &a, &b); printf("prefill last-token logits: max|diff| %.4f argmax %d vs %d\n", m, a, b);
    m = maxdiff(d1, d2, V, &a, &b); printf("next decode step:          max|diff| %.4f argmax %d vs %d\n", m, a, b);
    printf("sequential %d tok: %.1f tok/s | batched: %.1f tok/s (x%.1f)\n", N, N*1000.0/(t1-t0), N*1000.0/(t3-t2), (double)(t1-t0)/(t3-t2));
    return 0;
}
