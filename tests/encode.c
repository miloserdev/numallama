#define TESTING
#include "../src/runq.c"
int main(int argc, char** argv) {   // enc4 <tokenizer.bin> <vocab_size> <arch>
    ARCH = atoi(argv[3]); Tokenizer tk; build_tokenizer(&tk, argv[1], atoi(argv[2]));
    static char line[65536]; static int enc[65536];
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = 0;
        int n = 0;
        encode(&tk, line, 0, 0, enc, &n);
        for (int i = 0; i < n; i++) printf("%d ", enc[i]);
        printf("\n");
    }
    return 0;
}
