# numallama — NUMA-aware CPU inference for Llama 3.x / Qwen2.5 (AArch64 + NEON dotprod first, any Linux CPU works)
CC      ?= gcc
CFLAGS  ?= -Ofast -mcpu=native -pthread -Wall
LDLIBS  = -lm
TESTS   = tests/dump_logits tests/encode tests/check_batch tests/check_multi tests/check_draft tests/check_kv

.PHONY: all prof tests clean

all: runq

runq: src/runq.c src/*.h
	$(CC) $(CFLAGS) -o $@ src/runq.c $(LDLIBS)

# per-phase timing report at exit (ms per token per phase, per NUMA node)
prof: src/runq.c src/*.h
	$(CC) $(CFLAGS) -DPROFILE -o runq_prof src/runq.c $(LDLIBS)

tests: $(TESTS)

tests/%: tests/%.c src/runq.c src/*.h
	$(CC) $(CFLAGS) -Wno-unused-result -o $@ $< $(LDLIBS)

clean:
	rm -f runq runq_prof $(TESTS)
