# numallama — NUMA-aware CPU inference for Llama 3.x / Qwen2.5 (AArch64 + NEON dotprod first, any Linux CPU works)
CC      ?= gcc
CFLAGS  ?= -Ofast -mcpu=native -pthread -Wall
LDLIBS  = -lm
TESTS   = tests/dump_logits tests/encode tests/check_batch tests/check_multi tests/check_draft tests/check_kv tests/check_seqs tests/check_inkling

# Low-end boards: no -mcpu=native (build on the target) and no ARMv8.2 SDOT. The kernels then
# use the portable NEON/scalar path; the optimized binary above keeps SDOT.
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
PORTABLE_CFLAGS ?= -O2 -march=armv8-a -mtune=generic -pthread -Wall
else ifeq ($(UNAME_M),x86_64)
PORTABLE_CFLAGS ?= -O2 -march=x86-64 -mtune=generic -pthread -Wall
else
PORTABLE_CFLAGS ?= -O2 -pthread -Wall
endif

.PHONY: all prof tests portable clean

all: runq

runq: src/runq.c src/*.h
	$(CC) $(CFLAGS) -o $@ src/runq.c $(LDLIBS)

# per-phase timing report at exit (ms per token per phase, per NUMA node)
prof: src/runq.c src/*.h
	$(CC) $(CFLAGS) -DPROFILE -o runq_prof src/runq.c $(LDLIBS)

# portable binary for ARMv8.0 / weak boards: no SDOT, no -mcpu=native
portable: runq_portable

runq_portable: src/runq.c src/*.h
	$(CC) $(PORTABLE_CFLAGS) -o $@ src/runq.c $(LDLIBS)

tests: $(TESTS)

tests/%: tests/%.c src/runq.c src/*.h
	$(CC) $(CFLAGS) -Wno-unused-result -o $@ $< $(LDLIBS)

clean:
	rm -f runq runq_prof runq_portable $(TESTS)
