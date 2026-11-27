/* Thread placement. On a shared machine the scheduler initially stacks spin-waiting threads on
 * busy cores and needs seconds to spread them, which stalls every barrier. So at start-up we
 * pick the most idle CPUs (sampled from /proc/stat) among those allowed by taskset/numactl and
 * move thread i onto the i-th of them.
 *   RUNQ_PIN unset: place, then release the pin to the CPUs of that NUMA node
 *   RUNQ_PIN=1:     keep threads pinned
 *   RUNQ_PIN=0:     leave placement entirely to the scheduler */
static int* P_cpus = NULL; // chosen CPUs, one per thread

#if defined(__linux__)
static int read_cpu_idle(long long* idle, int max) {
    FILE* f = fopen("/proc/stat", "r");
    if (!f) return 0;
    char line[512];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        int cpu;
        long long v[8] = {0};
        if (sscanf(line, "cpu%d %lld %lld %lld %lld %lld %lld %lld %lld", &cpu, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) >= 5 &&
            cpu >= 0 && cpu < max) {
            idle[cpu] = v[3] + v[4]; // idle + iowait
            if (cpu + 1 > n) n = cpu + 1;
        }
    }
    fclose(f);
    return n;
}

static int NODES_REQ = 0; // -N: NUMA nodes to use, 0 = all usable
#if defined(__linux__)
static cpu_set_t P_node_mask[MAX_NODES]; // released threads may move only inside their node
#endif
/* Choose CPUs on one or more NUMA nodes (most idle first). nth <= 0: automatic thread count
 * (idle CPUs of each node minus a few, at most 96 per node). Sets NN, NODE_ID, NODE_T0, P_cpus.
 * Returns the number of threads. */
__attribute__((unused)) static int choose_cpus(int nth) {
    const char* mode = getenv("RUNQ_PIN");
    if (mode && !strcmp(mode, "0")) { NN = 1; return nth > 0 ? nth : 1; }
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return nth > 0 ? nth : 1;
    static long long i0[CPU_SETSIZE], i1[CPU_SETSIZE];
    read_cpu_idle(i0, CPU_SETSIZE);
    struct timespec ts = {0, 100 * 1000 * 1000};
    nanosleep(&ts, NULL);
    int n = read_cpu_idle(i1, CPU_SETSIZE);
    static int node_of[CPU_SETSIZE];
    for (int c = 0; c < CPU_SETSIZE; c++) node_of[c] = 0;
    int n_nodes = 1;
    for (int node = 0; node < 64; node++) {
        char path[96];
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        char list[4096];
        if (fgets(list, sizeof list, f)) {
            for (char* tok = strtok(list, ",\n"); tok; tok = strtok(NULL, ",\n")) {
                int lo, hi;
                if (sscanf(tok, "%d-%d", &lo, &hi) == 2) { for (int c = lo; c <= hi && c < CPU_SETSIZE; c++) node_of[c] = node; }
                else if (sscanf(tok, "%d", &lo) == 1 && lo < CPU_SETSIZE) node_of[lo] = node;
            }
        }
        fclose(f);
        if (node + 1 > n_nodes) n_nodes = node + 1;
    }
    if (n_nodes > 64) n_nodes = 64;
    long long period = 1;
    for (int c = 0; c < n; c++) if (i1[c] - i0[c] > period) period = i1[c] - i0[c]; // ticks of a fully idle CPU
    int idle_cnt[64] = {0};
    for (int c = 0; c < n; c++)
        if (CPU_ISSET(c, &allowed) && (i1[c] - i0[c]) * 10 >= period * 9) idle_cnt[node_of[c]]++;
    // nodes by idle CPUs, most idle first
    int order[64];
    for (int k = 0; k < n_nodes; k++) order[k] = k;
    for (int i = 1; i < n_nodes; i++) {
        int v = order[i], j = i - 1;
        while (j >= 0 && idle_cnt[order[j]] < idle_cnt[v]) { order[j + 1] = order[j]; j--; }
        order[j + 1] = v;
    }
    int want = NODES_REQ > 0 ? NODES_REQ : n_nodes;
    if (want > MAX_NODES) want = MAX_NODES;
    if (nth == 1) want = 1;
    NN = 0;
    for (int k = 0; k < n_nodes && NN < want; k++)
        if (idle_cnt[order[k]] >= (NODES_REQ > 0 ? 1 : 16) || NN == 0) NODE_ID[NN++] = order[k];
    int threads[MAX_NODES], total = 0;
    for (int k = 0; k < NN; k++) {
        if (nth <= 0) {
            // leave headroom on big nodes, but do not collapse a small board to a single worker
            int idle = idle_cnt[NODE_ID[k]];
            threads[k] = idle > 8 ? idle - 4 : idle > 1 ? idle - 1 : 1;
            if (threads[k] > 96) threads[k] = 96;
        } else {
            threads[k] = nth / NN + (k < nth % NN);
        }
        if (threads[k] > idle_cnt[NODE_ID[k]])
            fprintf(stderr, "[warning: only %d idle CPUs on NUMA node %d for %d threads; expect slowdowns]\n", idle_cnt[NODE_ID[k]], NODE_ID[k], threads[k]);
        NODE_T0[k] = total;
        total += threads[k];
    }
    NODE_T0[NN] = total;
    P_cpus = malloc(total * sizeof(int));
    for (int k = 0; k < NN; k++) {
        int cand[CPU_SETSIZE], nc = 0;
        for (int c = 0; c < n; c++)
            if (CPU_ISSET(c, &allowed) && node_of[c] == NODE_ID[k]) cand[nc++] = c;
        // most idle first; ties keep CPU order so threads land on neighbouring cores
        for (int i = 1; i < nc; i++) {
            int c = cand[i], j = i - 1;
            while (j >= 0 && (i1[cand[j]] - i0[cand[j]]) < (i1[c] - i0[c])) { cand[j + 1] = cand[j]; j--; }
            cand[j + 1] = c;
        }
        CPU_ZERO(&P_node_mask[k]);
        for (int i = 0; i < nc; i++) CPU_SET(cand[i], &P_node_mask[k]);
        for (int t = NODE_T0[k]; t < NODE_T0[k + 1]; t++) P_cpus[t] = nc ? cand[(t - NODE_T0[k]) % nc] : 0;
    }
    return total;
}
#endif

static void pin_thread(int tid) {
#if defined(__linux__)
    if (!P_cpus) return;
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(P_cpus[tid], &one);
    pthread_setaffinity_np(pthread_self(), sizeof(one), &one); // migrates the thread there now
    const char* mode = getenv("RUNQ_PIN");
    if (!(mode && !strcmp(mode, "1"))) {
        int k = 0;
        while (k < NN - 1 && tid >= NODE_T0[k + 1]) k++;
        pthread_setaffinity_np(pthread_self(), sizeof(P_node_mask[k]), &P_node_mask[k]); // soft: release the pin within the node
    }
#else
    (void)tid;
#endif
}

static void forward_batch_thread(int tid);
static void forward_multi_thread(int tid);
static void init_batch(Transformer* t);

/* Idle workers spin for a short while (tokens of one reply follow each other within ~1 ms) and
 * then sleep on a futex, so an idle chat does not keep a hundred cores busy. */
#include <linux/futex.h>
#include <sys/syscall.h>
#define IDLE_SPIN_NS 2000000LL
static inline long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static int wait_for_work(int seen) {
    int g;
    long long t0 = 0;
    for (unsigned spins = 0;; spins++) {
        if ((g = atomic_load_explicit(&P.go, memory_order_acquire)) != seen) return g;
        cpu_relax();
        if ((spins & 1023) == 1023) {
            long long t = now_ns();
            if (t0 == 0) t0 = t;
            else if (t - t0 > IDLE_SPIN_NS) {
                atomic_fetch_add(&P.sleepers, 1); // seq_cst: ordered against the waker's increment of `go`
                if (atomic_load(&P.go) == seen) syscall(SYS_futex, &P.go, FUTEX_WAIT_PRIVATE, seen, NULL, NULL, 0);
                atomic_fetch_sub(&P.sleepers, 1);
                t0 = 0;
            }
        }
    }
}
static inline void wake_workers(void) {
    atomic_fetch_add(&P.go, 1); // seq_cst
    if (atomic_load(&P.sleepers) > 0) syscall(SYS_futex, &P.go, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL, NULL, 0);
}

static void sample_thread(int tid);
static void copy_thread(int tid);
static void commit_thread(int tid);
// RUNQ_BARBENCH=1: measure the synchronization primitives alone (us per barrier)
static double now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3; }
static void barrier_bench_thread(int tid) {
    ThreadBuf* tb = &P.tb[tid];
    const int R = 20000;
    for (int i = 0; i < 2000; i++) barrier(tb); // warm-up
    double t0 = now_us();
    for (int i = 0; i < R; i++) barrier(tb);
    double t1 = now_us();
    for (int i = 0; i < R; i++) node_barrier(tb);
    barrier(tb);
    double t2 = now_us();
    if (tid == 0) fprintf(stderr, "[barrier bench] %d threads: global %.2f us, node %.2f us\n", P.nth, (t1 - t0) / R, (t2 - t1) / R);
}
static void* pool_worker(void* arg) {
    int tid = (int)(intptr_t)arg;
    pin_thread(tid);
    int seen = 0;
    for (;;) {
        seen = wait_for_work(seen);
        if (atomic_load_explicit(&P.quit, memory_order_acquire)) break;
        if (P.mode == 4) sample_thread(tid);
        else if (P.mode == 5) copy_thread(tid);
        else if (P.mode == 6) commit_thread(tid);
        else if (P.mode == 3) barrier_bench_thread(tid);
        else if (P.mode == 1) forward_batch_thread(tid);
        else if (P.mode == 2) forward_multi_thread(tid);
        else forward_thread(tid);
    }
    return NULL;
}

/* Model-specific runtime tables (RoPE frequencies, attention partials). Must be called right after
 * build_transformer() of that model (the header globals still hold its settings) and after the
 * pool layout (NN, NODE_T0, P.nth) is known. */
static void init_model_runtime(Transformer* t) {
    Config* p = &t->config;
    if (p->n_layers > P_max_layers) P_max_layers = p->n_layers; // counters are reset for the deepest model
    int head_size = p->head_dim;
    ROPE_THETA = t->g.rope_theta; ROPE_FACTOR = t->g.rope_factor; ROPE_LOW_FREQ = t->g.rope_low;
    ROPE_HIGH_FREQ = t->g.rope_high; ROPE_ORIG_CTX = t->g.rope_orig;
    {   // context chunks per kv head: enough items for the threads of the smallest node
        int min_t = P.nth, min_g = p->n_kv_heads;
        for (int k = 0; k < NN; k++) {
            int g0, g1;
            node_groups(p->n_kv_heads, k, &g0, &g1);
            if (NODE_T0[k + 1] - NODE_T0[k] < min_t) min_t = NODE_T0[k + 1] - NODE_T0[k];
            if (g1 - g0 < min_g) min_g = g1 - g0;
        }
        if (min_g < 1) min_g = 1;
        t->C = min_t / min_g > 1 ? min_t / min_g : 1;
    }
    t->part = calloc((size_t)p->n_heads * t->C * (head_size + 2), sizeof(float));
    t->rope_freq = malloc((head_size / 2) * sizeof(double));
    const int rdim = p->rot_dim > 0 ? p->rot_dim : head_size; // partial RoPE: frequencies over the rotated dims
    for (int j = 0; j < head_size / 2; j++) {
        double freq = j < rdim / 2 ? 1.0 / pow(ROPE_THETA, (2.0 * j) / rdim) : 0.0;
        if (ROPE_FACTOR > 0.0f) { // llama3 rope scaling (Llama 3.1 / 3.2)
            double low_wavelen = ROPE_ORIG_CTX / ROPE_LOW_FREQ;
            double high_wavelen = ROPE_ORIG_CTX / ROPE_HIGH_FREQ;
            double wavelen = 2.0 * M_PI / freq;
            if (wavelen > low_wavelen) {
                freq = freq / ROPE_FACTOR;
            } else if (wavelen >= high_wavelen) {
                double smooth = (ROPE_ORIG_CTX / wavelen - ROPE_LOW_FREQ) / (ROPE_HIGH_FREQ - ROPE_LOW_FREQ);
                freq = (1.0 - smooth) * freq / ROPE_FACTOR + smooth * freq;
            }
        }
        t->rope_freq[j] = freq;
    }
}

// make t the current model: restore its header settings and runtime tables
static Transformer* P_cur = NULL;
static void use_model(Transformer* t) {
    if (P_cur == t) return;
    GS = t->g.gs; ARCH = t->g.arch; QKV_BIAS = t->g.qkv_bias; SPECIAL_START = t->g.special_start;
    BOS_ID = t->g.bos; EOS_ID = t->g.eos; NORM_EPS = t->g.norm_eps;
    ROPE_THETA = t->g.rope_theta; ROPE_FACTOR = t->g.rope_factor; ROPE_LOW_FREQ = t->g.rope_low;
    ROPE_HIGH_FREQ = t->g.rope_high; ROPE_ORIG_CTX = t->g.rope_orig;
    P.rope_freq = t->rope_freq;
    P.part = t->part;
    P.C = t->C;
    P_cur = t;
}

void init_pool(Transformer* t, int nth) {
    Config* p = &t->config;
    int head_size = p->head_dim;
    if (nth < 1) nth = 1;
    if (p->dim % GS != 0 || p->hidden_dim % GS != 0) {
        fprintf(stderr, "group size %d must divide dim %d and hidden_dim %d; re-export the model\n",
                GS, p->dim, p->hidden_dim);
        exit(EXIT_FAILURE);
    }
    P.nth = nth;
    P.t = t;
    if (NN == 1 || NODE_T0[NN] != nth) { NN = 1; NODE_T0[0] = 0; NODE_T0[1] = nth; } // no (matching) placement
    if (!p->glm && p->n_heads / p->n_kv_heads > KV_MUL_MAX) {
        fprintf(stderr, "n_heads / n_kv_heads > %d is not supported\n", KV_MUL_MAX);
        exit(EXIT_FAILURE);
    }
    if (getenv("GLM_LAYERS")) GLM_LAYERS = atoi(getenv("GLM_LAYERS")); // debug: run only the first n layers
    if (getenv("GLM_DEBUG_SEL")) GLM_DEBUG_SEL = 1;
    init_model_runtime(t);
    use_model(t);
    int qmax = p->dim > p->hidden_dim ? p->dim : p->hidden_dim;
    if (p->q_dim > qmax) qmax = p->q_dim;
    if (p->n_experts > 0 && p->n_active * p->moe_hidden > qmax) qmax = p->n_active * p->moe_hidden;
    P_qmax = qmax;
    if (!p->glm && (p->n_experts > 256 || (p->n_experts > 0 && SPEC_MAX_TOK * p->n_experts > 4096))) { fprintf(stderr, "at most 256 experts are supported\n"); exit(EXIT_FAILURE); }
    if (p->n_active > MOE_MAX_ACTIVE) { fprintf(stderr, "at most %d active experts are supported\n", MOE_MAX_ACTIVE); exit(EXIT_FAILURE); }
    P.tb = aligned_alloc(128, nth * sizeof(ThreadBuf));
    memset(P.tb, 0, nth * sizeof(ThreadBuf));
    for (int i = 0; i < nth; i++) {
        P.tb[i].xb = calloc(p->q_dim > p->dim ? p->q_dim : p->dim, sizeof(float));
        P.tb[i].q = calloc(qmax, sizeof(int8_t));
        P.tb[i].s = calloc(qmax / GS + 1, sizeof(float));
        P.tb[i].rope_c = calloc(head_size / 2, sizeof(float));
        P.tb[i].rope_s = calloc(head_size / 2, sizeof(float));
        int k = 0;
        while (k < NN - 1 && i >= NODE_T0[k + 1]) k++;
        P.tb[i].node = k;
        P.tb[i].ltid = i - NODE_T0[k];
        P.tb[i].nthn = NODE_T0[k + 1] - NODE_T0[k];
    }
    if (head_size % GS != 0) {
        fprintf(stderr, "head_size %d must be a multiple of the group size %d\n", head_size, GS);
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < NN; k++) { // node-local buffers (bound to the node before first touch)
        int nthn = NODE_T0[k + 1] - NODE_T0[k];
        size_t qb = ((size_t)qmax * SPEC_MAX_TOK + 4095) & ~(size_t)4095;
        size_t sb = (((size_t)(qmax / GS + 1) * SPEC_MAX_TOK * sizeof(float)) + 4095) & ~(size_t)4095;
        size_t ssb = (((size_t)nthn * 16 * sizeof(float)) + 4095) & ~(size_t)4095;
        NB[k].q = aligned_alloc(4096, qb);
        NB[k].s = aligned_alloc(4096, sb);
        NB[k].ss = aligned_alloc(4096, ssb);
#if defined(__linux__)
        if (NN > 1) {
            bind_range(NB[k].q, qb, NODE_ID[k]);
            bind_range(NB[k].s, sb, NODE_ID[k]);
            bind_range(NB[k].ss, ssb, NODE_ID[k]);
        }
#endif
        memset(NB[k].q, 0, qb);
        memset(NB[k].s, 0, sb);
        memset(NB[k].ss, 0, ssb);
        if (nthn > 512) { fprintf(stderr, "too many threads per node\n"); exit(EXIT_FAILURE); }
        size_t syb = (sizeof(struct NodeSync) + 4095) & ~(size_t)4095;
        NB[k].sy = aligned_alloc(4096, syb);
#if defined(__linux__)
        if (NN > 1) bind_range(NB[k].sy, syb, NODE_ID[k]);
#endif
        memset(NB[k].sy, 0, syb);
        NB[k].router = aligned_alloc(4096, 4096 * sizeof(float));
#if defined(__linux__)
        if (NN > 1) bind_range(NB[k].router, 4096 * sizeof(float), NODE_ID[k]);
#endif
        memset(NB[k].router, 0, 4096 * sizeof(float));
        if (p->n_experts > 0) {
            size_t gb = ((size_t)SPEC_MAX_TOK * p->n_active * p->dim + 4095) & ~(size_t)4095;
            size_t gsb = ((size_t)SPEC_MAX_TOK * p->n_active * (p->dim / GS) * sizeof(float) + 4095) & ~(size_t)4095;
            NB[k].gq = aligned_alloc(4096, gb);
            NB[k].gs = aligned_alloc(4096, gsb);
#if defined(__linux__)
            if (NN > 1) { bind_range(NB[k].gq, gb, NODE_ID[k]); bind_range(NB[k].gs, gsb, NODE_ID[k]); }
#endif
            memset(NB[k].gq, 0, gb);
            memset(NB[k].gs, 0, gsb);
        }
    }
    for (int k = 0; k < 8; k++) atomic_store(&P.xflag[k].c, 0);
    atomic_store(&P.go, 0);
    atomic_store(&P.sleepers, 0);
    atomic_store(&P.quit, 0);
#if defined(__linux__)
#endif
    pin_thread(0);
    init_batch(t);
    for (int k = 0; k < NN; k++)
        for (int par = 0; par < 2; par++) {
            size_t bytes = ((size_t)(WORK_PHASES * 256 + 1) * sizeof(*P.work[k][par]) + 4095) & ~(size_t)4095; // any model up to 256 layers
            P.work[k][par] = aligned_alloc(4096, bytes);
#if defined(__linux__)
            if (NN > 1) bind_range(P.work[k][par], bytes, NODE_ID[k]);
#endif
            memset(P.work[k][par], 0, bytes);
        }
    P.seq = 0;
    prof_init(nth);
    P.tids = calloc(nth, sizeof(pthread_t));
    for (int i = 1; i < nth; i++) {
        if (pthread_create(&P.tids[i], NULL, pool_worker, (void*)(intptr_t)i) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            exit(EXIT_FAILURE);
        }
    }
    if (getenv("RUNQ_BARBENCH")) {
        P.mode = 3;
        wake_workers();
        barrier_bench_thread(0);
        P.mode = 0;
    }
}

void free_pool(void) {
    prof_report(P.nth);
    atomic_store_explicit(&P.quit, 1, memory_order_release);
    wake_workers();
    for (int i = 1; i < P.nth; i++) { pthread_join(P.tids[i], NULL); }
    for (int i = 0; i < P.nth; i++) {
        free(P.tb[i].xb); free(P.tb[i].q); free(P.tb[i].s);
        free(P.tb[i].rope_c); free(P.tb[i].rope_s);
    }
    free(P.tb); free(P.tids); free(P.part); free(P.rope_freq);
}

float* forward(Transformer* transformer, int token, int pos) {
    use_model(transformer);
    if (RECURRENT(&transformer->config)) { // the recurrent state can only advance, or restart at 0
        if (pos != 0 && pos != transformer->state.next_pos) {
            fprintf(stderr, "hybrid model: position %d out of order (state at %d)\n", pos, transformer->state.next_pos);
            exit(EXIT_FAILURE);
        }
        transformer->state.next_pos = pos + 1;
    }
    P.t = transformer;
    P.token = token;
    P.pos = pos;
    P.seq++;
    wake_workers();
    forward_thread(0);
#ifdef PROFILE
    PROF_TOKENS++;
#endif
    return transformer->state.logits;
}

