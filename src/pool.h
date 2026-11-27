// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += NORM_EPS;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

// ----------------------------------------------------------------------------
// Parallel runtime: a persistent pool of pinned threads that run the whole
// forward pass together, synchronizing with a spin barrier between phases.
// Small serial steps (rmsnorm, quantization of activations, attention merge)
// are computed redundantly by every thread into private buffers, which is
// cheaper than an extra barrier for models of this size.

#define MOE_MAX_ACTIVE 16
struct BarGroup { _Alignas(64) atomic_int c; };
typedef struct {
    _Alignas(128) float* xb; // (dim,) private normalized input / attention output (padded: no false sharing)
    int8_t* q;     // (max(dim, hidden_dim),) private quantized activations
    float* s;      // scale factors for q
    float* rope_c; // (head_size/2,) cos for the current position
    float* rope_s; // (head_size/2,) sin for the current position
    int node;      // NUMA node index (0..NN-1) of this thread
    int ltid, nthn; // index within the node, threads of the node
    int ep;          // barrier epoch of this thread (node-wide: every barrier of the node increments it)
    long long phase; // work phase of the last grab (the first grab of a phase takes the static ticket ltid)
    int npf;         // weight regions to prefetch while waiting at the next barrier
    const char* pf[MOE_MAX_ACTIVE * 2];
    int pflen[MOE_MAX_ACTIVE * 2];
} ThreadBuf;

typedef struct {
    int nth;            // number of threads, including the caller
    int C;              // attention context chunks per head
    Transformer* t;
    int token, pos;
    int mode;           // 0: one token (forward_thread), 1: a block of tokens (forward_batch_thread)
    struct WorkCounter { _Alignas(64) atomic_int v; }* work[MAX_NODES][2]; // per node, per token parity: next free row of each phase
    int seq;            // forward call counter (selects the counter set)
    pthread_t* tids;
    ThreadBuf* tb;
    float* part;        // (n_heads * C, head_size + 2) attention partials: max, sum, acc
    double* rope_freq;  // (head_size/2,)
    struct BarGroup xflag[8];            // cross-node level: epoch published by each node leader
    _Alignas(64) atomic_int go;
    _Alignas(64) atomic_int sleepers;   // workers blocked in futex_wait on `go`
    atomic_int quit;
} Pool;

static Pool P;

/* Dynamic row distribution for the decode matmuls: threads grab blocks of rows from a shared
 * counter, so a core that gets less memory bandwidth (mesh position) simply does fewer blocks
 * instead of making everybody wait at the next barrier. */
// phase k of a token, d-row matrix: the threads of a node take blocks of that node's rows only
// (the counters of a node live in that node's memory: a remote atomic per block was a bottleneck)
// Guided schedule over [0, n): tickets 0..nbig-1 are big chunks, the last quarter of the range goes
// in 4-row tickets, so the phase ends with short pieces instead of a long tail of one big chunk
// (under full memory load a thread streams only ~2 GB/s, a 16-row chunk would take ~15 us).
static inline int ticket_range(int t, int n, int nthn, int* b0, int* b1) {
    int chunk = (n / (nthn * 4) + 3) & ~3;
    if (chunk < 16) chunk = 16; // shorter row runs stream worse (prefetcher ramp-up per row)
    int nbig = (n - n / 4) / chunk;
    int b, e;
    if (t < nbig) { b = t * chunk; e = b + chunk; }
    else { b = nbig * chunk + (t - nbig) * 4; e = b + 4; }
    if (b >= n) return 0;
    *b0 = b;
    *b1 = e < n ? e : n;
    return 1;
}
// The first ticket of every thread in a phase is static (ticket ltid), so its rows are known before
// the phase starts and can be prefetched while the thread waits at the barrier (pf_first).
static inline int grab_ticket(atomic_int* ctr, int n, ThreadBuf* tb, int k, int* b0, int* b1) {
    long long key = ((long long)P.seq << 20) | k;
    int t;
    if (tb->phase != key) { tb->phase = key; t = tb->ltid; }
    else t = tb->nthn + atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
    return ticket_range(t, n, tb->nthn, b0, b1);
}
static inline int grab_rows(int k, int d, ThreadBuf* tb, int* r0, int* r1) {
    int lo, hi;
    node_rows(d, tb->node, &lo, &hi);
    if (!grab_ticket(&P.work[tb->node][P.seq & 1][k].v, hi - lo, tb, k, r0, r1)) return 0;
    *r0 += lo;
    *r1 += lo;
    return 1;
}
// like grab_rows, over a node-local virtual range [0, n) (MoE: the node's halves of the selected experts)
static inline int grab_virtual(int k, int n, ThreadBuf* tb, int* r0, int* r1) {
    return grab_ticket(&P.work[tb->node][P.seq & 1][k].v, n, tb, k, r0, r1);
}
// queue rows [r0, r1) of w (cols columns) for prefetching during the next barrier wait
static inline void pf_add(ThreadBuf* tb, const QuantizedTensor* w, int cols, int r0, int r1) {
    if (tb->npf >= MOE_MAX_ACTIVE * 2 || r1 <= r0) return;
    size_t bpr = w->bits == 4 ? cols / 2 : cols;
    tb->pf[tb->npf] = (const char*)(w->q + (size_t)r0 * bpr);
    tb->pflen[tb->npf] = (int)((r1 - r0) * bpr);
    tb->npf++;
}
// the rows of this thread's static first ticket of a node-split d-row matrix (virtual: n rows from 0)
static inline void pf_first(ThreadBuf* tb, const QuantizedTensor* w, int cols, int d, int virt) {
    int lo = 0, hi = d, b0, b1;
    if (!virt) node_rows(d, tb->node, &lo, &hi);
    if (ticket_range(tb->ltid, hi - lo, tb->nthn, &b0, &b1)) pf_add(tb, w, cols, lo + b0, lo + b1);
}
// one step of the prefetch queue (called from the barrier spin loops): a few lines into L2
static inline void pf_step(ThreadBuf* tb) {
    if (tb->npf == 0) return;
    int i = tb->npf - 1;
    for (int j = 0; j < 4 && tb->pflen[i] > 0; j++) {
#if defined(__aarch64__)
        __asm__ __volatile__("prfm pldl2keep, [%0]" : : "r"(tb->pf[i]));
#else
        __builtin_prefetch(tb->pf[i], 0, 2);
#endif
        tb->pf[i] += 64;
        tb->pflen[i] -= 64;
    }
    if (tb->pflen[i] <= 0) tb->npf--;
}
// rows [e*rows, (e+1)*rows) of a tensor as a tensor of its own (an expert of a MoE layer)
static inline QuantizedTensor expert_view(const QuantizedTensor* t, int e, int rows, int cols) {
    size_t bpr = t->bits == 4 ? cols / 2 : cols, spr = t->bits == 4 ? cols / 32 : cols / GS;
    QuantizedTensor v = {t->q + (size_t)e * rows * bpr, t->s + (size_t)e * rows * spr, t->bits};
    return v;
}
// MoE routing (Qwen3-MoE): softmax over the experts, top-k, optional renormalization
static void moe_route(const float* logits, int E, int k, int renorm, int* sel, float* wt) {
    float m = logits[0], z = 0.0f;
    for (int e = 1; e < E; e++) m = logits[e] > m ? logits[e] : m;
    for (int e = 0; e < E; e++) z += expf(logits[e] - m);
    for (int j = 0; j < k; j++) { sel[j] = -1; wt[j] = -1.0f; }
    for (int e = 0; e < E; e++) { // insertion into the sorted top-k
        float pr = expf(logits[e] - m) / z;
        int j = k - 1;
        if (pr <= wt[j]) continue;
        while (j > 0 && wt[j - 1] < pr) { wt[j] = wt[j - 1]; sel[j] = sel[j - 1]; j--; }
        wt[j] = pr;
        sel[j] = e;
    }
    if (renorm) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++) sum += wt[j];
        for (int j = 0; j < k; j++) wt[j] /= sum;
    }
}
// per-head RMSNorm (Qwen3 q_norm / k_norm) followed by RoPE on adjacent pairs
static inline void head_norm_rope(float* v, const float* w, int hs, const float* rc, const float* rs) {
    float ss = 0.0f;
    for (int i = 0; i < hs; i++) ss += v[i] * v[i];
    float inv = 1.0f / sqrtf(ss / hs + NORM_EPS);
    for (int i = 0; i < hs; i++) v[i] = w[i] * (v[i] * inv);
    for (int i = 0; i < hs; i += 2) {
        float v0 = v[i], v1 = v[i + 1];
        v[i] = v0 * rc[i / 2] - v1 * rs[i / 2];
        v[i + 1] = v0 * rs[i / 2] + v1 * rc[i / 2];
    }
}
#define WORK_PHASES 7 // per layer: q, k, v, wo, w1/w3 (or experts), w2, MoE router (+1 classifier)

// ---------------------------------------------------------------------------- profiling (-DPROFILE)
enum { PH_ROPE, PH_NORM, PH_QKV, PH_ATT, PH_MERGE, PH_WO, PH_FNORM, PH_W13, PH_QH, PH_W2, PH_CLS, PH_BARRIER, PH_NBARRIER, PH_IDLE, PH_N };
#ifdef PROFILE
static const char* PH_NAME[PH_N] = {"rope/setup", "att rmsnorm+quant", "qkv matmul+rope", "attention", "att merge+quant",
                                    "wo matmul", "ffn rmsnorm+quant", "w1/w3 matmul+swiglu", "quant hidden", "w2 matmul",
                                    "final norm+classifier", "barrier wait", "node barrier wait", "idle (outside forward)"};
#endif
#ifdef PROFILE
static inline uint64_t prof_ticks(void) { uint64_t v; __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(v)); return v; }
static inline uint64_t prof_freq(void) { uint64_t v; __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(v)); return v; }
typedef struct { _Alignas(64) uint64_t t[PH_N]; uint64_t last; int cur; } ProfThread;
static ProfThread PROF_T[1024];
static long PROF_TOKENS = 0;
#define PROF(tid, ph) do { ProfThread* pt_ = &PROF_T[tid]; uint64_t n_ = prof_ticks(); pt_->t[pt_->cur] += n_ - pt_->last; pt_->last = n_; pt_->cur = (ph); } while (0)
static void prof_init(int nth) {
    uint64_t n = prof_ticks();
    for (int i = 0; i < nth; i++) { PROF_T[i].last = n; PROF_T[i].cur = PH_IDLE; }
}
static void prof_report(int nth) {
    if (PROF_TOKENS == 0) return;
    double f = (double)prof_freq();
    double tot0 = 0, totm = 0;
    fprintf(stderr, "\n[profile] %ld forward calls, %d threads, ms per call: thread0 | mean over threads | max thread\n", PROF_TOKENS, nth);
    for (int ph = 0; ph < PH_N; ph++) {
        double t0 = PROF_T[0].t[ph] / f * 1e3 / PROF_TOKENS, mean = 0, mx = 0;
        for (int i = 0; i < nth; i++) { double v = PROF_T[i].t[ph] / f * 1e3 / PROF_TOKENS; mean += v; if (v > mx) mx = v; }
        mean /= nth;
        tot0 += t0; totm += mean;
        fprintf(stderr, "  %-24s %8.3f | %8.3f | %8.3f", PH_NAME[ph], t0, mean, mx);
        for (int k = 0; k < NN; k++) {
            double m = 0;
            for (int i = NODE_T0[k]; i < NODE_T0[k + 1]; i++) m += PROF_T[i].t[ph] / f * 1e3 / PROF_TOKENS;
            fprintf(stderr, " | node%d %8.3f", NODE_ID[k], m / (NODE_T0[k + 1] - NODE_T0[k]));
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "  %-24s %8.3f | %8.3f\n", "total", tot0, totm);
}
#else
#define PROF(tid, ph) do { } while (0)
static void prof_report(int nth) { (void)nth; }
static void prof_init(int nth) { (void)nth; }
#endif

#define KV_MUL_MAX 16 // max query heads per kv head (GQA group size)

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

static void barrier_impl(ThreadBuf* tb);
static void barrier(ThreadBuf* tb) {
#ifdef PROFILE
    int tid = (int)(tb - P.tb), saved = PROF_T[tid].cur;
    PROF(tid, PH_BARRIER);
    barrier_impl(tb);
    PROF(tid, saved);
#else
    barrier_impl(tb);
#endif
}
/* Barriers are trees: threads arrive at the counter of their group of BAR_GROUP, the last one of
 * each group arrives at the node counter, the last one of the node either releases the node
 * (node barrier) or first meets the representatives of the other nodes (global barrier). All
 * spinning happens on a node-local line; only NN threads touch the cross-socket counter. */
#define BAR_GROUP 16
static void node_arrive_release(ThreadBuf* tb, int global);
static void barrier_impl(ThreadBuf* tb) {
    if (P.nth == 1) return;
    node_arrive_release(tb, NN > 1);
}

/* Per-node cooperation. The input vector of every matmul (normalized + quantized) is produced
 * once per NUMA node by all threads of that node, each doing a slice, into a node-local buffer.
 * Every cache line of the shared fp32 activations crosses the socket link once per node instead
 * of once per thread, and nobody repeats the whole rmsnorm/quantization. */
typedef struct {
    int8_t* q;     // quantized input vector (max(dim, hidden_dim))
    float* s;      // its scales
    float* ss;     // per-thread partial sums of squares (one per cache line)
    struct NodeSync* sy; // barrier state of the node, in a page bound to the node
    float* router;       // MoE router logits computed by the node (every node routes on its own)
    int8_t* gq;          // MoE multi-token step: quantized ffn input of every (expert, token) pair
    float* gs;
} NodeBuf;
struct NodeSync {
    _Alignas(64) atomic_int rel;     // release epoch the node's threads spin on
    _Alignas(64) int xep;            // cross-node epoch (node leader only)
    struct BarGroup arr[512];        // per-thread arrival epochs (one line each)
    struct BarGroup garr[64];        // per-group arrival epochs
    struct BarGroup grel[64];        // per-group release epochs (set by the group leader)
};
static NodeBuf NB[MAX_NODES];
static int P_qmax = 0; // longest quantized node vector (NB slot size)
static int P_max_layers = 0; // most layers of any loaded model (work counter reset range)
#define SPEC_MAX_TOK 32 // tokens per multi-token decode step (speculative verification)
static void quantize_vec(int8_t* q, float* s, const float* x, int n);

/* Flag tree, no atomic read-modify-write: a thread publishes its epoch in its own cache line; the
 * fixed leader of each group of BAR_GROUP polls its members, the node leader (ltid 0) polls the group
 * leaders, meets the other node leaders through one flag line each (global barrier) and releases the
 * node with one store. Polling reads in parallel instead of serializing RMWs on a shared line. */
static void node_arrive_release(ThreadBuf* tb, int global) {
    struct NodeSync* sy = NB[tb->node].sy;
    const int lt = tb->ltid, e = ++tb->ep;
    pf_step(tb);
    if (lt % BAR_GROUP != 0) {
        atomic_store_explicit(&sy->arr[lt].c, e, memory_order_release);
    } else {
        const int end = lt + BAR_GROUP < tb->nthn ? lt + BAR_GROUP : tb->nthn;
        for (int j = lt + 1; j < end; j++)
            while (atomic_load_explicit(&sy->arr[j].c, memory_order_acquire) != e) { pf_step(tb); cpu_relax(); }
        if (lt != 0) {
            atomic_store_explicit(&sy->garr[lt / BAR_GROUP].c, e, memory_order_release);
            while (atomic_load_explicit(&sy->rel, memory_order_acquire) != e) { pf_step(tb); cpu_relax(); }
            atomic_store_explicit(&sy->grel[lt / BAR_GROUP].c, e, memory_order_release);
            tb->npf = 0;
            return;
        } else {
            const int ng = (tb->nthn + BAR_GROUP - 1) / BAR_GROUP;
            for (int g = 1; g < ng; g++)
                while (atomic_load_explicit(&sy->garr[g].c, memory_order_acquire) != e) { pf_step(tb); cpu_relax(); }
            if (global) {
                int xe = ++sy->xep;
                atomic_store_explicit(&P.xflag[tb->node].c, xe, memory_order_release);
                for (int k = 0; k < NN; k++)
                    if (k != tb->node)
                        while (atomic_load_explicit(&P.xflag[k].c, memory_order_acquire) - xe < 0) { pf_step(tb); cpu_relax(); }
            }
            atomic_store_explicit(&sy->rel, e, memory_order_release);
            atomic_store_explicit(&sy->grel[0].c, e, memory_order_release);
            tb->npf = 0;
            return;
        }
    }
    while (atomic_load_explicit(&sy->grel[lt / BAR_GROUP].c, memory_order_acquire) != e) { pf_step(tb); cpu_relax(); }
    tb->npf = 0;
}
static void node_barrier(ThreadBuf* tb) {
    if (tb->nthn == 1) return;
#ifdef PROFILE
    int tid = (int)(tb - P.tb), saved = PROF_T[tid].cur;
    PROF(tid, PH_NBARRIER);
    node_arrive_release(tb, 0);
    PROF(tid, saved);
#else
    node_arrive_release(tb, 0);
#endif
}

// groups [g0, g1) of an n-vector handled by this thread inside its node
static inline void node_groups_slice(int n, const ThreadBuf* tb, int* g0, int* g1) {
    int G = n / GS;
    *g0 = (int)((long)G * tb->ltid / tb->nthn);
    *g1 = (int)((long)G * (tb->ltid + 1) / tb->nthn);
}

// node buffer <- quantize(rmsnorm(x) * weight), cooperatively
static void node_norm_quant(ThreadBuf* tb, const float* x, const float* weight, int n) {
    NodeBuf* nb = &NB[tb->node];
    int g0, g1;
    node_groups_slice(n, tb, &g0, &g1);
    float ss = 0.0f;
    for (int i = g0 * GS; i < g1 * GS; i++) ss += x[i] * x[i];
    nb->ss[tb->ltid * 16] = ss;
    node_barrier(tb);
    float tot = 0.0f;
    for (int t = 0; t < tb->nthn; t++) tot += nb->ss[t * 16];
    const float inv = 1.0f / sqrtf(tot / n + NORM_EPS);
    float tmp[256];
    for (int g = g0; g < g1; g++) {
        for (int i = 0; i < GS; i++) tmp[i] = weight[g * GS + i] * (inv * x[g * GS + i]);
        quantize_vec(nb->q + (size_t)g * GS, nb->s + g, tmp, GS);
    }
    node_barrier(tb);
}

// node buffer <- quantize(v), cooperatively
static void node_quant(ThreadBuf* tb, const float* v, int n) {
    NodeBuf* nb = &NB[tb->node];
    int g0, g1;
    node_groups_slice(n, tb, &g0, &g1);
    if (g1 > g0) quantize_vec(nb->q + (size_t)g0 * GS, nb->s + g0, v + (size_t)g0 * GS, (g1 - g0) * GS);
    node_barrier(tb);
}
// multi-token versions: B vectors of length n at X + b*n -> node buffer (token b at q + b*n, s + b*n/GS)
static void node_norm_quant_multi(ThreadBuf* tb, const float* X, const float* weight, int n, int B) {
    NodeBuf* nb = &NB[tb->node];
    int g0, g1;
    node_groups_slice(n, tb, &g0, &g1);
    for (int b = 0; b < B; b++) {
        const float* x = X + (size_t)b * n;
        float ss = 0.0f;
        for (int i = g0 * GS; i < g1 * GS; i++) ss += x[i] * x[i];
        nb->ss[tb->ltid * 16 + b] = ss;
    }
    node_barrier(tb);
    float tmp[256];
    for (int b = 0; b < B; b++) {
        float tot = 0.0f;
        for (int t = 0; t < tb->nthn; t++) tot += nb->ss[t * 16 + b];
        const float inv = 1.0f / sqrtf(tot / n + NORM_EPS);
        const float* x = X + (size_t)b * n;
        for (int g = g0; g < g1; g++) {
            for (int i = 0; i < GS; i++) tmp[i] = weight[g * GS + i] * (inv * x[g * GS + i]);
            quantize_vec(nb->q + (size_t)b * n + (size_t)g * GS, nb->s + (size_t)b * (n / GS) + g, tmp, GS);
        }
    }
    node_barrier(tb);
}
static void node_quant_multi(ThreadBuf* tb, const float* V, int n, int B) {
    NodeBuf* nb = &NB[tb->node];
    int g0, g1;
    node_groups_slice(n, tb, &g0, &g1);
    if (g1 > g0)
        for (int b = 0; b < B; b++)
            quantize_vec(nb->q + (size_t)b * n + (size_t)g0 * GS, nb->s + (size_t)b * (n / GS) + g0, V + (size_t)b * n + (size_t)g0 * GS, (g1 - g0) * GS);
    node_barrier(tb);
}

// split the rows of the thread's NUMA node among the threads of that node, in blocks of 4 rows
static inline void part_range_node(int d, const ThreadBuf* tb, int* r0, int* r1) {
    int lo, hi;
    node_rows(d, tb->node, &lo, &hi);
    int blocks = (hi - lo) / 4;
    *r0 = lo + 4 * (int)((long)blocks * tb->ltid / tb->nthn);
    *r1 = (tb->ltid == tb->nthn - 1) ? hi : lo + 4 * (int)((long)blocks * (tb->ltid + 1) / tb->nthn);
}
// split d rows among threads in blocks of 4 rows
static inline void part_range(int d, int tid, int nth, int* r0, int* r1) {
    int blocks = d / 4;
    *r0 = 4 * (int)((long)blocks * tid / nth);
    *r1 = (tid == nth - 1) ? d : 4 * (int)((long)blocks * (tid + 1) / nth);
}

