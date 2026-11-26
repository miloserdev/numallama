/* Parallel memcpy on the thread pool: a serial 150 MB state snapshot cost ~25 ms (one core's
 * bandwidth), striped over the pool it takes ~1 ms. Each thread copies its slice. */
static struct {
  char *dst;
  const char *src;
  size_t n;
} CP;
static void copy_thread(int tid) {
  ThreadBuf *tb = &P.tb[tid];
  const size_t i0 = CP.n * tid / P.nth, i1 = CP.n * (tid + 1) / P.nth;
  memcpy(CP.dst + i0, CP.src + i0, i1 - i0);
  barrier(tb);
}
static void parallel_copy(void *dst, const void *src, size_t n) {
  if (P.nth <= 1 || n < (1 << 20)) {
    memcpy(dst, src, n);
    return;
  }
  CP.dst = dst;
  CP.src = src;
  CP.n = n;
  P.mode = 5;
  wake_workers();
  copy_thread(0);
  P.mode = 0;
}

/* Commit the accepted prefix of a speculative verify pass without a second weight read: the pass
 * saved every linear layer's fused input (BT.save_in -> SPEC_IN), and the state updates are a
 * fixed function of those inputs, so the recurrent / conv states are rebuilt from the snapshot
 * by replaying only the state updates (the recurrence output itself is discarded). */
static struct {
  int ntok;
  float *out; // (nth, nv dv) per-thread scratch for the discarded recurrence output
} SC;
static float *SPEC_OUT = NULL;
static void commit_thread(int tid) {
  Transformer *t = P.t;
  Config *p = &t->config;
  RunState *s = &t->state;
  ThreadBuf *tb = &P.tb[tid];
  const int LI = p->lin_in, nlin = p->n_layers - p->n_attn, nv = p->lin_nv, vd = nv * p->lin_dv;
  const int ntok = SC.ntok, nth = P.nth;
  float *out = SC.out + (size_t)tid * vd;
  for (int it = tid; it < nlin * LI; it += nth) { // conv + activations: (layer, channel) items, tokens in order
    int li = it / LI, r = it % LI;
    float *row = SPEC_IN + (size_t)li * SPEC_MAX * LI + r;
    for (int b = 0; b < ntok; b++) hyb_in_row(p, &t->weights, s->conv_state, li, r, row + (size_t)b * LI);
  }
  barrier(tb);
  const int ipb = (p->lin_dv + HYB_ROWS - 1) / HYB_ROWS;
  for (int it = tid; it < nlin * nv * ipb; it += nth) { // delta rule: (layer, head, row block) items
    int li = it / (nv * ipb), h = (it / ipb) % nv, c0 = (it % ipb) * HYB_ROWS;
    float *S = hyb_state(p, s, li, h, c0);
    for (int b = 0; b < ntok; b++) hyb_delta(p, S, SPEC_IN + (size_t)li * SPEC_MAX * LI + (size_t)b * LI, h, c0, out);
  }
  barrier(tb);
}
static void hybrid_spec_commit(Transformer *t, int ntok) { // state: spec_pos -> spec_pos + ntok, no weight reads
  if (!SPEC_OUT) SPEC_OUT = malloc((size_t)P.nth * t->config.lin_nv * t->config.lin_dv * sizeof(float));
  P.t = t;
  SC.ntok = ntok;
  SC.out = SPEC_OUT;
  P.mode = 6;
  wake_workers();
  commit_thread(0);
  P.mode = 0;
  t->state.next_pos += ntok;
}

