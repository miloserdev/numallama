}

#if defined(__linux__)
#include <sys/syscall.h>
// bind the pages fully inside [a, a+len) to a NUMA node (before they are touched)
static void bind_range(const void *a, size_t len, int node) {
  uintptr_t s0 = ((uintptr_t)a + 4095) & ~(uintptr_t)4095, e0 = ((uintptr_t)a + len) & ~(uintptr_t)4095;
  if (e0 <= s0) return;
  unsigned long mask[4] = {0};
  mask[node / 64] = 1UL << (node % 64);
  syscall(SYS_mbind, s0, e0 - s0, 2 /* MPOL_BIND */, mask, 256, 0);
}
// split every row-partitioned tensor across the nodes, matching node_rows()
static void place_tensor(QuantizedTensor *t, int n, int rows, int cols) {
  size_t bpr = t->bits == 4 ? cols / 2 : cols, spr = t->bits == 4 ? cols / 32 : cols / GS;
  for (int i = 0; i < n; i++)
    for (int k = 0; k < NN; k++) {
      int lo, hi;
      node_rows(rows, k, &lo, &hi);
      bind_range(t[i].q + lo * bpr, (hi - lo) * bpr, NODE_ID[k]);
      bind_range(t[i].s + lo * spr, (hi - lo) * spr * sizeof(float), NODE_ID[k]);
    }
}
// MoE: every expert block (rows [e*rows_per_expert, (e+1)*rows_per_expert)) is split between the
// nodes like a dense matrix, so the 8 experts of a token always put the same load on every node
static void place_experts(QuantizedTensor *t, int n, int experts, int rows_per_expert, int cols) {
  size_t bpr = t->bits == 4 ? cols / 2 : cols, spr = t->bits == 4 ? cols / 32 : cols / GS;
  for (int i = 0; i < n; i++)
    for (int e = 0; e < experts; e++)
      for (int k = 0; k < NN; k++) {
        int lo, hi;
        node_rows(rows_per_expert, k, &lo, &hi);
        size_t r0 = (size_t)e * rows_per_expert + lo;
        bind_range(t[i].q + r0 * bpr, (hi - lo) * bpr, NODE_ID[k]);
        bind_range(t[i].s + r0 * spr, (hi - lo) * spr * sizeof(float), NODE_ID[k]);
      }
}
static void place_weights(TransformerWeights *w, Config *p) {
  place_tensor(w->wq, p->n_attn, p->q_dim, p->dim);
  place_tensor(w->wk, p->n_attn, p->kv_dim, p->dim);
  place_tensor(w->wv, p->n_attn, p->kv_dim, p->dim);
  place_tensor(w->wo, p->n_attn, p->dim, p->q_dim);
  if (w->wgate) place_tensor(w->wgate, p->n_attn, p->q_dim, p->dim);
  if (p->hybrid) {
    place_tensor(w->w_in, p->n_layers - p->n_attn, p->lin_in, p->dim);
    place_tensor(w->w_out, p->n_layers - p->n_attn, p->dim, p->lin_nv * p->lin_dv);
  }
  if (p->n_experts > 0) {
    place_tensor(w->wg, p->n_layers, p->n_experts, p->dim);
    place_experts(w->w1, p->n_layers, p->n_experts, p->moe_hidden, p->dim);
    place_experts(w->w3, p->n_layers, p->n_experts, p->moe_hidden, p->dim);
    place_experts(w->w2, p->n_layers, p->n_experts, p->dim, p->moe_hidden);
  } else {
    place_tensor(w->w1, p->n_layers, p->hidden_dim, p->dim);
    place_tensor(w->w2, p->n_layers, p->dim, p->hidden_dim);
    place_tensor(w->w3, p->n_layers, p->hidden_dim, p->dim);
  }
  place_tensor(w->wcls, 1, p->vocab_size, p->dim);
}
#endif

/* Weight layout (Q4: see matmul_rows_q4). After loading, every int8 matrix is repacked in place into blocks of 4 rows:
 * per block and quantization group, GS/4 chunks of 16 bytes [row0 k..k+3 | row1 | row2 | row3],
 * and the scales of a block as [group][4 rows]. One `sdot` by lane (4 weight rows x 4 activations)
 * then yields the 4 rows in the 4 lanes, with no horizontal reductions; the batched kernel reuses
 * every weight vector for 8 tokens. Row offsets of multiples of 4 (node splits, experts) keep their
 * meaning, only the order inside a block changes. */
typedef struct { QuantizedTensor *t; int rows, cols; } PackJob;
static void *pack_job(void *arg) {
  PackJob *j = arg;
  if (j->t->bits == 4) { // Q4: chunk j of a group = bytes 4j..4j+3 of the 4 rows; scales [s0..s3 m0..m3]
    const int n = j->cols, ng = n / 32, bpr = n / 2;
    uint8_t *tq = malloc((size_t)4 * bpr);
    uint16_t *ts = malloc((size_t)4 * ng * 2 * sizeof(uint16_t));
    for (int b = 0; b < j->rows / 4; b++) {
      uint8_t *q = (uint8_t *)j->t->q + (size_t)b * 4 * bpr;
      uint16_t *sc = (uint16_t *)(j->t->s + (size_t)b * 4 * ng);
      memcpy(tq, q, (size_t)4 * bpr);
      memcpy(ts, sc, (size_t)4 * ng * 2 * sizeof(uint16_t));
      for (int g = 0; g < ng; g++) {
        for (int c = 0; c < 4; c++)
          for (int r = 0; r < 4; r++) memcpy(q + (size_t)g * 64 + c * 16 + r * 4, tq + (size_t)r * bpr + g * 16 + c * 4, 4);
        for (int r = 0; r < 4; r++) {
          sc[g * 8 + r] = ts[((size_t)r * ng + g) * 2];         // s
          sc[g * 8 + 4 + r] = ts[((size_t)r * ng + g) * 2 + 1]; // m
        }
      }
    }
    free(tq);
    free(ts);
    return NULL;
  }
  const int n = j->cols, ng = n / GS;
  int8_t *tq = malloc((size_t)4 * n);
  float *ts = malloc((size_t)4 * ng * sizeof(float));
  for (int b = 0; b < j->rows / 4; b++) {
    int8_t *q = j->t->q + (size_t)b * 4 * n;
    float *sc = j->t->s + (size_t)b * 4 * ng;
    memcpy(tq, q, (size_t)4 * n);
    memcpy(ts, sc, (size_t)4 * ng * sizeof(float));
    for (int g = 0; g < ng; g++) {
      for (int kk = 0; kk < GS / 4; kk++)
        for (int r = 0; r < 4; r++) memcpy(q + (size_t)g * GS * 4 + kk * 16 + r * 4, tq + (size_t)r * n + g * GS + kk * 4, 4);
      for (int r = 0; r < 4; r++) sc[g * 4 + r] = ts[(size_t)r * ng + g];
    }
  }
  free(tq);
  free(ts);
  return NULL;
}
static void pack_q8(TransformerWeights *w, Config *p) {
  struct { QuantizedTensor *t; int n, rows, cols; } list[] = {
    {w->wq, p->n_attn, p->q_dim, p->dim}, {w->wk, p->n_attn, p->kv_dim, p->dim}, {w->wv, p->n_attn, p->kv_dim, p->dim},
    {w->wo, p->n_attn, p->dim, p->q_dim}, {w->wgate, w->wgate ? p->n_attn : 0, p->q_dim, p->dim},
    {w->w_in, w->w_in ? p->n_layers - p->n_attn : 0, p->lin_in, p->dim},
    {w->w_out, w->w_out ? p->n_layers - p->n_attn : 0, p->dim, p->lin_nv * p->lin_dv},
    {w->w1, p->n_layers, p->n_experts > 0 ? p->n_experts * p->moe_hidden : p->hidden_dim, p->dim},
    {w->w3, p->n_layers, p->n_experts > 0 ? p->n_experts * p->moe_hidden : p->hidden_dim, p->dim},
    {w->w2, p->n_layers, p->n_experts > 0 ? p->n_experts * p->dim : p->dim, p->n_experts > 0 ? p->moe_hidden : p->hidden_dim},
    {w->wg, w->wg ? p->n_layers : 0, p->n_experts, p->dim}, {w->wcls, 1, p->vocab_size, p->dim}};
  int nj = 0;
  for (size_t k = 0; k < sizeof(list) / sizeof(list[0]); k++) nj += list[k].n;
  PackJob *jobs = malloc(nj * sizeof(PackJob));
  pthread_t *th = malloc(nj * sizeof(pthread_t));
  int m = 0;
  for (size_t k = 0; k < sizeof(list) / sizeof(list[0]); k++)
    for (int l = 0; l < list[k].n; l++) {
      if (list[k].rows % 4 || list[k].cols % GS) { fprintf(stderr, "Q8 matrix %d x %d: rows must be a multiple of 4\n", list[k].rows, list[k].cols); exit(EXIT_FAILURE); }
      jobs[m++] = (PackJob){list[k].t + l, list[k].rows, list[k].cols};
    }
  for (int a = 0; a < m; a += 128) { // at most 128 threads at a time
    int e = a + 128 < m ? a + 128 : m;
    for (int k = a; k < e; k++) pthread_create(&th[k], NULL, pack_job, &jobs[k]);
    for (int k = a; k < e; k++) pthread_join(th[k], NULL);
  }
  free(jobs);
  free(th);
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights, int *fd, float **data, ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  if (magic_number != 0x616b3432) {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }
  // read in the version number (uint32), has to be 2
  int version;
  if (fread(&version, sizeof(int), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  if (version != 2) {
    fprintf(stderr, "Bad version %d, need version 2\n", version);
    exit(EXIT_FAILURE);
  }
  int header_size = 256; // the header size for version 2 in bytes
  // read in the Config
  memset(config, 0, sizeof(Config));
  if (fread(config, 7 * sizeof(int), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // read in flags
  uint8_t shared_classifier; // a byte to indicate if the classifier is shared
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  int group_size; // the group size used in quantization
  if (fread(&group_size, sizeof(int), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  GS = group_size; // set as global, as it will be used in many places
  // optional rope parameters (zeros in files written by the original export.py)
  struct {
    float theta, factor, low, high;
    int orig;
  } rope = {0};
  if (fread(&rope, sizeof(rope), 1, file) == 1 && rope.theta > 0.0f) {
    ROPE_THETA = rope.theta;
    ROPE_FACTOR = rope.factor;
    ROPE_LOW_FREQ = rope.low;
    ROPE_HIGH_FREQ = rope.high;
    ROPE_ORIG_CTX = rope.orig;
  }
  struct {
    int arch;
    float eps;
    int qkv_bias, special_start, bos, eos;
  } ext = {0};
  if (fread(&ext, sizeof(ext), 1, file) == 1 && ext.eps > 0.0f) {
    ARCH = ext.arch;
    NORM_EPS = ext.eps;
    QKV_BIAS = ext.qkv_bias;
    SPECIAL_START = ext.special_start;
    BOS_ID = ext.bos;
    EOS_ID = ext.eos;
  }
  int q4mask = 0;
  if (fread(&q4mask, sizeof(int), 1, file) == 1) Q4_MASK = q4mask;
  int ext2[6] = {0}; // head_dim, n_experts, n_active, moe_hidden, qk_norm, norm_topk
  if (fread(ext2, sizeof(ext2), 1, file) != 1) memset(ext2, 0, sizeof(ext2));
  int q4_format = 0;
  if (fread(&q4_format, sizeof(int), 1, file) != 1) q4_format = 0;
  int hyb[8] = {0}; // full_interval, linear k heads, v heads, k dim, v dim, conv kernel, attention gate, rotary dims
  if (fread(hyb, sizeof(hyb), 1, file) != 1) memset(hyb, 0, sizeof(hyb));
  int gh[24] = {0}; // GLM block (export_glm5next.py)
  if (fread(gh, sizeof(gh), 1, file) != 1) memset(gh, 0, sizeof(gh));
  config->glm = ARCH == ARCH_GLM5;
  if (config->glm) {
    config->g_dense = gh[0]; config->g_ql = gh[1]; config->g_kvl = gh[2]; config->g_qk = gh[3]; config->g_vh = gh[4];
    config->g_H = gh[5]; config->g_dh = gh[6]; config->g_K = gh[7]; config->g_hc = gh[8]; config->g_sk = gh[9];
    memcpy(&config->g_hc_eps, &gh[10], 4); memcpy(&config->g_rscale, &gh[11], 4);
    memcpy(&config->g_limit, &gh[12], 4); memcpy(&config->g_lower, &gh[13], 4);
    config->g_ihh = gh[14]; config->g_ih = gh[15]; config->g_itopk = gh[16]; config->g_kpool = gh[17];
    config->g_amask[0] = (uint32_t)gh[18]; config->g_amask[1] = (uint32_t)gh[19]; config->g_q4 = gh[20]; config->g_tail = gh[21];
    if (config->g_hc > 8 || config->g_kpool > 16 || config->g_kvl > 1024 || config->g_dh > 256 || ext2[2] > 15) {
      fprintf(stderr, "%s: unsupported GLM dimensions\n", checkpoint);
      exit(EXIT_FAILURE);
    }
  }
  config->hybrid = ARCH == ARCH_QWEN35 && hyb[0] > 0;
  config->full_interval = hyb[0];
  config->n_attn = config->hybrid ? config->n_layers / hyb[0] : config->n_layers;
  config->lin_nk = hyb[1]; config->lin_nv = hyb[2]; config->lin_dk = hyb[3]; config->lin_dv = hyb[4];
  config->conv_k = hyb[5]; config->attn_gate = hyb[6];
  config->rot_dim = hyb[7] > 0 ? hyb[7] : 0;
  config->conv_dim = 2 * hyb[1] * hyb[3] + hyb[2] * hyb[4];
  config->lin_in = config->conv_dim + hyb[2] * hyb[4] + 2 * hyb[2];
  if (Q4_MASK && q4_format != 1) {
    fprintf(stderr, "%s: old symmetric Q4 format, please re-export it with scripts/export_hf_q8.py --q4\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  config->head_dim = ext2[0] > 0 ? ext2[0] : config->dim / config->n_heads;
  config->n_experts = ext2[1];
  config->n_active = ext2[2];
  config->moe_hidden = ext2[3];
  config->qk_norm = ext2[4];
  config->norm_topk = ext2[5];
  config->q_dim = config->n_heads * config->head_dim;
  config->kv_dim = config->n_kv_heads * config->head_dim;
  if (config->glm) { // KDA reuses the hybrid recurrent-state fields; n_attn = MLA layers
    int na = 0;
    for (int l = 0; l < config->n_layers; l++) na += (config->g_amask[l >> 5] >> (l & 31)) & 1;
    config->n_attn = na;
    config->lin_nk = config->lin_nv = config->g_H;
    config->lin_dk = config->lin_dv = config->g_dh;
    config->conv_k = config->g_K;
    config->conv_dim = 3 * config->g_H * config->g_dh;
  }
  // figure out the file size
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
  // read the whole checkpoint into anonymous memory (NUMA-local, huge pages) instead of mmap
  *fd = -1;
  if (posix_memalign((void **)data, 2 * 1024 * 1024, *file_size) != 0) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
#if defined(MADV_HUGEPAGE)
  madvise(*data, *file_size, MADV_HUGEPAGE);
#endif
  void *weights_ptr = ((char *)*data) + header_size; // skip header bytes. char is 1 byte
  weights->glm = NULL;
  if (config->glm) glm_map_weights(weights, config, weights_ptr);
  else memory_map_weights(weights, config, weights_ptr, shared_classifier); // pointers only, nothing touched yet
  EMB_PACKED = weights->wcls == weights->q_tokens;
#if defined(__linux__)
  if (NN > 1) { // each node gets its rows of every matrix
    if (config->glm) glm_place(config, weights->glm);
    else place_weights(weights, config);
  }
#endif
  fseek(file, 0, SEEK_SET);
  if (fread(*data, 1, *file_size, file) != (size_t)*file_size) {
    fprintf(stderr, "read failed!\n");
    exit(EXIT_FAILURE);
  }
  fclose(file);
  if (config->glm) { glm_pack(config, weights->glm); return; }
  pack_q8(weights, config); // embeddings stay quantized: the classifier may share them
  for (int k = 0; k < NN; k++) weights->wg_node[k] = weights->wg;
#if defined(__linux__)
  if (NN > 1 && config->n_experts > 0) { // the router is small (n_experts x dim): replicate it on every node
    size_t qb = (size_t)config->n_experts * config->dim, sb = qb / GS * sizeof(float);
    size_t per = ((qb + sb) + 4095) & ~(size_t)4095;
    for (int k = 0; k < NN; k++) {
      QuantizedTensor *c = malloc(config->n_layers * sizeof(QuantizedTensor));
      char *mem = aligned_alloc(4096, per * config->n_layers);
      bind_range(mem, per * config->n_layers, NODE_ID[k]);
      for (int l = 0; l < config->n_layers; l++) {
        c[l].q = (int8_t *)(mem + per * l);
        c[l].s = (float *)(mem + per * l + qb);
        c[l].bits = 8;
        memcpy(c[l].q, weights->wg[l].q, qb);
        memcpy(c[l].s, weights->wg[l].s, sb);
      }
      weights->wg_node[k] = c;
    }
  }
#endif
}

void build_transformer(Transformer *t, char *checkpoint_path, int ctx) {
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  t->g.gs = GS; t->g.arch = ARCH; t->g.qkv_bias = QKV_BIAS; t->g.special_start = SPECIAL_START;
  t->g.bos = BOS_ID; t->g.eos = EOS_ID; t->g.norm_eps = NORM_EPS;
  t->g.rope_theta = ROPE_THETA; t->g.rope_factor = ROPE_FACTOR; t->g.rope_low = ROPE_LOW_FREQ;
  t->g.rope_high = ROPE_HIGH_FREQ; t->g.rope_orig = ROPE_ORIG_CTX;
  t->rope_freq = NULL; t->part = NULL; t->C = 0;
  // limit the context window (the KV cache is allocated for seq_len positions)
  if (ctx > 0 && ctx < t->config.seq_len) {
    t->config.seq_len = ctx;
  }
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
#if defined(__linux__)
  if (NN > 1 && !t->config.glm) { // KV cache of kv head g lives on the node that runs its attention
    Config *p = &t->config;
    int hs = p->head_dim;
    size_t head_bytes = (size_t)p->seq_len * hs * sizeof(float);
    for (int l = 0; l < p->n_attn; l++)
      for (int k = 0; k < NN; k++) {
        int g0, g1;
        node_groups(p->n_kv_heads, k, &g0, &g1);
        size_t off = ((size_t)l * p->n_kv_heads + g0) * head_bytes;
        bind_range((char *)t->state.key_cache + off, (g1 - g0) * head_bytes, NODE_ID[k]);
        bind_range((char *)t->state.value_cache + off, (g1 - g0) * head_bytes, NODE_ID[k]);
      }
  }
#endif
}

void free_transformer(Transformer *t) {
  // free QuantizedTensors
  free(t->weights.q_tokens);
  free(t->weights.token_embedding_table);
  free(t->weights.wq);
  free(t->weights.wk);
  free(t->weights.wv);
  free(t->weights.wo);
  free(t->weights.w1);
  free(t->weights.w2);
  free(t->weights.w3);
  if (t->weights.wcls != t->weights.q_tokens) {
    free(t->weights.wcls);
  }
  // close the memory mapping
  free(t->data);
  // free the RunState buffers
  free_run_state(&t->state);
}

