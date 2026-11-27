// Inkling-2-Mini container (INK2JSON): a padded JSON manifest plus absolute-offset tensor
// blobs. The loader only maps the file and resolves names; the forward pass is separate.
// Q4 expert data is [expert][rows][cols] with packed nibbles then fp16 (scale, min) per 32.
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

typedef struct {
  char *name;
  int dtype;      // 0 f32, 1 q8, 2 q4
  int ndim, dims[4];
  void *data;
  size_t bytes;
} InkTensor;

typedef struct {
  int phys_layers, slots, dim, inter, heads, kv_heads, head_dim, vocab, unpadded;
  int experts, active, loop_start, loop_end, passes, gate_rank, d_rel, rel_extent;
  int window, conv_k, res_block, local_mask, eos, special_start;
  float eps, qk_clip, rel_clip, log_floor, log_alpha, final_softcap, gate_softcap, up_softcap, route_scale;
  InkTensor *t;
  int n_tensors;
  void *map;
  size_t map_size;
  JNode *manifest;
  char *manifest_text;
} InklingModel;

static inline int inkling_load(const char *path, InklingModel *m) {
#if defined(__linux__)
  memset(m, 0, sizeof *m);
  int fd = open(path, O_RDONLY);
  if (fd < 0) return 0;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 16) { close(fd); return 0; }
  m->map_size = (size_t)st.st_size;
  m->map = mmap(NULL, m->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (m->map == MAP_FAILED) { m->map = NULL; return 0; }
  const unsigned char *b = m->map;
  if (memcmp(b, "INK2JSON", 8) != 0) return 0;
  uint64_t jn;
  memcpy(&jn, b + 8, 8);
  if (jn > m->map_size - 16 || jn > (1 << 20)) return 0;
  m->manifest_text = malloc((size_t)jn + 1);
  if (!m->manifest_text) return 0;
  memcpy(m->manifest_text, b + 16, (size_t)jn);
  m->manifest_text[jn] = '\0';
  m->manifest = json_parse(m->manifest_text, NULL);
  if (!m->manifest) return 0;
  JNode *c = json_get(m->manifest, "config");
  if (!c) return 0;
  JNode *arr = json_get(m->manifest, "tensors");
  if (!arr || arr->type != J_ARR) return 0;
#define GI(field, key) do { JNode *v = json_get(c, key); if (!v || v->type != J_NUM) return 0; m->field = atoi(v->str); } while (0)
#define GF(field, key) do { JNode *v = json_get(c, key); if (!v || v->type != J_NUM) return 0; m->field = (float)atof(v->str); } while (0)
  GI(phys_layers, "phys_layers"); GI(slots, "slots"); GI(dim, "dim"); GI(inter, "inter");
  GI(heads, "heads"); GI(kv_heads, "kv_heads"); GI(head_dim, "head_dim"); GI(vocab, "vocab");
  GI(unpadded, "unpadded_vocab"); GI(experts, "experts"); GI(active, "active");
  GI(loop_start, "loop_start"); GI(loop_end, "loop_end"); GI(passes, "passes"); GI(gate_rank, "gate_rank");
  GI(d_rel, "d_rel"); GI(rel_extent, "rel_extent"); GI(window, "window"); GI(conv_k, "conv_k");
  GI(res_block, "res_block"); GI(eos, "eos"); GI(special_start, "special_start");
  GF(eps, "eps"); GF(qk_clip, "qk_clip"); GF(rel_clip, "rel_clip"); GF(log_floor, "log_floor");
  GF(log_alpha, "log_alpha"); GF(final_softcap, "final_softcap"); GF(gate_softcap, "gate_softcap");
  GF(up_softcap, "up_softcap"); GF(route_scale, "route_scale");
  JNode *lm = json_get(c, "local");
  if (lm && lm->type == J_ARR) for (JNode *v = lm->child; v; v = v->next) m->local_mask |= 1u << atoi(v->str);
  GS = 64; // INK2 Q8 groups are fixed at 64; the engine kernels and pack_job use this
  int n = 0;
  for (JNode *v = arr->child; v; v = v->next) n++;
  m->t = calloc((size_t)n, sizeof(InkTensor));
  if (!m->t) return 0;
  m->n_tensors = n;
  int i = 0;
  for (JNode *v = arr->child; v; v = v->next, i++) {
    JNode *nm = json_get(v, "name"), *dt = json_get(v, "dtype"), *sh = json_get(v, "shape");
    JNode *of = json_get(v, "offset"), *sz = json_get(v, "size");
    if (!nm || !dt || !sh || !of || !sz) return 0;
    InkTensor *t = &m->t[i];
    t->name = strdup(nm->str);
    t->dtype = !strcmp(dt->str, "f32") ? 0 : !strcmp(dt->str, "q8") ? 1 : 2;
    for (JNode *d = sh->child; d && t->ndim < 4; d = d->next) t->dims[t->ndim++] = atoi(d->str);
    size_t off = (size_t)strtoull(of->str, NULL, 10), bytes = (size_t)strtoull(sz->str, NULL, 10);
    if (off > m->map_size || bytes > m->map_size - off) return 0;
    t->data = (unsigned char *)m->map + off;
    t->bytes = bytes;
  }
  return 1;
#else
  (void)path; (void)m;
  return 0;
#endif
}

// Copy a row-major Q8/Q4 container tensor into a writable QuantizedTensor and repack it into the
// 4-row layout used by the engine kernels. The caller frees t->q and t->s.
static inline int inkling_pack_2d(const InkTensor *in, QuantizedTensor *t) {
  if (!in || in->ndim != 2 || (in->dtype != 1 && in->dtype != 2)) return 0;
  int rows = in->dims[0], cols = in->dims[1];
  if (rows % 4 || (in->dtype == 2 ? cols % 32 : cols % GS)) return 0;
  size_t qb = (size_t)rows * cols / (in->dtype == 2 ? 2 : 1);
  size_t sb = (size_t)rows * (in->dtype == 2 ? cols / 32 : cols / GS) * 4;
  if (qb + sb > in->bytes) return 0;
  t->q = malloc(qb);
  t->s = malloc(sb);
  if (!t->q || !t->s) { free(t->q); free(t->s); memset(t, 0, sizeof *t); return 0; }
  memcpy(t->q, in->data, qb);
  memcpy(t->s, (const char *)in->data + qb, sb);
  t->bits = in->dtype == 2 ? 4 : 8;
  PackJob job = {t, rows, cols};
  pack_job(&job);
  return 1;
}
static inline void inkling_pack_free(QuantizedTensor *t) { free(t->q); free(t->s); memset(t, 0, sizeof *t); }

static inline const InkTensor *inkling_find(const InklingModel *m, const char *name) {
  for (int i = 0; i < m->n_tensors; i++) if (!strcmp(m->t[i].name, name)) return &m->t[i];
  return NULL;
}

// Per-slot KV and short-conv state. The loop core reuses the 12 physical layers three times, but
// every visit has its own cache slot: 24 physical + 2 extra passes over 6..17 = 48 slots.
typedef struct {
  int ctx, slots;
  int8_t *kq, *vq;      // [slot][kv_head][pos][head_dim]
  float *ks, *vs;       // i8 per-64 scales, same order
  float *kf, *vf;       // f32 mirror used by the first validated forward
  float *conv;          // [slot][2*kv_dim + 2*dim][conv_k-1]
  float *x, *xb, *hb, *q, *k, *v, *att, *logits;
} InklingState;

static inline int inkling_slot_phys(const InklingModel *m, int slot) {
  return slot < m->phys_layers ? slot : m->loop_start + (slot - m->phys_layers) % (m->loop_end - m->loop_start);
}

static inline int inkling_state_alloc(const InklingModel *m, int ctx, InklingState *s) {
  memset(s, 0, sizeof *s);
  if (ctx <= 0) return 0;
  s->ctx = ctx; s->slots = m->slots;
  size_t row = (size_t)m->kv_heads * ctx * m->head_dim;
  s->kq = malloc((size_t)s->slots * row);
  s->vq = malloc((size_t)s->slots * row);
  s->ks = malloc((size_t)s->slots * row / GS * sizeof(float));
  s->vs = malloc((size_t)s->slots * row / GS * sizeof(float));
  s->kf = malloc((size_t)s->slots * row * sizeof(float));
  s->vf = malloc((size_t)s->slots * row * sizeof(float));
  size_t ch = 2 * (size_t)m->kv_heads * m->head_dim + 2 * (size_t)m->dim;
  s->conv = calloc((size_t)s->slots * ch * (m->conv_k - 1), sizeof(float));
  s->x = calloc(m->dim, sizeof(float)); s->xb = calloc(m->dim, sizeof(float));
  s->hb = calloc(m->inter > m->dim ? m->inter : m->dim, sizeof(float));
  s->q = calloc((size_t)m->heads * m->head_dim, sizeof(float));
  s->k = calloc((size_t)m->kv_heads * m->head_dim, sizeof(float));
  s->v = calloc((size_t)m->kv_heads * m->head_dim, sizeof(float));
  s->att = calloc((size_t)m->heads * ctx, sizeof(float));
  s->logits = calloc(m->vocab, sizeof(float));
  if (!s->kq || !s->vq || !s->ks || !s->vs || !s->kf || !s->vf || !s->conv || !s->x || !s->xb || !s->hb || !s->q ||
      !s->k || !s->v || !s->att || !s->logits) {
    free(s->kq); free(s->vq); free(s->ks); free(s->vs); free(s->kf); free(s->vf); free(s->conv); free(s->x); free(s->xb);
    free(s->hb); free(s->q); free(s->k); free(s->v); free(s->att); free(s->logits);
    memset(s, 0, sizeof *s);
    return 0;
  }
  return 1;
}

static inline void inkling_state_clear(const InklingModel *m, InklingState *s) {
  size_t row = (size_t)m->kv_heads * s->ctx * m->head_dim;
  memset(s->kq, 0, (size_t)s->slots * row);
  memset(s->vq, 0, (size_t)s->slots * row);
  memset(s->ks, 0, (size_t)s->slots * row / GS * sizeof(float));
  memset(s->vs, 0, (size_t)s->slots * row / GS * sizeof(float));
  memset(s->kf, 0, (size_t)s->slots * row * sizeof(float));
  memset(s->vf, 0, (size_t)s->slots * row * sizeof(float));
  size_t ch = 2 * (size_t)m->kv_heads * m->head_dim + 2 * (size_t)m->dim;
  memset(s->conv, 0, (size_t)s->slots * ch * (m->conv_k - 1) * sizeof(float));
}

static inline void inkling_state_free(InklingState *s) {
  free(s->kq); free(s->vq); free(s->ks); free(s->vs); free(s->kf); free(s->vf); free(s->conv); free(s->x); free(s->xb);
  free(s->hb); free(s->q); free(s->k); free(s->v); free(s->att); free(s->logits);
  memset(s, 0, sizeof *s);
}

static inline void inkling_free(InklingModel *m) {
  if (m->t) for (int i = 0; i < m->n_tensors; i++) free(m->t[i].name);
  free(m->t);
  json_free(m->manifest);
  free(m->manifest_text);
#if defined(__linux__)
  if (m->map) munmap(m->map, m->map_size);
#endif
  memset(m, 0, sizeof *m);
}
