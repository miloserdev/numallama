// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

/* Parallel sampling on the thread pool (the vocabulary is 128-152k entries: a serial softmax +
 * nucleus scan cost ~1.3 ms per token, more than 5% of a 30B-A3B decode step). Every thread takes a
 * static slice of the logits: (1) its max and sum of exp((l - max) / T) (or its argmax), barrier,
 * (2) the global max M and normalizer Z (online-softmax merge), and the slice's nucleus candidates
 * l >= M + T * ln(cutoff * Z), i.e. probability >= (1 - topp) / (n - 1) exactly as in sample_topp.
 * Thread 0 then sorts the few candidates. Same distribution as the serial code. */
static struct {
  const float *l;
  int n;
  float T, lnc; // temperature (0: argmax), ln of the nucleus cutoff probability
  float *m, *z; // per-thread max / partial normalizer (stride 16: one cache line per thread)
  int *arg, *cnt;
  int *cand;    // candidate indices, thread t writes from the start of its slice
  float M, Z;
} SP;

static void sample_thread(int tid) {
  ThreadBuf *tb = &P.tb[tid];
  const int n = SP.n, nth = P.nth;
  const int i0 = (int)((long)n * tid / nth), i1 = (int)((long)n * (tid + 1) / nth);
  const float *l = SP.l;
  float m = -FLT_MAX;
  int am = i0;
  for (int i = i0; i < i1; i++) if (l[i] > m) { m = l[i]; am = i; }
  SP.m[tid * 16] = m;
  SP.arg[tid * 16] = am;
  if (SP.T > 0.0f) {
    const float it = 1.0f / SP.T;
    float z = 0.0f;
    for (int i = i0; i < i1; i++) z += expf((l[i] - m) * it);
    SP.z[tid * 16] = z;
  }
  barrier(tb);
  if (SP.T <= 0.0f) return;
  float M = -FLT_MAX, Z = 0.0f;
  for (int t = 0; t < nth; t++) M = SP.m[t * 16] > M ? SP.m[t * 16] : M;
  for (int t = 0; t < nth; t++) Z += SP.z[t * 16] * expf((SP.m[t * 16] - M) / SP.T);
  const float thr = M + SP.T * (SP.lnc + logf(Z));
  int c = 0;
  for (int i = i0; i < i1; i++) if (l[i] >= thr) SP.cand[i0 + c++] = i;
  SP.cnt[tid * 16] = c;
  if (tid == 0) { SP.M = M; SP.Z = Z; }
  barrier(tb);
}
// builds the sorted nucleus of logits / T (0 < topp < 1) in pi; returns its size (all candidates),
// sets *last (last index inside the nucleus) and *cum (its probability mass)
static int par_nucleus(Sampler *sp, const float *logits, ProbIndex *pi, int *last, float *cum) {
  int n = sp->vocab_size;
  if (!SP.m) {
    SP.m = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(float));
    SP.z = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(float));
    SP.arg = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(int));
    SP.cnt = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(int));
  }
  static int cand_n = 0;
  if (cand_n < n) { free(SP.cand); SP.cand = malloc((size_t)n * sizeof(int)); cand_n = n; }
  SP.l = logits; SP.n = n; SP.T = sp->temperature;
  SP.lnc = logf((1.0f - sp->topp) / (n - 1));
  P.mode = 4;
  wake_workers();
  sample_thread(0);
  P.mode = 0;
  int n0 = 0;
  const float iz = 1.0f / SP.Z;
  for (int t = 0; t < P.nth; t++) {
    int i0 = (int)((long)n * t / P.nth);
    for (int c = 0; c < SP.cnt[t * 16]; c++) {
      int i = SP.cand[i0 + c];
      pi[n0].index = i;
      pi[n0].prob = expf((logits[i] - SP.M) / SP.T) * iz;
      n0++;
    }
  }
  qsort(pi, n0, sizeof(ProbIndex), compare);
  float cp = 0.0f;
  int li = n0 - 1;
  for (int i = 0; i < n0; i++) {
    cp += pi[i].prob;
    if (cp > sp->topp) { li = i; break; }
  }
  *last = li;
  *cum = cp;
  return n0;
}
static int par_argmax(const float *logits, int n) {
  if (!SP.m) {
    SP.m = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(float));
    SP.z = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(float));
    SP.arg = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(int));
    SP.cnt = aligned_alloc(64, (size_t)P.nth * 16 * sizeof(int));
  }
  SP.l = logits; SP.n = n; SP.T = 0.0f;
  P.mode = 4;
  wake_workers();
  sample_thread(0);
  P.mode = 0;
  int best = SP.arg[0];
  float bm = SP.m[0];
  for (int t = 1; t < P.nth; t++) if (SP.m[t * 16] > bm) { bm = SP.m[t * 16]; best = SP.arg[t * 16]; } // first max, like sample_argmax
  return best;
}
static inline int pool_ready(void) { return P.nth > 1 && P.tb != NULL; }

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f && pool_ready()) return par_argmax(logits, sampler->vocab_size);
  if (sampler->temperature > 0.0f && sampler->topp > 0 && sampler->topp < 1 && pool_ready()) {
    int last;
    float cum;
    par_nucleus(sampler, logits, sampler->probindex, &last, &cum);
    float r = random_f32(&sampler->rng_state) * cum, cdf = 0.0f;
    for (int i = 0; i <= last; i++) {
      cdf += sampler->probindex[i].prob;
      if (r < cdf) return sampler->probindex[i].index;
    }
    return sampler->probindex[last].index; // in case of rounding errors
  }
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

/* Speculative sampling with a deterministic draft token d (prompt lookup): returns d with
 * probability p(d) under the sampler's distribution (temperature + top-p), otherwise a sample from
 * that distribution with d removed. The output is distributed exactly like sample(). Greedy: argmax. */
int sample_draft(Sampler *sampler, float *logits, int d) {
  if (sampler->temperature == 0.0f) return pool_ready() ? par_argmax(logits, sampler->vocab_size) : sample_argmax(logits, sampler->vocab_size);
  int n = sampler->vocab_size;
  ProbIndex *pi = sampler->probindex;
  int n0 = 0, last;
  float cum = 0.0f;
  if (sampler->topp > 0 && sampler->topp < 1 && pool_ready()) {
    par_nucleus(sampler, logits, pi, &last, &cum);
  } else if (sampler->topp <= 0 || sampler->topp >= 1) {
    for (int q = 0; q < n; q++) logits[q] /= sampler->temperature;
    softmax(logits, n);
    for (int i = 0; i < n; i++) { pi[n0].index = i; pi[n0].prob = logits[i]; n0++; }
    last = n0 - 1;
    cum = 1.0f;
  } else {
    for (int q = 0; q < n; q++) logits[q] /= sampler->temperature;
    softmax(logits, n);
    const float cutoff = (1.0f - sampler->topp) / (n - 1);
    for (int i = 0; i < n; i++)
      if (logits[i] >= cutoff) { pi[n0].index = i; pi[n0].prob = logits[i]; n0++; }
    qsort(pi, n0, sizeof(ProbIndex), compare);
    last = n0 - 1;
    for (int i = 0; i < n0; i++) {
      cum += pi[i].prob;
      if (cum > sampler->topp) { last = i; break; }
    }
  }
  float pd = 0.0f; // probability of d in the (truncated, renormalized) distribution
  for (int i = 0; i <= last; i++) if (pi[i].index == d) { pd = pi[i].prob / cum; break; }
  if (random_f32(&sampler->rng_state) < pd) return d;
  // rejected: sample from the distribution without d
  float rest = cum - pd * cum;
  float r = random_f32(&sampler->rng_state) * rest, cdf = 0.0f;
  int fallback = -1;
  for (int i = 0; i <= last; i++) {
    if (pi[i].index == d) continue;
    fallback = pi[i].index;
    cdf += pi[i].prob;
    if (r < cdf) return pi[i].index;
  }
  return fallback >= 0 ? fallback : d;
}

/* Prompt lookup: find the most recent earlier occurrence of the last N tokens of hist (N = 4..2)
 * and propose the tokens that followed it. Returns the draft length. */
static int ngram_draft(const int *hist, int n, int *draft, int max_k) {
  for (int N = 4; N >= 2; N--) {
    if (n <= N) continue;
    const int *tail = hist + n - N;
    for (int s = n - N - 1; s >= 0; s--) { // most recent match first
      if (hist[s] != tail[0] || memcmp(hist + s, tail, N * sizeof(int)) != 0) continue;
      int k = 0;
      while (k < max_k && s + N + k < n) { draft[k] = hist[s + N + k]; k++; }
      if (k > 0) return k;
    }
  }
  return 0;
}

