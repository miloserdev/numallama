# Benchmarks

Machine: 2x Ampere Altra Max M128-30 (2 x 128 Neoverse N1 cores, 3.0 GHz), 2 NUMA nodes,
8 channels DDR4-3200 per socket (measured ~180 GB/s per socket, ~360 GB/s both), 1 TB RAM.
Date: 2026-11-18. Medians of 2–3 runs; the machine was otherwise idle.

Decode = token generation at short context (`tg`), prefill = 512-token prompt (`pp512`), tokens/s.

## numallama vs llama.cpp

llama.cpp built from master (2026-11-17) with `GGML_NATIVE=ON` (`neoverse-n1+dotprod`), OpenMP;
single socket under `numactl -N0 -m0 -t 96`; for 2 sockets the best of `--numa distribute` and
`numactl --interleave=all` with 96/112/128/192 threads (page cache dropped before each run).

| Model | llama.cpp 1 socket | numallama 1 socket | numallama 2 sockets | llama.cpp best 2-socket |
|---|---|---|---|---|
| Llama 3.2 3B Q8 | 36 / 512 | 48 / 1125 | **76** / **1172** | 4.6 / 249 |
| Llama 3.2 3B Q4 (llama.cpp Q4_0) | 64 / 803 | 69 / **865** | **99** / **985** | 21 / 394 |
| Llama 3.2 3B Q4_K_M (llama.cpp) | 64 / 568 | — | — | — |
| Llama 3.1 8B Q8 | 19 / 274 | 21 / 437 | **39** / **680** | 3.6 / 140 |
| Llama 3.1 8B Q4 (llama.cpp Q4_K_M) | 29 / 246 | 34 / **397** | **59** / **542** | 0.8 / 132 |

(decode / prefill 512 tokens)

- llama.cpp does not scale to the second socket on this machine: its per-op barriers collapse as
  soon as the thread count exceeds the truly idle cores, and the weights are not split by node.
- Prefill: Q8 weights are repacked at load into 4-row blocks interleaved by 4 bytes, so one `sdot` by
  lane gives 4 rows at once (no horizontal reductions) and each weight vector serves 8 tokens; every
  socket quantizes the block inputs into its own memory. Q4 uses the same layout; its min term is
  folded out of the inner loop (sum (s u + m) x = s sum u x + m sum x, with sum x once per token).

Quality, Llama 3.2 3B, mean KL divergence to full precision on the same 4 x 512-token chunks
(second half of each chunk scored, as `llama-perplexity --kl-divergence`):

| | mean KL | top-1 agreement |
|---|---|---|
| llama.cpp Q8_0 | 0.0004 | 99.5% |
| numallama Q8 | 0.0006 | 99.0% |
| **numallama Q4** (asymmetric g32 fp16 s+m, imatrix, wv in Q8; 2.22 GB) | **0.013** | **96.1%** |
| llama.cpp Q4_K_M | 0.015 | 95.8% |
| llama.cpp Q4_0 | 0.025 | 94.6% |
| numallama old Q4 (symmetric g32, no imatrix; removed) | 0.038 | 92.8% |

Where the 4-bit error came from (fp32 forward with fake-quantized weights, 2 chunks):
asymmetric g32 alone 0.031; + importance matrix 0.014; + wv in Q8 0.012; + w2 in Q8 in 1/3 of the
layers 0.010 (~5.4 bit/weight); + all w2 in Q8 0.008 (~5.9 bit/weight). The importance matrix is
collected by the engine on the Q8 model (`-m imatrix`, ~30 s for 19k tokens), per expert for MoE.

## Mixture of experts: Qwen3-30B-A3B-Instruct-2507 (abliterated), Q8 (32 GB)

128 experts x 768, 8 active per token (~3.3B active parameters out of 30.5B), 48 layers.

| | 1 socket | 2 sockets |
|---|---|---|
| decode | 47 tok/s | **71 tok/s** |
| prefill 512 tokens (batched, grouped by expert) | — | 523 tok/s (token by token: 69) |
| **Q4** (19.4 GB, imatrix): decode | **67 tok/s** | **91 tok/s** |
| Q4: prefill 512 tokens | — | 505 tok/s |

Q4 vs Q8 on the same 43 tokens: top-1 agreement 93%, KL 0.018.

Logits vs an fp32 numpy reference: top-1 97.7%, mean KL 0.0075.

## Hybrid linear attention: Qwen3.8-27B (abliterated), dense

64 layers: 48 Gated DeltaNet (16 k / 48 v heads x 128, conv 4) + 16 gated full attention (24 / 4 heads x 256,
RoPE on 64 dims), hidden 17408, vocab 248k. Recurrent state ~150 MB, independent of the context.

| | Q8 (28.6 GB) | Q4 (18.0 GB, imatrix) |
|---|---|---|
| decode, 1 socket | — | 10 tok/s |
| decode, 2 sockets | 11.4 tok/s | **17.2 tok/s** |
| prefill 512 tokens, 2 sockets | 195 tok/s | 172 tok/s |

Q8 vs an fp32 numpy reference (tests/ref_qwen35.py, 26 tokens): top-1 100%, KL 0.0005; batched prefill
last token KL 0.0012. Q4 vs Q8: top-1 96%, KL 0.006. Speculation is off for hybrid models (the recurrent
state cannot drop rejected drafts); a dense 27B reads all weights per token, hence ~5x slower than the
30B-A3B MoE, but it answers better.

## Synchronization rework (decode, 2 sockets)

| Step | Qwen3-30B-A3B | Llama 3.2 3B Q8 | Llama 3.1 8B Q8 |
|---|---|---|---|
| before | 55 | 67 | 35 |
| barrier state in node-bound pages, hierarchical cross-node level, padded ThreadBuf | 69 | | |
| q/k/v, qk-norm, attention with node barriers only; router replicated per node | 69 | | |
| flag-tree barriers (no atomic RMW), two-level release | 70 | | |
| prefetch of the next phase's first rows during barrier waits | **71** | **76** | **39** |

Barrier latency (`RUNQ_BARBENCH=1`): node 2.0 -> 1.1 us, cross-socket 7.4 -> 1.7 us.
Sampling (temperature + top-p over 152k tokens) moved onto the pool: 1.3 ms -> 0.09 ms per token.

Tried and dropped: smaller work chunks / byte-capped chunks (shorter row runs stream worse),
finer attention chunks (16 positions), Megatron-style column split of wo/w2 (two cross-socket
barriers per layer instead of four: +1% on MoE, 0 on 8B, +10 GB and +5 s load).
The remaining waiting (~30%) is load skew at ~12 barriers per layer, not barrier latency.

## Prefill rework (512 tokens, 2 sockets, tok/s)

| Step | 3B Q8 | 8B Q8 | Qwen3-30B-A3B |
|---|---|---|---|
| before | 492 | 214 | 215 |
| packed Q8 layout + sdot-by-lane 4x8 kernel | 611 | 382 | 249 |
| MoE: dynamic expert chunks, parallel gather | | | 328 |
| q+k+v as one balanced static range | 729 | 425 | 333 |
| per-node quantized block inputs | **1172** | **680** | **523** |

Tried and dropped: dynamic 16-row chunks for the batched matmuls (every chunk re-reads the inputs of
all tokens: 8B 382 -> 301).

## Speculative decoding, Qwen3-30B-A3B (MoE), 2 sockets, greedy, prompt lookup (`-k 8`)

| Task | plain | speculative | drafts accepted |
|---|---|---|---|
| fix typos in a Russian paragraph | 70 | **144–152** | 104/117 |
| print a file read with `cat` (context 1.6k) | 48–51 | **100** | 454/581 |
| free-form story | 68 | 67 | 15/75 |

A MoE verification step costs more than a dense one (each token brings its own experts): B=2 1.2–1.4x,
B=4 1.8–1.9x, B=8 2.5–3.3x of a single step. The draft length adapts (doubling on full acceptance,
shrinking to the accepted length), and after a fully rejected draft the lookup pauses for 2, 4, ... 32
tokens, so free text loses almost nothing.

## Speculative decoding (Llama 3.1 8B Q8, 2 sockets, greedy)

| Task | plain | prompt lookup (`-k 8`) | + draft Llama 3.2 1B (`-d`) |
|---|---|---|---|
| fix typos in a paragraph | 32.9 | **69.1** | — |
| summarize a tool output (Qwen2.5 7B) | 30.8 | **54.1** | — |
| free-form story | 31.1 | 31.9 | **46.5** |

Cost of one multi-token step relative to a single-token step (8B, 2 sockets):
B=2: 1.03x, B=4: 1.04x, B=8: 1.11x, B=16: 1.37x. The multi-token attention splits the context into
chunks (online-softmax merge): at 1.6k context, 3B, B=2 costs 1.05x instead of 2.8x.

## Profile (Llama 3.2 3B Q8, 1 socket, 96 threads)

Decode is memory-bound: ~155–180 GB/s of weights per socket, IPC ~0.7, ~78% backend stall cycles.
One core alone reaches ~8.5 GB/s (368 ms/token, 99.5% in matmul). Use `make prof` for a per-phase,
per-node breakdown.
