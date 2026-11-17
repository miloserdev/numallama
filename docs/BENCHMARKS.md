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
| Llama 3.2 3B Q8 | 36 / **512** | 46 / 489 | **64** / 490 | 4.6 / 249 |
| Llama 3.2 3B Q4 (llama.cpp Q4_0) | 64 / **803** | 69 / 430 | **83** / 490 | 21 / 394 |
| Llama 3.2 3B Q4_K_M (llama.cpp) | 64 / 568 | — | — | — |
| Llama 3.1 8B Q8 | 19 / **274** | 20 / 183 | **33** / 214 | 3.6 / 140 |
| Llama 3.1 8B Q4 (llama.cpp Q4_K_M) | 29 / **246** | 32 / 181 | **46** / 223 | 0.8 / 132 |

(decode / prefill)

- llama.cpp does not scale to the second socket on this machine: its per-op barriers collapse as
  soon as the thread count exceeds the truly idle cores, and the weights are not split by node.
- llama.cpp is faster at prefill (repacked 4x4 weight layout, tuned GEMM kernels).

Quality, Llama 3.2 3B, mean KL divergence to full precision on the same 4 x 512-token chunks
(second half of each chunk scored, as `llama-perplexity --kl-divergence`):

| | mean KL | top-1 agreement |
|---|---|---|
| llama.cpp Q8_0 | 0.0004 | 99.5% |
| numallama Q8 | 0.0006 | 99.0% |
| llama.cpp Q4_K_M | 0.015 | 95.8% |
| llama.cpp Q4_0 | 0.025 | 94.6% |
| numallama Q4 (symmetric, groups of 32) | 0.038 | 92.8% |

## Mixture of experts: Qwen3-30B-A3B-Instruct-2507 (abliterated), Q8 (32 GB)

128 experts x 768, 8 active per token (~3.3B active parameters out of 30.5B), 48 layers.

| | 1 socket | 2 sockets |
|---|---|---|
| decode | 36–43 tok/s | ~41 tok/s |
| prefill (batched, grouped by expert) | — | 204 tok/s (token by token: 55) |

Logits vs an fp32 numpy reference: top-1 97.7%, mean KL 0.0075. Decode is barrier-bound
(~10 synchronizations per layer x 48 layers): ~40% of the time is spent waiting; reducing the
number of cross-socket synchronizations is the next step for MoE.

## Speculative decoding (Llama 3.1 8B Q8, 2 sockets, greedy)

| Task | plain | prompt lookup (`-k 8`) | + draft Llama 3.2 1B (`-d`) |
|---|---|---|---|
| fix typos in a paragraph | 32.9 | **69.1** | — |
| summarize a tool output (Qwen2.5 7B) | 30.8 | **54.1** | — |
| free-form story | 31.1 | 31.9 | **46.5** |

Cost of one multi-token step relative to a single-token step (8B, 2 sockets):
B=2: 1.03x, B=4: 1.04x, B=8: 1.11x, B=16: 1.37x.

## Profile (Llama 3.2 3B Q8, 1 socket, 96 threads)

Decode is memory-bound: ~155–180 GB/s of weights per socket, IPC ~0.7, ~78% backend stall cycles.
One core alone reaches ~8.5 GB/s (368 ms/token, 99.5% in matmul). Use `make prof` for a per-phase,
per-node breakdown.
