# numallama — agent guide

CPU-only LLM inference engine in C (AArch64/NEON first, x86 works). Single translation unit:
`src/runq.c` only includes topical headers **in dependency order** — each header uses only what
earlier headers define. There is no build system beyond one `cc` invocation; a new `.h` must be
added to the include list in `src/runq.c` (and it is picked up by the Makefile's `src/*.h` dep).

## Layout (include order = dependency order)

| file | contents |
|---|---|
| `src/core.h` | `Config`, model/state structs, quantized tensors, checkpoint loading |
| `src/tools.h` | tool definitions/execution (`-f config/tools.json`); pulled in by `core.h` |
| `src/weights.h` | NUMA placement (`place_*`, `bind_range`) and weight repacking (`pack_q8`) |
| `src/inkling.h` | Inkling-2-Mini INK2 container loader (mmap + JSON manifest), Q8/Q4 packing, 48-slot state |
| `src/model_inkling.h` | Inkling-2-Mini text forward: embedding/head, depth mix, sconv, relative attention, loop controls, MoE |
| `src/inkling_server.h` | OpenAI `/v1/chat/completions` for `.ink2`: SSE + reasoning_content, one-turn-cached session, metrics, graceful shutdown |
| `src/pool.h` | thread pool, tree barriers, per-node reductions, `rmsnorm`, `-DPROFILE` hooks |
| `src/quant.h` | int8/int4 quantization, NEON matmul kernels (single-row and batched tiles) |
| `src/kv.h` | KV cache formats: f32 and int8 with per-group scales (`-K`), store/dot/axpy |
| `src/model_hybrid.h` | Qwen3.5/3.8 decode step: Gated DeltaNet linear attention + full attention |
| `src/glm.h` | GLM-5.3 decode step (KDA + MLA) |
| `src/model_dense.h` | dense Llama/Qwen decode step (`forward_thread`) |
| `src/runtime.h` | CPU/NUMA thread placement, pool workers, `build_transformer`, `use_model` |
| `src/batch.h` | batched prefill infra (`BT`), batched matmuls, MoE FFN, hybrid batched prefill |
| `src/glm_batch.h` | GLM batched prefill (+ per-position logits for speculation) |
| `src/decode_batch.h` | dense batched prefill, multi-token decode, `forward` / `forward_all` |
| `src/tokenizer.h` | BPE tokenizer |
| `src/sampler.h` | argmax / temperature / top-p, pool-parallel sampling |
| `src/spec.h` | pool-parallel memcpy, weightless speculative commit (`hybrid_spec_commit`) |
| `src/chat.h` | chat history, templates, tool calls, `chat_reply` (CLI + server decode loop) |
| `src/server.h` | OpenAI-compatible HTTP server (`-m server`): request loop, system-prefix slots, session journal; included from `runq.c` after the `sess_*` blob helpers (and skipped in `-DTESTING` builds) |
| `src/serve_batch.h` | continuous-batching scheduler for dense/MoE (`-J`): poll loop, resident per-session KV caches (LRU RAM budget, periodic dumps), batched `forward_sessions` decode; `server_serial` stays for hybrid/GLM/`-J 1` |

## Rules that keep the single-TU split working

- Define before use: a function/struct may only reference earlier headers (or forward-declare
  at the top of the earlier header, as `runtime.h` does for the pool thread entry points).
- Headers have no include guards on purpose — they are included exactly once, from `runq.c`.
- Everything is `static` except what tests need (`forward`, `forward_batch`, ...).
- Tests (`tests/*.c`) `#include "../src/runq.c"` with `-DTESTING` (drops `main` + server).
  Run `make tests` and at least `tests/check_batch <model> 192 64` after touching kernels (also with
  `RUNQ_KV=f32`), and `tests/check_seqs <model> 64` after touching `forward_sessions`/the KV store.
  `tests/check_inkling <model.ink2>` validates the INK2 container loader.
  The numpy reference checks (`tests/ref_*.py`, `dump_logits`) expect
  `RUNQ_KV=f32`: the default int8 KV adds ~0.005 KL and would break tight tolerances.
- Batched decode: `BT.tokpos` / `BT.tokv` give per-token positions and KV caches; with them NULL the
  same `forward_multi_thread` is the speculative path of one sequence. Hybrid/GLM never enter it.

## Conventions

- Build: `make` (`-Ofast -mcpu=native -pthread -Wall`, must stay warning-free). `make portable`
  builds `runq_portable` with `-march=armv8-a` and no SDOT for ARMv8.0 boards.
- Threading: one global pool `P`; work is split by row ranges per NUMA node
  (`node_rows`/`grab_rows`), sync via tree barriers — no locks on the hot path.
- Decode is bandwidth-bound: batching (many tokens per weight read) is the main lever;
  see `matmul_rows_batch`, `forward_batch`, `forward_all` (speculative verify). The KV cache is
  int8 by default (`-K f32` opts out; `kv.h`): K, V and the per-head queries are int8 with per-64
  group scales, scores are `sdot` products, V is dequantized once per position and FMA'd into the
  query heads.
- Documentation (README, docs/, AGENTS.md) and code comments are written in English.
- Token embeddings stay quantized: use `embed_row` / `embed_row_slice` (`quant.h`); never materialize
  a full fp32 embedding table (it is 1.2-5 GB on these vocabularies).
- Commit style: short imperative one-liners; never add Co-Authored-By trailers; commit with
  `--no-gpg-sign`.
