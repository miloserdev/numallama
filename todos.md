# numallama todos

Ordered by value/effort. Areas: memory, server, quality, speculation, long context,
new architectures, low-end hardware.

| # | Task | Area | Why / expected effect | Status |
|---|---|---|---|---|
| 1 | Dequantize token embeddings per row instead of materializing an fp32 table | memory | Drop 1.2-5.1 GB RSS and startup dequantization (Qwen3.8-27B vocab 248k x 5120 = 5.1 GB); only the current token's row is needed | done: qwen38 RSS 17.6 GB, qwen3 19.0 GB; shared-classifier packed layout handled |
| 2 | System-prefix slot floor (`SRV_SYS_MIN = 2048`) | server | Stop short side requests (opencode title, ~560 tok) from taking/evicting an agent preamble slot; keep the 11k prefix warm | done: 331-token prefix takes no slot; 2836-token prefix restores in-process and after restart |
| 3 | Hybrid Q4: keep DeltaNet projections (`li`, `lo`) in Q8 | quality | llm-in-c ablation: Q4 on `in_proj_qkv/in_proj_z/out_proj` breaks decisions; Q8 restores them. Our Q4 mask 507 quantizes `li/lo` | done, accepted as a quality variant: `/var/models-bin/qwen38_27b_abl_q4_liq8.bin` 20.40 GB (+2.43 GB, +13.5%), q4mask 123, maxerr 0.133. HF fp32 on 87 positions: **liq8 KL 0.0087 / top1 93.1% / max 0.0324** vs Q4 KL 0.0117 / 94.3% / 0.0418. Q4 vs liq8 answers identical on 3 decision prompts; 8k decode 12.2 vs 12.0 tok/s (no measurable cost). Standard Q4 stays the default; reproduce with `--q4 wq,wk,wo,w1,w2,w3` |
| 4 | Sampler `min_p`, `presence_penalty`, `frequency_penalty` | API | OpenAI-client compatibility; greedy stays penalty-free for exact speculation | done: `min_p` in the pool path, penalties counted within a completion; tests return coherent text |
| 5 | Tiled/vectorized prefill attention (block Q x KV, skip masked tiles) | prefill | `batch.h` prefill attention is the last scalar hot loop; long agent prompts | evaluated, no gain: porting dense chunked online-softmax to hybrid was neutral on 8k prefill (111 vs 112 tok/s at 128 threads), correct at 16-64 threads, so it was reverted to the scalar path; kept the larger `mpart` capacity (`BATCH_MAX x heads x (hs+2)`) to stop dense prefill from overflowing partials with few threads. The default i8 score dot already uses SDOT, so tiling alone cannot help |
| 6 | Chunked/interleaved prefill for continuous batching | server | A new long prompt currently blocks active streams; 512-token slices between decode steps | deferred |
| 7 | MTP for Qwen3.8-27B (exporter + second KV per session + adaptive depth) | speculation | llm-in-c blueprint: 1.5x aggregate, 2.2-2.4x on code; FC [h,2h], 3 norms, layer 64, 4-gram lookup, restricted draft vocab | researched (see notes): go, but it is a multi-stage feature: one extra full-attention layer (209 MB Q4) + 29.6 MB fc/norms, a second per-session KV, target-hidden capture during prefill, verify via the existing hybrid batched pass, adaptive depth + lookup |
| 8 | TurboQuant KV mode (`tq4`, long context / low RAM) | long context | 4-bit KV with rotation + Lloyd-Max codebook (~7x KV); after measuring simple q4 need | researched (see notes): TurboQuant 4+4-bit is ~2x smaller than our i8 (not 7x): ~8.4 vs ~16.5 bits/value; vLLM `4bit_nc` = 3.8x vs f32 with +2.71% PPL, `k8v4` = 2.6x with +1.17%. Do a simple q4-KV A/B before the full rotation/codebook path |
| 9 | Continuous batching for hybrid/GLM (per-session recurrent / MLA state) | server | Extend `-J` beyond dense/MoE; Qwen3.5/3.8 and GLM are serial today | deferred |
| 10 | Speculative decoding inside the batched step | server | Per-session draft chains (variable k) in one verify pass; MoE needs batch-aware verify | deferred |
| 11 | Turn slot + prefix-restore metrics | server | Keep the state at the previous turn boundary for retries/regenerate; log restore/save/prefill/TTFT | done: per-session recurrent turn slot (token hash, 4 GB LRU) for retries/regenerates, survives a disk restart; serial and dense-CB logs now print `restore/prefill/ttft` (serial also `save`); a retry restores at 23/24 cached tokens (prefill 725 -> 207 ms, disk restore 252 ms) with identical output; a branch now forces the next dump so the session is not lost to the dump throttle |
| 12 | NUMA binding for per-session i8 KV caches | memory | i8 caches are currently unbound; interleave/bind per node | done: `kv_cache_bind` applies the global-cache policy to every per-session cache (i8 and f32), binding each KV head group to the node that runs its attention; verified 75k `bind:` mappings during a live session; CB J4 110.3 tok/s vs 103.7-109.2 before (no regression) |
| 13 | KV offload to a file for very long sessions | long context | Spill a session cache when RAM is short; sessions already dump/restore | optional |
| 14 | GGUF import | formats | Postponed by request; converter `gguf -> our .bin` remains the plan | on hold |
| 15 | Lazy mmap of remaining large tensors | memory | Embeddings covered by #1; other rarely touched rows could stay page-cache-only | optional |
| 16 | Inkling-2-Mini support (text-only): loader/exporter, tokenizer, forward, batching/KV, Q8/Q4, publish | new architectures | Official GGUF exists and llama.cpp already runs it, but numallama has no Inkling ops; Apache-2.0; Mini 26.4B/3.4B active | Phase 2 + M1-M5 + server v2: full text forward (top-1 Paris, logit 27.37 vs 27.31, KL 0.059), packed-weight cache, raw/chat CLI, OpenAI server with SSE + `reasoning_content`, keep-alive/abort, one-session cache (prompt+state+logits), stop/min_p/penalties/usage, `/metrics`, graceful shutdown. Quality: `--experts q8` container (27.7 GB) validated, generic Q4/Q8 expert dispatch; oracle top-1 Paris, top-10 10/10 for both, Q8 KL 0.0535 vs Q4 0.0594, Q4-vs-Q8 KL 0.0083; greedy continuations agree ("Paris." then A/B/C choices, tails differ from IQ4_NL oracle). Next: single-thread speed, publish artifacts (HF upload needs separate approval), tools/multi-session |
| 17 | Low-end profile for microcomputers / weak ARM boards | low-end hardware | Must run on 2-8 GB SBCs with few cores: mmap weights, portable build, no-SDOT fallback, smaller defaults, RAM-aware `-J` | in progress: `make portable` -> `runq_portable` (`-march=armv8-a`, no SDOT; scalar fallback verified: check_kv OK, check_batch argmax equal). Auto `-T 0` now uses idle-1 workers on <=8-core nodes (a 4-core board gets 3, not 1) and idle-4 with the 96/node cap on big nodes. Remaining: lazy mmap (#15), RAM-aware `-c`/`-T`/`-J`, low-end benchmark matrix, optional single-binary runtime dispatch |
| 18 | NUMA-sharded replicas + session router | server/perf | Decode cannot scale linearly with 256 cores; shard by NUMA node, pin replicas, keep sessions sticky | v1 done: `scripts/serve_replicas.sh` + `scripts/replica_router.py` (round-robin assignment, sticky `session_id`, SSE relay). Measured Qwen3-30B Q4: single J16 229 / J32 238; 2xT64 404; 4xT32 516; **8xT16 588**; 16xT8 365; router 4xT32 503 vs 516 direct (~3%). Optimum: 8 replicas x T16 x J16; next: auto replica/J sizing for other machines |
| 19 | OpenAI server production hardening | server | Metrics, graceful shutdown, request limits, Inkling reasoning/cache/limits | done v1: main and Inkling servers expose `/metrics`; SIGINT/SIGTERM stop the accept loops; Inkling separates `reasoning_content`, caches one multi-turn session (prompt+state+logits, verified hit), streams keep-alives/abort, supports stop/min_p/penalties/usage, and rejects prompts over `INKLING_MAX_PROMPT` (default 4096). Turn latency: heartbeat stop is polled every 10 ms instead of joining a 2 s sleep (streaming TTFT 2.0 s -> 0.08-0.24 s); dense-CB session KV caches stay resident across turns with an LRU RAM budget and periodic dumps (16k follow-up TTFT 1.22 s -> 0.17 s). Next: drain active sessions on shutdown, per-replica metrics aggregation, tools for Inkling |

## Research notes: TurboQuant KV (item 8)

Algorithm (vLLM implementation, Google TurboQuant): keys get a Hadamard rotation, then a
per-coordinate Lloyd-Max scalar quantizer (3 or 4 bits) plus a per-vector fp16 norm; `norm
correction` renormalizes the centroid vector to unit norm before the inverse rotation. Values use
uniform 4-bit quantization with an fp16 scale and zero. QJL is deliberately omitted (it amplifies
softmax variance). vLLM presets measured against f32: `k8v4` 2.6x KV savings, +1.17% PPL;
`4bit_nc` 3.8x, +2.71%; `k3v4_nc` ~3.5x, +10.63%; `3bit_nc` 4.9x, +20.59%. Aggressive presets
skip the first/last two attention layers; this is disabled for hybrid models.

The 7x figure is versus f32 for a 3-bit preset. In numallama terms our default i8 is already
~16.5 bits/value (8-bit K and V plus scales); a 4+4-bit TurboQuant slot is ~8.1-8.5 bits/value
(4-bit indices + 2 B norm / 128 dims for K, 4-bit + 4 B scale/zero per 128 for V), i.e. roughly
2x less KV memory, not 7x. Head_dim 256 is a power of two, so a 256-point Hadamard transform fits.

Port sketch: (1) a simple uniform q4 KV first (4-bit K/V + per-64 scales) as the cheap A/B; if it
is not enough for long context, (2) add a `TQ4` mode with per-head rotation, a fixed 16-entry
Lloyd-Max codebook, fp16 K norms and uniform 4-bit V. Attention can dequantize K per score (or
use a codebook dot) and V once per position, as the i8 path does. Measure 30k context on Qwen3.8
and Qwen3-30B: KV bytes, decode tok/s, KL to `-K f32`. Risks: the rotation/Lloyd-Max dot raises
CPU cost, codebook selection must be per head distribution, and hybrid full-attention layers are
only 16 of 64 (boundary skipping is not useful).

## Research notes: Qwen3.8-27B MTP (item 7)

Source: llm-in-c `models/qwen3.8-27b/targets/apple-m3-pro` (Metal runtime) and the 4-bit
`mlx-community/Qwen3.8-27B-MTP-4bit` checkpoint it packs.

Shape of the feature: MTP is a **single extra full-attention layer at index 64** plus an fc
projection `[5120, 10240]` (concatenated embedding + hidden, affine Q4 g64 with BF16
scales/biases), and three RMS-norm vectors (pre-fc embedding, pre-fc hidden, final MTP norm).
The embedding table and LM head are shared with the target. Images: 209,436,672-byte layer +
29,556,736-byte extras. Drafting chains the current token through that layer (its own KV),
verifies the pending token plus up to 7 drafts in one batched target pass, and rolls the target
recurrent state back on a partial accept.

Measured on an M3 Pro (five workloads, greedy): 1.50x end-to-end aggregate, 2.10-2.39x on code,
1.17-1.33x on prose; 4 of 5 cases token-identical. Adaptive depth uses an acceptance EMA:
>0.85 -> 7, >0.70 -> 5, >0.55 -> 3, >0.40 -> 2, else 1. Below EMA 0.70 a four-gram
context lookup may replace the MTP chain (up to 7 followers, 16-step cooldown after zero accept).

Port sketch for numallama (does not touch the target model format):
1. New optional MTP companion file (one `ARCH_QWEN38_MTP` layer + fc/norms), loaded with a new
   flag; absent by default, so existing models are unchanged.
2. Per-session MTP KV (~8 MB i8 at 8k for 4 KV heads) and a per-session draft chain. Capture the
   target final hidden during prefill and feed the MTP layer to fill its KV.
3. Verify with the existing hybrid batched pass (`forward_all` / spec commit); reuse
   `hybrid_spec_snapshot` for the GDN rollback on partial accepts.
4. Adaptive depth and the existing `ngram_draft` lookup; add the restricted draft vocabulary only
   if measurements need it.
5. Measure on code and prose against prompt-lookup speculation and plain decode.

Risks: the published MTP is trained for Qwen3.8-27B, while the local target is an abliterated
variant (draft quality must be measured, not assumed); prefill gains no MTP benefit and pays one
extra layer; the CPU verify cost differs from Metal, so the best depth tiers must be re-measured.

## Track: Inkling-2-Mini (text-only)

Why this is a port, not a conversion: GGUF is only a container (weights + metadata); the engine
still needs a computation graph for the architecture. llama.cpp already has that graph for
Inkling-2, numallama does not, and it cannot read GGUF at all (#14). Exporting weights alone
produces a file nothing in numallama can execute. Existing model formats stay untouched: the new
architecture gets its own `ARCH_INKLING2` and appended config fields; old `.bin` files keep
loading as before.

Verified facts: Mini = 26.4B total / 3.4B active (5.4B effective with 3 loop passes), Apache-2.0,
19.5 GB HF safetensors; official GGUFs IQ4_NL 17.63 GB / Q8 27.82 GB / BF16 52.23 GB. The local
llama.cpp build 9048bad only knows `inkling2`; the Mini GGUF uses `inkling2_loop`, so it rejects
the file. Upstream origin/master (9fa7a33) has `inkling2_loop`; a fresh tree is being built in
`/dev/shm/llama-src` + `/dev/shm/llama-build` because `/ai` has only ~4 GB free. Small
(594B/17B active, Q4 ~300 GB) does not fit the current disks (246 GB free in `/dev/shm`,
27 GB in `/var`).

Phases:
1. **Reference/oracle (done).** `Inkling-2-Mini-IQ4_NL.gguf` is downloaded to
   `/dev/shm/inkling2` (17,627,940,768 bytes, GGUF). Upstream 9fa7a33 was built in
   `/dev/shm/llama-src` + `/dev/shm/llama-build` (`llama-completion`, raw `-no-cnv`); the old
   9048bad build cannot load `inkling2_loop`. Smoke run: "The capital of France is Paris.";
   load 22 s, prompt 10.9 tok/s / 5 tokens, decode 5.38 tok/s on 16 threads while the HF
   reference was running. Clean tok/s/RSS still to be recorded.
2. **Exporter + tokenizer (started).** Source: `thinkingmachines/Inkling-2-Mini`, Apache-2.0,
   sha `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, single 19.5 GB `model-00001-of-00001`
   safetensors (downloading to `/dev/shm/inkling2_hf`; config/tokenizer already there).
   Text config: 24 layers, hidden 2304, 24/8 heads, head_dim 128, vocab 201024 padded /
   200058 unpadded, intermediate 1152, MoE 128/8, route_scale 8, gate/up softcaps 4/25,
   d_rel 16, rel_extent 1024, sliding window 1024, sconv kernel 4, attn-res block 6,
   qk clip 22.627, rel clip 8, log-scale floor 8192/alpha 0.1, final softcap 30,
   loop core 6..18 x3 with gate_rank 64, 20 sliding layers. Weights are ModelOpt **NVFP4**:
   experts store `w13_weight`/`w2_weight` plus E4M3/UE8M0 scales and amax/denominator; the
   upstream `conversion/inkling2.py` has the exact dequant and name mapping. Tokenizer is BPE
   with the Llama-3 pre-tokenizer regex (our encoder matches), 60 added tokens; the chat
   template uses `<|message_user|>` style specials, so a special-token map is required.
   Done: `scripts/export_tokenizer.py` produced `/dev/shm/inkling2_hf/inkling2_tokenizer.bin`
   (201024 entries) and our encoder matches `llama-tokenize` exactly on a mixed
   English/Russian/Chinese/code sample (36/36 ids). The model header must set
   `special_start = 199998`; the chat-template branch for Inkling is still to be written.
   Safetensors header (complete file, 1244 tensors): 994 BF16, 154 F32, 48 F8_E4M3,
   48 U8. `model.llm.embed.weight` and `unembed.weight` are both [201024, 2304].
   Expert quantization differs by layer: **0..19 are NVFP4** (U8 weights
   [128, 2304, 1152]/[128, 2304, 576], F8_E4M3 scales [128, 2304, 144]/[128, 2304, 72],
   F32 amax [128] + one denominator); **20..23 are MXFP8** (F8_E4M3 weights
   [128, 2304, 2304]/[128, 2304, 1152], U8 scales [128, 2304, 72]/[128, 2304, 36]).
   Full-attention layers are 5, 11, 17, 23 (q/k/v input norms and dynamic conv taps);
   the other 20 are sliding-window. The loop core is 6..17 with controllers on those 12
   layers; layer 23 has none. Oracle: `/dev/shm/llama-build/bin/inkling_oracle` (libllama)
   wrote `/dev/shm/inkling2/logits_oracle.bin` (200058 floats) for
   "The capital of France is": tokens [371, 151451, 149879, 151194, 149884], top-1
   ` Paris` id 152717 logit 27.308.
   Dequant core validated from the real bytes (numpy, no torch): NVFP4 layer 0 w13
   dequantizes to [128, 2304, 2304], all finite, std 0.0208; MXFP8 layer 23 same shape,
   all finite, std 0.0208.
   Exporter implemented: `scripts/export_inkling.py` writes `INK2JSON` (magic + JSON manifest
   padded to 1 MiB + absolute tensor offsets): source names without `model.llm.`, Q4 experts
   as per-expert blocks, Q8 matrices, F32 norms/clips/convs/queries; vision/MTP skipped.
   Full export done: 694 tensors, 17,003,447,296 bytes at
   `/dev/shm/inkling2/inkling_mini_q4.ink2`; Q4 round-trip on layer 0/20 experts is finite with
   max error 0.008-0.014. C loader `src/inkling.h` maps the file, parses the manifest and
   resolves tensors by name; `inkling_pack_2d` copies a row-major Q8/Q4 tensor and repacks it into
   the engine's 4-row layout (GS=64); `tests/check_inkling` prints 24/48 layers/slots, 694
   tensors, and verifies Q8/Q4 matmuls against a direct row-major decode with max|diff| 0.00000.
   `InklingState` allocates the 48 KV slots (i8 + per-64 scales), conv history for K/V/attn/FFN,
   work buffers and logits; slot -> physical layer mapping verified (0,23,24,47,36 -> 0,23,6,17,6).
   Next: implement the layer math in `src/model_inkling.h`; no existing `.bin` format is touched.
3. **Forward v1.** `src/model_inkling.h`: one-token decode first, then prefill. Looped layers,
   relative attention, sliding window, short conv on K/V and post-conv, attention-residual
   blocks, sink, clipping/log scaling, MoE. Reuse existing matmuls/RMSNorm/SwiGLU/MoE routing.
   The upstream graph is `src/models/inkling2.cpp` (481 lines) + `inkling2-loop.cpp` (123);
   port it op by op and validate logits against llama.cpp (KL / top-1) on fixed prompts.
   This is the multi-week part: 24 layers with depth mixing, 4-tap conv state, relative bias
   `[16,1024]`, loop controls, and 128-expert routing are all new to numallama.

   Forward port checklist (M1 + M2 primitives done): `src/model_inkling.h` has Q8 embedding
   rows, RMSNorm via the pool helper, and the packed Q8 LM head with final softcap; a synthetic
   hidden vector gives the same top-1 (178515) and softcapped logit (3.264514) as a numpy
   reference. Depth mix, the causal 4-tap per-channel convolution, the L2 clip, the
   relative-bias/sink attention with sliding-window masking, and the sigmoid top-8 Q4 MoE
   (route_scale, situ 4/25) are implemented and exercised on real layer-0 weights
   (`check_inkling`, finite output). The full SWA layer (norms, projections, K/V sconv,
   clips, attention+gate, post-conv, residual/block push, depth mix, MoE, FFN post-conv)
   is assembled for layer 0. The same function has the full-attention branch: three separate
   Q/K/V input norms, dynamic short-conv taps from `weight_proj`, log-scaled positions and
   no sliding window; layer 5 runs finite. Loop passes are complete: prelude 0..5, three
   passes over 6..17 with separate slots 24..47, cross-pass bank/messages, down/gain/forget
   controllers, then post 18..23 and the output mix. The full text forward now runs: oracle
   prompt gives top-1 Paris (152717, logit 27.37 vs 27.31), KL 0.059, corr 0.979, top-10
   overlap 10/10. The raw CLI path in `runq` auto-detects INK2JSON and generates text
   (" Paris."); a one-turn chat path renders the Inkling message/effort template, hides
   service tokens and stops at `<|content_model_end_sampling|>`. Packed Q8/Q4 weights are
   cached (no per-token repack). Remaining: server/OpenAI path, quality/speed tuning, publish.

   Remaining layers checklist:
   - **Slots (state done):** 48 cache/conv slots = 24 physical + 2 extra passes over 6..17;
     prelude 0..5, core 6..17 x3, post 18..23. Physical weights are shared, KV/conv state is per
     slot. `layer_phys(slot) = slot < 24 ? slot : 6 + (slot - 24) % 12`; `InklingState` allocates
     i8 KV + per-64 scales, conv history and work buffers.
   - **Layer:** depth mix (RMSNorm the earlier block outputs + current, learned query softmax,
     mix back), optional loop-control message (RMSNorm concat(cur,msg) -> down -> tanh gain /
     forget -> cur + gain*msg - forget*cur), q/k/v input norms (full) or one attn norm (SWA),
     q/k/v/rel projections, 4-tap causal sconv on K/V (state + dynamic taps on full layers),
     L2 clips on q/rel, log-scaled positions on full layers, relative bias `[16,1024]`,
     per-head sink logit, attention gate, post-conv; MoE block does the same depth/control mix,
     ffn norm, router (sigmoid, top-8, route_scale), situ gate/up softcaps, w2, ffn post-conv.
   - **MoE:** w13 rows [gate; up], dequantized/quantized per expert; store as two arrays for
     the existing `expert_view` kernels. Router bias is `e_score_correction_bias`.
   - **Layout (done):** the exporter is row-major; `inkling_pack_2d` copies and repacks Q8/Q4
     tensors into the engine's 4-row layout at load, and `tests/check_inkling` matches a direct
     row-major decode (0.00000). Apply it per expert block and reuse the packed copies.
   - **Validation:** first a scalar single-token forward vs `logits_oracle.bin` (top-1 and KL);
     then the 48-slot cache, prefill, and finally batching/publish.
4. **Batching + KV.** KV semantics for window/relative attention and conv/loop state; i8 KV;
   a `check_seqs`-style batched-vs-single test.
5. **Quantize + publish.** Q8 first, then Q4; quality harness against the Q8/oracle; publish
   numallama-format weights under Apache-2.0 with NOTICE/attribution. The official GGUF already
   exists, so we publish our format, not another GGUF.

Out of scope initially: vision tower, MTP, Small 594B.

## Track: low-end hardware (microcomputers, weak ARM)

Goal: numallama must be usable on 2-8 GB / 2-8 core boards (Cortex-A53/A72 class, ARMv8.0),
not only on the current large NUMA machine. This track does not reorder the table; it starts
after the current top items.

Required work:
1. **Portable build + runtime dispatch.** `-mcpu=native` is not portable across hosts; add a
   `make portable` target (`-mcpu=generic`) while keeping the optimized build.
2. **No-SDOT fallback.** The Q8/Q4 kernels use ARMv8.2 `SDOT`; A53/A72 lack it and would SIGILL.
   Add a fallback (`vmull_s8`/`vpadalq_s16` or scalar) and select it via `getauxval` HWCAP
   (`HWCAP_ASIMDDP`) at startup.
3. **Memory.** Lazy mmap for all large tensors (#15), sequential access hints, no full-file read;
   log RSS and enforce a startup budget.
4. **Defaults.** Q4 weights + i8 KV on small boards; auto-limit `-c`, `-T` and `-J` from free
   RAM/cores (never 4 full caches on a 4 GB box); KV offload (#13) later if needed.
5. **Validation matrix.** 0.5B/1.5B/3B/7B Q4 on 2/4/8 GB boards: RSS, TTFT, tok/s, no SIGILL;
   scripts under `scripts/bench/lowend/`. Inkling-2-Mini is not a low-end target (17.6 GB at
   IQ4), it is only the new-architecture track.

Provisional acceptance: a 1.5B Q4 model with 4K context on a 4 GB/4-core ARMv8.0 board, and a
7B Q4 model on 8 GB, both within the RAM budget.

## Done

- int8 KV cache by default (`-K f32` opt-out), sessions v2 with head-major dumps.
- Continuous batching for dense/MoE (`-J`, default 4), per-session KV caches, poll loop with incremental readers, per-session streaming/abort, keep-alives.
- System-prefix slots in the batched path (floor #2 done), server-side tools inside the batch.
- Qwen3.8 liq8 Q4 variant (`li/lo` in Q8): HF-fp32 KL 0.0087 vs 0.0117, identical decision answers, no measurable decode cost.
- Hybrid prefill attention A/B: chunked online-softmax tiling was neutral and reverted; the default i8 SDOT path is already the fast one. `mpart` capacity now covers `BATCH_MAX` tokens so low-thread dense prefill cannot overflow it.
- Inkling-2-Mini oracle: official IQ4_NL GGUF loads in upstream llama.cpp 9fa7a33 (`inkling2_loop`) and generates correctly; local `/ai/llama.cpp` 9048bad is too old for this architecture.
- Server turn slot: per-session recurrent state at the prompt end, token-hash checked, 4 GB LRU, dumped/restored with the session; retries and regenerates skip the user prefill. Serial (`restore/prefill/ttft/save`) and dense-CB (`restore/prefill/ttft`) logs carry the timings.
- Per-session KV caches are NUMA-bound per KV head group (same policy as the global cache), for i8 and f32.
- `make portable` builds `runq_portable` for ARMv8.0 boards without SDOT (`-march=armv8-a`, scalar/NEON fallback); kernel and batched-prefill tests pass with equal argmax.
- Low-end auto threads: `-T 0` keeps idle-4 workers on large nodes, but on <=8-core nodes uses idle-1 (4 cores -> 3 workers) instead of collapsing to one; big-machine auto remains 96/node.
- 256-core throughput: NUMA replica launcher + sticky round-robin HTTP router. 8xT16/J16 replicas reached 588 tok/s (vs 229 single J16); 4xT32 through the router 503 vs 516 direct; 16xT8 falls to 365, so 8 x T16 is the measured optimum.
- OpenAI production v1: `/metrics` on main and Inkling servers, SIGTERM/SIGINT graceful stop, Inkling `reasoning_content`+SSE keep-alive/abort, one-session multi-turn cache (hit verified), stop/min_p/penalties/usage, and a clear `INKLING_MAX_PROMPT` 400 guard.
- Inkling-2-Mini INK2 exporter/loader: 694-tensor Q4 container (17.00 GB), Q4 round-trip max error 0.008-0.014, `tests/check_inkling` resolves the mapped tensors, verifies Q8/Q4 4-row packing against a direct decode (0.00000), and allocates the 48-slot KV/conv state with the correct physical-layer mapping.
- Inkling forward M1: Q8 embedding row, RMSNorm and packed Q8 LM head + final softcap; synthetic hidden top-1/logit matches the numpy reference exactly.
- Inkling forward M2 primitives: depth mix, causal 4-tap sconv, L2 clip, relative-bias/sink attention with sliding-window masking, and the sigmoid top-8 Q4 MoE (situ 4/25, route_scale) implemented; layer-0 MoE runs finite.
- Inkling forward M2 SWA layer assembled end-to-end (norms, projections, sconv, clips, attention+gate, post-conv, blocks/depth mix, MoE, FFN post-conv); layer 0 on real weights is finite.
- Inkling forward M3: full-attention branch with three input norms, dynamic conv taps, log-scaled positions and no sliding window; layer 5 runs finite.
- Inkling forward M4/M5: full loop (prelude, 3 passes over 6..17 with 48 slots, bank/messages/controllers, post 18..23, output mix) and raw `runq` generation. Oracle prompt: top-1 Paris, logit 27.37 vs 27.31, KL 0.059, top-10 overlap 10/10; text starts " Paris.".
- Inkling CLI: packed Q8/Q4 weight cache, one-turn chat template (`message_system`/effort/`message_user`, hidden service tokens, stop at `content_model_end_sampling`); raw generation 0.79 s/token at 17.1 GB RSS (still single-thread).
- Inkling server v2: auto-detected `.ink2`, `/v1/chat/completions` non-stream + SSE with separate `reasoning_content`, keep-alive/abort during prefill, one-session prompt+state+logits cache (verified hit), `stop`/`min_p`/penalties/`include_usage`, `/metrics`, API key and graceful SIGINT/SIGTERM shutdown. One request at a time; replicas/router provide concurrency.
- Inkling Q8 experts: `scripts/export_inkling.py --experts q8` writes per-expert Q8 blocks (27.71 GB container) and the loader packs them through the generic queue; oracle prompt top-1 Paris, top-10 10/10, KL 0.0535 (Q4 0.0594), Q4-vs-Q8 KL 0.0083; greedy continuations agree except the tail. `scripts/compare_inkling_logits.py` reproduces the metrics.
- Inkling MoE dispatches experts through `matmul_rows` (Q4/Q8) instead of the Q4-only kernels; Q4 regression checked.
- Server turn latency: keep-alive threads wake on stop within ~10 ms (no fixed 2 s join after prefill); dense-CB session KV caches are resident across turns with a RAM-derived LRU budget (`RUNQ_KV_RESIDENT_MB`, `RUNQ_DUMP_SECS`), removing per-turn restore/dump. Streaming TTFT 2.0 -> 0.02-0.24 s; 16k follow-up 1.22 -> 0.17 s.
- `tests/check_seqs` (batched step vs single decode), `scripts/bench/bench_cb.sh`.
