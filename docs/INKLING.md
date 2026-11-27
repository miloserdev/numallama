# Inkling-2-Mini in numallama

Text-only port of [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(source revision `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, Apache-2.0). This document records
provenance, the `INK2JSON` container, measured quality, run commands and the known limitations of
the port. No model weights are stored in the numallama repository.

## Model

| | |
|---|---|
| Parameters | 26.4B total, 3.4B active (5.4B effective with 3 loop passes) |
| Text backbone | 24 physical layers, hidden 2304, 24/8 heads, head_dim 128 |
| Loop core | layers 6..17, three passes with separate KV/conv slots (48 slots total) |
| Attention | sliding window 1024 on 20 layers, full attention on 5/11/17/23; relative bias 16x1024, per-head sink, 4-tap short conv, log-scaled positions, q/k L2 clips |
| MoE | 128 experts, top-8 sigmoid routing, gate/up softcaps 4/25, route scale 8 |
| Vocab | 201024 padded / 200058 unpadded, 60 special tokens, special_start 199998 |
| Quantized source | layers 0-19 experts NVFP4, 20-23 MXFP8; matrices BF16/F32 |

Out of scope for this port: vision tower, MTP head, the 594B model.

## Container (`INK2JSON`)

`scripts/export_inkling.py` reads the source safetensors, dequantizes NVFP4/MXFP8 experts to
dense, and writes a mmap-friendly container: magic `INK2JSON`, a JSON manifest padded to 1 MiB,
then tensor blobs at absolute offsets. Names are the source names without `model.llm.`; Q8
matrices carry f32 scales per 64 values, Q4 experts are packed nibbles plus fp16 (scale, min) per
32 values, one self-contained block per expert. The loader repacks matrices and experts into the
engine's 4-row SDOT layout without touching any existing `.bin` format.

| Artifact | Experts | Size | Notes |
|---|---|---|---|
| `inkling_mini_q4.ink2` | Q4 | 17.00 GB | default; source dequant -> Q4 |
| `inkling_mini_q8.ink2` | Q8 | 27.71 GB | source dequant -> Q8 (`--experts q8`) |
| `inkling2_tokenizer.bin` | — | 3.7 MB | 201024 entries, BPE matches `llama-tokenize` |

Published for download at
[`miloserdev/Inkling-2-Mini-numallama`](https://huggingface.co/miloserdev/Inkling-2-Mini-numallama)
together with the model card, tokenizer, exporter scripts, checksums and validation logits.

The official IQ4_NL GGUF remains useful as a runtime oracle: upstream llama.cpp with
`inkling2_loop` support wrote `logits_oracle.bin` for "The capital of France is".

## Quality

Oracle: official IQ4_NL GGUF, top-1 ` Paris` id 152717, logit 27.307991. Candidate logits are
produced by `tests/check_inkling`; `scripts/compare_inkling_logits.py` prints the table.

| Variant | Top-1 | Logit | KL(oracle) | Top-10 overlap | Correlation | Max logit diff |
|---|---|---|---|---|---|---|
| Q4 experts | 152717 | 27.3686 | 0.0594 | 10/10 | 0.97889 | 5.37 |
| Q8 experts | 152717 | 27.4252 | 0.0535 | 10/10 | 0.98145 | 5.08 |

Q4 vs Q8: KL 0.0083, correlation 0.99683, max diff 1.82, top-10 10/10. Most of the remaining
gap to the IQ4_NL oracle is therefore not expert quantization (the oracle is itself 4-bit and the
forward/quantization recipes differ). Greedy 24-token continuations from Q4 and Q8 agree up to
` Paris.` and the following A/B/C-style choices, then diverge in the tail; the IQ4_NL oracle takes
a different continuation after ` Paris.`. This is expected greedy divergence under different
quantization, not a top-1 failure.

## Reproduce

```bash
# export (source safetensors + config in the same directory)
python3 scripts/export_inkling.py model-00001-of-00001.safetensors inkling_mini_q4.ink2
python3 scripts/export_inkling.py model-00001-of-00001.safetensors inkling_mini_q8.ink2 --experts q8

# full forward, oracle prompt, writes logits_ours.bin
tests/check_inkling inkling_mini_q4.ink2
python3 scripts/compare_inkling_logits.py logits_oracle.bin logits_ours.bin

# raw generation / chat / OpenAI server
./runq inkling_mini_q4.ink2 -z inkling2_tokenizer.bin -m generate -i "The capital of France is" -n 24 -t 0
./runq inkling_mini_q4.ink2 -z inkling2_tokenizer.bin -m chat -i "What is the capital of France?" -n 128 -t 0
./runq inkling_mini_q4.ink2 -z inkling2_tokenizer.bin -m server -P 8080 -c 8192
```

## Server

`/v1/chat/completions` supports streaming and non-streaming replies, separate
`reasoning_content`, the Inkling chat template (system, effort, user, assistant, tool messages),
stop strings, `min_p`, presence/frequency penalties, seeds and `include_usage`. One multi-turn
session is cached in memory (prompt tokens + 48-slot state + last logits); a client that resends
the previous assistant turn reuses it. `/metrics` exposes requests/tokens/cache hits/errors and
SIGINT/SIGTERM stop the server gracefully.

## Limitations

- The forward is single-threaded and token-by-token: about 0.79 s/token on the reference
  machine. The server handles one request at a time; run replicas behind
  `scripts/replica_router.py` for concurrency. `-T` and `-J` do not apply.
- Long prompts are rejected with a clear 400 above `INKLING_MAX_PROMPT` (default 4096) rather
  than blocking for hours; set it to 0 to disable the guard.
- Tool calls are not implemented in the Inkling server; use the dense/MoE server for agent
  clients. Vision, audio and MTP are not included.
- The Q4/Q8 container is a numallama format, not a GGUF; the export script is the source of
  truth for reproducibility.

## Attribution

The weights and architecture are by Thinking Machines (Inkling-2-Mini, Apache-2.0). The
numallama port, INK2 exporter, Q4/Q8 containers and this document are derivative work; when
publishing the containers, include the upstream license, model card link and the source revision
above. The official GGUF by Thinking Machines is a separate artifact and is not redistributed
here.
