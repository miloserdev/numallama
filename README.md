# numallama — fast NUMA-aware CPU inference for Llama 3.x, Qwen2.5 and Qwen3 / Qwen3-MoE

numallama (code name) is a small C inference engine (one `.c` file + a header, no dependencies) for Llama 3.x, Qwen2.5 and
Qwen3 / Qwen3-MoE models, tuned for large Arm servers (Ampere Altra / Neoverse N1, NEON + dotprod) and multi-socket
NUMA machines. It started as a fork of [jameswdelancey/llama3.c](https://github.com/jameswdelancey/llama3.c),
itself derived from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

Highlights:

- **Mixture of experts (Qwen3-MoE, e.g. Qwen3-30B-A3B)**: int8 router, top-k routing, every expert
  split evenly between the NUMA nodes; batched prefill groups the tokens by expert so every expert is
  read once per block. Also Qwen3 attention (head_dim != dim / heads, per-head q/k RMSNorm).
- **int8 (Q8) and 4-bit (Q4) weights** with NEON `sdot` kernels (single token, 4x4 batched).
- **Multi-socket NUMA**: every weight matrix and the KV cache are split between the sockets
  (`mbind`), each socket's threads compute only its rows, per-node work counters and per-node
  activation buffers keep cross-socket traffic minimal; attention and the MoE router synchronize
  within a socket only. 2 sockets give +50..85% decode speed.
- **Persistent thread pool** with flag-tree barriers in node-local memory (~1 us within a socket,
  ~2 us across), dynamic row distribution, prefetch of the next phase's rows while waiting,
  futex sleep when idle, automatic choice of idle CPUs per NUMA node; sampling runs on the pool.
- **Batched prompt prefill** (matrix x matrix) and a **multi-token decode step**.
- **Speculative decoding**, exact (output distribution unchanged):
  prompt lookup (n-grams from the whole dialog) and an optional small **draft model** (`-d`).
- **Chat** with the native templates (Llama 3, ChatML), context trimming, and **tool calling**
  (`config/tools.json`: sandboxed `ls`/`cat`/`grep`/`find` via fork/execvp, no shell).
- Correct **tiktoken-style tokenizer** (Llama 3 / Qwen2 pre-tokenizer regex + byte-level BPE).
- **SmoothQuant** for Qwen (activation outliers folded into the weights at export time).

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for measurements, including a comparison with llama.cpp.
The binary is still called `runq` (the name of the upstream int8 runtime).

## Layout

| Path | Contents |
|---|---|
| `src/runq.c` | the engine: model loading, NUMA placement, thread pool, kernels, forward passes, tokenizer, sampler, chat |
| `src/tools.h` | tool calling: minimal JSON parser, `tools.json` loading, sandboxed command execution |
| `scripts/export_hf_q8.py` | Hugging Face safetensors → `.bin` (Q8 / Q4, optional SmoothQuant, numpy only) |
| `scripts/export_tokenizer.py` | `tokenizer.json` → `tokenizer.bin` |
| `scripts/calibrate.py` | activation statistics on a calibration text (for `--smooth` / `--imatrix`) |
| `config/tools.json` | tools offered to the model in chat mode |
| `data/tokenizer.bin` | Llama 3 tokenizer |
| `tests/` | fp32 numpy reference, tokenizer reference, engine consistency checks |
| `legacy/` | the original upstream fp32 runtime and PyTorch export scripts, for reference |

## Build

```bash
make            # ./runq          (-Ofast -mcpu=native -pthread)
make prof       # ./runq_prof     per-phase timing report at exit
make tests      # tests/*
```

Linux only (pthreads, NUMA syscalls, futex). AArch64 with dotprod gets the NEON kernels; other CPUs
use portable C fallbacks.

## Models

```bash
# Llama 3.x: weights + the bundled tokenizer
python3 scripts/export_hf_q8.py /path/to/Llama-3.1-8B-Instruct llama31_8b_q80.bin
# 4-bit (faster decode, lower quality); w2 is the most sensitive matrix
python3 scripts/export_hf_q8.py /path/to/model model_q4.bin --q4                 # all matrices
python3 scripts/export_hf_q8.py /path/to/model model_q4.bin --q4 wq,wk,wv,wo,w1,w3

# Qwen3-MoE (e.g. Qwen3-30B-A3B-Instruct-2507): ~32 GB in Q8
python3 scripts/export_hf_q8.py /path/to/Qwen3-30B-A3B-Instruct-2507 qwen3_30b_a3b_q80.bin
python3 scripts/export_tokenizer.py /path/to/Qwen3-30B-A3B-Instruct-2507 qwen3_tokenizer.bin

# Qwen2.5: calibrate, export with SmoothQuant, export its tokenizer
python3 scripts/calibrate.py /path/to/Qwen2.5-7B-Instruct calib.txt qwen7b.npz
python3 scripts/export_hf_q8.py /path/to/Qwen2.5-7B-Instruct qwen7b_q80.bin --imatrix qwen7b.npz --smooth 0.5
python3 scripts/export_tokenizer.py /path/to/Qwen2.5-7B-Instruct qwen7b_tokenizer.bin
```

`calib.txt` is any natural text of a few thousand words (a mix of the languages you use).

## Run

```bash
# chat (Llama 3 tokenizer is the default: data/tokenizer.bin)
./runq llama31_8b_q80.bin -m chat -t 0.7

# chat with tools, Qwen tokenizer
./runq qwen7b_q80.bin -z qwen7b_tokenizer.bin -m chat -f config/tools.json -t 0.6

# speculative decoding with a draft model (same tokenizer, smaller model)
./runq llama31_8b_q80.bin -m chat -d llama32_1b_q80.bin

# plain generation
./runq llama31_8b_q80.bin -t 0 -n 256 -i "Once upon a time"
```

| Option | Meaning |
|---|---|
| `-T <n>` | threads (default 0 = all idle CPUs of the used NUMA nodes, at most 96 per node) |
| `-N <n>` | NUMA nodes to use (default 0 = all); weights are split between them |
| `-c <n>` | context window (KV cache size), default 8192 |
| `-k <n>` | max prompt-lookup draft length, default 8, `0` = off (adaptive for MoE models) |
| `-d <file>` | draft model for speculative decoding |
| `-f <file>` | tools file, enables tool calling in chat |
| `-t`, `-p`, `-s` | temperature, top-p, seed |
| `-y`, `-i` | system prompt, first user message (chat) / prompt (generate) |

After every reply the chat prints `[prompt N tok X tok/s | reply M tok Y tok/s | context a/b | draft accepted c/d]`
to stderr. Thread placement: `RUNQ_PIN=1` keeps threads pinned, `RUNQ_PIN=0` disables placement.
`RUNQ_BARBENCH=1` prints the latency of the node and cross-node barriers at start-up.

For Russian (and other non-English text) with small models keep the temperature around 0.6–0.7.

## Tests

```bash
make tests
# logits vs an independent fp32 numpy reference (KL, top-1 agreement)
tests/dump_logits model.bin data/tokenizer.bin 64 logits.bin "some text" > ids.txt
python3 tests/ref_logits.py /path/to/hf-model ids.txt logits.bin
# tokenizer vs reference BPE built from tokenizer.json (NDIG=1 for Qwen)
tests/encode data/tokenizer.bin 128256 0 < corpus.txt > c.txt
TOKJSON=/path/to/tokenizer.json python3 tests/tok_ref.py < corpus.txt > p.txt && diff c.txt p.txt
# batched prefill / multi-token step / draft-model loop vs plain decoding
tests/check_batch model.bin 64 300
tests/check_multi model.bin
tests/check_draft main.bin draft.bin
```

## License

MIT (see `LICENSE`), as the upstream projects.
