#!/bin/bash
# Baseline (fp32 KV) decode/prefill at ~8k and ~30k prompt tokens for two models.
# Writes CSV: name,kv,ctx,prompt_tok,prefill_tok_s,reply_tok,decode_tok_s,wall_s
set -u
cd "$(dirname "$0")/../.." || exit 1
out=${1:-scripts/bench/baseline.csv}
echo "name,kv,ctx,prompt_tok,prefill_tok_s,reply_tok,decode_tok_s,wall_s" > "$out"
run() { echo "--- $*"; ./scripts/bench/bench_one.sh "$@" | tee -a "$out"; }
run qwen38_8k  /var/models-bin/qwen38_27b_abl_q4.bin  /var/models-bin/qwen38_tokenizer.bin 12288 scripts/bench/prompt_8k_qwen38.txt  5071 128 f32
run qwen38_30k /var/models-bin/qwen38_27b_abl_q4.bin  /var/models-bin/qwen38_tokenizer.bin 40960 scripts/bench/prompt_30k_qwen38.txt 5072 128 f32
run qwen3_8k   /var/models-bin/qwen3_30b_a3b_abl_q4.bin /var/models-bin/qwen3_tokenizer.bin 12288 scripts/bench/prompt_8k_qwen3.txt   5073 128 f32
run qwen3_30k  /var/models-bin/qwen3_30b_a3b_abl_q4.bin /var/models-bin/qwen3_tokenizer.bin 40960 scripts/bench/prompt_30k_qwen3.txt  5074 128 f32
echo "done: $out"
