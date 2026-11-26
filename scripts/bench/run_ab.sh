#!/bin/bash
# A/B: fp32 vs int8 KV at ~8k and ~30k prompt tokens for two models.
# CSV: name,kv,ctx,prompt_tok,prefill_tok_s,reply_tok,decode_tok_s,wall_s
set -u
cd "$(dirname "$0")/../.." || exit 1
out=${1:-scripts/bench/ab_f32_i8.csv}
echo "name,kv,ctx,prompt_tok,prefill_tok_s,reply_tok,decode_tok_s,wall_s" > "$out"
run() { echo "--- $*"; ./scripts/bench/bench_one.sh "$@" | tee -a "$out"; }
q38=/var/models-bin/qwen38_27b_abl_q4.bin; q38t=/var/models-bin/qwen38_tokenizer.bin
q3=/var/models-bin/qwen3_30b_a3b_abl_q4.bin;   q3t=/var/models-bin/qwen3_tokenizer.bin
run qwen38_8k  "$q38" "$q38t" 12288 scripts/bench/prompt_8k_qwen38.txt  5071 128 f32
run qwen38_8k  "$q38" "$q38t" 12288 scripts/bench/prompt_8k_qwen38.txt  5071 128 i8
run qwen38_30k "$q38" "$q38t" 40960 scripts/bench/prompt_30k_qwen38.txt 5072 128 f32
run qwen38_30k "$q38" "$q38t" 40960 scripts/bench/prompt_30k_qwen38.txt 5072 128 i8
run qwen3_8k   "$q3"  "$q3t"  12288 scripts/bench/prompt_8k_qwen3.txt   5073 128 f32
run qwen3_8k   "$q3"  "$q3t"  12288 scripts/bench/prompt_8k_qwen3.txt   5073 128 i8
run qwen3_30k  "$q3"  "$q3t"  40960 scripts/bench/prompt_30k_qwen3.txt  5074 128 f32
run qwen3_30k  "$q3"  "$q3t"  40960 scripts/bench/prompt_30k_qwen3.txt  5074 128 i8
echo "done: $out"
