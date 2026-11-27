#!/bin/bash
# Continuous-batching benchmark: J concurrent sessions, all started together.
# usage: bench_cb.sh <model> <tokenizer> <J> <clients> [max_tokens] [port] [threads]
set -u
model=$1 tok=$2 J=$3 n=$4 max_tokens=${5:-128} port=${6:-5110} threads=${7:-128}
cd "$(dirname "$0")/../.." || exit 1
tmp=$(mktemp -d /tmp/bench_cb.XXXXXX)
prompts=(
  "Write a long paragraph about the history of the Roman Empire."
  "Explain in detail how TCP congestion control works."
  "Write a detailed C function for a lock-free ring buffer and explain each line."
  "Describe the water cycle in detail for a student."
)
for i in $(seq 0 $((n - 1))); do
  python3 - "$tmp/req$i.json" "${prompts[$((i % 4))]}" "$i" "$max_tokens" <<'PY'
import json, sys
path, prompt, idx, mt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
json.dump({"model": "bench", "session_id": f"cb{idx}", "messages": [{"role": "user", "content": prompt}],
           "max_tokens": mt, "temperature": 0}, open(path, "w"))
PY
done
log=$tmp/server.log
./runq "$model" -z "$tok" -m server -P "$port" -c 8192 -T "$threads" -J "$J" -W none -t 0 >"$log" 2>&1 &
srv=$!
for _ in $(seq 1 600); do curl -fsS -m 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done
t0=$(date +%s.%N)
pids=""
for i in $(seq 0 $((n - 1))); do
  curl -sS -m 1200 -H 'Content-Type: application/json' --data-binary @"$tmp/req$i.json" \
    "http://127.0.0.1:$port/v1/chat/completions" -o "$tmp/resp$i.json" & pids="$pids $!"
done
for p in $pids; do wait "$p"; done
t1=$(date +%s.%N)
kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
python3 - "$tmp" "$n" "$t0" "$t1" "$J" <<'PY'
import json, sys, glob
d, n, t0, t1, J = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
tokens = 0
for f in glob.glob(d + "/resp*.json"):
    tokens += json.load(open(f))["usage"]["completion_tokens"]
wall = t1 - t0
print(f"J={J} sessions={n} wall={wall:.2f}s tokens={tokens} aggregate={tokens/wall:.1f} tok/s")
PY
rm -rf "$tmp"
