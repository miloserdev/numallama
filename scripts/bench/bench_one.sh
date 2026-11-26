#!/bin/bash
# One benchmark run: start the server, send a long prompt, parse the prefill/decode speeds.
# usage: bench_one.sh <name> <model> <tokenizer> <ctx> <prompt-file> <port> [threads] [kv]
set -u
name=$1 model=$2 tok=$3 ctx=$4 prompt=$5 port=$6 threads=${7:-128} kv=${8:-f32}
log=/tmp/bench_${name}_${ctx}_${kv}.log
body=/tmp/bench_${name}_${ctx}_${kv}.json
resp=/tmp/bench_${name}_${ctx}_${kv}.resp

python3 - "$prompt" "$body" <<'EOF'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
json.dump({"model": "bench", "messages": [{"role": "user", "content": text}],
           "max_tokens": 128, "temperature": 0, "stream": False},
          open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
EOF

./runq "$model" -z "$tok" -m server -P "$port" -c "$ctx" -T "$threads" -K "$kv" -W none -t 0 >"$log" 2>&1 &
pid=$!
ready=0
for _ in $(seq 1 600); do
  if curl -fsS -m 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 1
done
if [ "$ready" != 1 ]; then
  echo "$name,$ctx,FAILED_TO_START,,,,"; tail -5 "$log"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; exit 1
fi

t0=$(date +%s.%N)
curl -fsS -m 3600 -H 'Content-Type: application/json' --data-binary @"$body" \
  "http://127.0.0.1:$port/v1/chat/completions" -o "$resp"
rc=$?
t1=$(date +%s.%N)
sleep 1
kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")
line=$(grep -E '^\[server\] prompt [0-9]+ tok' "$log" | tail -1)
if [ "$rc" != 0 ] || [ -z "$line" ]; then
  echo "$name,$ctx,CURL_${rc}_OR_NO_LOG,,,${wall}"; tail -5 "$log"; exit 1
fi
# [server] prompt N tok (cached C) P tok/s | reply M tok D tok/s | ...
stats=$(printf '%s\n' "$line" | sed -E 's/^\[server\] prompt ([0-9]+) tok \(cached [0-9]+\) ([0-9.]+) tok\/s \| reply ([0-9]+) tok ([0-9.]+) tok\/s.*/\1,\2,\3,\4/')
echo "$name,$kv,$ctx,$stats,$wall"
