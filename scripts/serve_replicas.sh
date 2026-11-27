#!/bin/bash
# Launch NUMA-local server replicas for throughput (one process per pin group).
# usage: serve_replicas.sh <model> <tokenizer> [replicas=8] [threads=16] [J=16] [base_port=19000] [ctx=8192]
# Each replica is pinned to a NUMA node (round-robin), keeps its own KV/caches and queue.
set -euo pipefail
model=${1:?model}
tok=${2:?tokenizer}
R=${3:-8}
T=${4:-16}
J=${5:-16}
base=${6:-19000}
ctx=${7:-8192}
cd "$(dirname "$0")/.."

nodes=$(numactl --hardware 2>/dev/null | awk '/^available:/{print $2}')
[ -n "${nodes:-}" ] && [ "$nodes" -gt 0 ] || nodes=1
logdir=${NUMALLAMA_REPLICA_LOG:-/tmp/numallama_replicas}
mkdir -p "$logdir"
pidfile="$logdir/pids"
: > "$pidfile"

pids=()
cleanup() {
  trap - EXIT INT TERM HUP
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
  sleep 1
  for p in "${pids[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

echo "[replicas] $R replicas, T=$T J=$J ctx=$ctx, $nodes NUMA node(s), ports $base..$((base+R-1))"
for ((i=0; i<R; i++)); do
  node=$((i % nodes))
  port=$((base+i))
  numactl --cpunodebind="$node" --membind="$node" \
    ./runq "$model" -z "$tok" -m server -P "$port" -c "$ctx" -T "$T" -J "$J" -K i8 -W none -t 0 \
    >"$logdir/replica-$i.log" 2>&1 &
  p=$!
  pids+=("$p")
  echo "$p $port $node" >> "$pidfile"
done

for ((i=0; i<R; i++)); do
  port=$((base+i))
  ok=0
  for _ in $(seq 1 900); do
    if curl -fsS -m 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 1
  done
  if [ "$ok" != 1 ]; then
    echo "[replicas] replica $i on :$port failed; see $logdir/replica-$i.log" >&2
    exit 1
  fi
done

echo "[replicas] ready:"
for ((i=0; i<R; i++)); do echo "  127.0.0.1:$((base+i))  node $((i % nodes))"; done
echo "[replicas] logs: $logdir; one shared endpoint: scripts/replica_router.py 8080 127.0.0.1:$base ..."
wait
