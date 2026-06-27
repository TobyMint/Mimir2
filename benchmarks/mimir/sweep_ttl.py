#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Sweep MIMIR_PIN_TTL, restarting the pin-only server for each TTL and each
# repeat, with thorough GPU cleanup between runs. pin-only (no connector) at
# the validated pressure point (12 agents, scale 8).
#
# Robustness notes (learned from a previous run whose data was unreliable):
# - 12 agents at ~85-99% GPU usage is near a tipping point, so run-to-run
#   variance is large. We run each TTL REPEATS times with different seeds and
#   take the median, to separate TTL effect from noise.
# - Leftover GPU memory from a previous server shifts the next run's pressure.
#   We poll until GPU memory is actually freed (<500MB) before starting the
#   next server, and kill by compute-apps pid, not just by process name.
#
# Usage: bash benchmarks/mimir/sweep_ttl.py
set -u
cd "$(dirname "$0")/../.." || exit 1

# Activate the project env explicitly so the script works regardless of the
# caller's shell state.
source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate mimir2 2>/dev/null || true

PORT=8199
GPU=3
MODEL=/data/models/Qwen3-4B
TRACES=benchmarks/mimir/traces/batch30
OUTDIR=benchmarks/mimir/results
mkdir -p "$OUTDIR"

TTLS=(1 2 3 5 8)
REPEATS=3
SEEDS=(42 137 2718)

clean_gpu() {
  # Kill vllm serve + engine core + anything holding the GPU, then poll until
  # GPU memory is actually freed. Returns 0 if freed, 1 if timed out.
  pkill -9 -f "vllm serve" 2>/dev/null
  pkill -9 -f "EngineCore" 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null
  done
  for _ in $(seq 1 30); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null || echo 99999)
    [ "$used" -lt 500 ] && return 0
    # still occupied: kill any compute-apps pid again (they can respawn briefly)
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -9 "$pid" 2>/dev/null
    done
    sleep 3
  done
  echo "WARN: GPU$GPU still has ${used}MB after cleanup attempts"
  return 1
}

start_server() {
  local ttl=$1 log=$2
  CUDA_VISIBLE_DEVICES=$GPU VLLM_ENABLE_V1_MULTIPROCESSING=0 MIMIR_PIN_TTL=${ttl}.0 \
    nohup vllm serve "$MODEL" --port $PORT --scheduling-policy mimir \
      --max-model-len 40960 --max-num-batched-tokens 65536 \
      --gpu-memory-utilization 0.85 --dtype bfloat16 > "$log" 2>&1 &
  echo $!
}

wait_ready() {
  # Wait for /v1/models, then extra 12s for cudagraph capture to settle.
  local log=$1
  for _ in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q model; then
      sleep 12
      return 0
    fi
    sleep 5
  done
  echo "server failed to start; log tail:"; tail -15 "$log"
  return 1
}

for ttl in "${TTLS[@]}"; do
  for rep in $(seq 1 $REPEATS); do
    seed=${SEEDS[$((rep-1))]}
    echo "================ TTL=$ttl repeat=$rep seed=$seed ================"
    if ! clean_gpu; then
      echo "skip (GPU not clean)"; continue
    fi
    LOG=/tmp/vllm_sweep_ttl${ttl}_r${rep}.log
    SRV_PID=$(start_server "$ttl" "$LOG")
    if ! wait_ready "$LOG"; then
      kill -9 "$SRV_PID" 2>/dev/null; continue
    fi
    LABEL=mimir_pinonly_ttl${ttl}_r${rep}
    python benchmarks/mimir/replay_benchmark.py \
      --traces "$TRACES" \
      --num-agents 12 --arrival-rate 2.0 --scale 8 --port $PORT \
      --server-log "$LOG" --label "$LABEL" --seed "$seed" \
      --out "$OUTDIR/${LABEL}.json" 2>&1 | tail -3
    kill -9 "$SRV_PID" 2>/dev/null
  done
done

echo "================ SUMMARY (median of $REPEATS repeats) ================"
python3 - <<'PY'
import json, glob, statistics
from collections import defaultdict
by_ttl = defaultdict(list)
for f in glob.glob("benchmarks/mimir/results/mimir_pinonly_ttl*_r*_12x8.json") + \
         glob.glob("benchmarks/mimir/results/mimir_pinonly_ttl*_r*.json"):
    import re
    m = re.search(r"ttl(\d+)_r(\d+)", f)
    if not m: continue
    ttl = int(m.group(1))
    by_ttl[ttl].append(json.load(open(f)))
# also fold in the single-run validated ttl2
for f in glob.glob("benchmarks/mimir/results/mimir_pinonly_12x8.json"):
    by_ttl[2].append(json.load(open(f)))
print(f"{'TTL':>4} {'n':>3} {'JCT_med(med)':>13} {'JCT_p90(med)':>13} {'TTFT_med(med)':>14} {'hit%':>6} {'usage_max':>9}")
for ttl in sorted(by_ttl):
    runs = by_ttl[ttl]
    jct_med = statistics.median([r['job_completion_s']['median'] for r in runs])
    jct_p90 = statistics.median([r['job_completion_s']['p90'] for r in runs])
    ttft_med = statistics.median([r['ttft_s']['median'] for r in runs])
    hit = statistics.median([r['pressure_evidence']['prefix_hit_rate_last'] for r in runs])
    usage = max(r['pressure_evidence']['gpu_kv_usage_max'] for r in runs)
    print(f"{ttl:>4} {len(runs):>3} {jct_med:>13.1f} {jct_p90:>13.1f} {ttft_med:>14.2f} {hit:>6.1f} {usage:>9.1f}")
# per-run detail for transparency
print("\n--- per-run detail (to judge variance) ---")
for ttl in sorted(by_ttl):
    for r in by_ttl[ttl]:
        print(f"TTL={ttl}: JCT_med={r['job_completion_s']['median']:.1f} TTFT_med={r['ttft_s']['median']:.2f} "
              f"hit={r['pressure_evidence']['prefix_hit_rate_last']} usage_max={r['pressure_evidence']['gpu_kv_usage_max']}")
PY
