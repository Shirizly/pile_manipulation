#!/usr/bin/env bash
# Bounded foreground wait for the shared GPU: returns 0 once experiments/temp/gpu.lock is absent OR its PID is dead,
# AND nvidia-smi shows no compute process. $1 = max seconds (default 1500).
cd "$(dirname "$0")/../../.."
MAX=${1:-1500}; T0=$(date +%s)
free_gpu() {
  local lock=experiments/temp/gpu.lock
  if [ -f $lock ]; then
    local pid=$(awk '{print $1}' $lock)
    if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then return 1; fi
  fi
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${mem:-9999}" -lt 200 ]
}
while ! free_gpu; do
  if [ $(( $(date +%s) - T0 )) -ge $MAX ]; then echo "$(date) still busy: $(cat experiments/temp/gpu.lock 2>/dev/null) mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; exit 1; fi
  sleep 30
done
echo "$(date) GPU free"; exit 0
