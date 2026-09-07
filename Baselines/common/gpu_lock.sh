#!/usr/bin/env bash
# Serialise GPU jobs across concurrently-running baseline agents.
# There is ONE 8 GB RTX 4070 Laptop GPU on this machine; two trainings at once
# will OOM and take both down. Every training/eval run that touches CUDA goes
# through here:
#
#     Baselines/common/gpu_lock.sh python -m training.train --config ...
#
# Waits (up to 6 h) for the lock rather than failing, so an agent that queues
# behind another agent's training simply blocks instead of crashing.
set -euo pipefail
LOCK=/tmp/pile_manipulation_gpu.lock
exec 9>"$LOCK"
echo "[gpu_lock] $(date +%H:%M:%S) waiting for GPU lock: $*" >&2
flock -w 21600 9 || { echo "[gpu_lock] timed out waiting for GPU" >&2; exit 75; }
echo "[gpu_lock] $(date +%H:%M:%S) acquired; running: $*" >&2
"$@"
