#!/usr/bin/env bash
# Serialise GPU jobs across concurrently-running baseline agents.
#
#     Baselines/common/gpu_lock.sh python -m training.train --config ...
#
# WHY A COUNTING SEMAPHORE AND NOT AN EXCLUSIVE LOCK (changed 2026-09-08).
# This started as `flock` on one file, on the assumption that two jobs on one
# 8 GB card would OOM. Measured, that assumption was wrong by an order of
# magnitude: every baseline here is tiny (NFD UNet ~30k params, GNN ~38k,
# Schenck ~140k), and a live 100-epoch NFD training was using 0.39 GB of
# 8.19 GB. The exclusive lock stopped being a safety device and became the
# critical path -- a 90-minute training was blocking a 5-minute EVALUATION.
#
# So: N_SLOTS concurrent jobs, each holding one of N lock files. Still bounded,
# so a genuinely large future model cannot be swamped by unbounded concurrency,
# but no longer serialising work that has no reason to serialise.
#
# Evaluation/scoring runs do not need this wrapper at all -- they are short and
# small. Use it for training.
set -euo pipefail

# A fresh `bash` here drops a caller's `PYTHONPATH=. cmd` prefix assignment, and
# every baseline imports `Baselines.*` / `training.*` by absolute package path.
# Not exporting it killed B2-nfd's first run instantly with ModuleNotFoundError,
# so the wrapper owns this rather than each agent rediscovering it.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
case ":$PYTHONPATH:" in *":$REPO_ROOT:"*) ;; *) export PYTHONPATH="$REPO_ROOT:$PYTHONPATH" ;; esac

N_SLOTS="${GPU_LOCK_SLOTS:-3}"
DEADLINE=$(( $(date +%s) + 21600 ))   # 6 h

echo "[gpu_lock] $(date +%H:%M:%S) seeking 1 of $N_SLOTS slots: $*" >&2
while :; do
  for slot in $(seq 1 "$N_SLOTS"); do
    exec 9>"/tmp/pile_manipulation_gpu.slot${slot}.lock"
    if flock -n 9; then
      echo "[gpu_lock] $(date +%H:%M:%S) acquired slot $slot; running: $*" >&2
      "$@"
      exit $?
    fi
    exec 9>&-
  done
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[gpu_lock] timed out waiting for a GPU slot" >&2; exit 75
  fi
  sleep 5
done
