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

# Agents invoke this from the repo root but the wrapper eats a caller-set
# PYTHONPATH surprisingly often (it is a fresh `bash`, and `PYTHONPATH=. cmd`
# prefix assignments do not survive being passed as arguments here). Every
# baseline imports `Baselines.*` and `training.*` by absolute package path, so
# default it to the repo root rather than let each agent rediscover
# `ModuleNotFoundError: No module named 'Baselines'` the hard way -- which is
# exactly what killed B2-nfd's first run at 01:46.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
case ":$PYTHONPATH:" in *":$REPO_ROOT:"*) ;; *) export PYTHONPATH="$REPO_ROOT:$PYTHONPATH" ;; esac

LOCK=/tmp/pile_manipulation_gpu.lock
exec 9>"$LOCK"
echo "[gpu_lock] $(date +%H:%M:%S) waiting for GPU lock: $*" >&2
flock -w 21600 9 || { echo "[gpu_lock] timed out waiting for GPU" >&2; exit 75; }
echo "[gpu_lock] $(date +%H:%M:%S) acquired; running: $*" >&2
"$@"
