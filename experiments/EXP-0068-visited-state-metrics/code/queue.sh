#!/usr/bin/env bash
# EXP-0068 GPU chain (one job at a time, GPU lock experiments/temp/gpu.lock held for the whole chain):
#   pilot (cost only) -> RUN-0001 closed loop (states + quats + top-16) -> select_states (CPU) -> RUN-0002 DS-0023 pools
#   -> RUN-0003 scoring. python -u, pid files, .done markers (resume-safe), ledger lines in experiments/COMMANDS.jsonl.
cd "$(dirname "$0")/../../.."
PY=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0068-visited-state-metrics; R=$E/results; L=$E/logs; C=$E/code; mkdir -p $R $L
D=experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py
DS=datasets/DS-0023-visited-state-pools/data
ZOO="nfd_3ch_narrow_l20_v2 nfd_3ch_narrow_l20_v2_seed1 nfd_3ch_narrow_l20_v2_seed2 nfd_3ch_narrow_l20_v2_soft_s2 nfd_3ch_narrow_l20_v2_epoch10 linear_narrow_l20_v2_res64 linear_switched_soft nfd_3ch_randlen"
EXTRA="linear_narrow_l20_v2_res64_boot1 linear_narrow_l20_v2_res64_boot2 nfd_3ch_randlen_seed1"
G="letter_O letter_T letter_S letter_L letter_X letter_Z letter_C letter_H"
CELLS='{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25, "push_len": 0.02, "topk": 16}}'
SHA=$(git rev-parse --short HEAD)
LOCK=experiments/temp/gpu.lock
echo "$$ EXP-0068" > $LOCK
trap 'rm -f $LOCK' EXIT
ledger() { printf '{"ts":"%s","run_id":"EXP-0068/%s","event":"%s"%s}\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" "$3" >> experiments/COMMANDS.jsonl; }
step() {  # step <run_id> <tag> <cmd...>
  local rid=$1 tag=$2; shift 2
  if [ -f $R/$tag.done ]; then echo "$tag already done"; return 0; fi
  echo "$(date) START $tag"; ledger $rid start ",\"cmd\":\"$*\",\"commit\":\"$SHA\",\"dirty\":true"
  OMP_NUM_THREADS=4 PYTHONPATH=. "$@" > $L/$tag.log 2>&1 &
  echo $! > $L/$tag.pid; wait $!; rc=$?
  ledger $rid end ",\"status\":\"$([ $rc = 0 ] && echo ok || echo fail)\",\"returncode\":$rc"
  [ $rc = 0 ] && date > $R/$tag.done
  echo "$(date) END $tag rc=$rc"; return $rc
}
set -e
# (cost pilot dropped 2026-10-05 14:40: the GPU became free 5.5 h into the 7 h budget; CUT design = 1 start, 24 states)
step RUN-0001/closed_loop cl_full $PY -u $D --tag cl_full --results-dir $R --models $ZOO --planners cem --goals $G \
    --starts 40 --steps 13 --budget 0.5 --n-cand 1024 --cells "$CELLS" --legalize --legalize-fallback --planner-seed 0 --record-states --record-quats
CUDA_VISIBLE_DEVICES= step RUN-0002/select select $PY -u $C/select_states.py --episodes $R/cl_full.json --out $DS/starts.pt --starts 40 --n-per-cell 1
step RUN-0002/pools pools $PY -u $C/collect_pools.py --starts $DS/starts.pt --out $DS --pool 64 --n-envs 32
step RUN-0003/score_ds0023 score_ds0023 $PY -u $C/score_pools.py --corpus ds0023 --models $ZOO $EXTRA
step RUN-0003/score_ds0016 score_ds0016 $PY -u $C/score_pools.py --corpus ds0016 --models $ZOO $EXTRA
echo "$(date) QUEUE DONE" > $R/queue.done
