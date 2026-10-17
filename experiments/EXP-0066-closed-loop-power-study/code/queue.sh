#!/usr/bin/env bash
# EXP-0066 queue: sequential closed-loop cells (GPU exclusive), python -u, pid files, atomic per-step checkpoints
# (driver), ledger lines in experiments/COMMANDS.jsonl. Resume-safe: the driver skips complete episodes by tag.
cd "$(dirname "$0")/../../.."
PY=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0066-closed-loop-power-study; R=$E/results; L=$E/logs; mkdir -p $R $L
D=experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py
G="letter_O letter_T letter_S letter_X"; STARTS=$(seq 40 71)
CELLS='{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25, "push_len": 0.02}}'
SHA=$(git rev-parse --short HEAD)
LAST_START=$(date -d "03:50 tomorrow" +%s 2>/dev/null); [ $(date +%H) -lt 12 ] && LAST_START=$(date -d "03:50" +%s)
OPT_START=$((LAST_START - 65*60))
ledger() { printf '{"ts":"%s","run_id":"EXP-0066/RUN-0001/%s","event":"%s"%s}\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" "$3" >> experiments/COMMANDS.jsonl; }
run() {  # tag model planner_seed optional(0/1)
  local tag=$1 m=$2 ps=$3 opt=${4:-0}
  if [ -f $R/$tag.done ]; then echo "$tag already done"; return; fi
  local now=$(date +%s)
  if [ $now -gt $LAST_START ] || { [ $opt = 1 ] && [ $now -gt $OPT_START ]; }; then echo "$(date) SKIP $tag (budget)" | tee -a $R/skipped.txt; return; fi
  echo "$(date) START $tag"
  ledger $tag start ",\"cmd\":\"$D --tag $tag --models $m --planner-seed $ps (DESIGN.md settings)\",\"commit\":\"$SHA\",\"dirty\":true"
  OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u $D --tag $tag --results-dir $R --models $m --planners cem --goals $G --starts $STARTS \
    --steps 16 --budget 0.5 --n-cand 1024 --cells "$CELLS" --legalize --legalize-fallback --planner-seed $ps --record-states > $L/$tag.log 2>&1 &
  echo $! > $L/$tag.pid; wait $!; rc=$?
  ledger $tag end ",\"status\":\"$([ $rc = 0 ] && echo ok || echo fail)\",\"returncode\":$rc"
  [ $rc = 0 ] && date > $R/$tag.done
  echo "$(date) END $tag rc=$rc"
}
run nfd_s0_ps0   nfd_3ch_narrow_l20_v2          0
run lin_orig_ps0 linear_narrow_l20_v2_res64     0
run nfd_s1_ps0   nfd_3ch_narrow_l20_v2_seed1    0
run lin_b1_ps0   linear_narrow_l20_v2_res64_boot1 0
run nfd_s2_ps0   nfd_3ch_narrow_l20_v2_seed2    0
run lin_b2_ps0   linear_narrow_l20_v2_res64_boot2 0
run nfd_s0_ps1   nfd_3ch_narrow_l20_v2          1
run nfd_s0_ps2   nfd_3ch_narrow_l20_v2          2
run lin_orig_ps1 linear_narrow_l20_v2_res64     1
run lin_orig_ps2 linear_narrow_l20_v2_res64     2
run broad_s0_ps0 nfd_3ch_randlen                0 1
run broad_s1_ps0 nfd_3ch_randlen_seed1          0 1
echo "$(date) QUEUE DONE" > $R/queue.done
