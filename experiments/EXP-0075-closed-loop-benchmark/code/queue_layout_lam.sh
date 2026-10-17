#!/bin/bash
# EXP-0075 layout-weight study (LOG.md 2026-10-12 "LAYOUT WEIGHT"; results/layout_lam/). One driver call per cell (a cell = 32 episodes = one simulator chunk), 16 pushes (--allow-long), checkpoint per step, resumable.
# Cells lw_<w>_lam<L>: ens2bal, H=1, legal scoring, GD 20, 320 evals, fetch + OT pool; lam 0 = objective "cur" (control), else objective "layout" with layout_lam=L. Starts 40-47 x O/T/S/X. ONE process at a time (shared GPU).
# Priority order (budget-limited; later cells may not run): key refutation cells first.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python; E=experiments/EXP-0075-closed-loop-benchmark; D=$E/code/closed_loop.py; R=$E/results/layout_lam
mkdir -p $R $E/logs; STARTS="40 41 42 43 44 45 46 47"; SHA=$(git rev-parse --short HEAD)
C='"model":"ens2bal","planner":"cem","legal_score":true,"gd_steps":20,"fetch":true,"ot":true,"pool_mode":"ot","n_pool":64,"pop":64,"H":1'
run() { W=$1; L=$2; tag=w${W}_lam$L; [ -f $R/$tag.done ] && return; [ -f $R/STOP ] && return
  G="letter_O_w$W letter_T_w$W letter_S_w$W letter_X_w$W"
  if [ "$L" = 0 ]; then CELL="{\"lw_w${W}_lam$L\":{$C}}"; else CELL="{\"lw_w${W}_lam$L\":{$C,\"objective\":\"layout\",\"layout_lam\":$L}}"; fi
  rid="EXP-0075/layout_lam/$tag-$(date +%s)"; t0=$(date +%s)
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $G --starts $STARTS --steps 16 --allow-long --cells $CELL\",\"note\":\"EXP-0075 layout-weight study, HEAD $SHA + uncommitted (dirty)\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $G --starts $STARTS --cells "$CELL" --steps 16 --allow-long > $E/logs/layout_lam_$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $R/$tag.done; }
for wl in "15 0" "15 20" "15 2" "10 0" "10 20" "10 2" "15 6" "10 6" "15 0.5" "10 0.5"; do set -- $wl; run $1 $2; done
echo "$(date) DONE" > $R/queue.done
