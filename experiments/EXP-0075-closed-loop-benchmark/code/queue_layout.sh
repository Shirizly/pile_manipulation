#!/bin/bash
# EXP-0075 layout-transport objective test (LOG.md 2026-10-12, results/layout/). One driver call per (width, H) group; driver checkpoints results/layout/<group>.json after every step and resumes;
# ledger lines in experiments/COMMANDS.jsonl. Cells lo_<w>_H<H>_<obj>: obj cur = control (current objective), lay = + 2.0 x Sinkhorn W1 to the uniform_layout slots (layout_term.LayoutTransport, objective "layout").
# All arms: ens2bal, 10 pushes, legal scoring, GD 20 steps, fetch + OT pool (pool_mode ot, n_pool 64, pop 64 = 320 evaluations), H=2 cumspill.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
R=$E/results/layout
mkdir -p $R $E/logs
STARTS="40 41 42 43 44 45 46 47"
SHA=$(git rev-parse --short HEAD)
C='"model":"ens2bal","planner":"cem","legal_score":true,"gd_steps":20,"fetch":true,"ot":true,"pool_mode":"ot","n_pool":64,"pop":64'
run() { tag=$1; goals=$2; cells=$3; [ -f $R/$tag.done ] && return; rid="EXP-0075/layout/$tag-$(date +%s)"; t0=$(date +%s)
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $goals --starts $STARTS --steps 10 --cells <see queue_layout.sh>\",\"note\":\"EXP-0075 layout-transport cells, HEAD $SHA + uncommitted (dirty)\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $goals --starts $STARTS --cells "$cells" --steps 10 > $E/logs/layout_$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $R/$tag.done; }
for H in 1 2; do for W in 15 10; do
  G="letter_O_w$W letter_T_w$W letter_S_w$W letter_X_w$W"
  OM=""; [ $H = 2 ] && OM=',"obj_mode":"cumspill"'
  run w${W}_H$H "$G" "{\"lo_w${W}_H${H}_cur\":{$C,\"H\":$H$OM}, \"lo_w${W}_H${H}_lay\":{$C,\"H\":$H$OM,\"objective\":\"layout\",\"layout_lam\":2.0}}"
done; done
echo "$(date) DONE" > $R/queue.done
