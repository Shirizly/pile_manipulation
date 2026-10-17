#!/bin/bash
# EXP-0075 mass paired comparison (LOG.md 2026-10-11 "MASS COMPARISON"): OT pool (320 evals + GD 20) vs fl pool (320 + GD) vs vanilla CEM (1,280 evals, no fetch/legal/GD, terminal objective),
# H=1,2,4, letters O/T/S/X, fresh DS-0006 starts 48-111 in blocks of 8 (+ block 40 = cross-check), 10 pushes, ens2bal, planner seed 0. One driver call per (width, start block) = 9 cells x 32 episodes;
# results/mass/<tag>.json checkpointed every step, resume of incomplete cells; ledger lines in experiments/COMMANDS.jsonl. Order: w20 b0,b1; w15 b0,b1; then alternate; w10 last.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
R=$E/results/mass
SHA=$(git rev-parse --short HEAD)
C='"model":"ens2bal","planner":"cem","legal_score":true,"gd_steps":20,"fetch":true,"n_pool":64,"pop":64'
[ -f $R/STOP ] && exit 0
run() { w=$1; s0=$2; tag=w${w}_s${s0}; [ -f $R/$tag.done ] && return; [ -f $R/STOP ] && return
  STARTS=$(seq -s' ' $s0 $((s0+7))); G="letter_O_w$w letter_T_w$w letter_S_w$w letter_X_w$w"
  cells="{"; for H in 1 2 4; do
    OBJ=""; [ $H -gt 1 ] && OBJ=',"obj_mode":"cumspill"'
    cells="$cells\"mass_w${w}_H${H}_ot\":{$C,\"H\":$H$OBJ,\"ot\":true,\"pool_mode\":\"ot\"},\"mass_w${w}_H${H}_fl\":{$C,\"H\":$H$OBJ,\"pool_mode\":\"fl\",\"n_extra\":16},\"mass_w${w}_H${H}_van\":{\"model\":\"ens2bal\",\"planner\":\"cem\",\"H\":$H},"; done
  cells="${cells%,}}"
  rid="EXP-0075/mass/$tag-$(date +%s)"; t0=$(date +%s)
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $G --starts $STARTS --steps 10 --cells <see queue_mass.sh>\",\"note\":\"EXP-0075 mass comparison, HEAD $SHA + uncommitted (dirty), GPU shared\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $G --starts $STARTS --cells "$cells" --steps 10 > $E/logs/mass_$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $R/$tag.done; }
run 20 48; run 20 56; run 15 48; run 15 56
for s in 64 72 80 88 96 104; do run 20 $s; run 15 $s; done
run 10 48; run 10 56; run 20 40; run 15 40
echo "$(date) DONE" > $R/queue.done
