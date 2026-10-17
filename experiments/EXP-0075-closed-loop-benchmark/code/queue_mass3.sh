#!/bin/bash
# EXP-0075 mass comparison extension to w10 (LOG 2026-10-12 MASS w10): ot vs fl only, letters O/T/S/X, fresh starts 48-55 (32 episodes), 10 pushes, same cell configs as queue_mass2.sh.
# One driver call per H (order 2, 4, 1); results/mass/w10_s48_H<H>.json checkpointed every step; ledger lines in experiments/COMMANDS.jsonl. Stop with results/mass/STOP3.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
R=$E/results/mass
SHA=$(git rev-parse --short HEAD)
C='"model":"ens2bal","planner":"cem","legal_score":true,"gd_steps":20,"fetch":true,"n_pool":64,"pop":64'
w=10; s0=48
run() { H=$1; tag=w${w}_s${s0}_H${H}; [ -f $R/$tag.done ] && return; [ -f $R/STOP3 ] && return
  STARTS=$(seq -s' ' $s0 $((s0+7))); G="letter_O_w$w letter_T_w$w letter_S_w$w letter_X_w$w"
  OBJ=""; [ $H -gt 1 ] && OBJ=',"obj_mode":"cumspill"'
  cells="{\"mass_w${w}_H${H}_ot\":{$C,\"H\":$H$OBJ,\"ot\":true,\"pool_mode\":\"ot\"},\"mass_w${w}_H${H}_fl\":{$C,\"H\":$H$OBJ,\"pool_mode\":\"fl\",\"n_extra\":16}}"
  rid="EXP-0075/mass/$tag-$(date +%s)"; t0=$(date +%s)
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $G --starts $STARTS --steps 10 --cells <see queue_mass3.sh>\",\"note\":\"EXP-0075 mass w10 extension, HEAD $SHA + uncommitted (dirty), GPU shared\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $G --starts $STARTS --cells "$cells" --steps 10 > $E/logs/mass_$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $R/$tag.done; }
run 2; run 4; run 1
echo "$(date) DONE" > $R/queue3.done
