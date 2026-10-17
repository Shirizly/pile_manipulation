#!/bin/bash
# EXP-0075 planner-fix cells (LOG.md 2026-10-11). One driver call per tag; results/<tag>.json rewritten each step; ledger lines in experiments/COMMANDS.jsonl.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
GOALS="letter_O_w20 letter_T_w20 letter_S_w20 letter_X_w20"
STARTS="40 41 42 43 44 45 46 47"
SHA=$(git rev-parse --short HEAD)
run() { tag=$1; cells=$2; [ -f $E/results/$tag.done ] && return; rid="EXP-0075/$tag-$(date +%s)"; t0=$(date +%s)
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --goals $GOALS --starts $STARTS --steps 10 --cells <see queue_fixes.sh>\",\"note\":\"EXP-0075 planner-fix cells, HEAD $SHA + uncommitted fixes (dirty)\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --goals $GOALS --starts $STARTS --cells "$cells" --steps 10 > $E/logs/$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $E/results/$tag.done; }
run fix_H1full '{"base1F": {"model":"ens2bal","planner":"cem","H":1}, "fetch1F": {"model":"ens2bal","planner":"cem","H":1,"fetch":true}, "legal1F": {"model":"ens2bal","planner":"cem","H":1,"legal_score":true}, "fl1F": {"model":"ens2bal","planner":"cem","H":1,"fetch":true,"legal_score":true}}'
run fix_L20 '{"base1_20": {"model":"ens2bal","planner":"cem","H":1,"push_len":0.02}, "legal1_20": {"model":"ens2bal","planner":"cem","H":1,"push_len":0.02,"legal_score":true}, "fl1_20": {"model":"ens2bal","planner":"cem","H":1,"push_len":0.02,"fetch":true,"legal_score":true}}'
run fix_H2 '{"base2F": {"model":"ens2bal","planner":"cem","H":2}, "cum2F": {"model":"ens2bal","planner":"cem","H":2,"obj_mode":"cumulative"}, "cs2F": {"model":"ens2bal","planner":"cem","H":2,"obj_mode":"cumspill"}, "cs2F_fl": {"model":"ens2bal","planner":"cem","H":2,"obj_mode":"cumspill","fetch":true,"legal_score":true}}'
echo "$(date) DONE" > $E/results/queue_fixes.done
