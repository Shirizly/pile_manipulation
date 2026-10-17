#!/bin/bash
# EXP-0075 gate G6 planner-seed-1 replication of the fl cells; pre-registration LOG.md 2026-10-15 07:18.
# Same mechanics as queue_gates2.sh: one driver call per cell, strictly serial, ledger lines in experiments/COMMANDS.jsonl.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
R=$E/results/gates
STARTS="40 41 42 43 44 45 46 47"
SHA=$(git rev-parse --short HEAD)
run() { tag=$1; GOALS=$2; cells=$3; [ -f $R/$tag.done ] && return; rid="EXP-0075/gates/$tag-$(date +%s)"; t0=$(date +%s)
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $GOALS --starts $STARTS --steps 10 --bank-per-env 4096 --planner-seed 1 --cells '${cells//\"/\\\"}'\",\"note\":\"EXP-0075 gate G6 seed-1 cell, HEAD $SHA + uncommitted (dirty)\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $GOALS --starts $STARTS --cells "$cells" --steps 10 --bank-per-env 4096 --planner-seed 1 > $E/logs/gates_$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $R/$tag.done
  $P -u $E/code/analyse_gates.py > $E/logs/analyse_gates.log 2>&1; }
cell() { echo "{\"$1\":{$2}}"; }
V='"model":"vanilla64bal","H":1'; N='"model":"ens2bal","H":1'; FL=',"fetch":true,"legal_score":true'
for w in w10 w15; do
  G="letter_O_$w letter_T_$w letter_S_$w letter_X_$w"
  run g6s1_${w}_v64_fl "$G" "$(cell g6s1_${w}_v64_fl "$V,\"planner\":\"cem\"$FL")"
  run g6s1_${w}_ens_fl "$G" "$(cell g6s1_${w}_ens_fl "$N,\"planner\":\"cem\"$FL")"
done
echo "$(date) DONE" > $R/queue_g6s1.done
