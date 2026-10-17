#!/bin/bash
# EXP-0075 gates G3 (seed) + P0-2 remainder (budget ladders) + G5 (cheap-model proposals), RAL run plan Tier 1-2; pre-registration LOG.md 2026-10-15 02:55.
# One driver call per cell (results/gates/<cell>.json rewritten each step, resumes finished chunks); strictly serial; ledger lines in experiments/COMMANDS.jsonl.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
R=$E/results/gates
GOALS="letter_O_w20 letter_T_w20 letter_S_w20 letter_X_w20"
STARTS="40 41 42 43 44 45 46 47"
SHA=$(git rev-parse --short HEAD)
run() { tag=$1; bank=$2; cells=$3; [ -f $R/$tag.done ] && return; rid="EXP-0075/gates/$tag-$(date +%s)"; t0=$(date +%s)
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $GOALS --starts $STARTS --steps 10 --bank-per-env $bank --cells '${cells//\"/\\\"}'\",\"note\":\"EXP-0075 gate G3/P0-2/G5 cell, HEAD $SHA + uncommitted (dirty)\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $GOALS --starts $STARTS --cells "$cells" --steps 10 --bank-per-env $bank > $E/logs/gates_$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $R/$tag.done
  $P -u $E/code/analyse_gates.py > $E/logs/analyse_gates.log 2>&1; }
cell() { echo "{\"$1\":{$2}}"; }
V='"model":"vanilla64bal","H":1'; N='"model":"ens2bal","H":1'; L='"model":"lf","H":1'
# A. G3 seed gate (cem 1,280); seed-0 ens = g1_ens_cem
run g3_v128_s1 4096 "$(cell g3_v128_s1 '"model":"vanilla128_s1","H":1,"planner":"cem"')"
run g3_z128_s1 4096 "$(cell g3_z128_s1 '"model":"zoom128_s1","H":1,"planner":"cem"')"
run g3_ens_s1 4096 "$(cell g3_ens_s1 '"model":"ens128_s1","H":1,"planner":"cem"')"
run g3_v128_s0 4096 "$(cell g3_v128_s0 '"model":"vanilla128bal","H":1,"planner":"cem"')"
run g3_z128_s0 4096 "$(cell g3_z128_s0 '"model":"zoom128bal","H":1,"planner":"cem"')"
# B. P0-2 budget ladders (tag suffix bXY = X.Y s)
for b in 0.1 0.3 1.0 3.0; do t=p02_ens_b${b/./}; run $t 4096 "$(cell $t "$N,\"planner\":\"cem\",\"budget_s\":$b")"; done
for b in 0.3 1.0 3.0; do t=p02_ens_gd_b${b/./}; run $t 4096 "$(cell $t "$N,\"planner\":\"cem_gd\",\"budget_s\":$b")"; done
for b in 0.1 0.3 1.0; do t=p02_v64_gd_b${b/./}; run $t 4096 "$(cell $t "$V,\"planner\":\"cem_gd\",\"budget_s\":$b")"; done
for b in 0.1 0.3 1.0; do t=p02_lf_b${b/./}; run $t 4096 "$(cell $t "$L,\"planner\":\"cem\",\"budget_s\":$b")"; done
# C. G5 proposals (cem 1,280)
run g5_v64_fl 4096 "$(cell g5_v64_fl "$V,\"planner\":\"cem\",\"fetch\":true,\"legal_score\":true")"
run g5_ens_fl 4096 "$(cell g5_ens_fl "$N,\"planner\":\"cem\",\"fetch\":true,\"legal_score\":true")"
run g5_ens_fl_spill 4096 "$(cell g5_ens_fl_spill "$N,\"planner\":\"cem\",\"fetch\":true,\"legal_score\":true,\"spill_filter\":1.0")"
echo "$(date) DONE" > $R/queue2.done
