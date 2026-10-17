#!/bin/bash
# EXP-0075 gate tests G2 (vanilla64 CEM budget ladder) then G1 (model x planner at 1,280 evals), RAL run plan Tier 1; pre-registration LOG.md 2026-10-15 01:44.
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
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $GOALS --starts $STARTS --steps 10 --bank-per-env $bank --cells '$cells'\",\"note\":\"EXP-0075 gate G1/G2 cell, HEAD $SHA + uncommitted (dirty)\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $GOALS --starts $STARTS --cells "$cells" --steps 10 --bank-per-env $bank > $E/logs/gates_$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $R/$tag.done; }
V='"model":"vanilla64bal","H":1'; N='"model":"ens2bal","H":1'
# G2
run g2_v64_b01 4096 "{\"g2_v64_b01\":{$V,\"planner\":\"cem\",\"budget_s\":0.1}}"
run g2_v64_b03 4096 "{\"g2_v64_b03\":{$V,\"planner\":\"cem\",\"budget_s\":0.3}}"
run g2_v64_b10 8192 "{\"g2_v64_b10\":{$V,\"planner\":\"cem\",\"budget_s\":1.0}}"
# G1 (ens cem = base1F same-session control first)
run g1_ens_cem 4096 "{\"g1_ens_cem\":{$N,\"planner\":\"cem\"}}"
run g1_v64_cem 4096 "{\"g1_v64_cem\":{$V,\"planner\":\"cem\"}}"
run g1_v64_rank 4096 "{\"g1_v64_rank\":{$V,\"planner\":\"rank\",\"rank_n\":1280}}"
run g1_ens_rank 4096 "{\"g1_ens_rank\":{$N,\"planner\":\"rank\",\"rank_n\":1280}}"
run g1_v64_cemgd 4096 "{\"g1_v64_cemgd\":{$V,\"planner\":\"cem_gd\",\"gd_steps\":20}}"
run g1_ens_cemgd 4096 "{\"g1_ens_cemgd\":{$N,\"planner\":\"cem_gd\",\"gd_steps\":20}}"
# G2 bank-size control
run g2_v64_b03_bank8k 8192 "{\"g2_v64_b03_bank8k\":{$V,\"planner\":\"cem\",\"budget_s\":0.3}}"
echo "$(date) DONE" > $R/queue.done
