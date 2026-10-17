#!/bin/bash
# EXP-0075 OT-seeded candidate pool test (LOG.md 2026-10-11, results/ot_pool/). One driver call per (H, budget) group of three pool modes (fl = control, ot, mix);
# results/ot_pool/<group>.json rewritten each step (the driver resumes incomplete cells); ledger lines in experiments/COMMANDS.jsonl. Priority order a..d (brief).
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
R=$E/results/ot_pool
GOALS="letter_O_w20 letter_T_w20 letter_S_w20 letter_X_w20"
STARTS="40 41 42 43 44 45 46 47"
SHA=$(git rev-parse --short HEAD)
C='"model":"ens2bal","planner":"cem","fetch":true,"legal_score":true,"gd_steps":20'
run() { tag=$1; cells=$2; [ -f $R/$tag.done ] && return; rid="EXP-0075/ot_pool/$tag-$(date +%s)"; t0=$(date +%s)
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $GOALS --starts $STARTS --steps 10 --cells <see queue_ot_pool.sh>\",\"note\":\"EXP-0075 OT-pool cells, HEAD $SHA + uncommitted (dirty)\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $GOALS --starts $STARTS --cells "$cells" --steps 10 > $E/logs/ot_pool_$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $R/$tag.done; }
# a. H=1, 320 evaluations (pool 64 + 4 x 64)
run H1_E320 "{\"ot_fl_H1_E320\":{$C,\"H\":1,\"pool_mode\":\"fl\",\"n_pool\":64,\"pop\":64,\"n_extra\":16}, \"ot_ot_H1_E320\":{$C,\"H\":1,\"ot\":true,\"pool_mode\":\"ot\",\"n_pool\":64,\"pop\":64}, \"ot_mix_H1_E320\":{$C,\"H\":1,\"ot\":true,\"pool_mode\":\"mix\",\"n_pool\":64,\"pop\":64}}"
# b. H=2, 1,280 evaluations, cumspill objective
run H2_E1280 "{\"ot_fl_H2_E1280\":{$C,\"H\":2,\"obj_mode\":\"cumspill\",\"pool_mode\":\"fl\"}, \"ot_ot_H2_E1280\":{$C,\"H\":2,\"obj_mode\":\"cumspill\",\"ot\":true,\"pool_mode\":\"ot\"}, \"ot_mix_H2_E1280\":{$C,\"H\":2,\"obj_mode\":\"cumspill\",\"ot\":true,\"pool_mode\":\"mix\"}}"
# c. H=4, 1,280 evaluations, cumspill objective
run H4_E1280 "{\"ot_fl_H4_E1280\":{$C,\"H\":4,\"obj_mode\":\"cumspill\",\"pool_mode\":\"fl\"}, \"ot_ot_H4_E1280\":{$C,\"H\":4,\"obj_mode\":\"cumspill\",\"ot\":true,\"pool_mode\":\"ot\"}, \"ot_mix_H4_E1280\":{$C,\"H\":4,\"obj_mode\":\"cumspill\",\"ot\":true,\"pool_mode\":\"mix\"}}"
# d. H=1, 1,280 evaluations (fl + GD is also the GD-vs-no-GD check against fl1F)
run H1_E1280 "{\"ot_fl_H1_E1280\":{$C,\"H\":1,\"pool_mode\":\"fl\"}, \"ot_ot_H1_E1280\":{$C,\"H\":1,\"ot\":true,\"pool_mode\":\"ot\"}}"
echo "$(date) DONE" > $R/queue.done
