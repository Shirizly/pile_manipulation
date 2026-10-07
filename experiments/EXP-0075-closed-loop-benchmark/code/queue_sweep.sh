#!/bin/bash
# EXP-0075 reward-hacking probe (user 2026-10-08): LOW planner budgets, horizon 1, ensemble model. Evaluations per decision: rank n; cem (pool, pop, iters) -> pool + pop*iters.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
GOALS="letter_O_w20 letter_T_w20 letter_S_w20 letter_X_w20"; STARTS="40 41 42 43 44 45 46 47"
run() { tag=$1; cells=$2; [ -f $E/results/$tag.done ] || { PYTHONPATH=. $P -u $D --tag $tag --goals $GOALS --starts $STARTS --cells "$cells" --steps 10 > $E/logs/$tag.log 2>&1 && touch $E/results/$tag.done; }; }
mk() { suf=$1; pl=$2   # suf: Lfull or L20 ; pl: extra push_len json fragment
  echo "{\"rank32_$suf\": {\"model\":\"ens2bal\",\"planner\":\"rank\",\"H\":1,\"rank_n\":32$pl}, \"rank128_$suf\": {\"model\":\"ens2bal\",\"planner\":\"rank\",\"H\":1,\"rank_n\":128$pl}, \"rank640_$suf\": {\"model\":\"ens2bal\",\"planner\":\"rank\",\"H\":1,\"rank_n\":640$pl}, \"cem128i1_$suf\": {\"model\":\"ens2bal\",\"planner\":\"cem\",\"H\":1,\"n_pool\":128,\"pop\":128,\"iters\":1$pl}, \"cem256i1_$suf\": {\"model\":\"ens2bal\",\"planner\":\"cem\",\"H\":1,\"n_pool\":256,\"pop\":256,\"iters\":1$pl}, \"cem256i2_$suf\": {\"model\":\"ens2bal\",\"planner\":\"cem\",\"H\":1,\"n_pool\":256,\"pop\":256,\"iters\":2$pl}, \"cem128i4_$suf\": {\"model\":\"ens2bal\",\"planner\":\"cem\",\"H\":1,\"n_pool\":128,\"pop\":128,\"iters\":4$pl}}"; }
run sweep_Lfull "$(mk Lfull '')"
run sweep_L20 "$(mk L20 ',"push_len":0.02')"
echo "$(date) DONE" > $E/results/queue_sweep.done
