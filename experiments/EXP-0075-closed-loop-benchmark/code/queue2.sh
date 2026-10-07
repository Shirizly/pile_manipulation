#!/bin/bash
# EXP-0075 follow-ups (run concurrently with queue.sh; planners are iteration-bounded, not wall-clock bounded, so sharing the GPU does not change outcomes).
#  members : single-model cells (zoom128 / vanilla128, balance on) at cem H=1, 20 mm  -> is the ensemble the best closed-loop model?
#  budget  : cem H=1 and H=2 with 4x the evaluations (pop 1024, pool 1024)           -> is the H=2/H=4 deficit a search-budget problem?
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
GOALS="letter_O_w20 letter_T_w20 letter_S_w20 letter_X_w20"
STARTS="40 41 42 43 44 45 46 47"
run() { tag=$1; cells=$2; [ -f $E/results/$tag.done ] || { PYTHONPATH=. $P -u $D --tag $tag --goals $GOALS --starts $STARTS --cells "$cells" --steps 10 > $E/logs/$tag.log 2>&1 && touch $E/results/$tag.done; }; }
MEM='{"zoom_cem1_L20": {"model":"zoom128bal","planner":"cem","H":1,"push_len":0.02}, "vanilla_cem1_L20": {"model":"vanilla128bal","planner":"cem","H":1,"push_len":0.02}}'
BUD='{"cem1x4_L20": {"model":"ens2bal","planner":"cem","H":1,"push_len":0.02,"n_pool":1024,"pop":1024,"iters":4}, "cem2x4_L20": {"model":"ens2bal","planner":"cem","H":2,"push_len":0.02,"n_pool":1024,"pop":1024,"iters":4}}'
run members_L20 "$MEM"
run budget_L20 "$BUD"
echo "$(date) DONE" > $E/results/queue2.done
