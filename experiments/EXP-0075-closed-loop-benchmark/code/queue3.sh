#!/bin/bash
# EXP-0075 follow-up: horizon planners with the prefix-min objective (value of the best state along the predicted sequence), full push range and 20 mm; waits for queue2 to free the GPU.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
GOALS="letter_O_w20 letter_T_w20 letter_S_w20 letter_X_w20"; STARTS="40 41 42 43 44 45 46 47"
run() { tag=$1; cells=$2; [ -f $E/results/$tag.done ] || { PYTHONPATH=. $P -u $D --tag $tag --goals $GOALS --starts $STARTS --cells "$cells" --steps 10 > $E/logs/$tag.log 2>&1 && touch $E/results/$tag.done; }; }
PM='{"cem2pm_Lfull": {"model":"ens2bal","planner":"cem","H":2,"obj_mode":"prefixmin"}, "cem4pm_Lfull": {"model":"ens2bal","planner":"cem","H":4,"obj_mode":"prefixmin"}, "cem2pm_L20": {"model":"ens2bal","planner":"cem","H":2,"push_len":0.02,"obj_mode":"prefixmin"}}'
until [ -f $E/results/main_Lfull.done ]; do sleep 60; done
run prefixmin "$PM"
echo "$(date) DONE" > $E/results/queue3.done
