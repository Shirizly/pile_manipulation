#!/bin/bash
# EXP-0075 closed-loop benchmark queue. One driver call per push mode (chunk = one cell = 4 letters x 8 starts = 32 episodes). <= 10 steps. Atomic per chunk (results/<tag>.json rewritten each step).
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
GOALS="letter_O_w20 letter_T_w20 letter_S_w20 letter_X_w20"
STARTS="40 41 42 43 44 45 46 47"
run() { tag=$1; cells=$2; [ -f $E/results/$tag.done ] || { PYTHONPATH=. $P -u $D --tag $tag --goals $GOALS --starts $STARTS --cells "$cells" --steps 10 > $E/logs/$tag.log 2>&1 && touch $E/results/$tag.done; }; }
L20='{"rank_L20": {"model":"ens2bal","planner":"rank","H":1,"push_len":0.02}, "cem1_L20": {"model":"ens2bal","planner":"cem","H":1,"push_len":0.02}, "cem2_L20": {"model":"ens2bal","planner":"cem","H":2,"push_len":0.02}, "cem4_L20": {"model":"ens2bal","planner":"cem","H":4,"push_len":0.02}}'
LF='{"rank_Lfull": {"model":"ens2bal","planner":"rank","H":1}, "cem1_Lfull": {"model":"ens2bal","planner":"cem","H":1}, "cem2_Lfull": {"model":"ens2bal","planner":"cem","H":2}, "cem4_Lfull": {"model":"ens2bal","planner":"cem","H":4}}'
BASE='{"nfd64_rank_L20": {"model":"vanilla64bal","planner":"rank","H":1,"push_len":0.02}, "nfd64_cem1_L20": {"model":"vanilla64bal","planner":"cem","H":1,"push_len":0.02}, "nfd64_rank_Lfull": {"model":"vanilla64bal","planner":"rank","H":1}, "nfd64_cem1_Lfull": {"model":"vanilla64bal","planner":"cem","H":1}}'
run main_L20 "$L20"
run main_Lfull "$LF"
run base64 "$BASE"
echo "$(date) DONE" > $E/results/queue.done
