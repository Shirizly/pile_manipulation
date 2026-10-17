#!/bin/bash
# EXP-0070 loop: wait for iter1, analyse, apply the pre-declared lam2 rule, refit m1, run iter2.
cd /home/alon/Code/pile_manipulation
PY=/home/alon/anaconda3/envs/pme/bin/python; E=experiments/EXP-0070-time-optimal-mpc
while pgrep -f "tag iter1 " >/dev/null; do sleep 15; done
$PY $E/code/analyse_pilot.py iter1 > $E/results/analysis_iter1.txt 2>&1
$PY $E/code/equal_time.py iter1 > $E/results/equal_time_iter1.txt 2>&1
$PY $E/code/loop_step.py iter1 $E/results/etc_map_m0.json etc_map_m1 > $E/results/loop_step_iter1.txt 2>&1 || exit 1
LAM2=$(grep LAM2 $E/results/loop_step_iter1.txt | awk '{print $2}')
OMP_NUM_THREADS=4 $PY scripts/run_probe.py --tag exp0070_iter2 --exp EXP-0070 -- $PY $E/code/time_mpc.py --tag iter2 --cells lyap,blend$LAM2 --starts 40 41 42 43 44 45 --map2 $E/results/etc_map_m1.json --map-key "gap|td"
$PY $E/code/analyse_pilot.py iter2 > $E/results/analysis_iter2.txt 2>&1
$PY $E/code/equal_time.py iter2 > $E/results/equal_time_iter2.txt 2>&1
echo LOOP_DONE
