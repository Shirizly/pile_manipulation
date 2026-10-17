#!/bin/bash
# EXP-0070 RUN-0012: ablation -- is iteration 1's gain the time term or the mass-gap re-rank alone? (after the loop queue)
cd /home/alon/Code/pile_manipulation
PY=/home/alon/anaconda3/envs/pme/bin/python; E=experiments/EXP-0070-time-optimal-mpc
until grep -q LOOP_DONE runs/exp0070_queue_loop.log; do sleep 15; done
OMP_NUM_THREADS=4 $PY scripts/run_probe.py --tag exp0070_ablation --exp EXP-0070 -- $PY $E/code/time_mpc.py --tag ablation --cells lyap,blend1.0,blendnt1.0 --starts 40 41 42 43 44 45 --map2 $E/results/etc_map_m0.json --map-key "gap|td"
$PY $E/code/analyse_pilot.py ablation > $E/results/analysis_ablation.txt 2>&1
$PY $E/code/equal_time.py ablation > $E/results/equal_time_ablation.txt 2>&1
echo ABLATION_DONE
