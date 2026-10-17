#!/bin/bash
PY=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0070-time-optimal-mpc
while pgrep -f "tag pilot_v1" >/dev/null; do sleep 10; done
$PY $E/code/analyse_pilot.py pilot_v1 > $E/results/analysis_pilot_v1.txt 2>&1
OMP_NUM_THREADS=4 $PY scripts/run_probe.py --tag exp0070_chain_C --exp EXP-0070 -- $PY $E/code/chain_probe.py --mode chain --n-states 24 && $PY $E/code/chain_analyse.py > $E/results/chain_analyse.txt 2>&1 && $PY $E/code/chain_model_eval.py nfd_3ch_randlen > $E/results/chain_model_eval.txt 2>&1
OMP_NUM_THREADS=4 $PY scripts/run_probe.py --tag exp0070_pilot_b01 --exp EXP-0070 -- $PY $E/code/time_mpc.py --tag pilot_b01 --cells lyap,time_rerank --budget 0.1 --starts 40 41 42 43 44 45 --etc-map $E/results/etc_map_v0.json
$PY $E/code/analyse_pilot.py pilot_b01 > $E/results/analysis_pilot_b01.txt 2>&1
echo QUEUE3_DONE
