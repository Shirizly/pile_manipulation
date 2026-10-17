#!/bin/bash
# EXP-0070 session 2 queue (after the 2026-10-05 power loss). From repo root.
PY=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0070-time-optimal-mpc
while kill -0 19016 2>/dev/null; do sleep 10; done
$PY $E/code/analyse_pilot.py pilot_v0b > $E/results/analysis_pilot_v0b.txt 2>&1
OMP_NUM_THREADS=4 $PY scripts/run_probe.py --tag exp0070_chain_C --exp EXP-0070 -- $PY $E/code/chain_probe.py --mode chain --n-states 24
$PY $E/code/chain_analyse.py > $E/results/chain_analyse.txt 2>&1
$PY $E/code/chain_model_eval.py nfd_3ch_randlen > $E/results/chain_model_eval.txt 2>&1
$PY $E/code/refit_merge.py pilot_v0b time_rerank $E/results/etc_map_v0.json etc_map_v1 > $E/results/etc_map_v1.txt 2>&1
OMP_NUM_THREADS=4 $PY scripts/run_probe.py --tag exp0070_pilot_v1 --exp EXP-0070 -- $PY $E/code/time_mpc.py --tag pilot_v1 --cells time_rerank --starts 40 41 42 43 44 45 --etc-map $E/results/etc_map_v1.json
OMP_NUM_THREADS=4 $PY scripts/run_probe.py --tag exp0070_pilot_b01 --exp EXP-0070 -- $PY $E/code/time_mpc.py --tag pilot_b01 --cells lyap,time_rerank --budget 0.1 --starts 40 41 42 43 44 45 --etc-map $E/results/etc_map_v0.json
echo QUEUE2_DONE
