#!/bin/bash
# EXP-0070 RUN-0004: policy-iteration refit (etc from the time_rerank cell's own episodes) + 0.1 s budget pair. From repo root.
set -e
PY=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0070-time-optimal-mpc
until grep -q "wrote" runs/exp0070_pilot_v0b.log; do sleep 10; done
$PY $E/code/analyse_pilot.py pilot_v0b > $E/results/analysis_pilot_v0b.txt
$PY $E/code/refit_merge.py pilot_v0b time_rerank $E/results/etc_map_v0.json etc_map_v1 > $E/results/etc_map_v1.txt
OMP_NUM_THREADS=4 $PY scripts/run_probe.py --tag exp0070_pilot_v1 --exp EXP-0070 -- $PY $E/code/time_mpc.py --tag pilot_v1 --cells time_rerank --starts 40 41 42 43 44 45 --etc-map $E/results/etc_map_v1.json
OMP_NUM_THREADS=4 $PY scripts/run_probe.py --tag exp0070_pilot_b01 --exp EXP-0070 -- $PY $E/code/time_mpc.py --tag pilot_b01 --cells lyap,time_rerank --budget 0.1 --starts 40 41 42 43 44 45 --etc-map $E/results/etc_map_v0.json
echo FOLLOWUP_DONE
