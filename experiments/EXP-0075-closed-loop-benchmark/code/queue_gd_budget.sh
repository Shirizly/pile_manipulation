#!/bin/bash
# waits for the 10 gd_budget plans, then runs the simulator replay and the analysis
cd "$(dirname "$0")/../../.."; P=/home/alon/anaconda3/envs/pme/bin/python; E=experiments/EXP-0075-closed-loop-benchmark; export PYTHONPATH=.
until [ "$(ls $E/results/gd_budget/*.json 2>/dev/null | wc -l)" -ge 10 ]; do sleep 30; done
$P -u $E/code/sim_gd_budget.py > $E/logs/sim_gd_budget.log 2>&1; $P $E/code/analyse_gd_budget.py > $E/logs/analyse_gd_budget.log 2>&1; echo finished >> $E/logs/analyse_gd_budget.log
