#!/bin/bash
# tail-switch experiment: continue each recorded 0.8-crossing state (96 = 64 ens-bulk + 32 v64-bulk) with the ens and the v64 planner, 2 seeds each, max 8 tail pushes
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
B=experiments/EXP-0076-expert-demos/code/beam.py
C="--scenarios experiments/EXP-0076-expert-demos/results/tail_scenarios.json --all --n 20 --B 1 --k 1 --E 96 --max-depth 8 --no-store"
code/run.sh TS_ens_s0 $B --tag TS_ens_s0 $C --seed 0
code/run.sh TS_v64_s0 $B --tag TS_v64_s0 $C --seed 0 --model vanilla64bal --n-pool 1024 --pop 1024
code/run.sh TS_ens_s1 $B --tag TS_ens_s1 $C --seed 1
code/run.sh TS_v64_s1 $B --tag TS_v64_s1 $C --seed 1 --model vanilla64bal --n-pool 1024 --pop 1024
