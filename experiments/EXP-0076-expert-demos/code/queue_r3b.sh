#!/bin/bash
# noise floor: baseline feasible config, planner seed 1, same 36 scenarios (after round 3)
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 159555 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
S="scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter"
code/run.sh F0_seed1 $B --tag F0_seed1 --n 20 --strata $S --first 2 --count 4 --B 1 --k 1 --E 36 --seed 1
