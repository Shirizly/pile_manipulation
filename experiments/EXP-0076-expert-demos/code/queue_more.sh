#!/bin/bash
# extra (after the replay pass): feasible restart (seed 4) on the still-unsolved n20 scenarios, then 12 more dense n50 scenarios (indices 1-2)
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 798899 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh R20_unsolved_s4 $B --tag R20_unsolved_s4 --n 20 --ids $(cat results/ids_n20_still_unsolved.txt) --B 1 --k 1 --E 32 --seed 4 --max-depth 16
code/run.sh f50d_feas2 $B --tag f50d_feas2 --n 50 --strata inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --first 1 --count 2 --B 1 --k 1 --E 12 --max-depth 14
