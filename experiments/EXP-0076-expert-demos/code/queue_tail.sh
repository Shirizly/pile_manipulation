#!/bin/bash
# tail queue (02:00): after E20_unsolved -> feasible restarts (portfolio, seeds 2 and 3) on the 78 slow n20 scenarios -> dense n50 attempts -> replay verification
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 577960 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
for s in 2 3; do code/run.sh R20_slow_s$s $B --tag R20_slow_s$s --n 20 --ids $(cat results/ids_beam_n20_slow.txt) --B 1 --k 1 --E 78 --seed $s; done
code/run.sh f50d_feas $B --tag f50d_feas --n 50 --strata inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 1 --B 1 --k 1 --E 6 --max-depth 14
