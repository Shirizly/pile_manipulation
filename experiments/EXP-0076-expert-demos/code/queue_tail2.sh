#!/bin/bash
# tail queue v2 (02:20): cheaper beam B2k8 (16 sims / decision, 8 concurrent) on the 59 remaining unsolved n20 -> feasible restarts (seeds 2, 3) on the 78 slow -> 6 dense n50
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh E20u_b2k8 $B --tag E20u_b2k8 --n 20 --ids $(cat results/ids_beam_n20_unsolved_rest.txt) --B 2 --k 8 --E 128 --max-depth 14 --mu 0.05
for s in 2 3; do code/run.sh R20_slow_s$s $B --tag R20_slow_s$s --n 20 --ids $(cat results/ids_beam_n20_slow.txt) --B 1 --k 1 --E 78 --seed $s; done
code/run.sh f50d_feas $B --tag f50d_feas --n 50 --strata inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 1 --B 1 --k 1 --E 6 --max-depth 14
