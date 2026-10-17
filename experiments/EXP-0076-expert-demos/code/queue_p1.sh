#!/bin/bash
# pilot round 1 (n20, 18 scenarios = first 2 per stratum): A feasible (B1 k1), B sim-verified greedy (B1 k16), C beam (B4 k8)
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh p1A_feas $B --tag p1A_feas --n 20 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 2 --B 1 --k 1 --E 32
code/run.sh p1B_greedy16 $B --tag p1B_greedy16 --n 20 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 2 --B 1 --k 16 --E 128
code/run.sh p1C_beam4x8 $B --tag p1C_beam4x8 --n 20 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 2 --B 4 --k 8 --E 128
