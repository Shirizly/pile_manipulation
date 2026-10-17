#!/bin/bash
# pilot round 1b (n50, 18 scenarios): A feasible (B1 k1, E32), B sim-verified greedy top-8 (B1 k8, E64). Waits for queue_p1.
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while pgrep -f queue_p1.sh >/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh p2A_feas50 $B --tag p2A_feas50 --n 50 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 2 --B 1 --k 1 --E 32
code/run.sh p2B_greedy8_50 $B --tag p2B_greedy8_50 --n 50 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 2 --B 1 --k 8 --E 64
