#!/bin/bash
# after p2A (PID 67792): p1B resume, p1C, p2B
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 67792 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh p1B_greedy16 $B --tag p1B_greedy16 --n 20 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 2 --B 1 --k 16 --E 128
code/run.sh p1C_beam4x8 $B --tag p1C_beam4x8 --n 20 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 2 --B 4 --k 8 --E 128
code/run.sh p2B_greedy8_50 $B --tag p2B_greedy8_50 --n 50 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 2 --B 1 --k 8 --E 64
