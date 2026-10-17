#!/bin/bash
# dense n50 (inbetween / piled), feasible planner, 1 scenario per (init, goal kind) = 6 scenarios, max 14 pushes; after queue_main
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 353297 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh f50d_feas $B --tag f50d_feas --n 50 --strata inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --count 1 --B 1 --k 1 --E 6 --max-depth 14
