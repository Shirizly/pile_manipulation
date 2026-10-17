#!/bin/bash
# margin test: X5 = X2 config with acceptance at 0.97 (stored as expert_robust), same 12 scenarios; replay X5
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 893808 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh X5_beam_margin97 $B --tag X5_beam_margin97 --n 20 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter --count 2 --E 128 --B 4 --k 8 --mu 0.05 --success 0.97 --pipeline-name expert_robust
code/run.sh replay_X5 experiments/EXP-0076-expert-demos/code/replay.py --n 20 --run X5_beam_margin97 --tag _a
