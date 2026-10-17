#!/bin/bash
# round 4 (after round 3): n50 scattered feasible baseline (45 scenarios), then expert-beam ablations on the 12 scattered/inbetween pilot scenarios (paired with p1C)
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 $(cat logs/queue_r3.pid) 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh f50s_feas $B --tag f50s_feas --n 50 --strata scattered-disc scattered-square scattered-letter --count 15 --B 1 --k 1 --E 45
X="--n 20 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter --count 2 --E 128"
code/run.sh X1_beam8x4 $B --tag X1_beam8x4 $X --B 8 --k 4
code/run.sh X2_beam4x8_mu05 $B --tag X2_beam4x8_mu05 $X --B 4 --k 8 --mu 0.05
code/run.sh X3_beam4x8_H2ot $B --tag X3_beam4x8_H2ot $X --B 4 --k 8 --H 2 --pool ot
