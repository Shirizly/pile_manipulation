#!/bin/bash
# re-check beam test: X4 = X2 config + --recheck on the same 12 scenarios; replay X2 and X4 solved records; then 6 more dense n50
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 833823 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh X4_beam4x8_mu05_recheck $B --tag X4_beam4x8_mu05_recheck --n 20 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter --count 2 --E 128 --B 4 --k 8 --mu 0.05 --recheck
R=experiments/EXP-0076-expert-demos/code/replay.py
code/run.sh replay_X2 $R --n 20 --run X2_beam4x8_mu05 --tag _a
code/run.sh replay_X4 $R --n 20 --run X4_beam4x8_mu05_recheck --tag _a
code/run.sh f50d_feas2 $B --tag f50d_feas2 --n 50 --strata inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter --first 1 --count 1 --B 1 --k 1 --E 6 --max-depth 14
