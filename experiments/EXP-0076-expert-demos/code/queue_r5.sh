#!/bin/bash
# round 5: time-aware re-ranking for the feasible pipeline (36 n20 ablation scenarios), after round 4
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 184284 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
S="scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter"
C="--n 20 --strata $S --first 2 --count 4 --B 1 --k 1 --E 36"
code/run.sh F6_tmu02 $B --tag F6_tmu02 $C --time-mu 0.02
code/run.sh F7_tmu05 $B --tag F7_tmu05 $C --time-mu 0.05
code/run.sh F8_tailrule $B --tag F8_tailrule $C --phases '[{"until_mass":0.8,"n_pool":128,"pop":128},{"n_pool":512,"pop":512,"n_extra":256,"time_mu":0.02}]'
