#!/bin/bash
# master queue (rebuilt 22:45): after f50s_feas -> expert-beam ablations X1, X2 -> feasible time-aware / tail-rule cells F6-F8 -> expert beam on n20 (E20_*)
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 307331 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
S="scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter"
X="--n 20 --strata scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter --count 2 --E 128"
code/run.sh X1_beam8x4 $B --tag X1_beam8x4 $X --B 8 --k 4
code/run.sh X2_beam4x8_mu05 $B --tag X2_beam4x8_mu05 $X --B 4 --k 8 --mu 0.05
C="--n 20 --strata $S --first 2 --count 4 --B 1 --k 1 --E 36"
code/run.sh F6_tmu02 $B --tag F6_tmu02 $C --time-mu 0.02
code/run.sh F7_tmu05 $B --tag F7_tmu05 $C --time-mu 0.05
code/run.sh F8_tailrule $B --tag F8_tailrule $C --phases '[{"until_mass":0.8,"n_pool":128,"pop":128},{"n_pool":512,"pop":512,"n_extra":256,"time_mu":0.02}]'
BEAMCFG=$(cat code/r6_beamcfg.txt)
code/run.sh E20_unsolved $B --tag E20_unsolved --n 20 --ids $(cat results/ids_beam_n20_unsolved.txt) $BEAMCFG
code/run.sh E20_slow $B --tag E20_slow --n 20 --ids $(cat results/ids_beam_n20_slow.txt) $BEAMCFG
