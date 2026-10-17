#!/bin/bash
# round 3: feasible-pipeline ablations, n20, 36 scenarios (strata x indices 2..5); baseline = f20_all on the same scenarios
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 $(cat logs/f20.pid) 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
S="scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter"
C="--n 20 --strata $S --first 2 --count 4 --B 1 --k 1 --E 36"
code/run.sh F1_v64big $B --tag F1_v64big $C --model vanilla64bal --n-pool 1024 --pop 1024 --gd 20
code/run.sh F2_ensH2ot $B --tag F2_ensH2ot $C --H 2 --pool ot
code/run.sh F3_phaseH2H1 $B --tag F3_phaseH2H1 $C --phases '[{"until_mass":0.75,"H":2,"pool":"ot"},{"H":1,"pool":"fl"}]'
code/run.sh F4_ensbig $B --tag F4_ensbig $C --n-pool 512 --pop 512
code/run.sh F5_v64H2ot $B --tag F5_v64H2ot $C --model vanilla64bal --H 2 --pool ot --n-pool 1024 --pop 1024
