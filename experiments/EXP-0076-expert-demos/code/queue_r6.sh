#!/bin/bash
# round 6: expert beam over the n20 scenarios the feasible run left unsolved, then the slow ones (>= 7 pushes)
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 291550 2>/dev/null; do sleep 15; done
BEAMCFG=$(cat code/r6_beamcfg.txt)
B=experiments/EXP-0076-expert-demos/code/beam.py
code/run.sh E20_unsolved $B --tag E20_unsolved --n 20 --ids $(cat results/ids_beam_n20_unsolved.txt) $BEAMCFG
code/run.sh E20_slow $B --tag E20_slow --n 20 --ids $(cat results/ids_beam_n20_slow.txt) $BEAMCFG
