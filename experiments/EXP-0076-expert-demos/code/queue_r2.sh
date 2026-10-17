#!/bin/bash
# round 2: bulk n20 feasible baseline over all 270 n20 scenarios (ens2bal fl H1 CEM 1280 + GD 20), after p1C finishes
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 $(cat logs/p1C.pid) 2>/dev/null; do sleep 15; done
B=experiments/EXP-0076-expert-demos/code/beam.py
S="scattered-disc scattered-square scattered-letter inbetween-disc inbetween-square inbetween-letter piled-disc piled-square piled-letter"
code/run.sh f20_all $B --tag f20_all --n 20 --strata $S --count 30 --B 1 --k 1 --E 64
