#!/bin/bash
# final: open-loop replay verification of every stored n20 demo (expert, feasible), one pass each, after the dense n50 run
cd /home/alon/Code/pile_manipulation/experiments/EXP-0076-expert-demos
while kill -0 743762 2>/dev/null; do sleep 15; done
R=experiments/EXP-0076-expert-demos/code/replay.py
code/run.sh replay_final_expert_n20 $R --n 20 --pipeline expert --tag _final
code/run.sh replay_final_feasible_n20 $R --n 20 --pipeline feasible --tag _final
