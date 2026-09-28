#!/bin/bash
# EXP-0036 (TODO H1): 3 additional seeds of the world-frame NFD baseline, sequential
# (one GPU). Each run resumes from its own checkpoint if cut off (Trainer resume).
set -e
cd /home/alon/Code/pile_manipulation
for k in 1 2 3; do
  PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u Baselines/NFD/train_nfd.py \
     experiments/EXP-0036-seed-noise-floor/configs/nfd_3ch_randlen_seed$k.yaml --seed $k
done
