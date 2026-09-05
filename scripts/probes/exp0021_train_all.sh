#!/bin/bash
# EXP-0021: train all 7 UNetFilm cells sequentially on one GPU.
# Each config is configs/training/exp0021_unetfilm_<tag>.yaml, built from
# configs/training/unetfilm_corl_limited_100e_fixedraster.yaml (see that
# commit's message for what was changed). Run via scripts/run_probe.py so
# output is unbuffered, captured to a file, and PID-trackable.
set -e
cd /home/alon/Code/pile_manipulation

TAGS="blind_n5 blind_n10 blind_n20 contact_n5 contact_n10 contact_n20 blind_n50"

for tag in $TAGS; do
  echo "=================================================================="
  echo "=== training cell: $tag  ($(date))"
  echo "=================================================================="
  python -u -m training.train "configs/training/exp0021_unetfilm_${tag}.yaml" --no-resume
done
echo "=== ALL 7 CELLS DONE ($(date)) ==="
