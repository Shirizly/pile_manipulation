#!/usr/bin/env bash
# EXP-0021 / RUN-0001 -- generic-LeJEPA encoder seed replicates.
#
# Recipe is EXP-0020's, byte-for-byte, so that seed 0 does NOT need retraining:
# experiments/EXP-0020-lejepa-sigreg-on-z/artifacts/RUN-0001/encoder_lejepa.pt
# IS the seed-0 member of this set (same script, same flags, same 252 files,
# same 12x250 steps / bs 192 / lr 1e-3 / latent 256 / proj 256, lambda_sigreg
# 0.02 on both p and z). Only seeds 1 and 2 are new.
#
# Training corpus is EXP-0019/0020's file set (overnight_randlen_train + 60
# strided Sean files), NOT DS-0002. Deliberate: it keeps these encoders
# directly comparable to the frozen-random floor and the lejepa-debug table,
# which are the baselines P1 is measured against. Recorded as a deviation from
# the record's provenance.data, to be noted in "What was actually run".
set -euo pipefail

REPO=/home/alon/Code/pile_manipulation
PY=/home/alon/anaconda3/envs/pme/bin/python
OUT="$REPO/experiments/temp/exp0021-encoders"
TRAIN="$REPO/experiments/EXP-0019-lejepa-encoder-pushlen-switched/code/train_encoder.py"

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH=.
export OMP_NUM_THREADS=4

for SEED in 1 2; do
    DST="$OUT/lejepa_generic_seed${SEED}.pt"
    if [ -f "$DST" ]; then echo "seed $SEED already done, skipping"; continue; fi
    echo "=== generic LeJEPA, seed $SEED -> $DST"
    # python -u: buffered stdout through a redirect makes a healthy job look hung.
    "$PY" -u "$TRAIN" \
        --seed "$SEED" \
        --sigreg-on-z 0.02 \
        --lambda-sigreg 0.02 \
        --sean-files 60 \
        --epochs 12 --steps-per-epoch 250 --bs 192 --lr 1e-3 \
        --latent-dim 256 --proj-dim 256 --res 64 --n-res-blocks 2 \
        --out "$DST" 2>&1 | tee "$OUT/train_seed${SEED}.log"
done

echo "=== all seeds done"
ls -la "$OUT"
