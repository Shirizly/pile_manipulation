#!/usr/bin/env bash
# EXP-0021 -- generic-LeJEPA encoders at 32x32 input (per direction 2026-09-17:
# match the linear-foresight canonical resolution). Same recipe as the 64px
# seeds in every other respect, so resolution is the only variable.
set -euo pipefail
REPO=/home/alon/Code/pile_manipulation
PY=/home/alon/anaconda3/envs/pme/bin/python
OUT="$REPO/experiments/temp/exp0021-encoders32"
mkdir -p "$OUT"; cd "$REPO"; export PYTHONPATH=.; export OMP_NUM_THREADS=4
for SEED in 0 1 2; do
    DST="$OUT/lejepa32_seed${SEED}.pt"
    [ -e "$DST" ] && { echo "seed $SEED done, skip"; continue; }
    "$PY" -u experiments/EXP-0021-five-way-ranking-comparison/code/train_encoder_res.py \
        --seed "$SEED" --res 32 --sigreg-on-z 0.02 --lambda-sigreg 0.02 \
        --sean-files 60 --epochs 12 --steps-per-epoch 250 --bs 192 --lr 1e-3 \
        --latent-dim 256 --proj-dim 256 --n-res-blocks 2 \
        --out "$DST" 2>&1 | tee "$OUT/train32_seed${SEED}.log"
done
echo "=== done"; ls -d "$OUT"/*.pt
