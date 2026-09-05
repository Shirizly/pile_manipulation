#!/bin/bash
# EXP-0021: evaluate all 7 trained cells (swept-region metric, baselines,
# contact stratification, action ablation). Run after
# scripts/probes/exp0021_train_all.sh has produced all 7 checkpoints.
set -e
cd /home/alon/Code/pile_manipulation

declare -A CFG
CFG[blind_n5]=configs/dataset/genesis_granularity_blind_n5.yaml
CFG[blind_n10]=configs/dataset/genesis_granularity_blind_n10.yaml
CFG[blind_n20]=configs/dataset/genesis_granularity_blind_n20.yaml
CFG[contact_n5]=configs/dataset/genesis_granularity_contact_n5.yaml
CFG[contact_n10]=configs/dataset/genesis_granularity_contact_n10.yaml
CFG[contact_n20]=configs/dataset/genesis_granularity_contact_n20.yaml
CFG[blind_n50]=configs/dataset/genesis_granularity_blind_n50.yaml

for tag in blind_n5 blind_n10 blind_n20 contact_n5 contact_n10 contact_n20 blind_n50; do
  echo "=================================================================="
  echo "=== evaluating cell: $tag  ($(date))"
  echo "=================================================================="
  PYTHONPATH=. python -u scripts/probes/exp0021_eval.py \
    "${CFG[$tag]}" "runs_granularity/unetfilm_${tag}" --tag "$tag"
done
echo "=== ALL 7 EVALS DONE ($(date)) ==="
