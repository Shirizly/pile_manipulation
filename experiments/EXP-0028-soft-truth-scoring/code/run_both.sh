#!/bin/bash
# EXP-0028: the eval_report harness re-run with SOFT ground-truth scoring, default and many goal sets.
set -e
cd /home/alon/Code/pile_manipulation
for GS in default many; do
  mkdir -p experiments/EXP-0028-soft-truth-scoring/artifacts/RUN-0001-soft-$GS
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u Baselines/common/eval_report.py --truth-scoring soft --goal-set $GS \
     --out-prefix experiments/EXP-0028-soft-truth-scoring/artifacts/RUN-0001-soft-$GS/report --models nfd_randlen,nfd_warped_randlen,nfd_warped_randlen_flipaug,nfd_warped_randlen_flipaug_epoch30,nfd_residual_warped_flipaug_randlen,nfd_residual_worldframe_noaug_ep43,linear_switched_res32,linear_switched_res64,linear_single_res32,linear_single_res64,gnn_l20l40,gnn_randlen_n30 --corpora L20mm,L40mm,randlen_test
done
