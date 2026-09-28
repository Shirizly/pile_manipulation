#!/bin/bash
# EXP-0036: accuracy + slateN (30 goals, soft truth) on randlen_test for the 4 seeds,
# via eval_report's nfd_randlen spec with --ckpt pointed at each seed (NFD_CKPT is overwritten by the loader). CPU (EXP-0039
# holds the GPU). One report per seed, written atomically by eval_report; a finished
# seed is skipped on rerun.
cd /home/alon/Code/pile_manipulation
P=/home/alon/anaconda3/envs/pme/bin/python
OUT=experiments/EXP-0036-seed-noise-floor/results/randlen_test
mkdir -p $OUT
for s in "" _seed1 _seed2 _seed3; do
  tag=seed${s:-_seed0}; tag=${tag#seed_}
  [ -f $OUT/$tag.json ] && { echo "skip $tag"; continue; }
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
    $P -u Baselines/common/eval_report.py --models nfd_randlen --corpora randlen_test --goal-set many \
    --device cpu --truth-scoring soft --no-reference --out-prefix $OUT/$tag \
    --ckpt nfd_randlen=Baselines/NFD/runs/nfd_3ch_randlen$s/unet_best.pth
  echo "done $tag $(date)"
done
