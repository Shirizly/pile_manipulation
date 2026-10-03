# MODEL-0008 -- NFD (3-ch UNet, `nfd_randlen` recipe) on FleX binary image masks, DS-0020 **v2**, seed 0

**Status:** active

## What this is

EXP-0062 (rerun of EXP-0061 with the new in-domain training collection): the same architecture and
recipe as MODEL-0005 (`nfd-unet3ch`, features [4,8,16], residual, MSE on sigmoid, Adam 1e-4 constant,
batch 32, x8 rot/flip augmentation, bf16 AMP, grad clip 1.0), trained on **DS-0020 v2** (1997 blob/spread
trajectories in DS-0019's scene; trajectories 0-99 excluded for DS-0019 leakage; splits.json train
1707 / val 190 trajectories = 15,093 / 1,665 kept rows). Input occ0 AND target occ1 = the **binary
top-down image masks** (`FlexData/image_mask.py`, `occ_source="image_mask"`), 64x64 grid +-7.2 FleX
units, row = X = x_flex, col = Y = -z_flex, plate 2.4 x 0.225 units. Particles never a model input.

## Read this before reusing

- **Stopped by the plateau rule at epoch 129 of a 250 cap** (patience 20, `patience_min_rel_delta`
  0.005: 20 epochs without a > 0.5 % relative val improvement); best epoch 128, val MSE 0.010244.
  Best-val improvement over the last 20 epochs at the stop was 0.3-0.4 % -- comparable to the
  epoch-to-epoch val noise (sd 0.5 %), still creeping down, not strictly flat. Curve:
  `experiments/EXP-0062-*/figures/nfd_v2_seed0_convergence.png`.
- Val loss is on the DS-0020 **v2** val split -- not comparable with MODEL-0005's v1 val loss.
- Native FleX units, image-mask input: meaningful only on the FleX grid (DS-0019 / DS-0020 instance configs).
- Score with ground truth = binary image mask (`eval_report.py --corpora flex_ds0019_mask --truth-scoring image`).

## Provenance

- commit d72bb304 (dirty tree: `training/trainer.py` gained `patience_min_rel_delta` in this run;
  plus the EXP-0061 uncommitted files), 2026-10-02 14:20-15:41 CEST, single RTX 4070 Laptop GPU (alone; ~37 s/epoch)
- `PYTHONPATH=. python -u Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_3ch_flex_mask_v2.yaml --seed 0`
  (never interrupted; full-state checkpointing on, resume with `--resume`)
- data: DS-0020 v2 (`configs/dataset/flex_ds0020v2_train_ds0019_test.yaml`), image-mask cache
  `datasets/DS-0020-*/cache/image_masks.npz`
- training run dir: `Baselines/NFD/runs/nfd_3ch_flex_mask_v2_seed0/` (unet_best/last/epoch_*, last_state.pt, tfevents);
  log `experiments/EXP-0062-flex-v2-train-rerun/logs/train_nfd_seed0.log`

## Contents

`checkpoint.pth` = `Baselines/NFD/runs/nfd_3ch_flex_mask_v2_seed0/unet_best.pth` (sha256 9f8c5790...,
epoch 128); `config.yaml` (resolved run config); `model_card.yaml`. Load with
`Baselines/NFD/predictor.py::build_predictor`, or `eval_report.py --ckpt nfd_randlen=weights/MODEL-0008-nfd-flex-mask-v2-seed0/checkpoint.pth`.

## Regeneration

The command above, ~80 min alone on the GPU. Not bitwise reproducible (cuDNN/AMP nondeterminism).

## Test history

See `tests.md`.
