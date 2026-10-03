# MODEL-0005 -- NFD (3-ch UNet, `nfd_randlen` recipe) on FleX binary image masks, seed 0

**Status:** active

> **Trained on DS-0020 v1 (the OLD payload), not the current DS-0020.** On 2026-10-02 the user
> replaced DS-0020's payload with a new blob/spread collection (v2, `datasets/DS-0020-*/DATASET.md`).
> This instance was fit on the v1 data (2000 trajectories from ONE uniform-spread start state), now archived at
> `datasets/DS-0020-training-data-flex-N864/old_data/` (raw) + `old_data/_ported_v1/` (config.yaml, splits.json,
> cache/ incl. image_masks.npz, DATASET.md). Every DS-0020 path, row count and split below refers to v1; load it
> with `old_data/_ported_v1/config.yaml` (its `config.yaml` was repointed there). Its numbers do not transfer to v2.

## What this is

EXP-0061 (FleX rerun of EXP-0001): the EXP-0001 `nfd_randlen` architecture/recipe
(`nfd-unet3ch`, features [4,8,16], residual, MSE on sigmoid, Adam 1e-4, batch 32, x8 rot/flip
augmentation, bf16 AMP) trained on DS-0020 with input occ0 AND target occ1 = the **binary top-down
image masks** (`FlexData/image_mask.py`, `occ_source="image_mask"`), 64x64 grid +-7.2 FleX units,
row = X = x_flex, col = Y = -z_flex, plate 2.4 x 0.225 units. One of 3 seeds (MODEL-0005/0006/0007
= seeds 0/1/2, identical recipe).

## Read this before reusing

- **80 epochs, not the presentation-matched 150** (budget cut, see the config header item 5); val
  loss was still drifting down ~1 %/10 epochs at the end (best epoch 80 of 80), so this
  is not a converged model.
- Native FleX units, image-mask input: meaningful only on the FleX grid (DS-0019/DS-0020 instance
  configs), never on Genesis corpora.
- Score with ground truth = binary image mask (`eval_report.py --corpora flex_ds0019_mask
  --truth-scoring image`); the default `--truth-scoring soft` is the particle splat.

## Provenance

- commit d72bb304 (dirty tree), 2026-10-01
- `PYTHONPATH=. python -u Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_3ch_flex_mask.yaml
  --seed 0 --override training.shuffle_seed=0 output.log_dir=Baselines/NFD/runs/nfd_3ch_flex_mask_seed0` (interrupted twice on purpose
  and continued with `--resume`, see `experiments/EXP-0061-*/RUNS_nfd.md`)
- data: DS-0020 train 17,837 / val 1,991 rows (splits.json, by trajectory); image-mask cache
  `datasets/DS-0020-*/old_data/_ported_v1/cache/image_masks.npz` (DS-0020 **v1**, archived 2026-10-02) (FlexData/image_mask.py sha256 6a243bb8...)
- training run dir: `Baselines/NFD/runs/nfd_3ch_flex_mask_seed0/` (unet_best/unet_last/unet_epoch_*, last_state.pt, tfevents)

## Contents

`checkpoint.pth` = `Baselines/NFD/runs/nfd_3ch_flex_mask_seed0/unet_best.pth` (ModelTrainingWrapper state dict, epoch 80,
val loss 0.008696); `config.yaml` (resolved run config); `model_card.yaml`. Load with
`Baselines/NFD/predictor.py::build_predictor` (env `NFD_CKPT`, or `eval_report.py --ckpt
nfd_randlen=weights/MODEL-0005-nfd-flex-mask-seed0/checkpoint.pth`).

## Regeneration

The command above (~2 h with 3 seeds sharing the GPU; ~50 s/epoch alone). Not bitwise
reproducible (cuDNN/AMP nondeterminism: same-seed epoch-1 loss differs ~3e-4 relative).

## Test history

See `tests.md`.
