# MODEL-0014 -- NFD (3-ch UNet, EXP-0062 `nfd_3ch_flex_mask_v2` recipe) on FleX binary image masks, DS-0021 (count-group carrots), seed 0

**Status:** active

**Read this before reusing:** plateau-stopped at epoch 95 of a 250 cap (best epoch 75, val MSE 0.007288 on the DS-0021
val split -- not comparable with MODEL-0008's DS-0020 val loss). One seed. Trained 2026-10-04 05:36-06:20 CEST on an
RTX 4070 Laptop GPU (alone, ~30 s/epoch).

## What this is

MODEL-0008's architecture and recipe (`nfd-unet3ch`, features [4,8,16], residual, MSE on sigmoid, Adam 1e-4 constant,
batch 32, x8 rot/flip aug, AMP, grad clip 1.0, 250-epoch cap + plateau stop patience 20 / 0.5 %), trained on
**DS-0021** (EXP-0064 count-group FleX carrot piles; split = EXP-0064 per-group 90/10 by state; train 15,193 / val 1,741
kept rows). Input occ0 and target occ1 = binary top-down colour-image masks, 64x64 grid +-7.2 FleX units.

## Provenance

- config `Baselines/NFD/configs/nfd_3ch_flex_mask_ds0021.yaml` (copy: EXP-0064 `runs/RUN-0014-nfd-train/config.yaml`)
- `PYTHONPATH=. python -u Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_3ch_flex_mask_ds0021.yaml --seed 0`
  via `scripts/run_probe.py` from `experiments/EXP-0064-obj-count-effect-study/code/nfd_after_queue.sh`
- run dir `Baselines/NFD/runs/nfd_3ch_flex_mask_ds0021_seed0/`; `checkpoint.pth` = its `unet_best.pth` (sha256 in `checkpoint.sha256`)
- commit 3373e65b, dirty

## Load

`eval_report.py --ckpt nfd_randlen=weights/MODEL-0014-nfd-flex-mask-countgroups-seed0/checkpoint.pth`; scored by
EXP-0064 `code/score_image_metrics.py` (model key `nfd14`).
