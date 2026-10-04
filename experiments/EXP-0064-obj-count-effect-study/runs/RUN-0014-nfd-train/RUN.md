# RUN-0014 — NFD (EXP-0062 recipe) on DS-0021 image masks, seed 0 → MODEL-0014

- what: `Baselines/NFD/configs/nfd_3ch_flex_mask_ds0021.yaml` (copy in this dir) = EXP-0062's
  `nfd_3ch_flex_mask_v2.yaml` with only the dataset instance (DS-0021 train/val; Trainer test pass DS-0022) and
  log_dir changed: 3-ch UNet [4,8,16] residual, MSE, Adam 1e-4 constant, batch 32, x8 aug, AMP, grad clip 1,
  250-epoch cap + plateau stop (patience 20, min rel delta 0.5 %), seed 0, full-state checkpoints (resume `--resume`).
  Data: DS-0021 train 15,193 / val 1,741 kept rows (dataset build verified on CPU before launch).
- GPU gating: the overnight GNN queue (`code/overnight_queue.sh`, RUN-0008..0011 + RUN-0012 evals) owns the GPU;
  this run must not start before `QUEUE DONE` appears in `artifacts/RUN-0012-eval-overnight/queue_status.log`.
  The chain `code/nfd_after_queue.sh` (launched detached 2026-10-04 00:16 CEST, status in
  `artifacts/RUN-0014-nfd-train/chain_status.log`) waits for that line, then trains
  (`python scripts/run_probe.py --tag exp0064_run0014_nfd_train --threads 8 -- python Baselines/NFD/train_nfd.py
  Baselines/NFD/configs/nfd_3ch_flex_mask_ds0021.yaml --seed 0`), copies `unet_best.pth` to
  `weights/MODEL-0014-nfd-flex-mask-countgroups-seed0/checkpoint.pth` (+ run_config.yaml, sha256), scores it
  (RUN-0015 `score --models nfd14 --device cuda`) and re-runs RUN-0015 `analyze`.
- commit 3373e65b, dirty.
- run dir: `Baselines/NFD/runs/nfd_3ch_flex_mask_ds0021_seed0/`; log `artifacts/RUN-0014-nfd-train/exp0064_run0014_nfd_train.log`.
- STATUS: done (Outcome below).

## Outcome

- 2026-10-04 05:36 CEST: chain saw `QUEUE DONE` (05:33:51) and started training; RTX 4070 Laptop alone, ~30 s/epoch.
- Plateau stop at epoch 95 (20 epochs without > 0.5 % relative val improvement); best epoch 75, val MSE 0.007288;
  Trainer test pass on DS-0022: hard IoU 0.677, changed-pixel MSE 0.191 (copy baseline 1.0).
- Promoted 06:20 to `weights/MODEL-0014-nfd-flex-mask-countgroups-seed0/checkpoint.pth` (sha256 595030d3...),
  scored (RUN-0015, cuda, 3 s) and analysed; chain `CHAIN DONE` 06:21. DS-0022 slateN_K overall 0.929 [0.924, 0.935],
  accuracy 0.520 [0.510, 0.530].
