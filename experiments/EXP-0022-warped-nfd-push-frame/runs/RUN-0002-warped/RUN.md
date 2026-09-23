# RUN-0002 -- warped, no wall channel (arm 2 of 3)

Config: `Baselines/NFD/configs/nfd_train_warped_L20mm_pilot.yaml` (copied in as
`config.yaml`). Same NFD UNet architecture as RUN-0001, but predicting in the canonical
push frame (`transforms.functional.push_frame_roundtrip`) via `nfd-unet-warped` /
`nfd-genesis-3ch-warped`, `plate_mode: canonical`, `wall_channel: false`, `in_channels: 3`.
Same dataset (`slates_multistep/n20_L20mm_train`, identical split/resolution/min-push-length),
same loss/lr/scheduler/augmentation/batch/epochs as RUN-0001, so the only difference vs
RUN-0001 is world-frame vs. canonical-push-frame prediction.

## Code state
- Commit: `a175b98138372ab54e36f84d933ec157efa230bb`
- Working tree: dirty, 61 files changed (pre-existing; unrelated to this run)

## Exact argv
```
OMP_NUM_THREADS=4 PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u \
  Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_train_warped_L20mm_pilot.yaml --no-resume
```

## Timing
- Start: 2026-09-22 09:39:51 UTC (immediately after RUN-0001 finished and GPU was
  confirmed idle again, 15 MiB used / 0% util)
- End: 2026-09-22 09:52 UTC
- Exit status: 0
- 20 epochs in 12:25 (745 s) => ~37.3 s/epoch, essentially flat across the run (unlike
  RUN-0001, no warm-up speedup trend visible) -- consistent with the warp/plate-geometry
  path adding a roughly constant per-batch overhead vs. the unwarped arm.

## Loss (world-frame MSE -- the warped model's internal prediction is in push-frame
coordinates, but the loss reported here and the val/test numbers below are converted back
to world frame by the training code before computing `eulerian_combined`, so they are
directly comparable to RUN-0001/RUN-0003)
- Final (epoch 20): train=0.005491, val=0.005565
- Best val: 0.005565 at epoch 20 (i.e. the best-val epoch is the LAST epoch; val loss was
  noisier than RUN-0001's -- e.g. epoch 19 val=0.005761 was a local uptick before epoch 20
  came back down to 0.005565)
- No divergence, no NaNs. Curve decays smoothly through epoch ~9-10 then flattens with
  epoch-to-epoch noise of about +/-0.0002 in val loss for the remaining 10 epochs --
  slightly noisier / later-converging than RUN-0001, but not a red flag on its own.

## Test-set metrics (recorded for completeness, not part of the comparison spec)
- prob_mse: 0.005812, zero_mse: 0.027773, copy_mse: 0.014259, hard_iou: 0.778427,
  hard_dice: 0.867004, changed_mse: 0.215465

## Checkpoint
- `Baselines/NFD/runs/nfd_warped_L20mm_pilot/unet_best.pth` (best, epoch 20)
- `Baselines/NFD/runs/nfd_warped_L20mm_pilot/unet.pth` (final, epoch 20 -- same as best here)
- `model_card.yaml`, `run_config.yaml`, TensorBoard events also in that directory.

## Anomalies
None. Clean run. GPU was idle and no other job was running before/during this run (only
this run's own process + its dataloader workers were present).
