# RUN-0003 -- warped + wall channel (arm 3 of 3)

Config: `Baselines/NFD/configs/nfd_train_warped_walls_L20mm_pilot.yaml` (copied in as
`config.yaml`). Identical to RUN-0002 except `wall_channel: true` and `in_channels: 4`
(a 4th workspace-extent channel). Same dataset, split, loss, lr, scheduler, augmentation,
batch, epochs as RUN-0001/RUN-0002.

## Code state
- Commit: `a175b98138372ab54e36f84d933ec157efa230bb`
- Working tree: dirty, 61 files changed (pre-existing; unrelated to this run)

## Exact argv
```
OMP_NUM_THREADS=4 PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u \
  Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_train_warped_walls_L20mm_pilot.yaml --no-resume
```

## Timing
- Start: 2026-09-22 09:53:03 UTC (immediately after RUN-0002 finished and GPU was
  confirmed idle again, 15 MiB used / 0% util, no leftover processes)
- End: 2026-09-22 10:05 UTC
- Exit status: 0
- 20 epochs in 12:38 (758 s) => ~37.9 s/epoch, essentially flat across the run, in line
  with RUN-0002's ~37.3 s/epoch (both warped arms cost about the same per epoch; the
  extra wall channel adds negligible overhead vs. the warp path itself).

## Loss (world-frame MSE, same `eulerian_combined` mse=1.0-only loss, converted back to
world frame before the reported numbers -- directly comparable to RUN-0001/RUN-0002)
- Final (epoch 20): train=0.005388, val=0.005508
- Best val: 0.005446 at epoch 16 (train at that epoch: 0.005462)
- Loss curve itself decays smoothly and monotonically the whole run, no NaNs, no
  divergence. Best val is NOT the last epoch here (epochs 17-20 hover 0.0055-0.0055
  without a sustained further improvement) -- similar late-training noise pattern to
  RUN-0002.

### Anomalous early-training transient (epochs 1-2)
`hard_iou` was very low for the first two epochs (0.2182 at epoch 1, 0.3319 at epoch 2)
before jumping to 0.77 by epoch 3 and behaving normally thereafter (0.78-0.79 for the
rest of training, in line with the other two arms). Train/val loss themselves fell
smoothly across epochs 1-3 (0.173/0.113 -> 0.079/0.051 -> 0.036/0.022) with no
discontinuity, so this reads as a slow-to-converge hard-threshold IoU metric during the
model's first couple of epochs (plausible with an extra input channel changing the
early optimization trajectory) rather than a training failure -- but it is the one
genuinely anomalous-looking number across all three arms and is flagged here rather than
silently smoothed over. RUN-0001 and RUN-0002's epoch-1 IoU were 0.63 and 0.54
respectively, both much higher than RUN-0003's 0.22, so the effect is specific to this
arm's first two epochs.

## Test-set metrics (recorded for completeness, not part of the comparison spec)
- prob_mse: 0.005729, zero_mse: 0.027773, copy_mse: 0.014259, hard_iou: 0.779022,
  hard_dice: 0.867904, changed_mse: 0.204065

## Checkpoint
- `Baselines/NFD/runs/nfd_warped_walls_L20mm_pilot/unet_best.pth` (best, epoch 16)
- `Baselines/NFD/runs/nfd_warped_walls_L20mm_pilot/unet.pth` (final, epoch 20)
- `model_card.yaml`, `run_config.yaml`, TensorBoard events also in that directory.

## Anomalies
Early-epoch IoU dip described above (epochs 1-2), self-resolved by epoch 3. No crash, no
NaN, no restart needed. GPU was idle and exclusive to this job throughout (confirmed idle
before start; no other train_nfd processes observed running concurrently at any point
across all three runs in this experiment).
