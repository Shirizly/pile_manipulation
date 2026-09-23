# RUN-0001 -- unwarped control (arm 1 of 3)

Config: `Baselines/NFD/configs/nfd_train_3ch_L20mm_pilot.yaml` (copied in as `config.yaml`
in this directory). Plain `nfd-genesis-3ch` dataset + `nfd-unet3ch` model, world-frame
prediction, on `slates_multistep/n20_L20mm_train` only (5%/5% val/test split, `resolution_scale`
0.5, `min_push_length_m` 0.0198, augmentation on, batch 32, 20 epochs, `eulerian_combined`
loss with mse=1.0 and all other loss terms 0). This is the control against which the two
warped arms (RUN-0002, RUN-0003) are compared -- no existing checkpoint had been trained on
L20mm alone.

## Code state
- Commit: `a175b98138372ab54e36f84d933ec157efa230bb`
- Working tree: dirty, 61 files changed (pre-existing dirty state at start of this task;
  none of it was touched by this run)

## Exact argv
```
OMP_NUM_THREADS=4 PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u \
  Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_train_3ch_L20mm_pilot.yaml --no-resume
```
(see `COMMAND.txt`)

## Timing
- Start: 2026-09-22 09:28:09 UTC (local 09:28, foreground process launch)
- End:   2026-09-22 09:39 UTC (approx, from checkpoint mtimes; process observed exited
  before 09:38 wall-clock check loop returned)
- Exit status: 0 (background wrapper reported `exit code 0`)
- 20 epochs in 10:43 (643 s) => ~32.2 s/epoch average across the run (first epoch ~39 s,
  settling to ~29 s/epoch by the end, presumably as dataloader worker warm-up / OS caching
  effects wore off)

## Loss (world-frame MSE, `eulerian_combined` with mse=1.0 only)
- Final (epoch 20): train=0.005218, val=0.005143
- Best val: 0.005102 at epoch 19 (train at that epoch: 0.005244)
- Monotonic, smooth decay across all 20 epochs, no divergence, no NaNs, no plateau before
  epoch ~15 (val loss still inching down through epoch 19). Curve looks healthy.

## Test-set metrics (from the script's own held-out test split, NOT part of the
comparison spec but recorded for completeness)
- prob_mse: 0.005390, zero_mse: 0.027773, copy_mse: 0.014259, hard_iou: 0.783553,
  hard_dice: 0.869927, changed_mse: 0.239256

## Checkpoint
- `Baselines/NFD/runs/nfd_3ch_L20mm_pilot/unet_best.pth` (best, epoch 19)
- `Baselines/NFD/runs/nfd_3ch_L20mm_pilot/unet.pth` (final, epoch 20)
- `Baselines/NFD/runs/nfd_3ch_L20mm_pilot/model_card.yaml`, `run_config.yaml`, TensorBoard
  events file also present in that directory.

## Anomalies
None. Clean run, no GPU contention (GPU was idle before start: 15 MiB used, 0% util),
no restarts needed.
