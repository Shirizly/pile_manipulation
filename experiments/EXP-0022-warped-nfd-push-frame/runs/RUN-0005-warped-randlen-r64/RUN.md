# RUN-0005 -- warped NFD (no wall channel), canon_res=64, on overnight_randlen

The main-run arm that actually decides the experiment. Config:
`Baselines/NFD/configs/nfd_train_warped_randlen.yaml` (copied in as `config.yaml`).
Same NFD UNet architecture family as the world-frame baseline, but predicting in the
canonical push frame (`transforms.functional.push_frame_roundtrip`) via
`nfd-unet-warped` / `nfd-genesis-3ch-warped`, `plate_mode: canonical`, `wall_channel:
false`, `canon_res: null` (-> batch's own grid resolution, 64 at
`resolution_scale=0.5` -- the setting that matches the LinearForesight baseline's own
warp exactly). Trained on `overnight_randlen_train` (all 5 spawn-mode/particle-count
groups, 89,081 train transitions), same split/recipe/30-epoch gradient-step count as
`nfd_train_3ch_randlen.yaml`, whose checkpoint (`Baselines/NFD/runs/nfd_3ch_randlen/
unet_best.pth`) is the world-frame NFD baseline recorded in EXP-0001. The baseline
itself was NOT retrained for this run -- it is the fixed comparison point.

## Code state
- Commit: `a175b98138372ab54e36f84d933ec157efa230bb`
- Working tree: dirty (pre-existing at run start; unrelated to this run -- same dirty
  tree noted in RUN-0001..0004)

## Exact argv
```
OMP_NUM_THREADS=4 PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u \
  Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_train_warped_randlen.yaml --no-resume
```
(see `COMMAND.txt`)

## Timing
- Start: 2026-09-22 10:18:15 CEST (08:18:15 UTC), GPU idle beforehand.
- 30 epochs in 2:42:57 (9,777 s epoch-loop time per the in-process tqdm log), ~326
  s/epoch average -- consistent with the ~363 s/epoch estimate in `LOG.md` from the
  first 7 epochs before the sequential wrapper script was killed (see Anomalies).
- Total wall time including data load and the post-training test-set evaluation pass:
  ~2h43m (process exit, inferred from the run directory's own `stdout.log`/`stderr.log`
  mtimes, ~13:01 CEST / 11:01 UTC).
- Exit status: 0.

## Loss (world-frame MSE, converted back from push-frame internally by the training
code before computing `eulerian_combined`, so directly comparable to the baseline)
- Final (epoch 30): train=0.009176, val=0.009300
- Best val: **0.009284 at epoch 28** (val was noisier after ~epoch 22 -- small
  +/-0.0001 upticks at epochs 23, 25-27, 29-30 around a flat plateau; no divergence,
  no NaNs)
- Hard IoU: 0.8802 at epoch 30 (training-loop metric); **0.884540 on the held-out test
  split** (post-training test-set pass)

## Test-set metrics (recorded for completeness)
- prob_mse: 0.009209, zero_mse: 0.070880, copy_mse: 0.017508, hard_iou: 0.884540,
  hard_dice: 0.936928, changed_mse: 0.201596

## Checkpoint
- `Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth` (best, epoch 28)
- `Baselines/NFD/runs/nfd_warped_randlen/unet.pth` (final, epoch 30)
- `model_card.yaml` (confirms `plate_mode: canonical`, `wall_channel: false`, `scale:
  1.0`, `canon_res: null`), `run_config.yaml`, TensorBoard events also in that
  directory.

## Anomalies
- **The sequential wrapper script (`code/run_randlen_arms.sh`) was killed at epoch
  7/30, per user direction, to pause and test this arm before committing the rest of
  the GPU time to RUN-0006/0007/0008.** The training PROCESS ITSELF (the `train_nfd.py`
  invocation this RUN.md documents) was NOT killed and continued running to completion
  under its own process group, unattended, and exited with status 0 as recorded above.
  RUN-0006/0007/0008's run directories are empty placeholders; those arms were never
  started.
- **Consequence for `experiments/COMMANDS.jsonl`**: because the wrapper was killed
  mid-run, no `end` event was written for RUN-0005 by the script that writes `start`/
  `end` events automatically. An `end` event has been added by hand after the fact,
  from the checkpoint/log timestamps described under Timing above (not from a live
  process-exit hook), and is explicitly labelled `"reconstructed": true` in that JSONL
  entry -- it is not a directly logged timestamp and should not be read as one.
- No other anomalies. GPU was idle before the run started (confirmed via `nvidia-smi`,
  per `LOG.md`); no divergence, no NaNs, exit status 0.
