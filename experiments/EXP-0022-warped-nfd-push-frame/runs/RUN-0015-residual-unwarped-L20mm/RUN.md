# RUN-0015 -- R1, unwarped residual control

NOTE (post-run refactor, pure code move, no behaviour/retrain change): the
`Baselines/NFD/residual_nfd_lib.py` path below moved to `model/residual_nfd/lib.py`
after this run -- the registered type name `nfd-unet3ch-residual` is unchanged.

Config: `Baselines/NFD/configs/nfd_train_residual_unwarped_L20mm_pilot.yaml` (copied in as
`config.yaml`). Model `nfd-unet3ch-residual` (`Baselines/NFD/residual_nfd_lib.py::
ResidualUnwarpedWrapper`): plain world-frame NFD UNet, `residual: false` (its own
occ0-into-logit skip OFF, asserted), explicit `clamp(occ0 + tanh(delta), 0, 1)`
reconstruction, converted back to a logit so the UNCHANGED `eulerian_combined` loss
(mse=1.0 only) scores it against the absolute `occ1` target -- same target/loss as
RUN-0001. Dataset `nfd-genesis-3ch` (existing, unchanged) on
`slates_multistep/n20_L20mm_train` only. Recipe otherwise identical to RUN-0001
(resolution_scale 0.5, min_push_length_m 0.0198, val/test 5/5, batch 32, 20 epochs,
augmentation true).

## Code state
- Commit: `a175b98138372ab54e36f84d933ec157efa230bb`
- Working tree: dirty (pre-existing dirty state at start of this task, per gitStatus at
  session start; this task's own additions are new/untracked files plus one added import
  line in `train_nfd.py` and additive dict entries in `eval_report.py` -- no existing
  behaviour changed. RUN-0010 confirmed alive before/during/after this run, never touched.)

## Exact argv
```
OMP_NUM_THREADS=4 PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u \
  Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_train_residual_unwarped_L20mm_pilot.yaml --no-resume
```
(see `COMMAND.txt`)

## Timing -- CONTENTION WARNING

Ran concurrently with RUN-0010 (the standing overnight job) AND, for most of its
duration, RUN-0016 (R2, launched shortly after to save wall-clock under this task's
budget) -- i.e. 2-way contention for the first ~2 minutes, then 3-way for the rest.
**These timings are inflated by contention and are not comparable to RUN-0001's clean
~32s/epoch.**

- Start: 2026-09-23 10:17:03 (local)
- End: 2026-09-23 ~10:39:09 (local, from checkpoint mtime)
- 20 epochs in ~1326s => **~66.3 s/epoch average** (~2.1x RUN-0001's clean 32.2 s/epoch)
- Exit status: 0

## Loss (world-frame MSE, `eulerian_combined` mse=1.0 only)
- Final (epoch 20): train=0.005080, val=0.004989
- Best val: 0.004989 at epoch 20 (still improving at the last epoch)
- Monotonic, smooth decay across all 20 epochs, no divergence, no NaNs. Slightly BETTER
  final val loss than RUN-0001's 0.005143 (not yet known whether this translates to
  slateN/accuracy -- eval pending).

## Test-set metrics
- prob_mse: 0.005230, zero_mse: 0.027773, copy_mse: 0.014259, hard_iou: 0.782792,
  hard_dice: 0.869536, changed_mse: 0.224411

## Checkpoint
- `Baselines/NFD/runs/nfd_residual_unwarped_L20mm_pilot_2/unet_best.pth` (best, epoch 20)
  -- copied into `../artifacts/RUN-0015-residual-unwarped-L20mm/` alongside its
  `model_card.yaml`.
- Directory auto-suffixed `_2` by the trainer because the 2-epoch smoke test (run
  earlier in this same task, to validate the registration/wiring before committing to
  the full 20-epoch run) already occupied the un-suffixed `nfd_residual_unwarped_L20mm_pilot/`.

## Dead-gradient / mechanism checks (done BEFORE this run, on the 2-epoch smoke
checkpoint -- see `../results/residual_pilot.md` for the full writeup)
- `clamp` active on 96.81% of all pixels (harmless -- static background), but only
  12.92% of pixels that actually need to change (`|occ1-occ0|>0.05`) are gradient-starved.
  Not a training blocker.

## Anomalies
None beyond the documented contention-inflated timing. Clean run, no restarts needed.
