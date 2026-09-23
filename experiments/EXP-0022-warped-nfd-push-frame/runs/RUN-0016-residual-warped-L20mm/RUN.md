# RUN-0016 -- R2, warped residual (the arm that matters)

NOTE (post-run refactor, pure code move, no behaviour/retrain change): the
`Baselines/NFD/residual_nfd_lib.py` path below moved to `model/residual_nfd/lib.py`
after this run -- the registered type name `nfd-unet-warped-residual` is unchanged.

Config: `Baselines/NFD/configs/nfd_train_residual_warped_L20mm_pilot.yaml` (copied in as
`config.yaml`). Model `nfd-unet-warped-residual`
(`Baselines/NFD/residual_nfd_lib.py::ResidualWarpedWrapper`): `residual: false` (asserted),
warps the PRISTINE world-frame `occ0` into the canonical push frame, predicts a signed
residual there (`tanh` head), unwarps the RESIDUAL FIELD (not a reconstructed occupancy)
back to world, adds it to the SAME pristine `occ0`, clamps to [0,1] -- NO validity-mask
blend. Converted back to a logit so the UNCHANGED `eulerian_combined` loss (mse=1.0 only)
scores it against the absolute `occ1` target -- same target/loss as RUN-0002. Dataset
`nfd-genesis-3ch-warped` (existing, unchanged) on `slates_multistep/n20_L20mm_train` only.
Recipe otherwise identical to RUN-0002 (`plate_mode: canonical`, no wall channel,
`canon_res: null` -> 64, `scale: 1.0`, resolution_scale 0.5, min_push_length_m 0.0198,
val/test 5/5, batch 32, 20 epochs, augmentation true).

## Code state
- Commit: `a175b98138372ab54e36f84d933ec157efa230bb`
- Working tree dirty (pre-existing at session start, per this task's own additions -- see
  RUN-0015's identical note). RUN-0010 confirmed alive before/during/after this run.

## Exact argv
```
OMP_NUM_THREADS=4 PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u \
  Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_train_residual_warped_L20mm_pilot.yaml --no-resume
```

## Timing -- CONTENTION WARNING (worse than RUN-0015)

Ran concurrently with RUN-0010 for its ENTIRE duration, and with RUN-0015 (R1) for most
of it (both launched together, deliberately, to save wall-clock under this task's
budget). **These timings are inflated by contention (up to 3-way for most of the run)
and are not comparable to RUN-0002's clean ~37-38s/epoch.**

- Start: 2026-09-23 10:19:37 (local)
- End: 2026-09-23 ~10:44:30 (local, approx from log/checkpoint timestamps)
- 20 epochs in ~1493s => **~74.7 s/epoch average** (~2x RUN-0002's clean 37-38 s/epoch)
- Exit status: 0

## Loss (world-frame MSE, `eulerian_combined` mse=1.0 only)
- Final (epoch 20): train=0.005073, val=0.005076
- Best val: 0.005045 at epoch 18
- Monotonic, smooth decay across all 20 epochs, no divergence, no NaNs, no plateau before
  the last couple epochs (consistent with a converged fit at this budget).

## Test-set metrics
- prob_mse: 0.005438, zero_mse: 0.027773, copy_mse: 0.014259, hard_iou: 0.780310,
  hard_dice: 0.867804, changed_mse: 0.241662

## Checkpoint
- `Baselines/NFD/runs/nfd_residual_warped_L20mm_pilot_2/unet_best.pth` (best, epoch 18) --
  copied into `../artifacts/RUN-0016-residual-warped-L20mm/` alongside `model_card.yaml`.
- Directory auto-suffixed `_2` for the same reason as RUN-0015 (smoke test occupied the
  un-suffixed dir first).

## Eval result (RUN-0017, see `../results/residual_pilot.md` for the full four-arm table)

Scored against RUN-0001 (unwarped, direct), RUN-0002 (warped, direct), and R1 (unwarped
residual) via `Baselines/common/eval_report.py` on the L20mm eval cell:

- **`accuracy`: 0.4006** -- clearly above RUN-0002's 0.3596 (the direct warped arm this
  is the residual analogue of), and now essentially on par with the UNWARPED arms
  (RUN-0001 0.4018, R1 0.4091) despite paying the warp's resampling cost. This is a real
  gain over the direct warped formulation, even though the model-free residual accuracy
  CEILING (0.6852, see `results/residual_pilot.md` section 1) is barely above the direct
  ceiling (0.6640) -- i.e. the gain looks like it comes from something about the
  residual parameterisation helping the network's fit/optimisation, not from the
  resampling-ceiling argument the design was originally justified by.
- **`slateN`** (averaged over 3 goals): lyapunov 0.8608, mass_in_region 0.8586, signed_mass
  0.8066 -- beats RUN-0002 (0.8342 / 0.8366 / 0.8042) on 2 of 3 value functions and ties on
  the third within pilot noise; roughly comparable to the unwarped arms (RUN-0001 0.8810 /
  0.8233 / 0.7646; R1 0.8720 / 0.7969 / 0.7824) -- no single arm dominates all three value
  functions at this pilot's scale (60 slates, single push length -- treat as a shakeout,
  per `results/pilot_eval.md`'s own caveat, not a generalisation verdict).

## Anomalies
None beyond the documented contention-inflated timing. Clean run, no restarts needed.
