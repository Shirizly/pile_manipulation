# RUN-0001 — capacity intervals, persisted MLP, and the inner-CV leak test

**Experiment:** EXP-0017. **Status:** completed (see `stdout.log`, `stdout_cvleak.log`).
**Device:** CPU only (`CUDA_VISIBLE_DEVICES=""`), 4-core box shared with another agent;
threads capped at 3 (part A) and 1 (part C) so the two parts could run concurrently.

## What executed

Two processes, both under `code/`:

* `run0001_capacity_intervals.py` (parts A+B) — the capacity x value-function x
  repeated-split sweep through `value_readout.ValueReadout`, held-out-GOAL
  (4 split seeds 100..103) and held-out-STATE (3 slate-split seeds 0..2).
  Persists the MLP readouts.
* `run0001b_cv_leak.py` (part C) — EXP-0015's Fourier-order sweep re-run with the
  inner ridge-lambda CV folds made SLATE-AWARE, against the row-random folds the
  original code used.

## Inputs (reused, not re-derived)

* `experiments/temp/desc-value-readout/stage3_features_and_targets.pt` — EXP-0015's
  `phi_state` (1000,87), `phi_goal` (500,87), `slate_of_state` (20 slates),
  `values` {lyapunov, mass_in_region, signed_mass_in_region} (1000,500), `n_fourier=8`.
* Part C re-rasterises DS-0001 (`Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm`,
  step 0, 200 states/slate) because the Fourier order has to vary.
* Goals for part C: `Baselines/common/goals.py` `quadrant_mask(...,0)`, `letter_mask("T")`.

## Resolved configuration

`max_train_rows=25000`, `max_eval_rows=20000` (both parts A/B), `seed=0`,
`N_TEST_GOALS=100/500`, `N_TEST_SLATES=4/20`, MLP `(128,128)` relu `alpha=1e-3`
`max_iter=300` `early_stopping=True`, ridge alphas `logspace(-2,6,20)`,
poly-2 alphas `logspace(-1,7,12)`. Part C lambda grid and protocol copied from
EXP-0015 `stage0_upper_bound.py` unchanged except the fold construction.

## Outputs

* `artifacts/RUN-0001/cells.json` — every (split_kind, seed, capacity, mode, value_fn) cell.
* `artifacts/RUN-0001/cv_leak.json` — the Fourier x inner-CV grid, with the selected lambdas.
* `experiments/temp/exp0017-value-readout/readout_mlp_diff_<value_fn>_goalsplit100.joblib` —
  fitted `ValueReadout` objects (config + estimator + metrics), reloadable with
  `ValueReadout.load`. Logged in `experiments/TEMP_LOG.md`.
* `results/RESULTS.md` — the analysed tables.
