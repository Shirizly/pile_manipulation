# EXP-0017 results — RUN-0001

All numbers are `value_readout_r2` (METRICS.md), SS_tot about the TRAINING mean.
Features are EXP-0015's 87-dim analytic occupancy descriptors (n_fourier=8) of the
state and of the goal-as-legal-configuration; 1000 states over 20 slates, 500 goals.
`diff` = phi_state - phi_goal (EXP-0015's mode); `concat` = [phi_state, phi_goal],
the mode a learned latent would use when the goal embedding need not share the space.
CPU only. 25k train rows / 20k eval rows per cell.

## Headline

**At MLP capacity the readout beats `goal_only` on held-out goals for all three value
functions, by 4-8x the goal-split noise floor (~0.05-0.07).** At LINEAR capacity with
`diff` features it does NOT, for either mass function — reproducing EXP-0015's caution.

| value_fn | best held-out-GOAL model | best `goal_only` (any capacity) | increment | floor |
|---|---|---|---|---|
| lyapunov | mlp/diff **+0.871 +- 0.010** (n=3) | +0.327 +- 0.064 (linear) | **+0.544** | ~0.06 |
| mass_in_region | mlp/diff **+0.644 +- 0.033** (n=3) | +0.364 +- 0.064 (linear) | **+0.280** | ~0.06 |
| signed_mass_in_region | mlp/diff **+0.703 +- 0.026** (n=3) | +0.363 +- 0.057 (mlp) | **+0.340** | ~0.06 |

`goal_only` is quoted at its BEST capacity, not at the model's capacity, so the baseline
is not handicapped by a bad capacity choice (see "MLP on goal_only diverges", below).

## Capacity ordering, with an interval (EXP-0015's missing cell)

Held-out GOAL, `diff` features, mean +- sd over independent 400/100 goal splits.

| value_fn | linear (n=4) | poly-2 (n=3) | MLP(128,128) (n=3) | EXP-0015 single split |
|---|---|---|---|---|
| lyapunov | +0.420 +- 0.051 | +0.794 +- 0.016 | +0.871 +- 0.010 | 0.362 / 0.751 / 0.893 |
| mass_in_region | +0.285 +- 0.051 | +0.585 +- 0.061 | +0.644 +- 0.033 | 0.213 / 0.499 / 0.689 |
| signed_mass_in_region | +0.278 +- 0.050 | +0.597 +- 0.055 | +0.703 +- 0.026 | 0.205 / 0.482 / 0.749 |

* **linear -> poly-2 is resolved** for every value function: gaps 0.30-0.37 against
  spreads <= 0.06.
* **poly-2 -> MLP is resolved for `lyapunov`** (+0.077 vs spreads 0.016/0.010) and
  **weakly for `signed_mass_in_region`** (+0.106 vs 0.055/0.026), and is
  **UNRESOLVED for `mass_in_region`** (+0.059 against spreads 0.061/0.033 — inside the
  floor; do not rank these two).
* EXP-0015's single-split numbers all sit within ~1-2 sd of the repeated means, so its
  ordering claim survives; only its precision was missing.

## Full held-out-GOAL cell table (all modes, all baselines)

See `tables_capacity.md`. Notable rows:

| value_fn | capacity | diff | concat | state_only | goal_only | mean_only |
|---|---|---|---|---|---|---|
| lyapunov | linear | +0.420 | +0.451 | +0.122 | +0.327 | 0.000 |
| mass_in_region | linear | +0.285 | +0.378 | +0.010 | +0.364 | 0.000 |
| signed_mass_in_region | linear | +0.278 | +0.384 | +0.020 | +0.361 | 0.000 |

At linear capacity `diff` is BELOW `goal_only` for both mass functions (-0.080, -0.083);
`concat` is above it by +0.013 / +0.023, i.e. **inside the noise floor -> UNRESOLVED**.
`concat` beats `diff` at linear capacity for all three value functions (+0.03 to +0.11),
which matters because `concat` is the mode a learned latent will use.

## Held-out STATE — the negative-R2 anomaly, partially resolved

Held-out STATE = 4 of 20 slates withheld, all 500 goals in training.

| value_fn | linear/diff | mlp/diff (n=1) | linear/state_only | goal_only (linear) |
|---|---|---|---|---|
| lyapunov | +0.129 +- 0.074 | +0.585 | +0.105 | +0.329 +- 0.042 |
| mass_in_region | -0.075 +- 0.237 | +0.332 | +0.005 | +0.381 +- 0.021 |
| signed_mass_in_region | -0.134 +- 0.239 | +0.108 | +0.009 | +0.377 +- 0.021 |

The negative held-out-STATE R2 EXP-0015 reported (-0.12, -0.16) **reproduces, and is a
LINEAR-CAPACITY effect**: raising capacity to the MLP moves both mass functions positive
(+0.332, +0.108). So the sign flip is not a data pathology — a linear map in `diff`
features extrapolates worse than a constant on unseen slates. **It is only partly
resolved**: even at MLP capacity both mass functions stay BELOW `goal_only` on held-out
STATE (0.332 vs 0.424; 0.108 vs 0.425), and the MLP cells are a SINGLE state split with
no interval. Why the held-out-STATE deficit persists while the held-out-GOAL increment is
large is UNEXPLAINED.

## The inner-CV leak — mechanism CONFIRMED

Read from the code first (`stage0_upper_bound.py::fit_eval_cv`, and identically
`stage3b_regression.py` and `persist_final_models.py::cv_lambda`):

    folds = np.array_split(rng_cv.permutation(n), 3)

The OUTER split is slate-aware (`split_state`); the INNER folds selecting the ridge
lambda are row-random. Post-push states within one slate share a pile, so row-random
inner folds put near-duplicates on both sides of the fold.

Re-running the Fourier sweep with the inner folds grouped BY SLATE, everything else
byte-identical (`tables_cv_leak.md`):

| goal / value_fn | nf=8 | nf=16 | nf=24 | nf=32 |
|---|---|---|---|---|
| quadrant0 / lyapunov, row-random | +0.921 | +0.301 | -0.617 | +0.071 |
| quadrant0 / lyapunov, slate-aware | +0.890 | +0.789 | +0.597 | +0.591 |
| quadrant0 / mass, row-random | +0.222 | -1.283 | -4.367 | -1.257 |
| quadrant0 / mass, slate-aware | +0.775 | +0.760 | +0.621 | +0.634 |
| letter_T / lyapunov, row-random | +0.633 | **-18.66** | -0.704 | -0.806 |
| letter_T / lyapunov, slate-aware | +0.836 | +0.518 | +0.490 | +0.476 |
| letter_T / mass, row-random | -5.774 | **-38.90** | -13.22 | -0.778 |
| letter_T / mass, slate-aware | +0.070 | +0.302 | +0.588 | +0.497 |

**CONFIRMED.** The selected lambda is the smoking gun: row-random folds pick
lambda in 0.01-1 in 15 of 16 cells; slate-aware folds pick 30-3000 in 12 of 16. Under
slate-aware folds **no cell is negative** and the catastrophic monotone reversal is gone.

What is NOT established: that higher Fourier order *helps*. Under slate-aware folds the
trend is mixed — `quadrant0/lyapunov` still drifts down 0.890 -> 0.591, `letter_T/mass`
rises 0.070 -> 0.497 — and the spreads (0.04-0.49) cover most of it. Fourier order is
**UNRESOLVED**, not reversed. EXP-0015's result (2) is an artifact of the fold
construction and should not be cited in either direction.

## Persisted objects

`experiments/temp/exp0017-value-readout/readout_mlp_diff_<value_fn>_goalsplit100.joblib`
— three fitted `ValueReadout` objects (config + sklearn estimator + metrics), the
capacity EXP-0015 measured but did not save. Reload with
`ValueReadout.load(path)`; then `.predict(X_state, X_goal)` or
`.dv(z_now, z_pred, x_goal)` for the action-ranking signal.
