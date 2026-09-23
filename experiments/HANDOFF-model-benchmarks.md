# Handoff — general model testing and benchmarks

For a session picking up the cross-model benchmark line (EXP-0023 and its
successors). This is about HOW models are tested, not about any one model.

## Why this line exists

Every model comparison in this project before EXP-0023 scored models as
**rankers**: fix a pool of randomly sampled candidate actions, have the model
order them, measure how good its pick was (`slateN`). That is not how a
dynamics model is used in MPC, where the model is an **objective optimised
against** and the controller follows its gradients into regions of action space
no sampled pool contained.

**These are different abilities and a model can have one without the other** —
optimisation actively seeks out wherever the model is most wrong in the
optimistic direction, and ranking a plausible pool never probes that.

## What EXP-0023 established

`experiments/EXP-0023-model-as-gradient-source/` (`DESIGN.md` has the full
protocol; `EXPERIMENT.md` the record). 10 states from DS-0001
(`n20_scatter_s20a1000_L20-70mm`), `corner`/`lyapunov`, shared 100-action seed
pool, 6 arms, 120 Adam steps each, Genesis-CEM oracle at 4 iterations x 32 envs.

`dv` is a COST here (more negative is better):

| reference | true `dv` |
|---|---|
| random action from the pool | +0.0023 (the average push slightly HURTS) |
| best model's **rank-only** pick from 100 | -0.0332 |
| **best action the pool actually contained** | -0.0446 |
| best model's pick **after gradient descent** | -0.0644 |
| Genesis-CEM oracle | -0.1001 |

Three findings that stand:

1. **Models do not pick the pool's best.** The best rank-only pick (-0.033)
   falls short of what the pool contained (-0.045) — ~25% of available value
   lost to ranking error alone.
2. **Gradients genuinely escape the pool.** Optimisation reaches -0.064,
   beating the pool's own ceiling by ~44%. This is NOT "just sample more
   actions": GD finds actions no pool member matched.
3. **Model error is the binding constraint, not the pool or the optimiser.**
   Best arm captures only **0.629** of the oracle.

Two things that do NOT stand:

- **The power check FAILED, by the design's own pre-registered criterion.**
  Between-arm sd of `gradient_gain` (0.0105) < between-state sd (0.0163).
  **The arm ranking in that record must not be read.** Verdict is
  `inconclusive` for exactly this reason.
- **`gradient_gain` was negative in 18 of 60 (arm, state) cells** — optimising
  against the model was often worse than just picking from the pool, and this
  was not concentrated in one bad arm.

Also worth knowing: the top-scoring arm spent **32% of its optimisation steps
pinned to the UPPER push-length bound on 7/10 states**. The models want longer
pushes than the 20-70 mm range they were fitted on, so that row is partly a
statement about the constraint, not the model.

## The first thing to do

**Fix the power (TODO M2).** Re-run with 30-50 states instead of 10, and/or
more goals and value functions, until between-arm variation exceeds
between-state variation. The harness exists and **adding a model is one entry**
in `simple_mpc/adapters.py::OCC_ADAPTERS`. Until that passes, this benchmark
can say things about the PROBLEM (pool vs model error) but cannot rank MODELS.

## Reusable infrastructure built for this

- **`simple_mpc/adapters.py`** — a second adapter family alongside the original:
  `make_occ_adapter` / `OCC_ADAPTERS` registry, `OccupancyGradientAdapter`,
  `PredictorGradientAdapter`, `SwitchedLinearGradientAdapter`, `SlateRawStub`,
  `occ_from_particles`, `assert_dv_convention`. `PredictorGradientAdapter`
  calls each predictor's own `predict_occ` through `__wrapped__` (the function
  the `@torch.no_grad()` decorator wrapped), so the differentiable and offline
  paths are literally the same forward and cannot drift apart.
  **Note the pre-existing limit**: the ORIGINAL `make_adapter` supports only
  Eulerian wrappers and `PropNetDiffDenModel` and raises `NotImplementedError`
  for NFD/Schenck/linear operators. The `predict_occ` predictors are an
  OFFLINE scoring interface, not the live adapter contract. The new family is
  what makes NFD-style models optimisable at all.
- **`Baselines/LinearForesight/model.py`** — `soft_bin_weights`,
  `predict_switched_soft`: a differentiable triangular interpolation between
  adjacent push-length bin operators, agreeing exactly with the hard gate at
  every bin centre. The hard gate is untouched and still backs every existing
  register row. **Correction worth carrying**: the hard gate's defect is
  NARROWER than "not differentiable" — the warp, operator and occupancy path
  all carry gradient, and only the *choice of operator* is a step function in
  push length, so exactly one term of the length gradient is missing.
  `predict_switched`'s masked in-place assignment DOES survive autograd.
- **`tests/test_occ_gradient_adapters.py`** — 16 tests; invariant
  `occ-gradient-adapter-matches-offline-predictor` (`holds`).

## Metric infrastructure you must know

- **Lead with `slateN`; `accuracy` is a diagnostic, not a verdict.** They have
  repeatedly disagreed about which model is better. Live example: LinearForesight
  switched res32 beats NFD on `slateN`/lyapunov at both L20mm (0.902 vs 0.853)
  and L40mm (0.943 vs 0.932) while scoring roughly HALF NFD's image accuracy.
- **`accuracy` is a ratio of population MEANS, not a mean of per-row ratios.**
  Computing it per-row explodes to values in the millions when a row has
  near-zero persistence error. This bug was hit once already.
- **Use `random` as the ranking floor, never `persistence`** — persistence
  predicts `dv = 0` for every candidate, so `argmin` returns row 0 and the
  induced order is row order. It is degenerate as a ranker.
- **`dv`'s sign is NOT uniform across value functions** — see the SIGN section
  of `experiments/METRICS.md`. It is a COST for `lyapunov` and a REWARD for
  `mass_in_region`/`signed_mass_in_region`. `slateN` is normalised and always
  reads higher-is-better; the difference metrics (`gradient_gain`,
  `pool_escape`, `regret_vs_oracle`) have their subtraction order hard-coded
  for the COST sense and are currently correct **only for lyapunov**.
  **TODO M4** is to give them a `higher_is_better` flag so the wrong case fails
  loudly. A doc note alone will not stop this recurring.
- **Ground truth is NOT noisy** (EXP-0024, invariant
  `genesis-snapshot-restore-repeat-determinism`, holds): repeating the same
  action from the same snapshot gives a within-action/between-action `dv`
  variance ratio <= 5.3e-5. So the `slateN` plateau at 0.85-0.95 is model
  error, not noisy targets — there IS real headroom. Caveat: established for
  n20 single pushes through the snapshot-restore path only; a separate archived
  probe saw genuine 30-38% divergence at n=100 on longer diagonal pushes.

## Reference baselines, one harness, three corpora

`Baselines/common/eval_report.py`. `slateN` goal-averaged, leading;
`accuracy` beside it.

| model | corpus | accuracy | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|---|
| NFD (`nfd_3ch_randlen`) | L20mm | 0.407 | 0.853 | 0.820 | 0.753 |
| | L40mm | 0.509 | 0.932 | 0.783 | 0.841 |
| | randlen_test | 0.456 | 0.942 | 0.889 | 0.866 |
| LinearForesight switched res32 | L20mm | 0.262 | 0.902 | 0.830 | 0.705 |
| | L40mm | 0.437 | 0.943 | 0.821 | 0.882 |
| | randlen_test | 0.248 | 0.856 | 0.788 | 0.784 |
| LinearForesight switched res64 | randlen_test | 0.208 | 0.711 | 0.734 | 0.723 |
| LinearForesight single (global) res32 | randlen_test | 0.147 | 0.629 | 0.570 | 0.428 |
| warped NFD + flip aug + residual | randlen_test | 0.462 | 0.938 | 0.876 | 0.853 |

Switching by push length is worth a lot (switched >> single everywhere). The
linear model's weakness is generalisation to the broad corpus, not ranking.

## The largest hole in ALL of this

**No seed-level noise floor anywhere.** Every arm in every experiment is a
single training run, so margins of 0.01-0.05 cannot be ordered and several
verdicts rest on "the direction is consistent across three corpora" rather than
on a margin. This is **TODO H1** and it gates how much any close call is worth.

## Other open items

`experiments/TODO.md` is the prioritised list. `experiments/OPEN_ISSUES.md` has
two defects worth knowing: `Baselines/LinearForesight/runs/operators_res64.pt`
was missing from disk and restored from an untracked stray (provenance
unverified — the `linear_*_res64` rows above rest on it), and
`nfd_train_3ch_randlen.yaml`'s header states its gradient-step count 8x too low
(it quotes the no-augmentation arithmetic; comparisons are unaffected, anything
SIZED by trusting it is not).
