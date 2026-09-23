# EXP-0023 — a dynamics model as a SOURCE OF GRADIENTS, not just a ranker

## Why this test exists

Every model comparison in this repo so far scores models as **rankers**: fix
a pool of randomly sampled candidate actions, have the model order them, and
measure how good its pick was (`slateN`). That is not how a dynamics model is
actually used. In MPC the model is an **objective to be optimised against** —
the controller follows the model's gradients into regions of action space
that no sampled pool contained.

These are different abilities and **a model can have one without the other.**
A model can rank a fixed pool well while its gradients point somewhere
useless, because optimisation actively seeks out the model's errors: gradient
descent will happily walk to wherever the model is most wrong in the
optimistic direction. Ranking a fixed pool never exposes that failure mode,
because a pool of plausible actions never probes the model's worst
extrapolations.

**So the headline quantity is not "how good is the optimised action" but the
DIFFERENCE between a model's rank-only score and its post-optimisation
score.** That difference can be negative, and a negative value is the single
most informative outcome this test can produce.

This also directly answers a standing question: whether the random candidate
pool, rather than the ranker, is what limits control performance.

## The design

### Common protocol, per state

Every arm starts from **the same ~100-action seed pool** for that state, so
no arm gets a better starting point than another.

1. **Seed pool.** ~100 actions sampled the way the corpus samples them
   (pile-aware sampling, `Genesis/binned_slate_collection.py`; push length in
   the corpus's 20-70 mm range).
2. **Rank-only pick** (`a_rank`). The action the model scores best in the
   seed pool. This reproduces the existing `slateN`-style measurement and is
   the baseline each arm is compared against *itself* on.
3. **Gradient-optimised pick** (`a_grad`). Starting from `a_rank`, run the
   repo's gradient-descent MPC (`simple_mpc/mpc.py`) through the model,
   optimising the action parameters against the value objective.
4. **Oracle-optimised pick** (`a_oracle`). From the same seed pool, run CEM
   (`simple_mpc/sampling_optimizers.py`) with **Genesis itself as the model**
   (`simple_mpc/genesis_oracle.py`). Dynamics error is zero by construction,
   so this is a ceiling, NOT a fair competitor — it costs orders of magnitude
   more compute. **CEM, not MPPI**: one optimiser, chosen and held fixed, so
   arms differ only in the model.
5. **Ground truth.** Execute `a_rank`, `a_grad` and `a_oracle` in Genesis and
   measure the true value change `dv` of each.

### Metrics

Let `dv(a)` be the true, simulated value change.

| metric | definition | what it answers |
|---|---|---|
| `dv_rank` | `dv(a_rank)` | the existing ranker measurement |
| `dv_grad` | `dv(a_grad)` | how good the optimised action actually is |
| `dv_oracle` | `dv(a_oracle)` | the ceiling |
| **`gradient_gain`** | `dv_rank - dv_grad` (see SIGN note below) | **the headline. Negative = optimising against this model makes things WORSE than just picking from the pool.** |
| `capture_vs_oracle` | `dv_grad / dv_oracle` | the user's requested "capture relative to the oracle-optimised action" |
| `regret_vs_oracle` | `dv_grad - dv_oracle` (see SIGN note) | absolute headroom left |
| `pool_ceiling` | `max over seed pool of dv` | best action the pool CONTAINED |
| **`pool_escape`** | `pool_ceiling - dv_grad` (see SIGN note) | **did optimisation find something better than anything in the pool?** This is what separates "a wider pool would do" from "gradients add real value." |

**SIGN NOTE — corrected 2026-09-23, after this design was written and executed.**
As first drafted, the three difference metrics above were written as
`dv_grad - dv_rank` and glossed "negative = worse". Those contradict each
other, because **`dv = value(after) - value(before)` is a COST in this repo**
(lower is better), as `binned_pool_cache.py` and `METRICS.md` define it. The
formulae have been negated here so that every difference metric reads
**positive = better**, which is what the glosses always meant. EXP-0023's
executed record caught this and used the corrected sign throughout, and now
asserts the convention in code (`simple_mpc.adapters.assert_dv_convention`).
`capture_vs_oracle = dv_grad / dv_oracle` needs no change: both are costs
measured from the same `v0`, so the ratio already reads correctly.

`capture_vs_oracle` is a ratio and is unstable when `dv_oracle` is near zero,
so report the raw `dv` values alongside it and exclude or flag degenerate
states rather than letting a ratio blow up. `dv` sign convention must be
fixed once, written down, and asserted in code — a sign error here inverts
every conclusion.

### Scope

- **10 states**, `n=20`, scatter spawn — `Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm` (DS-0001) is exactly this and already has a 1000-candidate pool per slate, so the 100-action seed pool can be drawn from it and its true `dv` values may already be cached (`scripts/probes/binned_pool_cache.py`).
- **Goals/value functions**: start with `lyapunov` and one goal shape. Widen
  only if cheap — breadth matters for `slateN`, but this test's unit of
  analysis is the per-state optimisation, not the pool statistic.
- **Arms**: the baseline models that can be made differentiable — NFD
  (`nfd_3ch_randlen`), warped NFD, residual warped NFD (EXP-0022's best),
  LinearForesight switched. Plus **floors**: `random` action from the pool,
  and the seed pool's own best-by-true-`dv` (`pool_ceiling`).

## Known blockers, which are most of the work

1. **`simple_mpc/adapters.py::make_adapter` does not support these models.**
   `docs/CODEMAP.md` is explicit: it supports only Eulerian wrappers and
   `PropNetDiffDenModel`, and raises `NotImplementedError` for NFD, Schenck
   and the fitted linear operators. The `predict_occ` predictors are an
   **offline scoring** interface, not the live adapter contract. Writing
   those adapters is the bulk of this task, and they are a reusable
   capability that belongs in `simple_mpc/adapters.py`, not in this
   experiment's `code/`.
2. **LinearForesight's bin switching is not differentiable.**
   `Baselines/LinearForesight/model.py::bin_index` uses `torch.bucketize` to
   pick a per-length operator. Under gradient descent the push length moves
   continuously and crosses bin boundaries, where the gradient is either zero
   or undefined. The fix is a **differentiable soft gate** — interpolate
   between the two adjacent bins' operators as a function of where the length
   sits between bin centres — implemented alongside the existing hard gate,
   not replacing it (the hard gate backs existing register rows). Document
   the relaxation as a deviation from the fitted model.
3. **Differentiability of the whole chain must be verified, not assumed.**
   The action reaches the model through `draw_plate_soft` (soft precisely so
   it is differentiable) and, for warped arms, through the push-frame warp.
   A silently-zero gradient would make this benchmark measure nothing while
   producing plausible numbers.

   **CORRECTED after execution:** this section originally also flagged
   `predict_switched`'s masked in-place assignment into a cloned tensor as an
   autograd risk. **It is not** — EXP-0023 verified gradient survives it. And
   the hard gate's real defect is narrower than "not differentiable": the
   warp, the operator and the occupancy path all carry gradient, and only the
   *choice of operator* is a step function in push length, so exactly one
   term of the length gradient is missing. "bucketize isn't differentiable"
   is easily overread into "this model cannot be optimised against", which is
   false.
4. **The action must stay legal.** Optimisation will push the action out of
   the workspace, through walls, or to a length outside the 20-70 mm range
   the models were fit on. Constrain by projection or reparameterisation, and
   **report how often each arm hits a bound** — an arm that is always clamped
   is being scored on the constraint, not on its gradients.

## The result that would most embarrass this design

Every arm's `gradient_gain` comes out near zero and within noise, so the test
cannot distinguish models that are good gradient sources from ones that are
not. Guard: report per-state values, not just means, and compare the spread
across states against the spread across arms. If between-arm variation is
smaller than between-state variation, this design lacks the power to answer
its own question and the honest report says so.

## Why this is worth building properly

It is intended as **the basis for a more thorough model benchmark**, not a
one-off. The adapters and the differentiable gate are reusable; the driver
should be written so adding a model is one registration, in the same spirit
as `Baselines/common/eval_report.py`'s `MODELS` dict.
