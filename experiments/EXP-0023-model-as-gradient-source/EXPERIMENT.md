---
# ---- identity -------------------------------------------------------------
id: EXP-0023
title: >
  A dynamics model as a SOURCE OF GRADIENTS, not just a ranker: per-state
  gradient-descent MPC through six baseline models against a Genesis-CEM
  oracle ceiling on DS-0001, measuring gradient_gain = the true-dv difference
  between a model's gradient-optimised action and its own best pick from the
  same 100-action seed pool
tier: T1
mode: exploratory
date: 2026-09-23

# ---- the claim ------------------------------------------------------------
claim: >
  MEASURED, EXPLORATORY. On 10 DS-0001 states, `corner`/`lyapunov`, with
  every arm starting from the same 100-action seed pool, gradient-descent MPC
  through each of six frozen baseline models improves the TRUE dv of the
  chosen action ON AVERAGE (`gradient_gain` +0.0021 to +0.0312 Lyapunov
  units, positive for all six arm means) but NOT reliably: every arm has 2 to
  4 of 10 states where optimising against the model made the outcome WORSE
  than its own best pick from the pool. The arm ordering CANNOT be read: the
  sd of the six arm means (0.0105) is SMALLER than the sd of the ten state
  means (0.0163), which is exactly the failure `DESIGN.md` pre-named as the
  one that would embarrass it. What the record does establish, because the
  effects are large relative to that spread: (a) `nfd_residual_warped` is a
  bad gradient source in absolute terms -- `gradient_gain` +0.0021, capture
  0.112 of the oracle, and `pool_escape` NEGATIVE on 10 of 10 states, i.e.
  optimising against it never once beat the best action the seed pool already
  contained; (b) no arm reaches the Genesis-CEM ceiling -- the best captures
  0.63 of it -- so dynamics-model error, not the candidate pool, is the
  binding constraint at this pool size; (c) optimisation does escape the pool
  for four of six arms (`pool_escape` positive on 4-6 of 10 states), so
  gradients are not merely a slower way to search the pool.

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: a175b981
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0023-model-as-gradient-source/code/stage1_optimise.py
  code_path: experiments/EXP-0023-model-as-gradient-source/code/
  data: ["DS-0001"]
  seed: 0
  split: >
    Nothing is FITTED in this record. Every model is a pre-existing frozen
    checkpoint; DS-0001 (`Genesis/data/slates_binned/n20_scatter_s20a1000_
    L20-70mm`, step 0) is used only as (a) the source of the 10 evaluation
    states and (b) the shared 100-action seed pool per state, drawn with
    `torch.randperm(seed=0)` from that state's 1000 candidates. The SAME seed
    pool is handed to every arm, so no arm gets a better starting point.
    MODEL-0001 and MODEL-0003 were fit on a legacy split that overlaps
    `overnight_randlen_test` (EXP-0021's `split` note); that affects their
    standing as models, not this record's within-arm rank-vs-gradient
    contrast, which compares each arm against ITSELF.
  runtime: >
    RUN-0000 gradient gate ~40 s; RUN-0001 (model side) 188 s; RUN-0002
    (Genesis) ~13 min for 10 states plus ~100 s scene build and kernel
    compilation; RUN-0003 (analysis) ~5 s. All under 3-way GPU contention
    (EXP-0022's RUN-0010 training throughout, plus two other subagents'
    jobs) -- every timing here is INFLATED and none is a benchmark.
  runs: ["RUN-0001-stage1-optimise", "RUN-0002-stage2-genesis",
         "RUN-0003-stage3-analyse"]
  env: "python 3.10, torch 2.11.0+cu130 (anaconda3/envs/pme), RTX 4070 Laptop 8 GB, Genesis 1.3.3"

budget:
  declared: "~2.5 h wall-clock, ~350k tokens (task brief)"
  spent: "~1 h wall-clock total; well inside the declared 2.5 h"
  outcome: within

design:
  varied:
    arm: [nfd_3ch_randlen, nfd_3ch_finetuned, nfd_warped_randlen,
          nfd_residual_warped, linear_switched_soft, linear_switched_hard]
    action_source: [rank-only-from-pool, gradient-optimised, genesis-cem-oracle,
                    pool-ceiling-by-true-dv]
    state: "10 slates of DS-0001 (slate_idx 0..9)"
  held_fixed:
    pool: "100 actions per state, drawn once with seed 0 and shared by every arm"
    value_fn: "lyapunov"
    goal: "corner"
    grid: 64
    bounds: "+-0.064 m both axes"
    rasteriser: "transforms.functional.particles_to_occupancy, footprint_radius 1.25 vox (identical to scripts/probes/binned_pool_cache.py)"
    optimiser_model_side: "Adam, lr 1.5e-3 on the raw [sx,sy,ex,ey] metres, 120 steps, best-iterate-by-PREDICTED-dv kept"
    optimiser_oracle_side: "CEM (simple_mpc.sampling_optimizers.CEMOptimizer), 4 iterations x 32 Genesis envs, n_elite 8, momentum 0.5, std_floor 0.005, init_std_frac 0.5 -- CEM and only CEM, per DESIGN.md, so arms differ only in the model"
    constraints: "projection after every step: endpoints clamped to +-0.060 m, push length rescaled into [0.020, 0.070] m along the current direction. The SAME projection is applied to every CEM candidate, so the oracle searches the identical legal set."
    horizon: "n_ahead = 1 (DS-0001 is a single-step pool)"
  baselines:
    - "the arm's OWN rank-only pick from the seed pool (`dv_rank`) -- the within-arm control that `gradient_gain` is a difference against; this is the do-nothing-new baseline for an optimiser"
    - "pool_ceiling: the best action the 100-action seed pool actually CONTAINED, by TRUE dv -- separates 'a wider pool would have done' from 'gradients added value'"
    - "Genesis-CEM oracle: zero dynamics error by construction, the ceiling (not a fair competitor)"
    - "persistence is NOT used as a floor here: it predicts dv=0 for every candidate, so its argmin returns row order (CODEMAP/binned_pool_cache trap). The pool's own true-dv distribution plays the floor role instead."
  metric: "gradient_gain (a difference of true `lyapunov` dv), with capture_vs_oracle, pool_escape, regret_vs_oracle"

noise_floor: >
  Resimulation noise is ZERO here, not merely small: invariant
  `genesis-snapshot-restore-repeat-determinism` (established by EXP-0024)
  says re-executing an identical action from an identical restored particle
  snapshot returns bit-identical terminal positions and dv (within_var 0 to
  1.2e-7 against between_var 2.5e-4 to 2.3e-3). Every dv in this record is
  produced through exactly that seam, so the per-cell measurement has no
  resimulation variance and the only spread is genuine between-state and
  between-arm variation. The relevant "floor" is therefore the DESIGN'S OWN
  POWER CHECK, reported in results/: the sd of the six arm means of
  gradient_gain against the sd of the ten state means. If the former is not
  larger than the latter, this design cannot separate arms at n=10 states
  and the record says so rather than reading its arm ordering.

depends_on:
  - genesis-snapshot-restore-repeat-determinism
  - goal-mask-axis-convention-row-y-col-x
  - push-frame-warp-roundtrip
  - slates-binned-uniform-difficulty

establishes:
  - occ-gradient-adapter-matches-offline-predictor

result: >
  Per-arm means over 10 states (Lyapunov units; dv is a COST, and the three
  difference metrics are written positive = better, see the body):

    arm                    dv_rank   dv_grad  grad_gain(sem)  neg  escape  esc+  capture
    nfd_warped_randlen    -0.03320  -0.06436  +0.0312(.0121)  3/10 +0.0197 5/10  0.629
    linear_switched_soft  -0.03925  -0.05998  +0.0207(.0101)  4/10 +0.0153 6/10  0.623
    nfd_3ch_randlen       -0.03422  -0.05488  +0.0207(.0086)  2/10 +0.0102 4/10  0.552
    linear_switched_hard  -0.03696  -0.04929  +0.0123(.0059)  2/10 +0.0046 5/10  0.486
    nfd_3ch_finetuned     -0.02625  -0.03383  +0.0076(.0043)  4/10 -0.0108 3/10  0.348
    nfd_residual_warped   -0.00871  -0.01080  +0.0021(.0018)  3/10 -0.0338 0/10  0.112

  Floors on the same 10 pools: `random` (pool mean true dv) +0.00234 -- the
  average push in this pool slightly HURTS -- and `pool_ceiling` (pool best)
  -0.04464. Oracle -0.10014, range -0.0564 to -0.1430 per state.
  POWER CHECK FAILS: between-arm sd 0.0105 < between-state sd 0.0163
  (residual sd 0.0183). Known-number reproduction against
  `binned_pool_cache.py`'s cached `dv_true`, on 190 re-executed seed-pool
  rows: Pearson r 0.959, MAE 0.00497, rms 0.00655 against a cache sd of
  0.02263, means +0.00103 (Genesis) vs +0.00196 (cache).

verdict: inconclusive
downgrades: [imprecision, provenance]
grade: low
---

# EXP-0023 — a dynamics model as a source of gradients

The plan, the rationale, the blockers and the result that would embarrass
this design are in `DESIGN.md` in this directory, written and committed
before the run. This file records what was actually built and measured; it
does not restate the design.

## What was actually run

Four runs, each with its own `runs/RUN-*/RUN.md`:

- **RUN-0000** — the differentiability gate. All seven registered arms pass:
  finite, non-zero gradients in every one of the four action components, no
  zero-gradient rows, and 100 % sign agreement between the analytic
  directional derivative and a central finite difference. **This ran before
  any optimisation**, per `DESIGN.md` blocker 3.
- **RUN-0001** — model side, 6 arms x 10 states: rank-only pick from the
  shared pool, then 120 Adam steps on the model's own predicted `dv` with
  projected constraints.
- **RUN-0002** — Genesis: the CEM oracle and the full-fidelity execution of
  every arm's `a_rank`/`a_grad`, plus 190 cached-pool rows for the
  known-number check.
- **RUN-0003** — metrics.

The three `DESIGN.md` blockers were all real and all resolved:

1. **MPC adapters for the NFD family and the linear operators.** Added to
   `simple_mpc/adapters.py` as a SECOND adapter family
   (`make_occ_adapter` / `OCC_ADAPTERS`), because the existing `make_adapter`
   surface is observation-and-camera-driven and these models are not. No
   prediction math is duplicated: `PredictorGradientAdapter` calls each
   predictor's own `predict_occ` through `__wrapped__`, the function
   `@torch.no_grad()` decorated, so the differentiable path and the offline
   slate-scoring path are literally the same forward. Pinned by
   `tests/test_occ_gradient_adapters.py` and the new invariant
   `occ-gradient-adapter-matches-offline-predictor`.
2. **A differentiable length gate for the switched-linear operators.**
   `Baselines/LinearForesight/model.py::soft_bin_weights` /
   `predict_switched_soft`, alongside the untouched hard gate. Triangular
   interpolation between adjacent bin centres; exact agreement with the hard
   gate at every centre, 50/50 halfway between. Both gates are run as
   separate arms here so the relaxation's cost is visible, not assumed.
3. **Verified differentiability.** RUN-0000, above.

**Working tree was dirty** (61 pre-existing modified files at task start,
untouched by this record, plus this record's own new/modified files:
`simple_mpc/adapters.py`, `Baselines/LinearForesight/model.py`,
`tests/test_occ_gradient_adapters.py`, the four docs, `METRICS.md`,
`INVARIANTS.md`, and this directory). Nothing was committed.

**Two deliberate deviations from `DESIGN.md`**, both stated rather than
silently substituted:

- The sign of the three difference metrics (next section). `DESIGN.md`'s
  formulae and its own glosses disagree under this repo's cost convention.
- The switched-linear arm uses `weights/MODEL-0001-stage2-visual-switched`
  (the promoted 32x32 switched-linear visual operator, and the instance whose
  numbers on this exact pool are already cached) rather than
  `Baselines/LinearForesight/runs/operators_res*.pt`. Same family, same bin
  scheme, same warp; MODEL-0001 is the one with a prior on this corpus.

## Sign convention — fixed once, asserted in code

`dv = value(after) - value(before)` with `value = control_utility_test
.lyapunov`, i.e. a **COST: lower is better**, identical to
`scripts/probes/binned_pool_cache.py` and to `experiments/METRICS.md`'s
`lyapunov` row. `simple_mpc.adapters.assert_dv_convention` asserts it from a
synthetic near/far occupancy pair and is called at the top of every arm in
both stage 1 and the gradient gate.

**`DESIGN.md` writes `gradient_gain = dv_grad - dv_rank` and glosses
"negative = optimising made things WORSE". Those two are inconsistent under
this repo's cost convention**, where a *lower* dv is better and so
`dv_grad - dv_rank < 0` means optimisation HELPED. This record keeps the
design's MEANING and negates its formula, for the three difference metrics:

    gradient_gain    = dv_rank      - dv_grad      > 0 => optimising helped
    pool_escape      = pool_ceiling - dv_grad      > 0 => beat the whole pool
    regret_vs_oracle = dv_grad      - dv_oracle    > 0 => headroom remains

`capture_vs_oracle = dv_grad / dv_oracle` is kept exactly as `DESIGN.md`
writes it: both are costs measured from the same `v0`, so the ratio already
reads as "fraction of the oracle's improvement captured".

## Results

### The headline: `gradient_gain` is positive on average and unreliable per state

Every arm's mean `gradient_gain` is positive — optimising against the model
beat its own best pick from the pool, on average. But **every arm also has 2
to 4 of 10 states where it is negative**, i.e. where following the model's
gradients produced a worse action than the model's own rank-only choice. The
per-state table is in `results/metrics.json`; the spread within an arm
(sd 0.006 to 0.038) is two to three times its mean. `DESIGN.md` called a
negative value "the single most informative outcome this test can produce" —
it occurs, on 18 of the 60 (arm, state) cells, and it is not concentrated in
one bad arm.

### The power check FAILS — the arm ordering must not be read

sd of the six arm means 0.0105, sd of the ten state means 0.0163, residual
sd 0.0183. `DESIGN.md` named this exact outcome as the one that would
embarrass the design, with the guard "if between-arm variation is smaller
than between-state variation, this design lacks the power to answer its own
question and the honest report says so." **It is smaller, so this report says
so.** Ten states is not enough to rank six models on this quantity. Nothing
below should be read as "model A is a better gradient source than model B"
unless the gap is large compared with 0.016.

Two differences ARE large compared with that spread and are reported as
findings rather than as an ordering:

- **`nfd_residual_warped` is a poor gradient source in absolute terms.**
  `gradient_gain` +0.0021 (sem 0.0018), capture 0.112 of the oracle, and
  `pool_escape` negative on **10 of 10** states. Optimising against it never
  once found an action better than the best the seed pool already contained.
  It is also the weakest RANKER here (`dv_rank` -0.0087, against a pool mean
  of +0.0023 and a pool best of -0.0446), so this is not a case of a good
  ranker with bad gradients — it is weak on both axes. (Note its checkpoint
  is the L20mm pilot, trained on a single push-length corpus, while the two
  strongest arms are `randlen`-trained. That is a plausible mechanism and is
  NOT tested here.)
- **No arm approaches the oracle.** Best capture 0.629; mean regret 0.036 to
  0.089 Lyapunov units against an oracle mean of -0.100. The pool is not what
  limits control performance at this pool size — dynamics-model error is.

### `pool_escape`: gradients are not just a slower pool search

Four of six arms escape the pool on average (`pool_escape` > 0), on 4 to 6 of
10 states. Two do not (`nfd_3ch_finetuned` -0.0108 on 3/10,
`nfd_residual_warped` -0.0338 on 0/10). So for the better arms the answer to
`DESIGN.md`'s standing question is: a wider pool would NOT simply substitute
for gradients — optimisation reaches actions the 100-candidate pool did not
contain — but it also does not close the gap to the oracle.

**Caveat on `pool_escape` specifically.** `pool_ceiling` comes from the
CACHED `dv_true` of the 100 pool rows, produced by the corpus collection's
code path; `dv_grad` comes from this record's Genesis re-execution. Those two
paths agree at r = 0.959 with MAE 0.00497 (see below), not exactly, and
`pool_escape` is a difference between them — so a `|pool_escape|` under about
0.005 should be treated as unresolved. That affects
`linear_switched_hard`'s +0.0046 and, marginally, `nfd_3ch_randlen`'s
+0.0102; it does not affect `nfd_residual_warped`'s -0.0338 or
`nfd_warped_randlen`'s +0.0197.

### The soft length gate vs the hard one

`linear_switched_soft` beats `linear_switched_hard` on all three headline
numbers — `gradient_gain` +0.0207 vs +0.0123, `pool_escape` +0.0153 vs
+0.0046, capture 0.623 vs 0.486 — on the SAME operators, the same seed pool
and the same optimiser, differing only in whether the push-length gate is
differentiable. **The gap (0.0084 in `gradient_gain`) is smaller than the
between-state sd (0.0163), so this is suggestive, not established**, and the
soft gate also hits the length bound more often (0.235 vs 0.033 of steps).
The mechanism is nonetheless the one predicted: the hard gate's length
gradient is missing the term that says "a longer push would use a different
operator", and the soft arm moves push length much more freely as a result.

### Constraint bounds — how often each arm was scored on the constraint

Fraction of the 120 Adam steps at which a projection actually moved the
action, averaged over states, and the number of states where any projection
fired at all:

| arm | box | length | states with any bound hit |
|---|---|---|---|
| `nfd_warped_randlen` | 0.012 | **0.324** | **7/10** |
| `linear_switched_soft` | 0.026 | 0.235 | 4/10 |
| `nfd_3ch_randlen` | 0.087 | 0.177 | 3/10 |
| `nfd_3ch_finetuned` | 0.082 | 0.103 | 2/10 |
| `linear_switched_hard` | 0.000 | 0.033 | 2/10 |
| `nfd_residual_warped` | 0.000 | 0.023 | 1/10 |

**The best-scoring arm is also the most constrained.** `nfd_warped_randlen`
spends about a third of its optimisation steps pinned to the 20–70 mm push
length limit, on 7 of 10 states. Its `gradient_gain` is therefore partly a
statement about the constraint (the models want longer pushes than they were
fitted on) rather than purely about its gradients. The direction is always
the upper bound — the models are extrapolating toward longer pushes, which is
precisely the "optimisation walks to where the model is most wrong in the
optimistic direction" failure mode the design was built to expose, here
caught by the constraint rather than by the model.

### Known-number reproduction

190 seed-pool rows (19 per state) re-executed in Genesis from the restored
snapshot, against `binned_pool_cache.py`'s cached `dv_true`:

| | value |
|---|---|
| Pearson r | 0.959 |
| MAE | 0.00497 |
| rms | 0.00655 |
| cache sd (the scale) | 0.02263 |
| means | +0.00103 (Genesis) vs +0.00196 (cache) |

The two agree well — 96 % correlated, with an error 22 % of the quantity's
own sd and a bias of 0.0009 — but **they are not identical**, so this
pipeline reproduces the cached numbers rather than reproducing them exactly.
This is NOT in tension with invariant
`genesis-snapshot-restore-repeat-determinism`, which is about repeating the
same action from the same snapshot in the same process; here the comparison
crosses code paths (the corpus's own collection driver vs this record's
snapshot-restore-and-execute), which is why this record carries the
`provenance` downgrade.

## Unrelated findings

- **`predict_switched`'s masked in-place assignment into a clone does survive
  autograd.** `DESIGN.md` flagged it as a differentiability risk. It is not
  one: the hard-gate arm passes the full gradient gate with non-zero
  gradients in all four action components. The hard gate's actual defect is
  narrower than "no gradient" — the warp, the operator and the occupancy path
  all carry gradient, and only the CHOICE of operator is a step function in
  push length, so what is missing is one term in the length gradient. Worth
  stating because "bucketize is not differentiable" is easily overread into
  "this model cannot be optimised against at all", which is false.
- **Reduced- and full-fidelity Genesis rollouts give materially different
  `dv`.** The CEM oracle selects under the halved settle budget
  (`rollout_settle_steps: 50`) and is re-executed at full fidelity; the two
  differ by a few percent typically and by 21 % on one state (slate 5:
  planning best -0.1088, achieved -0.0854). Anyone quoting a planning
  rollout's own cost as an achieved result will overstate it. Recorded in
  `docs/oracle_mpc_design.md`.
- **`GenesisOracleEnv.restore_snapshot` is exact.** `restore_err` (max abs
  difference between the particle state asked for and the state the env then
  holds) is 0.0 on all 10 states, so an arbitrary corpus state can be used as
  an oracle start state with no fidelity loss at the restore step. Also
  documented in `docs/oracle_mpc_design.md`.
- **The average push in a DS-0001 pool slightly HURTS** under
  `corner`/`lyapunov`: pool mean true `dv` +0.00234, and only 35-61 % of
  candidates improve the value. The `random` floor for this task is therefore
  marginally worse than doing nothing, which is worth knowing before reading
  any capture-style ratio on this corpus.
- **`experiments/temp/multistep-rollout/rollout.py::RawStub` was being
  re-derived per experiment.** Promoted to `simple_mpc.adapters.SlateRawStub`
  with a test pinning its geometry.

## What would change the verdict

- A larger seed pool. `pool_ceiling` is computed over 100 of the 1000
  candidates DS-0001 holds; a 1000-action pool would raise the bar
  `pool_escape` is measured against and could flip its sign.
- A deeper oracle. 4 CEM iterations x 32 candidates is 128 Genesis rollouts
  per state; `capture_vs_oracle` is a ratio against THAT, not against the
  true optimum, so it is an upper bound on captured fraction.
- More states. The power check below is the honest limit on how far the arm
  ordering can be read.
