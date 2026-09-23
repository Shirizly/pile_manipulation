---
# ---- identity -------------------------------------------------------------
id: EXP-0015
title: >
  A value readout for descriptor-space predictions: representing goals as
  legal same-count material configurations fixes the goal-generalisation
  blow-up, and the remaining barrier is linearity, not information
tier: T1
mode: exploratory
date: 2026-09-15
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  A scalar value function of a pile state and a goal region is recoverable
  from world-frame analytic occupancy descriptors of the two
  (`dmdc_baseline.occupancy_descriptors`, 87-dim), but NOT by a linear map:
  on a 500-goal dataset with entire goal shapes held out, `value_readout_r2`
  rises monotonically with model capacity -- linear 0.362 / poly-2 0.751 /
  MLP(128,128) 0.893 for `lyapunov`, 0.213 / 0.499 / 0.689 for
  `mass_in_region`, 0.205 / 0.482 / 0.749 for `signed_mass_in_region`.
  Separately, representing the goal as a LEGAL same-object-count material
  configuration rather than a solid rasterised mask reduces the held-out-goal
  failure of the linear model by two to three orders of magnitude (lyapunov
  -189.9 -> -1.39; signed_mass -366 -> -1.55), confirming that solid masks
  and granular states occupy different regions of descriptor space and their
  difference vector is largely uninformative about value.

prediction: null

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0015-descriptor-value-readout/code/stage3b_regression.py
  data: ["DS-0001"]
  code_path: experiments/EXP-0015-descriptor-value-readout/code/
  seed: 0
  split: >
    Two splits reported as a pair throughout (see METRICS.md
    `value_readout_r2`): held-out STATE (slate-aware, 16/4 slates, 3 repeats)
    and held-out GOAL (leave-one-goal-out over 6 goals in RUN-0002; a real
    400/100 goal split, 3 repeats, in RUN-0003). States are the POST-push
    states of DS-0001 step 0, which are all distinct -- the PRE-push states
    are not (a same-state slate shares one pile across its 1000 candidates,
    so splitting on those rows would leak exactly).
  runtime: "~52 min total across RUN-0001..RUN-0004, CPU"
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004]

budget:
  declared: "120 min / 200k tokens (delegated subagent, declared HARD)"
  spent: "~52 min / ~139k tokens"
  outcome: stopped-early

design:
  varied:
    RUN-0001: {n_fourier: [8, 16, 24, 32], value_fn: [lyapunov, mass_in_region, signed_mass_in_region], goal: 4}
    RUN-0002: {goal_representation: [solid-mask, legal-configuration]}
    RUN-0003: {n_goals: [6, 500]}
    RUN-0004: {capacity: [linear, poly-2, MLP-128-128]}
  held_fixed: {dataset: DS-0001, step: 0, descriptor: "dmdc_baseline.occupancy_descriptors",
               grid: 64, n_objects: 20, ridge: "CV-chosen over 0.01..1e6"}
  baselines: [mean_only, state_only, goal_only]
  metric: "value_readout_r2"

noise_floor: >
  RUN-0003's goal-split repeat spread, ~0.05-0.07 R2. RUN-0004's capacity
  gaps (0.36 -> 0.75 -> 0.89 for lyapunov) are several times that, so the
  capacity ORDERING is credible; the individual RUN-0004 numbers are from a
  SINGLE split and carry no interval of their own. RUN-0002's before/after
  differences span orders of magnitude and are far outside any plausible
  floor.

depends_on: [occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x,
             readout-cv-folds-slate-aware]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  (1) Upper bound, goal fixed, value from the TRUE state descriptor: value IS
  recoverable, so there is no information ceiling at zero -- `lyapunov`
  reaches R2 0.90 (sem 0.07) and 0.97 (0.01) on large smooth goals, but
  -1.47 on a thin goal (letter_O), and `mass_in_region`/
  `signed_mass_in_region` are not recoverable at n_fourier=8 for any of 4
  fixed goals (+0.31 down to -8.9). (2) Raising Fourier order 8->16->24->32
  made every cell WORSE (quadrant0/lyapunov 0.90 -> 0.12 -> -0.76 -> -0.72),
  the opposite of the coordinator's stated prior; MECHANISM NOT ESTABLISHED
  -- the leading suspect is that the inner CV folds selecting the ridge
  lambda are row-random rather than slate-aware, and post-push states within
  a slate are near-duplicates. This sub-result is NOT usable until that is
  checked. (3) Goal-as-configuration vs goal-as-mask, held-out-goal R2:
  lyapunov -189.9+/-263 -> -1.39+/-2.11; mass_in_region -5.48+/-3.49 ->
  -1.73+/-1.52; signed_mass -366+/-645 -> -1.55+/-1.05. Still negative at 6
  goals, which is too few to test goal generalisation. (4) At 500 goals,
  held-out-goal turns positive: lyapunov diff 0.406+/-0.056 vs goal_only
  0.311+/-0.069; but for both mass functions diff (0.271, 0.265) sits BELOW
  goal_only (0.346, 0.342), i.e. goal identity still dominates. Held-out-
  STATE R2 for the two mass functions went NEGATIVE at 500 goals (-0.12,
  -0.16) having been +0.42 at 6 goals -- unexplained, marked unresolved. (5)
  Capacity, held-out goal, single split: linear/poly-2/MLP = 0.362/0.751/
  0.893 (lyapunov), 0.213/0.499/0.689 (mass_in_region), 0.205/0.482/0.749
  (signed_mass). Generator: 1560 configurations generated, non-penetration
  assertion passed on all of them; generated-config pixel mass 96.0+/-5.4 vs
  real-state 94.6+/-2.1.
verdict: inconclusive
downgrades: [incomplete-design, imprecision, inconsistency, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## What was uncommitted (dirty tree)

`Baselines/common/goal_configs.py` (new), the `docs/CODEMAP.md` edit, this
record's own files, and everything under `experiments/temp/desc-value-readout/`
and `experiments/temp/desc-value-regression/` were uncommitted at run time.

## Why the verdict is `inconclusive`, not `supported`

The capacity result (5) is the finding worth having, and it is from a
**single split** with no interval of its own, and its fitted MLP was **not
persisted** — only its R2 was recorded — so it cannot be re-tested without a
refit, and a refit is only probably the same object. Under this skill's own
rule, a design whose decisive cell is missing has tested nothing: the
decisive cell here is "does the capacity ordering survive repeats", and it
was not run. The goal-representation result (3) is robust to this and is the
part a later record can lean on.

## What was actually run

**RUN-0001** — upper-bound probe: value regressed on `phi(state)` alone with
the goal held fixed, ridge with CV-chosen lambda, slate-aware outer split,
swept over descriptor Fourier order.

**RUN-0002** — the goal-configuration generator
(`Baselines/common/goal_configs.py::mask_to_configuration`) and the
mask-vs-configuration rerun of the value regression over 6 goals, matched to
the protocol of the earlier `experiments/temp/desc-value-regression/` run so
the before/after table is comparable.

**RUN-0003** — a 500-goal dataset (300 rectangles with area
`0.15 + 0.35*Beta(2,5)` of the workspace, 200 letters T/A/O/S/G, all at
random rotation), then the same regression with a real 400/100 goal split.

**RUN-0004** — capacity escalation: linear, degree-2 polynomial features, and
an MLP(128,128), scored on held-out goals.

**RUN-0005 (the decoder) was NOT run** — budget was exhausted after RUN-0004.
See "What would change the verdict".

## The generator, and why it is project code rather than experiment code

`Baselines/common/goal_configs.py::mask_to_configuration` turns a goal mask
into a legal material configuration: `n_objects` cubes, none outside the
mask, spread roughly uniformly, with NO penetration. Non-penetration is exact
rather than checked-by-sampling: a cube of edge `a` at any yaw has
axis-aligned footprint half-extent at most `a*sqrt(2)/2`, so a grid pitch
`p >= a*sqrt(2)` with per-axis jitter at most `(p - a*sqrt(2))/2` admits
arbitrary yaw and jitter without overlap. The pitch is found by bisection so
that the number of in-mask grid points is 1.0-1.2x `n_objects`, and the
surplus is dropped at random; thin masks fall back to rejection sampling and
mask dilation. This is a reusable capability (any goal shape, any object
count), not a composition of existing ones, so it lives beside
`Baselines/common/goals.py` per `project-overview`'s ownership rule.

## Unrelated findings

The earlier attempt recorded in `experiments/temp/desc-value-regression/`
(same question, solid-mask goals, 6 goals) additionally found that its
apparent held-out-STATE skill was a between-goal artifact: a `goal_only`
baseline that ignores the state entirely and predicts one constant per goal
matched or beat the difference model (0.427 vs 0.428 for `mass_in_region`;
0.422 vs 0.300 for `signed_mass_in_region`), and per-goal-conditional R2 with
the goal fixed was near zero or negative almost everywhere. That is why
`goal_only` is a mandatory baseline in this metric's definition.

## What would change the verdict

- **Repeats on the capacity comparison, and persisting the MLP.** ~20 min.
  This is the single cell that would move the verdict from `inconclusive`.
- **The descriptor->raster decoder** (not run). It is the more useful
  instrument than a value regression: fitted once, it serves every goal and
  every value function through the existing value-function code unchanged,
  and it can be scored against the true raster directly instead of through a
  scalar. It should be benchmarked against the MLP, not the linear baseline.
- **The Fourier-order anomaly.** Re-run RUN-0001 with slate-aware inner CV
  folds before anyone cites a Fourier-order number.
- **The held-out-state regression at 500 goals** (positive at 6 goals,
  negative at 500, for the two mass functions) is unexplained and should be
  isolated before this readout is used for ranking.
- **Nothing here has been scored under `slateN`.** This record measures a
  readout's regression quality, which is a proxy for the ranking ability the
  motivating question (EXP-0014) is actually about. Until a readout is
  scored under `slateN` on DS-0001, no claim about MODEL-0002's ranking
  ability follows from these numbers.

## Amendment 2026-09-15 — result (2) invalidated by EXP-0017 (inner-CV leak)

EXP-0017 confirmed, by reading this record's own code and then testing it,
that `stage0_upper_bound.py::fit_eval_cv` (line 94) and
`stage3b_regression.py::fit_eval_cv` (line 80) build their INNER
cross-validation folds as `np.array_split(rng_cv.permutation(n), 3)` — row-
random — while the outer split is slate-aware via `split_state`. Post-push
states within one slate are near-duplicates, so a row-random inner fold puts
near-copies of each validation row into its own training fold and selects a
ridge `lambda` far too small. New tag: `readout-cv-folds-slate-aware`
(`broken`).

**Result (2) — that raising Fourier order 8→16→24→32 made every cell
monotonically worse — is `invalidated`.** Regrouping the inner folds by slate
and changing nothing else turns all 16 cells positive (12 of 16 had been
negative; letter_T/`mass_in_region` at nf=16 goes -38.90 → +0.302), and the
selected `lambda` rises from 0.01–1 in 15/16 cells to 30–3000 in 12/16. The
measured reversal was an artifact of the fold construction, not a property of
Fourier order. Under slate-aware folds the Fourier-order trend is mixed and
inside its spread — **unresolved, not reversed**. The "leading suspect"
recorded in this record's own result (2) was therefore correct.

**Results (1), (3) and (4), and the persisted stage2/stage3 ridge models, are
`narrowed`, not refuted.** They ran through the same leaky `lambda` selection,
so their linear-capacity numbers are pessimistic — but (3)'s before/after
comparison is matched (both sides used the same selection), so its two-to-
three-order-of-magnitude direction stands, and C-021 is unaffected in
direction. Result (5)'s capacity ORDERING survives: EXP-0017 re-measured it
under slate-aware folds across 3 goal splits and this record's single-split
numbers all lie within ~1–2 sd. The MLP this record did not persist is now
persisted by EXP-0017.
