---
# ---- identity -------------------------------------------------------------
id: EXP-0017
title: >
  Turning the descriptor value readout into a reusable, persisted instrument:
  the readout beats `goal_only` on held-out goals only at nonlinear capacity,
  and EXP-0015's Fourier-order reversal is a confirmed inner-CV leak
tier: T1
mode: exploratory
date: 2026-09-15
hypothesis: >
  EXP-0015's suspected mechanism for its Fourier-order anomaly -- that the
  inner CV folds selecting the ridge lambda are row-random rather than
  slate-aware, while post-push states within a slate are near-duplicates --
  is true of the code and accounts for the anomaly.

# ---- the claim ------------------------------------------------------------
claim: >
  On DS-0001 step 0 with 500 legal-configuration goals and 87-dim analytic
  occupancy descriptors (n_fourier=8), a value readout
  h(phi_state, phi_goal) -> scalar beats the mandatory `goal_only` baseline on
  HELD-OUT GOALS only at nonlinear capacity: MLP(128,128) on `diff` features
  reaches `value_readout_r2` 0.871+-0.010 (lyapunov), 0.644+-0.033
  (mass_in_region), 0.703+-0.026 (signed_mass_in_region) over 3 independent
  400/100 goal splits, against a best-capacity `goal_only` of 0.327+-0.064 /
  0.364+-0.064 / 0.363+-0.057 -- increments of +0.544 / +0.280 / +0.340,
  4-8x the ~0.05-0.07 goal-split noise floor. At LINEAR capacity with `diff`
  features the readout is BELOW `goal_only` for both mass functions
  (-0.080, -0.083), reproducing EXP-0015's caution. The capacity ordering
  linear < poly-2 < MLP is resolved for linear->poly-2 in every value function
  and for poly-2->MLP only in `lyapunov` (and weakly in
  `signed_mass_in_region`); poly-2 vs MLP for `mass_in_region` is inside the
  floor and is UNRESOLVED. Separately, EXP-0015's Fourier-order reversal is
  CONFIRMED to be caused by row-random inner CV folds: making the folds
  slate-aware, with nothing else changed, removes every negative cell
  (16/16) and raises the selected lambda from 0.01-1 to 30-3000.
  Scope: analytic descriptors only; no learned latent and no `slateN` score.

prediction:
  discriminating: true
  statement: >
    If the inner-CV leak is the mechanism, slate-aware inner folds will select
    a larger lambda and remove the monotone Fourier-order collapse; if it is
    not, the collapse will persist under slate-aware folds. Separately, if the
    readout is usable for action ranking, the held-out-GOAL increment over
    `goal_only` (not raw R2) will exceed the ~0.05-0.07 goal-split noise floor
    at some capacity; if it does not at ANY capacity, the readout approach
    cannot rank actions and the latent programme needs a decoder instead.
  outcome: >
    Both discriminating cells ran. Leak: CONFIRMED. Increment over `goal_only`:
    exceeded the floor by 4-8x at MLP capacity for all three value functions.

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0017-value-readout-instrument/code/run0001_capacity_intervals.py
  data: ["DS-0001"]
  code_path: experiments/EXP-0017-value-readout-instrument/code/
  seed: 0
  split: >
    Reported as a pair per METRICS.md `value_readout_r2`. Held-out GOAL:
    random 400/100 goal splits, seeds 100..103 (4 for linear, 3 for poly-2 and
    MLP, 1 for poly-2 `goal_only`). Held-out STATE: slate-aware, 16/4 of the 20
    slates, seeds 0..2 (1 split for the MLP cells). States are POST-push states
    of DS-0001 step 0. Features/targets reused verbatim from EXP-0015
    (`experiments/temp/desc-value-readout/stage3_features_and_targets.pt`), so
    the two records' cells are directly comparable.
  runtime: "~35 min wall-clock, CPU only (CUDA_VISIBLE_DEVICES=\"\"), 4-core box"
  runs: [RUN-0001]

budget:
  declared: "75 min / 150k tokens (delegated subagent, declared HARD, CPU only)"
  spent: "~60 min / ~95k tokens"
  outcome: within

design:
  varied:
    capacity: [linear, poly2, "MLP-128-128"]
    feature_mode: [diff, concat, state_only, goal_only, mean_only]
    value_fn: [lyapunov, mass_in_region, signed_mass_in_region]
    split_kind: [held-out-goal, held-out-state]
    split_seed: [100, 101, 102, 103]
    inner_cv_folds: [row_random, slate_aware]
    n_fourier: [8, 16, 24, 32]
  held_fixed: {dataset: DS-0001, step: 0, descriptor: "dmdc_baseline.occupancy_descriptors",
               n_fourier: 8, grid: 64, n_objects: 20, n_goals: 500,
               goal_representation: "legal same-count material configuration",
               max_train_rows: 25000, max_eval_rows: 20000,
               mlp: "(128,128) relu alpha=1e-3 max_iter=300 early_stopping"}
  baselines: [mean_only, goal_only, state_only]
  metric: "value_readout_r2"

noise_floor: >
  ~0.05-0.07 R2, the goal-split spread measured by EXP-0015 RUN-0003 and
  reproduced here as the sd over 3-4 independent goal splits (0.010-0.064
  across all held-out-GOAL cells). The `mean_only` baseline is exactly 0.000
  with sd 0.000 in every cell, as it must be by construction, which confirms
  the R2 normalisation is against the TRAINING mean. Held-out-STATE spreads
  are much larger (up to 0.24), so held-out-STATE differences below ~0.3 are
  not rankable at n=3.

depends_on: [occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  (1) HEADLINE -- the readout DOES beat `goal_only` on held-out goals, but only
  at nonlinear capacity. MLP `diff`: lyapunov +0.871+-0.010, mass_in_region
  +0.644+-0.033, signed_mass +0.703+-0.026 (3 goal splits) vs best-capacity
  `goal_only` +0.327+-0.064 / +0.364+-0.064 / +0.363+-0.057; increments +0.544
  / +0.280 / +0.340 against a ~0.06 floor. At LINEAR capacity `diff` is BELOW
  `goal_only` for both mass functions (-0.080, -0.083) and `concat` beats it by
  only +0.013 / +0.023, i.e. inside the floor -> UNRESOLVED. (2) Capacity with
  an interval (EXP-0015's missing cell): linear / poly-2 / MLP =
  0.420+-0.051 / 0.794+-0.016 / 0.871+-0.010 (lyapunov),
  0.285+-0.051 / 0.585+-0.061 / 0.644+-0.033 (mass_in_region),
  0.278+-0.050 / 0.597+-0.055 / 0.703+-0.026 (signed_mass). linear->poly-2 is
  resolved everywhere; poly-2->MLP is resolved for lyapunov (+0.077) and weakly
  for signed_mass (+0.106) but UNRESOLVED for mass_in_region (+0.059 vs spreads
  0.061/0.033). EXP-0015's single-split numbers all lie within ~1-2 sd of these
  means, so its ordering survives; only its precision was missing. (3) THE
  INNER-CV LEAK IS CONFIRMED, by reading the code and then testing it. Both
  `stage0_upper_bound.py::fit_eval_cv` and `stage3b_regression.py::fit_eval_cv`
  build inner folds as `np.array_split(rng_cv.permutation(n), 3)` -- row-random
  -- while the outer split is slate-aware. Re-running the Fourier sweep with
  the inner folds grouped by slate and nothing else changed: every one of the
  16 cells is positive (was negative in 12/16), the catastrophic values vanish
  (letter_T/mass nf=16: -38.90 -> +0.302; letter_T/lyapunov nf=16: -18.66 ->
  +0.518), and the selected lambda rises from 0.01-1 (15/16 cells) to 30-3000
  (12/16 cells). What is NOT established is that higher Fourier order helps:
  under slate-aware folds the trend is mixed (quadrant0/lyapunov drifts
  0.890 -> 0.591, letter_T/mass rises 0.070 -> 0.497) with spreads 0.04-0.49
  covering most of it -- Fourier order is UNRESOLVED, not reversed. (4) The
  negative held-out-STATE R2 at 500 goals reproduces (-0.075+-0.237,
  -0.134+-0.239) and is shown to be a LINEAR-CAPACITY effect: the MLP moves
  both mass functions positive (+0.332, +0.108). Only partly resolved -- both
  stay below `goal_only` (0.424, 0.425) on held-out STATE even at MLP capacity,
  and those MLP cells are a single state split. (5) The instrument exists and
  is persisted: `code/value_readout.py::ValueReadout` takes arbitrary state and
  goal embedding matrices (`concat` mode does not require matching dims), fits
  at three capacities, computes the three mandatory baselines on every fit,
  exposes `predict` and `dv(z_now, z_pred, goal)`, and save/load; three fitted
  MLP readouts are saved under `experiments/temp/exp0017-value-readout/`.
verdict: supported
downgrades: [indirectness, incomplete-design, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## What was uncommitted (dirty tree)

`git status --porcelain` at run time, on branch `baselines/overnight` at
`6ea03278`. Modified and uncommitted: `.claude/skills/experiment-log/SKILL.md`,
`.claude/skills/project-overview/SKILL.md`, `Genesis/sandbox_manipulation_clean.py`,
`docs/piled_collection.md`, `experiments/{INVARIANTS,METRICS,REGISTER,TEMP_LOG}.md`,
`experiments/COMMANDS.jsonl`, `scripts/probes/{pool_common,pool_inspect,pool_survey}.py`,
`weights/MODEL-000{1,2,3}-*/tests.md`. Untracked: `.claude/orchestrator-notes.md`,
`.claude/skills/{data-collection,subagent-experimenter}/`,
`Baselines/common/goal_configs.py`, `Genesis/binned_slate_{collection,dataset}.py`,
`Genesis/data/{Sean,slates_binned}/`, `datasets/`, `docs/CODEMAP.md`,
`docs/experimental_design/jepa_based_encoder.md`,
`experiments/EXP-001{1,2,3,4,5,6,7}-*/`, `scripts/probes/binned_pool_cache.py`.

Load-bearing for this record: `Baselines/common/goal_configs.py` (produced the goal
configurations), `Genesis/binned_slate_dataset.py` (corpus loader, used by the Fourier
re-run), `experiments/EXP-0015-*/` (the reused features/targets and the code whose CV
folds were audited), and this record's own tree.

## Why the verdict is `supported`

Both cells the prediction named as discriminating ran to completion: the
`goal_only` increment exceeded the noise floor at MLP capacity for all three
value functions, and the leak hypothesis was tested by changing exactly one
line of fold construction. `mode: exploratory` is retained because the
capacity sweep itself carries no committed prediction -- only the leak test
does.

`downgrades`: `indirectness` (this measures a regression R2, not `slateN`; no
claim about ranking ability follows until a readout is scored under `slateN`),
`incomplete-design` (the cells below), `untested-dependency` (both
`depends_on` tags are not `holds`/`fixed`).

## What was actually run

**RUN-0001** in two concurrent CPU processes (see `runs/RUN-0001-capacity-intervals/RUN.md`):

* Parts A+B — capacity x feature-mode x value-function x repeated split, held-out
  GOAL and held-out STATE, through `ValueReadout`, persisting the MLP.
* Part C — the Fourier-order sweep re-run with row-random vs slate-aware inner
  CV folds, on a reduced grid (2 goals x 2 value functions x 4 Fourier orders x
  3 repeats x 2 fold constructions) copied from EXP-0015's `stage0_upper_bound.py`
  with only `fit_eval_cv`'s fold construction changed.

No pilot revealed an outcome before the prediction was written: the only pilot
was a three-cell runtime measurement used to size the sweep, and its R2 values
reproduced EXP-0015's already-published stage-4 numbers.

## Cells NOT run (the `incomplete-design` downgrade)

| cell | why | cost |
|---|---|---|
| poly-2 `goal_only`, splits 101-103 | poly-2 costs ~63 s/fit on 4 cores; 1 split only | ~4 min |
| poly-2 / MLP with `concat` features | `concat` doubles the feature count; poly-2 would be ~4x cost | ~25 min |
| MLP held-out-STATE, splits 1-2 | left 1 split; this is why (4) is only partly resolved | ~5 min |
| `slateN` for any readout | needs the pool-ranking harness, not a regression split | ~30 min |
| the slate-aware-CV re-fit of EXP-0015's PERSISTED stage2/stage3 ridge models | those used the same leaky `cv_lambda`; see below | ~10 min |

## Consequence for EXP-0015 (for the coordinator — REGISTER.md not edited here)

Per `register-validator`'s "When a bug is found", the coordinator should consider:

1. The leak is in EXP-0015's own code, not in a shared invariant, so no
   `INVARIANTS.md` tag goes `broken`. It is a defect of one experiment's method.
2. **EXP-0015's result (2), the Fourier-order sweep, is `invalidated`** — the
   measurement is an artifact of row-random inner folds, now confirmed. EXP-0015
   already flagged it as "NOT usable until checked"; it can now be marked
   invalidated with `invalidated_by: EXP-0017`.
3. **EXP-0015's results (1), (3), (4) and the stage2/stage3 persisted ridge
   models are affected in the same direction but not overturned**: they all ran
   through the same row-random `fit_eval_cv` / `cv_lambda`, so their lambdas are
   likely too small and their linear numbers are, if anything, pessimistic. The
   capacity result (5) does NOT go through it (stage 4 uses sklearn `RidgeCV`),
   and it reproduces here. Suggested status: `narrowed`, not `refuted`.
4. Any `REGISTER.md` row citing an EXP-0015 Fourier-order number should move to
   `invalidated`; a row citing its capacity ordering can be upgraded to cite
   EXP-0017's interval instead.
5. The invariant-test analogue worth adding: a test that any inner CV in this
   repo's readout code groups by `slate_of_state`. Not written here.

## Unrelated findings

* **An MLP fitted on `goal_only` features can diverge.** `mlp`/`goal_only`/
  `lyapunov` on held-out goals scored **-3.010 +- 4.612** over 3 splits — one
  split produced a large negative R2 while the linear `goal_only` on the same
  splits is a stable +0.327 +- 0.064. A baseline evaluated at the model's
  capacity is therefore not automatically the right reference; this record
  quotes `goal_only` at its BEST capacity so the comparison cannot be won by
  handicapping the baseline. Anyone reusing `ValueReadout` should read the
  per-capacity baseline rows, not just the model's own capacity row.
* **`concat` beats `diff` at linear capacity** for all three value functions
  (+0.031 to +0.106 held-out GOAL). EXP-0015 only ever used `diff`. This is
  practically relevant for the LeJEPA programme: a learned latent has no reason
  for `phi_state - phi_goal` to be meaningful, and `concat` — which is also the
  only mode that works when state and goal embeddings differ in dimension — is
  not worse and is somewhat better at the capacity where it was tested.
* `mean_only` returns exactly 0.000 +- 0.000 in all 27 cells where it ran,
  which is a useful check that the METRICS.md training-mean normalisation is
  implemented correctly.

## What would change the verdict

- **`slateN` on DS-0001 using a persisted MLP readout.** This record measures
  regression quality; the motivating question (EXP-0014) is ranking. The
  `dv()` method exists for exactly this and was not exercised. ~30 min.
- **MLP held-out-STATE over 3+ splits**, to close the last piece of the
  negative-R2 anomaly. ~5 min.
- **Re-running EXP-0015's stage2/stage3 fits with slate-aware `cv_lambda`**,
  so its persisted linear models are not fitted at a leaked lambda. ~10 min.
- **A learned-latent drop-in.** Nothing here shows the instrument works on a
  LeJEPA latent; it shows it works on analytic descriptors and that its API
  does not assume them.
