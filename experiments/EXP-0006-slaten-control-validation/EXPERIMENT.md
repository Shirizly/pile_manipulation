---
# ---- identity -------------------------------------------------------------
id: EXP-0006
title: slateN control-ranking validation of the stage-2/3 model family on real same-state slates -- switching helps on corner, loses on stripe, and a descriptor-only model scoring 0 image accuracy ranks actions well above chance
tier: T1
mode: exploratory
date: 2026-09-13
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On `Genesis/data/slates_multistep` real same-state candidate pools
  (`n20_L10mm`/`n20_L20mm`/`n20_L40mm`, step-0 slates, K=N slateN, corner
  Lyapunov goal unless noted): (a) the stage-2 switched visual/hybrid
  operator (EXP-0005) beats persistence and random by a wide, well-powered
  margin on every dataset (e.g. pooled corner switched +0.919 vs
  persistence -0.093); (b) switching by push-length-bin beats the
  matching global/unswitched operator on the `corner` goal once pooled
  across datasets (19W/6L/35T of 60, ~3.5x paired sem) but LOSES to global
  on the `stripe` goal (0W/7L/33T of 40) -- the direction is goal-dependent,
  not a uniform win; (c) the stage-3 descriptor-only model (D-all-local,
  94-dim, scores exactly 0.0 image-space `accuracy` by construction) scores
  a real, well-powered positive slateN (pooled corner +0.461, sem 0.061)
  via a point-mass COM readout of its own descriptor prediction -- image
  accuracy and control-ranking usefulness are not the same currency for
  this model.

prediction: null  # exploratory

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 0ddab20f
  dirty: true                      # same pre-existing unrelated dirty tree
                                    # as EXP-0004/EXP-0005; this record's own
                                    # inputs are untracked temp scratch.
  data_commit: "unrecorded for slates_multistep's own generation provenance -- not checked in this promotion pass"
  script: "experiments/temp/stage2-slaten/eval_slaten.py; experiments/temp/stage3-slaten/eval_slaten_latent.py"
  data: ["Genesis/data/slates_multistep/{n20_L10mm,n20_L20mm,n20_L40mm} eval splits (20 slates x 128 candidates each, step 0 only)"]
  code_path: "Baselines.common.data.load_cell (real same-state pool loader) + control_utility_test.lyapunov_weights + Baselines.common.goals.slate_n_capture; stage-2 operators refit via hybrid-vis-desc/fit_hybrid.py::fit_cell; stage-3 operators refit via latent-switched/fit_latent_switched.py::fit_cell; descriptor-only operator refit from dmdc-lenbins's D-all-local cache"
  seed: 0
  split: "slates_multistep's own pre-built eval manifests (not this repo's randlen file split) -- 20 held-out slates per dataset, K=N=128 candidates per slate, no resampling at K=N; K=32 fixed-reference uses 300 without-replacement draws per slate"
  runtime: "not recorded precisely per source RESULTS.md; both runs completed within their sessions, nothing left incomplete for the cells that were attempted"
  runs: []

budget:
  declared: "not separately declared for this promotion; stage2-slaten Run 1 + Run 2 and stage3-slaten each ran under their own (unstated) per-session budgets"
  spent: "promotion pass: ~20 min / ~25k tokens"
  outcome: within

design:
  varied:
    goal: [corner, center, stripe]
    dataset: [n20_L10mm, n20_L20mm, n20_L40mm]
    model: ["visual switched", "visual global", "hybrid14 switched", "hybrid94 switched", "latent+desc94 switched", "latent+desc94 global", "latent-only switched", "descriptor-only (point-mass readout)", persistence, random]
  held_fixed:
    value_function: "lyapunov_weights (control_utility_test), the corner/stripe/center Lyapunov distance-transform potential"
    K: "K=N=128 (the real full pool, exact closed form, no resampling); K=32 fixed reference also reported per METRICS.md's cross-dataset comparability requirement"
    degeneracy_screen: "any (dataset, goal) cell with frac(dv_true==0) > 0.90 excluded as degenerate (center on n20_L10mm/n20_L20mm; stripe on n20_L10mm) -- ties would be forced to 1.000 for every model including persistence, not a skill measurement"
  baselines: [persistence, random]
  metric: "slateN (K=N=128) and slate32 (K=32 fixed reference), per experiments/METRICS.md -- the standard control metric, reported with wins/losses/ties and paired sem throughout, per METRICS.md's own ties/effective-n caveat"

noise_floor: "not a single number -- paired sem on the head-to-head dV/capture diff is reported per comparison (e.g. switched-vs-global on pooled corner: mean diff -0.0039, paired sem 0.0011, ~3.5x; on pooled stripe: global wins outright, 0/7/33). Tie rates (the metric's known weakness) range 0-100% across cells and are reported alongside every capture number, not hidden."

depends_on: [goal-mask-axis-convention-row-y-col-x, occ-rasteriser-consistency, push-frame-warp-roundtrip]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  Every fitted model family beats persistence and random by a wide,
  well-powered margin on nearly every non-degenerate (dataset, goal) cell
  (e.g. pooled corner: visual switched +0.919 vs persistence -0.093, vs
  random +0.120). Switched-vs-global for the stage-2 visual/hybrid family is
  goal-dependent: switching wins on corner (pooled 19W/6L/35T of 60,
  ~3.5x paired sem) but LOSES on stripe (pooled 0W/7L/33T of 40, global
  wins outright). The stage-1/2/3 (hybrid14 vs hybrid94) descriptor-width
  question is a dead heat everywhere under slateN (never once disagree on
  any dataset/goal). Stage-3's descriptor-only model, which scores exactly
  0.0 image-space accuracy, scores a real positive pooled slateN (+0.461,
  sem 0.061, 35W/5L/20T of 60 vs persistence) via a point-mass COM readout
  -- image accuracy is not a reliable stand-in for control usefulness for
  this model. Stage-3's switched latent+descriptor operator LOSES to its
  own global/unswitched counterpart on the shortest-push dataset
  (n20_L10mm, where all pushes fall in one weak length bin) -- an
  extrapolation weakness of switching specific to a held-out distribution
  narrower than the training bins.
verdict: supported
downgrades: [selection, incomplete-design, inconsistency, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If EXP-0005's image-space margins (switched beats global, hybrid barely
beats visual) did not reflect real control-ranking differences, slateN
would show no separation anywhere, or a separation uncorrelated with the
image-space ranking. Instead: (1) the visual-family switched-vs-global
separation DOES appear, but only once pooled to 60 slates and only on one
of two goals tested -- exactly the pattern METRICS.md's power caveat
predicts for a real-but-small effect at low per-cell n, not evidence of "no
effect." (2) The hybrid14-vs-hybrid94 tie (EXP-0005's own near-noise 0.0001
image-space gap) reproduces EXACTLY as a tie under slateN on every single
dataset/goal -- a null that agrees with the null it was predicting, which
is itself a discriminating result (a metric that always agreed regardless of
the underlying effect would not be informative). (3) The descriptor-only
reordering (0.0 accuracy, +0.461 slateN) is the sharpest discriminating
case: these two numbers could not disagree more, and the mechanism (a
point-mass COM readout the image metric cannot express at all) is
identified, not just observed.

## What was actually run

`experiments/temp/stage2-slaten/` ran in two passes: Run 1 (one dataset,
one cell, one goal, `corner`) found a directional-but-underpowered
(2 non-tied slates) hint that switching helps; Run 2 widened to the full
stage-2 family (visual/hybrid14/hybrid94/global), 3 datasets, and 3 goals
(`center`/`corner`/`stripe`), with a degeneracy screen applied before
scoring each (dataset, goal) cell. `experiments/temp/stage3-slaten/` scored
the stage-3 latent/descriptor family on the same 3 datasets, `corner` goal
only (time-boxed -- `center`/`stripe` were not attempted for this family).

**`n50_L20mm` was never scored in either pass** -- it has raw data but no
prepared eval-split config/manifest (checked exhaustively against
`configs/dataset/*slates_multistep*`); this is a genuine missing cell, not
a silent drop, and is the reason `slaten-broad` (in flight, see below) exists.

Neither stage-2-slaten nor stage-3-slaten refit stage 2's visual operator
side-by-side with stage 3's latent family in the SAME run, so the
"does the coarse accuracy ranking (visual > latent > descriptors) survive
under slateN" question is answered only PARTIALLY here: stage2-slaten's own
visual-family numbers (this record's own table) and stage3-slaten's
latent/descriptor numbers were never scored in one head-to-head pass. The
stage-3 RESULTS.md flags this explicitly as a budget-driven gap.

**A broader validation run is in flight and NOT reflected in this record**:
`experiments/temp/slaten-broad/` (created same day, per
`experiments/TEMP_LOG.md`) is widening this validation further --
excluding `n20_L10mm` this time, adding goal shapes/functions beyond
corner/center/stripe, and scoring against the overnight holdout split
rather than `slates_multistep` alone. That run belongs to a different
concurrent agent and was not read as a numeric input here (per this task's
instruction to only read, not touch, `experiments/temp/slaten-broad/`).
**This record will need a follow-up or amendment once that run lands** --
in particular, it may sharpen or reverse the goal-dependence finding in (b)
above, since `slaten-broad` explicitly adds more goal shapes than the 3
tested here.

## Numbers

**Pooled slateN (K=N), by goal, stage-2 visual family:**

| goal | n | visual switched | global (unswitched) | persistence | random |
|---|---|---:|---:|---:|---:|
| corner | 60 | +0.919 (sem .024) | +0.788 (sem .039) | -0.093 (sem .071) | +0.120 (sem .052) |
| stripe | 40 | +0.976 (sem .010) | +0.993 (sem .003) | +0.275 (sem .138) | +0.443 (sem .141) |
| center | 20 (L40mm only) | +1.000 | +1.000 | -0.461 | +0.311 |

**Head-to-head switched vs global, pooled:**

| goal | W/L/T | mean dV diff | paired sem |
|---|---|---:|---:|
| corner (n=60) | 19/6/35 | -0.0039 | 0.0011 |
| stripe (n=40) | 0/7/33 | +0.0022 | 0.0013 |

**Stage-3 family, pooled corner (n=60):**

| cell | slateN | sem | W/L/T vs persistence |
|---|---:|---:|---|
| latent+desc94 switched | +0.697 | 0.052 | 51/4/5 |
| latent+desc94 global (unswitched) | +0.811 | 0.035 | 57/2/1 |
| latent-only switched | +0.689 | 0.054 | 50/4/6 |
| descriptor-only (point-mass readout) | +0.461 | 0.061 | 35/5/20 |
| persistence | -0.093 | 0.071 | -- |
| random | +0.120 | 0.052 | 28/14/18 |

Full per-(dataset, goal) tables, K=32 fixed-reference numbers, and the
excluded-degenerate-cell log are in
`experiments/temp/stage2-slaten/results_slaten_stage2family.json` and
`experiments/temp/stage3-slaten/results_slaten_latentfamily.json`.

## What would change the verdict

- **`n50_L20mm`'s eval split**, if built, would let all three
  push-length regimes be compared side by side; currently `n20_L10mm` is
  the only "short push" evidence for the switching-hurts-on-short-pushes
  finding.
- **`slaten-broad`'s wider goal set** (ring/T/quadrant, mass-in-region and
  signed-mass value functions) is the direct, already-in-flight answer to
  whether the corner-vs-stripe goal-dependence is a 2-goal coincidence or a
  real pattern; this record should be amended once it lands.
- **A single joint run** scoring stage-2's visual operator and stage-3's
  latent/descriptor family together, same datasets/goals, would close the
  "does the accuracy ranking survive under slateN, in full" gap flagged
  above.

## Threats

- **`selection`**: `corner` was adopted as the primary goal specifically
  because `center` proved degenerate (>90% zero-effect candidates) in Run 1
  -- a real, disclosed, data-driven exclusion, not silently dropped, but a
  post-hoc choice nonetheless. Which datasets got which goal cells (e.g.
  stage-3 only scoring `corner`) was also chosen under time pressure rather
  than a pre-registered design.
- **`incomplete-design`**: `n50_L20mm` never had an eval split built;
  `ring`/`T`/`quadrant` goal shapes (available in `Baselines/common/
  goals.py`) were never scored for either family; no latent-width sweep
  (inherited from EXP-0005) feeds into the stage-3 numbers here either.
- **`inconsistency`**: switching helps on `corner`, pooled and
  well-powered (19W/6L/35T, ~3.5x sem), but LOSES on `stripe` (0W/7L/33T)
  -- the effect's sign is goal-dependent, not a fixed property of switching.
  This is reported as a real, governing condition (per the skill's guidance
  to name conditions rather than average over them), not hidden by
  averaging the two goals together.
- **`untested-dependency`**: `goal-mask-axis-convention-row-y-col-x` and
  `occ-rasteriser-consistency` are both `unchecked` in `INVARIANTS.md`,
  inherited from EXP-0001..0003 for the same reason -- this record's goal
  masks (`corner`/`center`/`stripe`) and rasterisation both rest on the
  same unverified conventions.
- Considered and dismissed: **provenance mismatch** between the refit
  operators and their EXP-0005 counterparts -- both stage2-slaten and
  stage3-slaten refit via the SAME `fit_cell`/`predict_image` functions
  EXP-0005's own scripts define, with the same recipe/lam, on the same
  cached train tensors; not a new model.

## Amendment (2026-09-13, following EXP-0008) -- superseded, not invalidated

**This record's switching-vs-goal-dependence interpretation is superseded by
EXP-0008; the numbers measured here stand unchanged.** EXP-0008 re-ran the
identical stage-3 `latent+desc94` switched-vs-global comparison with
`n20_L10mm` excluded (a dataset the user has separately ruled contaminated
-- every push in it falls in the weakest push-length bin, forcing the
switched operator into one bad bin-specific operator for every candidate)
and found the reversal reported above (pooled corner: switched +0.697 <
global +0.811) disappears: with L10mm excluded, pooled corner/lyapunov
switched (+0.938) beats global (+0.911), and the overnight holdout (a
corpus with no L10mm-style contamination at all) shows switched beating
global in **all 15** (shape, value-function) cells tested there. This is a
**supersession, not an invalidation**: nothing about `push-frame-warp-
roundtrip`, `occ-rasteriser-consistency`, or `goal-mask-axis-convention-
row-y-col-x` was found broken, and the numbers above were computed
correctly on the data given to this record (`n20_L10mm` included, as this
record's own design specified). The measurement is correct; the CORPUS was
the confound, and EXP-0008 is the follow-up this record itself flagged as
in-flight in "What was actually run," above. Per the `experiment-log`
skill's guidance, do not delete or rewrite the original numbers in
"Numbers" above -- they remain the correct answer to "what did stage-3's
switched vs. global operator score, pooled across all three
`slates_multistep` push-length datasets including `n20_L10mm`."

The goal-shape-dependence claim in (b)/`result` above (`stripe` loses to
global, `corner` sees switching win) is similarly **narrowed, not
overturned**: EXP-0008 found this holds only under the `lyapunov` value
function on `stripe` (0W/7L/33T, reproducing this record's `corner`-only
finding's flavour exactly); under `mass_in_region` and
`signed_mass_in_region`, scored on the SAME `stripe` shape, switching wins
instead (12W/7L/21T and 9W/7L/24T respectively). So "stripe is bad for
switching" is better read as "stripe scored by Lyapunov distance is bad for
switching" -- a real, narrower condition this record's 2-goal, 1-value-
function design could not have distinguished from "stripe is bad for
switching" outright.

See `EXP-0008` for the full widened design (5 goal shapes x 3 value
functions x 2 slates_multistep datasets, L10mm excluded, + a 15-cell
overnight-holdout replication) and `experiments/REGISTER.md`'s revised
`C-005` for the register-level wording change this amendment drives.

## Unrelated findings

- `experiments/temp/stage2-slaten/operators/*.pt` and
  `experiments/temp/stage3-slaten/{latent_operators,desc_operator}.pt` were
  persisted, unlike EXP-0005's own stage-2/3 runs (which only saved
  metrics, not the fitted operators) -- Run 1 of stage2-slaten had to refit
  from scratch for exactly this reason. The visual operator has now been
  promoted out of `temp/` into `weights/MODEL-0001-stage2-visual-switched/`
  as part of this same promotion pass (see task instructions); the
  remaining operators (hybrid14/hybrid94, latent family, descriptor-only)
  are still only in `experiments/temp/`, not yet promoted, since nothing
  outside this record currently compares against them directly.

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- Asymmetric goal masks (T / random_quadrant / ring_O / stripe / letters) in this record were scored BEFORE the goal-axis fix `28271c09` (2026-09-17) and were never rescored; transpose-invariant goals (corner, center, ...) are unaffected (ISS-003). The existing amendment (superseded by EXP-0008) does not cover this.
