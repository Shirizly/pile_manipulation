---
id: EXP-0007
title: >
  P1 refuted: deconfounded FSS skill is largest at r=1 and falls monotonically,
  not a rise-to-saturation near the EXP-0003 blur sigma
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: null

claim: >
  On Genesis/data/cube_spectrum/n20 (piled cubes, mask view, sigma 0), the
  linear operator's Fractions-Skill-Score advantage over persistence and over
  mean-delta -- FSS(operator) minus FSS(baseline) at each neighbourhood radius
  r -- rises with r and plateaus (per-step gain <0.02) somewhere in r in
  [3,5] px, with the plateau radius within ~2 px of the sigma~1-1.5 that
  helped in EXP-0003.

prediction:
  supports: "the FSS-advantage-over-persistence curve rises with r and its per-step gain falls below 0.02 first at some r in [3,5]"
  refutes: "the curve is flat, monotonically FALLS with r, or its plateau (if any) sits outside r in [1,8]"
  discriminating: true

provenance:
  commit: e9b83f99
  dirty: true                     # BACKFILLED 2026-09-05: `scripts/probes/fss_scale_decomp.py`
                                  # did not exist at e9b83f99; it was written in the same
                                  # session and first committed at aac084e3. So this sha
                                  # bounds the run from below only -- the analysis code
                                  # that actually ran is the aac084e3 version of that file.
  script_first_committed: aac084e3
  script: scripts/probes/fss_scale_decomp.py
  data: ["Genesis/data/cube_spectrum/n20/*_data.pt"]
  code_path: particles_to_occupancy (via occupancy_foresight.load_transition_fields, view="mask")
  seed: 0
  split: "episode-level, 4/16 files held out, seed 0 (same rule as scripts/probes/regimes.py)"
  runtime: "~15s CPU per sigma arm (sigma=0 and sigma=1 both run)"

budget:
  declared: "35 min, 120k tokens"
  spent: "~30 min, ~85k tokens"
  outcome: within

design:
  varied: {r: [1, 2, 3, 5, 8, 12, 16], model: [persistence, identity_warp, mean_delta, linear_operator], sigma: [0.0, 1.0]}
  held_fixed: {view: mask, cube_size: 0.005, min_grains: 1.0, grid: 64, canon_res: 64, crop: 1.0, ridge: 1.0, estimator: "ridge toward identity", region: "swept_region_mask, half_width=0.5*plate+2px, pad=0.5*plate (plate=0.04/0.128*64=20px)", split_rule: identical across models}
  baselines: [persistence, identity_warp (warp round-trip only, A=I), mean_delta]
  metric: >
    M1: FSS(r) = 1 - [sum_region(box_r(pred)-box_r(truth))^2] /
    [sum_region box_r(pred)^2 + sum_region box_r(truth)^2], pooled over all
    region-weighted pixels of the held-out set (not averaged per-sample), box_r
    = 2r+1 uniform filter. Region = the same swept-region mask used elsewhere
    in the repo, so untouched background (persistence-exact for every model by
    the warp-blend invariant) cannot inflate the score.
    M5: Laplacian pyramid (4 levels, repo's own _gaussian_blur2d + 2x pooling)
    of the error field (pred-truth) and the signal field (truth-persistence);
    per-band ratio = sqrt(sum_region err_band^2 / sum_region sig_band^2),
    region mask average-pooled down the same pyramid.

noise_floor: >
  Not measured for this design (single seed-0 split, no fold sweep -- same gap
  EXP-0003 left open). The reported effects (FSS-advantage 0.04-0.55; M5
  ratios spanning 0.2-1.08) are 1-2 orders of magnitude larger than the
  ~0.001-0.004 fold sd reported elsewhere in this repo
  (`linear_foresight_report.md` §2.2b), so almost certainly not noise, but this
  is a plausibility argument, not a measured floor.

depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, footprint-splat]
establishes: []

result: >
  FSS(operator)-FSS(persistence) is 0.55 at r=1 and falls monotonically to
  0.04 at r=16 (sigma=0); same shape at sigma=1 (0.54 -> 0.04). No rise, no
  plateau near r=3-5. M5 error/change ratio for the operator falls from 0.69
  (finest band, ~1px) to 0.22 (~8px band) then rises to 0.37 at the coarsest
  residual band -- a genuine scale structure, but not the one P1 predicted and
  not at the predicted radius.

verdict: refuted
downgrades: [imprecision, indirectness]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

P1 makes a shape claim, not just a sign claim: the operator's advantage should
*build* with neighbourhood size and level off close to where EXP-0003's blur
helped. If instead the advantage is already near its ceiling at the smallest r
and shrinks as r grows -- because persistence's own score approaches 1 for
free as everything gets smoothed together -- that is the opposite shape, and
distinguishes "the operator has skill at a believable coarse scale" from "the
operator has skill everywhere, most visibly at the finest scale, and the
apparent large-r improvement is the confound the task specifically warned
about."

## What was actually run

Built `scripts/probes/fss_scale_decomp.py`: box-filter FSS at r in
{1,2,3,5,8,12,16} and a 4-level Laplacian pyramid (+ residual), both computed
against the swept-region mask so the ~95%-untouched background (identical
across models by construction) cannot inflate any curve. Four models:
persistence (`o0`), identity-warp (`A=I` through the same canonicalise / warp /
blend pipeline as every other model -- isolates warp-round-trip cost from
operator skill), mean-delta (canonical-frame mean delta, blended back to world
frame the same way `predict_world` does), and the ridge-toward-identity linear
operator (`res=64, crop=1.0, ridge=1.0`), fit on 3632 train transitions and
scored on 1208 held-out (4/16 episodes). Ran both a sigma=0 (native sharpness,
the point of the exercise) and a sigma=1 arm (the task allowed this if budget
permitted; it did, cost was 15s).

Deviation from the literal FSS(r) formula given in the task prompt: I pooled
numerator and denominator sums over all region-weighted pixels of the test set
rather than computing a per-sample ratio and averaging, because the region
mask is small (~675 px/transition) and several models' `frac_pred` is exactly
0 for many samples at r=1 (a flat prediction), which makes a per-sample ratio
denominator-unstable. Pooling is the standard FSS convention in the
meteorology literature this metric comes from (Roberts & Lean 2008) and gives
the same large-sample behaviour; noted here since it is a literal deviation
from the prompt's formula.

## Numbers

FSS(r), sigma=0.0 (pooled over 1208 test transitions, region-weighted):

| r | persistence | identity_warp | mean_delta | linear_operator |
|---|---|---|---|---|
| 1  | 0.3419 | 0.3526 | 0.7081 | 0.8877 |
| 2  | 0.4171 | 0.4233 | 0.7779 | 0.9384 |
| 3  | 0.4892 | 0.4936 | 0.8230 | 0.9601 |
| 5  | 0.6359 | 0.6382 | 0.8835 | 0.9782 |
| 8  | 0.8048 | 0.8058 | 0.9243 | 0.9871 |
| 12 | 0.9078 | 0.9083 | 0.9356 | 0.9902 |
| 16 | 0.9496 | 0.9499 | 0.9414 | 0.9908 |

Deconfounded skill, sigma=0.0 (FSS difference vs. the zero-skill references):

| r | operator - persistence | operator - mean_delta | identity_warp - persistence |
|---|---|---|---|
| 1  | +0.5458 | +0.1795 | +0.0107 |
| 2  | +0.5214 | +0.1606 | +0.0062 |
| 3  | +0.4709 | +0.1371 | +0.0044 |
| 5  | +0.3423 | +0.0947 | +0.0022 |
| 8  | +0.1823 | +0.0629 | +0.0009 |
| 12 | +0.0825 | +0.0546 | +0.0006 |
| 16 | +0.0412 | +0.0493 | +0.0004 |

sigma=1.0 arm (same shape, slightly smaller gaps): operator-persistence runs
0.538 (r=1) -> 0.043 (r=16); operator-meandelta runs 0.161 (r=1) -> 0.039
(r=16). Every raw FSS value at sigma=1 is 1-6 points higher than at sigma=0
(e.g. operator at r=1: 0.937 vs 0.888) -- consistent with EXP-0003's "blur is
a genuinely easier target", not evidence for or against P1 itself.

M5 scale-decomposed error / change ratio (1.0 = no better than persistence, 0
= perfect), sigma=0.0, band sizes roughly [~1px, ~2px, ~4px, ~8px, coarse
residual]:

| model | L0 | L1 | L2 | L3 | residual |
|---|---|---|---|---|---|
| persistence | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| identity_warp | 0.8127 | 0.9558 | 0.9887 | 0.9953 | 0.9986 |
| mean_delta | 0.7633 | 0.7224 | 0.5980 | 0.5716 | 1.0802 |
| linear_operator | 0.6864 | 0.4943 | 0.2774 | 0.2208 | 0.3742 |

sigma=1.0: linear_operator 0.5876 / 0.4154 / 0.2526 / 0.2036 / 0.2780 -- same
V-shape (improves to L3, worsens at the residual), less pronounced reversal.

## What would change the verdict

Nothing cheap reverses the *shape* finding -- it reproduces at two blur levels
with the same qualitative curve both times, and the FSS/M5 disagreement about
*which* scale is "believable" (FSS-relative says none; M5 says somewhere
around the L3 band, ~8px, ~4 cube widths) is itself informative rather than a
gap to close. What would matter:
- A per-episode fold sweep (LORO, 16 folds, ~5 min at this runtime) to turn
  the imprecision downgrade into a real noise floor -- cheap, was skipped only
  because the single split already gives an unambiguous non-saturating shape.
- Running M2 (EMD) or M3 (SAL) from the same ideas-log table on the same held-
  out set, to see whether a metric that is not neighbourhood-bounded (FSS is
  capped at 1) shows the same or a different scale structure -- FSS's bounded
  range may itself be why its "skill" measure compresses toward the ceiling at
  large r while M5's unbounded ratio does not. That comparison is the natural
  next cell and is outside this record's scope.

## Threats

- `imprecision`: single seed-0 episode split (4/16 held out), as in EXP-0003
  and EXP-0002 pre-LORO. No noise floor computed (see `noise_floor`).
- `untested-dependency`: `swept-region-metric` and `episode-split` are
  `unchecked` in INVARIANTS.md; `footprint-splat` and `canonical-warp` and
  `warp-blend` are `holds`, carried for completeness since the pipeline uses
  all three.
- Considered and dismissed: `provenance` -- everything here goes through one
  code path (`particles_to_occupancy` via `load_transition_fields`), never
  crossing into the `PileSweepData` raster that `grid-convention` /
  `rasteriser-identity` mark broken, so that domain does not apply.
- Considered and dismissed: `indirectness` for THIS record specifically -- P1
  is a claim about the shape of a metric curve, not about control utility, and
  the curve was measured directly. Indirectness would apply to any record that
  tried to use these numbers to say something about MPC ranking (C-030's
  larger claim), which this record does not attempt.
- The FSS metric is bounded in [0,1] and both the operator and persistence
  approach the ceiling as r grows; this compresses the *difference* between
  them at large r almost by construction, independent of any real change in
  skill. This is very likely *why* deconfounded FSS skill declines with r
  rather than evidence that the operator's advantage is actually concentrated
  at fine scales -- M5's unbounded ratio, which does show a genuine (if
  differently-shaped and non-monotonic) scale structure, is the more trustworthy
  read of "where is the skill" from this record.

## Grade note, 2026-09-05

`untested-dependency` dropped: this record's `depends_on` tags all
hold as of the invariant tests added today (`tests/test_metric_invariants.py`,
`tests/test_grid_convention.py`). Grade very-low -> low. The evidence did not
change; what changed is that the assumptions it rests on are now checked.

## Unrelated findings

- `scripts/probes/regimes.py` imports `from sand_foresight import
  load_sand_arrays, BOUNDS`, but no `sand_foresight.py` exists in the repo
  (only `occupancy_foresight.py`, which defines `load_transition_fields` with
  a matching but differently-named signature). `python scripts/probes/regimes.py`
  fails at import with `ModuleNotFoundError` as of this commit -- the script
  referenced in the ideas-log §7 entry points and in this task's own prompt as
  "available" currently does not run. Not fixed here.
- `mean_delta`'s M5 residual-band ratio (0.9135-1.0802, both sigma arms) is
  *worse* than persistence's by definition (>1.0) at the coarsest scale --
  the zero-parameter canonical mean-delta baseline actively hurts at the
  lowest spatial frequency even though it clearly helps at every finer band.
  Logged, not investigated.



## Reviewer amendment, 2026-09-05

Written by the task-giver after re-running this record's own script and adding
one number it did not compute. **The measurements stand; the reading of the FSS
half does not.** Recorded here rather than by editing the analysis above, so the
original reasoning stays visible.

### The FSS difference has a forced decline

`FSS(model) − FSS(persistence)` was chosen to deconfound the fact that raw FSS
rises toward 1 for every model at large r. But **both** terms converge to 1, so
their difference is forced to 0 at large r regardless of skill. A monotone
decline is therefore not evidence about skill-vs-scale — it is the ceiling of a
bounded score, the exact failure mode this record's own feedback identified in
the abstract and then did not apply to its own instrument.

### The standard criterion says the opposite

Roberts & Lean's usable-skill threshold is `FSS_useful = 0.5 + f0/2`, where f0
is the base rate in the scored area. Measured on this data: **f0 = 0.0978, so
FSS_useful = 0.549.** Against that:

| model | FSS(r=1) | believable scale (first r with FSS ≥ 0.549) |
|---|---|---|
| linear operator | **0.888** | **≤ 1** (the smallest radius tested) |
| mean-delta | 0.708 | ≤ 1 |
| persistence | 0.342 | ≈ 4 px |
| identity (warp only) | 0.353 | ≈ 4 px |

**The operator already has usable skill at the finest scale tested.** P1
predicted skill would *rise* to a plateau at r ≈ 3–5, which presupposes it has
little skill below that. It has plenty. So P1 is refuted — but for the opposite
reason to the one recorded above, and the distinction matters for what to do
next.

### What this does to C-030

The strong form of C-030 — "pixel error is mostly measuring detail no model can
predict" — is **weakened**, not supported: the operator predicts fine scales
better than persistence does, by a wide margin.

The M5 half, which this record reported as "a genuine but different structure",
is the part that still supports the programme, and should be read as evidence
rather than as a curiosity. Operator error as a fraction of the signal, by band:

| band | ~1 px | ~2 px | ~4 px | ~8 px | coarse residual |
|---|---|---|---|---|---|
| operator | 0.686 | 0.494 | 0.277 | **0.221** | 0.374 |

The operator captures ~31% of the finest band and ~78% of the 8-px band. Fine
detail is **harder, not impossible** — a 3x difference in relative error across
scales. That is real scale structure and it is what C-030 needs; it is just a
weaker statement than "the fine band is noise".

The coarsest residual reversing to 0.374 is worth noting separately: that band
is essentially total amplitude, so the operator is getting *how much* material
moved somewhat wrong even while getting *where* right. That is an amplitude
error, and SAL (M3) is the metric that would separate it.

### Consequences

- `indirectness` added to the downgrades and the grade drops to `very-low`: the
  FSS half measured a quantity whose scale-dependence is partly an artifact of
  the score's ceiling, not of the models.
- P1 is **rewritten** in `docs/ideas_log_signal_vs_detail.md` — as originally
  worded it did not name the exact quantity ("FSS skill" could mean raw FSS,
  FSS against a baseline, or FSS against FSS_useful, and the three disagree).
  That is a defect in the prediction, not in this experiment.
- The decisive experiment for C-030 (§4 of the ideas log: which metric
  rank-correlates with realised control performance) is untouched by all of
  this and remains the thing to run.
