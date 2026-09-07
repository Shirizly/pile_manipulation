---
id: EXP-0026_v2
title: >
  C-046's clean "systematic degradation arms are K-invariant, only
  independent-per-candidate noise degrades with selection pressure" split
  was measured under ONE functional (a distance transform). Re-run under a
  sharper, fixed-target functional (ind-corner, r<=1 spectral energy 0.541
  vs dist-corner's 0.629) on the SAME two datasets/caches as EXP-0024_v2:
  the split NARROWS on n20_heap_5mm and partially INVERTS on n20_L20mm
  (displacement becomes the largest K-decliner, hf-noise goes flat). What
  does NOT move: mean-delta's own capture fraction declines with K under
  every functional tested, on both datasets
tier: T1
mode: confirmatory
date: 2026-09-07
hypothesis: C-046
claim: >
  EXP-0026/EXP-0026_v1 established, under the register's one functional
  (dist-corner), that selection pressure amplifies a degradation's control
  cost only when the error is independent across candidates (hf-noise,
  mean-delta), leaving systematic error (displacement, amplitude, blur)
  K-invariant. Because that split was read as a fact about ERROR TYPES, it
  should reproduce under a different cost functional applied to the SAME
  degraded fields, the SAME predictions, the SAME candidate slates. If the
  split instead depends on which functional scores it, C-035/C-039/C-046's
  "systematic is free, independent noise is not" story is a property of the
  metric's own smoothness, not of the errors themselves.
prediction:
  supports: >
    under ind-corner (fixed target, sharper functional), the same arms that
    were K-invariant under dist-corner (displacement, amplitude, blur) stay
    within 3 points from K=4 to K=max on BOTH datasets, and the same arms
    that declined under dist-corner (hf-noise, mean-delta) still decline by
    more than that -- the split is functional-independent.
  refutes: >
    an arm that was K-invariant under dist-corner degrades by more than 3
    points under ind-corner, or an arm that declined under dist-corner goes
    flat under ind-corner, on either dataset -- the split is
    functional-dependent and C-046's mechanism must be restated with the
    functional named.
  discriminating: true
provenance:
  commit: db8f9a28
  dirty: false
  dirty_note: >
    Identical to EXP-0024_v2 -- same commit, same session, same dv caches
    (this record reuses them unmodified, computing no new predictions).
  data_commit: >
    Identical to EXP-0024_v2/EXP-0026/EXP-0026_v1 -- no new data collection.
  script: >
    scripts/probes/exp0026_kcurve_exact.py (unchanged) against
    runs_exp0024_v2/dv_cache_A.pt and runs_expB/n20_L20mm_v2_dv_cache.pt
    (EXP-0024_v2's caches, which already carry the 13 EXP-0025/EXP-0026
    degradation arms under every goal, since `_degradation_arms` runs once
    per cache build and its output is scored under whichever goals
    `--goals` names).
  data: ["runs_exp0024_v2/dv_cache_A.pt", "runs_expB/n20_L20mm_v2_dv_cache.pt
          (both from EXP-0024_v2, reused unmodified)"]
  code_path: >
    Identical to EXP-0024_v2/EXP-0026/EXP-0026_v1 -- registry/PileSweepData
    throughout, EXP-0025's degradation-arm recipes imported unchanged.
  seed: "Identical to EXP-0024_v2 (dataset split seeds), 1234 (degradation-arm noise, EXP-0025's own), 0 (K-sweep)."
  split: "Identical to EXP-0024_v2 -- dataset A all 50 n20_heap_5mm slates; dataset B EXP-0024_v1's 30/20 split, step-0 subset of n20_L20mm."
  runtime: "~1 min CPU (2 kcurve_exact calls x 2 functionals x 2 datasets against already-built caches, no new predictions)."
budget:
  declared: "shared with EXP-0024_v2 (~3h wall-clock, ~250k tokens total for both records)"
  spent: "~15 min of the shared budget -- this record adds no new computation beyond re-running exp0026_kcurve_exact.py with --models against EXP-0024_v2's existing caches"
  outcome: within
design:
  varied:
    functional: ["dist-corner (baseline)", "ind-corner (sharpest non-degenerate fixed-target functional from EXP-0024_v2)"]
    degradation: ["displacement k=1/4", "amplitude a=0.5/1.5", "blur s=1.0/2.0",
                  "hf-noise m=0.5/1.0/2.0", "mean-delta (own K-decline)", "wrong-physics"]
    dataset: ["A: n20_heap_5mm, K=4->31", "B: n20_L20mm step-0, K=4->128"]
  held_fixed:
    predictions: "identical to EXP-0024_v2 -- same fit/checkpoint per dataset, same degraded fields (EXP-0025's recipes, noise seed 1234); only the SCORING functional (dist-corner vs ind-corner) changes"
    K_range: "K=4 (start) to K=max per dataset (31 or 128) -- the same endpoints EXP-0026/EXP-0026_v1 used"
    arm_set: "the 13 arms EXP-0025/EXP-0026 defined, unchanged"
  baselines: [persistence, "linear (undegraded)", mean-delta]
  metric: "slateK_exact (closed form, docs/experiments/METRICS.md; the same quantity slate4 is the K=4 special case of), K=4 vs K=max, per arm per functional per dataset"
noise_floor: >
  Per-arm K=4->K=max deltas are within-model, within-functional comparisons
  (one arm's own slateK_exact at two K values), not paired model
  differences, so no sem is computed for them individually -- exactly
  EXP-0026/EXP-0026_v1's own convention for reporting the degradation
  sweep's deltas. Where a model comparison IS paired (UNet-linear), see
  EXP-0024_v2, which this record does not repeat.
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend,
             swept-region-metric, episode-split, settled-state]
establishes: []
result: >
  PREDICTION REFUTED on both datasets, in two different ways. Dataset A
  (n20_heap_5mm): under dist-corner the split is clean (systematic arms move
  0.011-0.045 from K=4->31; hf-noise moves 0.055-0.113); under ind-corner
  EVERY arm now moves 0.048-0.066 (displacement k=4 moves 0.173, the LARGEST
  decline of any arm at either functional) -- the split narrows to
  near-invisibility and displacement k=4 becomes the standout decliner, not
  hf-noise. Dataset B (n20_L20mm): under dist-corner the split reproduces
  EXP-0026_v1's own published numbers exactly (mean-delta 0.5515->0.2871,
  hf-noise m=1.0 0.8072->0.7236, amplitude/blur move <=2.4 points,
  displacement k=4 actually RISES +0.0644); under ind-corner displacement
  k=4 INVERTS to a large decline (-0.1296) while hf-noise goes flat or
  slightly RISES (m=1.0 +0.0037, m=2.0 +0.0339) -- the split does not just
  narrow here, it partially SWAPS which arms look K-sensitive. What survives
  unchanged on BOTH datasets under BOTH functionals: mean-delta's own
  capture fraction declines with K (A: dist-corner -0.036, ind-corner
  -0.048; B: dist-corner -0.264, ind-corner -0.342) -- always the largest or
  near-largest decline in the table, at every functional, on every dataset.
  VERDICT: refuted -- the arm-by-arm systematic/independent split is
  functional-dependent; the model-level (mean-delta's own K-decline) finding
  is not.
verdict: refuted
downgrades: [imprecision, untested-dependency, inconsistency]
grade: very-low
supersedes: []
invalidated_by: null
---

**Tier note.** Same as every record in this family: T1 rather than T2
because of the `settled-state` dependency, despite a pre-registered
prediction.

## Why this test discriminates

C-046's evidence (EXP-0026, EXP-0026_v1) is entirely arm-by-arm K-dependence
under `dist-corner`: displacement/amplitude/blur stay flat, hf-noise and
mean-delta decline, and the mechanism proposed is the winner's curse --
independent-per-candidate error gets more chances to promote a mediocre
action as the pool grows, while a shared/systematic perturbation moves every
candidate alike and cancels under comparison. That mechanism is stated as a
fact about the ERROR, but every measurement of it ran through one smooth,
low-pass functional. If `dist-corner`'s own smoothness is what makes a
"shared" perturbation cancel (because nearby pixels get nearly the same
weight, so a spatial shift or amplitude scaling barely changes `V`), then a
sharper functional -- which assigns very different weight to pixels a few
cells apart -- should make exactly those "systematic" perturbations start to
bite, since they now move mass across weight boundaries the old functional
was too smooth to register. This design re-scores the IDENTICAL degraded
fields (no new predictions, same noise seed) under `ind-corner` and checks
whether the arm-by-arm story holds.

## What was actually run

No new predictions and no new degraded fields: `EXP-0024_v2`'s two dv caches
already carry all 13 EXP-0025/EXP-0026 degradation arms scored under every
goal in their `--goals` list (because `_degradation_arms` builds the
degraded occupancy fields once per cache, and the goal loop then scores
EVERY entry in `preds` -- base models and arms alike -- under EVERY goal
named). This record simply calls `exp0026_kcurve_exact.py --models
<arm-list>` against `runs_exp0024_v2/dv_cache_A.pt` and
`runs_expB/n20_L20mm_v2_dv_cache.pt` at `--goal corner` and `--goal
ind-corner`, `--ks 4,31` (A) / `--ks 4,128` (B).

## Numbers

**Dataset A (n20_heap_5mm), `slateK_exact` K=4 -> K=max, dist-corner vs ind-corner:**

| arm | dist-corner K=4 | K=31 | Δ | ind-corner K=4 | K=31 | Δ |
|---|---|---|---|---|---|---|
| linear (undegraded) | 0.9767 | 0.9581 | -0.0186 | 0.9739 | 0.9233 | -0.0507 |
| mean-delta | 0.8471 | 0.8107 | **-0.0364** | 0.8591 | 0.8107 | **-0.0484** |
| displacement k=1 | 0.9729 | 0.9482 | -0.0247 | 0.9648 | 0.9031 | -0.0617 |
| displacement k=4 | 0.9258 | 0.8807 | -0.0451 | 0.8901 | 0.7174 | **-0.1727** |
| amplitude a=0.5 | 0.9760 | 0.9593 | -0.0167 | 0.9738 | 0.9241 | -0.0496 |
| amplitude a=1.5 | 0.9723 | 0.9545 | -0.0178 | 0.9715 | 0.9152 | -0.0563 |
| blur s=1.0 | 0.9719 | 0.9605 | -0.0114 | 0.9709 | 0.9190 | -0.0519 |
| blur s=2.0 | 0.9649 | 0.9422 | -0.0227 | 0.9655 | 0.9109 | -0.0547 |
| hf-noise m=0.5 | 0.9457 | 0.8609 | -0.0848 | 0.9715 | 0.9199 | -0.0516 |
| hf-noise m=1.0 | 0.9149 | 0.8022 | -0.1127 | 0.9653 | 0.9056 | -0.0596 |
| hf-noise m=2.0 | 0.7929 | 0.7380 | -0.0550 | 0.9395 | 0.8736 | -0.0659 |
| wrong-physics | -0.0554 | -0.0834 | -0.0279 | -0.0447 | -0.0421 | +0.0026 |

Under dist-corner the split is exactly EXP-0026's shape: systematic arms
(displacement k=1, amplitude, blur) move 0.011-0.025, hf-noise moves
0.055-0.113. **Under ind-corner every arm moves 0.048-0.066** --
displacement k=1 (0.062) and amplitude/blur (0.050-0.056) now decline about
as much as hf-noise (0.052-0.066), and displacement k=4 (0.173) declines
MORE than any hf-noise arm at either functional. The split has not reversed,
but it has become nearly invisible: sharpening the functional makes
"systematic" perturbations behave like independent-noise perturbations for
K-dependence purposes.

**Dataset B (n20_L20mm step-0), `slateK_exact` K=4 -> K=128:**

| arm | dist-corner K=4 | K=128 | Δ | ind-corner K=4 | K=128 | Δ |
|---|---|---|---|---|---|---|
| linear (undegraded) | 0.9533 | 0.9670 | +0.0136 | 0.9431 | 0.9346 | -0.0085 |
| mean-delta | 0.5515 | 0.2871 | **-0.2644** | 0.6319 | 0.2897 | **-0.3422** |
| displacement k=1 | 0.9370 | 0.9220 | -0.0150 | 0.9274 | 0.8953 | -0.0321 |
| displacement k=4 | 0.7945 | 0.8589 | **+0.0644** | 0.7777 | 0.6480 | **-0.1296** |
| amplitude a=0.5 | 0.9532 | 0.9666 | +0.0134 | 0.9429 | 0.9346 | -0.0083 |
| amplitude a=1.5 | 0.9421 | 0.9657 | +0.0235 | 0.9365 | 0.9337 | -0.0028 |
| blur s=1.0 | 0.9373 | 0.9554 | +0.0181 | 0.9382 | 0.9407 | +0.0026 |
| blur s=2.0 | 0.9053 | 0.9117 | +0.0063 | 0.9270 | 0.9055 | -0.0215 |
| hf-noise m=0.5 | 0.8982 | 0.8671 | -0.0311 | 0.9404 | 0.9346 | -0.0058 |
| hf-noise m=1.0 | 0.8072 | 0.7236 | -0.0836 | 0.9325 | 0.9362 | +0.0037 |
| hf-noise m=2.0 | 0.5989 | 0.4853 | -0.1136 | 0.8814 | 0.9153 | +0.0339 |
| wrong-physics | -0.0018 | 0.0966 | +0.0984 | 0.0050 | 0.0610 | +0.0561 |

This reproduces EXP-0026_v1's own published dist-corner numbers exactly
(mean-delta 0.5515->0.2871, hf-noise m=1.0 0.8072->0.7236, displacement k=4
+0.0644 -- all to 4 decimals, confirming the cache is unchanged from
EXP-0024_v1/EXP-0026_v1's own). **Under ind-corner the split does not just
narrow, it partially swaps**: displacement k=4 flips from the register's
best-behaved "systematic, even slightly helped by K" arm (+0.0644) to its
worst K-decliner after mean-delta (-0.1296); hf-noise, the flagship
independent-noise exemplar under dist-corner (down to -0.1136 at m=2.0),
goes FLAT to slightly RISING under ind-corner (+0.0037, +0.0339 at m=1.0/2.0).

**What is functional-independent**: mean-delta's own capture fraction
declines with K on BOTH datasets under BOTH functionals, always among the
largest declines in the table (A: -0.036/-0.048; B: -0.264/-0.342). This is
the one piece of C-046-adjacent evidence (EXP-0026_v1's "the model whose
across-candidate variation is warp/state-driven, i.e. independent per
candidate, degrades with K" mechanism) that this record's sharpening does
not disturb.

## What this means

**C-046's arm-by-arm mechanism, as measured, is functional-dependent.** The
specific claim "displacement/amplitude/blur are K-invariant, hf-noise is
not" was true under `dist-corner` and is not reliably true under `ind-corner`
on either dataset -- on A the split collapses (every arm becomes
K-sensitive), on B it partially inverts (displacement becomes the worst
K-decliner, hf-noise goes flat). The winner's-curse mechanism EXP-0026
proposed -- a shared perturbation "cancels under comparison" because nearby
candidates get nearly the same weight -- is a property of a SMOOTH weight
field specifically: sharpening the field means two states differing by a
spatial shift or amplitude scale of the SAME degraded prediction no longer
get nearly the same score, because the weight now varies enough over the
relevant length scale to distinguish them. That is a mechanistic account of
why the split should be functional-dependent, not just an empirical
observation that it is.

**This qualifies, but does not overturn, EXP-0026_v1's headline addition.**
mean-delta's own K-decline -- the first model-level (not just arm-level)
demonstration of C-046's mechanism -- reproduces under both functionals on
both datasets, and by a WIDER margin under ind-corner on dataset B (-0.342
vs -0.264). Whatever makes mean-delta's own across-candidate variation
behave like independent-per-candidate noise (its single fixed canonical-
frame delta interacting differently with each candidate's own push
geometry) is not specific to a smooth cost functional the way the
displacement/blur/amplitude arms' apparent K-invariance was.

**Consequence for C-046 and C-035/C-039.** The register's stated form of
C-046 ("selection pressure amplifies control cost only for independent
error") should be restated as: *under the register's own distance-transform
functional*, this split holds cleanly; whether an arbitrary perturbation
looks systematic or independent under selection pressure is not a fixed
property of the perturbation, it interacts with how smoothly the scoring
functional treats nearby pixels. C-035/C-039 ("blur/amplitude nearly free,
hf-noise expensive") inherit the same qualification, since their evidence is
also `dist-corner`-only.

## What would change the verdict

- **A denser functional sweep** (distclip-r8/r4/r2, this record's companion
  EXP-0024_v2 already has the caches) would locate WHERE on the sharpness
  axis the split starts to narrow, rather than only bracketing it at two
  points.
- **A mechanistic decomposition**: directly measuring how much of an arm's
  `dV` variance across a slate is "shared" (correlated with the undegraded
  linear operator's own across-candidate variance) vs "residual" would test
  the winner's-curse account directly rather than inferring it from which
  arms happen to decline.
- **Dataset C** (L10mm, L40mm) would show whether the inversion seen on B
  (n20_L20mm) is push-length-specific or general -- not run this session,
  named under `incomplete-design` in EXP-0024_v2.

## Threats

- `imprecision`: identical to EXP-0024_v2 -- one checkpoint/fit per dataset,
  no seed sweep.
- `untested-dependency`: `settled-state`, inherited unchanged.
- `inconsistency`: dataset A and dataset B do not even agree on HOW the
  split breaks (A: collapses, every arm K-sensitive; B: partially inverts,
  displacement becomes worst, hf-noise goes flat) -- reported as the finding
  rather than resolved, since forcing a single story onto two datasets that
  disagree would be worse than stating the disagreement. What both datasets
  DO agree on (mean-delta's own decline, functional-independent) is called
  out separately for exactly this reason.
- Considered and dismissed: `provenance` -- no new predictions or degraded
  fields were computed; this record only re-scores EXP-0024_v2's own caches
  under a different goal, and the dist-corner column reproduces
  EXP-0026/EXP-0026_v1's own published numbers to 4 decimals on both
  datasets.
- Considered and dismissed: `selection` -- both datasets and all 13 arms are
  reported under both functionals, including the direction (B's inversion)
  least convenient for a clean story.

## Unrelated findings

- `wrong-physics`'s own K=4 value is near-zero under `dist-corner` on both
  datasets (as EXP-0026/EXP-0026_v1 found) but becomes modestly POSITIVE
  under `ind-corner` on dataset B (0.0050 -> 0.0610) and stays near-zero on A
  -- too small and inconsistent a signal to classify either way, same call
  EXP-0026_v1 made for this arm under dist-corner.
- This record needed no new GPU or CPU-heavy computation at all -- every
  number comes from re-running `exp0026_kcurve_exact.py`'s existing
  `--models`/`--goal` arguments against caches EXP-0024_v2 already built,
  confirming that script's design goal (cache once, sweep cheaply) extends
  cleanly to a second scoring axis (functional) that its authors did not
  anticipate.
