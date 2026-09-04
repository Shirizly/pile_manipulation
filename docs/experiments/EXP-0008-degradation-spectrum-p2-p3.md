---
# ---- identity -------------------------------------------------------------
id: EXP-0008
title: >
  P2 refuted (no crossing: hf-noise hurts control utility more than
  displacement, agreeing with rms); P3 refuted for FSS(r=1) specifically --
  rms rank-correlates with control utility better than FSS(r=1) does
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  Under synthetic degradation of the fitted operator's held-out prediction on
  Genesis/data/cube_spectrum/n20 (piled cubes, mask view, sigma 0), rms
  penalises high-frequency noise more than small pixel displacement while
  Lyapunov dV rank-correlation (control utility) penalises displacement more
  than noise -- the two orderings cross (P2); and across the full degradation
  spectrum, FSS(r=1) rank-correlates with control utility (both the dV
  Spearman correlation and the slate-of-4 selection regret) more strongly
  than pixel rms does (P3, partial -- FSS only, no EMD/SAL).

prediction:
  supports: "the displacement family (k=1,2,4) shows LARGER control-utility drop (Spearman dV correlation, and slate4 regret) than the hf-noise family (m=0.5,1,2) at matched or lower rms cost, reversing rms's own ordering; AND |spearman(FSS@1-across-spectrum, control-metric-across-spectrum)| > |spearman(rms-across-spectrum, control-metric-across-spectrum)| on both goals tested"
  refutes: "hf-noise damages control utility at least as much as displacement does at matched rms cost (no reversal); OR |spearman(rms, control)| >= |spearman(FSS@1, control)|"
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: e9b83f99
  script: scripts/probes/degradation_spectrum.py
  data: ["Genesis/data/cube_spectrum/n20/*_data.pt"]
  code_path: particles_to_occupancy (via occupancy_foresight.load_transition_fields, view="mask")
  seed: 0
  split: "episode-level, 4/16 files held out, seed 0 (identical rule to EXP-0007 and scripts/probes/regimes.py)"
  runtime: "~15s CPU per goal (center, corner both run; ~30s total)"

budget:
  declared: "40 min, 130k tokens"
  spent: "~45 min, ~100k tokens"
  outcome: exceeded

design:
  varied:
    degradation: [displacement k=1/2/4, amplitude alpha=0.5/0.75/1.25/1.5,
                  hf-noise m=0.5/1.0/2.0, blur sigma=1/2, wrong-physics]
    goal: [center, corner]
  held_fixed:
    view: mask
    cube_size: 0.005
    min_grains: 1.0
    grid: 64
    canon_res: 64
    crop: 1.0
    ridge: 1.0
    estimator: "ridge toward identity (same fit as EXP-0007)"
    region: swept_region_mask, half_width=0.5*plate+2px, pad=0.5*plate
    split_rule: identical to EXP-0007
  baselines: [persistence (delta=0, cannot rank), oracle (perfect, upper anchor)]
  metric: >
    Accuracy: (a) pixel rms, pooled region-weighted sqrt(mean (pred-truth)^2)
    over the swept-region mask; (b) FSS(r=1), reused verbatim from
    scripts/probes/fss_scale_decomp.py's fss_pooled, read against Roberts &
    Lean's FSS_useful = 0.5 + f0/2 (measured f0=0.0985, FSS_useful=0.5493) per
    the EXP-0007 reviewer amendment -- not as a difference of two bounded
    scores. Control utility: control_utility_test.py's lyapunov_weights
    ("center", "corner" -- both simple convex regions; no letter shapes, no
    literal "point" goal since none exists in the code, corner substitutes as
    the second convex goal), lyapunov, and rank_metrics reused unmodified;
    dv_true is computed from the REAL recorded post-push occupancy (o1te),
    i.e. is the realised dV of an actually-executed push, satisfying the
    ideas-log sec.6 circularity guard for this arm. Reported control-utility
    scalars: Spearman(dv_pred, dv_true) and the slate-of-4 selection regret
    (rank_metrics' own K=4 column), both from control_utility_test.py.

noise_floor: >
  Not measured (single seed-0 split, no fold sweep -- same gap as EXP-0007).
  Effect sizes here are large relative to fold sd reported elsewhere in this
  repo (control-utility Spearman drops of 0.35-0.9 vs a documented ~0.001-0.004
  fold sd for scalar metrics; `linear_foresight_report.md` Sec 2.2b) --
  plausibility argument, not a measured floor.

depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, footprint-splat]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  P2 refuted: no crossing. hf-noise m=0.5 (rms 0.241, close to or below
  displacement k=1's rms 0.281) collapses control utility (Spearman dV corr
  0.474->0.089 center-goal, 0.971->0.353 corner-goal) far more than
  displacement k=4 (rms 0.353, the LARGEST rms cost in the displacement
  family) does (0.474->0.339 center, 0.971->0.914 corner). rms and control
  utility AGREE that hf-noise is the more damaging error type; they do not
  cross. P3 (FSS arm) refuted: |spearman(rms-spectrum, control-spectrum)| =
  0.68-0.78 across both goals and both control operationalisations (dV
  Spearman, slate4 regret) exceeds |spearman(FSS@1-spectrum, same control
  metric)| = 0.51-0.59 in every one of the 4 combinations tested. rms
  rank-correlates with control utility BETTER than FSS(r=1) does on this
  spectrum, the opposite of what P3 predicted.

verdict: refuted
downgrades: [imprecision, untested-dependency, incomplete-design]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If the signal-vs-detail hypothesis (C-030) is right, rms and control utility
should disagree about which error type is worse: rms should overweight
zero-mean high-frequency noise (a texture error, invisible to a distance-
weighted mass functional) and underweight small displacement (which changes
per-pixel location but not much of where the aggregate mass sits). If instead
both metrics agree on the ordering, or rms tracks control utility as well as
or better than a signal-sensitive metric, the hypothesis fails on its own
chosen ground -- this experiment measures exactly that ordering, on both a
per-degradation-type basis (P2) and a whole-spectrum rank correlation (P3).

## What was actually run

Built `scripts/probes/degradation_spectrum.py`, reusing EXP-0007's exact fit
(ridge-toward-identity linear operator, res=64/crop=1/ridge=1, same episode
split) to get one held-out prediction per test transition (1208 of 4840 kept
pushes, 4/16 episodes). Took `delta = linear_op - persistence` and produced
14 degraded fields plus 2 anchors (persistence, oracle):

- **displacement**: `torch.roll(delta, shifts=k, dims=-1)`, k in {1,2,4}px.
- **amplitude**: `alpha * delta`, alpha in {0.5, 0.75, 1.25, 1.5}.
- **hf-noise**: zero-mean Gaussian noise run through the same 4-level
  Laplacian pyramid used by `fss_scale_decomp.py`, keeping only the finest
  band, rescaled to {0.5, 1.0, 2.0}x the std of the TRUE finest-band signal
  (`o1_true - persistence`, same pyramid) over the scored region -- so
  magnitudes are anchored to the actual signal's own fine-scale energy,
  not an arbitrary pixel unit.
- **blur**: the repo's own `_gaussian_blur2d(delta, sigma)`, sigma in {1, 2}.
- **wrong-physics**: `torch.roll(delta, shifts=1, dims=0)` -- substitutes the
  adjacent test transition's predicted delta for this one's.

Each degraded delta was added back to persistence and clamped to [0,1] before
scoring (reconstructing a full occupancy field, since both rms/FSS and the
Lyapunov functional need a field, not a delta). Ran the full spectrum on two
convex goals (`center`, `corner`) for a cheap multiverse check per the skill's
discipline; both agree on every qualitative conclusion below, so only
`center`'s numbers are discussed in prose (both are in the table).

**Deviation from plan**: EMD/sliced-Wasserstein and the M5 scale-decomposed
band error (both "if budget allows" in the task) were not run -- the FSS arm
alone already answered P3 clearly and unexpectedly (see below), and writing
this up honestly took longer than planned. Took the `incomplete-design`
downgrade for this rather than silently dropping it. FSS was tested at r=1
only (not swept over r=1..16 as in EXP-0007) -- a coarser radius might behave
differently and is the natural next cell (see "What would change the
verdict"). Only one seed / one split was run (no LORO fold sweep).

## Numbers

FSS(r=1), rms, and control-utility columns, goal=center (1208 test
transitions, pooled region-weighted; `dv_true` mean +0.00509, sd 0.02129, 7%
of pushes helpful for this goal):

| model | rms | FSS@1 | >=useful (0.549) | pearson(dV) | spearman(dV) | sign% | slate4 |
|---|---|---|---|---|---|---|---|
| oracle | 0.0000 | 1.0000 | yes | 0.999 | 0.999 | 100% | 0.795 |
| persistence | 0.3965 | 0.3419 | no | -- | -- | -- | 0.000 |
| operator (undegraded) | 0.2182 | 0.8877 | yes | 0.806 | 0.474 | 83% | 0.586 |
| displacement k=1 | 0.2813 | 0.8251 | yes | 0.780 | 0.443 | 82% | 0.551 |
| displacement k=2 | 0.3189 | 0.7263 | yes | 0.737 | 0.408 | 82% | 0.512 |
| displacement k=4 | 0.3531 | 0.5799 | yes | 0.633 | 0.339 | 81% | 0.445 |
| amplitude a=0.5 | 0.2710 | 0.7088 | yes | 0.812 | 0.472 | 83% | 0.585 |
| amplitude a=0.75 | 0.2308 | 0.8381 | yes | 0.809 | 0.473 | 83% | 0.586 |
| amplitude a=1.25 | 0.2266 | 0.8857 | yes | 0.801 | 0.470 | 83% | 0.577 |
| amplitude a=1.5 | 0.2401 | 0.8709 | yes | 0.795 | 0.465 | 83% | 0.572 |
| hf-noise m=0.5 | 0.2412 | 0.8662 | yes | 0.030 | 0.089 | 80% | -0.043 |
| hf-noise m=1.0 | 0.2958 | 0.8065 | yes | -0.075 | 0.050 | 80% | -0.120 |
| hf-noise m=2.0 | 0.4190 | 0.6475 | yes | -0.133 | 0.012 | 80% | -0.192 |
| blur s=1.0 | 0.2524 | 0.8397 | yes | 0.793 | 0.463 | 83% | 0.557 |
| blur s=2.0 | 0.2892 | 0.7419 | yes | 0.777 | 0.473 | 82% | 0.561 |
| wrong-physics | 0.3978 | 0.3668 | no | 0.010 | -0.015 | 77% | -0.185 |

Same table, goal=corner (`dv_true` mean +0.01710, sd 0.06686, 42% helpful):
qualitatively identical shape -- displacement k=1..4 takes spearman(dV) from
0.971 down to only 0.914 (a 6% relative drop); hf-noise m=0.5..2.0 takes it
from 0.971 down to 0.152 (an 84% relative drop), at rms costs (0.241-0.419)
that bracket the displacement family's (0.281-0.353) rather than exceeding
them.

Whole-spectrum rank correlations (n=14 degraded models, anchors excluded),
both goals and both control-utility operationalisations:

| control metric | goal | spearman(rms, control) | spearman(FSS@1, control) |
|---|---|---|---|
| dV Spearman corr | center | -0.680 | +0.508 |
| dV Spearman corr | corner | -0.778 | +0.590 |
| slate4 regret | center | -0.765 | +0.557 |
| slate4 regret | corner | -0.761 | +0.569 |

(A "good" accuracy metric should be strongly *negatively* correlated with rms
and strongly *positively* correlated with FSS if lower rms / higher FSS both
mean higher control utility. rms wins all four comparisons.)

## What would change the verdict

- **FSS swept over r, not just r=1.** EXP-0007 found the operator's advantage
  over persistence is largest at r=1 and shrinks with r; this record tested
  only r=1, the radius where FSS is bounded tightest against the FSS_useful
  ceiling for models with fine-scale skill (most of this spectrum scores
  0.58-0.89, all near or above the 0.549 threshold). It is plausible a
  smaller-r or a *finer-than-pixel* variant (no box averaging at all) would
  track hf-noise damage better, since a 3x3 box filter (r=1) already averages
  away much of a zero-mean noise field's rms footprint -- a mechanism, not
  yet a measurement. Cost: ~10 min, reusing `fss_pooled` at the existing
  radius list.
- **EMD / SAL**, per the original plan, would show whether a metric that is
  not neighbourhood-bounded behaves differently. Not run here (budget). Cost:
  moderate (`compare_model_emd.py` exists but needs wiring to this harness).
- **A fold sweep (LORO, 16 folds)** to convert the imprecision downgrade into
  a real noise floor. Cost: ~10x this run's runtime (~2-3 min).
- **Why hf-noise hurts control utility so much despite being zero-mean**: the
  degraded field is clamped to [0,1] after adding noise, which is not a
  linear operation -- clamping can turn zero-mean noise into a net biased
  mass change. Not measured here; worth a one-line check (mean of the applied
  noise before vs. after clamp) before trusting the mechanism story.

## Threats

- `imprecision`: single seed-0 episode split (4/16 held out), same as
  EXP-0007. No fold sweep, no noise floor measured directly.
- `untested-dependency`: `swept-region-metric` and `episode-split` are
  `unchecked` in INVARIANTS.md; `canonical-warp`, `warp-blend`,
  `footprint-splat` are `holds`.
- `incomplete-design`: EMD/SAL not run; FSS tested at one radius, not the
  full curve; only two convex goals (center, corner), not the literal
  "point" goal named in the task (does not exist in `control_utility_test.py`
  -- corner substitutes as the second convex region).
- Considered and dismissed: `provenance` -- single code path throughout
  (`particles_to_occupancy` via `load_transition_fields`), matching EXP-0007's
  reasoning exactly; never touches the broken `PileSweepData` raster.
- Considered and dismissed: `indirectness` for the control-utility side --
  `dv_true` is computed from the actually-recorded post-push occupancy, i.e.
  is the realised dV of an executed push, per the ideas-log sec.6 circularity
  guard; this is not a proxy stacked on a proxy the way one-step pixel error
  standing in for MPC performance was.
- **Circularity guard, explicitly**: rms is defined on raw pixel values; FSS
  is defined on box-filtered fractional coverage within a neighbourhood;
  control utility is defined on a distance-transform-weighted mass ratio
  (Lyapunov V). None of the three share a definition with each other, so a
  correlation between any pair is not arithmetic. EMD/Wasserstein -- the arm
  the ideas log flags as most exposed to circularity -- was not run at all,
  so that risk did not need to be managed here; it should be flagged again if
  EMD is added later, since EMD's "mass moved" and V's "mass-weighted
  distance" are close enough in spirit to warrant real scrutiny.

## Unrelated findings

- `scripts/probes/regimes.py` still fails at import (`ModuleNotFoundError:
  sand_foresight`) -- same defect logged in EXP-0007, not touched here.
- FSS(r=1) is nearly uninformative about hf-noise damage specifically: it
  falls only from 0.888 (undegraded) to 0.648 (noise m=2.0, the harshest
  level tested) -- an 8-point range -- while control utility over the same
  span falls to near zero or negative. A 3x3 box filter suppresses much of a
  zero-mean noise field's energy by construction (averaging ~9 i.i.d. samples
  reduces its std by ~3x), which may be *why* FSS underperforms rms here --
  the metric built to fix the double-penalty problem may, at this radius,
  also fix away the exact error type this experiment cares about. Logged as
  a candidate mechanism, not verified.
- `wrong-physics` (swap in the adjacent transition's delta) scores *worse*
  than persistence on both rms (0.398 vs 0.397) and FSS (0.367 vs 0.342, just
  barely above) and has negative or near-zero control utility (slate4 -0.185
  center, sign% only 55-77%) -- a plausible-looking wrong answer is worse
  than doing nothing on every metric tried, which is reassuring (no metric
  is fooled by it) but was not otherwise analysed.

## Reviewer amendment, 2026-09-05

Written by the task-giver from this record's own table. **The measurements
stand and the two stated verdicts stand as written; the summary "rms and
control utility agree hf-noise is worse" does not follow from the numbers, and
the most important result in the table went unreported.**

### The crossing is there — read the rows at matched rms

The table compares degradation families at matched *index* (k=1 vs m=0.5),
which is arbitrary — the levels are not calibrated to each other. Compare at
matched **rms**, which is the question P2 actually asks:

| pair | rms | dV Spearman | slate4 |
|---|---|---|---|
| hf-noise m=0.5 | **0.2412** | 0.089 | **−0.043** |
| displacement k=1 | **0.2813** | 0.443 | **+0.551** |
| hf-noise m=1.0 | **0.2958** | 0.050 | **−0.120** |
| displacement k=2 | **0.3189** | 0.408 | **+0.512** |

In both pairs the noise arm has the **lower** rms and the **catastrophically
worse** control utility — negative slate4, i.e. worse than choosing at random.
So rms orders these two error types the *opposite* way to control utility.
That is a crossing, and it is the finding.

### P2 is still refuted, because its direction was backwards

P2 predicted rms would over-penalise noise and under-penalise displacement.
The truth is the reverse: **rms under-penalises noise and over-penalises
everything else.** Refuted as written — but the underlying claim that rms is
the wrong instrument survives, and is now C-035.

### The unreported result: amplitude and blur are free

| degradation | rms | dV Spearman | slate4 |
|---|---|---|---|
| operator (undegraded) | 0.2182 | 0.474 | 0.586 |
| amplitude a=0.5 (halve the delta!) | 0.2710 | **0.472** | **0.585** |
| blur σ=2.0 | 0.2892 | **0.473** | **0.561** |

Halving the predicted displacement, and blurring it heavily, cost rms 0.05–0.07
and cost control utility **nothing**. This is the strongest single piece of
evidence in the programme so far, it is directly the "blur is free for control"
intuition the ideas log opened with, and it is not mentioned in the analysis
above.

### Why the whole-spectrum correlation missed all of this

The pooled Spearman over 14 models is the wrong statistic for a question about
error *types*. Three of the four families (amplitude, blur, displacement)
behave monotonically and drag the pooled correlation toward "rms works"; the
one family that dissociates is averaged away. **A pooled rank correlation
cannot detect a type-specific dissociation, by construction.** The rms-beats-FSS
conclusion is correct as a statement about global ordering and says nothing
about the claim under test.

### The mechanism, and why it matters for MPC

Ranking is destroyed by **variance**, not by **bias**. Displacement, amplitude
and blur are systematic — they perturb every candidate the same way, so they
cancel when candidates are *compared*. High-frequency noise is drawn
independently per candidate, so it does not cancel; it swamps the small
between-candidate differences that `dV` ranking depends on.

Consequence, and it is actionable: **for MPC, prefer a low-variance predictor
even at the cost of bias.** That is an argument for heavy regularisation,
shrinkage toward identity, and smoothing — and it retro-explains why the
ridge-toward-identity operator and σ≈1 blur have done well throughout.

### Caveat that limits all of the above

Candidate slates are drawn from *different states* (`linear_foresight_report.md`
§2.4), so "independent noise per candidate" is guaranteed here in a way it might
not be in a real MPC loop, where one state's candidates share a prediction
context. A same-state slate could reduce the noise penalty substantially. That
collection (~50 states × 16 actions) is now the highest-value item in the
programme.

### Process failures worth recording

- **Claim-id collision.** This record allocated `C-033`, which was already in
  use by EXP-0007, and overwrote an existing `C-030` row. The validator did not
  catch it. Renumbered to C-034 on review; a duplicate-id check has been added.
- The `budget.outcome: exceeded` disclosure was correct and is exactly what the
  budget rule is for.
