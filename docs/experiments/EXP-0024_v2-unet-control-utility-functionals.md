---
id: EXP-0024_v2
title: >
  CORRECTED 2026-09-07 (coordinator review). The first pass of this record
  wrongly diagnosed `ind-square8` as degenerate because "a single push
  never crosses an 8x8 px boundary"; the real cause is a PLACEMENT BUG --
  the target (rows/cols 8-15) never overlapped the pile's actual support
  (measured centroid ~(31.5,31.5), support rows/cols 25-39). With that
  fixed (three new pile-relative goals, centred on a data-measured
  centroid, never hard-coded), the sharp end of the axis is now genuinely
  tested and the answer is NOT "functional-independent": on n20_heap_5mm
  the linear operator collapses to 0.57-0.63 slateK_exact and the
  UNet-linear gap grows up to ~7x; on n20_L20mm the linear operator ALSO
  collapses (to 0.65-0.77) but the gap goes NEGATIVE -- UNet becomes worse
  than linear at the sharpest functional tested. Both halves are
  well-powered and neither is explained by a diagnosed confound
tier: T1
mode: confirmatory
date: 2026-09-07
hypothesis: C-045
claim: >
  C-044/C-045/C-046 all rest on rank_metrics/slateK_exact computed from a
  distance-transform Lyapunov cost, which is low-pass by construction. If
  the "linear operator near the ceiling, UNet's edge small" finding is an
  artifact of this cost being unable to see the fine-detail band the
  UNet's own advantage lives in (C-044), sharpening the functional at a
  FIXED, PILE-INTERSECTING target should erode the linear operator's
  capture and change the UNet-linear gap. The first version of this record
  varied functional sharpness only on a HALF-PLANE target (`corner`), which
  a coordinator review found could never reach the intended regime: a
  half-plane indicator is a single step edge spanning the whole image --
  still a low-frequency weight field (r<=1 energy only fell from 0.629 to
  0.541) -- and its one attempt at a genuinely small, sharp target
  (`ind-square8`) was placed at a FIXED location that never overlapped the
  pile's actual support, giving a trivial zero (V==1 identically, `dV`==0
  for a placement reason, not a "sharpness" reason). This version adds
  targets sized and shaped to intersect the pile's DATA-MEASURED location,
  reaching r<=1 energy 0.060-0.216.
prediction:
  supports: >
    as the weight field sharpens (r<=1 energy falling from 0.629 toward
    <0.10), the linear operator's slateK_exact falls materially -- below
    0.90 at the sharpest non-degenerate functional -- AND the paired
    UNet-linear gap grows to more than twice its value under dist-corner,
    against the paired sem.
  refutes: >
    linear slateK_exact stays >= 0.95 and the UNet-linear gap stays within 1
    sem of its dist-corner value at every non-degenerate functional. Then the
    control results are functional-independent and considerably STRONGER
    than currently claimed.
  discriminating: true
provenance:
  commit: 49c285b1
  dirty: false
  dirty_note: >
    control_utility_test.py (pile_centroid_and_support, the three
    pile-relative goal keys), exp0026_selection_pressure.py/
    expB_multistep_eval.py (compute the centroid from occ0 and thread
    pile_center through) and spectral_concentration.py (the new fields'
    table) were committed at 49c285b1 before the caches below were rebuilt
    against the clean tree (an earlier pass ran against a dirty tree while
    the fix was being written; every number quoted here was regenerated
    after the commit and is bit-identical to the dirty-tree pass, confirming
    determinism, but only the clean-tree numbers are used). No retraining:
    same checkpoints as the first pass (runs_exp0024/unetfilm_cube_spectrum_n20,
    runs_expB/unetfilm_slates_multistep_n20_L20mm).
  data_commit: "Identical to the first pass -- no new data collection."
  script: >
    scripts/probes/exp0026_selection_pressure.py --goals (dataset A),
    scripts/probes/expB_multistep_eval.py --goals (dataset B), both with
    --degradations, now also computing pile_centroid_and_support(occ0) and
    passing pile_center to lyapunov_weights for any goal ending "-pile";
    scripts/probes/exp0026_kcurve_exact.py and exp0026_kcurve.py (unchanged)
    for the K-sweep; scripts/probes/spectral_concentration.py (extended);
    scripts/probes/functional_degeneracy_screen.py (unchanged).
  data: ["runs_exp0024_v2/dv_cache_A.pt (50 slates x up to 32 candidates,
          n20_heap_5mm, 11 goals)",
         "runs_expB/n20_L20mm_v2_dv_cache.pt (20 slates x 128 candidates,
          step-0 subset of n20_L20mm, 11 goals)"]
  code_path: >
    registry.dataset_registry PileSweepData (type: genesis) throughout,
    unchanged. `pile_centroid_and_support` pools occ0 (the exact tensor the
    dv cache already loads) and computes a mass centroid + bounding box --
    no new data path, no new representation.
  seed: "Identical to the first pass (dataset split seeds, K-sweep seed 0, degradation-arm noise seed 1234)."
  split: "Identical to the first pass -- dataset A all 50 n20_heap_5mm slates; dataset B EXP-0024_v1's 30/20 split, step-0 subset of n20_L20mm."
  runtime: "~5 min CPU total (2 dv-cache rebuilds, ~8 kcurve_exact/kcurve sweeps per dataset, 1 spectral-concentration pass, 2 degeneracy screens) -- no GPU."
budget:
  declared: "~1.5h wall-clock for this correction pass (coordinator-set), on top of the original ~3h/250k token budget shared with EXP-0026_v2"
  spent: "~55 min wall-clock for this correction pass"
  outcome: within
design:
  varied:
    functional: ["dist-corner (baseline, r<=1 energy 0.629)",
                 "distclip-corner-r8/r4/r2 (0.619/0.582/0.557)",
                 "ind-corner (0.541)", "dist-square8 (0.683, size control)",
                 "ind-square8 (0.060, PLACEMENT-DEGENERATE -- corrected diagnosis, see Numbers)",
                 "ind-square8-pile (0.060, pile-centred, WORKS)",
                 "ind-square16-pile (0.216, pile-centred, DEGENERATE -- near-total containment, see Numbers)",
                 "ind-stripe-thin-pile (0.132, pile-centred, WORKS)"]
    dataset: ["A: n20_heap_5mm (K<=31, 50 slates)", "B: n20_L20mm step-0 (K<=128, 20 slates x 128 real candidates)"]
    K: [2, 4, 8, 16, 24or32, 64, 128or31]
  held_fixed:
    predictions: "identical to EXP-0024's fit/checkpoint (dataset A) and EXP-0024_v1's fit/checkpoint (dataset B) -- nothing about the models changes; only the weight field d/w and (secondarily) K vary"
    pile_center: "computed once per dataset from occ0, pooled over ALL candidates in that dataset's cache -- (31.577, 31.319) for A, (31.265, 31.546) for B, both from pile_centroid_and_support, never hard-coded"
    metric: "slateK_exact/worstK/rank_profile (docs/experiments/METRICS.md, closed form) at K=2..31 (A) or K=2..128 (B), sampled slateK/regret_dv/pick_pctile at K=4 and K=max as the bridge"
  baselines: [persistence, mean-delta, oracle]
  metric: "slateK_exact (headline), worstK, rank_profile, regret_dv, pick_pctile, plus sampled slate4/slateK bridge, all per docs/experiments/METRICS.md"
noise_floor: >
  Same second-order pairing as the first pass: for each slate, (gap under
  functional X) - (gap under dist-corner), paired across the SAME 50 (or
  20) slates, so a functional's effect on the gap is tested against its own
  noise floor rather than the gap's absolute one.
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend,
             swept-region-metric, episode-split, settled-state]
establishes: []
result: >
  CORRECTED DIAGNOSIS: `ind-square8`'s degeneracy is a PLACEMENT BUG, not a
  reachability limit. Measured (`pile_centroid_and_support`, pooled occ0):
  pile support rows 25-39, cols 25-39 on BOTH datasets (centroid ~(31.5,
  31.5)) -- `ind-square8`'s fixed target (rows/cols 8-15) is entirely
  disjoint from this, so the target is empty at every state, V==1
  identically, and dV==0 follows trivially (a saturated indicator, C-040's
  failure mode in new clothes). DEGENERACY SCREEN on the three new
  pile-relative goals: `ind-square8-pile` PASSES on both datasets (A: 2.9%
  helpful, sd 0.119, 1507 distinct values, 50/50 slates clear -- skewed but
  not degenerate by the numeric screen; B: 37.3% helpful, sd 0.103, 20/20
  clear). `ind-stripe-thin-pile` PASSES cleanly on both (A: 14.3% helpful,
  sd 0.111; B: 43.6% helpful, sd 0.075). `ind-square16-pile` FAILS on BOTH
  (0.0% helpful on A and B -- the 16x16 px target is large enough,
  relative to the pile's own ~14x14 px support, to nearly contain it, so
  every dispersing push increases cost almost deterministically; excluded
  from the verdict). THE PREDICTION IS NOW GENUINELY TESTED, and splits by
  dataset. Linear slateK_exact falls WELL below 0.90 on every working sharp
  cell: dataset A `ind-square8-pile` 0.7649 (K=4) -> 0.6294 (K=31);
  `ind-stripe-thin-pile` 0.7145 -> 0.5680; dataset B `ind-square8-pile`
  0.8689 (K=4) -> 0.7192 (K=128); `ind-stripe-thin-pile` 0.8322 -> 0.7693.
  The FIRST clause of "supports" is decisively confirmed everywhere. The
  SECOND clause (gap doubles) holds on dataset A (`ind-stripe-thin-pile`
  K=4 gap +0.0706 vs dist-corner's +0.0103, a 6.84x ratio, gap-vs-corner
  t=+4.05; `ind-square8-pile` K=31 gap +0.1051 vs +0.0176, 5.97x, t=+1.77)
  but FAILS on dataset B, where the gap REVERSES SIGN (`ind-square8-pile`
  K=4 gap -0.0155, sem 0.0065, t=-2.39 vs dist-corner's +0.0173,
  gap-vs-corner t=-4.30; `ind-stripe-thin-pile` K=4 gap -0.0101, t=-0.93).
  Neither "supports" nor "refutes" describes both datasets. VERDICT:
  inconclusive -- the discriminating test worked and both halves are
  well-powered, but the two datasets disagree on the gap's direction with
  no diagnosed confound (unlike EXP-0024_v1's warp-limited L10mm) to
  explain the disagreement away.
verdict: inconclusive
downgrades: [imprecision, untested-dependency, inconsistency]
grade: very-low
supersedes: []
invalidated_by: null
---

**Tier note.** Same reason as every record in this family: T1 rather than
T2 because of the `settled-state` dependency, despite a pre-registered
prediction.

## Correction, 2026-09-07: the original diagnosis was wrong

The first version of this record reported `ind-square8` as degenerate
"because a single push never crosses an 8x8 px boundary this far from the
pile's reachable footprint" -- a reachability claim. **A coordinator review
found the actual cause: the target simply never overlapped the pile.**
`ind-square8`'s mask sat at rows/cols [8:16] (an "eighth-side square" offset
one side-length in from the corner, chosen to avoid the array-boundary
placement issue documented in the first pass). Measuring the pile's actual
location directly (`pile_centroid_and_support`, pooling occ0 over every
candidate in each dataset) gives support rows 25-39, cols 25-39 on BOTH
`n20_heap_5mm` and `n20_L20mm` -- disjoint from [8:16] on both axes. The
target was never reachable not because a push is too short, but because it
was never IN THE PILE'S NEIGHBOURHOOD at all: `V = d^T y / ||y||_1` with `d`
an indicator of an always-empty region is `1 - 0/mass = 1` identically, so
`dV = 0` follows without any physics being tested. This is C-040's
degeneracy (a target whose relationship to the pile makes `dV` trivially
zero) wearing different clothes -- saturated at "always full" there,
"always empty" here.

**Consequence, corrected**: the sharp end of the functional-sharpness axis
was UNTESTED by the first pass, not refuted. This version tests it properly
with three new goals, each requiring an explicit `pile_center` computed from
the data (`pile_centroid_and_support(occ0)`, never hard-coded), so the
target genuinely intersects the pile's own footprint.

## Why this test discriminates

Unchanged from the first pass: every one of C-030/C-035/C-039/C-044/
C-045/C-046 is computed from a distance-transform Lyapunov cost, low-pass
by construction. What changes here is that the first pass's sharpness axis
(`corner` -> `distclip-corner-r*` -> `ind-corner`) never left a HALF-PLANE
target -- a single step edge spanning the whole image is itself a
low-frequency weight field (r<=1 energy only fell from 0.629 to 0.541,
never below ~0.5), so it could not distinguish "sharpening the functional
doesn't matter" from "this design never got sharp enough to see." Adding a
target sized to the pile itself (8x8 or a 4px stripe, against a ~14x14 px
pile) reaches r<=1 energy 0.060-0.132 -- the regime the pre-registered
prediction actually names ("<0.10").

## What was actually run

**`pile_centroid_and_support(occ, thresh=1e-6)`** (new, `control_utility_test.py`):
pools an occupancy tensor's leading dims, takes the mean field, and returns
the mass centroid `(cy, cx)` and the tight bounding box of rows/cols whose
pooled mass exceeds `thresh`. Run on dataset A's 1597 candidates and dataset
B's 2560 step-0 candidates independently: both give support rows/cols
25-39 (14 px wide) and centroid within 0.3 px of (31.5, 31.5) -- i.e. the
pile sits almost exactly centred in the 64x64 grid on both datasets, which
is also why `center` (rows/cols 16-48, entirely containing this support) is
degenerate (C-040: the whole pile is always "inside", nothing to score).

**Three new goal keys** (`control_utility_test.py::lyapunov_weights`, all
require `pile_center`): `ind-square8-pile`/`ind-square16-pile` (8x8 / 16x16
px indicator centred on `pile_center`, clamped to stay in-bounds);
`ind-stripe-thin-pile` (4-px-wide, full-image-width indicator stripe
centred on `pile_center`'s row). `exp0026_selection_pressure.py`/
`expB_multistep_eval.py` compute the centroid from each dataset's own occ0
once (only if any requested goal ends `-pile`) and pass it through.

**Spectral concentration**: unchanged methodology, extended to the three
new fields. Confirmed position-invariant as expected (an indicator's
`|FFT|` does not depend on which in-bounds `pile_center` is used -- verified
`ind-square8-pile` reproduces `ind-square8`'s own 0.060/0.514/0.819 to the
digit, despite sitting at a completely different location).

**Degeneracy screen, K-sweep**: identical scripts/methodology to the first
pass, run against the rebuilt caches.

## Numbers

**Spectral concentration** (64x64, demeaned |FFT|^2, fraction within radius
r of DC) -- new rows only, the six from the first pass are unchanged:

| weight field | r<=1 | r<=4 | r<=8 |
|---|---|---|---|
| ind-square8-pile | 0.060 | 0.514 | 0.819 |
| ind-square16-pile | 0.216 | 0.803 | 0.900 |
| ind-stripe-thin-pile | 0.132 | 0.488 | 0.797 |

**Pile location** (`pile_centroid_and_support`, pooled occ0): A centroid
(31.577, 31.319), support rows/cols (25, 39, 25, 39). B centroid (31.265,
31.546), support rows/cols (25, 39, 25, 39). Essentially identical on both
datasets -- the pile is centred in the tray by construction in both
collections.

**Degeneracy screen** (`functional_degeneracy_screen.py`):

| goal | dataset | n | mean dv_true | sd | %helpful | %\|dv\|<eps | slates clearing | status |
|---|---|---|---|---|---|---|---|---|
| ind-square8 (old, corner-relative) | A | 1597 | 0.0000 | 0.0000 | 0% | 100% | 0/50 | DEGENERATE (placement bug, corrected) |
| ind-square8 (old, corner-relative) | B | 2560 | 0.0000 | 0.0000 | 0% | 100% | 0/20 | DEGENERATE (placement bug, corrected) |
| **ind-square8-pile** | A | 1597 | +0.2171 | 0.1192 | **2.9%** | 0.0% | 50/50 | ok (skewed -- see caveat below) |
| **ind-square8-pile** | B | 2560 | +0.0266 | 0.1027 | 37.3% | 18.9% | 20/20 | ok |
| ind-square16-pile | A | 1597 | +0.2891 | 0.1351 | **0.0%** | 0.3% | 50/50 | **DEGENERATE** |
| ind-square16-pile | B | 2560 | +0.0617 | 0.0963 | **0.0%** | 50.4% | 20/20 | **DEGENERATE** |
| **ind-stripe-thin-pile** | A | 1597 | +0.1101 | 0.1111 | 14.3% | 0.2% | 50/50 | ok |
| **ind-stripe-thin-pile** | B | 2560 | -0.0057 | 0.0751 | 43.6% | 19.3% | 20/20 | ok |

`ind-square16-pile` fails on both datasets in the same way `center` does
(C-040): a target large enough to nearly contain the pile's own support
makes almost every dispersing push increase cost, so `%helpful` rounds to
0.0% on both -- reported and excluded, not silently dropped.
`ind-square8-pile` on dataset A passes the numeric screen (2.9% > the 1%
cutoff) but is heavily skewed toward "harmful" -- caveated below, not
excluded, since it still has real variance (1507 distinct values, 50/50
slates clear) and a skewed sign distribution does not by itself prevent a
meaningful ranking test.

**`slateK_exact`, linear, at K=4 and K=max**:

| functional (r<=1) | A K=4 | A K=31 | B K=4 | B K=128 |
|---|---|---|---|---|
| dist-corner (0.629) | 0.9767 | 0.9581 | 0.9533 | 0.9670 |
| ind-corner (0.541) | 0.9739 | 0.9233 | 0.9431 | 0.9346 |
| **ind-square8-pile (0.060)** | **0.7649** | **0.6294** | **0.8689** | **0.7192** |
| **ind-stripe-thin-pile (0.132)** | **0.7145** | **0.5680** | **0.8322** | **0.7693** |

Linear's capture collapses at the genuinely sharp end -- 0.57-0.77 across
all four (dataset x functional) working cells, well below the 0.90
supports-threshold everywhere, and below it even at K=4 on both datasets
for `ind-stripe-thin-pile`. UNet:

| functional | A K=4 | A K=31 | B K=4 | B K=128 |
|---|---|---|---|---|
| ind-square8-pile | 0.7866 | 0.7346 | 0.8534 | 0.6532 |
| ind-stripe-thin-pile | 0.7851 | 0.6725 | 0.8221 | 0.7244 |

UNet also collapses on dataset B (0.85->0.65 for square8-pile, 0.82->0.72
for stripe) -- BELOW linear's own K=128 value for `ind-square8-pile`
(0.6532 vs 0.7192), the sign flip driving the negative gap below.

**Paired UNet-linear gap, and gap-vs-dist-corner (second-order pairing)**:

| dataset | functional | K | gap (own sem, t) | dist-corner gap | ratio | gap-vs-corner (sem, t) |
|---|---|---|---|---|---|---|
| A | ind-square8-pile | 4 | +0.0217 (0.0126, 1.72) | +0.0103 | 2.10x | +0.0113 (0.0126, +0.90) |
| A | ind-square8-pile | 31 | +0.1051 (0.0506, 2.08) | +0.0176 | 5.97x | +0.0875 (0.0494, +1.77) |
| A | ind-stripe-thin-pile | 4 | +0.0706 (0.0151, 4.67) | +0.0103 | 6.84x | +0.0603 (0.0149, **+4.05**) |
| A | ind-stripe-thin-pile | 31 | +0.1045 (0.0456, 2.29) | +0.0176 | 5.94x | +0.0869 (0.0437, +1.99) |
| B | ind-square8-pile | 4 | **-0.0155** (0.0065, -2.39) | +0.0173 | -0.89x | -0.0327 (0.0076, **-4.30**) |
| B | ind-square8-pile | 128 | -0.0659 (0.0425, -1.55) | +0.0151 | -4.35x | -0.0811 (0.0472, -1.72) |
| B | ind-stripe-thin-pile | 4 | **-0.0101** (0.0108, -0.93) | +0.0173 | -0.58x | -0.0274 (0.0115, **-2.37**) |
| B | ind-stripe-thin-pile | 128 | -0.0449 (0.0651, -0.69) | +0.0151 | -2.97x | -0.0601 (0.0706, -0.85) |

On dataset A the gap grows with sharpness, clearing its own gap-vs-corner
sem decisively for `ind-stripe-thin-pile` at K=4 (t=+4.05) and reaching
~6x at K=31 (t=+1.77-1.99, the same K=max under-powering EXP-0024_v1's
reviewer amendment already flagged). On dataset B the gap **reverses
sign and clears its own sem in the negative direction** at K=4 for both
functionals (t=-4.30, t=-2.37) -- UNet is measurably WORSE than the linear
operator here, the opposite of every other cell in this register.

**worstK at K=4** (adversarial pool): A `ind-square8-pile` linear 0.9971 /
UNet 0.9573 (UNet still better, worst-case); A `ind-stripe-thin-pile`
linear 1.1983 / UNet 1.0957 (UNet better); B `ind-square8-pile` linear
1.0883 / UNet **1.2460** (UNet WORSE, matching the average-case reversal);
B `ind-stripe-thin-pile` linear 1.0207 / UNet 1.1059 (UNet worse again).
worstK agrees with the average-case sign flip on B and the average-case
advantage on A -- not a metric artifact confined to one statistic.

**Sampled `slate4`/`slateK` bridge** (with replacement, `regret_dv`/
`pick_pctile` alongside): dataset A `ind-square8-pile` linear/UNet
0.704/0.728 (K=4) -> 0.630/0.735 (K=31), regret_dv linear 0.0174->0.0537,
UNet 0.0157->0.0361 (UNet's absolute regret stays lower at K=31 despite the
closed-form capture numbers being close, since regret's denominator moves
too); dataset B `ind-square8-pile` linear/UNet 0.803/0.782 (K=4) ->
0.719/0.653 (K=128), regret_dv linear 0.0105->0.0480, UNet 0.0119->0.0588
(UNet's regret is now WORSE in absolute terms too, not just in the bounded
ratio) -- the reversal on dataset B is real by both the bounded and
unbounded metric, not an artifact of `slateK_exact`'s own normalisation.

## What this means

**The original "functional-independent" conclusion is WITHDRAWN for the
genuinely sharp regime.** It was correct only for the range the first
pass actually tested (r<=1 energy 0.541-0.683, a half-plane target at
varying steepness) -- within that range, sharpening indeed barely moved
anything, and that finding stands unchanged (see the first pass's Numbers,
reproduced above for the dist-corner/ind-corner rows). But that range never
reached the regime the pre-registered prediction named ("<0.10"), because a
half-plane's own geometry -- a single edge spanning the whole image -- is
inherently low-frequency regardless of how steep the transition is made.
**Geometry, not just functional steepness, sets how sharp a weight field
can get**, and this record's targets (an 8x8 px blob or a 4px stripe sized
to the pile itself) are the first in this family to reach r<=1 < 0.15.

**At the genuinely sharp end, the linear operator's near-ceiling
performance does NOT survive.** slateK_exact falls to 0.57-0.77 across
every working cell -- a 20-40 point drop from its dist-corner value of
0.95-0.98. This is the first evidence in the register that the "linear
operator captures ~95% of the oracle's advantage" finding (C-045) is
functional-dependent after all, once the functional is sharp AND
pile-intersecting rather than merely a steep half-plane.

**Whether the UNet benefits from this is dataset-dependent, and this
record cannot resolve why.** On `n20_heap_5mm` (dataset A, EXP-0024's
original checkpoint/fit, 32 sampler-drawn candidates) the UNet's edge grows
substantially and significantly at K=4 for the stripe functional (6.84x,
t=4.05). On `n20_L20mm` (dataset B, EXP-0024_v1's checkpoint, 128 REAL
candidates, per-length in-distribution training) the UNet's edge not only
fails to grow, it reverses sign with high confidence at K=4 for both
functionals (t=-2.39, t=-4.30) -- the linear operator becomes the BETTER
model once the cost is sharp enough to reward fine detail, exactly
backwards from the "UNet's fine-detail advantage should show up more under
a sharp cost" mechanism C-044 proposes. **Both results are well-powered
(clear t-statistics at K=4, the least tie-inflated K); neither is explained
by a diagnosed confound the way EXP-0024_v1's L10mm warp-limited regime
explained away that record's one discrepant cell.** This is reported as a
genuine, unresolved disagreement, not smoothed into either a "supports" or
"refutes" reading.

## What would change the verdict

- **A mechanism for the A/B disagreement.** Candidate hypotheses, untested
  here: (a) dataset A's UNet was trained on `cube_spectrum/n20` (blind to
  this exact slate collection) while dataset B's was trained
  in-distribution on `n20_L20mm` itself -- a sharp, pile-local cost might
  reward whichever model's errors happen to be spatially correlated with
  the SPECIFIC training distribution in a way a smooth cost cannot see; (b)
  the two datasets' pushes differ in length/sampler (contact-aware
  sampler-drawn vs placement-aware real candidates) in a way that
  interacts with a small, pile-local target differently. Testing either
  requires a third dataset or a controlled swap (same checkpoint, both
  slate collections), out of this correction's budget.
- **`ind-square8-pile`'s skew on dataset A** (97% of candidates increase
  cost) means its ranking signal there is thinner than `ind-stripe-thin-pile`'s;
  a stripe or blob NOT centred exactly on the pile centroid (offset so
  roughly half of candidates are net-helpful) would give a better-balanced
  same-target sharp cell to compare against dataset B's better-balanced
  36-44% helpful.
- **`ind-square16-pile`'s degeneracy** could potentially be worked around
  with an OFF-centre 16x16 target (still sharp, but not large enough
  relative to the pile to risk near-total containment) -- not attempted
  here, budget-limited.

## Threats

- `imprecision`: one UNet training seed, one linear fit per dataset,
  unchanged from the first pass.
- `untested-dependency`: `settled-state` unchecked for the rigid-cube path.
- `inconsistency`: dataset A and dataset B disagree on the sign of the
  UNet-linear gap at the sharp end, both at a level that clears its own
  paired sem (t=+4.05 on A, t=-4.30 on B for the K=4 comparisons), with no
  diagnosed confound distinguishing them -- reported as the finding, not
  resolved. This is the reason the verdict is `inconclusive` rather than
  `supported` or `refuted`.
- Considered and dismissed: `provenance` -- every number reuses EXP-0024/
  EXP-0024_v1's own checkpoints and fits through their own unmodified code
  path; the only new code (`pile_centroid_and_support`, the three new goal
  keys, the two probe scripts' centroid-computation lines) is a measurement
  and a mask placement, not a change to model inference, fitting, or the
  metric implementations. The rebuilt caches reproduce the first pass's own
  `corner`/`ind-corner`/etc. numbers to the digit (dirty-tree pass vs
  clean-tree pass, bit-identical, confirming determinism).
- Considered and dismissed: `selection` -- all three new goals are
  reported, including the one that failed the screen (`ind-square16-pile`)
  and the dataset (B) whose result is least convenient for a clean
  "sharpening helps the UNet" narrative.

## Unrelated findings

- The pile in BOTH `n20_heap_5mm` and `n20_L20mm` sits almost exactly
  centred in the 64x64 grid (centroid within 0.3 px of (31.5, 31.5), a
  ~14x14 px support) -- this is WHY `center` (rows/cols 16-48) has been
  degenerate in every record that has used it (C-040): the target contains
  the pile's entire reachable footprint, so nothing can leave it. The two
  datasets' near-identical centroid and support size, despite different
  collection procedures (contact-aware sampler-drawn vs placement-aware
  real candidates), was not something either prior record measured
  directly.
- An indicator's spectral concentration is confirmed position-invariant a
  second time, now across a much larger displacement (corner-relative
  [8:16] vs pile-relative [~24:32]): `ind-square8-pile` reproduces
  `ind-square8`'s own 0.060/0.514/0.819 to 3 decimals despite sitting
  roughly 20 px away. This makes `pile_centroid_and_support`'s exact
  centroid non-load-bearing for the SPECTRAL table (any in-bounds
  placement gives the same numbers) even though it is essential for the
  `dV` numbers themselves (which very much depend on where the mass
  actually is).
- `worstK` (the adversarial-pool metric) agrees with the average-case sign
  flip on dataset B and the average-case advantage on dataset A at every
  sharp functional tested -- the reversal is not confined to one summary
  statistic's own normalisation quirk.
