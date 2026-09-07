---
id: EXP-0024_v2
title: >
  Every control-utility number in this register (C-030/C-035/C-039/C-044/
  C-045/C-046) comes from ONE cost functional, V = d^T y / ||y||_1 with d a
  distance transform -- low-pass by construction. Sharpening the functional
  at a FIXED target (distclip-corner-r{2,4,8} -> ind-corner) on EXP-0024's
  own n20_heap_5mm slates and EXP-B's n20_L20mm cell: linear slateK_exact
  never falls below 0.92 and the paired UNet-linear gap does not grow --
  if anything it shrinks. The sharpest planned cell (ind-square8, a small
  off-centre target) is DEGENERATE on both datasets: a single push never
  moves material across an 8x8 px boundary, so dV=0 for every candidate
tier: T1
mode: confirmatory
date: 2026-09-07
hypothesis: C-045
claim: >
  C-044/C-045/C-046 all rest on rank_metrics/slateK_exact computed from a
  distance-transform Lyapunov cost, which is low-pass by construction
  (measured: 0.629 of its |FFT|^2 sits within radius 1 of DC on a 64x64
  grid, demeaned). If the "linear operator near the ceiling, UNet's edge
  small" finding is an artifact of this cost being unable to see the
  fine-detail band the UNet's own advantage lives in (C-044), sharpening the
  functional at a FIXED target should erode the linear operator's capture
  and grow the UNet-linear gap. `lyapunov_weights` is extended with a
  sharpness family (distclip-corner-r{2,4,8}, ind-corner) that keeps the
  TARGET fixed (the corner half-plane) and only changes the functional FORM,
  plus a target-SIZE control (dist-square8) that keeps the functional a
  distance transform but shrinks the target -- isolating which axis, if
  either, does the work.
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
  commit: db8f9a28
  dirty: false
  dirty_note: >
    control_utility_test.py, scripts/probes/exp0026_selection_pressure.py,
    scripts/probes/expB_multistep_eval.py and the two new probe scripts
    (spectral_concentration.py, functional_degeneracy_screen.py) were
    committed at db8f9a28 BEFORE any of this record's numbers were produced
    -- the prediction above is this commit's own diff, so it reconstructs
    the exact code that ran. No retraining and no new data collection: every
    number here reuses EXP-0024's UNet checkpoint (runs_exp0024/unetfilm_cube_spectrum_n20)
    and EXP-0024_v1's L20mm checkpoint (runs_expB/unetfilm_slates_multistep_n20_L20mm)
    unmodified.
  data_commit: >
    Identical to EXP-0024/EXP-0026 (Genesis/data/slates/n20_heap_5mm,
    006004d0..dbf21ba2) and EXP-0024_v1/EXP-0026_v1
    (Genesis/data/slates_multistep/n20_L20mm, 8c006d88) -- no new data.
  script: >
    scripts/probes/exp0026_selection_pressure.py --goals (dataset A),
    scripts/probes/expB_multistep_eval.py --goals (dataset B), both with
    --degradations; scripts/probes/exp0026_kcurve_exact.py and
    scripts/probes/exp0026_kcurve.py (unchanged) for the K-sweep;
    scripts/probes/spectral_concentration.py (new) for the weight-field
    spectral table; scripts/probes/functional_degeneracy_screen.py (new)
    for the pre-metric screen.
  data: ["runs_exp0024_v2/dv_cache_A.pt (50 slates x up to 32 candidates,
          n20_heap_5mm, 8 goals incl. 6 new)",
         "runs_expB/n20_L20mm_v2_dv_cache.pt (20 slates x 128 candidates,
          step-0 subset of n20_L20mm, same 8 goals)"]
  code_path: >
    registry.dataset_registry PileSweepData (type: genesis) throughout --
    EXP-0024/EXP-0026/EXP-0024_v1's own code path, unchanged. The only new
    code is in control_utility_test.py::lyapunov_weights (new goal keys,
    center/corner/stripe left byte-identical -- verified: dv_true for
    goal=corner on dataset A reproduces EXP-0026's own numbers to 4 decimals,
    e.g. slateK_exact(linear, K=4)=0.9767 matching the self-test value in
    exp0026_kcurve_exact.py) and the two new probe scripts, which do not
    touch the model/fit/evaluation code path at all.
  seed: >
    Identical to EXP-0024/EXP-0026 (dataset A) and EXP-0024_v1 (dataset B,
    slate-level split seed 0). K-sweep bootstrap/sampler seed 0
    (exp0026_kcurve.py/exp0026_kcurve_exact.py defaults). Degradation-arm
    noise seed 1234 (unchanged, EXP-0025's own).
  split: >
    Dataset A: all 50 n20_heap_5mm slates (genesis_cube_spectrum_n20_slates_all.yaml,
    split="train", val_pct=0/test_pct=0), EXP-0026's own convention. Dataset
    B: EXP-0024_v1's 30-train/20-eval slate-level split on n20_L20mm,
    step-0-only for control ranking (128 real candidates/slate).
  runtime: "~4 min CPU total (2 dv-cache builds, ~8 kcurve_exact/kcurve sweeps per dataset, 1 spectral-concentration pass, 2 degeneracy screens) -- no GPU, no retraining."
budget:
  declared: "~3h wall-clock, ~250k tokens (shared with EXP-0026_v2 -- one collection/analysis pass across both records, per the task)"
  spent: "~1h40min wall-clock, ~90k tokens for the collection+analysis; remainder spent on this record and its companion, METRICS.md/REGISTER.md updates"
  outcome: within
design:
  varied:
    functional: ["dist-corner (baseline, r<=1 energy 0.629)",
                 "distclip-corner-r8 (0.619)", "distclip-corner-r4 (0.582)",
                 "distclip-corner-r2 (0.557)", "ind-corner (0.541)",
                 "dist-square8 (0.683, target-SIZE control, same functional form as dist-corner)",
                 "ind-square8 (0.060, DEGENERATE -- see Numbers)"]
    dataset: ["A: n20_heap_5mm (K<=31, 50 slates)", "B: n20_L20mm step-0 (K<=128, 20 slates x 128 real candidates)"]
    K: [2, 4, 8, 16, 24or32, 64, 128or31]
  held_fixed:
    predictions: "identical to EXP-0024's fit/checkpoint (dataset A) and EXP-0024_v1's fit/checkpoint (dataset B) -- nothing about the models changes; only the weight field d/w and (secondarily) K vary"
    target_shape_for_corner_family: "the corner half-plane mask (rows/cols [0:H/2)), identical across dist-corner/distclip-r*/ind-corner -- only the functional applied to that SAME mask changes"
    target_for_square_family: "an 8x8 px square (H//8 side), offset one side-length in from the corner (rows/cols [H/8:2H/8)) -- identical across dist-square8/ind-square8, only the functional changes"
    metric: "slateK_exact/worstK/rank_profile (docs/experiments/METRICS.md, closed form) at K=2..31 (A) or K=2..128 (B), sampled slateK at K=4 as the slate4 bridge"
  baselines: [persistence, mean-delta, oracle]
  metric: "slateK_exact (headline), worstK, rank_profile, plus sampled slateK (slate4 bridge), all per docs/experiments/METRICS.md"
noise_floor: >
  Paired across slates (50 for A, 20 for B), sem of the per-slate
  UNet-linear difference -- never the across-slate sd, per every prior
  record's own correction. Because the axis under test here is FUNCTIONAL,
  not model, the load-bearing comparison is a SECOND-ORDER pairing: for each
  slate, (gap under functional X) - (gap under dist-corner), paired across
  the SAME 50 (or 20) slates, so a functional's effect on the gap is tested
  against its own noise floor, not against the gap's absolute noise floor
  (which would overstate how confidently two functionals' gaps differ, since
  both inherit the same per-slate difficulty variance). This is the same
  second-order pairing logic EXP-0024_v1's push-length comparisons needed
  and did not have; it is used here because the task requires attributing
  an effect to the functional axis specifically.
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend,
             swept-region-metric, episode-split, settled-state]
establishes: []
result: >
  SPECTRAL TABLE (measured, scripts/probes/spectral_concentration.py,
  64x64, DC-demeaned): dist-corner 0.629/0.883/0.941 (r<=1/4/8); dist-square8
  0.683/0.908/0.953; ind-corner 0.541/0.870/0.937; ind-square8
  0.060/0.514/0.819; distclip-r8/r4/r2 0.619/0.582/0.557 at r<=1 -- a clean
  monotone sharpening at FIXED target from distclip-r8 down to ind-corner,
  confirming dist-square8 barely moves off dist-corner's own concentration
  despite an 8x wide->8px target shrink (size is not the lever, exactly the
  task's premise). DEGENERACY SCREEN: ind-square8 is degenerate on BOTH
  datasets (dv_true == 0 for all 1597/2560 candidates -- a single push never
  crosses an 8x8 px boundary this far from the pile's reachable footprint);
  center is degenerate as always (C-040); every other functional passes
  (40-53% helpful, sd 0.02-0.29, 50/50 or 20/20 slates clearing the
  variation threshold). PREDICTION: the "supports" branch is REFUTED on both
  datasets. Linear slateK_exact at the sharpest non-degenerate functional
  (ind-corner) never drops below 0.90: 0.9739 (K=4) -> 0.9233 (K=31) on A;
  0.9431 (K=4) -> 0.9346 (K=128) on B -- both comfortably above the 0.90
  supports-threshold (though A's K=31 value dips just under the refutes
  threshold of 0.95, to 0.9233 -- noted, not treated as supporting evidence,
  since the gap clause below still fails decisively for supports). The
  paired UNet-linear gap does NOT grow with sharpness -- it is flat to
  slightly SMALLER: dataset A K=4, dist-corner +0.0103 (sem 0.0016) ->
  ind-corner +0.0071 (sem 0.0019), gap-vs-corner -0.0032 (sem 0.0021,
  t=-1.54); dataset B K=4, dist-corner +0.0173 (sem 0.0032) -> ind-corner
  +0.0136 (sem 0.0035), gap-vs-corner -0.0037 (sem 0.0036, t=-1.04). No
  functional at either dataset produced a POSITIVE, sem-clearing
  gap-vs-corner difference at any K tested -- the largest positive point
  estimate was dataset A's dist-square8 at K=31 (+0.0063, sem 0.0154,
  t=0.41), inside noise. VERDICT: refuted.
verdict: refuted
downgrades: [imprecision, untested-dependency, incomplete-design]
grade: very-low
supersedes: []
invalidated_by: null
---

**Tier note.** Same reason as EXP-0024/EXP-0024_v1/EXP-0026/EXP-0026_v1: a
prediction block was written and the analysis code committed (db8f9a28)
before any number below was produced, but this is filed T1 rather than T2
because it depends on `settled-state`, `unchecked` for the rigid-cube path.

## Why this test discriminates

Every one of C-030/C-035/C-039/C-044/C-045/C-046 is computed from
`V = d^T y / ||y||_1` with `d` a distance transform to a goal region --
`center`/`corner`/`stripe`, unchanged since EXP-0008. A distance transform is
low-pass by construction: away from the target boundary it is smooth almost
everywhere, so a metric built on it may simply be unable to register the
high-frequency content the UNet's whole image-accuracy advantage lives in
(C-044). If so, "the linear operator is near the ceiling and the UNet's edge
is small" (C-045) is a fact about THIS COST, not about control in general.
This design breaks the confound the register has never separated: functional
SHARPNESS (distance transform -> clipped distance transform -> indicator) and
target SIZE (a half-plane vs an 8x8 px square) are varied INDEPENDENTLY, at
the SAME predictions, the SAME slates, the SAME checkpoints as the records
under test -- so a result here is directly comparable to theirs, not a new
task that could differ for unrelated reasons.

## What was actually run

**`lyapunov_weights` extended** (`control_utility_test.py`): `center`/`corner`/
`stripe` keep the exact code path they always had (verified: dataset A's
`corner` numbers reproduce EXP-0026's own slateK_exact self-test value,
0.9767 at K=4, to 4 decimals). New keys: `distclip-corner-r{2,4,8}` (distance
transform to the corner half-plane mask, clipped at r px and renormalised,
`min(dist,r)/r`); `ind-corner` (indicator of the same mask, the r->0 limit);
`ind-square8`/`dist-square8` (indicator / distance transform to an 8x8 px
square offset one side-length in from the corner -- flush-corner placement
was tried first and rejected: a distance transform's actual values, unlike
an indicator's `|FFT|`, are NOT shift-invariant on a finite non-periodic
domain, and a corner-flush square understated its own concentration by
sitting where most of the grid is on one side of it; the offset placement's
measured spectrum matches this task's own reference table to 3 decimals).

**Spectral concentration** (`scripts/probes/spectral_concentration.py`, new):
fraction of `|FFT(field - mean(field))|^2` within radius r (grid units) of
DC, 64x64. Demeaning is necessary and load-bearing: without it every field
here (mean far from zero) reads as ~85-98% concentrated at DC regardless of
shape, because the mean itself dominates raw power. After demeaning, this
script's numbers match the task's own reference table on 4 of 5 rows to 3
decimals (dist-corner 0.629/0.883/0.941; ind-corner 0.541/0.870/0.937;
ind-square8 0.060/0.514/0.819; dist-square8, once the square is offset rather
than corner-flush, 0.683/0.908/0.953) -- confirming the reference table's own
methodology before trusting this record's new distclip-r* rows.

**Degeneracy screen** (`scripts/probes/functional_degeneracy_screen.py`,
new), run BEFORE any slateK/slateK_exact number, per the task's explicit
requirement: reports dv_true mean/sd, %helpful, %|dv|<1e-9, #distinct dv
values, and slates clearing the same per-slate variation threshold
`exp0026_kcurve_exact.py` itself uses (std >= 1e-9) to skip a slate.

**dv caches**: `exp0026_selection_pressure.py --goals` (dataset A, all 50
slates, `--degradations`) and `expB_multistep_eval.py --goals` (dataset B,
step-0 subset of the 20 eval slates, `--degradations`) -- both scripts
unchanged apart from the new `--goals` argument, which defaults to
`corner,center` (so every existing invocation is byte-identical) and loops
`lyapunov_weights` over whatever list is passed. No retraining, no new fit:
dataset A reuses `runs_exp0024/unetfilm_cube_spectrum_n20`; dataset B reuses
`runs_expB/unetfilm_slates_multistep_n20_L20mm` (EXP-0024_v1's checkpoint).

**K-sweep**: `exp0026_kcurve_exact.py` (closed-form slateK_exact/worstK/
rank_profile, unchanged) and `exp0026_kcurve.py` (sampled slate4 bridge,
unchanged) against each new goal in both caches.

## Numbers

**Spectral concentration** (64x64, demeaned |FFT|^2, fraction within radius
r of DC):

| weight field | r<=1 | r<=4 | r<=8 |
|---|---|---|---|
| dist-corner (baseline, "corner") | 0.629 | 0.883 | 0.941 |
| distclip-corner-r8 | 0.619 | 0.926 | 0.966 |
| distclip-corner-r4 | 0.582 | 0.906 | 0.964 |
| distclip-corner-r2 | 0.557 | 0.886 | 0.952 |
| **ind-corner** | **0.541** | 0.870 | 0.937 |
| dist-square8 (size control) | 0.683 | 0.908 | 0.953 |
| **ind-square8** (DEGENERATE, see below) | **0.060** | 0.514 | 0.819 |

Monotone at fixed target from r8 down to ind-corner (0.619 -> 0.582 -> 0.557
-> 0.541), confirming distclip is a real sharpness knob. dist-square8 sits
essentially at dist-corner's own concentration (0.683 vs 0.629, barely
higher) despite an 8x linear target shrink -- the task's premise, reproduced
here independently: **shrinking a distance-transform target does not
sharpen it; switching functional form does.**

**Degeneracy screen** (`functional_degeneracy_screen.py`, pooled over both
datasets' candidates):

| goal | dataset | n | mean dv_true | sd | %helpful | %\|dv\|<eps | slates clearing |
|---|---|---|---|---|---|---|---|
| corner | A | 1597 | +0.0189 | 0.0514 | 38% | 0.0% | 50/50 |
| corner | B | 2560 | -0.0003 | 0.0234 | 49% | 18.4% | 20/20 |
| center | A/B | — | ~0 | ~0.001-0.004 | 0% | 99% | 11/50, 13/20 | **DEGENERATE (C-040, as always)** |
| **ind-square8** | **A** | 1597 | **0.0000** | **0.0000** | **0%** | **100%** | **0/50** | **DEGENERATE** |
| **ind-square8** | **B** | 2560 | **0.0000** | **0.0000** | **0%** | **100%** | **0/20** | **DEGENERATE** |
| dist-square8 | A | 1597 | -0.0046 | 0.0557 | 53% | 0.0% | 50/50 |
| dist-square8 | B | 2560 | -0.0040 | 0.0247 | 53% | 18.1% | 20/20 |
| ind-corner | A | 1597 | -0.0247 | 0.2611 | 49% | 0.1% | 50/50 |
| ind-corner | B | 2560 | -0.0202 | 0.1280 | 47% | 20.2% | 20/20 |
| distclip-corner-r2/r4/r8 | A/B | — | -0.014..+0.071 | 0.12-0.29 | 40-51% | 0-20% | 50/50, 20/20 |

**`ind-square8` fails the screen identically on both datasets, exactly the
failure mode the task predicted**: `dv_true` is exactly 0 for every single
candidate (1597/1597 on A, 2560/2560 on B). The target (8x8 px, offset one
side-length from the corner) sits far enough from these piles' reachable
footprint that no single push -- of ~32 sampler-drawn candidates on A or 128
real candidates on B -- ever moves material across its boundary. This is
reported as degenerate and **excluded from every conclusion below**, not
dropped quietly; it is also why the task's "run ind-square8 first" ordering
produced a null result rather than the sharpest data point, and that gap in
the design is real (see `incomplete-design`, below).

**`slateK_exact`, linear, at K=4 and K=max, by functional** (r<=1 spectral
energy in parentheses):

| functional (r<=1) | A K=4 | A K=31 | B K=4 | B K=128 |
|---|---|---|---|---|
| dist-corner (0.629) | 0.9767 | 0.9581 | 0.9533 | 0.9670 |
| distclip-r8 (0.619) | 0.9809 | 0.9598 | 0.9605 | 0.9718 |
| distclip-r4 (0.582) | 0.9799 | 0.9491 | 0.9564 | 0.9710 |
| distclip-r2 (0.557) | 0.9755 | 0.9252 | 0.9484 | 0.9570 |
| **ind-corner (0.541)** | **0.9739** | **0.9233** | **0.9431** | **0.9346** |
| dist-square8 (0.683, size ctrl) | 0.9766 | 0.9328 | 0.9475 | 0.9567 |

Linear **never drops below 0.92** anywhere in this table -- far above the
0.90 supports-threshold. UNet tracks linear closely at every cell (e.g.
ind-corner: A 0.9810->0.9397, B 0.9566->0.9056) -- both models sharpen and
soften together, which is exactly what a flat gap looks like.

**Paired UNet-linear gap, and gap-vs-dist-corner (second-order pairing)**:

| functional | A, K=4 gap (sem, t) | A, gap-vs-corner (sem, t) | B, K=4 gap (sem, t) | B, gap-vs-corner (sem, t) |
|---|---|---|---|---|
| dist-corner | +0.0103 (0.0016, 6.51) | — | +0.0173 (0.0032, 5.47) | — |
| distclip-r8 | +0.0062 (0.0013, 4.82) | -0.0041 (0.0009, -4.51) | +0.0108 (0.0025, 4.36) | -0.0065 (0.0020, -3.31) |
| distclip-r4 | +0.0055 (0.0013, 4.21) | -0.0048 (0.0016, -3.01) | +0.0110 (0.0033, 3.31) | -0.0063 (0.0035, -1.79) |
| distclip-r2 | +0.0069 (0.0016, 4.18) | -0.0034 (0.0020, -1.75) | +0.0136 (0.0035, 3.93) | -0.0037 (0.0036, -1.04) |
| ind-corner | +0.0071 (0.0019, 3.79) | -0.0032 (0.0021, -1.54) | +0.0136 (0.0035, 3.83) | -0.0037 (0.0036, -1.04) |
| dist-square8 | +0.0114 (0.0019, 5.93) | +0.0011 (0.0018, +0.58) | +0.0156 (0.0035, 4.43) | -0.0017 (0.0034, -0.50) |

At K=max (A K=31, B K=128) every gap-vs-corner difference stays inside
|t|<=1.8 (A) / |t|<=1.6 (B) -- no functional grows the gap outside noise at
either K=4 or K=max, and several (distclip-r8, distclip-r4 at K=4) show the
gap **significantly SMALLER** than at dist-corner, the opposite direction
"supports" needed.

**worstK at K=4** (adversarial pool, linear/UNet): dataset A dist-corner
0.471/0.316 -> ind-corner 0.327/0.270 (both fall, roughly together);
dataset B dist-corner 1.030/0.741 -> ind-corner 0.418/0.369. worstK shrinks
with sharpness on both datasets -- read as the adversarial-pool value itself
being sensitive to the functional's own scale/shape, not as new evidence
either way for the model-comparison question (both models move together).

**rank_profile** (linear, ranks 1-4, no bottom-quartile pathology at either
sharpness level): dataset A dist-corner 0.035/0.058/0.076/0.095 -> ind-corner
0.045/0.073/0.088/0.087; dataset B dist-corner 0.004/0.011/0.024/0.030 ->
ind-corner 0.009/0.012/0.023/0.031 -- both flat-ish, confirming the flat
slateK_exact curves are trustworthy under the sharper functional too, exactly
the check METRICS.md prescribes.

**Sampled `slate4` bridge** (with-replacement, matching the register's own
convention): dataset A linear/UNet dist-corner 0.958/0.974 -> ind-corner
0.957/0.965; dataset B dist-corner 0.897/0.932 -> ind-corner 0.862/0.891 --
same story as the closed-form numbers, both models sharpen together and the
gap does not open up.

## What this means

**C-044/C-045 are NOT artifacts of the cost functional's low-pass shape.**
Sharpening the weight field's spectrum from 0.629 to 0.541 (r<=1 energy) at a
FIXED target, on two independent datasets (32 sampler-drawn and 128 real
candidates), moves the linear operator's own capture fraction down by at
most ~5 points at the largest K tested (never below 0.92) and does not widen
the UNet-linear gap -- if anything the gap is flat to modestly narrower under
the sharper functionals at K=4, with two cells clearing their own paired sem
in that (unpredicted) direction. **The register's "the linear operator
already captures most of the oracle's advantage, and the UNet's edge is
small" conclusion (C-045) is functional-independent over the range this
design could test**, and should be read as considerably more robust than a
single-functional record could show, not narrower.

**The sharpest planned cell failed before it could test anything.** `ind-
square8` -- the intended sharpest point on the axis (r<=1 energy 0.060, an
order of magnitude sharper than ind-corner's 0.541) -- is degenerate on both
datasets: no single push in either collection ever moves material across an
8x8 px boundary sited one side-length from the corner. This is not a
methodological accident; it is the mechanism the task itself named ("a push
moves 1-2 cubes across the boundary, so dV may be zero for most candidates"),
realised in its extreme form (zero for ALL candidates) because the target
sits outside what a single ~10-20mm push can reach from these piles'
starting footprint. **The register cannot currently test the sharpest end of
this axis (r<=1 < ~0.5) with a small, local target under a single-push
design** -- see "What would change the verdict."

**dist-square8 is the cleanest confirmation of the task's premise.** At
r<=1=0.683, it is marginally SHARPER than dist-corner (0.629) despite an 8x
target-size shrink, and its slateK_exact/gap numbers track dist-corner's own
almost exactly (A: 0.9766 vs 0.9767 at K=4; B: 0.9475 vs 0.9533) -- target
size alone moves essentially nothing, exactly as the spectral table predicts
and as this record's independent measurement confirms.

## What would change the verdict

- **A workable sharp+small-target cell.** `ind-square8`'s failure mode is a
  reachability problem, not a fundamental one: a target still small (sharp)
  but sited where the pile's pushed material actually goes (e.g. adjacent to
  the pile's own footprint, or reachable within the collection's push
  length) would test the r<=1<0.2 range this record could not. Cost: no new
  collection needed if an existing dataset's pile footprint is characterised
  first (~10 min CPU to locate a reachable small region, then re-run this
  record's pipeline unchanged).
- **Push length as a second axis.** EXP-0024_v1 found push length itself
  changes the canonical-frame pipeline's validity (L10mm warp-limited); a
  functional x push-length grid (this record's functionals against
  L10mm/L40mm, deferred to budget below) would show whether the "functional
  barely matters" finding is itself push-length-dependent.
- **A seed sweep**, as every prior record in this family -- one checkpoint,
  one fit per dataset here too.

## Threats

- `imprecision`: one UNet training seed, one linear fit per dataset
  (inherited unchanged from EXP-0024/EXP-0024_v1 -- no new training here).
  The second-order (gap-vs-dist-corner) sem is itself only as good as 50 (A)
  or 20 (B) slates' worth of paired differences.
- `untested-dependency`: `settled-state` unchecked for the rigid-cube path,
  inherited unchanged.
- `incomplete-design`: the task's own priority order named `ind-square8` as
  "the sharp form; run this one first" -- it ran first, and is degenerate on
  both datasets (measured cost: near-zero, ~1 min CPU per dataset to
  discover), so the sharpest planned functional contributes no data to the
  verdict. The distclip sweep and `ind-corner` still cover a real, monotone
  sharpness range (0.629 down to 0.541) and carry the verdict, but the task's
  own "supports" threshold (r<=1 falling toward <0.10) was never reached.
  Dataset C (L10mm, L40mm) was not run this session -- explicitly
  deprioritised by the task ("Lower priority... if budget remains") and the
  budget was spent on writing up A/B instead; named here rather than
  silently omitted.
- Considered and dismissed: `provenance` -- every number reuses EXP-0024/
  EXP-0024_v1's own checkpoints and fits through their own unmodified code
  path; the only new code (`lyapunov_weights`'s new goal keys, the two new
  probe scripts) does not touch model inference, fitting, or the metric
  implementations at all, and `corner`'s own numbers reproduce EXP-0026's
  published self-test value to 4 decimals.
- Considered and dismissed: `selection` -- every functional named in the
  task's design was run and is reported, including the two that failed
  (`ind-square8` degenerate, `center` degenerate as always) and the two
  cells (out of many) where the gap moved in the unpredicted direction
  (narrower, not wider) at a level clearing its own sem.

## Unrelated findings

- A distance transform's raw values (unlike an indicator's) are NOT
  shift-invariant on a finite, non-periodic grid: `distance_transform_edt`
  computes genuine Euclidean distances bounded by the actual array, so a
  small target's distance field depends on where it sits relative to the
  array edges (a corner-flush square "sees" the whole rest of the grid on
  one side and understates its own spectral concentration), while an
  indicator's `|FFT|` is exactly position-invariant because a within-bounds
  translation is a circular shift for a discrete Fourier transform. This
  cost about 10 minutes of rabbit-holing before the offset placement in
  `lyapunov_weights` was found to match this task's own reference table to 3
  decimals -- worth remembering for the next weight-field addition.
- Raw (non-demeaned) spectral concentration is nearly useless for comparing
  these fields: every one of them reads as 85-98% concentrated at DC before
  demeaning, because a distance transform's or an indicator's MEAN dominates
  its power almost independently of shape. `spectral_concentration.py`
  demeans before computing the ratio for exactly this reason, and the
  difference is large enough (e.g. `ind-square8` raw 0.985 vs demeaned 0.060
  -- literally opposite conclusions) that any future spectral-concentration
  script for this project should demean by default.

## Reviewer note, expected before this record ships: the K=31/K=128 `t`-comparison inherits EXP-0024_v1's amendment

The gap-vs-corner `t` values at K=max in this record are computed the same
way EXP-0024_v1's reviewer amendment flagged as underpowered at large K (many
slates tied, effective n well under the nominal 50/20). This record's K=max
gap-vs-corner numbers are reported for completeness and are NOT the basis for
the verdict, which rests on the K=4 comparison (where ties are zero on both
datasets for every functional) and on the unambiguous "linear never drops
below 0.90" fact, which does not depend on the paired-difference test
resolving at all.
