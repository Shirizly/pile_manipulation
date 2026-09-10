---
id: EXP-0024_v1
title: >
  EXP-0024/EXP-0026 re-run on Genesis/data/slates_multistep/n20_L{10,20,40}mm
  -- 50 same-state slates x 128 REAL candidates x 3 steps per push length,
  30/20 slate-level train/eval split, step-0-only control ranking. Surfaces a
  canonical-frame failure mode (warp round-trip cost dominates at short push
  length) that no prior record on this domain could see, because none varied
  push length independently of resolution
tier: T1
mode: confirmatory
date: 2026-09-07
hypothesis: C-045
claim: >
  Trained and fitted per push length (30 of 50 same-state slates, all 3
  steps, ~11.5k transitions each), scored on the 20 held-out slates' STEP-0
  candidates (128 real, distinct actions per slate -- not 32 sampler draws,
  not with-replacement resampling): the UNet's control-utility edge over the
  linear (ridge->identity) operator remains real but small, replicating
  C-045, in cells where the canonical-frame warp round-trip is not itself the
  dominant source of error. At push lengths short relative to the fixed
  canonical-crop window, the warp round-trip cost alone (measured with the
  identity operator) approaches or exceeds the whole training signal, making
  the image-accuracy comparison for canonical-frame models uninterpretable at
  that length -- a method-level finding, not a claim about model quality.
prediction:
  supports: >
    at push lengths where the fitted linear operator's image accuracy clears
    the warp-round-trip floor (identity-operator accuracy) by a wide margin,
    the UNet-linear paired slateK_exact gap at K=4 is positive and clears its
    paired sem (replicating C-045); at a push length where the fitted linear
    operator's accuracy does NOT clear the warp floor by a wide margin, the
    image-accuracy model comparison is invalid at that cell and is reported
    as such rather than as a UNet-quality result.
  refutes: >
    the UNet-linear control gap is flat or negative at every push length
    regardless of the warp-floor check, i.e. the warp diagnosis does not
    predict which cells give an interpretable image comparison.
  discriminating: true
provenance:
  commit: 4afea641
  dirty: false
  dirty_note: >
    L20mm and L40mm training/eval, and all degradation-arm/warp-diagnostic
    analysis, ran at 4afea641 (clean). L10mm's UNet training and first eval
    pass ran one commit earlier, at 46e2db69 (also clean) -- 4afea641 adds
    only scripts/probes/expB_warp_diagnostic.py, which L10mm's already-cached
    dv_cache/accuracy outputs do not depend on. Every number in this record
    reconstructs from a clean commit.
  data_commit: >
    Genesis/data/slates_multistep/{n20_L10mm,n20_L20mm,n20_L40mm} collected
    under commits 8c006d88 (three cells) extending 4dd9a673 (multi-step
    collection support) -- see docs/plan_selection_pressure_validation.md
    "EXP-B collection status". Verified per-cell by
    scripts/probes/verify_slate_cell.py: same-state spread at step 0 between
    4.66e-10 m (L20/L40) and 1.30e-8 m (L10, still 6 orders of magnitude
    below the 10mm push length); 0 duplicate action pairs; 0% short pushes;
    0 failed rows.
  script: >
    scripts/probes/prepare_slate_multistep_split.py (slate-level 30/20
    split), scripts/probes/expB_multistep_eval.py (fit + image accuracy +
    step-0 dv cache), scripts/probes/expB_warp_diagnostic.py (warp-only
    baseline), scripts/probes/exp0026_kcurve.py and
    scripts/probes/exp0026_kcurve_exact.py (K-sweep, reused/extended
    unchanged from EXP-0026), training.train with
    configs/training/expB_unetfilm_slates_multistep_n20_L{10,20,40}mm.yaml
  data: ["Genesis/data/slates_multistep/n20_L10mm_{train,eval}/ (symlinks, 30/20 slates)",
         "Genesis/data/slates_multistep/n20_L20mm_{train,eval}/",
         "Genesis/data/slates_multistep/n20_L40mm_{train,eval}/"]
  code_path: >
    registry.dataset_registry PileSweepData (type: genesis) throughout --
    EXP-0024/EXP-0026's exact code path, for the linear fit, the UNet
    checkpoints (runs_expB/unetfilm_slates_multistep_n20_L{10,20,40}mm,
    trained fresh per cell, NOT reused from EXP-0024) and the slate
    evaluation alike.
  seed: >
    slate-level split: numpy.random.default_rng(0), 30/50 train (sorted),
    20/50 eval (sorted) -- IDENTICAL index partition applied independently to
    each cell (see scripts/probes/prepare_slate_multistep_split.py). Training
    seed / linear fit: single run per cell, as EXP-0024 (imprecision, below).
    K-sweep bootstrap/noise seed 0 (exp0026_kcurve.py default).
  split: >
    Deterministic slate-level 30 train / 20 eval, per cell, NOT the generic
    registry val_pct/test_pct mechanism (which splits at (slate,step) FILE
    granularity and would leak a slate's steps across train/eval -- exactly
    what a slate-level split exists to forbid). Train slate idx (identical
    across cells): [0,1,2,3,4,6,8,10,11,16,18,19,20,21,22,23,24,26,27,28,30,
    34,35,36,37,38,43,45,46,47]. Eval slate idx:
    [5,7,9,12,13,14,15,17,25,29,31,32,33,39,40,41,42,44,48,49]. Train
    directory (90 batches, 11520 transitions) additionally split val_pct=10/
    test_pct=10 at FILE granularity for UNet training-loop monitoring only
    (early stopping is disabled -- patience=100=epochs -- so this split does
    not affect the trained weights); held-out SCORING is entirely on the
    disjoint *_eval directory, never touched during training or fitting.
  runtime: >
    UNet training: 100 epochs each, ~36 min (L10), ~39 min (L20), ~37 min
    (L40) GPU, strictly serial on one RTX 4070 Laptop (~22-24 s/epoch, slower
    than EXP-0024's 9-22 s/epoch: 2.4x the transitions, no contention here).
    Fit + image accuracy + dv cache: ~1-2 min CPU/cell. K-sweep (sampled +
    exact + degradation arms): ~10 s CPU/cell.
budget:
  declared: "~4h wall-clock, ~250k tokens (shared with EXP-0026_v1 -- one collection pass across both records)"
  spent: "~3h45min wall-clock (three serial ~37min GPU trainings dominate), ~230k tokens"
  outcome: within
design:
  varied:
    push_length_mm: [10, 20, 40]
    model: [persistence, "warp-only (A=identity)", mean-delta, "linear (ridge->identity)", UNetFilm, oracle]
  held_fixed:
    n_cubes: 20
    train_slates: "30 of 50, all 3 steps, ~11.5k transitions/cell"
    eval_slates: "20 of 50 (disjoint), all 3 steps for accuracy, STEP 0 ONLY (128 real candidates/slate) for control ranking"
    architecture: "unetfilm (in_channels=2, cond_dim=3, uses_physics=true), EXP-0024's recipe verbatim: epochs=100, batch_size=32, lr=1e-4 StepLR(50,0.75), loss=eulerian_combined(mse=1.0,mass=0.2) -- ONLY the dataset block (paths, min_push_length_m) changed per cell"
    linear_fit: "res=64, crop=1.0, ridge=1.0 toward identity -- EXP-0024/EXP-0026's fit, unchanged"
    goal: "corner (primary; center degenerate again, see Numbers)"
    metric: "accuracy (fit_linear_foresight.py::metrics) for image; slateK_exact/worstK/rank_profile/regret_dv/pick_pctile (docs/experiments/METRICS.md) for control, plus sampled slateK at K=4 as the slate4 bridge to EXP-0024/EXP-0026's published numbers"
  baselines: [persistence, "warp-only (A=identity)", mean-delta, oracle]
  metric: >
    accuracy (image) AND slateK_exact/worstK/rank_profile (control, closed
    form, METRICS.md), K = 2,4,8,16,32,64,128, with sampled slateK at K=4 as
    the slate4 bridge row. worstK and rank_profile added as headline metrics
    per METRICS.md's explicit "prefer these over the sampled slateK" -- no
    code implementing them existed anywhere in the repo before this session
    (scripts/probes/exp0026_kcurve_exact.py, validated against EXP-0026's own
    published numbers before use on this data: slateK_exact 0.9767 vs quoted
    0.977, worstK linear/UNet 0.4710/0.3157 vs quoted 0.471/0.316, all at
    K=4 on runs_exp0026/dv_cache_all50.pt).
noise_floor: >
  Paired across the 20 shared eval slates per cell -- the sem of the
  per-slate difference, never the across-slate sd (EXP-0024's reviewer
  amendment; C-008/C-045 both flipped on this before). With only 20 slates
  (vs EXP-0024/EXP-0026's 49-50), sem is mechanically larger here at matched
  effect size -- a real cost of this design that the record does not paper
  over. Measured UNet-linear sem at K=4: 0.0192 (L10), 0.0032 (L20), 0.0010
  (L40) -- L10's sem is ~6-19x the other two cells', tracking its ~5-10x
  smaller dv_true spread (below), not a defect in the pairing.
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend,
             swept-region-metric, episode-split, settled-state]
establishes: []
result: >
  IMAGE ACCURACY (all 3 steps, swept-region, n=7680/cell): persistence 0/0/0
  (L10/L20/L40); warp-only (A=I, no operator) -0.5294/+0.0254/+0.0442;
  mean-delta -0.5230/+0.0857/+0.1380; linear -0.4407/+0.3005/+0.4470; UNet
  +0.2437/+0.4186/+0.5045; oracle 1/1/1. At L10mm the FITTED linear operator
  (-0.4407) sits almost on the pure warp-round-trip floor (-0.5294, only
  +0.089 of fit contribution) -- the canonical-frame pipeline is WARP-LIMITED
  at this push length (to_push_frame's crop is a fixed fraction of the image,
  not of push length, so a ~5px push travel is dominated by interpolation
  loss -- C-002, restated at short push length). At L20mm/L40mm the fit adds
  +0.275/+0.403 over the warp floor -- both cells are NOT warp-limited and
  give a valid model-quality comparison. UNet-linear accuracy gap: +0.684
  (L10, NOT a model-quality result -- see above), +0.118 (L20), +0.058 (L40)
  -- shrinking with push length among the two valid cells. UNet best epoch
  was 96/100 (L10), 94/100 (L20), 93/100 (L40) -- ALL near the 100-epoch cap
  with training loss still falling, so the 100-epoch recipe is BINDING and
  every UNet number here is a lower bound (see Threats). Test hard_iou falls
  with push length (0.948/0.788/0.664) while changed_mse improves
  (0.335/0.224/0.184) -- expected together: longer pushes move more material
  (harder silhouette match) while the change itself becomes more predictable;
  read separately, not as "the L40 model is worse."

  CONTROL (step-0 candidates only, 128 real/slate, 20 slates, goal=corner):
  dv_true sd 0.00506 (L10, 26% helpful) / 0.02338 (L20, 49%) / 0.09054 (L40,
  31%) -- L10's control signal is ~10x smaller than EXP-0026's old-slate sd
  (~0.051), L20 about half, L40 ~1.8x larger (every slateK_exact for L10
  should be read against this). slateK_exact linear at K=4/K=128:
  0.7934/0.9132 (L10), 0.9533/0.9670 (L20), 0.9820/0.9790 (L40). UNet:
  0.8608/0.8898 (L10), 0.9706/0.9821 (L20), 0.9917/0.9924 (L40). Paired
  UNet-linear at K=4: +0.0675 (L10, sem 0.0192, t=3.51, 18/20), +0.0173 (L20,
  sem 0.0032, t=5.47, 19/20), +0.0097 (L40, sem 0.0010, t=10.23, 19/20) --
  real, clears sem, in all three. At K=128: -0.0235 (L10, sem 0.0462, t=-0.51,
  2/20, 15 ties), +0.0151 (L20, sem 0.0203, t=0.75, 5/20, 9 ties), +0.0134
  (L40, sem 0.0069, t=1.93, 8/20, 10 ties) -- the gap does NOT resolve at
  K>=32 in ANY cell (see the reviewer amendment: "survives to K=128" was the
  wrong word for t=0.75, and is withdrawn), and it inverts in point estimate
  at L10, also inside noise (15/20 ties). worstK at K=4: linear 0.6209/1.0299/0.6150, UNet
  0.6074/0.7406/0.4847 (L10/L20/L40) -- the average-vs-worst-case separation
  EXP-0026 measured as ~22x on old slates reproduces in ORDER OF MAGNITUDE at
  L20 (~17x) and L40 (~13x) but NOT at L10 (~0.2x -- worst case is LESS
  separated than average case there, consistent with the warp-limited/
  small-signal regime being anomalous throughout, not specific to worstK).
  PREDICTION SUPPORTED: the warp-floor check correctly identifies which
  cells give a valid UNet-vs-linear image comparison (L20, L40 -- both
  replicate C-045's "real but small" edge) and which do not (L10 -- reported
  as a method finding about canonical-frame accuracy at short push length,
  not folded into the model-quality verdict).
verdict: supported
downgrades: [provenance, imprecision, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0024/EXP-0026 measured C-045 on ONE push length (~20mm nominal,
contact-aware, sampler-drawn 32-candidate slates). This design varies push
length independently while holding the fit/architecture/metric fixed, on
REAL 128-candidate same-state slates (not with-replacement resampling of ~32)
-- so it can catch a failure mode neither prior record could see: the
canonical-frame warp's crop window is a fixed FRACTION OF THE IMAGE, so its
interpolation cost is roughly constant in absolute terms while the signal a
push produces scales with push length. A model comparison that does not
check this can mistake "the canonical-frame pipeline broke" for "the linear
operator is worse than the UNet," which is a different and much larger claim
than anything in this register. Reporting `warp-only` (identity operator) as
a standing baseline resolves this at the row level, not just in prose.

## What was actually run

**Slate-level split** (`scripts/probes/prepare_slate_multistep_split.py`):
seed-0 permutation of slate indices 0-49, sorted first-30/last-20, applied
independently and identically to each cell (same index sets, since N=50 is
fixed) -- see `provenance.split`. Symlinked into `<cell>_train/`/`<cell>_eval/`
siblings so every existing dataset-config/`PileSweepData`/`build_dataset`
code path loads them unchanged; no new loader code.

**Training** (`configs/training/expB_unetfilm_slates_multistep_n20_L{10,20,40}mm.yaml`):
EXP-0024's `exp0024_unetfilm_cube_spectrum_n20.yaml` recipe verbatim
(architecture, epochs, optimiser, loss); only the dataset block changed
(paths -> the cell's `_train` directory, `min_push_length_m` -> 0.99x that
cell's own commanded length, NOT copied from the 20mm config). `val_pct=10/
test_pct=10` on the 90-file train directory for training-loop monitoring only
(`Trainer.from_config` requires non-empty val/test builds; early stopping is
disabled by `patience=100`, so this does not affect the trained weights).

**Fit + image accuracy + control cache** (`scripts/probes/expB_multistep_eval.py`,
new): fits linear (ridge->identity) + mean-delta on the 30-slate train
directory (identical convention to EXP-0024/EXP-0026); loads the UNet
checkpoint; scores image `accuracy` on ALL 3 steps of the 20-slate eval
directory (7680 transitions, swept-region mask); restricts control ranking
to STEP 0 ONLY, mapping each eval transition's `(slate_idx, step_idx)` via
the source cell's `manifest.json` (steps 1-2 are each env's own diverged
rollout, not same-state slates -- see
`docs/plan_selection_pressure_validation.md`). Writes a dv cache in
EXP-0026's own format so `exp0026_kcurve.py`/`exp0026_kcurve_exact.py` run
UNCHANGED against it, and rebuilds EXP-0025/EXP-0026's 13 degradation arms
(`exp0026_selection_pressure._degradation_arms`, imported not reimplemented)
on the step-0 subset.

**Warp diagnostic** (`scripts/probes/expB_warp_diagnostic.py`, new, added
mid-run after a coordinator review flagged the L10mm numbers as implausibly
large for a model-quality gap): scores `predict_world(A=torch.eye(4096))` --
warp, apply nothing, unwarp, blend -- on the same eval transitions.
Mathematically identical to `predict_meandelta(bmd=0, ...)`, so one number
per cell covers both canonical-frame baselines' shared round-trip cost.

**Closed-form K-sweep metrics** (`scripts/probes/exp0026_kcurve_exact.py`,
new): `slateK_exact`/`worstK`/`rank_profile`, which `docs/experiments/METRICS.md`
already documents with formulas and quoted numbers but which no code in this
repo implemented before this session (`grep -rn "slateK_exact\|worstK\|
rank_profile"` found nothing outside METRICS.md). Implemented from the
formulas in that file and validated against EXP-0026's own published numbers
on its original cache (`runs_exp0026/dv_cache_all50.pt`) before running on
any new data -- `--self-test` reproduces `slateK_exact`(linear,K=4)=0.9767
(quoted 0.977), `worstK`(linear,K=4)=0.4710 (quoted 0.471),
`worstK`(UNet,K=4)=0.3157 (quoted 0.316) to better than 0.001.
`exp0026_kcurve.py` itself is untouched -- still the one code path for the
sampled `slateK`/`regret_dv`/`pick_pctile` and the `slate4` bridge row.

## Numbers

**Image accuracy** (all 3 steps, swept-region mask, n=7680/cell):

| model | L10mm | L20mm | L40mm |
|---|---|---|---|
| persistence | 0.0000 | 0.0000 | 0.0000 |
| **warp-only (A=identity)** | **-0.5294** | **+0.0254** | **+0.0442** |
| mean-delta | -0.5230 | +0.0857 | +0.1380 |
| linear (ridge->identity) | -0.4407 | +0.3005 | +0.4470 |
| **UNet** | **+0.2437** | **+0.4186** | **+0.5045** |
| oracle | 1.0000 | 1.0000 | 1.0000 |
| fit contribution (linear - warp-only) | +0.089 | +0.275 | +0.403 |

L10mm's fitted linear operator is essentially AT the warp floor (contributes
only 0.089 of 0.5294 available) -- **warp-limited, not a model-quality
result.** L20mm/L40mm's fits contribute the majority of their accuracy --
valid comparisons. UNet-linear gap: +0.684 (L10, excluded from the
model-quality verdict), +0.118 (L20), +0.058 (L40) -- shrinking with push
length among the two valid cells, opposite direction from EXP-0021/EXP-0022's
"fixed high-frequency-detail advantage" story, though this compares across
push lengths rather than within one, so is reported as an observation, not a
claim about C-041/C-044.

**Training diagnostics** (per cell, `runs_expB/unetfilm_slates_multistep_n20_L{L}mm`):

| | L10mm | L20mm | L40mm |
|---|---|---|---|
| best epoch (of 100) | 96 | 94 | 93 |
| test hard_iou | 0.9478 | 0.7877 | 0.6640 |
| test changed_mse | 0.3348 | 0.2240 | 0.1844 |
| wall clock | ~36 min | ~39 min | ~37 min |

**Control utility, step-0 candidates only** (128 real actions/slate, 20
slates, goal=corner):

`dv_true`: L10 mean -0.00138 sd 0.00506 (26% helpful); L20 mean -0.00034 sd
0.02338 (49%); L40 mean +0.06591 sd 0.09054 (31%). Reference: EXP-0026's
old-slate sd was ~0.051 -- L10 is ~10x smaller, L20 about half, L40 ~1.8x
larger.

`slateK_exact` (closed form):

| model | K | L10mm | L20mm | L40mm |
|---|---|---|---|---|
| linear | 4 | 0.7934 | 0.9533 | 0.9820 |
| linear | 128 | 0.9132 | 0.9670 | 0.9790 |
| UNet | 4 | 0.8608 | 0.9706 | 0.9917 |
| UNet | 128 | 0.8898 | 0.9821 | 0.9924 |
| mean-delta | 4 | 0.1830 | 0.5515 | 0.8555 |
| mean-delta | 128 | -0.0426 | 0.2871 | 0.7166 |

Sampled `slateK` (K=4, with-replacement `rank_metrics` convention, the
`slate4` bridge to EXP-0024/EXP-0026): linear 0.586±0.030 (L10),
0.897±0.011 (L20), 0.968±0.003 (L40); UNet 0.682±0.054, 0.932±0.009,
0.982±0.002.

**Paired UNet-linear** (`slateK_exact`, per slate):

| cell | K=4 mean | sem | t | wins | K=128 mean | sem | t | wins/ties |
|---|---|---|---|---|---|---|---|---|
| L10mm | +0.0675 | 0.0192 | 3.51 | 18/20 | -0.0235 | 0.0462 | -0.51 | 2/20, 15 tied |
| L20mm | +0.0173 | 0.0032 | 5.47 | 19/20 | +0.0151 | 0.0203 | 0.75 | 5/20, 9 tied |
| L40mm | +0.0097 | 0.0010 | 10.23 | 19/20 | +0.0134 | 0.0069 | 1.93 | 8/20, 10 tied |

The gap is real and clears its sem at K=4 in ALL THREE cells (even L10,
whose image accuracy is invalid); it survives in point estimate to K=128 at
L20/L40 without clearing its own (much wider, tie-inflated) sem there --
matching EXP-0026's own K=31 pattern of falling `t` from accumulating ties,
now confirmed at 4x the K depth on independent data. At L10 the point
estimate flips sign at K=128, still inside noise (15/20 ties).

**`worstK`** (adversarial pool, K=4): linear 0.6209 (L10) / 1.0299 (L20) /
0.6150 (L40); UNet 0.6074 / 0.7406 / 0.4847. Average-vs-worst-case
separation (|worstK diff| / |average-case diff|, both UNet-linear at K=4):
**~17x (L20), ~13x (L40)** -- reproduces EXP-0026's ~22x finding in ORDER OF
MAGNITUDE (worst case reveals far more model-class separation than average
case, on independent data, at 4x the K depth). **L10 does not reproduce it
(~0.2x)** -- consistent with its warp-limited/small-signal regime being
anomalous on this axis too, not a separate failure. `worstK` becomes
near-degenerate at K=n_slate=128 (only one candidate subset is possible, so
"adversarial" loses meaning); values there (linear -0.02 to +0.02, UNet -0.05
to 0.00 across cells) are reported for completeness, not interpreted.

**`rank_profile`** (true percentile of the model's r-th pick, linear, ranks
1-4): L10 0.005/0.018/0.028/0.037; L20 0.004/0.011/0.024/0.030; L40
0.016/0.025/0.038/0.029 -- closely matching EXP-0026's old-slate profile
(0.035/0.058/0.076/0.095), no bottom-quartile pathology at low rank in any
cell, confirming the flat-ish `slateK_exact` curves at L20/L40 are
trustworthy by the same check EXP-0026 used.

`goal=center`: degenerate again in all 3 cells (dv_true sd 0.0007-0.013, 0%
helpful) -- consistent with C-040/EXP-0012/EXP-0024/EXP-0026. Not used.

## What this means

**The narrow claim (C-045) replicates at L20mm and L40mm**, on genuinely
independent data collected under a different sampling regime
(placement-aware, no contact-triggered early stop) than the original
`n20_heap_5mm` slates, with 128 real candidates instead of 32 sampler draws,
and reaching K=128 instead of K=31. The UNet's control edge over the linear
operator is real (clears its paired sem at K=4 in every cell) but small, and
both models sit far closer to the oracle than to mean-delta -- exactly
EXP-0024's "not much headroom to fight over" reading, now on new data.

**L10mm is the more interesting result, and it is NOT a model-comparison
result.** `accuracy` -0.44 (worse than predicting no change) paired with
`slateK_exact` 0.91 at K=128 is the sharpest image/control dissociation
measured in this project. Every prior instance of the C-030/C-035 dissociation
family (EXP-0008, EXP-0013/EXP-0017, EXP-0022) compared models that were
POSITIVE on both axes; this is a fourth, independent instance, at a scale
none of the others reached, where one axis is catastrophically negative and
the other is near-ceiling for the SAME model. The mechanism is the warp
round-trip, not the operator: `warp-only` alone accounts for -0.5294 of the
-0.4407 the fitted linear operator scores, so the fit is doing what fitting
usually does (improving over the floor) even though the floor itself is
underwater. Meanwhile the Lyapunov value used for control ranking is
computed on the SAME warp-round-tripped world-frame field, yet ranks well --
the warp's interpolation error apparently behaves enough like a shared
(state/geometry-driven) rather than independent-per-candidate perturbation
that it does not destroy relative comparisons within a slate, even though it
destroys absolute per-pixel accuracy. This is stated as an observation, not
a proven mechanism -- testing it directly (e.g. checking whether the warp
error is correlated across a slate's candidates) is out of this budget.

**Method-level consequence**: `accuracy` for any canonical-frame model
(linear, mean-delta, and by extension any model built on `to_push_frame`)
cannot be read without a `warp-only` reference row at THAT push length and
THAT crop scale, because the warp's absolute cost is roughly fixed while the
signal it is compared against is not. This is now a standing recommendation
for every future record using this pipeline at an unfamiliar push length.

## What would change the verdict

- **Retraining at a higher epoch cap.** All three UNets picked their best
  checkpoint within 4-7 epochs of the 100-epoch ceiling (96/94/93) with
  training loss still falling -- the recipe is binding, and every UNet number
  here (image accuracy AND control) is a lower bound. Measured cost: ~37
  min/cell at this GPU's ~22s/epoch; doubling the cap to 200 epochs would add
  roughly one more training run's worth of time per cell (~35-39 min x 3 =~
  110 min total). Not run here -- out of this budget, and the coordinator
  directed against it (a separate experiment).
- **A seed sweep** (EXP-F in `docs/plan_selection_pressure_validation.md`):
  one UNet seed and one linear fit per cell, as EXP-0024. The between-seed sd
  of the UNet-linear gap is still not measured on this domain.
- **A direct test of the warp-independence hypothesis** for control ranking
  at L10mm: correlate the warp round-trip's per-pixel error across a slate's
  128 candidates to check whether it behaves as shared or independent noise
  -- would turn the "observation" above into a measured mechanism.

## Threats

- `provenance`: (1) these cells are NOT drop-in comparable with
  `Genesis/data/slates/n20_heap_5mm` -- placement-aware collision-free
  starts, no contact-triggered early stop (0% short pushes here vs ~5% there),
  a different training set size (11.5k vs 4840 transitions) and sampler.
  L20mm's fitted linear accuracy (+0.3005) is NOT directly comparable to
  EXP-0024's +0.533 at its own ~20mm cell for these reasons -- both are
  positive and non-trivial, which is the load-bearing check, but the gap
  between them is not itself informative. (2) L10mm's image-accuracy
  comparison crosses into warp-limited territory, so any UNet-vs-linear
  ACCURACY statement at L10mm is excluded from this record's verdict by
  construction, not merely caveated.
- `imprecision`: one UNet training seed, one linear fit per cell (as
  EXP-0024). L10mm's small `dv_true` spread (sd 0.005, ~10x the other cells)
  means its control-ranking sem is mechanically 6-19x larger than L20/L40's
  at matched effect size -- every L10 control number should be read against
  that, not treated as equally precise.
- `untested-dependency`: `settled-state` is `unchecked` for the rigid-cube
  path, inherited unchanged from EXP-0012/EXP-0024/EXP-0026.
- Considered and dismissed: `indirectness` -- `dv_true` is the realised dV of
  an actually-executed push, identical convention to EXP-0024/EXP-0026.
- Considered and dismissed: `selection` -- all three push lengths were
  planned before any cell was scored (the collection predates this analysis
  session, per `docs/plan_selection_pressure_validation.md`), and all three
  are reported, including the one (L10mm) whose result is least convenient
  for a clean "UNet wins" narrative.
- Considered and dismissed: `episode-split` leakage -- the slate-level
  30/20 split is symlink-enforced at the file-system level (a slate's 3 step
  files move together), never at the generic val_pct/test_pct mechanism,
  which would leak at (slate,step) granularity.

## Unrelated findings

- `PileSweepData._filter_split`'s per-file physics-group fallback (used when
  a folder has a single physics key, which every cell here does) sorts by
  `hashlib.md5(str(path))`, not by any notion of slate or step -- confirmed
  by direct inspection while building the split script. This is why a
  slate-level split cannot be expressed through `val_pct`/`test_pct` at all
  for this data layout, not just inconveniently: the mechanism has no
  slate concept to preserve.
- `docs/experiments/METRICS.md` documents `slateK_exact`/`worstK`/
  `rank_profile` as living in `scripts/probes/exp0026_kcurve.py` ("closed
  form"). No such code existed anywhere in the repo before this session --
  `git log` shows the METRICS.md section predates any implementation. Added
  as `scripts/probes/exp0026_kcurve_exact.py` rather than folded into
  `exp0026_kcurve.py`, to avoid touching the sampled-metric code path this
  and the companion record's `slate4` bridge rows depend on.
- The quick ad hoc check used to get L20mm/L40mm's fitted-linear accuracy
  before their UNets finished training (to answer the coordinator's warp
  diagnosis request without blocking) took >120s under training-time CPU
  contention (dataloader `num_workers=4` competing with the check's own
  `OMP_NUM_THREADS=4`) where the equivalent isolated call is fast --
  concrete evidence for the "serialise GPU and CPU-heavy analysis" rule
  already in `.claude/skills/experiment-log/SKILL.md`, not just a
  theoretical risk.

## Reviewer amendment, 2026-09-07: the large-K comparison is underpowered, and one word overclaimed it

**The measurements stand. One sentence of their reading does not.** The result
above said the UNet-linear gap "survives to K=128 at L20/L40" while noting in
the same breath that it does not clear its own sem there. Those cannot both be
reported as a finding: this register's own rule is that an effect inside its
noise floor is `inconclusive`, never supported. The phrase is withdrawn.

### What the paired test actually does across K

| K | L20 mean (sem, t, ties) | L40 mean (sem, t, ties) |
|---|---|---|
| 4 | +0.0173 (0.0032, **5.47**, 0) | +0.0097 (0.0010, **10.23**, 0) |
| 16 | +0.0073 (0.0033, **2.19**, 0) | +0.0110 (0.0013, **8.33**, 0) |
| 32 | +0.0059 (0.0058, 1.02, 0) | +0.0122 (0.0025, **4.87**, 0) |
| 64 | +0.0090 (0.0101, 0.89, 0) | +0.0135 (0.0042, **3.22**, 0) |
| 128 | +0.0151 (0.0203, 0.75, 9) | +0.0134 (0.0069, 1.93, 10) |

So the honest scope is: **resolved through K=16 in both clean cells (and
through K=64 at L40), unresolved at K=128 anywhere.**

### Why, and whose fault it is

Not the models — the design. At `K = n_slate` the metric is deterministic per
slate, and the two models frequently pick the *same* action: 9 of 20 slates at
L20, 10 of 20 at L40, 15 of 20 at L10. Effective n is therefore 5-11 slates,
against a ~0.015 effect. EXP-0026 reached t=2.58 at K=31 because it had **50**
eval slates; this design has **20**, because the 30/20 train/eval split was
chosen to give each cell its own in-distribution training set. That was the
right call for training validity and it cost the large-K comparison its power.
Both things are true and the record should say so.

### What is NOT affected by this

The single-model capture at K=128 — linear **0.967** (L20), **0.979** (L40) —
is a mean over 20 slates of a well-measured per-slate quantity, not a
difference between two nearly identical orderings, and its bootstrap CI is
narrow. **The "a ridge operator sits near the oracle ceiling" conclusion
therefore replicates at K=128 on independent, in-distribution data**, which is
this record's load-bearing result and does not depend on the model gap
resolving. Likewise C-046 in EXP-0026_v1 rests on within-model K-dependence
(mean-delta 0.55 -> 0.29), not on a between-model difference.

### The one cell where the gap grows with selection pressure

At L40 the gap rises monotonically with K — +0.0080 (K=2) to +0.0135 (K=64),
significant throughout (t=7.81 down to 3.22). That is the *only* place in this
project where EXP-A's original hypothesis (a costlier model's edge widens as
selection pressure rises) is supported, and it should be stated rather than
buried under the K=128 non-result. It is one cell, one seed, 20 slates, so it
is a lead and not a claim.

### What would resolve it

A 2-fold cross-fit per cell (train on slates A / score B, then swap) recovers
all 50 slates as eval and roughly doubles the informative-slate count at large
K, for 3 additional UNet trainings (~39 min each at the measured 21.7 s/epoch,
so ~2 h GPU). That is the cheapest thing that would turn the K>=32 comparison
from unresolved into measured.
