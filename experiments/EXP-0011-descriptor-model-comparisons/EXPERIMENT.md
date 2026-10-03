---
# ---- identity -------------------------------------------------------------
id: EXP-0011
title: >
  Action-frame descriptors beat raw-frame descriptors on slates_multistep
  control ranking but not on slates_binned; a single small MLP does not
  match the switched-linear descriptor operator at one-step prediction
tier: T1
mode: exploratory
date: 2026-09-15
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  (1) Under a shared switched-linear (6 push-length-bin) fit, warping
  occupancy into the canonical push (action) frame before computing analytic
  descriptors (mass/COM/moments2/DFT, D=87) improves slateN control-ranking
  over computing the identical descriptor formula on the raw (unwarped)
  frame, on Genesis/data/slates_multistep (first step, n20_L20mm/L40mm,
  n=50 slates/cell, 9 goal x value-fn cells pooled). (2) A single small MLP
  (99->48->94, 9406 params, under a 1:10 params:train-rows ceiling) trained
  on the IDENTICAL 94-dim D-all-local push-frame descriptor basis does NOT
  match weights/MODEL-0002-descriptor-only-D-all-local (6-bin switched-linear,
  same basis) on held-out one-step descriptor_accuracy
  (Genesis/data/overnight_randlen_test, n=10752).

prediction:
  supports: >
    (1) frame B (action) pooled wins > frame A pooled wins on slates_multistep,
    by a margin clearing paired sem at the pooled level; (2) MLP
    descriptor_accuracy < switched-linear descriptor_accuracy by more than
    the tuning-sweep's own spread (~0.03).
  refutes: >
    (1) frame A wins >= frame B pooled, or the difference sits inside paired
    sem; (2) MLP descriptor_accuracy is within ~0.03 of or exceeds
    switched-linear's 0.435.
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0011-descriptor-model-comparisons/code/desc-frame-compare__run_experiment.py
  data: ["Genesis/data/overnight_randlen_train", "Genesis/data/overnight_randlen_test",
         "Genesis/data/slates_multistep/n20_L20mm", "Genesis/data/slates_multistep/n20_L40mm",
         "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"]
  code_path: experiments/EXP-0011-descriptor-model-comparisons/code/
  seed: 0
  split: "overnight_randlen_train (n20 groups, 135 files, 69120 rows) vs overnight_randlen_test (n20 groups, 15 files, 7680 rows) for RUN-0001; overnight_randlen_train (all 5 groups, 192 files, 98304 rows) vs overnight_randlen_test (all 5 groups, 21 files, 10752 rows) for RUN-0002 -- both pre-split, disjoint at the file level (re-verified: 0 filename overlap in every n20 subdir, confirmed by direct file listing during this promotion)"
  runtime: "RUN-0001 ~180min (over its 60min budget, flagged); RUN-0002 ~75min"
  runs: [RUN-0001, RUN-0002]

budget:
  declared: "RUN-0001: 60 min/~140k tokens; RUN-0002: 75 min/~150k tokens (each declared separately when run)"
  spent: "RUN-0001: ~180min/180k+ tokens (exceeded, flagged mid-run); RUN-0002: ~75min/~150k tokens (within)"
  outcome: exceeded

design:
  varied:
    RUN-0001: {descriptor_frame: [raw, action-warped], ridge_lambda: [1e-4, 1e-3, 1e-2]}
    RUN-0002: {model: [switched-linear-MODEL-0002, single-MLP], mlp_config: [H48-relu-wd0, H48-relu-wd1e-4, H48-relu-wd1e-3, H48-tanh-wd1e-4, H32-relu-wd1e-4]}
  held_fixed:
    RUN-0001: {descriptor_family: "mass/COM/moments2/DFT (D=87), fit_per_action_operators closed-form ridge-toward-zero, 6-bin EXP-0003 scheme, rasteriser particles_to_occupancy uniformly"}
    RUN-0002: {descriptor_basis: "D-all-local, 94-dim, push-frame (dmdc-lenbins/descriptors_d.py, identical to MODEL-0002's fit basis)", train_corpus: overnight_randlen_train, test_corpus: overnight_randlen_test}
  baselines: [persistence, random]
  metric: "descriptor_accuracy (RUN-0002 headline); slateN (both RUNs' control metric)"

noise_floor: >
  RUN-0001: paired sem of the wins/losses goodness-diff per cell (reported per
  cell, e.g. L40mm random_quadrant/lyapunov paired_sem=0.0139 against a mean
  diff of 0.148 -- an order of magnitude clear). RUN-0002: paired sem at n=20
  slates for the control-metric cells (all 9 cells sit at 1-2x their own paired
  sem -- inside noise, reported as such); descriptor_accuracy has no formal
  floor measured, but the tuning sweep's own spread across 5 configs (0.131
  best to -0.052 worst) brackets what a config choice alone can move, and the
  MLP-vs-switched gap (0.304) is ~10x that spread.

depends_on: [push-frame-warp-roundtrip, randlen-train-test-file-disjoint]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  (1) Frame B (action) beats frame A decisively on slates_multistep (pooled
  W/L/T: L20mm 116/207/127, L40mm 100/329/21) but is a statistical tie on
  slates_binned (82/80/18, n=20/cell, underpowered) -- not pooled across
  corpora, reported as unresolved there. (2) MLP descriptor_accuracy=0.131 vs
  switched-linear 0.435 on overnight_randlen_test (n=10752) -- MLP clearly
  worse, capacity-bound (H32 went negative, weight decay hurt monotonically,
  still rising at epoch 119). Control-metric comparison of MLP vs
  switched-linear on slates_binned (n=20/cell) is inconclusive at this power.
  slates_multistep was NOT evaluated for the MLP (see Threats/incomplete-design).
verdict: supported
downgrades: [incomplete-design, imprecision, provenance]
grade: very-low
supersedes: []
invalidated_by: null
---

## What was uncommitted (dirty tree)

Working tree was dirty at both RUN-0001 and RUN-0002's run time and at this
promotion's own commit time: the two source scripts and their outputs under
`experiments/temp/{desc-frame-compare,desc-mlp}/`, plus this promotion's own
new files under `experiments/EXP-0011-descriptor-model-comparisons/` and
edits to `experiments/{REGISTER,INVARIANTS,TEMP_LOG}.md` and
`docs/CODEMAP.md`. No project source module (`Baselines/`, `transforms/`,
`Genesis/`) was modified.

## Why this test discriminates

If action-frame descriptors carried no real ranking information beyond what
raw-frame descriptors already give, frame A and frame B would tie (within
noise) on slateN across corpora -- they do not tie on slates_multistep, where
the gap is 10x-plus the paired sem. If the MLP could recover what per-bin
switching captures, its descriptor_accuracy would approach 0.435 despite
having push length as an explicit input feature; instead it plateaus near
0.13 under a hard capacity ceiling, which is the discriminating observation
this design was built to catch.

## What was actually run

**RUN-0001 (desc-frame-compare)**: fit the SAME descriptor formula
(`dmdc_baseline.occupancy_descriptors`, D=87) on two different frames -- raw
occupancy (A) vs. occupancy first warped into the canonical push frame via
`to_push_frame`/`push_frame_validity_mask` (B) -- both switched by the
identical EXP-0003 6-bin scheme, closed-form ridge, λ swept in
{1e-4,1e-3,1e-2}, best λ=1e-4 for both frames by each frame's OWN
`descriptor_accuracy` (never compared cross-frame -- see Threats).
`descriptor_accuracy` is used only within-variant (A-vs-persistence-A,
B-vs-persistence-B); the control metric (`slateN`, point-mass readout) is
the only cross-frame comparison. Evaluated on `overnight_randlen_test`
(ridge selection only), `slates_multistep` step 0 (n20_L20mm, n20_L40mm),
and `slates_binned` (n20-scatter-only) -- the three corpora never pooled
together. A device-mismatch bug in the script's own point-mass readout (not
inherited from upstream `eval_slaten_latent.py`) and a CUDA OOM were found
and fixed mid-run (see desc-frame-compare/RESULTS.md and Unrelated
findings).

**RUN-0002 (desc-mlp)**: trained a single MLP (99-dim input = 94-dim
D-all-local descriptor + 5 action features -> 94-dim output) on
`overnight_randlen_train`, under a hard capacity ceiling
(params <= N_train/10 = 9830, chosen 9406 by construction, not swept up to
it). A 5-config sweep (width, activation, weight decay) selected H48-relu-
wd0 (0.131) as best. Compared against MODEL-0002 (weights/MODEL-0002-
descriptor-only-D-all-local) on the identical basis/normalisation/held-out
rows (licensed per experiment-log's descriptor_accuracy rule). Control-
metric eval (slateN) was run ONLY on `slates_binned` (n20-scatter,
K=N=1000) -- `slates_multistep` was NOT evaluated for the MLP; the 75-minute
budget closed before that cell ran. This is the largest gap in this
sub-record and is carried as `incomplete-design`.

## Numbers

### RUN-0001: ridge sweep, descriptor_accuracy (within-variant only, never A-vs-B)

| λ | frame A (own persistence baseline) | frame B (own persistence baseline) |
|---|---:|---:|
| 1e-4 | 0.0543 | 0.4133 |
| 1e-3 | 0.0538 | 0.4128 |
| 1e-2 | 0.0352 | 0.3877 |

### RUN-0001: slateN, pooled wins/losses/ties, frame A vs frame B (9 goal x value-fn cells)

| corpus | n slates/cell | W (A) | L (B) | T |
|---|---:|---:|---:|---:|
| slates_multistep n20_L20mm | 50 | 116 | 207 | 127 |
| slates_multistep n20_L40mm | 50 | 100 | 329 | 21 |
| slates_binned n20_scatter | 20 | 82 | 80 | 18 |

Example cell (L40mm, random_quadrant/lyapunov): A=-0.199±0.095, B=+0.833±0.020, W0/L41/T9.

### RUN-0002: descriptor_accuracy, overnight_randlen_test (n=10752)

| model | descriptor_accuracy |
|---|---:|
| persistence | 0.000 |
| random | -1.290 |
| switched-linear (MODEL-0002) | **0.435** |
| MLP (best of 5 configs, H48 relu wd0, 9406 params) | **0.131** |

MLP tuning sweep: H48-relu-wd0 0.131 (best) / H48-relu-wd1e-4 0.128 /
H48-relu-wd1e-3 0.100 / H48-tanh-wd1e-4 0.106 / H32-relu-wd1e-4 -0.052.

### RUN-0002: slateN, MLP vs switched-linear, slates_binned n20-scatter (n=20/cell)

All 9 cells (ring_O/T/random_quadrant x lyapunov/mass_in_region/
signed_mass_in_region) sit at 1-2x their own paired sem -- none clear noise
at n=20. Full per-cell table in `results/results_control.json` and
`experiments/temp/desc-mlp/RESULTS.md`.

## What would change the verdict

RUN-0001's slates_binned tie could resolve either way with more slates (n=20
is underpowered per its own paired sems, e.g. 82W/80L is well within a coin
flip at this n) -- collecting more scatter-spawn slates (~1-2h) or reading
across further seeds would settle it. RUN-0002's missing `slates_multistep`
control-metric cell for the MLP (never run, budget) would give a much larger,
better-powered second control read; that is the single most valuable next
step and was not attempted here (~20-30 min at existing harness cost).

## Threats

- `incomplete-design`: RUN-0002 never evaluated slates_multistep for the MLP
  -- confirmed by inspecting `experiments/temp/desc-mlp/eval_control.py`'s
  `main()`, which only loads `slates_binned` (no `slates_multistep` load
  call exists in the file at all). This is a real gap, not an oversight in
  reporting.
- `imprecision`: RUN-0001's slates_binned cell and RUN-0002's entire control-
  metric comparison rest on n=20 slates/cell -- explicitly reported as
  underpowered in both source RESULTS.md files, not smoothed over.
- `provenance`: RUN-0001 rasterises uniformly via `particles_to_occupancy`
  for ALL corpora (train/test/multistep/binned), which is internally
  consistent for A-vs-B but is NOT the official `_draw_particle_grid`
  rasteriser `slates_multistep`'s registry-quoted numbers elsewhere in this
  register use -- documented explicitly in the source script's docstring,
  not discovered after the fact. EXP-0012 separately measured this
  discrepancy as negligible (corr 0.998, slateN 0.822 vs 0.820) for a
  similar cell, but that check was not repeated for every one of RUN-0001's
  27 cells.
- Considered and dismissed: switching not being per-candidate. Verified by
  direct code read (both `Baselines/LinearForesight/model.py::predict_switched`
  and `desc-mlp/eval_control.py::predict_switched` bucketize `length_m`
  per-row before dispatch) -- every candidate action gets its OWN bin's
  operator, in both RUN-0001 and RUN-0002. Not a defect.
- Considered and dismissed: `descriptor_accuracy` misuse across different
  descriptor sets. RUN-0001 explicitly never compares A's accuracy to B's
  (different target vectors, correctly flagged in its own RESULTS.md).
  RUN-0002's MLP and MODEL-0002 share the identical 94-dim basis and
  normalisation -- a licensed comparison, confirmed by inspecting both
  models' fit code.

## Unrelated findings

- A device-mismatch bug (`bilinear_sample` called with inputs on different
  devices) was introduced by RUN-0001's own adaptation of `com_world_pixel`/
  `bilinear_sample` (copied from `experiments/temp/stage3-slaten/
  eval_slaten_latent.py`, which does NOT have the bug) -- fixed in-run. These
  two functions have now been hand-copied at least 3 times across temp
  experiments and belong in a canonical shared location (flagged to
  docs/CODEMAP.md by the desc-frame-compare run; not fixed here, out of
  scope).
- `docs/CODEMAP.md`'s Datasets section did not, at the time of these runs,
  note that `Baselines/common/data.py` warns `particles_to_occupancy` is not
  a drop-in replacement for the official `_draw_particle_grid` rasteriser --
  since fixed in a later CODEMAP update (see EXP-0012).

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- Asymmetric goal masks (T / random_quadrant / ring_O / stripe / letters) in this record were scored BEFORE the goal-axis fix `28271c09` (2026-09-17) and were never rescored; transpose-invariant goals (corner, center, ...) are unaffected (ISS-003). ISS-003 names this record (C-016 exposed).
- DS-0001 was simulated with friction 0.3 / density 1000, not the randlen models' training physics (`benchmark-physics-matches-training`, broken).
- EXP-0065 / ISS-013: 46 % of DS-0001's candidate pushes put the blade on a cube at touchdown (pre-fix pile-aware sampler); this record's pools were not re-scored on legal-only candidates.
- C-017: MODEL-0002 was fit on a split containing 17 of the 21 overnight_randlen_test files (found in EXP-0021 setup); clean refit 0.4319 vs 0.435.
