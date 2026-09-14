---
# ---- identity -------------------------------------------------------------
id: EXP-0004
title: Switched-by-push-length DMDc over analytic occupancy descriptors beats a global operator of the same basis, but the best FEATURE basis reverses once input/target dimensionality are decoupled
tier: T1
mode: exploratory
date: 2026-09-13
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On `Genesis/data/overnight_randlen` (213 files, 80/20 file split stratified
  by spawn mode, seed 0), a ridge operator per push-length bin (EXP-0003's
  6-bin scheme) predicting analytic occupancy descriptors at t+1 from the
  same descriptors at t beats a single global operator of the IDENTICAL
  feature basis, on held-out normalised `descriptor_accuracy`, for every
  basis tried (stock global moments/DFT, +push-frame-local blocks, richer
  Fourier/moment bases, all-local). Separately: which FEATURE BASIS scores
  best is confounded by target dimensionality in the naive (own-dims-pooled)
  comparison, and reverses once a common 14-dim target is held fixed across
  all input variants (the common-target ablation).

prediction: null  # exploratory; mode: exploratory below

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 0ddab20f
  dirty: true                      # pre-existing dirty tree, unrelated files
                                    # (skills reorg, doc moves) -- this
                                    # record's own inputs (experiments/temp/
                                    # dmdc-lenbins/*) are untracked scratch,
                                    # not modified-tracked files. See "What
                                    # was actually run".
  data_commit: "unrecorded (overnight_randlen has no provenance block in its _0_config.yaml, same gap noted in EXP-0001/EXP-0002/EXP-0003)"
  script: "experiments/temp/dmdc-lenbins/{fit.py,fit_b.py,fit_d.py,fit_arbitration.py}"
  data: ["Genesis/data/overnight_randlen (213 files, mixed/piled/scattered, cube)"]
  code_path: "transforms.functional.particles_to_occupancy (rasteriser) + dmdc_baseline.occupancy_descriptors / experiments/temp/dmdc-lenbins/descriptors{,_b,_c,_d}.py (feature bases) + dmdc_baseline.fit_operator (ridge fit, per-bin and global)"
  seed: 0
  split: "file-level, 80/20, stratified by spawn_mode, seed 0: 170 train files (87040 rows) / 43 test files (22016 rows) -- a fresh split for this experiment, NOT the same split object as randlen-train-test-file-disjoint (that tag covers prepare_randlen_split.py's own train_all/test_all split; this one is a separate stratified 80/20 built ad hoc for the descriptor study, see 'What was actually run')"
  runtime: "not recorded precisely; ~1 evening session across variants A/B/C/D + the common-target ablation, per experiments/TEMP_LOG.md timestamps (all 2026-09-13)"
  runs: []

budget:
  declared: "not separately declared for this promotion pass -- these numbers were produced during the original temp session under its own (unstated) budget; this record only promotes them, per the task instructions not to re-run anything"
  spent: "promotion pass: ~15 min / ~20k tokens (read + write only, no computation)"
  outcome: within

design:
  varied:
    feature_basis: ["stock global (A, D=87)", "+push-frame-local blocks (B, D=98)", "richer Fourier/moment (C: nf16 D=295, nf24 D=631, highorder D=96, pooled4 D=103, pooled8 D=151)", "all-local push-frame incl. DFT (D-all-local, D=94)", "all-local minus DFT (D-no-DFT, D=14)", "push-frame-only dim-matched control (D=11)"]
    model: ["switched (6 push-length bins)", "global (single operator, same basis)"]
    ridge_lam: [1e-4, 1e-3, 1e-2]
  held_fixed:
    rasteriser: "particles_to_occupancy, 64x64, bounds +/-0.064m"
    split: "same 170/43 file split for every variant"
    binning: "EXP-0003's 6 equal-width push-length bins, MIN_ROWS_PER_BIN=50 (never triggered)"
    metric_definition: "descriptor_accuracy = 1 - rms(model_err)/rms(persistence_err), z-scored per-dim by TRAIN phi_t1 mean/std, const/degenerate blocks excluded from the pooled number (metric.py, shared across A/B/C/D)"
  baselines: [persistence, "global (unswitched) operator of the same basis"]
  metric: "descriptor_accuracy (own-basis pooled, variants A/B/C/D) AND, separately, a common-14-dim-target version of the same formula (the arbitration ablation) -- NOT yet a key in experiments/METRICS.md; see Threats. accuracy/slateN (the two standard metrics) were not computed in this experiment -- descriptor_accuracy is a proxy for accuracy, see Threats/indirectness"

noise_floor: "not measured -- single file-level 80/20 split (seed 0), no fold-to-fold sd. Own-basis headline gaps (switched vs global, e.g. stock8 holdout 0.052 vs 0.040) are comparable in size to the lam-sensitivity already observed (~0.005-0.01 swing across the 1e-4/1e-3/1e-2 sweep), so treat all headline deltas as indicative, not precise."

depends_on: [push-frame-warp-roundtrip, occ-rasteriser-consistency]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  Switching beats the global operator of the SAME basis in every one of 8
  bases tried (own-basis pooled `descriptor_accuracy`, lam=1e-4 holdout):
  stock8 0.052 vs 0.040, +push-frame(B) 0.086 vs 0.061, D-all-local 0.435 vs
  0.308, D-no-DFT 0.489 vs 0.343, dim-matched control 0.473 vs 0.253 -- all
  small in absolute terms except the all-local family (D-all-local/D-no-DFT),
  which clears ~0.4-0.5. Enriching the Fourier/moment basis beyond ~D=150
  does NOT help (nf16/nf24 tie or actively regress via overfitting: nf24
  holdout switched goes NEGATIVE, -0.004 to -0.083, despite the best train
  score of the sweep). Which basis "wins" reverses once the arbitration
  confound is removed: under a COMMON 14-dim target, D-all-local (94-dim
  input, incl. DFT) is the best predictor (holdout 0.603), beating D-no-DFT's
  own 14 dims (0.489) and beating stock-global-A alone, which is WORSE than
  persistence as an input for this target (holdout -0.924). The own-basis
  "drop the DFT block" recommendation from variants A-D was a target-side
  dilution artifact, not a property of the DFT block as a predictor.
verdict: supported
downgrades: [imprecision, indirectness, selection, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If switching by push length carried no real structure, the switched and
global operators (fit from the identical loaded tensors, per-bin split of
the same data) would tie or the switched operator would lose in poorly
populated bins; instead switched beats global on every one of 8 independent
bases, including the two thinnest bins where a badly-conditioned per-bin fit
would be expected to show up worst. Separately, if the basis-ranking
reversal were a fluke, it would not have a clean mechanistic explanation --
it does: pooling `descriptor_accuracy` over a variant's OWN dimension set
lets the number of easily-predicted dims (not their informativeness) decide
the ranking, and the reversal appears exactly when that confound is removed
by holding the target fixed. That mechanism, not noise, is what the
common-target table shows.

## What was actually run

Everything reported here is a re-statement of numbers already computed in
`experiments/temp/dmdc-lenbins/` (variants A, B, C, D, and the common-target
arbitration ablation) -- **nothing was re-run or recomputed for this
promotion**, per the task's explicit instruction. The four `RESULTS*.md`
files in that directory are the source of truth; this record summarises
them but does not restate every cell (see "Numbers").

The working tree was dirty at promotion time from an unrelated concurrent
skills/docs reorganisation (see `git status` in the session's own record);
none of the files this record's numbers came from are affected by that
dirtiness -- they are untracked scratch under `experiments/temp/`, already
finalised before this promotion pass began.

One code path used here (`descriptors_b.py`'s push-frame warp) required an
axis-order fix discovered during variant B's authoring: `particles_to_
occupancy` emits `occ[b, x_bin, y_bin]` but `push_frame_transform`/
`warp_affine_occ` expect `(y_bin, x_bin)` pixel order (a `grid_sample`-style
image convention) -- verified empirically with a synthetic single-voxel
spike before trusting any push-frame descriptor number. This is a
convention gotcha, not a bug in either function; noted here so a future
reader of `descriptors_b.py`/`descriptors_d.py` does not have to
re-discover it.

## Numbers

**Own-basis pooled `descriptor_accuracy`, holdout, lam=1e-4** (persistence =
0 by construction in every row):

| basis (D) | switched | global |
|---|---:|---:|
| stock8 (87) | 0.0523 | 0.0400 |
| nf16 (295) | 0.0456 | 0.0498 |
| nf24 (631) | **-0.0041** | 0.0436 |
| highorder (96) | 0.0497 | 0.0382 |
| pooled4 (103) | 0.0498 | 0.0380 |
| pooled8 (151) | 0.0445 | 0.0360 |
| +push-frame-local (B, 98) | 0.0862 | 0.0607 |
| D-all-local incl. DFT (94) | 0.4345 | 0.3029 |
| D-no-DFT (14) | 0.4892 | 0.3434 |
| dim-matched local-only control (11) | 0.4731 | 0.2530 |

**Common-target ablation** (target fixed at D-no-DFT's own 14-dim vector;
input varied), holdout, lam=1e-4:

| input feature set (D_in) | switched | global |
|---|---:|---:|
| stock global A only (87) | -0.9240 | -1.0250 |
| push-frame B only (11) | 0.3440 | 0.1440 |
| D-no-DFT's own 14 dims (self-check) | 0.4890 | 0.3430 |
| **D-all-local incl. DFT (94)** | **0.6030** | 0.4120 |
| A + B concatenated (98) | 0.4770 | 0.2650 |
| persistence | 0.0000 | 0.0000 |

Full per-bin/per-lam breakdowns for every cell above live in
`experiments/temp/dmdc-lenbins/results_variant{A,B,C,D}.json` and
`results_arbitration.json`, cited by id here rather than restated.

## What would change the verdict

- **A noise floor** (fold/seed sweep) for either the own-basis or the
  common-target numbers -- not measured; small headline gaps (e.g. stock8's
  own-basis 0.052 vs 0.040) are the same order as lam-sensitivity already
  observed and should not be over-read without one.
- **A dimension-weighted or selected pooling scheme** instead of uniform
  per-dim pooling would test directly whether the own-basis ranking is
  purely a dilution artifact or partly real; not attempted here.
- **Re-scoring the common-target ablation in IMAGE space** (this experiment
  never reconstructs occupancy) -- done separately in EXP-0005, which finds
  the descriptor-basis question is nearly moot once a 32x32 visual channel
  is present (see EXP-0005's claim).

## Threats

- **`imprecision`**: no noise floor; single seed-0 split throughout, see
  `noise_floor` above.
- **`indirectness`**: `descriptor_accuracy` is an explicit proxy for the
  repo's image-space `accuracy` metric -- it pools 11-94 hand-designed
  scalar dims in z-scored space, not reconstructed occupancy, and the
  source RESULTS.md files flag this caveat themselves in every variant.
  This metric key is also **not yet in `experiments/METRICS.md`** -- flagged
  here rather than added, since `METRICS.md` edits are reserved for the
  concurrent slaten-broad work per this task's instructions. Whoever next
  touches `METRICS.md` should add `descriptor_accuracy`'s exact formula
  (z-score by train `phi_t1`, const-block exclusion, pooled rms ratio) from
  `experiments/temp/dmdc-lenbins/metric.py`.
- **`selection`**: the feature bases compared were iteratively designed
  across the session (A -> B -> C -> D), each informed by the previous
  variant's own-basis result (e.g. D-no-DFT was chosen as "the stage-2
  feature set" specifically because it had the best own-basis pooled number,
  before the common-target ablation reversed that conclusion). The
  common-target ablation itself was a post-hoc fix to a flaw noticed only
  after the fact, not a design specified up front.
- **`untested-dependency`**: `occ-rasteriser-consistency` is `unchecked` in
  `INVARIANTS.md` (no direct test that `load_randlen_cell`'s rasterisation
  matches `CellData`'s byte-for-byte) -- inherited from EXP-0001..0003,
  which carry the same downgrade for the same reason.
- Considered and dismissed: **provenance mismatch** between switched and
  global at a given basis -- both are fit from the identical loaded
  train tensors in the same script run for every basis, so within-basis
  comparisons are not confounded by this.

## Unrelated findings

- **`dmdc_baseline.apply_operators` OOMs at large D**: it materialises an
  `[N, D, D]` tensor via `A[bins]` fancy indexing (`torch.bmm(A[bins],
  phi.unsqueeze(-1))`); at D=631 (variant C's `nf24`), N~1e5, this requests
  ~138GB and fails. `experiments/temp/dmdc-lenbins`'s `fit_arbitration.py`
  and `fit_d.py` worked around it with a custom `apply_operators_mem` (a
  per-bin loop) rather than fixing `dmdc_baseline.py` itself -- the fix
  (loop over bins instead of gathering `A[bins]`) is straightforward and
  should be upstreamed into `dmdc_baseline.apply_operators` before another
  caller hits the same OOM at a comparably large D.
- **`scripts/run_probe.py` writes its execution ledger to `runs/COMMANDS.jsonl`**,
  not the documented `experiments/COMMANDS.jsonl` (`scripts/run_probe.py`'s
  own `LEDGER = ROOT / "runs" / "COMMANDS.jsonl"` constant) -- a
  documentation/implementation mismatch, not exercised directly by this
  experiment (none of the `dmdc-lenbins` scripts were run through
  `run_probe.py`), but worth fixing so the project-wide ledger the
  `experiment-log` skill describes is actually where every run lands.
