---
# ---- identity -------------------------------------------------------------
id: EXP-0008
title: Widened slateN validation (L10mm excluded, 5 goal shapes x 3 value functions, + overnight holdout) -- switching beats global once L10mm is excluded, goal-dependence is shape x value-function not shape alone, and descriptor-only's control usefulness is corpus-specific
tier: T1
mode: exploratory
date: 2026-09-13
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On `Genesis/data/slates_multistep` (`n20_L20mm`/`n20_L40mm`, `n20_L10mm`
  EXCLUDED as a contaminated dataset per user instruction) and, separately,
  on the 43-file `overnight_randlen` holdout corpus (never trained on, split
  disjointness independently re-verified here), scoring stage-2 visual and
  stage-3 latent+desc94 switched-by-push-length-bin vs global operators,
  and the stage-3 descriptor-only point-mass operator, under slateN (K=32
  fixed reference, 150 without-replacement resamples/slate) across 5 goal
  shapes (`corner`, `stripe`, `random_quadrant`, `ring_O`, `T`) x 3 value
  functions (`lyapunov`, `mass_in_region`, `signed_mass_in_region`): (a)
  with `n20_L10mm` excluded, switched beats global pooled on `corner`/
  `lyapunov` (+0.938 vs +0.911) and in all 15 (shape,valuefn) cells on the
  overnight holdout -- reversing EXP-0006's stage-3 pooled finding that
  global beat switched; (b) the earlier "switching hurts on stripe" finding
  is specific to `lyapunov` (stripe/lyapunov: 0W/7L/33T) and does NOT hold
  under `mass_in_region` (12W/7L/21T) or `signed_mass_in_region`
  (9W/7L/24T) on the SAME goal shape -- goal-dependence is shape x
  value-function joint, not shape alone; (c) descriptor-only's positive
  slateN under `mass_in_region`/`signed_mass_in_region` (EXP-0006 only
  tested `lyapunov`) survives, weaker, on the in-corpus `slates_multistep`
  data, but does NOT generalise: on the overnight holdout it degrades
  sharply and reverses sign on `ring_O`/`lyapunov` (-0.313, worse than
  persistence) and goes ~0 on `ring_O`/`mass_in_region` (-0.003); (d) the
  coarse accuracy-based ranking (visual > latent+desc94 > descriptor-only >
  persistence) reproduces on the independent holdout corpus with markedly
  fewer ties than on `slates_multistep`.

prediction: null  # exploratory

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 0ddab20f
  dirty: true                      # same pre-existing unrelated dirty tree as
                                    # EXP-0004..EXP-0007; this record's own
                                    # inputs are untracked temp scratch.
  data_commit: "unrecorded for slates_multistep's own generation provenance, and for overnight_randlen's -- not re-checked in this promotion pass, same as EXP-0006"
  script: "experiments/temp/slaten-broad/eval_slaten_broad.py"
  data: ["Genesis/data/slates_multistep/{n20_L20mm,n20_L40mm} eval splits (n20_L10mm EXCLUDED)", "43-file overnight_randlen holdout split (seed 0, holdout_frac=0.2), reassembled into 128-candidate same-state slates from experiments/temp/hybrid-vis-desc/cache/test_cache.pt"]
  code_path: "Baselines.common.goals.slate_n_capture (generalised value-function form) + Baselines.common.goals.{mass_in_region,signed_mass_in_region,letter_mask,quadrant_mask}; operators reused from disk, none refit: experiments/temp/stage2-slaten/operators/visual_lam1.0.pt, experiments/temp/stage3-slaten/{latent_operators.pt,desc_operator.pt}, experiments/temp/latent-switched/encoder_decoder.pt"
  seed: 0
  split: "slates_multistep's own pre-built eval manifests (20 slates x 128 candidates, step 0 only), n20_L10mm excluded; overnight holdout: descriptors.py::list_files()+split_files(seed=0,holdout_frac=0.2), independently re-derived and set-compared against test_cache.pt/train_cache.pt's own file_id fields -- exact match (170 train / 43 test, zero overlap)"
  runtime: "not recorded precisely; completed within the producing session, no cell left incomplete for the ones attempted"
  runs: []

budget:
  declared: "not separately declared for this promotion -- slaten-broad ran under its own producing session's budget before this record existed"
  spent: "promotion pass: ~30 min / ~55k tokens"
  outcome: within

design:
  varied:
    dataset: [n20_L20mm, n20_L40mm, overnight_holdout_43file]
    goal_shape: [corner, stripe, random_quadrant, ring_O, T]
    value_fn: [lyapunov, mass_in_region, signed_mass_in_region]
    model: ["visual switched", "visual global", "latent+desc94 switched", "latent+desc94 global", "descriptor-only (point-mass readout, extended to all 3 value fns)", persistence, random]
  held_fixed:
    K: "K=32 fixed reference, 150 without-replacement resamples/slate (reduced from stage2/3-slaten's 300 for time budget -- sems marginally noisier, direction unaffected)"
    degeneracy_screen: "any (dataset, shape, valuefn) cell with frac(slates with true value constant across all 128 candidates) >= 0.90 excluded -- none of the 45 multistep or 15 holdout cells hit this"
    excluded_dataset: "n20_L10mm excluded from every number in this report per user instruction (contaminated, all pushes in the weakest length bin)"
    hybrid14_hybrid94_latentonly: "NOT re-run here -- already shown statistically tied with visual/latent+desc94 in stage2-slaten/stage3-slaten, and none of this record's 4 questions bear on that tie"
  baselines: [persistence, random]
  metric: "slateN (K=32 fixed reference), per experiments/METRICS.md's generalised-value-function section -- lyapunov (cost) and mass_in_region/signed_mass_in_region (value) all reused verbatim from METRICS.md, not reinvented here"

noise_floor: "paired sem on head-to-head diffs, reported per (shape,valuefn) cell in results_slaten_broad.json alongside wins/losses/ties -- e.g. pooled corner/lyapunov switched-vs-global: 7W/6L/27T; stripe/lyapunov: 0W/7L/33T (the one clear loss). Tie rates range widely by cell (6-34 of 40 multistep, 6-19 of 43 holdout) -- reported per METRICS.md's own power caveat, not hidden."

depends_on: [goal-mask-axis-convention-row-y-col-x, occ-rasteriser-consistency, push-frame-warp-roundtrip]
establishes: [hybrid-vis-desc-holdout-disjoint-from-train]

# ---- outcome --------------------------------------------------------------
result: >
  With n20_L10mm excluded, pooled corner/lyapunov switched (+0.938) beats
  global (+0.911), 7W/6L/27T -- reversing EXP-0006's L10mm-contaminated
  pooled finding; the overnight holdout (a corpus never trained on)
  confirms this independently, switched beating global in all 15
  (shape,valuefn) cells, often by a wide margin with far fewer ties.
  Goal-dependence survives but is finer than shape alone: stripe loses to
  global under lyapunov (0/7/33) but wins under mass_in_region (12/7/21)
  and signed_mass_in_region (9/7/24); the newly-scored T shape shows the
  largest, most consistent switched win across all 3 value functions.
  Descriptor-only's positive slateN partially survives the two new value
  functions in-corpus (weaker than under lyapunov, still positive) but
  fails out-of-corpus on annular/thin targets: ring_O/lyapunov on the
  holdout goes to -0.313 (worse than persistence). The coarse accuracy-
  based ranking (visual > latent+desc94 > descriptor-only > persistence)
  reproduces cleanly on the holdout, with markedly fewer ties than on
  slates_multistep. Independent re-verification of the holdout split
  (file-id set comparison against the artifact the operators actually
  consumed) found zero train/test overlap (170+43=213).
verdict: supported
downgrades: [incomplete-design, inconsistency, untested-dependency]
grade: very-low
supersedes: [EXP-0006]
invalidated_by: null
---

## Why this test discriminates

If the earlier switched-vs-global reversal (global beating switched, pooled,
in EXP-0006's stage-3 numbers) were a real property of switching rather than
an artifact of one contaminated dataset, removing that dataset should not
change the sign. It does: pooled corner/lyapunov flips from +0.697/+0.811
(switched/global, L10mm included) to +0.938/+0.911 (switched/global, L10mm
excluded) -- and the overnight holdout, built from a wholly different
corpus with no L10mm-style short-push contamination, independently confirms
switched winning in all 15 cells. A coincidental confound would not be
expected to reproduce on an unrelated corpus. Similarly, if "stripe is bad
for switching" were a property of the goal SHAPE, it should hold under any
value function scored on that shape; instead it holds under `lyapunov` only
and reverses under the two mass-based value functions on the identical
shape -- discriminating "shape" from "shape x value-function" as the real
governing condition.

## What was actually run

A single evaluation pass, `eval_slaten_broad.py`, reusing every operator
from disk (none refit): stage-2 visual (`experiments/temp/stage2-slaten/
operators/visual_lam1.0.pt`), stage-3 latent+descriptor and descriptor-only
(`experiments/temp/stage3-slaten/{latent_operators.pt,desc_operator.pt}`),
and the encoder/decoder (`experiments/temp/latent-switched/
encoder_decoder.pt`). Two data sources were scored on the identical
(shape, valuefn) grid: (1) `slates_multistep`'s `n20_L20mm`/`n20_L40mm`
(L10mm excluded), and (2) a NEW 43-slate corpus built directly from the
43-file overnight_randlen holdout by grouping `test_cache.pt` rows by
`file_id` and taking each file's first 128 rows (verified against
`Baselines/common/randlen_data.py`'s row-major-block convention). The
descriptor-only point-mass readout was extended from `lyapunov`-only
(its only mode in EXP-0006) to `mass_in_region`/`signed_mass_in_region` by
also reading the model's predicted total mass and placing all of it at the
predicted world COM -- the natural point-mass approximation of `V = sum(occ
* mask)`, not a new formula. `hybrid14`/`hybrid94`/`latent-only` cells and
`n50_L20mm` were not run (see Threats/incomplete-design). The holdout split
disjointness was independently re-derived (not assumed) by re-running
`descriptors.py::list_files()+split_files(seed=0, holdout_frac=0.2)` inside
this script and comparing file-id sets against `train_cache.pt`/
`test_cache.pt`'s own recorded file ids: exact match, zero overlap.

## Numbers

**Pooled slateN (K=32), n20_L20mm + n20_L40mm (L10mm excluded), n=40/row --
selected cells (full 45 in results_slaten_broad.json):**

| shape | value fn | visual_sw | visual_gl | latent+desc94_sw | latent+desc94_gl | desc-only | persist | random |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| corner | lyapunov | +0.972 | +0.952 | +0.938 | +0.911 | +0.663 | -0.167 | +0.040 |
| stripe | lyapunov | +0.976 | +0.993 | +0.906 | +0.903 | +0.419 | +0.275 | +0.173 |
| stripe | mass_in_region | +0.386 | +0.310 | +0.290 | +0.325 | +0.315 | -0.021 | -0.003 |
| stripe | signed_mass_in_region | +0.527 | +0.471 | +0.483 | +0.501 | +0.446 | +0.015 | +0.005 |
| ring_O | signed_mass_in_region | +0.532 | +0.429 | +0.433 | +0.652 | +0.242 | +0.057 | +0.017 |
| T | lyapunov | +0.901 | +0.475 | +0.735 | +0.552 | +0.347 | -0.035 | +0.139 |

**Head-to-head visual switched vs global (W/L/T of 40):** corner/lyapunov
7/6/27 (equiv. 4/2/34 under the strict-tie-per-model convention used in the
source RESULTS.md's other rows); stripe/lyapunov **0/7/33** (only cell where
global wins outright); T/lyapunov **31/2/7** (largest, cleanest switched
win).

**Overnight holdout (43 files, never trained on), selected cells (full 15
in JSON; switched beats global in all 15):**

| shape | value fn | visual_sw | visual_gl | latent+desc94_sw | latent+desc94_gl | desc-only | persist | random |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| corner | lyapunov | +0.900 | +0.547 | +0.735 | +0.553 | +0.197 | +0.099 | +0.030 |
| ring_O | lyapunov | +0.864 | +0.586 | +0.675 | +0.429 | **-0.313** | +0.041 | +0.021 |
| ring_O | mass_in_region | +0.785 | +0.437 | +0.586 | +0.253 | -0.003 | -0.048 | +0.022 |
| T | signed_mass_in_region | +0.732 | +0.208 | +0.429 | +0.263 | -0.057 | -0.017 | -0.010 |

Full per-(dataset,shape,valuefn,model) numbers, degeneracy fractions, and
every wins/losses/ties/paired-sem pair are in
`experiments/EXP-0008-slaten-broad-widened-goals/results/
results_slaten_broad.json`, with the narrative writeup preserved verbatim
in `experiments/EXP-0008-slaten-broad-widened-goals/results/RESULTS.md`
(both copied from `experiments/temp/slaten-broad/`, which is gitignored).

## What would change the verdict

- **`n50_L20mm`'s eval split**, if built, would let the L10mm-exclusion
  finding be checked on a third push-length regime.
- **hybrid14/hybrid94/latent-only rescored on this same 5x3 grid** would
  close the last gap in "does the full accuracy-based family ranking
  survive under every goal/valuefn combination," not just visual/
  latent+desc94/descriptor-only.
- **An R^2 validation of the point-mass mass-in-region/signed-mass-in-region
  readout against ground truth** (never done for either value function) --
  the sharp holdout failures in (c) are read directly off slateN, with no
  independent check of whether the readout itself is a reasonable estimator
  under these value functions specifically.
- **A per-slate-seeded `random_quadrant` mask** (currently one fixed mask
  per dataset cell) -- would tighten but is not expected to reverse any
  finding above (documented as a scope reduction, not a bias risk for
  switched-vs-global comparisons specifically).

## Threats

- **`incomplete-design`**: `n50_L20mm` still has no eval split (inherited
  gap from EXP-0006, still open); `hybrid14`/`hybrid94`/`latent-only` were
  not rescored on this grid; the point-mass mass-in-region/signed-mass-in-
  region readout has no independent R^2 validation; K=32 used 150
  without-replacement resamples here vs 300 in stage2/3-slaten (time
  budget) -- sems are marginally noisier as a result, direction unaffected
  per spot-checks in the source RESULTS.md.
- **`inconsistency`**: switched does not beat global in every cell tested
  -- stripe/lyapunov is a clear, well-powered global win (0/7/33), and
  ring_O/signed_mass_in_region also favours global (+0.652 vs +0.433).
  This is reported as a real governing condition (shape x value-function),
  not averaged away.
- **`untested-dependency`**: `goal-mask-axis-convention-row-y-col-x` and
  `occ-rasteriser-consistency` remain `unchecked` in `INVARIANTS.md`,
  inherited from EXP-0001/EXP-0006 -- this record's new goal shapes
  (`random_quadrant`, `ring_O`, `T`) and both new value functions rest on
  the same unverified conventions.
- Considered and dismissed: **`selection`** -- the L10mm exclusion was a
  standing instruction from before this run started (not a post-hoc choice
  made after seeing results), the reduced K=32/150-resample setting was a
  declared time-budget tradeoff stated up front, and no cell reported here
  was chosen after seeing its own outcome (all 45+15 cells in the design
  are reported, not a subset).
- Considered and dismissed: **provenance mismatch** between the operators
  scored here and their EXP-0005/EXP-0006 counterparts -- all four fitted
  operators were loaded byte-identical from the `.pt` files EXP-0006 itself
  persisted, not refit.

## Unrelated findings

- `random_quadrant`'s mask is one fixed quadrant per dataset cell, seeded
  by a hash of the dataset name rather than truly per-slate as `Baselines/
  common/goals.py`'s own docstring recommends -- documented in the source
  RESULTS.md as a scope reduction, not silently dropped.
- The 213-file corpus underlying the holdout split (`descriptors.py::
  list_files()`) globs both n20 and n50 overnight_randlen subfolders; the
  43-file holdout used here inherits whatever n20/n50 mix `build_data.py`
  produced, not separated by group size -- a possible confound between
  "holdout" and "n50-heavier" that was not checked in the source run.
