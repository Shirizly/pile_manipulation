---
# ---- identity -------------------------------------------------------------
id: EXP-0001
title: NFD_randlen beats both GNN variants on image accuracy in every corpus tested; all three models' slateN capture is positive everywhere
tier: T1
mode: exploratory
date: 2026-09-10
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  Across all 3 corpora tested (L20mm, L40mm, overnight_randlen held-out
  test), NFD_randlen's swept-region image `accuracy` exceeds both GNN
  variants' (gnn_l20l40, gnn_randlen_n30) `accuracy` on that same corpus,
  and every one of the 81 (model x corpus x goal x value-function) `slateN`
  capture cells is positive (better than a random pick from the same
  same-state pool).

# prediction: omitted -- mode is exploratory, no threshold was committed
# before running (this record reports newly-built evaluation infrastructure
# and its first measurement pass, not a pre-registered hypothesis test).

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 36b38878
  dirty: true                     # working tree modified when this ran -- see
                                  # "What was actually run" for the exact file list
  data_commit: "e612df7b (slates_multistep cells); unrecorded (overnight_randlen -- no provenance block in its _0_config.yaml)"
  script: Baselines/common/eval_report.py
  data:
    - "configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml"
    - "configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml"
    - "configs/dataset/genesis_overnight_randlen_test_all.yaml"
  code_path: "PileSweepData raster (registry.dataset_registry.build_dataset), via Baselines.common.data.load_cell (slate cells) and Baselines.common.randlen_data.load_randlen_cell (randlen, N-agnostic)"
  seed: 0
  split: "L20mm/L40mm: manifest-defined slate-level held-out eval cell (20 of 50 slates). randlen_test: file-level held-out split, seed 0, scripts/probes/prepare_randlen_split.py, all 5 groups"
  runtime: "~4 min total (3 corpus loads + 9 model-scoring passes), CPU+single GPU, shared machine"

budget:
  declared: "not declared in advance -- this record was filed after the fact, per user direction to build the report first and record it"
  spent: "~3.5 h across pipeline design, implementation, smoke tests, full run, and this record"
  outcome: exceeded

design:
  varied:
    model: [gnn_l20l40, gnn_randlen_n30, nfd_randlen]
    corpus: [L20mm, L40mm, randlen_test]
    goal: [random_quadrant, ring_O, T]
    value_function: [lyapunov, mass_in_region, signed_mass]
  held_fixed:
    grid_resolution: "64x64"
    swept_region_mask_params: "0.04/0.128*W plate width, 0.5*plate_px+2.0 half-width, 0.5*plate_px pad -- same as scripts/probes/expB_multistep_eval.py"
    slateN_pool: "K = full pool (128 candidates), no replacement, step-0 same-state pools only"
    gnn_cube_size_m: 0.005
    gnn_no_orientation: true
  baselines: [persistence]        # via metrics()'s own denominator; mean-delta/linear not
                                  # re-fit in this record (out of scope -- see "What would
                                  # change the verdict")
  metric: "accuracy, slateN -- slateN generalised to 3 value functions (lyapunov, mass_in_region, signed_mass), see docs/experiments/METRICS.md"

noise_floor: "not measured -- single run per (model, corpus) cell, no repeated seeds/folds. Flagged as a real gap (see Threats), not silently assumed zero."

depends_on: [randlen-train-test-file-disjoint, randlen-step0-pool-size-128, occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x]
establishes: []

# ---- outcome --------------------------------------------------------------
result: "accuracy: nfd_randlen 0.407/0.509/0.456 (L20mm/L40mm/randlen_test) vs gnn_l20l40 0.278/0.313/0.071 and gnn_randlen_n30 0.242/0.325/0.148 -- NFD highest in all 3. slateN capture range across all 81 cells: [0.076, 0.988], none negative."
verdict: supported
downgrades: [untested-dependency, imprecision, indirectness]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If NFD's grid-native representation carried no real advantage over GNN's
node-count-bottlenecked one, accuracy could as easily favour either GNN
variant on any given corpus -- particularly `gnn_randlen_n30`, trained on
the same corpus family as `nfd_randlen`. Finding NFD strictly ahead on
accuracy in all 3 corpora, including the 2 corpora neither NFD nor
`gnn_randlen_n30` trained on directly (L20mm/L40mm, in-distribution only
for `gnn_l20l40`), is the pattern a real representational gap would
produce and a coincidence would not reliably reproduce across 3 different
corpora. Similarly, if the corrected (camera-only) GNN pipeline had
somehow broken control-ranking ability, some (model, corpus, goal,
value-function) cells would show near-zero or negative capture (no better
than random); finding none negative across 81 cells is evidence the
pipeline fix (SPEC.md's "CORRECTION" section) did not silently break
ranking even though it changed what the model is allowed to see.

## What was actually run

This record documents the FIRST run of newly-built evaluation
infrastructure (`Baselines/common/eval_report.py`, `goals.py`,
`randlen_data.py`, and `Baselines/GNN/perception.py`'s
`rasterize_nodes_as_cubes`/`resample_occupancy_through_nodes`), built in
the same session as this record, per an explicit user request for a
cross-corpus, multi-metric report across GNN and NFD. No prediction was
committed beforehand -- this is exploratory by construction, not a
threshold test.

**Working tree was dirty at run time** (`provenance.dirty: true`). The
modified/new files, none of which were reverted before this run:
`Baselines/GNN/{LOG.md,SPEC.md,predictor.py,perception.py (new),
dataset/dataset_genesis_gnn.py,train/train_genesis_gnn_dyn.py,
runs/ckpt_{best,last}.pth (retrained this session),
runs/randlen_train_all_n30/ (new checkpoint, this session),
scripts/verify_rasterizer.py (deleted this session)}`,
`Baselines/common/{data.py,eval_baseline.py,eval_randlen_indist.py,
eval_report.py (new),goals.py (new),randlen_data.py (new)}`,
`docs/experiments/METRICS.md`, and 12 `configs/dataset/genesis_*.yaml`
files (documentation-only edits to those 12 -- citation cleanup, no
functional change; see git diff for the exact set). In short: the checkpoints
being scored (`ckpt_best.pth`, `randlen_train_all_n30/ckpt_best.pth`) and
the scoring code itself (`predictor.py`, `perception.py`) were BOTH
produced in this same uncommitted working-tree state -- there is no
earlier commit these specific numbers could be reproduced against.

**Models scored:**
- `gnn_l20l40` -- `Baselines/GNN/runs/ckpt_best.pth`, retrained this
  session on the corrected (camera-raster-only) pipeline, pooled
  L20mm+L40mm training data, `n_particles=20`, epoch 300, val_mse 4.02e-6.
- `gnn_randlen_n30` -- `Baselines/GNN/runs/randlen_train_all_n30/ckpt_best.pth`,
  trained this session on the pooled overnight_randlen TRAIN corpus (all 5
  groups, n20+n50), `n_particles=30`, epoch 101, val_mse 5.66e-6.
- `nfd_randlen` -- `Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth`,
  pre-existing checkpoint from an earlier session (not retrained here);
  its own training provenance was not re-verified in this record.

**GNN accuracy uses a node-count-bottlenecked ground truth** (both
`occ0`/`occ1` FPS-resampled to the model's own `n_particles` and rasterised
back before comparison -- `docs/experiments/METRICS.md`, "GNN accuracy:
node-count-bottlenecked comparison"), NOT raw ground truth. NFD's accuracy
uses raw ground truth directly (no such bottleneck exists for a grid-native
model). This is a deliberate asymmetry (fair to each model's own
representation), not an oversight, but it means GNN's and NFD's `accuracy`
numbers are not numerically "the same quantity" in the strictest sense --
flagged here and in Threats, not hidden.

**No `mean-delta`/`linear` reference operator was fit for this record**
(unlike `Baselines/GNN/LOG.md`'s earlier reports) -- out of scope for what
was asked; `persistence` (built into `metrics()`'s own denominator) is the
only baseline present.

## Numbers

**Accuracy** (swept-region, vs persistence; higher is better, 1 = perfect):

| corpus | gnn_l20l40 | gnn_randlen_n30 | nfd_randlen |
|---|---|---|---|
| L20mm | 0.2783 | 0.2416 | **0.4071** |
| L40mm | 0.3127 | 0.3250 | **0.5088** |
| randlen_test | 0.0712 | 0.1484 | **0.4564** |

**slateN capture**, per (goal, value function), averaged over the goal's
step-0 same-state pools (n_slates: L20mm=20, L40mm=20, randlen_test=21):

`gnn_l20l40`:

| goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| L20mm / random_quadrant | 0.9583 | 0.9318 | 0.9662 |
| L20mm / ring_O | 0.9565 | 0.8021 | 0.4199 |
| L20mm / T | 0.7315 | 0.7054 | 0.7500 |
| L40mm / random_quadrant | 0.9710 | 0.9058 | 0.9358 |
| L40mm / ring_O | 0.9342 | 0.7937 | 0.8771 |
| L40mm / T | 0.9377 | 0.8163 | 0.8824 |
| randlen_test / random_quadrant | 0.5153 | 0.4956 | 0.4539 |
| randlen_test / ring_O | 0.3698 | 0.2897 | 0.3520 |
| randlen_test / T | 0.4461 | 0.2408 | 0.1480 |

`gnn_randlen_n30`:

| goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| L20mm / random_quadrant | 0.8764 | 0.7977 | 0.8600 |
| L20mm / ring_O | 0.8441 | 0.4251 | 0.0763 |
| L20mm / T | 0.4827 | 0.3710 | 0.4929 |
| L40mm / random_quadrant | 0.9542 | 0.8189 | 0.8923 |
| L40mm / ring_O | 0.7860 | 0.6441 | 0.6362 |
| L40mm / T | 0.8960 | 0.6155 | 0.8454 |
| randlen_test / random_quadrant | 0.8145 | 0.7810 | 0.7794 |
| randlen_test / ring_O | 0.6258 | 0.5263 | 0.4589 |
| randlen_test / T | 0.8127 | 0.4512 | 0.3932 |

`nfd_randlen`:

| goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| L20mm / random_quadrant | 0.9224 | 0.9210 | 0.9142 |
| L20mm / ring_O | 0.8405 | 0.7036 | 0.4138 |
| L20mm / T | 0.8162 | 0.7134 | 0.7404 |
| L40mm / random_quadrant | 0.9876 | 0.9044 | 0.9464 |
| L40mm / ring_O | 0.8052 | 0.6570 | 0.7192 |
| L40mm / T | 0.9014 | 0.7512 | 0.8372 |
| randlen_test / random_quadrant | 0.9499 | 0.8516 | 0.9403 |
| randlen_test / ring_O | 0.8926 | 0.8527 | 0.8499 |
| randlen_test / T | 0.9706 | 0.8466 | 0.7414 |

**Averaged over the 3 goals, per value function** (also in
`Baselines/common/runs/cross_corpus_report.json`):

| model | corpus | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| gnn_l20l40 | L20mm | 0.8821 | 0.8131 | 0.7120 |
| gnn_l20l40 | L40mm | 0.9476 | 0.8386 | 0.8984 |
| gnn_l20l40 | randlen_test | 0.4437 | 0.3420 | 0.3180 |
| gnn_randlen_n30 | L20mm | 0.7344 | 0.5313 | 0.4764 |
| gnn_randlen_n30 | L40mm | 0.8787 | 0.6928 | 0.7913 |
| gnn_randlen_n30 | randlen_test | 0.7510 | 0.5862 | 0.5438 |
| nfd_randlen | L20mm | 0.8597 | 0.7793 | 0.6894 |
| nfd_randlen | L40mm | 0.8981 | 0.7709 | 0.8343 |
| nfd_randlen | randlen_test | 0.9377 | 0.8503 | 0.8439 |

Full per-slate raw numbers are not retained (only per-goal means were
written to the JSON) -- see "What would change the verdict" for what
re-running with per-slate output would add.

## What would change the verdict

- **A noise floor.** Re-running with several seeds (where the model/eval
  path has any stochasticity -- FPS's random start point, currently seeded
  per-row so this specific run is reproducible, but re-seeding differently
  would show run-to-run spread) or a fold-level split would tell whether
  the accuracy gaps (e.g. 0.407 vs 0.278 on L20mm) are inside or outside
  plausible noise. Cheap: re-run `eval_report.py` with a different global
  seed offset, ~5 min.
- **`mean-delta`/`linear` reference rows**, fit per corpus, would show
  whether any of the 3 models are actually beating a trivial baseline by
  more than the baseline itself beats persistence -- this record only
  established accuracy relative to persistence, not relative to the
  cheapest non-trivial operator.
- **Per-slate capture distributions** (not just the mean) -- METRICS.md's
  own caveat on `slateN` (ties, effective sample size) was not checked
  here; a tie-rate/spread analysis could show the averaged numbers above
  rest on very few effectively-distinct pools, especially for
  `randlen_test`'s smaller `piled_n50` group (1 held-out file only).
- **A byte-identical rasteriser check** between `load_cell` and
  `load_randlen_cell`'s occ0/occ1 construction (the `occ-rasteriser-consistency`
  tag, currently `unchecked`) would retire that dependency.

## Threats

- **`untested-dependency`**: 3 of 4 `depends_on` tags are `unchecked`
  (`occ-rasteriser-consistency`, `goal-mask-axis-convention-row-y-col-x`)
  -- believed true by construction/code comparison, not tested.
- **`imprecision`**: no noise floor was measured (single run per cell, no
  seed/fold repetition) -- the accuracy/capture gaps reported are not
  benchmarked against any measured run-to-run spread.
- **`indirectness`**: GNN's `accuracy` is measured against a node-resampled
  ground truth (its own representational ceiling), not raw ground truth --
  a deliberate, documented choice (`METRICS.md`), but it means GNN's and
  NFD's accuracy numbers, while both called "accuracy", are not measuring
  literally the same comparison target. Considered and kept anyway because
  the alternative (comparing GNN against raw ground truth) would conflate
  "wrong dynamics" with "fewer nodes than cubes exist" -- a worse
  indirectness in the other direction.
- Considered and dismissed: **`provenance`** mismatch between GNN and NFD
  code paths. Both ultimately consume the same `occ0`/`occ1` built by the
  same registry rasteriser (`load_cell`/`load_randlen_cell`), and neither
  predictor's OWN rasterisation (GNN's `rasterize_nodes_as_cubes`, NFD's
  native grid output) touches the other's code -- this is an inherent
  property of comparing two different model families, not an inconsistency
  within one.

## Unrelated findings

- `Baselines/GNN/runs/gnn_randlen/` (a checkpoint directory predating this
  session, trained on overnight_randlen under the OLD privileged-state
  pipeline) still exists on disk and is now stale relative to the
  corrected `predictor.py` -- scoring it today would silently feed it
  raster-derived node positions its weights were never trained on. Not
  touched or deleted here.
- `Baselines/NFD/predictor.py`'s `NFD_CKPT` env var has no safety check
  against an unset/wrong value (defaults silently to the in-distribution
  `nfd_3ch` checkpoint) -- already flagged once before in this project's
  history (`Baselines/common/eval_randlen_indist.py`'s own module
  docstring describes an earlier incident of exactly this), and still
  true today; this record set `NFD_CKPT` explicitly each time, but a
  future caller could still hit the same silent-wrong-checkpoint failure
  mode.
- `Baselines/common/randlen_data.py`'s `piled_n50` group has only 1
  held-out test file (13 total, 12 train/1 test) -- thin relative to the
  other 4 groups (45-50 files each, 3-5 held out). Not a bug, just a
  size imbalance worth knowing before reading `randlen_test` results as
  balanced across groups.
