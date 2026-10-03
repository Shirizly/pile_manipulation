---
# ---- identity -------------------------------------------------------------
id: EXP-0002
title: Both randlen-trained models' accuracy is stable across piled/scattered/mixed spawn modes; NFD_randlen beats gnn_randlen_n30 in every mode
tier: T1
mode: exploratory
date: 2026-09-10
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  For the two models trained on overnight_randlen (gnn_randlen_n30,
  nfd_randlen), scored separately on the piled/scattered/mixed held-out
  test files (instead of EXP-0001's pooled view), swept-region image
  `accuracy` for each model varies by less than 0.04 across the 3 spawn
  modes, nfd_randlen's accuracy exceeds gnn_randlen_n30's in every spawn
  mode, and every one of the 54 (model x spawn-mode x goal x
  value-function) `slateN` capture cells is positive.

# prediction: omitted -- mode is exploratory, same as EXP-0001 (this is a
# follow-up data-slicing pass on the same measurement, not a pre-registered
# threshold test).

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 36b38878
  dirty: true                     # same dirty tree as EXP-0001, plus 2 new
                                  # yaml configs and one small dict edit to
                                  # eval_report.py -- see "What was actually run"
  data_commit: "unrecorded (overnight_randlen -- no provenance block in its _0_config.yaml, same as EXP-0001)"
  script: Baselines/common/eval_report.py
  data:
    - "configs/dataset/genesis_overnight_randlen_test_piled_all.yaml"
    - "configs/dataset/genesis_overnight_randlen_test_scattered_all.yaml"
    - "configs/dataset/genesis_overnight_randlen_test_n20_mixed.yaml"
  code_path: "PileSweepData raster (registry.dataset_registry.build_dataset), via Baselines.common.randlen_data.load_randlen_cell -- identical code path to EXP-0001's randlen_test cell, just three smaller file sets instead of one pooled one"
  seed: 0
  split: "overnight_randlen held-out test files, file-level split (scripts/probes/prepare_randlen_split.py, seed 0), partitioned by spawn mode instead of pooled: piled (piled_n20 5 files + piled_n50 1 file = 6), scattered (scattered_n20 5 + scattered_n50 5 = 10), mixed (mixed_n20 5 -- no mixed_n50 group exists in this corpus at all)"
  runtime: "~70s total (3 corpus loads + 6 model-scoring passes)"

budget:
  declared: "not declared in advance -- direct follow-up to EXP-0001, same session"
  spent: "~25 min (2 new configs, a 6-line CORPORA dict addition, one smoke test, the full run, this record)"
  outcome: within

design:
  varied:
    model: [gnn_randlen_n30, nfd_randlen]     # gnn_l20l40 excluded: not trained on randlen
    spawn_mode: [piled, scattered, mixed]
    goal: [random_quadrant, ring_O, T]
    value_function: [lyapunov, mass_in_region, signed_mass]
  held_fixed:
    grid_resolution: "64x64"
    swept_region_mask_params: "same as EXP-0001"
    slateN_pool: "K = full pool per file (128 candidates for piled/scattered/mixed alike), no replacement, step-0 only"
    gnn_cube_size_m: 0.005
    gnn_no_orientation: true
    scoring_code: "byte-identical to EXP-0001 (Baselines/common/eval_report.py unchanged except the CORPORA dict addition; predictor.py/perception.py untouched since EXP-0001)"
  baselines: [persistence]
  metric: "accuracy, slateN -- slateN generalised to 3 value functions (lyapunov, mass_in_region, signed_mass), see experiments/METRICS.md"

noise_floor: "not measured, same gap as EXP-0001. Additionally: n_slates per spawn mode is small (piled=6, scattered=10, mixed=5), so per-spawn-mode capture means rest on very few pools -- flagged in Threats, not treated as precise."

depends_on: [randlen-train-test-file-disjoint, randlen-step0-pool-size-128, occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x]
establishes: []

# ---- outcome --------------------------------------------------------------
result: "accuracy (piled/scattered/mixed): gnn_randlen_n30 0.1447/0.1374/0.1744 (range 0.037); nfd_randlen 0.4474/0.4568/0.4653 (range 0.019). NFD ahead of GNN in all 3 modes by a wide margin (~0.29-0.30). slateN capture range across 54 cells: [0.215, 1.000], none negative."
verdict: supported
downgrades: [untested-dependency, imprecision, indirectness]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0001 pooled all spawn modes together for the randlen corpus, so it
could not distinguish "both models generalise evenly across
initialisation styles" from "the pooled number is an average masking a
mode a model handles badly." If either model were spawn-mode-brittle
(e.g. much worse on `mixed`, the only mode combining scattered piles and
clumps in one scene), stratifying would surface a large per-mode swing;
finding both models' accuracy varying by <0.04 across modes is the
pattern even-generalisation would produce and brittleness would not. NFD
leading GNN in every stratum (not just on average) rules out the
possibility that EXP-0001's pooled NFD-ahead-of-GNN finding was carried by
one spawn mode alone.

## What was actually run

Direct follow-up to EXP-0001, same session, same scoring code
(`Baselines/common/eval_report.py`, `Baselines/common/goals.py`,
`Baselines/GNN/perception.py`, `Baselines/GNN/predictor.py` -- none of
these changed between the two records). Two things were added to run
this: two new dataset configs
(`genesis_overnight_randlen_test_{piled,scattered}_all.yaml`, pooling each
spawn mode's n20+n50 groups, mirroring `genesis_overnight_randlen_test_all.yaml`'s
own pooling) and 3 new entries in `eval_report.py`'s `CORPORA` dict
(`randlen_piled`, `randlen_scattered`, `randlen_mixed` -- the last reusing
the pre-existing `genesis_overnight_randlen_test_n20_mixed.yaml` directly,
since no `mixed_n50` group was ever collected for this corpus at all).

**Model scope narrowed relative to EXP-0001**: `gnn_l20l40` (trained on
`slates_multistep`, not `overnight_randlen`) was deliberately excluded --
this record only asks the spawn-mode question for models that actually
trained on this corpus.

**Working tree dirty, same as EXP-0001**, plus this session's two new
yaml configs and the small `CORPORA` dict addition to `eval_report.py`
(see `provenance.dirty`). The checkpoints/predictor/perception code being
scored are unchanged from EXP-0001's dirty state.

## Numbers

**Accuracy** (swept-region, vs persistence; higher is better):

| spawn mode | n files (test) | gnn_randlen_n30 | nfd_randlen |
|---|---|---|---|
| piled | 6 (5 n20 + 1 n50) | 0.1447 | **0.4474** |
| scattered | 10 (5 n20 + 5 n50) | 0.1374 | **0.4568** |
| mixed | 5 (n20 only) | 0.1744 | **0.4653** |

(Compare EXP-0001's pooled `randlen_test`: gnn_randlen_n30 0.1484,
nfd_randlen 0.4564 -- both pooled values fall within the per-mode range
above, as expected for an average.)

**slateN capture**, per (goal, value function), averaged over each mode's
step-0 same-state pools:

`gnn_randlen_n30`:

| spawn mode / goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| piled / random_quadrant | 0.9247 | 0.5885 | 0.5875 |
| piled / ring_O | 0.6906 | 0.4515 | 0.3385 |
| piled / T | 0.7892 | 0.2152 | 0.2403 |
| scattered / random_quadrant | 0.5441 | 0.8971 | 0.8799 |
| scattered / ring_O | 0.5562 | 0.3815 | 0.5333 |
| scattered / T | 0.8160 | 0.5260 | 0.4097 |
| mixed / random_quadrant | 0.8427 | 0.8689 | 0.8810 |
| mixed / ring_O | 0.6297 | 0.7572 | 0.7127 |
| mixed / T | 0.9113 | 0.3943 | 0.2621 |

`nfd_randlen`:

| spawn mode / goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| piled / random_quadrant | 0.9934 | 0.8407 | 0.9443 |
| piled / ring_O | 0.8413 | 0.7077 | 0.8843 |
| piled / T | 0.9715 | 0.7711 | 0.6075 |
| scattered / random_quadrant | 0.9729 | 0.9371 | 0.9361 |
| scattered / ring_O | 0.9128 | 0.9378 | 0.8413 |
| scattered / T | 0.9691 | 0.8151 | 0.6974 |
| mixed / random_quadrant | 0.9363 | 0.7661 | 0.9543 |
| mixed / ring_O | 0.9136 | 0.8564 | 0.8260 |
| mixed / T | 0.9724 | 1.0000 | 0.9900 |

**Averaged over the 3 goals, per value function:**

| model | spawn mode | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| gnn_randlen_n30 | piled | 0.8015 | 0.4184 | 0.3888 |
| gnn_randlen_n30 | scattered | 0.6388 | 0.6015 | 0.6076 |
| gnn_randlen_n30 | mixed | 0.7945 | 0.6734 | 0.6186 |
| nfd_randlen | piled | 0.9354 | 0.7732 | 0.8120 |
| nfd_randlen | scattered | 0.9516 | 0.8967 | 0.8249 |
| nfd_randlen | mixed | 0.9408 | 0.8742 | 0.9234 |

Full JSON: `Baselines/common/runs/cross_corpus_report_spawnmode.json`.

## What would change the verdict

- **More held-out files per spawn mode**, especially `piled` (only 1 n50
  file) and `mixed` (5 files total, all n20) -- the current per-mode
  capture means rest on very few same-state pools (n_slates: piled=6,
  scattered=10, mixed=5), so a single unusual pool could move a mode's
  mean noticeably. Re-collecting more overnight_randlen test files per
  mode (particularly `piled_n50`) would tighten this.
- **A per-slate tie-rate/spread check** (as flagged in EXP-0001) is still
  not done here either.
- **`mixed_n50` does not exist as a collected group at all** for this
  corpus -- if a true apples-to-apples n20-vs-n50 comparison per spawn
  mode is wanted later, `mixed` would need new data collection, not just a
  new config.

## Threats

- **`untested-dependency`**: same 2 unchecked tags as EXP-0001
  (`occ-rasteriser-consistency`, `goal-mask-axis-convention-row-y-col-x`).
- **`imprecision`**: no noise floor measured, and per-mode slate counts
  are small (5-10) -- the accuracy stability claim (<0.04 spread) and the
  capture means are both averages over few pools, not statistically
  powerful ones.
- **`indirectness`**: same GNN-vs-NFD accuracy asymmetry as EXP-0001 (GNN
  scored against a node-resampled ground truth, NFD against raw ground
  truth) -- see that record's Threats for the full reasoning, unchanged
  here.

## Unrelated findings

- `piled_n50` has only 1 held-out test file out of 13 total (12
  train/1 test) -- already flagged as an unrelated finding in EXP-0001;
  repeating here because this record's `piled` spawn-mode result rests
  disproportionately on that single n50 file plus 5 n20 files.
- `mixed_n50` was never collected for this corpus (confirmed: no such
  directory exists under `Genesis/data/overnight_randlen{,_train,_test}`)
  -- not a bug, just a corpus-composition fact worth knowing before
  reading `mixed` as symmetric with `piled`/`scattered`.

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- GNN rows carry `gnn-node-sampling-consistent-within-state` (broken: node sampling seeded by batch row).
- `slates_multistep` (L20mm/L40mm) was simulated with friction 0.3 / density 1000 -- off-training-physics for randlen-trained models (`benchmark-physics-matches-training`, broken); EXP-0048 later found physics sets barely change 20 mm single-push rankings on scatter (C-053), untested for piles/longer pushes.
- Asymmetric goal masks (T / random_quadrant / ring_O / stripe / letters) in this record were scored BEFORE the goal-axis fix `28271c09` (2026-09-17) and were never rescored; transpose-invariant goals (corner, center, ...) are unaffected (ISS-003).
- Spawn-mode strata rest on 5-10 files per mode; no later record re-tests spawn mode as a model-ranking stratifier with power (closest: EXP-0053's scatter vs clump accuracy split).
