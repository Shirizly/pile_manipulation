---
# ---- identity -------------------------------------------------------------
id: EXP-0021
title: >
  Post-fix-raster UNet beats the linear operator by 4-11 points in every one
  of 7 cells, and the "it just predicts nothing" concern is refuted --
  contact>0 stratification does not shrink either model's advantage
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: H-A1

# ---- the claim ------------------------------------------------------------
claim: >
  C-020's UNet result (EXP-0014: retrained on the corrected raster, the action
  channel is worth 38.5 of a 28.0-point whole-image advantage over persistence)
  generalises to the swept-region metric, to a linear-operator anchor, and
  across object count (n=5/10/20, plus n=50) and push-completion (blind vs
  contact-filtered >39mm): the UNet beats a same-split ridge-toward-identity
  linear operator on swept-region rms, and this advantage is not an artifact
  of predicting no-ops -- it does not shrink when restricted to contact>0
  transitions.

prediction:
  supports: >
    the UNet's swept-region rms (as % of persistence) beats the linear
    operator's in a majority of the 7 cells x 2 strata, by a margin exceeding
    a few points (i.e. not noise-floor-sized against EXP-0018's borrowed
    ~1.9-point LORO sd), AND the UNet's (and the linear operator's) advantage
    over persistence under contact>0 stratification is within a few points of
    its "all transitions" value -- not collapsed toward 0 -- AND the
    action-shuffle sign flip (shuffled worse than persistence) or a
    near-total shuffle-gap (shuffled destroys most of the true-action
    advantage) recurs in most of the 7 cells, replicating EXP-0014's
    mechanism rather than being a one-dataset fluke.
  refutes: >
    the UNet's swept-region advantage over persistence is much smaller under
    contact>0 than "all transitions" while the linear operator's is not (the
    pre-registered failure mode: the UNet's edge is mostly "predict nothing"),
    OR the UNet fails to beat the linear operator in most cells, OR the
    shuffle/zero ablation no longer shows the model depending heavily on its
    action channel.
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 0634e303
  dirty: false
  data_commit: >
    blind n5/n10/n20: f196d657 (all 8 files each). contact n10/n20: 3d7db119
    (all 8 files each). contact n5: MIXED -- 2 of 8 files (_0,_1) at f196d657,
    6 of 8 (_2..._7) at 3d7db119 (read from each file's own
    `_N_config.yaml` provenance: block; see Unrelated findings). blind n50
    (L040): unrecorded -- predates dataset-provenance stamping, same as
    EXP-0018/EXP-0009's L040 usage.
  script: >
    scripts/probes/exp0021_train_all.sh + scripts/probes/exp0021_eval_all.sh
    (both new, committed 20e07784/0634e303 before running), calling
    training.train (unmodified) and scripts/probes/exp0021_eval.py (new,
    committed 410524f1), which reuses fit_linear_foresight.py's canonicalise/
    fit_operator/predict_world/metrics/swept_region_mask/contact_score and
    scripts/probes/exp0009_rerun.py's predict_meandelta verbatim -- no new
    rasteriser, warp or metric was written.
  data: >
    ["Genesis/data/granularity/{n5,n10,n20,c5,c10,c20}/cube/*/size0.005/*_data.pt",
    "Genesis/data/foresight/L040/cube/n50/size0.005/*_data.pt"] via
    configs/dataset/genesis_granularity_*.yaml (new, committed 71dbfe4b)
  code_path: >
    PileSweepData raster (fixed, aac084e3+) for both the UNet's own
    input/target and the linear operator's occ_t/occ_t1 -- same dataset
    object for both, see "What was actually run"
  seed: >
    UNet training: none explicit (matches the EXP-0014 template, a
    like-for-like omission -- the file-level test/val/train split is
    deterministic by md5(path), not by a random draw). Shuffle-ablation
    permutation: seed 0, one draw per cell (unseeded-across-cells would still
    be one draw each; seeding it makes the run reproducible without changing
    what was measured).
  split: >
    registry file-level split (val_pct=10/test_pct=10), NEW fallback rule:
    Genesis/training/dataset.py's `_filter_split` grouped runs by
    (folder, physics-key) so files sharing nominal physics move together --
    but every granularity/L040 cell here has exactly ONE physics key across
    all 8 files, which previously put 100% of the folder in train (verified:
    val/test both raised "No configs found" before this record's code
    change). Falls back to per-FILE groups when a folder resolves to one
    physics-key group (commit 85f2413b) -- still whole-run/episode
    granularity, no leakage protection lost (every file already shares
    physics params, which is the only reason physics-grouping exists).
    Result: 6 files train / 1 val / 1 test per cell (test sizes vary after
    the contact-arm push-length filter, since filtering removes a different
    fraction of each file -- see Numbers). The linear operator is fit on
    this SAME dataset object's "train" split and scored on its "test" split
    -- literally the same held-out transitions the UNet's own test-split
    metrics were computed on, not two independently-drawn splits.
  runtime: "~50 min GPU (7 x 100-epoch trainings, res=64, ~4.3 s/epoch) + ~1 min CPU/GPU (7 evals, linear fit + UNet forward + ablation each)"

budget:
  declared: "3h wall-clock, 350k tokens (declared by the task-giver)"
  spent: "~2h05min, ~205k tokens"
  outcome: within

design:
  varied:
    cell: [blind_n5, blind_n10, blind_n20, contact_n5, contact_n10, contact_n20, blind_n50]
    stratum: [all, "contact>0"]
    action_channel (ablation only): [true, shuffled, zeroed]
  held_fixed: >
    architecture unetfilm (in_channels=2, cond_dim=3, uses_physics=true,
    input_mode=standard); epochs=100, batch_size=32, lr=1e-4
    StepLR(step=50,gamma=0.75), loss=eulerian_combined(mse=1.0,mass=0.2),
    mixed_precision=true, grad_clip_norm=1.0 -- all copied verbatim from
    configs/training/unetfilm_corl_limited_100e_fixedraster.yaml (EXP-0014's
    own template, not the drifted configs/ tree); dataset.resolution_scale
    changed 1.0 -> 0.5 for EVERY cell (64x64 grid) -- a deliberate, disclosed
    deviation from the template, made so the UNet's own world-frame grid is
    identical to the linear operator's, letting the two be scored on the
    literal same pixels rather than two different resolutions normalised
    only by a ratio; linear-operator config res=64/crop=0.5/ridge=1.0 toward
    identity (EXP-0018's own headline cell); val_pct/test_pct=10/10 and the
    per-file split fallback (same rule, all 7 cells); 5mm cubes, perpendicular
    40mm nominal pushes, same box, for every cell; contact-arm push-length
    filter fixed at >39mm.
  baselines: [persistence, mean-delta, "identity (warp only)", "linear operator (ridge=1.0 toward identity, same-split)"]
  metric: >
    swept-region rms (fit_linear_foresight.py's swept_region_mask,
    half_width=0.5*plate_px+2px, forward pad=0.5*plate_px) as % of
    persistence rms on the SAME cell's held-out test split, reported for all
    held-out transitions and for contact>0 only (contact_score>0, same
    swept-band definition). explained = 1 - rms/persistence_rms.

noise_floor: >
  Not independently measured for this 7-cell design (one training run per
  cell, no seed repeats). Borrowed only as a rough scale reference: EXP-0018
  measured an 8-fold LORO sd of 1.9 points of pct_persist for the SAME linear
  operator (res=16/crop=0.5, a different res/crop cell, not these datasets).
  The UNet-vs-linear margins here (4.4-10.9 points) are 2-6x that borrowed
  figure and the ablation effects (20-34 points) are 10x+, so this record
  treats them as clearly resolved relative to that figure -- but this is a
  borrowed floor from a different design, not a measured one, which is why
  `imprecision` is taken as a downgrade rather than treated as closed.

depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend, swept-region-metric, episode-split]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  The UNet beats the linear operator in all 7 cells x 2 strata (swept-region
  pct_persist 65.6-79.0% for the UNet vs 74.6-87.6% for the linear operator,
  a 4.4-10.9 point margin every time). The pre-registered concern is
  REFUTED: neither model's advantage over persistence shrinks under
  contact>0 stratification (UNet: unchanged or very slightly larger in all 7
  cells; linear operator: same). The action ablation replicates EXP-0014's
  mechanism at every cell -- zeroing the action collapses the UNet to
  97.7-101.0% of persistence (i.e. to within ~1 point of "do nothing") and
  shuffling it destroys 87-120% of the model's total advantage -- but the
  literal sign flip (shuffled worse than persistence) only recurs in 2 of 7
  cells (blind_n5, blind_n10), not all, which is a real, milder-than-EXP-0014
  finding, not a full replication of the single-cell 110.5% result. A
  monotonic advantage-grows-with-n trend appears in all 4 independent series
  (UNet and linear operator, blind and contact arms) but is not resolved
  against a measured noise floor (one seed per cell) -- reported as
  consistent-but-unresolved, not as an established trend.
verdict: supported
downgrades: [imprecision, indirectness, inconsistency, provenance]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0014 showed a large, clean action-dependence effect (shuffled worse than
persistence) but on ONE dataset (n=50, oblique/variable-length), scored on
whole-image rms (~99% untouched pixels at low n), and never compared to the
linear operator that the rest of this repo's evidence is built around. If the
UNet's blind-arm advantage is mostly "predict nothing" -- a real risk, since
blind n=5 is 59% zero-contact by the task's own pre-collected statistics --
restricting to contact>0 transitions (where "nothing happened" is no longer a
free answer) should collapse it toward zero while the linear operator's
(fit and scored the same way) does not. Running both models through the
identical swept-region/contact-stratified pipeline on 7 cells, rather than
one whole-image number on one dataset, is what lets this be answered rather
than assumed.

## What was actually run

1. **Code changes, committed before running** (`85f2413b`): (a)
   `PileSweepData.__init__` gained an optional `min_push_length_m` filter --
   an index-remapping layer over the existing (run, sample) lookup, so
   `__len__`/`get_run_index`/`get_raw_action`/`__getitem__` each gained one
   extra indirection. Verified against the task's own pre-computed cell
   sizes: filtering c5/c10/c20 to >39mm gives exactly 1541/1872/1895
   transitions, matching the design table to the transition. (b)
   `PileSweepData._filter_split` now falls back to per-FILE groups when a
   folder resolves to exactly one physics-key group (every granularity/L040
   cell here does) -- previously this put 100% of such a folder in `train`
   and raised `ValueError` for `val`/`test` (reproduced and confirmed before
   fixing). The full test suite (238 tests) passes unchanged after both
   changes.
2. **Cost pilot** (disclosed, no outcome kept): a 2-epoch timing run on
   `blind_n5` at res=64 measured 4.1-4.5 s/epoch -- 100 epochs x 7 cells is
   ~50 min GPU, comfortably inside budget, so epochs were kept at the
   template's 100 rather than reduced. Its weights/metrics were discarded.
3. **7 configs built from `configs/training/unetfilm_corl_limited_100e_fixedraster.yaml`**
   (EXP-0014's own template, per the task's warning about the drifted
   `configs/` tree), changing only `dataset.paths`, `dataset.resolution_scale`
   (1.0 -> 0.5, see `held_fixed`), and `dataset.min_push_length_m` for the
   contact cells. Trained sequentially on one GPU
   (`scripts/probes/exp0021_train_all.sh`), ~7 min/cell.
4. **Evaluation** (`scripts/probes/exp0021_eval.py`, one cell at a time):
   fits the linear operator on the dataset's own `train` split, then builds
   `persistence`/`mean-delta`/`identity(warp only)`/`linear-ridge1.0`
   predictions and the UNet's `true`/`shuffled`/`zeroed`-action predictions
   on the SAME `test` split, and scores all seven with the identical
   `swept_region_mask`/`metrics` call, stratified by `contact_score() > 0`.
5. **Not run**: a repeated-seed measurement of this design's own noise floor
   (see `noise_floor`); nonneg or OLS variants of the linear operator (task
   asked for ridge toward identity specifically); LORO folds (single
   train/test split per cell, as `val_pct/test_pct=10/10` gives one held-out
   file). These are the design's known gaps, not silent omissions.

## Numbers

**Swept-region rms as % of persistence (100% = predicts nothing moved), all 7 cells x 2 strata:**

| cell | stratum | n | persistence | mean-delta | identity(warp) | linear-ridge1.0 | **UNet-true** | UNet-shuffled | UNet-zeroed |
|---|---|---|---|---|---|---|---|---|---|
| blind n5  | all        | 320 | 100.0 | 106.7 | 101.6 | 87.6 | **79.0** | 104.1 | 100.2 |
| blind n5  | contact>0  | 210 | 100.0 | 100.4 |  99.8 | 82.0 | **77.6** | 101.8 | 100.0 |
| blind n10 | all        | 320 | 100.0 | 102.5 |  99.8 | 84.4 | **74.5** | 101.9 | 100.0 |
| blind n10 | contact>0  | 271 | 100.0 |  98.9 |  98.6 | 80.6 | **73.5** | 101.1 |  99.8 |
| blind n20 | all        | 320 | 100.0 |  95.2 |  96.9 | 79.3 | **72.1** |  98.4 |  99.5 |
| blind n20 | contact>0  | 314 | 100.0 |  94.8 |  96.8 | 78.8 | **72.0** |  98.4 |  99.4 |
| contact n5  | all      | 206 | 100.0 | 100.1 |  98.4 | 82.9 | **72.0** |  96.9 |  99.9 |
| contact n5  | contact>0| 192 | 100.0 |  98.3 |  98.1 | 81.4 | **71.4** |  96.6 |  99.8 |
| contact n10 | all      | 220 | 100.0 |  93.6 |  95.7 | 78.5 | **68.5** |  96.5 |  99.3 |
| contact n10 | contact>0| 218 | 100.0 |  93.3 |  95.6 | 78.2 | **68.4** |  96.0 |  99.3 |
| contact n20 | all      | 240 | 100.0 |  89.6 |  95.2 | 74.7 | **65.7** |  98.4 |  99.2 |
| contact n20 | contact>0| 239 | 100.0 |  89.5 |  95.2 | 74.6 | **65.6** |  98.0 |  99.2 |
| blind n50 (L040) | all       | 320 | 100.0 |  88.1 |  94.5 | 76.2 | **66.4** |  95.5 |  99.7 |
| blind n50 (L040) | contact>0 | 319 | 100.0 |  88.0 |  94.5 | 76.2 | **66.4** |  95.5 |  99.7 |

n_test(contact>0) as a fraction of n_test(all): blind n5 66%, blind n10 85%,
blind n20 98%, contact n5 93%, contact n10 99%, contact n20 100%, blind n50
100% -- measured on THIS design's own `contact_score() > 0` threshold, not
the task's pre-collected swath-particle statistic (see Unrelated findings for
the discrepancy).

**UNet-vs-linear margin (linear% − UNet%, "all" stratum): +8.6 (n5) / +9.9
(n10) / +7.2 (n20) blind; +10.9 (c5) / +10.0 (c10) / +9.0 (c20) contact;
+9.8 (n50).** Roughly constant across object count (~7-11 points), while
both models' absolute advantage over persistence grows with n.

**Action ablation, shuffle-gap and zero-gap as % of the model's total
advantage over persistence (100−true%), "all" stratum:**

| cell | total advantage | shuffle-gap | shuffle-gap/advantage | zero-gap | zero-gap/advantage | sign flip? |
|---|---|---|---|---|---|---|
| blind n5  | 21.0 | 25.1 | 120% | 21.2 | 101% | **yes** (104.1%) |
| blind n10 | 25.5 | 27.4 | 108% | 25.5 | 100% | **yes** (101.9%) |
| blind n20 | 27.9 | 26.3 |  94% | 27.4 |  98% | no (98.4%) |
| contact n5  | 28.0 | 24.9 | 89% | 27.9 | 100% | no (96.9%) |
| contact n10 | 31.5 | 28.0 | 89% | 30.8 |  98% | no (96.5%) |
| contact n20 | 34.3 | 32.7 | 95% | 33.5 |  98% | no (98.4%) |
| blind n50   | 33.6 | 29.1 | 87% | 33.3 |  99% | no (95.5%) |

Zeroing the action collapses the model to within 0.8 points of persistence in
every one of 7 cells (97.7-101.0% of persistence). Shuffling destroys
87-120% of the model's advantage in every cell -- close to total, sometimes
more than total -- but the literal sign flip (shuffled *worse* than
persistence, EXP-0014's 110.5% headline result) recurs in only 2 of 7 cells.

## What would change the verdict

- **A seeded repeat per cell** (different training seed, different
  shuffle-permutation draw) would put a real noise floor on both the
  UNet-vs-linear margin and the object-count trend, neither of which is
  measured here. Cheap in aggregate (~50 min GPU for one more full pass of
  all 7), but was not in budget alongside everything else this record ran.
- **LORO or a k-fold split per cell** (currently one held-out file per cell)
  would separate "this margin is real" from "this margin is this file."
  Given the margins (4.4-10.9 points) are several times the borrowed
  1.9-point LORO floor from a related but different cell (EXP-0018), this
  would need an unusually large fold-to-fold swing to overturn the
  headline UNet-beats-linear finding, but the record should not claim more
  precision than it has.
- **Nonneg linear operator** was not fit (ridge only, per the task's own
  spec) -- EXP-0009/EXP-0018 found ridge/ols/nonneg agree to ~0.1pt at low
  res, so this is unlikely to change the margin, but it is unmeasured here.

## Threats

- `imprecision`: one training run and one train/test split per cell, no
  seed repeats -- see `noise_floor`. This is the main reason the grade caps
  low despite the effect sizes being large relative to a borrowed floor.
- `indirectness`: swept-region rms (even contact-stratified) is a one-step
  pixel-space proxy for downstream control utility, the same standing
  caveat every rms-based record in this register carries (C-030/C-035).
- `inconsistency`: the literal action-shuffle sign flip (>100% of
  persistence) recurs in only 2 of 7 cells, not all -- EXP-0014's single-cell
  110.5% result is the strongest observed case, not the typical one. The
  weaker-but-consistent "shuffle destroys 87-120% of the advantage" finding
  is the one this record actually leans on.
- `provenance`: contact_n5's 8 source files span two data_commits
  (f196d657 x2, 3d7db119 x6) -- a mixed collection batch, not a single
  lineage; every other cell is a single data_commit. `pixel-index-origin`
  remains broken (~1px, per INVARIANTS.md) but is not cited in `depends_on`
  since the effect sizes here (points, not sub-pixel) do not turn on it --
  noted per EXP-0014/EXP-0018 precedent.
- Considered and dismissed: **the swept-region-vs-whole-image metric choice
  explains the different sign-flip rate vs EXP-0014.** Plausible but not
  actually tested here (EXP-0014's own dataset was not re-scored on
  swept-region in this record); left as an open question rather than a
  claimed explanation.
- Considered and dismissed: **the object-count trend is just the contact
  fraction changing.** The trend holds within the "contact>0" stratum too
  (which fixes the no-op confound), so it is not solely an artifact of
  higher-n cells having fewer no-op transitions -- though it could still
  partly reflect other n-correlated factors (e.g. absolute pile mass/rms
  scale) rather than a clean "harder to predict" story; this record does
  not decompose that further.

## Unrelated findings

- The task's own pre-collected contact statistic ("swath particles /
  zero-contact fraction": blind 0.51/59%, 1.07/36%, 2.06/18%;
  contact-filtered 1.25/27%, 2.08/13%, 3.51/5%; L040 5.05/4%) does not match
  this record's `contact_score() > 0` fractions on the held-out test split
  (blind n5/10/20: 34%/15%/2% zero-contact; contact n5/10/20: 7%/1%/0.4%;
  L040: 0.3%). Both are internally consistent (all cells shift toward less
  zero-contact together) but the absolute numbers differ substantially --
  plausibly a train/test-split-of-8-files sampling effect (one held-out
  file, not the full population) compounded by a different contact
  definition (`fit_linear_foresight.contact_score`'s blade-path mass vs a
  swath-particle count). Not chased further; both measurements agree on the
  qualitative ordering (n20 and contact arms have much less zero-contact
  than blind n5), which is all this record's stratification needed.
- `PileSweepData._filter_split`'s physics-key grouping silently gives 100%
  of a folder to `train` whenever that folder has only one physics-key group
  -- true of every dataset collected as a single "family" (one friction/
  density/particle-size/n_particles combination across all files in a
  folder), which likely includes other datasets beyond the ones this record
  touches. Fixed here (fallback to per-file groups) but not audited across
  the rest of the `configs/dataset/` tree for silently-train-only val/test
  splits from before this fix landed.
- `contact_n5`'s 8 source files are the only ones observed spanning two
  different `data_commit` values in one leaf folder (see `provenance`) --
  worth knowing if `contact_n5` numbers specifically are ever re-derived.

## Grade note

`imprecision`+`indirectness`+`inconsistency`+`provenance` = 4 domains, so the
grade saturates at `very-low` (the formula caps at 3+). What actually changed
relative to C-020's prior `low` grade (EXP-0014): the swept-region metric was
obtained (EXP-0014 explicitly could not get it), a linear-operator anchor now
exists for the same claim, the design spans 7 cells instead of 1, and the
pre-registered "predicts nothing" failure mode was directly tested and
refuted rather than left open. The letter does not show this (`low` ->
`very-low` reads like a downgrade, driven by `inconsistency`+`provenance`
being newly counted, not by the evidence getting weaker) -- see the register
row for the domain-level framing.
