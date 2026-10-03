---
# ---- identity -------------------------------------------------------------
id: EXP-0018
title: >
  A value readout trained on 23.5k DIVERSE after-sweep states from five source
  corpora against a 2000-goal library: held-out-GOAL and held-out-STATE now
  agree, and the readout survives leaving an entire source corpus out for
  `lyapunov` but not for `mass_in_region`
tier: T1
mode: exploratory
date: 2026-09-15

# ---- the claim ------------------------------------------------------------
claim: >
  With 23,520 after-sweep states sampled across FIVE source corpora
  (overnight_randlen_train, Sean inbetween/piled/scattered, slates_multistep),
  87-dim analytic occupancy descriptors (dmdc_baseline.occupancy_descriptors,
  n_fourier=8) as the state and goal embedding, and a 2000-goal library whose
  goals are legal same-object-count 20-cube configurations, the static readout
  h(phi_state, phi_goal) -> value beats the best-capacity `goal_only` baseline
  on HELD-OUT GOALS by `value_readout_r2` +0.637+-0.019 (lyapunov),
  +0.422+-0.011 (mass_in_region) and +0.598+-0.014 (signed_mass_in_region) at
  MLP(128,128) capacity over 3 goal splits, and by +0.511/+0.199/+0.437 at
  LINEAR capacity. Training on diverse states removes EXP-0017's held-out-STATE
  pathology: held-out-STATE (grouped) now tracks held-out-GOAL to within ~0.08
  `value_readout_r2` in every cell and is POSITIVE at linear capacity for all
  three value functions (+0.642/+0.438/+0.617), where EXP-0017 measured
  -0.075/-0.134 on a single slate pool. Leaving an entire SOURCE CORPUS out
  degrades the readout and the degradation is corpus- and value-function
  specific: for `lyapunov` the increment over `goal_only` stays positive on all
  five held-out corpora (linear +0.21 to +0.71, mean +0.522, spread 0.220);
  for `mass_in_region` it collapses to +0.121+-0.095 and for
  `signed_mass_in_region` it is dominated by one outright failure
  (held-out randlen, -0.277). Scope: analytic descriptors only, `diff`
  feature mode, a 6000-state stratified subsample for fitting, and NO `slateN`
  score -- nothing about action ranking follows from these numbers.

prediction:
  discriminating: false
  statement: >
    None committed. This is exploratory engineering (get the data and the
    harness right, find bugs); the record reports what came out.

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0018-value-readout-diverse-states/code/stage3_fit.py
  data: ["Genesis/data/overnight_randlen_train", "Genesis/data/Sean",
         "Genesis/data/slates_multistep/n20_L20mm",
         "Genesis/data/slates_multistep/n20_L40mm"]
  code_path: experiments/EXP-0018-value-readout-diverse-states/code/
  seed: 0
  split: >
    Three families, ALL built over GROUPS, never over rows. `group_id` is the
    source `*_data.pt` file for randlen and Sean (independent transitions) and
    `ms:<corpus>:s<slate_idx>` for slates_multistep, whose three steps are one
    trajectory and therefore near-duplicates; 1405 groups over the 6000 fitted
    states. (1) held-out GOAL: random 1600/400 splits of the 2000-goal library,
    seeds 100-102, all states in both sides. (2) held-out STATE: 80/20 over
    GROUPS, seeds 200-201, all goals shared. (3) held-out SOURCE CORPUS:
    leave-one-corpus-out, 5 folds, all goals shared. Inner ridge-lambda CV is
    `GroupKFold(3)` (`GroupRidgeCV`, added here) grouped by state-group for the
    state/corpus families and by goal id for the goal family -- NOT `RidgeCV`'s
    default leave-one-out GCV, which is row-random.
  runtime: "~12 min for the 51 fits; ~3 min for the cache; ~1.5 min for the goal library. CPU only (CUDA_VISIBLE_DEVICES=\"\", OMP_NUM_THREADS=4)"
  runs: [RUN-0001]

budget:
  declared: "80 min / 160k tokens (delegated subagent, declared HARD, CPU only)"
  spent: "~70 min / ~110k tokens"
  outcome: within

design:
  varied:
    split_family: [held-out-goal, held-out-state, held-out-source-corpus]
    held_out_corpus: [randlen, sean_inbetween, sean_piled, sean_scattered, slates_multistep]
    capacity: [linear, poly2, "MLP-128-128"]
    value_fn: [lyapunov, mass_in_region, signed_mass_in_region]
    split_seed: [100, 101, 102, 200, 201]
  held_fixed:
    states: "23520 after-sweep (`states_`) states, 5 corpora; 6000 stratified for fitting"
    embedding: "dmdc_baseline.occupancy_descriptors(n_fourier=8), 87-dim, for BOTH state and goal"
    feature_mode: diff
    rasteriser: "transforms.functional.particles_to_occupancy, footprint_radius=0.5*CUBE_SIZE/pitch=0.5*0.005/0.002=1.25 vox, 64x64, DEFAULT_BOUNDS -- ONE function for states and goals"
    goal_representation: "legal same-object-count 20-cube configuration (mean occupancy over K=3 samples), NOT a solid mask"
    n_goals: 2000
    value_targets: "computed from the goal MASKS, per Baselines/common/goals.py and control_utility_test.lyapunov"
    mlp: "(128,128) relu alpha=1e-3 max_iter=120 early_stopping"
  baselines: [mean_only, goal_only, state_only]
  metric: "value_readout_r2"

noise_floor: >
  Held-out GOAL: sd over 3 splits is 0.002-0.021 `value_readout_r2`, so
  differences below ~0.05 are UNRESOLVED. Held-out STATE: sd over 2 splits is
  0.001-0.041, n=2, so treat ~0.08 as the floor. Held-out SOURCE CORPUS has
  NO repeat per fold: each corpus is one number and the +-0.195 to +-0.408
  quoted for that family is BETWEEN-CORPUS variation, not noise -- no ranking
  of corpora against each other is claimed. `mean_only` is exactly 0.000 in all
  51 cells, confirming the R2 normalisation is against the TRAINING mean.

depends_on: [occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x,
             readout-cv-folds-slate-aware]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  (1) HEADLINE, held-out GOAL, metric `value_readout_r2`, increment over
  best-capacity `goal_only`, MLP capacity, 3 splits: lyapunov +0.637+-0.019
  (raw 0.776+-0.021 vs goal_only 0.138), mass_in_region +0.422+-0.011 (raw
  0.671+-0.002 vs 0.248), signed_mass_in_region +0.598+-0.014 (raw 0.779+-0.006
  vs 0.181). At LINEAR capacity the increments are +0.511/+0.199/+0.437 -- all
  far outside the ~0.05 floor, unlike EXP-0017 where linear `diff` was BELOW
  `goal_only` for both mass functions. Capacity ordering linear < poly2 < MLP
  reproduces for lyapunov (0.649 / 0.697 / 0.776); poly2 was run in ONE cell
  only. (2) THE NEAR-DUPLICATE PROBLEM IS FIXED. EXP-0017, training on one
  slate pool, got NEGATIVE held-out-STATE `value_readout_r2` at linear capacity
  (-0.075, -0.134) and a large gap to held-out-GOAL. Here held-out-STATE
  (grouped) is +0.642/+0.438/+0.617 linear and +0.693/+0.564/+0.746 MLP, within
  ~0.08 of the corresponding held-out-GOAL cells for every value function. The
  selected ridge lambda is 62-2976 across all 31 linear/poly2 cells -- the sane
  regime, not the 0.01-1 that the row-random-fold bug produced. (3) HELD-OUT
  SOURCE CORPUS is the honest test and it separates the value functions.
  `lyapunov`, linear, increment over goal_only per held-out corpus: randlen
  +0.706, sean_inbetween +0.305, sean_piled +0.205, sean_scattered +0.691,
  slates_multistep +0.702 (mean +0.522, between-corpus spread 0.220) -- always
  positive, so a `lyapunov` readout transfers to a pile distribution it never
  saw, though the magnitude is not. `mass_in_region` is far weaker
  (+0.121+-0.095: randlen +0.100, sean_scattered -0.011, slates_multistep
  +0.069) and `signed_mass_in_region` is inconsistent (+0.757 on sean_scattered,
  +0.744 on slates_multistep, but -0.277 on held-out randlen and +0.010 on
  sean_piled). The MLP does NOT reliably beat linear under corpus shift: it is
  better on slates_multistep (+0.788 vs +0.702) and worse on randlen (+0.643 vs
  +0.706) and much worse on sean_inbetween (-0.048 vs +0.305, the only negative
  `lyapunov` increment anywhere) -- i.e. the extra capacity overfits the
  training corpora's state distribution. (4) A BUG FOUND: `ValueReadout` as
  written for EXP-0017 selected its ridge lambda with `RidgeCV`, whose default
  is leave-one-out generalized CV, i.e. ROW-RANDOM -- exactly the
  `readout-cv-folds-slate-aware` defect that INVARIANTS.md recorded the
  instrument as being free of. That note has been corrected, and
  `GroupRidgeCV` + `ValueReadout.fit(..., cv_group_by=)` added so grouped inner
  folds can be requested (they must be requested; the default is unchanged).
  (5) Artefacts persisted: 2000-goal library `experiments/temp/goal-states/dataset_v2.pt`
  (dataset.pt untouched), the full feature/value/split cache
  `experiments/temp/exp0018-value-readout/cache.pt` (23520x87 states, 2000x87
  goals, three 23520x2000 value matrices, per-state corpus/group/push-length
  labels and the reproducible `(corpus, group, file, row)` state index), the
  fit subsample index, and all 51 fitted readouts as joblib.
verdict: inconclusive
downgrades: [indirectness, incomplete-design, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## What was uncommitted (dirty tree)

Branch `baselines/overnight` at `6ea03278`. Modified and uncommitted at run time:
`.claude/skills/{experiment-log,project-overview}/SKILL.md`,
`Genesis/sandbox_manipulation_clean.py`, `docs/piled_collection.md`,
`experiments/{COMMANDS.jsonl,INVARIANTS.md,METRICS.md,REGISTER.md,TEMP_LOG.md}`,
`scripts/probes/{pool_common,pool_inspect,pool_survey}.py`,
`weights/MODEL-000{1,2,3}-*/tests.md`. Untracked: `.claude/orchestrator-notes.md`,
`.claude/skills/{data-collection,subagent-experimenter}/`,
`Baselines/common/goal_configs.py`, `Genesis/binned_slate_{collection,dataset}.py`,
`Genesis/data/{Sean,slates_binned}/`, `datasets/`, `docs/CODEMAP.md`,
`docs/experimental_design/jepa_based_encoder.md`, `experiments/EXP-001{1..8}-*/`,
`scripts/probes/binned_pool_cache.py`.

Load-bearing for this record: `Baselines/common/goal_configs.py` (the goal
configurations), `Genesis/data/Sean/` (three of the five source corpora),
`experiments/EXP-0017-*/code/value_readout.py` (the instrument, **modified by this
record** — `GroupRidgeCV` and the `cv_group_by` argument), and this record's own tree.
This record's own numbers were produced by uncommitted code, so `commit: 6ea03278`
does not reconstruct the run on its own; the scripts are in `code/`.

## Why the verdict is `inconclusive`

No prediction was committed (`mode: exploratory`), and the cell that would decide
whether any of this is usable — a `slateN` score through this readout — was not run.
`downgrades`: `indirectness` (a regression R² on a static mapping, not a ranking
metric; `slateN` is what decides), `incomplete-design` (the cells below),
`untested-dependency` (all three `depends_on` tags are `unchecked`/`broken`).

## What was actually run

**RUN-0001**, three stages, all CPU:

* `code/stage1_generate_goals_v2.py` — 2000 goals (800 rectangles, 810 letters over
  T/A/O/S/G/L/E/H/C with rotation and scale, 250 ellipses, 140 two-blob), each with
  K=3 legal 20-cube configurations; 6000/6000 passed `assert_no_penetration`.
  Written to `experiments/temp/goal-states/dataset_v2.pt`; EXP-0015's `dataset.pt`
  was not touched.
* `code/stage2_build_cache.py` — 23,520 after-sweep states: randlen 7680 (all five
  spawn/particle subdirectories, 40 rows/file), sean_inbetween 3640, sean_piled 4200,
  sean_scattered 2000 (8 rows/file over 1230 files), slates_multistep 6000 (both
  L20mm and L40mm, 20 rows/file over all 3 steps). Push length spans 0.2–80.0 mm,
  mean 39.2 mm. One rasteriser for everything.
* `code/stage3_fit.py` — 51 fits, all persisted.

## Cells NOT run (cost if someone wants them)

* `slateN` through this readout — the cell that decides usability. ~20 min.
* `poly2` anywhere except held-out-GOAL/rep0/lyapunov. It is by far the most
  expensive capacity here (3915 features from the 87-dim `diff`), ~80 s/cell.
* MLP on the corpus split for `mass_in_region` and `signed_mass_in_region`
  (only `lyapunov` ran). ~10 fits, ~5 min — this is the one I would run first.
* `concat` feature mode (needed once the state embedding is a learned latent
  whose dimension differs from the goal's). ~5 min.
* Any repeat of a corpus fold, so the corpus family has no within-fold interval.
* A learned-encoder latent in place of the descriptors — the follow-up this
  cache was built for. `cache.pt` carries the state index so the same states
  can be re-embedded.

## Unrelated findings

* **`ValueReadout` did NOT use grouped inner CV.** EXP-0017's instrument selected
  its ridge lambda with `RidgeCV(alphas=...)`, whose default is leave-one-out
  generalized CV over ROWS — the `readout-cv-folds-slate-aware` defect that
  `INVARIANTS.md` explicitly recorded the instrument as being free of. EXP-0017's
  held-out-STATE linear cells are the ones most exposed (a held-out-slate outer
  split with row-random inner folds). The invariant's note is corrected, and
  grouped folds are now available via `cv_group_by`, but **the default is still
  row-random** — a later caller who omits the argument gets the bug back.
* **`Genesis/data/overnight_randlen_train` is five subdirectories**, not a flat
  file list: `mixed_n20`, `piled_n20`, `piled_n50`, `scattered_n20`,
  `scattered_n50` (192 `*_data.pt` total). Spawn mode and particle count live in
  the directory name. A `glob('*.pt')` at the top level returns nothing, which is
  a silent-empty failure mode. Added to `docs/CODEMAP.md`.
* **`slates_multistep/*/manifest.json::batches`** is the only mapping from
  `_{batch_idx}_data.pt` to `(slate_idx, step_idx)`; the filename index is neither
  slate-major nor step-major in an inferable way. Added to `docs/CODEMAP.md`.
* All three value functions are one matmul away from the occupancy: with
  `A = occ @ dist_fields` and `B = occ @ masks`, `lyapunov = A/mass`,
  `mass_in_region = B`, `signed_mass_in_region = 2B - mass`. 23520x2000x3 targets
  take 6 s on 4 CPU threads this way; the per-goal Python loop EXP-0015 used does
  not scale to 2000 goals.
* `mask_to_configuration` costs ~1 ms in the `grid` branch but the per-goal cost is
  strongly shape-dependent (letters fall back to rejection sampling): 2000 goals x 3
  samples took 82 s, with the first 500 in 0.7 s and the last 500 in ~14 s.

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- ISS-002 (open): goal FEATURES were descriptors of rasterised configurations while TARGETS came from the raw (pre-fix, mirrored) mask arrays; this record's cache and its 51 persisted readouts are stale and must be regenerated before any number here is reused.
