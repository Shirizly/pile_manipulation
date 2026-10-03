---
# ---- identity -------------------------------------------------------------
id: EXP-0064
title: >
  FleX carrot piles grouped by object count: the ported GNN read every push z-mirrored; with the frame fixed,
  node accuracy still falls with pile size but slateN no longer rises -- and the trained GNN only matches an
  untrained push-field heuristic, beating it on small piles and losing on large ones
tier: T1
mode: exploratory
date: 2026-10-03
hypothesis: H-scene (object count changes task difficulty and model usefulness)

# ---- the claim ------------------------------------------------------------
claim: >
  On FleX carrot piles from one centred blob whose size is set by the carrot count (DS-0021 train,
  DS-0022 test: 50 same-state slates of 51-100 clean obj-biased pushes per group), for the dyn-res GNN
  (PropNetDiffDenModel) trained with the source repo's own loop on <= 30 FPS-sampled true particles,
  scored on all-particle truth over 17 random point goals and over the project's 3 mask goals x 3 value
  functions, with pools equalised to K = 51 and escaped rows dropped: (a) node_accuracy falls with object
  count; (b) slateN rises with object count; (c) the slateN rise from 10-30 to 400-500 is larger for the
  trained GNN than for the untrained `field` baseline (s0 + the GNN's own action-input field), i.e. part
  of the rise is the model's, not the task's. As-run (MODEL-0011, mirrored action) and corrected-frame
  (MODEL-0012) models are both scored.

# prediction: omitted -- exploratory. The study was run in the source repo without a committed
# prediction; the re-scoring here was designed after seeing the as-run per-group table.

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: d72bb304
  dirty: true                     # other sessions' uncommitted work + this port's files; nothing committed per user instruction
  data_commit: "source repo ~/Code/dyn-res-pile-manip @ 5c9eca9, dirty (collectors uncommitted there) -- issues.md I-10"
  script: "RUN-0003/0004 (source repo): code/train_gnn_dyn_grouped.py, code/eval_grouped_capture.py; RUN-0005/0006/0007 (this repo): code/train_gnn_dyn_grouped.py, code/eval_extended.py + code/summarize_extended.py, code/frame_check.py"
  data: [DS-0021, DS-0022]
  code_path: "Baselines/GNN/model/gnn_dyn.py::PropNetDiffDenModel (byte-identical to source); code/dataset_grouped_particles.py (ported adapter, + action_z_sign); nearest-node carry to all particles; mask goals via Baselines/common/goals.py on a local 64x64 +-7.2 binary disk raster (row x, col -z) -- NOT eval_report's FleX image-mask path"
  seed: "training 42; eval FPS np.random.seed(1000*rep + state_idx), 3 reps; K-equalised subsets rng(0)"
  split: "DS-0021 per-group 90/10 by state (1800/200 windows, frames 0-5 of each trajectory); DS-0022 all 200 states test"
  runtime: "source: collection ~1 day, training ~1h20m GPU; here: retrain ~100 min GPU (RTX 4070 Laptop), extended eval ~6 min per model"
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004, RUN-0005, RUN-0006, RUN-0007]

budget:
  declared: "port + reorganise + standards re-score within a 3 h session (user)"
  spent: "~2.5 h incl. the corrected retrain"
  outcome: within

design:
  varied:
    object_count_group: ["10-30", "50-70", "100-150", "400-500"]
    model: [MODEL-0011 (as run, z-mirrored action), MODEL-0012 (corrected frame, RUN-0005), field (untrained), persistence]
    scoring: [as-run definition (goal 0, tracked-node truth, escaped kept, full pool), 17 point goals all-particle truth, K=51 equalised, mask goals x 3 value functions]
  held_fixed:
    pile_generator: "count_target single centred blob; footprint grows with count (issues.md I-9)"
    action_sampler: "obj_biased (start near a particle + jitter, uniform end), push length 0.96-9.6 FleX units"
    node_budget: "min(30, target count) -- the 10-30 group gets fewer nodes (I-8)"
    training_recipe: "source loop unchanged (user decision); adapter's build_action_delta encoding (I-3); escaped training rows NOT filtered (I-2)"
    epochs: "100 for both models (as-run stopped by hand at 100 of 300; RUN-0005 configured to 100)"
  baselines: [persistence, field, random]
  metric: "slateN_point_goal, slateN (mask goals: random_quadrant/ring_O/T x lyapunov/mass_in_region/signed_mass), node_accuracy (experiments/METRICS.md)"

noise_floor: "Training seed noise unmeasured (one seed per model). FPS node-sampling: 3 seeded reps averaged per state; the seeded re-score reproduces the unseeded as-run overall numbers within 0.01 (accuracy 0.219 vs 0.219, capture 0.609 vs 0.616). State-bootstrap 95% CIs per group (n = 50 states) are reported for every cell; group contrasts are unpaired (different states)."

depends_on: [flex-action-frame-neg-z]
establishes: []

# ---- outcome --------------------------------------------------------------
result: "Corrected-frame MODEL-0012 (K=51, 17 point goals, all-particle truth): node_accuracy 0.579/0.570/0.557/0.481 by group (as-run MODEL-0011 0.223/0.243/0.214/0.156); slateN_point_goal 0.907/0.918/0.921/0.886, 400-500 minus 10-30 -0.020 [-0.041, +0.001] (as-run +0.125 [+0.063, +0.186], field +0.056 [+0.043, +0.069]); mask slateN mean3 0.574/0.693/0.693/0.612, +0.037 [-0.007, +0.084] (field +0.100). MODEL-0012 minus untrained field, paired: point +0.026 [+0.009, +0.043] / +0.007 / +0.003 / -0.057 [-0.076, -0.039]; mask -0.057 / -0.030 / -0.090 / -0.136 [-0.169, -0.104]. (a) supported, (b) and (c) refuted for the corrected model; the as-run rise was an artefact of the mirrored action."
verdict: refuted
downgrades: [indirectness, provenance, selection, imprecision]
grade: very-low
---

## Why this test discriminates

The as-run study read "capture rises with object count" as the GNN becoming more useful for control on
bigger piles. That reading needs the rise to (i) survive pool-size equalisation (valid pools shrink with
count, and slateN falls with pool size, C-031), (ii) survive goal breadth and all-particle truth, and (iii)
exceed the rise of a ranker that has no learned dynamics at all. The `field` baseline (predict
s0 + the GNN's own action-input displacement field) is that ranker: if it rises as much, the rise is a
property of the scene/task/normalisation, not of the model.

## What was actually run

- RUN-0001/0002 (source repo, ported): collected DS-0021 / DS-0022 with object-count-targeted piles.
- RUN-0003/0004 (source repo, ported): trained MODEL-0011 with the source loop; scored per group
  (`artifacts/RUN-0004-eval/eval_results_grouped.json`; the original write-up is `reports/REPORT.md`).
- RUN-0007 (here): found the adapter reads the stored action as (x, z) while the data's frame is (x, -z)
  (`results/frame_check.json`; issues.md I-1). The invariant `flex-action-frame-neg-z` now `holds` via a pytest
  on DS-0021/22 with a negated control.
- RUN-0005 (here): the same training with `action_z_sign: -1` as the ONLY change (100 epochs) → MODEL-0012.
- RUN-0006 (here): extended re-score of MODEL-0011, MODEL-0012, `field`, `persistence` (see RUN.md).
- Dirty tree: commit d72bb304 plus other sessions' uncommitted edits (see `git status`) and this port's own uncommitted files (EXP-0064 code/, the adapter's `action_z_sign` knob, tests/test_flex_dataset.py additions); the user asked for no commits this session.
- Not run, by user instruction: anything through the source repo; FleX re-collection (not possible here).
  Not run: escape-filtered or baseline-action-encoding retrains (issues.md I-2, I-3), training seeds.

## Numbers

Full tables: `results/extended_summary.md` (generated; JSON beside it). Headline cells, K = 51, escaped rows
dropped, mean [95% state-bootstrap CI], 50 states per group:

| model | metric | 10-30 | 50-70 | 100-150 | 400-500 | 400-500 − 10-30 |
|---|---|---|---|---|---|---|
| MODEL-0011 as run | slateN_point_goal (17 goals, full) | 0.561 | 0.636 | 0.650 | 0.686 | +0.125 [+0.063, +0.186] |
| MODEL-0011 as run | slateN mask mean3 | 0.288 | 0.408 | 0.405 | 0.412 | +0.123 [+0.064, +0.182] |
| MODEL-0011 as run | node_accuracy (as-run def) | 0.223 | 0.243 | 0.214 | 0.156 | |
| field (untrained) | slateN_point_goal | 0.886 | 0.904 | 0.922 | 0.942 | +0.056 [+0.043, +0.069] |
| field (untrained) | slateN mask mean3 | 0.649 | 0.722 | 0.770 | 0.749 | +0.100 [+0.063, +0.136] |
| field (untrained) | node_accuracy | 0.504 | 0.424 | 0.342 | 0.075 | |
| MODEL-0012 corrected | slateN_point_goal | 0.907 | 0.918 | 0.921 | 0.886 | −0.020 [−0.041, +0.001] |
| MODEL-0012 corrected | slateN mask mean3 | 0.574 | 0.693 | 0.693 | 0.612 | +0.037 [−0.007, +0.084] |
| MODEL-0012 corrected | node_accuracy | 0.579 | 0.570 | 0.557 | 0.481 | |
| MODEL-0012 − field (paired) | slateN_point_goal | +0.026 [+0.009, +0.043] | +0.007 | +0.003 | −0.057 [−0.076, −0.039] | |
| MODEL-0012 − field (paired) | slateN mask mean3 | −0.057 [−0.104, −0.011] | −0.030 | −0.090 | −0.136 [−0.169, −0.104] | |
| persistence | all | 0 | 0 | 0 | 0 | |

Reproduction of the as-run table (as-run scoring definition, seeded FPS): accuracy 0.223/0.243/0.214/0.156
vs source 0.224/0.245/0.212/0.156; capture 0.470/0.639/0.637/0.691 vs 0.475/0.646/0.633/0.709.

Pool facts (DS-0022): valid actions/state 96.1/92.8/87.7/78.6; escaped-in-valid rows/state
0.56/1.90/2.24/2.80; share of the pool that moves none of the tracked nodes 0.105/0.051/0.029/0.007;
mean − best true spread (17 goals) 3.55/3.10/2.65/1.60 FleX units.

## What would change the verdict

- MODEL-0012 (corrected frame, run): (b) and (c) fail -- the source report's "bigger piles make the model more
  useful" reading is withdrawn. What remains open, one seed: the trained GNN's edge over the untrained field
  heuristic shrinks and reverses with pile size (C-068).
- Training seeds (each ~100 min) for both arms; an escape-filtered and a baseline-encoding retrain
  (issues.md I-2/I-3).
- Closed-loop use: none of this is closed-loop. FleX closed-loop MPC does not exist in this repo.

## Threats

- Object count is confounded with pile footprint and with node count (I-8, I-9): this is a pile-SIZE
  axis, not object count at fixed footprint, and not clumped-vs-scattered.
- Normalised capture is not comparable across groups even at equal K when the true spread differs ~2x
  (METRICS.md caveat 3); regret in FleX units is reported beside it in `results/extended_summary.md`.
- The mask-goal slateN uses a local binary raster, not eval_report's image-mask path; not comparable
  with EXP-0061/0062 numbers.
- Unpaired group contrasts (different states per group).

## Interpretation (2026-10-03, after RUN-0005/0006)

- The as-run "capture rises with object count" came from a model whose push input was mirrored: such a model
  ranks mostly by where mass is, which a bigger pile makes easier. With the push read correctly the rise is gone.
- The corrected GNN's node_accuracy (0.48-0.58) far exceeds the field heuristic's (0.08-0.50), yet the
  heuristic ranks pushes as well (point goals) or better (mask goals), increasingly so on large piles --
  another case of accuracy and slateN disagreeing across model types (H-metric), and a model x scene
  interaction (H-scene): a learned particle model with a fixed 30-node budget is worth less, relative to a
  geometric push heuristic, as the pile grows. One seed; node budget and pile footprint are confounded with
  count (I-8, I-9).

## Unrelated findings

- The FleX solver's explosion rate rises with pile size (valid 96 % → 79 %), and the collector's validity
  flag misses escaped particles in 3.5-14.6 % of DS-0021 rows (rising with count) — a data-quality
  issue for any FleX carrot corpus from this collector (DS-0019/20 audits found the same class).
- The source report's two infrastructure fixes (PyFleX `voxelize.cpp` unbounded loop; native segfault
  crash ledger) live uncommitted in the source repo (issues.md I-10).
