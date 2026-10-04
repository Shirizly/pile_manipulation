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
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004, RUN-0005, RUN-0006, RUN-0007, RUN-0013, RUN-0014, RUN-0015]   # RUN-0013..0015: image-based families on the same program (section below)

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

## Overnight follow-ups (RUN-0008..0012, 2026-10-04): escape filter, original encoding, seeds

Each is RUN-0005 with ONE change, 100 epochs, scored by the same harness (K = 51, all-particle truth).
A second untrained baseline, `field_orig` = s0 + the ORIGINAL baseline action encoding, was added.

| model | node acc (clean) | slateN point | slateN mask | 400-500 − 10-30 point | − own untrained field, point (overall) | − own field, mask (overall) |
|---|---|---|---|---|---|---|
| tube, seed 42 (MODEL-0012) | 0.596 | 0.908 | 0.643 | −0.020 [−0.041, +0.001] | −0.005 [−0.015, +0.004] | −0.078 [−0.098, −0.058] |
| tube, seed 43 (RUN-0010) | 0.592 | 0.914 | 0.678 | −0.012 [−0.032, +0.008] | −0.000 [−0.009, +0.008] | −0.039 [−0.058, −0.020] |
| tube, seed 44 (RUN-0011) | 0.594 | 0.901 | 0.649 | −0.027 [−0.048, −0.007] | −0.016 [−0.026, −0.006] | −0.065 [−0.086, −0.045] |
| tube, escaped rows dropped (RUN-0008) | 0.593 | 0.906 | 0.645 | −0.019 [−0.040, +0.002] | −0.009 | −0.077 |
| **orig encoding (RUN-0009)** | **0.640** | **0.958** | **0.706** | −0.008 [−0.015, +0.001] | **+0.011 [+0.008, +0.014]** vs field_orig | **+0.027 [+0.012, +0.043]** vs field_orig |
| untrained field (tube) | 0.388 | 0.914 | 0.722 | +0.056 [+0.043, +0.069] | | |
| untrained field_orig | 0.636 | 0.948 | 0.682 | −0.027 [−0.034, −0.020] | | |

By group, GNN minus its own untrained field (point goals): tube seeds +0.016..+0.026 at 10-30 and
−0.047..−0.070 at 400-500 (the C-068 pattern replicates in all 3 seeds); orig encoding −0.003 [−0.007, +0.001]
at 10-30 rising to +0.018..+0.020 at 100-500 (mask −0.059 → +0.055; moved-node accuracy +0.001 → +0.030).

Reading:
- **Seed noise** (3 tube seeds): node accuracy sd ~0.002, slateN point ~0.007, mask ~0.019 -- the C-068 sign
  pattern is above it.
- **Escape filter (I-2): no measurable effect.**
- **Encoding (I-3) matters more than learning.** The baseline's own encoding lifts every metric (accuracy
  +0.045, slateN point +0.05, mask +0.06 over tube), but most of that is already in the encoding: the
  untrained field_orig reaches 0.636 / 0.948 / 0.682. On this program the trained GNN adds at most ~0.02
  slateN (point) / ~0.06 (mask) and ~0.03 moved-node accuracy over its own action-input heuristic.
- **C-068 is encoding-specific (narrowed):** with the tube encoding the GNN's edge over its heuristic
  reverses on large piles; with the baseline encoding the (small) edge GROWS with pile size. The robust
  pile-size statement is only that untrained heuristics are already strong here and the learned
  increment is small at every size.

## Unrelated findings

- The FleX solver's explosion rate rises with pile size (valid 96 % → 79 %), and the collector's validity
  flag misses escaped particles in 3.5-14.6 % of DS-0021 rows (rising with count) — a data-quality
  issue for any FleX carrot corpus from this collector (DS-0019/20 audits found the same class).
- The source report's two infrastructure fixes (PyFleX `voxelize.cpp` unbounded loop; native segfault
  crash ledger) live uncommitted in the source repo (issues.md I-10).

## NFD and LinearForesight on the same program (2026-10-04, RUN-0013 / RUN-0014 / RUN-0015)

Added the project's two image-based families on the SAME train/test program, through the EXP-0061/0062
FlexData pipeline (not a reimplementation): DS-0021/22 now have FlexData instance configs, particle caches and
binary colour-image-mask caches (grid +-7.2, 64 px; DS-0021 split = this record's per-group 90/10 by state, so
val = the GNN's val; render check: frame peak (0,0), 99.7 / 99.9 % of removed mask pixels inside the swept region
vs 47 / 55 % with z negated -- DS-0021/22 DATASET.md). Loader flags drop escaped / out-of-grid / null rows (train
15,193 of 18,000 kept; test 16,417 of 17,761, clean pools 50-98).

- **LF switched (MODEL-0013, RUN-0013)**: EXP-0062 recipe, bins/lambda picked on DS-0021 val only (equal bins,
  lambda 300; val accuracy 0.360, single 0.266). CPU.
- **NFD (MODEL-0014, RUN-0014)**: EXP-0062 `nfd_3ch_flex_mask_v2` recipe on DS-0021, seed 0, plateau stop. Gated on
  the overnight GNN queue (`QUEUE DONE`); runs via the detached chain `code/nfd_after_queue.sh`, which also scores it
  and regenerates `results/nfd_lf_image_metrics.*`. **Its numbers are not in the table below unless stated** (see RUN-0014 Outcome).
- **GNN on the image (RUN-0015)**: MODEL-0012's node displacements (z sign -1, tube encoding, FPS rep 0) carried onto
  the input mask by nearest node (flex_predictor.render's rule); `gnn_truecap` = the same carry with the true node
  motion (renderer cap); `field` = EXP-0064's untrained push field through the same carry. Out-of-domain references:
  MODEL-0008 (NFD) / MODEL-0009 (LF) trained on DS-0020 v2 (same scene and grid, different pile generator).

Scoring (RUN-0015, `code/score_image_metrics.py`): truth = binary image mask (= `eval_report --truth-scoring image`);
slateN = 3 goals x 3 value fns, ties averaged (persistence = 0 exactly); K-equalised to K = 50 (smallest clean pool,
50 subsets/state); accuracy = swept-region, ratio of sums. State-bootstrap 95 % CIs, 50 states per group.

slateN (K = 50) / accuracy, mean [95 % CI] in `results/nfd_lf_image_metrics.md`:

| model | 10-30 | 50-70 | 100-150 | 400-500 | overall |
|---|---|---|---|---|---|
| LF switched MODEL-0013 | 0.830 / 0.338 | 0.866 / 0.386 | 0.899 / 0.417 | 0.883 / 0.424 | 0.869 [0.860, 0.879] / 0.396 |
| LF single MODEL-0013 | 0.599 / 0.242 | 0.690 / 0.261 | 0.750 / 0.273 | 0.688 / 0.284 | 0.682 / 0.267 |
| GNN MODEL-0012 on mask | 0.556 / -0.046 | 0.681 / 0.036 | 0.674 / 0.059 | 0.546 / 0.045 | 0.614 [0.592, 0.632] / 0.030 |
| GNN renderer cap (true node motion) | 0.970 / 0.523 | 0.959 / 0.476 | 0.941 / 0.400 | 0.853 / 0.223 | 0.930 / 0.392 |
| field (untrained) | 0.660 / 0.103 | 0.729 / 0.139 | 0.769 / 0.156 | 0.748 / 0.074 | 0.727 / 0.118 |
| NFD MODEL-0008 (DS-0020 v2, out of domain) | 0.894 / 0.456 | 0.927 / 0.517 | 0.948 / 0.541 | 0.916 / 0.492 | 0.922 [0.914, 0.929] / 0.505 |
| LF switched MODEL-0009 (DS-0020 v2) | 0.800 / 0.309 | 0.859 / 0.370 | 0.899 / 0.410 | 0.877 / 0.411 | 0.859 / 0.381 |
| persistence | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| random | -0.019 / (noise) | 0.018 | -0.001 | -0.015 | -0.004 [-0.022, 0.014] |

Paired, per state (K = 50 slateN): LF switched minus GNN +0.274 / +0.186 / +0.225 / +0.337 (all CIs > 0); GNN minus
field -0.104 / -0.048 / -0.096 / -0.202 (all CIs < 0); LF minus field +0.13..+0.17 (all > 0); MODEL-0013 minus
MODEL-0009 +0.011 [+0.005, +0.017] overall (in-domain fit helps only the 10-30 group, +0.030). 400-500 minus 10-30
(unpaired): LF switched +0.052 [+0.026, +0.079], GNN -0.011 [-0.064, +0.043], field +0.088, GNN cap -0.117.

**Does accuracy correlate with slateN here?**
- (a) Across models within a group: **yes**. 7 non-baseline models (incl. field and the GNN cap): Kendall tau
  +0.90 / +0.81 / +0.62 / +0.81 by group (p 0.003 / 0.011 / 0.069 / 0.011), overall +0.71 (p 0.03); with persistence
  and random (n = 9) tau 0.78-0.89, all p <= 0.002. The one discordance is the GNN render cap (top slateN, mid accuracy).
- (b) Across groups within a model (n = 4, underpowered by construction): **not consistently**. Spearman +1.0 for
  MODEL-0008 and the GNN cap, +0.8 for both switched LFs, +0.4 for LF single and field, 0.0 for the GNN. Accuracy
  rises monotonically with group for every LF while slateN peaks at 100-150.
- (c) Per state within a model: **yes, positively and moderately**. Spearman(state accuracy, state slateN_K) over 200
  states: LF switched +0.60 [+0.49, +0.69], LF single +0.39 [+0.27, +0.51], GNN +0.50 [+0.38, +0.60], field +0.42
  [+0.29, +0.54], MODEL-0008 +0.62, MODEL-0009 +0.73, GNN cap +0.75; random +0.05 [-0.10, +0.19]. Within single
  groups (n = 50) it is weaker and strongest in 400-500 (+0.51..+0.83); several 10-30 / 50-70 / 100-150 cells include 0.
  Caveat: per-state correlation also carries state difficulty common to both metrics (a state where pushes are easy to
  predict tends to be easy to rank), so it is not a within-state model-quality signal.

Reading: on this program the image-based switched LF ranks pushes far better than the particle GNN read through the
image (the GNN is below even its own untrained push field on image-mask truth; with true node motion the same renderer
reaches 0.93, so the gap is the GNN's dynamics, not the renderer, except at 400-500 where the 30-node cap itself
drops to 0.85). Image slateN does not fall with pile size for the image models; it rises 10-30 -> 100-150 and stays.
Image-mask slateN is NOT comparable with this record's particle-raster mask slateN (RUN-0006) or point-goal slateN.
Threats: one seed per model (LF is deterministic); MODEL-0013 excludes escaped rows that the GNN trained on (I-2);
the GNN on the image uses one FPS rep; NFD MODEL-0014 pending at the time of writing.
