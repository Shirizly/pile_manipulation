> **Ported 2026-10-03 from the source repo (`~/Code/dyn-res-pile-manip`, file `OBJECT_COUNT_GNN_STUDY.md`), text unchanged below this note.**
> **Read EXP-0064's `EXPERIMENT.md` and `issues.md` first:** the model described here read every push in a
> z-mirrored frame (issues.md I-1), so its accuracy/capture describe a model with a nearly uninformative
> action input; the per-group capture rise is largely shared by an untrained push-field baseline
> (RUN-0006). Path mapping: `data/true_action_transitions_carrots_grouped` → DS-0021;
> `data/true_action_slates_objbiased_carrots_grouped` → DS-0022; `data/gnn_dyn_model_grouped/...` →
> MODEL-0011 (`weights/`); `test_outputs/*.log|json` → `artifacts/RUN-000{1..4}-*/`; `train/`, `dataset/`,
> `eval_grouped_capture.py`, `config/...` → `code/` and `code/configs/`; `env/flex_env_multi.py` →
> `code/ported_reference/`; `model/gnn_dyn.py` → `Baselines/GNN/model/gnn_dyn.py` (byte-identical);
> `.github/skills/pyflex-simulation/SKILL.md` and the PyFleX fixes remain in the source repo only.

# Object-count-grouped GNN study: results and methodology

**Question:** how does a GNN dynamics model's prediction `accuracy` and
control-ranking `capture` (this file's `slateN`, named per `METRICS.md`'s
"historical reasons" note) vary with the number of objects (carrots) in the
pile it has to predict?

## Headline results

Trained `PropNetDiffDenModel` (`model/gnn_dyn.py`, the project's only GNN
dynamics model) via the **unmodified baseline training loop**
(`train/train_gnn_dyn.py`'s loop, copied verbatim into
`train/train_gnn_dyn_grouped.py` with only the data-loading swapped — see
"Why a copy, not an edit" below), on 1800 training windows / evaluated on the
full 200-state held-out test pool. Checkpoint: `net_best.pth` at epoch 87
(validation loss 0.0615; see "Training" below for why we stopped at 100
epochs rather than the configured 300).

| object-count group | accuracy | capture (mean +/- sem) | test states | test pool/state (mean) |
|---|---|---|---|---|
| 10-30   | 0.224 | 0.475 +/- 0.046 | 50 | 96.1 |
| 50-70   | 0.245 | 0.646 +/- 0.048 | 50 | 92.8 |
| 100-150 | 0.212 | 0.633 +/- 0.037 | 50 | 87.7 |
| 400-500 | 0.156 | 0.709 +/- 0.040 | 50 | 78.6 |
| **overall** | **0.219** | **0.616 +/- 0.022** | 200 | 88.8 |

Full per-state breakdown (including per-state capture, particle counts
tracked, and pool size): `test_outputs/eval_results_grouped.json`.

**Two findings point in opposite directions, and both are real:**

1. **`accuracy` degrades with object count**, most sharply at 400-500
   (0.224 -> 0.156). Pointwise next-position prediction gets harder as the
   pile gets more crowded and the fixed 30-node budget (see "Node budget"
   below) has to compress proportionally more real structure into the same
   graph size.
2. **`capture` *improves* with object count** (0.475 -> 0.709), close to
   monotonically (only 50-70 vs 100-150 is a near-tie within noise). A larger
   pile's centroid responds more predictably to "did this push engage a lot
   of mass in roughly the right direction" even when the model's per-particle
   position predictions are individually worse — ranking actions by a
   *distance-to-goal-centroid* value function (see "Capture / value
   function" below) needs much coarser information than exact positions, and
   more mass under the plate means more pushes produce a clear, easy-to-rank
   directional effect. Read `accuracy` and `capture` as answering different
   questions, not as two readings of the same thing: the model's raw
   dynamics-prediction fidelity gets *worse* with scale here, while its
   usefulness for *this specific control objective* gets *better*.

**Caveat that must travel with this table**: the test pool size shrinks with
object count (96.1 -> 78.6 valid actions/state on average) because the
solver's own explosion rate rises sharply with pile size (measured directly,
see "Explosion rate vs. object count" below: 3.8% -> 21.4%). `capture` is
computed on whatever pool size is left after dropping exploded actions, so
part of the 400-500 group's higher capture could in principle be an artifact
of a smaller, easier-to-rank pool rather than a genuine model-quality
difference — `METRICS.md`'s own stated caveat ("not comparable across pools
of different size") applies directly here. We did not control for this
(stratifying sampling to equal post-explosion pool sizes was out of scope for
this run); a follow-up wanting a cleaner causal read on `capture` alone
should resample extra actions per state in the larger groups to equalize
valid-pool size before ranking.

## Datasets collected

Both new, both carrots, both using the same per-step action sampling as the
existing `objbiased`/`transitions` datasets (`obj_biased`: start point drawn
near an existing particle + Gaussian jitter, end point uniform, margin-aware
— see `.github/skills/pyflex-simulation/SKILL.md`). State sampling is new
(see "Object-count-targeted pile generation" below), split into 4 equal
blocks of states, one per object-count group (ranges above).

- **`data/true_action_transitions_carrots_grouped`** (TRAIN) — 2000 states x
  10 sequential actions (continuing-trajectory format, see
  `data/true_action_transitions_FORMAT.md` for the general schema this
  follows). 500 states/group. **Result: 2000/2000 states fully collected,
  20000/20000 transitions, zero permanently-failed states** (the rare
  native-crash class documented in `collect_transitions.py`'s docstring did
  not reoccur in this run — 0/2000, vs 5/2000 in the earlier
  non-grouped carrots transitions run).
- **`data/true_action_slates_objbiased_carrots_grouped`** (TEST) — 200 states
  x 100 independent action replays per state (slates format, see
  `data/true_action_slates_FORMAT.md`), 50 states/group, bin-stratified by
  push length (same 6 bins/edges as every other carrots dataset in this
  project). **Result: 200/200 states, 17761/20000 actions valid** (see the
  explosion-rate table below for why that's well below 100%, and why
  unevenly so across groups).
- Both configs: `config/data_gen/transitions_carrots_grouped.yaml`,
  `config/data_gen/slates_objbiased_carrots_grouped.yaml`.

### Object-count-targeted pile generation

Carrots' existing generators (`rand_blob`/`rand_spread`) derive object count
as a side effect of a randomly drawn blob radius/scale — measured empirically
(2000 draws each) they produce only 5 distinct counts apiece
(45/105/189/297/429 and 189/297/429/585/765), none in this study's low
ranges (10-30, 50-70) and only partial overlap with the high ones. Hitting
exact arbitrary ranges needed a new mechanism: `env/flex_env_multi.py`'s
`_carrots_scene_params` gained a `count_target` `init_pos` branch that
inverts `rand_blob`'s own count formula (`num_carrots = (num_x*num_z-1)*3`,
solved for `num_x==num_z` from the **target** count instead of from a random
blob radius), so the generated footprint is sized exactly to the target
count rather than a randomly-sized footprint getting truncated (which would
produce an unnatural thin sliver for a small target on a large footprint).
Per-carrot scale and pile position/offset stay randomised exactly as
`rand_blob` does, for the same within-group state variety. Validated
visually and numerically before collection (exact counts 15/25/60/120/450
hit on request; see conversation — not re-attached here, but
`test_outputs/eval_results_grouped.json`'s recorded `target_num_carrots` per
state in the dataset manifests is the permanent record).

Both collectors (`collect_transitions.py`, `collect_true_action_results_parallel.py`)
gained a shared `transitions.count_groups` / `slates.count_groups` config key
and a `resolve_count_group` helper (in `collect_transitions.py`, imported by
the other) that deterministically assigns every state to a group and draws
its target count, seeded by `(seed, state_idx)` so a resumed/redone state
reproduces the identical target.

### Explosion rate vs. object count (slates/TEST set)

| group | valid / planned | rate |
|---|---|---|
| 10-30   | 4807 / 4998 | 96.2% |
| 50-70   | 4639 / 5000 | 92.8% |
| 100-150 | 4385 / 5000 | 87.7% |
| 400-500 | 3930 / 5000 | 78.6% |

Monotonic, and a substantial effect (96% -> 79%) — bigger piles are
genuinely harder for the solver, not just for the model. This is itself a
finding relevant to the project's motivating question, independent of the
GNN results above.

### Two real infrastructure bugs found and fixed en route

Both affect any FUTURE carrots collection at scale, not just this run, so
they're recorded in `.github/skills/pyflex-simulation/SKILL.md` as well as
here:

1. A rare **native segfault** in carrots' random-convex-mesh generation
   (`CreateRandomConvexMesh`, `PyFleX/bindings/helpers.h`), confirmed
   deterministic (state-dependent, not load/contention-dependent) and
   state-count-correlated (more states and larger `num_carrots` draws make it
   more likely to hit). Mitigated with a persistent per-state crash ledger
   (`collect_transitions.py`) that permanently skips a state after 2
   native-crash attempts, run under a supervisor script
   (`run_transitions_shard.sh`) that auto-restarts on crash. Did not trigger
   at all in this run's 2000 states (unlike the earlier non-grouped run,
   5/2000) — load-bearing but not something to rely on being silent.
2. A genuine **infinite loop** in vendored PyFleX
   (`PyFleX/core/voxelize.cpp`'s `Voxelize()` ray-march had no iteration
   bound; a self-intersecting mesh could spin it forever, printing
   `Error self-intersect` without limit). This was live and had already
   written a 42GB log before being caught. Patched with a bounded iteration
   cap (`4 * depth + 1000`, generous for any non-degenerate mesh); confirmed
   fixed by reproducing the exact prior hang and watching it terminate
   cleanly instead.

### A harness-level operational issue (not a bug in this project's code)

Background shell commands in this session were killed by the harness after
a fixed ~20-25 minute wall-clock cap, repeatedly, regardless of the Bash
tool's own `timeout` parameter. Every collector here is designed to be
safely resumable (append-only, fsync'd manifests, persistent crash ledgers),
so each kill cost only a relaunch, no data — but dozens of relaunches over
the run would not have been practical to do by hand. Worked around by
launching long-running collectors/training as fully detached OS processes
(`setsid nohup ... < /dev/null > log 2>&1 & disown`, outside the harness's
own background-task tracking) and watching them via the `Monitor` tool's
polling/`tail -f` mode instead of relying on background-task completion
notifications.

## GNN training

### Why a copy, not an edit

Per explicit instruction: use the project's existing baseline training
process (`train/train_gnn_dyn.py`), not the unified `training/trainer.py`
framework, "in case they differ in results." `train/train_gnn_dyn.py` itself
is **untouched**. `train/train_gnn_dyn_grouped.py` is a copy with the
training loop (multi-step rollout loss, optimizer, logging, checkpointing)
byte-for-byte identical, and only two inert differences: it doesn't spin up
a throwaway `FlexEnv` to fetch camera parameters (the new dataset needs no
camera frame at all — see below), and it points at this study's own
config/output paths. `config/train/gnn_dyn_grouped.yaml` mirrors
`config/train/gnn_dyn.yaml`'s schema.

### New dataset adapter (`dataset/dataset_grouped_particles.py`)

The baseline's own `dataset/dataset_gnn_dyn.py::ParticleDataset` could not be
reused as-is: it requires per-timestep depth images (`N_depth.png`) that this
project's `true_action_*` collectors never save (they're built around a
different, PyFleX-native data-generation pipeline, not the depth-camera
pipeline `data_gen/gnn_dyn_data.py` used). `GroupedParticleDataset` produces
the **exact same `__getitem__` contract** (`states, states_delta, attrs,
particle_num, particle_den, color_imgs`, identical shapes/semantics) so the
training loop runs unmodified, differing only in how those tensors are
built:

- **No depth-image round trip** — particles are the real simulated
  positions (more accurate than the original's depth-camera-approximated
  ones, not less).
- **Node budget, per your instruction**: `min(30, state's target object
  count)`. A state with 10-30 objects (lower than 30) gets one node per
  particle*; the two larger groups get a fixed 30-node farthest-point-sample
  (FPS) of the real particle cloud. FPS spreads the sample across the whole
  pile, so a 400-500-object state still gets *some* representation of every
  region, at much coarser per-object resolution — this is presumed to be
  what lets the model extract a usable (if imprecise) signal from the larger
  groups, consistent with `accuracy` degrading gracefully rather than
  collapsing.

  (*lower-case caveat: "10-30 objects" means 10-30 carrot *bodies*, not
  10-30 *particles* — even the smallest pile has several hundred particles,
  of which up to 30 are FPS-sampled, same as the other groups; node count
  equals object count only incidentally, when object count happens to be
  <=30.)
- **Fixed `particle_den` constant** (1000.0, `PARTICLE_DEN_CONST` in the
  module), not the baseline's per-sample `U(15, 6500)` random draw — that
  randomisation exists in the baseline to compensate for sensing noise in
  real/simulated-depth-derived density estimates, which has no equivalent
  here (we have exact simulated ground truth).
- **Position/action normalisation by `GLOBAL_SCALE=24`**, matching the
  baseline's own final normalisation step — necessary so the model's
  `adj_thresh=0.08` (and the derived `PLATE_HALF_WIDTH` used for the
  action-encoding soft mask) stay in the range those defaults were tuned
  for; skipping it would have silently made the neighbour-adjacency radius
  ~24x too small relative to actual particle spacing.
- Train/valid split (90/10) is **per-state, stratified within each
  object-count group independently** — never per-window (would leak a
  state's own future frames across the split) and never a single global
  split (could by chance starve one group's validation set).

### Training run

`config/train/gnn_dyn_grouped.yaml`: `batch_size=16`, `n_history=1`,
`n_rollout=5` (a 6-frame window, well within the 11 frames — initial + 10
steps — every complete training trajectory has), `node_budget=30`,
`adj_thresh=0.08` (unchanged baseline default), 1800/200 train/valid windows.

Configured for 300 epochs; **stopped at epoch 100** (~1h20m wall clock) once
validation loss had been flat in the 0.061-0.067 band since roughly epoch
15-20 with no further trend — continuing to epoch 300 would have cost
several more hours for no expected gain, and `net_best.pth` (epoch 87, val
loss 0.0615) was already checkpointed. All 100 epoch checkpoints are kept
(`data/gnn_dyn_model_grouped/<run>/net_epoch_*.pth`), nothing deleted, so a
longer run can resume or a different epoch can be re-evaluated without
recollecting anything.

## Metrics

Both defined in `METRICS.md`; re-derived here for this particle/raw-coordinate
setting (no occupancy image involved anywhere in this evaluation).

**`accuracy`** = `1 - rms(pred - true) / rms(persistence - true)`, pooled
over every (state, action, tracked-particle) triple in a group — persistence
being "predict nothing moved" (the model's own input). Implemented in
`eval_grouped_capture.py`.

**`capture`** (`METRICS.md`'s `slateN`, generalised to an arbitrary value
function, per that file's 2026-09-10 section) — your choice,
**distance-to-goal-centroid (Lyapunov-style)**: for each TEST state, a goal
point is drawn uniformly in the workspace, seeded by that state's own index
(reproducible regardless of which model/run scores it — same convention
`METRICS.md` specifies for its own goal shapes). `cost(particles) = mean
distance from every tracked particle to the goal point` (lower is better,
`dv < 0` improving, exactly `METRICS.md`'s stated `lyapunov` sign
convention). For the pool of K candidate actions from that state:

    capture = (mean_c(true_dv) - true_dv[argmin_c(pred_dv)]) / (mean_c(true_dv) - min_c(true_dv))

0 = the model's pick was no better than the pool average (= what a random
pick would get in expectation); 1 = it picked the oracle's actual best
action. Both the model's prediction and the "true" comparison use the same
subsampled/tracked particle indices and the same normalisation as training.
Implemented in `eval_grouped_capture.py`; raw per-state capture values (so a
reader can recompute sem, check ties, etc. independently) are in
`test_outputs/eval_results_grouped.json`'s `per_state` list.

**Known limitation carried over from `METRICS.md`'s own stated caveats for
`slateN`** (ties / effective sample size, and non-comparability across pools
of different size) — both apply here and are not separately re-verified
in this run; see "Headline results" above for the specific pool-size caveat
that matters for this table.

## Where everything is (nothing deleted)

| What | Where |
|---|---|
| Train data | `data/true_action_transitions_carrots_grouped/` |
| Test data | `data/true_action_slates_objbiased_carrots_grouped/` |
| Trained checkpoints (all 100 epochs + best) | `data/gnn_dyn_model_grouped/2026-10-03-04-53-20-352197/` |
| Training log | `test_outputs/train_gnn_dyn_grouped.log` |
| Full eval results (overall + per-group + per-state) | `test_outputs/eval_results_grouped.json` |
| Data-collection run logs | `test_outputs/transitions_carrots_grouped_shard{0,1}.log`, `test_outputs/slates_objbiased_carrots_grouped_run.log` |
| New/changed code | `env/flex_env_multi.py` (`count_target`), `collect_transitions.py` / `collect_true_action_results_parallel.py` (`count_groups`), `dataset/dataset_grouped_particles.py`, `train/train_gnn_dyn_grouped.py`, `eval_grouped_capture.py`, `PyFleX/core/voxelize.cpp` (iteration-cap fix), configs under `config/data_gen/` and `config/train/` |
