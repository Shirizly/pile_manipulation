# Refactor index — where loose code should move

**Status: pass 1 of N.** Covers the repository root, `scripts/`, `Genesis/` top level,
and the dead trees. Not yet audited: `Baselines/*`, `simple_mpc/`, `dino_wm/`, `le-wm/`,
`model/futureintegration/`, `experiments/*/code/` — see [Not yet covered](#not-yet-covered).

This is a **proposal index**, not a changelog. Nothing here has been moved. Each row says
*what*, *where to*, *why*, and *what breaks*. Work it top-down by phase; the phases are
ordered by risk, not by value.

---

## Constraints that shape every move

Three properties of this repo make a naive "move files into packages" refactor destructive.
Read these before acting on any row below.

1. **Everything imports by bare module name from the repo root.** `from
   fit_linear_foresight import canonicalise`, `from dmdc_baseline import
   load_transition_arrays`, `from control_utility_test import lyapunov`. There is no package
   (`setup.py`/`pyproject.toml` do not exist; every caller relies on cwd being the repo root
   or on `PYTHONPATH=.`). Moving such a module renames a public symbol path used in **57
   places** (`fit_linear_foresight`) with no import-rewrite safety net beyond grep.

2. **`experiments/EXP-*/code/` is a frozen evidence layer.** Those files are the record of
   how a claim in `experiments/REGISTER.md` was produced. Rewriting their imports edits
   evidence. **Recommended rule: never touch `experiments/*/code/`.** Every move of a
   root-level module that experiment code imports must therefore leave a
   re-export shim at the old path, or the record stops being re-runnable.

3. **`experiments/COMMANDS.jsonl` records literal command lines.** Moving a script named in
   it invalidates the recorded command. Only `scripts/run_probe.py` appears there today
   (8 entries), so this constraint is currently cheap — but it prices every future move up.

**Consequence.** The high-traffic modules (§1) should move *with a shim*, and the shim should
be deleted only after a separate, explicitly-scoped import-rewrite pass over live code
(`scripts/`, `Baselines/`, `tests/`) that skips `experiments/`.

---

## §1 — Root-level modules that are actually libraries

These are imported far more than they are run. They sit at the root only because that is
where they were first written. This is the single largest source of the "scripts strewn
around" feeling, and the highest-value fix.

| file | lines | importers | move to | why |
|---|---|---|---|---|
| [utils.py](../utils.py) | 862 | **69** | split — see below | not one module; four unrelated concerns fused |
| [fit_linear_foresight.py](../fit_linear_foresight.py) | 633 | **57** | `Baselines/LinearForesight/fit_pixel_operator.py` | `Baselines/LinearForesight/` already owns `model.py` + `fit_switched.py`; this is the third piece of the same baseline, marooned at root |
| [control_utility_test.py](../control_utility_test.py) | 338 | **34** | `Baselines/common/control_utility.py` | it is the `lyapunov` / `lyapunov_weights` / `rank_metrics` library that `Baselines/common/goals.py` is the natural neighbour of. **The `_test` suffix is a lie** — pytest's default `python_files` glob matches `*_test.py`, so `pytest` from the root imports this module every run and collects 0 tests from it |
| [dmdc_baseline.py](../dmdc_baseline.py) | 678 | **33** | `Baselines/Descriptors/dmdc.py` (new dir) | owns `occupancy_descriptors` / `descriptor_slices` / `fit_per_action_operators`, which `MODEL-0002`'s whole basis is defined against. A descriptor baseline deserves a sibling dir to `LinearForesight/`, `NFD/`, `GNN/`, `SchenckCNN/` |
| [occupancy_foresight.py](../occupancy_foresight.py) | 281 | **16** | `Baselines/LinearForesight/fit_occupancy_operator.py` | same fit as `fit_linear_foresight`, different loader; belongs beside it (its own docstring says the shared method is load-bearing) |
| [loro_foresight.py](../loro_foresight.py) | 190 | 4 | `Baselines/LinearForesight/eval_loro.py` | leave-one-run-out CV *of* the LinearForesight operator |
| [variance_decomposition.py](../variance_decomposition.py) | 193 | 4 | `Baselines/common/variance_decomposition.py` | imported as a library by three other analyses (`density_stratified`, `nonlinearity_probe`, `probes/exp0019_contact_linearity`) for its `load` / `features` / `r2` / `band_frame`, and it backs live claims C-007/C-009 |

### `utils.py` — proposed split

862 lines, 69 importers, and about five lineages fused into one file. It is imported by
`dino_wm/` (7), `Baselines/GNN/` (7), and a dozen experiment records, most of which want one
function each. Suggested target: a `common/` package at the root (kept at root so the
shim story is a one-line re-export).

| functions | move to |
|---|---|
| `fps`, `fps_rad`, `fps_np`, `recenter`, `downsample_pcd`, `np2o3d`, `findClosestPoint`, `opengl2cam`, `depth2fgpcd`, `pcd2pix`, `calc_dis` | `common/pointcloud.py` |
| `resize`, `crop`, `adjust_*`, `lighten_img`, `rmbg`, `drawRotatedRect`, `drawPushing`, `write_video_frame` | `common/imaging.py` |
| `gen_goal_shape`, `gen_ch_goal`, `gen_subgoal`, `scale_subgoal_to_material_pixels`, `gt_rewards`, `gt_rewards_norm_by_sum` | **merge into `Baselines/common/goals.py`** — that file already owns goal masks and scoring; these are a duplicate lineage. `gen_subgoal` and `gen_goal_shape` are each **defined twice inside `utils.py`** (lines 337/777 and 367/747) — the second definition silently wins. Resolve before moving. |
| `set_seed`, `load_yaml`, `save_yaml`, `Tee`, `AverageMeter`, `get_lr`, `git_provenance`, `get_current_YYYY_MM_DD_hh_mm_ss_ms`, `count_*_parameters`, `to_var`, `to_np`, `rand_float`, `rand_int`, `norm`, `combine_stat`, `init_stat` | `common/runtime.py` |
| `rect_from_coord`, `check_side`, `check_within_rect`, `preprocess_action_segment`, `preprocess_action_repeat*` | `transforms/action_encoding.py` — these are action↔grid encodings, which `ARCHITECTURE.md`'s stated philosophy puts in `transforms/`. Note `Baselines/SchenckCNN/action_encoding.py` already exists; check for overlap first. |

`utils.py` then becomes a shim that re-exports all of the above, and is deleted after the
live-code rewrite pass.

---

## §2 — Root-level entry points (run, not imported)

These are legitimately executables, but an executable is not a root-directory citizen.

| file | lines | move to | notes |
|---|---|---|---|
| [run_experiments.py](../run_experiments.py) | 1281 | `simple_mpc/cli/run_experiments.py` | reads `simple_mpc/config/experiments/*.yaml`; the config already lives in `simple_mpc/` |
| [run_experiment_batch.py](../run_experiment_batch.py) | 156 | `simple_mpc/cli/run_experiment_batch.py` | thin subprocess wrapper over the above; move together |
| [run_oracle_mpc.py](../run_oracle_mpc.py) | 308 | `simple_mpc/cli/run_oracle_mpc.py` | its logic is already `simple_mpc/oracle_mpc.py` + `simple_mpc/genesis_oracle.py` |
| [debug_mpc_gui.py](../debug_mpc_gui.py) | 765 | `simple_mpc/gui/debug_mpc_gui.py` | |
| [human_mpc_gui.py](../human_mpc_gui.py) | 510 | `simple_mpc/gui/human_mpc_gui.py` | drives `simple_mpc/human_mpc.py` + `human_grid_search.py`; imports tile helpers from `debug_mpc_gui`, so the two must move together |
| [visualize.py](../visualize.py) | 458 | `scripts/viz/visualize_dataset.py` | 0 importers; its design doc is already archived (`docs/archive/2026-07/VISUALIZE_PLAN.md`) |
| [visualize_training.py](../visualize_training.py) | 216 | `scripts/viz/inspect_unet_predictions.py` | 0 importers. Current name says nothing about what it inspects (NFDUNetFiLM on the Genesis test split) |
| [compare_model_emd.py](../compare_model_emd.py) | 413 | `scripts/probes/compare_model_emd.py` | 0 importers; a one-shot NFD-vs-X EMD comparison |
| [run_pile_analysis.py](../run_pile_analysis.py) | 58 | `scripts/probes/run_pile_analysis.py` | 0 importers; `subprocess`-shells three other root scripts by literal filename — **its command strings must be updated in the same commit as any §1 or §3 move** |

---

## §3 — Root-level one-shot analyses

Zero or near-zero importers, each answering one question once. `scripts/probes/` is the
established home for exactly this (it already holds 40 of them).

| file | lines | move to | notes |
|---|---|---|---|
| [deltav_predictability.py](../deltav_predictability.py) | 146 | `scripts/probes/deltav_predictability.py` | |
| [density_stratified.py](../density_stratified.py) | 91 | `scripts/probes/density_stratified.py` | imports `variance_decomposition` — update with §1 |
| [nonlinearity_probe.py](../nonlinearity_probe.py) | 82 | `scripts/probes/nonlinearity_probe.py` | imports `variance_decomposition` — update with §1 |
| [interpret_foresight.py](../interpret_foresight.py) | 91 | `scripts/probes/interpret_foresight.py` | parses args at import time (no `main()`), so it cannot be imported — fine for a probe, blocks reuse |
| [interpret_by_contact.py](../interpret_by_contact.py) | 78 | `scripts/probes/interpret_by_contact.py` | no argparse at all; the dataset path is hardcoded at line 9 |
| [model_zoo.py](../model_zoo.py) | 263 | `scripts/probes/model_zoo_sweep.py` | 0 importers. Name implies a model registry; it is a one-shot comparison sweep, and `registry/model_registry.py` is the actual registry. **Rename is the point of this move** |
| [koopman_skeleton.py](../koopman_skeleton.py) | 170 | `archive/` | 0 importers, no docstring, comments in the second person ("First: define..."). Superseded by `experiments/EXP-0016-*/code/model.py`, which is the same idea actually implemented and measured |

---

## §4 — `scripts/probes/` — the retired experiment numbering

**This is the most dangerous item in the repo and should be phase 1.**

Fourteen probes are named `exp00NN_*` / `expB_*` after an experiment numbering that was
retired at commit `f33ae178` ("Archived old experiments due to confused results"). The
current `experiments/EXP-0001…EXP-0020` records are a *different, unrelated* series, and
the numbers now **collide**:

| file | its actual subject | what `experiments/EXP-00NN` is today |
|---|---|---|
| `exp0016_loro.py` | L040 crop/res LORO sweep | LeJEPA random-encoder floor |
| `exp0018_config_sweep.py` | crop/res config sweep for C-001 | value readout on diverse states |
| `exp0019_contact_linearity.py` | C-007/C-009 adversarial re-checks | LeJEPA encoder push-length switched |
| `exp0009_rerun.py` | pixel-operator re-run post grid fix | time-budget benchmark |
| `exp0021_*`, `exp0022_*`, `exp0024_*`, `exp0026_*`, `exp0029_*`, `expB_*` | UNet-vs-linear, selection pressure, L10mm mechanism | *no such record exists* |

An agent told "check what EXP-0019 did" will find the wrong file. `experiments/METRICS.md` defines `slateK_exact` and points at `scripts/probes/pool_common.py`
and `pool_survey.py` for it — but `exp0026_kcurve_exact.py`'s own docstring records that an
earlier METRICS.md cited *it* as the definition source for `slateK_exact` / `worstK` /
`rank_profile`. Check that citation is fully migrated before deleting anything here.

**Proposed:** `scripts/probes/legacy/`, with each file renamed to its subject and an
`scripts/probes/legacy/README.md` mapping old name → new name → the retired claim it served.
Suggested renames: `exp0016_loro.py` → `loro_crop_res_sweep.py`, `exp0018_config_sweep.py` →
`operator_config_sweep.py`, `exp0019_contact_linearity.py` → `contact_linearity_recheck.py`,
`exp0021_eval.py` → `unet_vs_linear_swept.py`, `exp0022_blur_fair.py` →
`unet_vs_linear_blur_fair.py`, `exp0024_control_eval.py` → `control_eval_cube_spectrum.py`,
`exp0026_selection_pressure.py` / `exp0026_kcurve*.py` → `selection_pressure_*.py`,
`exp0029_l10mm_mechanism.py` → `l10mm_mechanism.py`, `expB_*` → `multistep_*`.
`METRICS.md` must be updated in the same commit.

**Do not move** the probes `docs/CODEMAP.md` currently points at as live tooling:
`pool_common.py`, `pool_inspect.py`, `pool_survey.py`, `binned_pool_cache.py`.

### `scripts/` top level — minor

| file | move to | why |
|---|---|---|
| `probe_dense_pile.py`, `probe_pile_depth.py`, `probe_pyramid_diversity.py`, `probe_pyramid_setups.py` | `scripts/probes/` | four probes sitting one level above the probes directory, for no reason |
| `cube_spectrum_analysis.sh`, `cube_spectrum_summary.py`, `overnight_foresight.sh` | `scripts/probes/legacy/` | drive the retired cube-spectrum / L040 line |
| `fix_nvrtc.sh` | `scripts/env/` | environment repair, not a probe or a tool |
| `check_register.py`, `summarise_register.py`, `run_probe.py`, `describe_dataset.py` | **stay** | the evidence-layer tooling; `check_register.py` is referenced by two skills and must exit 0 |

---

## §5 — `Genesis/` top level

24 modules in one flat directory mixing the simulator wrapper, collection drivers,
dataset readers, and scratch files.

| file | lines | move to | why |
|---|---|---|---|
| `tmp.py` | 47 | **delete** | references `self._scene` at module top level — it is a copy-pasted fragment that cannot run |
| `test.py` | 17 | **delete** | not a test; 17 lines of scratch |
| `parallelogram differentiable.py` | 99 | `archive/` | **a space in the filename** makes it unimportable by any Python import statement |
| `pile_training.py` | 12 | **delete** | 12 lines, 0 importers, imports the pre-`_clean` sandbox |
| `pile_training_old.py` | 211 | `archive/` | 0 importers; imports `utilities.materials`, a package that no longer exists |
| `parameter_optimization.py` | 66 | `archive/` | 0 importers; depends on `bayes_opt` (in `requirements.txt`, but nothing else in the repo uses it) |
| `sandbox_manipulation.py` | — | `archive/` | superseded by `sandbox_manipulation_clean.py` (8 live importers vs this one's stragglers). Keeping both invites an agent to extend the dead one |
| `sandbox_manipulation_single_env.py` | — | `archive/` | 0 importers |
| `data_collection.py`, `data_collection_multi_env.py` | 407 / 93 | `archive/` | superseded by `data_collection_clean.py` (8 importers); `data_collection_multi_env.py` has 0 |
| `convert_pkl_to_pt.py` | 170 | `scripts/data/` | a one-off format migration, not a simulator module |
| `benchmark_n_envs.py` | 130 | `scripts/probes/` | a throughput measurement whose only output is a default for `simple_mpc` |
| `binned_slate_collection.py`, `cube_spectrum_collection.py`, `same_state_slate_collection.py`, `run_collection.py`, `data_collection_clean.py` | — | `Genesis/collection/` | the five live collection drivers, as a group. The `data-collection` skill documents them as one family; the directory should say so |
| `binned_slate_dataset.py` | — | `Genesis/datasets/` or top-level `datasets/readers/` | it is **Genesis-free** (a pure reader). Sitting under `Genesis/` implies a simulator dependency it does not have, and `datasets/` now exists at root |
| `state_library.py`, `action_sampling.py`, `placement_sampling.py`, `spawn_geometry.py`, `transition_buffer.py`, `sandbox_manipulation_clean.py` | — | **stay** | the actual simulator layer |

---

## §6 — Dead trees (delete, no shim, no archive)

| path | what it is |
|---|---|
| `MPC/` | **53 files, all `.pyc`, zero `.py`** — a `__pycache__` skeleton of a directory layout that no longer exists (`MPC/__pycache__/run_oracle_mpc.cpython-310.pyc`, etc.). Untracked. Pure noise at the top of every `ls` |
| `GranularDynamics2/` | same: one `myClasses/__pycache__/UNetModels.cpython-310.pyc` and nothing else. The real copy is `oldscripts/GranularDynamics2/` |
| `__pycache__/`, `.pytest_cache/` | **78 `.pyc` files are tracked in git.** `git rm -r --cached` + a `.gitignore` entry |
| `oldscripts/` | 19 tracked training scripts + a 40 MB-class `.pth` checkpoint + `.png`s. Belongs in `archive/`, and the checkpoint belongs in `weights/` or nowhere |

---

## Suggested phasing

1. **§6 + §4** — deletions and the experiment-number collision. Zero import risk, removes the
   two things most likely to mislead an agent *today*.
2. **§3 + §5** — move the zero-importer one-shots and the Genesis scratch. Low risk;
   grep-verifiable.
3. **§2** — entry points. Medium risk: `run_pile_analysis.py` shells others by literal path,
   and the two GUIs are coupled.
4. **§1** — the libraries, each **with a shim at the old path**, then a separate rewrite pass
   over `scripts/`, `Baselines/`, `tests/` only. `utils.py`'s split is its own project and
   should be last; resolve its two duplicate definitions first.

Every phase must update `docs/CODEMAP.md` in the same commit — it is the file agents are
told to read before grepping, so a stale path there costs more than the move saves.

---

## Not yet covered

- `Baselines/` internals — `common/` has 11 modules and is becoming its own grab-bag.
- `simple_mpc/` — 13 modules, not yet checked for dead entries.
- `dino_wm/`, `le-wm/` — vendored third-party trees; need a "vendored, do not refactor"
  marker rather than a move.
- `model/futureintegration/` — its README already says some entries are broken/skipped.
- `experiments/*/code/` — **out of scope by rule** (constraint 2), but worth a separate
  index of which helpers there have been hand-copied between records. `CODEMAP.md` already
  flags `com_world_pixel` / `bilinear_sample` as copied into ≥3 experiments with
  device-placement bugs re-introduced each time; that is a promotion candidate for
  `Baselines/common/`.
- `configs/`, `env/`, `training/`, `registry/`, `transforms/`, `physics/` — appear coherent
  on inspection, not audited in depth.
