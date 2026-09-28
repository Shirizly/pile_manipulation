---
name: project-overview
description: "Project map for pile_manipulation: purpose, major subsystems, and which doc file owns what — including where reusable datasets, trained model instances, and experiment storage live. Use before working on this repo for the first time in a session, before any change that touches a major information flow, a module's responsibility, a default value, or adds a feature, or whenever it's unclear which module/doc owns something."
argument-hint: "Optionally name the subsystem you're touching, e.g. 'oracle mpc', 'losses', 'Genesis wrapper', 'adapters'"
user-invocable: true
---

# Pile Manipulation — Project Overview

## Purpose

Research codebase for granular pile manipulation: pushing granular material
(cubes/spheres of e.g. chickpeas) toward a target shape/region with a
plate-style pusher, in the Genesis simulator. It supports **learned dynamics
models driven by gradient-descent MPC**, a **Genesis-as-model sampling MPC
(CEM/MPPI) ceiling baseline** that removes dynamics-model error entirely,
and a **human-piloted variant of that same ceiling baseline** (grid-search-
refined manual actions via a GUI) that also removes optimizer/sampling
limitations from the comparison, plus the shared training/loss/transform
infrastructure all three depend on.

## Major parts

| Directory / file | Role |
|---|---|
| `Genesis/` | Low-level simulator wrapper (`SandboxManipulation`) — build/reset/execute a push/read state; batched multi-env data collection; the `n_envs` throughput benchmark |
| `env/genesis_env.py` | `GenesisEnv` — single-env bridge from `SandboxManipulation` to the MPC-facing observation/action interface |
| `model/` | **Model architecture code**: learned (`NFDUNetFilm.py`, `gnn_dyn.py`, `futureintegration/`) and checkpoint-free heuristics (`eulerian_wrapper.py`'s push-model registry). Not to be confused with `weights/` — this directory holds code, not trained instances |
| `training/` | Model training loop (`trainer.py`), the loss registry (`losses.py`), typed batch/output contracts (`types.py`) |
| `transforms/` | Stateless, dependency-light representation conversions (particle↔occupancy, action↔camera coords) shared by datasets, models, and both MPC variants |
| `registry/` | `register_model`/`build_model`, `register_dataset`/`build_dataset` factories |
| `physics/` | `PhysicsBounds` — the one normalization source of truth |
| `simple_mpc/` | Three MPC variants: `mpc.py`/`adapters.py` (gradient-descent, learned/heuristic models via the adapter pattern); `oracle_mpc.py`/`genesis_oracle.py`/`sampling_optimizers.py` (Genesis-as-model CEM/MPPI ceiling baseline); `human_mpc.py`/`human_grid_search.py` (human-piloted variant of the same ceiling baseline, grid-search-refined) |
| `run_experiments.py` / `run_oracle_mpc.py` / `human_mpc_gui.py` | Batch/entry-point drivers for the three MPC variants |
| `tests/` | pytest suite (Genesis-free tests run without a GPU; a few require Genesis) |
| `Baselines/<Name>/` | Baseline model implementations (GNN, NFD, SchenckCNN, ...), each mirroring the project's code/weights split internally: baseline code stays in `Baselines/<Name>/`, trained instances live in `Baselines/<Name>/weights/MODEL-####-slug/` |
| `experiments/` | Per-experiment scientific records (`EXP-####-slug/`: design, runs, artifacts, results, reports), the project-wide claim ledger (`REGISTER.md`, `INVARIANTS.md`, `METRICS.md`, `COMMANDS.jsonl`), and a `temp/` scratch area (below the ledger's lowest tier) for quick, uncited probes. Owned by `experiment-log`; validated by `register-validator` (`scripts/check_register.py`) |
| `datasets/` | Reusable dataset instances (`DS-####-slug/`), each with its own resolved config and authoritative metadata |
| `weights/` | Reusable, resolved **fitted-object instances** (`MODEL-####-slug/`) — network checkpoints, a fitted linear operator, a heuristic model's parameters — each with its own resolved config, payload, and a short test-history pointer. Distinct from `model/`, which is code, not instances |
| `configs/` | Reusable config **templates/schemas** only — copied into a dataset/model/experiment instance and resolved there. A template edit must never encode one instance's value; a field name must mean the same thing in every template that declares it; and a template change that adds fields is a new, versioned template (`template: <path>@<sha>` recorded on the instance), never a silent edit of the one existing instances already point at |
| `scripts/` | Thin project entry points and maintenance/validation tools; substantial reusable logic belongs in the owning module, not in scripts |

## Doc map — read (and update) the one that owns what you're touching

| Doc | Owns |
|---|---|
| `docs/CODEMAP.md` | **Index of where things already are** — which function in which file does what (representations, operators, goals, metrics, datasets, timing), plus known traps. Check it before grepping; add to it when you find something missing |
| `docs/ARCHITECTURE.md` | Repository module map, entry points, the **Design Philosophy** (modularity / division-of-responsibility patterns to preserve), extension recipes (add a model/dataset/loss), training config schema |
| `docs/INTERFACES.md` | Data contracts: batch dict keys per representation, `ModelOutput`, the MPC adapter surface (§3.4) and its `per_sample` loss-cost variant (§3.5), coordinate conventions |
| `docs/UTILITIES.md` | Utility ownership boundaries: what belongs in `transforms/functional.py` vs `utils.py` vs a scoped module, and the on_phase-hook / write_video_frame pattern as the reference example |

Minor subsystem design docs (each owns the design of one major subsystem, not the whole repo):
| `docs/oracle_mpc_design.md` | Full design reference for the oracle MPC subsystem: snapshot/restore state management, sampling optimizers, occupancy-representation caveats, config schema, known limitations |
| `docs/linear_visual_foresight_baseline.md` | Suh & Tedrake 2020 switched-linear visual foresight as a comparison baseline: paper summary, what the repo already supports, the integration plan, and the perpendicular-push / fixed-length action restriction (§7, implemented) |
| `docs/piled_collection.md` | Piled (multi-layer, centred) particle spawns and pile-aware action sampling: why they exist, what they guarantee, and every flag/config that activates them |
| `docs/human_demo_design.md` | Full design reference for the human-demonstration subsystem: the 5D action convention, local grid-search refinement, GUI interaction model, output-schema/recording parity with `run_oracle_mpc.py` |
| `experiments/` | The evidence layer and experiment storage — structure and ownership are covered in the Major Parts row above, not restated here; this row exists so the documentation-policy rule below has something to point at. |

**2026-09-10 — claim register reset, and a stale-citation note.** The prior
`docs/experiments/` (every record, `REGISTER.md`, `INVARIANTS.md`,
`METRICS.md`), `reports/`, and the narrative docs that cited them were
archived wholesale to `archive/2026-09-10_pre-reset/` to start the claim
ledger clean; trained checkpoints and datasets were left in place. Citations
into `reports/...` from `docs/ARCHITECTURE.md`, `docs/UTILITIES.md`,
`docs/piled_collection.md`, and `docs/linear_visual_foresight_baseline.md`
now resolve under `archive/2026-09-10_pre-reset/` instead of the live tree —
those docs were not themselves rewritten by the reset, so a citation that
looks dead is not necessarily wrong, check the archive before assuming so.

If you're not sure where something belongs, it's almost certainly one of
these `docs/` files or the Major Parts table above, not a new file — check
the Design Philosophy in `ARCHITECTURE.md` first; it explains *why* the
boundaries are drawn where they are, which usually settles where a change's
documentation belongs too.

## Visualization — where the plotting code already is

There is no single `viz/` module and there should not be one; plotting lives
with the thing being plotted. Before writing a new figure, check this table —
the pool-diagnostic scripts in particular already produce the standard
per-slate figure and should be extended, not reimplemented.

| what you want to draw | where it is |
|---|---|
| **one action pool / same-state slate**: dv histogram with each model's pick, the action pool on the step-0 occupancy (dots + sampled heading arrows), the chosen actions with swept rectangles, `\|R_K\|` vs K, model-rank-vs-true-rank scatter, true-percentile bars | `scripts/probes/pool_inspect.py` (the figure) + `scripts/probes/pool_common.py` (cache loading, `rk_curve`, `action_geometry`, `swept_rectangle_corners`, `MODEL_COLORS`) |
| which pools are worth plotting (typical / worst / near-tie, per model) | `scripts/probes/pool_survey.py` — run it first and take the slate ids it names |
| the dV cache those two read | `scripts/probes/exp0026_selection_pressure.py`, `expB_multistep_eval.py`, and — for `Genesis/data/slates_binned/*`, whose `step{k}.pt` layout the other two cannot read — `scripts/probes/binned_pool_cache.py` |
| particles → an image to plot at all | `transforms/functional.py::particles_to_occupancy`; the plate as a channel is `draw_plate_soft` |
| world action → pixel endpoints | `fit_linear_foresight.py::actions_to_pixels` (`pool_common.action_geometry` wraps it) |
| a push drawn on a raw image (OpenCV, not matplotlib) | `utils.py::drawPushing`, `drawRotatedRect` |
| simulator video of an episode | `utils.py::write_video_frame` + the `on_phase`/`on_step` hooks on `SandboxManipulation.execute_action` (see `docs/UTILITIES.md` — this hook pattern is the documented reference example) |
| predicted-vs-true rollout video | `simple_mpc/debug_vis.py::save_predicted_trajectory_video` |
| OT planner diagnostics (distributions, vector field, divergence) | `simple_mpc/ot_planner.py::plot_*` |
| per-episode reward curve | `simple_mpc/human_mpc.py::_plot_episode_reward` |
| **one (state, push) transition through one model** (2x3: input, the model's own action channels, truth, RMS/accuracy/changed-IoU bars, prediction, signed error map) | `scripts/probes/transition_panel.py --model <id> --index i ...` |
| push-by-push GIF of a recorded closed-loop episode (goal mask, cubes, blade; side-by-side cells) | `experiments/EXP-0051-*/code/demo_gifs.py::draw_frame` / `write_gif`; for EXP-0055/0056 cells `experiments/EXP-0055-*/code/demo_gifs.py`. Draw the world **x right, y DOWN** -- the letter goals read correctly only that way |
| a rotated-square cube footprint (from yaw), a push arrow, or a blade/plate line segment perpendicular to a push -- the 3 primitives that were independently re-derived in `transition_panel.py`/`demo_gifs.py`/the retrieval debug figure below | `scripts/probes/cube_viz.py::cube_patch`/`push_arrow`/`blade_endpoints` (matplotlib patches only; not a general framework -- callers still own their own figure/axes) |
| **retrieval-model visual debug** (EXP-0059): global-frame query state + push, query vs its top-1/top-5 neighbours' pre-push states with Hungarian correspondence, 1-NN and combined (k5 cube-median) transferred predictions vs truth, per-cube mm error + confidence, all in the query's own push frame | `experiments/EXP-0059-retrieval-transition-model/code/retrieval_debug.py::plot_retrieval_debug` (function) + its CLI (batch of ~12 queries → `experiments/EXP-0059-*/figures/retrieval_debug/`, indexed by `README.md`) |

Corpus-level structure (trajectories, per-slate sequences, per-bin selections)
is queried through `Genesis/binned_slate_dataset.py::BinnedSlateCorpus`, not by
re-reading the files — see the `data-collection` skill.

## Configuration and code ownership

Project-level files under `configs/` define reusable formats, defaults, and
schemas (see the Major Parts row above for the exact rule governing template
versioning). When an object is created, copy the relevant template into the
object's own directory and configure it there; the instance copy is the
authoritative resolved configuration for that object, and later edits to the
project template must never silently change it.

Experiment-local code is allowed, but should remain small and genuinely
local. A new composition of existing project functionality normally belongs
in the experiment's local entry point/orchestration rather than in a new
reusable module. A new reusable utility or capability belongs in the
appropriate project module and should be added to the architecture map when
another part of the codebase is expected to use it. Before creating a new
helper, search for an existing implementation and for the appropriate owning
module — and do not create a generic `utils.py`/`analysis.py`-style dumping
ground inside an experiment when a clearer local name, or the existing
architecture, already fits.

## Documentation policy (do this as part of the change)

**Any change to a major information flow, a module's responsibility, a
default value, an added feature, or a reusable capability updates the
relevant doc(s) above in the same change — not as a follow-up, and not only
when explicitly asked.**

Concretely, before considering such a change done:
- New reusable module, class, or function → add it to `ARCHITECTURE.md`'s module map (and `UTILITIES.md` if it is a reusable utility rather than a subsystem-specific piece).
- New or changed batch dict key, model output shape, or adapter method → update `INTERFACES.md`.
- New config template/default, changed hyperparameter meaning, or a new knob → update the owning config/doc and `ARCHITECTURE.md` / subsystem design doc if structural.
- Anything touching the oracle MPC subsystem specifically → update `docs/oracle_mpc_design.md`'s relevant section.
- Any experiment, probe, fit, sweep or benchmark whose number might be cited → record it through the `experiment-log` skill. A number in a commit message or chat log is not an evidence record.
- A genuinely new subsystem on the scale of oracle MPC → give it its own `docs/<name>_design.md` following the existing design-doc pattern and link it from `ARCHITECTURE.md` and this map.

**Prefer small, targeted architectural changes.** Do not create a new
project-level module or document merely to house one experiment's composition
of capabilities.

## Every job must survive being cut off (standing rule, 2026-09-24)

A run can be killed at any moment (session restart, OOM, a shared-GPU
collision). The cost of that must be minutes, not the whole run. So every
script that runs for more than a few minutes -- collection, training,
evaluation, sweeps, probes -- **writes its progress to disk as it goes**:

- **Checkpoint after every unit of work** (per model, per corpus, per slate,
  per batch, per epoch): rewrite the results / manifest file with everything
  finished so far. Replacing the previous checkpoint each step is fine -- the
  point is that the latest completed unit is always on disk.
- **Write atomically**: write `<file>.tmp`, then `os.replace` it onto the
  target, so a kill mid-write never leaves a truncated file.
- **Keep a manifest** beside large outputs: what has been produced, with what
  config, up to which unit -- so a later session can see exactly how far a run
  got without re-reading the payload.
- A restart-from-midpoint path is NOT required in advance; a run whose
  partial results are on disk can always be completed or re-scoped later.
  A run whose results exist only in memory cannot.

Reference implementations: `Baselines/common/eval_report.py::_write_json_atomic`
(results rewritten after every model), `simple_mpc/gt_bank.py` (atomic per-state
files), EXP-0027 `stage1_grad_many.py` (saved after every arm).

## Environment

Genesis-dependent code (`Genesis/*`, `env/genesis_env.py`,
`simple_mpc/genesis_oracle.py`, and anything importing them) requires the
`genesis` package and a GPU; everything else in the module map is Genesis-free
and covered by the fast pytest suite. Activate the project's conda environment
before running Python: `conda activate pme`.
