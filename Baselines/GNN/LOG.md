# GNN baseline — LOG

## SUMMARY (agent B1-gnn-impl, implementation phase, 2026-09-08)

**Status: implemented, training in progress / scored.** (Update this line as
the run progresses; see "Running notes" below for the live trail if this
agent crashes mid-run.)

**Files delivered (all new, per SPEC.md section 7 — none of the vendored
`Baselines/GNN/{dataset,train,model}/*` files were touched):**
- `Baselines/GNN/geometry.py` — `compute_s_delta` (SPEC.md section 6 recipe,
  heading from `p_stop - p_start` only, never `angles`), and the three
  chosen constants: `PUSHER_W=0.02` (sourced from `plate.size`),
  `SOFTNESS=0.01` (kept from the reference), `ADJ_THRESH=0.012` (chosen
  after the geometry check below), `PARTICLE_DENS=1000.0` (fixed constant,
  per SPEC.md hazard 3).
- `Baselines/GNN/scripts/check_geometry.py` — the pre-training sanity check
  SPEC.md section 8 demands. Results below.
- `Baselines/GNN/dataset/dataset_genesis_gnn.py` — `GenesisGNNDataset`,
  reads raw `_*_data.pt` files directly (NOT `dataset/dataset_gnn_dyn.py`),
  flattens each file's 128 rows to independent single-step examples.
- `Baselines/GNN/train/train_genesis_gnn_dyn.py` — Adam(lr=1e-3,
  betas=(0.9,0.999)) + StepLR(100, 0.5), single-step (`n_rollout=1`) MSE
  loss on xyz only, batch 128, checkpoints every 10 epochs to
  `Baselines/GNN/runs/{ckpt_best,ckpt_last}.pth` + `history.json`.
- `Baselines/GNN/predictor.py` — `build_predictor()` for
  `Baselines/common/eval_baseline.py`; loads a checkpoint, runs
  `model.predict_one_step`, reattaches the INPUT quaternion (model has no
  orientation head), rasterises per-row via
  `Baselines.common.data.rasterize_particles` (harness's own routine, not
  reimplemented).

**Geometry sanity check (`check_geometry.py`, run BEFORE training):**
- s_delta mask lights up sensibly, not saturated/empty: for `n20_L20mm`
  (20mm push), ~5-16 of 20 cubes get non-trivial `s_delta` per candidate
  (mean 7.1), matching the short push's narrow longitudinal window
  (`x_loc` in `(0, 0.02m)`); for `n20_L40mm` (40mm push), 16-20 of 20 cubes
  are active (mean 19.1) — the whole pile falls inside the longer window,
  which is the expected behaviour of a hard longitudinal gate whose length
  scales with push length, not a bug.
- Printed the per-cube `(x_loc, y_loc, l_mask, w_mask)` breakdown directly
  (not just a lumped "distance to segment", which conflates the hard
  longitudinal gate with the soft lateral one and is misleading — caught
  this myself mid-check and rewrote it): cubes with `|y_loc| < PUSHER_W
  (0.02m)` correctly get `w_mask≈1.0`; cubes outside decay smoothly over
  the `SOFTNESS=0.01m` scale. No axis swap, no all-or-nothing collapse.
- Adjacency degree distribution swept over `adj_thresh ∈
  {0.008,0.010,0.012,0.015,0.020}` m (all built on `s_cur + s_delta`, exactly
  `PropNetDiffDenModel.predict_one_step`'s own construction, not
  reimplemented — `check_geometry.py` mirrors only the adjacency half to
  avoid a full forward pass): min in-degree is 2-4 at every value tested
  (never an isolated node), and `adj_thresh=0.012` is NOT saturated at the
  hard top-10 cap (`frac_at_cap` 0.47 for L20mm, 0.81 for L40mm — a real
  distance-threshold effect, not "always top-10-NN regardless of the
  literal value", the failure mode SPEC.md hazard 2 warned about).
  **Chose `adj_thresh=0.012` m** — matches SPEC.md's own recommended range
  (2-3x cube edge = 0.010-0.015m) and sits in the middle of a healthy,
  non-degenerate part of the sweep.

**Training recipe chosen:** 500 epochs, batch 128, Adam lr=1e-3, pooled
train pool (23,040 transitions), StepLR(100,0.5). Measured ~2.5s/epoch on
the shared GPU including a full validation pass on the pooled eval split
(15,360 transitions) — a 500-epoch run is ~20 min, well inside budget (the
38k-parameter model trains far faster than SPEC.md's own conservative
estimate). Smoke-tested first (`--smoke`, 2 epochs) before committing to
the full run, per the task's priority-order rule.

---

# GNN baseline — agent A1-gnn-spec log (design phase, superseded by the
# implementation above where the two disagree — SPEC.md's own
# ORCHESTRATOR AMENDMENT already documents the one place they did)

## Summary

Task: write `SPEC.md` (design doc, no training) for adapting the vendored
`PropNetDiffDenModel` GNN (Wang/Li et al., "Dynamic-Resolution Model
Learning for Object Pile Manipulation") to train on our Genesis
`slates_multistep` cube-pile data, at a **constant** N=20 node count
(resolution regressor explicitly out of scope per user decision recorded
in `Baselines/ORCHESTRATION_LOG.md`).

**Delivered:** `Baselines/GNN/SPEC.md`. No training run, no model code
changes — as instructed.

**Headline findings:**
- Graph = one node per cube (N=20 fixed), edges = distance-threshold AND
  top-10-nearest-neighbour, built on the *anticipated* post-action
  position `s_cur + s_delta`, not the current position.
- The action never becomes extra "pusher" nodes — despite the code's
  comments, `a_cur` (the node-type attribute) is always zero in both the
  reference dataset and the existing `simple_mpc/adapters.py::GNNAdapter`.
  The action instead becomes a per-node vector field `s_delta`
  (direction-to-tool-end, masked to the swept lane) fed into the particle
  encoder and used (only) to shape the graph topology.
- Architecture confirmed from the vendored checkpoint's `state_dict`
  (not guessed): `nf_effect=64` throughout, `pstep=3` message-passing
  rounds (hardcoded in `model/gnn_dyn.py:160`, not a config value), total
  **38,403 trainable parameters** — a tiny model.
- Three numeric constants from the reference recipe (`adj_thresh=0.08`,
  `pusher_w=0.8/24`, `particle_dens` sampled `Uniform(15,6500)`) are all
  tuned to a different simulator's own arbitrary coordinate scale and must
  not be reused verbatim on our metre-scale data. Found a sourced
  replacement for `pusher_w` (0.02 m, from the blade's own
  `plate.size: [0.04, 0.002, 0.01]` in a cell's `_0_config.yaml`);
  `adj_thresh` and the mask softness are argued from first principles but
  not empirically validated — flagged as the top open risk.
- Found and numerically **confirmed** a genuine frame hazard: our
  `angles` field is not the push heading, it's `heading + 90°` (blade face
  orientation), verified exactly across 4 samples. Not needed as a model
  input at all under this recipe, but a serious footgun if fed naively.
- Our data is much simpler to consume than theirs: no RGBD, no FPS/KDTree
  particle tracking, no camera/OpenGL transform chain — states are exact
  per-cube positions already. Recommended a new dataset+train script
  rather than patching the vendored ones.
- `manifest.json`'s 50-slates×3-steps×128-envs structure does not record
  which of a slate's 128 step-0 candidates was executed to reach step 1 —
  so multi-step (`n_rollout>1`) supervision, which the paper's own recipe
  uses, is not reconstructable without more digging. Recommended
  `n_rollout=1` (single-step) training as the safe, data-matching choice;
  flagged multi-step as an open question.
- Confirmed the "~23k transitions" figure in the task brief exactly:
  90 train files/cell × 2 cells (L20mm+L40mm pooled) × 128 rows = 23,040,
  using the existing `prepare_slate_multistep_split.py` 30/20 slate-level
  split (which must be reused, not re-derived, to avoid leaking a slate's
  3 steps across train/eval).

## Running notes (chronological)

- Read `Baselines/ORCHESTRATION_LOG.md` first, as instructed. Confirmed
  scope: GNN only, no resolution regressor, N=20 fixed. Confirmed
  `model/gnn_dyn.py` at repo root is byte-identical to
  `Baselines/GNN/model/gnn_dyn.py` (already known from the orchestration
  log; not re-verified by hand, took the log's word for it — cheap and
  low-risk to trust here since both are read-only references).
- Read `model/gnn_dyn.py`, `model/eulerian_wrapper.py`,
  `model/diff_mass_push.py` in full. Noted `eulerian_wrapper.py` and
  `diff_mass_push.py` are **not used by the GNN path at all** — they
  belong to the sibling Eulerian/NFD baseline (`SplatPushModel` etc.).
  They were in this agent's source list but turned out irrelevant to the
  GNN spec beyond confirming that `_gen_s_delta`-style differentiable
  action-conditioning exists elsewhere in the repo too (parallel
  construction, different model family).
- Read `train/train_gnn_dyn.py`, `dataset/dataset_gnn_dyn.py`,
  `data_gen/gnn_dyn_data.py`, both yaml configs. Found the collate
  function pads to the batch's max `particle_num` — a variable-N
  accommodation that's moot for our fixed N=20.
- Loaded the vendored checkpoint's `state_dict` directly (no conda
  activation needed — the plain shell python already had torch on path)
  and printed every tensor's shape. This is the ground truth for §2 of
  the spec; matched every shape back to the corresponding `nn.Module`
  constructor call in `model/gnn_dyn.py` by hand.
- Loaded a reference sample episode
  (`Baselines/GNN/data/gnn_dyn_data/0/`): particle npy is raw un-normalized
  PyFlex world coordinates (4 columns, 4th overwritten to 1.0 for the
  homogeneous transform); `actions.p` is a plain pickled
  `(10,4)` float64 array, range ≈ ±5, matching `wkspc_w=5.0`.
- Loaded our sample cell
  (`Genesis/data/slates_multistep/n20_L20mm/_0_data.pt`) and did the
  measurements in SPEC.md §4: confirmed all 128 rows share one starting
  state (std ≈ 4.8e-12), push length is exactly 0.02 ± 1e-6 m (matches
  cell name), push is planar (z-component of the push vector ≤ 9.4e-7 m),
  and — the one non-obvious finding — `angles` is exactly
  `atan2(dy,dx) + π/2`, not the heading itself, checked across the first
  4 rows with all 4 differences landing on π/2 to float precision.
- Read `simple_mpc/adapters.py::GNNAdapter` and its `_gen_s_delta` helper
  in full (as instructed — "someone previously intended to drive this
  model"). Confirmed it independently re-derives the same `s_delta`
  construction as `dataset_gnn_dyn.py`, and confirms `a_cur` is zeroed
  there too — good cross-check that the "no real pusher-node attribute"
  finding isn't a one-off dataset quirk.
- Extracted the paper's method section (PDF pages 3-4, via `pymupdf`
  since `pdftoppm`/`poppler-utils` was not installed and the `Read` tool's
  page-render path failed — used `fitz`/`pymupdf`'s text extraction
  instead, which was available). Confirmed the paper's high-level
  description (node/edge encoders, multi-step message passing "following
  Li et al.", MSE loss over a rollout horizon) matches the code's actual
  implementation; the paper itself carries no appendix with layer-size
  hyperparameters in these 11 pages (no separate supplementary-material
  PDF was found in `docs/reference_papers/`) — all architecture numbers
  in the spec come from the code/checkpoint, exactly as the task
  requested ("say which came from which").
- Grepped `Genesis/*.py` for the box/cube/blade physical constants needed
  to replace the reference recipe's now-inapplicable numeric constants.
  Found cube edge (0.005 m) and box volume (0.128×0.128×0.04 m) in a
  cell's own `_0_config.yaml` (`data_collection.cube_size`, `box.vol`) —
  matches the task brief's stated "0.128 m across" exactly. Found the
  blade/plate geometry the same way (`plate.size: [0.04, 0.002, 0.01]`),
  giving a sourced value for the pusher half-width (0.02 m) instead of
  guessing. Did **not** find an explicit named "adj_thresh"-equivalent
  contact radius anywhere in the Genesis code — that constant has no
  physical-simulator counterpart to source from, so §5/§9 of the spec
  correctly labels it as argued-from-first-principles, not measured.
- Read `scripts/probes/prepare_slate_multistep_split.py` in full — this
  is where the "50 slates × 3 steps × 128 envs, 30/20 slate-level split"
  structure and the exact 23,040-transition arithmetic came from. Also
  where the "which of the 128 step-0 candidates continues to step 1"
  gap was noticed (checked `manifest.json`'s `batches` list directly:
  fields are `batch_idx, slate_idx, state_library_index, step_idx,
  env_count, unresolved_after_resampling` — no executed-candidate index).
- Did not read `docs/experiments/METRICS.md` or re-derive
  `slateK_exact`/`accuracy` end-to-end — out of this agent's declared
  scope (design doc for the *model*, not the eval harness) and flagged
  explicitly as an unverified assumption in SPEC.md §9 item 4, rather than
  silently assumed.

## Tool-call budget

Used roughly 40 tool calls (reads, greps, small python snippets via Bash,
two Write calls). Stayed within the 40-55 budget given in the brief.
