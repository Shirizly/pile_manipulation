# Interfaces

This document defines data contracts between dataset wrappers, model wrappers, losses, metrics, and MPC adapters.

## 1. Representation-Agnostic Batch Contract

All dataset wrappers should return a dict with representation-specific keys. Optional keys are omitted (not set to `None`).

### 1.1 Common keys

```python
{
    "representation": str,      # eulerian | lagrangian | ...
    "physics": Tensor[B, P],    # optional, normalized when used by model
    "metadata": dict,           # optional run/sample metadata
}
```

### 1.2 Eulerian batch example

```python
{
    "input":   Tensor[B, C, H, W],
    "target":  Tensor[B, H, W],
    "physics": Tensor[B, P],
    "push_px": Tensor[B, 4],   # optional — see below
}
```

`push_px` (optional): `[start_col, start_row, end_col, end_row]`, pixels,
**(col, row)** order — the same convention `transforms/functional.py`'s
push-frame warp (`to_push_frame` et al.) uses, and what a push-frame /
canonical-frame model needs alongside the occupancy. When present,
`training/trainer.py::_augment_eulerian_batch`'s 4-rotation x 2-flip
spatial augmentation carries it through the SAME `rot90`/`flip` pixel
transform applied to `input`/`target` (see
`model/warped_nfd/WARPED_NFD_NOTES.md` for the derivation), so the endpoints
stay geometrically consistent with the augmented images; when absent,
augmentation behaviour is unchanged from before this key existed.

### 1.3 Lagrangian batch example

```python
{
    "a_cur":            Tensor[B, N],
    "s_cur":            Tensor[B, N, 3],
    "s_delta":          Tensor[B, N, 3],
    "target_particles": Tensor[B, N, 3],
    "particle_dens":    Tensor[B],
    "particle_nums":    Tensor[B],       # optional for variable particle count
    "physics":          Tensor[B, P],    # optional
}
```

Rules:
- Consumers must branch by available keys (or `representation`) rather than assuming `input`/`target` always exist.
- Eulerian channel conventions are representation-local and documented under the Eulerian section.
- Representation conversion (particle to grid, action rasterization) should happen in transform/wrapper layers, not in generic trainers.

Extended optional keys (migration-safe):

```python
{
    "current_occupancy": Tensor[B, H, W],
    "target_occupancy":  Tensor[B, H, W],
    "score_map":         Tensor[H, W] | Tensor[B, H, W],  # ScoreMapWeightedLoss only
}
```

When present, Eulerian losses should prefer explicit occupancy keys over implicit channel slicing.

`score_map` is a fixed goal-reward landscape (e.g. from
`simple_mpc.occupancy_reward.OccupancyReward.compute_score_tensor`), consumed
only by `ScoreMapWeightedLoss` (`training/losses.py`) — it lets MPC
optimization use literally the same objective as reward reporting. Other
losses ignore it.

## 2. Model Output Contract

External consumers should treat model output as a structured pair:

```python
ModelOutput(
    logits=Tensor[B, 1, H, W] | Tensor[B, H, W],
    probabilities=Tensor[B, 1, H, W] | Tensor[B, H, W] | None,
)
```

Contract:
- `logits` are raw unconstrained outputs.
- `probabilities` are occupancy probabilities in `(0, 1)`.
- If `probabilities` is missing, callers derive it with `torch.sigmoid(logits)`.

This removes ambiguity at train/eval/MPC boundaries.

## 3. Core Signatures

### 3.1 Generic training wrapper

```python
ModelTrainingWrapper.forward(batch: dict) -> Tensor | ModelOutput
```

This signature is representation-agnostic; tensor shapes depend on model family.

### 3.2 Eulerian model signatures

```python
forward(x: Tensor[B, 2, H, W], props: Tensor[B, 3]) -> Tensor[B, 1, H, W]
```

Returns raw logits. Sigmoid is applied by losses/metrics or wrappers.

### 3.3 Lagrangian model signatures (GNN)

```python
PropNetDiffDenModel.predict_one_step(
    a_cur: Tensor[B, N],
    s_cur: Tensor[B, N, 3],
    s_delta: Tensor[B, N, 3],
    particle_dens: Tensor[B],
) -> Tensor[B, N, 3]
```

Lagrangian dynamics operate on particle states directly, with no occupancy-grid requirement at model input.

### 3.4 Adapter surface for MPC

```python
obs_to_state(obs_np) -> state
expand_state(state, n_sample: int) -> state_batch
predict_step(state_batch, act_batch) -> state_batch
compute_reward(state_batch) -> reward_batch
```

Adapter state is representation-specific:
- Eulerian state: occupancy grid tensor `(B, Nx, Ny)`
- Lagrangian state: particle tensor `(B, N, 3)`

`simple_mpc.oracle_mpc` (the Genesis-as-model sampling MPC — see
`docs/oracle_mpc_design.md`) does not implement this adapter surface at all:
its "model" is the Genesis simulator, so `predict_step` is a real physics
rollout (`GenesisOracleEnv.rollout_candidates`), not a learned forward pass,
and there is no `compute_reward`/`ModelAdapter` object to swap. It reuses the
same occupancy conventions (§4.1) and the training loss registry (via
`per_sample=True`, see below) rather than duplicating the adapter contract.

### 3.4b Occupancy-grid gradient adapter surface (EXP-0023)

The §3.4 surface is built around a rendered `(H,W,5)` observation and a
camera. The `Baselines/` occupancy models (NFD family, warped/residual NFD,
switched-linear pixel operators) have no camera: they consume a 64x64
world-frame occupancy over the fixed +/-64 mm slate workspace and are scored
with `control_utility_test.lyapunov`. `simple_mpc.adapters` therefore carries
a SECOND adapter family for them, with its own surface:

```python
state_from_particles(states)   -> occ          # (B,n_particles,>=3) metres -> (B,64,64), = encode_state(raster)
encode_state(occ)              -> occ_in       # raster -> this model's INPUT representation
expand_state(state, n_sample)  -> occ_batch
predict_step(occ, act)         -> occ1         # act (B,4) world-metre [sx,sy,ex,ey]
value(occ)                     -> (B,)         # Lyapunov COST, lower is better
dv(occ0, act)                  -> (B,)         # value(after) - value(before)
```

**`predict_step` and `dv` are required to be differentiable w.r.t. `act`.**
The plate yaw is derived from the travel direction by
`transforms.functional.action_to_pose` (matching the corpora's own `angles`),
so the action has exactly four free parameters and no yaw argument.

**`encode_state` is identity except for soft-occupancy models** (EXP-0063:
`input_blur_sigma > 0`, read by `Baselines/NFD/predictor.py::input_blur_sigma_from_run`
from the checkpoint's own `run_config.yaml`). Apply it to every model input
built from particles (`occ_from_particles`), and NEVER to a predicted
occupancy being rolled forward -- the prediction is already in the model's
representation. Callers that bypass `state_from_particles` must call it
themselves: `eval_extended.py::_encode`, `multistep_eval.py::run_occ`,
`learned_mpc.ModelObjective` (per ensemble member, with that member's own
start value).

Selection is by string id, not by `isinstance`:
`make_occ_adapter(model_id, device, goal_shape)` over the `OCC_ADAPTERS`
registry -- the live-gradient counterpart of
`Baselines/common/eval_report.py`'s `MODELS` dict. Registering a model is one
entry.

**A `Baselines/*` predictor's `predict_occ(batch) -> (B,H,W)` (§ `Baselines/
common/eval_baseline.py`) is an OFFLINE SCORING interface and is
`@torch.no_grad()`.** It is not this contract. `PredictorGradientAdapter`
bridges the two by calling the predictor's own `predict_occ.__wrapped__`, so
the differentiable path and the offline path are literally the same code.

**SIGN CONVENTION, repo-wide for `dv`:** `dv = value(after) - value(before)`,
a COST -- lower is better, an optimiser MINIMISES it. Asserted by
`simple_mpc.adapters.assert_dv_convention`; identical to
`scripts/probes/binned_pool_cache.py` and `experiments/METRICS.md`.

### 3.5 Loss contract's `per_sample` mode (MPC cost use)

Losses in `training/losses.py` accept an optional `per_sample: true` config
key (see its module docstring). With it set, `total_loss` is an unreduced
`Tensor[B]` (one cost per batch item) instead of a scalar — this is how
`simple_mpc.oracle_mpc` ranks candidates using the exact same registered
losses training uses, rather than a parallel cost-specific implementation.
Training code should never set this; it's for candidate-ranking callers only.

## 4. Eulerian Representation Details

This section is specific to Eulerian wrappers and occupancy-based models.

### 4.1 Coordinate conventions

Genesis world frame:
- `world_x`: table right
- `world_y`: table forward
- `world_z`: up

EulerianWrapper occupancy convention:
- 2D grid uses `dim 0 = world_x`
- 2D grid uses `dim 1 = -world_y`

Dataset/grid convention:
- Grid tensors use `dim 0 = world_y` (rows)
- Grid tensors use `dim 1 = world_x` (cols)

Bridge requirement:
Any conversion path between wrapper-state and dataset-state must apply the same flip/transpose policy in forward and inverse directions.

FleX corpora (`FlexData/dataset.py`, DS-0019/DS-0020; EXP-0061):
- Units are native FleX workspace units, not metres; every geometric quantity
  (grid bounds, plate width, thresholds) comes from the dataset's instance
  `config.yaml`, exposed through the same `raw.to_pxl` / `raw.ctr_in_PXL` /
  `raw.configs[0]["plate"]["size"]` / `raw.configs[0]["box"]["vol"]` /
  `raw.workspace_bounds` attributes `PileSweepData` has, plus
  `raw.plate_width_px` (read by `fit_linear_foresight.plate_width_px`) and
  `raw.particles_before(i)` / `raw.particles_after(i)` (variable particle
  count, so no `states` tensor).
- DS-0020 has two payloads: v2 (current, `datasets/DS-0020-*/config.yaml`,
  `cache_format: traj_manifest_v2`, image-mask input only) and the archived
  v1 (`old_data/_ported_v1/config.yaml`, `chunks_v1`). Both give the same
  row / group / step interface (row = (trajectory, push k); before = state k,
  after = state k + 1).
- Table frame: `X = x_flex`, `Y = -z_flex` (FleX y is up). This is the frame
  the stored actions are in; the particle files' z is negated on load.
  Grid: `dim 0 = X`, `dim 1 = Y`; pixel coordinate `world * to_pxl + ctr`,
  occupancy pixel centres at integer coordinates (same as `draw_plate_soft`),
  so `truth_for_scoring`'s Genesis `-1.0` px offset is replaced by
  `raw.score_uv_offset_px = 0.0`.
- The batch dict keys/shapes are identical to the Genesis `nfd-genesis-3ch`
  (`channels: 3`) / `genesis` (`channels: 2`) datasets; `physics` is a zero
  vector (omitted unless `include_physics: true`).

### 4.2 Sigmoid requirement at MPC boundary (Eulerian)

Reward paths clamp states to `[0, 1]`. Feeding raw logits into this clamp saturates gradients and can silently kill optimization.

Required behavior:
- MPC-facing predictions must be occupancy probabilities in `(0, 1)`.
- If model internals are logit-based, convert before returning from the MPC-facing wrapper.

## 5. Lagrangian Representation Details

This section is specific to particle-based models and adapters.

### 5.1 State and action semantics
- State is particle positions (and optionally per-particle attributes), not an occupancy grid.
- Actions are converted to per-particle displacement fields before model rollout.
- Rewards for optimization should match particle representation when possible.

### 5.2 Reporting compatibility
- Cross-model comparison may project particles to occupancy for reporting only.
- This projection should not replace representation-native optimization losses/rewards.

## 6. Compatibility Notes

- Legacy callers can still pass raw tensors; helper conversion utilities should normalize this to the ModelOutput contract.
- New code should prefer explicit typed contracts in `training/types.py`.
- Representation-specific losses/metrics should validate required keys early and fail loudly on mismatch.

Known gaps:
- `GNNAdapter.compute_reward` (the `'default'` particle reward for MPC
  optimization) is unported: the original FlexEnv `config_reward_ptcl` has no
  Genesis equivalent yet, so it raises `NotImplementedError`. The `'iou'` and
  `'eulerian'` reward types remain available for reporting. Lagrangian MPC
  optimization is blocked until a particle reward is ported.
