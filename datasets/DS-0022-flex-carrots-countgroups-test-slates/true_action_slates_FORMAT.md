# `true_action_slates_*` data format

Ground-truth (state, action) -> outcome triples for a pile-pushing task.
Produced by `collect_true_action_results.py` / `collect_true_action_results_parallel.py`
(the two are byte-for-byte interchangeable — same seeding, same on-disk layout).
Copy the whole dataset folder (e.g. `true_action_slates_objbiased/`) as a unit;
this doc describes any of the folders in `data/`:

- `true_action_slates_objbiased` — 100 states x 200 actions, carrots, action
  starts biased near the pile (`action_sampler: obj_biased`).
- `true_action_slates_mixed` — 100 states x 200 actions, carrots, actions
  uniform over the workspace.
- `true_action_slates_binned` — 20 states x 120 actions, carrots, uniform
  actions (smaller/older run, same format).
- `true_action_slates_objbiased_coffee` — 100 states x 200 actions, **coffee**
  beans instead of carrots, otherwise identical config to `objbiased` (same
  seed, bin edges, `obj_biased` sampler, particle-count target) — collected
  for cross-material transferability testing. See "Coffee vs. carrots" below
  for what differs physically.
- `true_action_slates_objbiased_capsule` — 100 states x 200 actions,
  **capsule** pieces instead of carrots/coffee, otherwise identical config to
  `objbiased`. Same per-state-jitter variety mechanism as coffee (see
  "Coffee vs. carrots" below; capsule's particle-count calibration keeps
  `particle_r` at its original 0.075 rather than coarsening it further --
  see `.github/skills/pyflex-simulation/SKILL.md`'s capsule resolution table
  for why coffee had more headroom to do that).

For each state, one initial particle configuration is fixed, then
`n_actions` independently-sampled pushes are each replayed from that *same*
initial state (not a trajectory — every action is a fresh branch off the same
start). This isolates (state, action) -> outcome from any particular
rollout/policy.

## Directory layout

```
true_action_slates_<name>/
  manifest.jsonl          # one JSON record per line; the source of truth
  run_config.json         # the full config this run was launched with
  <state_idx>/            # one folder per initial state, e.g. 0/, 1/, ...
    initial_particles.npy
    initial_color.png
    initial_state.npz     # parallel collector only (see below)
    <action_idx>_after_particles.npy
    <action_idx>_after_color.png
```

`state_idx` runs `0 .. n_states-1`. `action_idx` runs `0 .. n_actions-1`, but
not contiguous per state: an action whose push caused a solver explosion (or
that never got a clean replay) is marked invalid and has no `_after_*` files,
so some `action_idx` values are missing from a state's directory. **Always go
through `manifest.jsonl`** to know which actions exist and whether they're
valid, instead of listing files.

## `manifest.jsonl`

Append-only JSON Lines log, two record types, identified by `"type"`:

### `state_init`

```json
{"type": "state_init", "state_idx": 0, "n_particles": 4221,
 "positions_path": "0/initial_particles.npy",
 "color_path": "0/initial_color.png",
 "init_pos": "rand_blob", "timestamp": 1789722624.88}
```

One per state, logged once when that state's initial pile is settled.
`init_pos` is only present when the run mixes initial-pile shapes
(`slates.init_pos_mix` in the config, e.g. `objbiased`/`mixed` alternate
`rand_blob`/`rand_spread` by `state_idx % 2`); when a run uses a single fixed
shape (`binned`) it's omitted — see `run_config.json`'s `dataset.init_pos` /
`slates.init_pos_mix` instead.

### `action`

```json
{"type": "action", "state_idx": 0, "action_idx": 0,
 "action": [x0, z0, x1, z1], "valid": true, "env_slot": 0,
 "bin": 0, "push_length": 1.21,
 "after_positions_path": "0/0_after_particles.npy",
 "after_color_path": "0/0_after_color.png", "timestamp": 1789710981.06}
```

- `action`: `[start_x, start_z, end_x, end_z]` — the pusher's straight-line
  path in world/workspace coordinates (XZ plane; the push height is derived
  at runtime from the pile's current max particle height, not stored here).
  Coordinates lie in `[-wkspc_w, wkspc_w]` minus `action_margin` (see
  `run_config.json`'s `dataset.wkspc_w` / `dataset.action_margin`).
- `push_length`: euclidean length of that path, `norm([x1,z1] - [x0,z0])`.
- `bin`: index into `slates.bins.edges_abs` / `edges_frac` in
  `run_config.json` — actions are planned in equal-count length bins so a
  batch of pushes costs about what its longest push costs. `bin` 0 = shortest
  pushes. (`-1` would mean "no binning", not used in these three datasets.)
- `valid`: whether the push completed without the solver blowing up. When
  `false`, there are no `after_*` files for this `action_idx` — see
  `"failure"` (`"solver_explosion"` mid-push, or `"repeated_batch_abort"`
  after repeated retries). Invalid actions are still recorded exactly once
  (never silently dropped), so `done` bookkeeping/resume logic can treat the
  manifest as complete.
- `env_slot`: which tile of the parallel batch produced this result — an
  implementation detail of the parallel collector, not meaningful for
  training.

Records are **not** sorted by `action_idx` within a state — the parallel
collector processes a whole length-bin as one batch, so build an index (e.g.
`{(state_idx, action_idx): record}`) rather than assuming order, and filter
on `valid` before using `after_*` paths.

## Per-state files

- `initial_particles.npy`: flat `float32` array, length `4 * n_particles`.
  Reshape to `(n_particles, 4)` for `(x, y, z, inv_mass)` per particle (FleX's
  native particle layout; `inv_mass` is not physically meaningful for this
  task — all carrot particles share the same mass).
- `<action_idx>_after_particles.npy`: same shape/layout, the particle state
  after that action's push settles. Both `initial_particles.npy` and every
  `_after_particles.npy` for a state have the same `n_particles` (also
  logged in the state's `state_init` record) and index particles in the same
  order, so `after[i] - initial[i]` is a valid per-particle displacement.
- `initial_color.png` / `<action_idx>_after_color.png`: `720x720` RGB render
  of the pile from the fixed camera (`dataset.cam_idx`), background pixels
  (depth above a global-scale-derived threshold) whitened to pure white —
  same convention as `data_gen/gnn_dyn_data.py`. No depth channel is saved
  (the renderer produces RGBD; only RGB is written to disk).
- `initial_state.npz` (parallel collector runs only, i.e. all three of these
  datasets): the full resumable sim state for this state — `positions`
  `(n_particles, 4)`, `velocities` `(n_particles, 3)` (zeroed), and the rigid
  cluster shape-matching state `rigid_rotations` `(n_rigids, 4)` /
  `rigid_translations` `(n_rigids, 3)`. This is what's teleported back in
  before every action's push (`env.set_all_envs_state`), so it's the actual
  "state" half of each (state, action) pair — `initial_particles.npy` is a
  derived convenience view of `initial_state.npz['positions']`.

## `run_config.json`

The exact config the run was launched with (so results are reproducible /
interpretable without hunting down which YAML was used), plus
`parallel_resolved` (actual `n_envs`, tile grid shape, tile spacing/seed —
present in all three since all three were collected with the parallel
collector). Notable fields:

- `dataset.global_scale`, `wkspc_w`: scene scale and half-width of the
  square action workspace (actions and `push_length` are in these units).
- `dataset.action_margin`: inset subtracted from `wkspc_w` before sampling
  action start/end points (keeps the pusher off the walls).
- `slates.n_states`, `n_actions`, `seed`: dataset size and the RNG seed that
  deterministically generates every initial state and every action (same
  `seed` + `state_idx`/`action_idx` always reproduces the same push — this is
  what makes the serial and parallel collectors produce identical datasets).
- `slates.bins.edges_abs` / `edges_frac`: push-length bin edges (`edges_abs`
  in workspace units directly; `edges_frac` as a fraction of the maximum
  possible push length `sqrt(2) * (hi - lo)`, see `plan_binned_actions` in
  `collect_true_action_results_parallel.py`).
- `slates.action_sampler`: `"uniform"` (both endpoints uniform in the
  workspace) or `"obj_biased"` (start point drawn near an actual pile
  particle, end point uniform) — only changes *where* actions are sampled
  within a bin, not the bin edges, so datasets with different samplers stay
  comparable bin-for-bin.
- `slates.init_pos_mix`: if present, the initial pile shape alternates across
  this list by `state_idx % len(init_pos_mix)` (e.g. `objbiased`/`mixed`
  alternate `"rand_blob"`/`"rand_spread"`); if absent, every state uses
  `dataset.init_pos` (e.g. `binned` always uses `"spread"`). **Absent and
  meaningless for `objbiased_coffee`** — see below.

## Coffee vs. carrots (`objbiased_coffee`)

`objbiased_coffee` swaps `dataset.obj` from `"carrots"` to `"coffee"` but
otherwise reuses every size/sampling config from `objbiased` verbatim (same
`seed`, `n_states`, `n_actions`, `action_sampler`, bin edges) — the intent is a
same-size dataset for testing whether a model trained on one material
transfers to the other. Two things are genuinely different, not just
relabeled:

- **Particle representation.** Carrots are random-convex-mesh rigid clusters;
  coffee beans are a fixed `coffee_bean.ply` mesh, voxelized into particles at
  a configurable resolution (`dataset.particle_r`, repurposed for `obj:
  coffee` as the bean's packing radius — unrelated to its use for `obj:
  ball`). `objbiased_coffee` uses `particle_r: 0.105` and `dataset.num_objects:
  117` beans, calibrated so the pile totals ~4,248 particles, matching
  `objbiased`'s 4,221 carrot particles for comparable per-push simulation cost
  — but a coffee particle and a carrot particle do not correspond to the same
  physical granule size or count; don't assume per-particle correspondence
  across the two datasets, only per-pile / per-push aggregate comparisons.
- **Initial-state variety.** Carrots' `rand_blob`/`rand_spread` (`init_pos_mix`)
  relocate and reshape the *whole pile* per state. Coffee's underlying scene
  (`yx_Coffee`) has no such generator — it always builds the same fixed grid
  footprint at the same workspace location — so `objbiased_coffee` instead
  draws a per-bean random position jitter (half a bean's grid pitch, uniform
  per axis), seeded by the per-state seed the same way carrots' pile shape is.
  Every state therefore has a visually similar footprint/location with a
  genuinely different bean-by-bean layout, rather than carrots' varied
  footprints/locations. `slates.init_pos_mix` plays no role here (coffee
  ignores `dataset.init_pos` entirely).

## Quick loading sketch

```python
import json, numpy as np

records = [json.loads(l) for l in open(f'{root}/manifest.jsonl')]
actions = [r for r in records if r['type'] == 'action' and r['valid']]

for r in actions:
    before = np.load(f'{root}/{r["state_idx"]}/initial_particles.npy').reshape(-1, 4)
    after = np.load(f'{root}/{r["after_positions_path"]}').reshape(-1, 4)
    action = np.array(r['action'])  # [x0, z0, x1, z1]
    bin_id = r['bin']
```
