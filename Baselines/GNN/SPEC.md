# GNN baseline — design spec (constant node count)

**See the "CORRECTION (2026-09-10)" section at the end of this file
before reading anything below it as current.** The original design here
(and the implementation it described) fed ground-truth simulator cube
centroids directly to the graph nodes. That was a mistake, not a
deliberate simplification: node positions must be constructed from the
top-down occupancy raster (foreground-extraction + FPS), since a real
deployment only ever has a camera, never privileged 3D object pose. The
rest of this file (architecture, `s_delta`, hazards) is otherwise still
accurate and is kept as the original record; only §1's node-construction
description and the "N=20 fixed" framing below are superseded.

**Status: design document + a since-corrected implementation.** Written
for an implementer who has NOT read the paper. Every architecture number
is cited to `model/gnn_dyn.py` and to the vendored checkpoint's
`state_dict`, not guessed. Scope, per `Baselines/ORCHESTRATION_LOG.md`
(user decision, 2026-09-08): **no resolution regressor** -- the paper's
own §III-C Bayesian optimization over ω (choosing HOW MANY nodes to use,
`res_rgr`/`res_regressor.py`/`train_res_rgr.py`/`dataset_res_rgr.py`) is
out of scope and not discussed further below. This is a narrower
exclusion than the original version of this doc implied: node count
(`n_particles`) is a fixed, chosen hyperparameter (originally 20, matching
our cells' true cube count; since generalized to other values, e.g. 30,
decoupled from any cell's true particle count -- see the CORRECTION
section), but WHICH `n_particles` positions the graph gets is no longer
looked up from privileged state -- it is constructed via
foreground-extraction + Farthest Point Sampling from the occupancy raster,
exactly as the paper's own perception module does. Only the *adaptive
choice of ω itself* (the resolution regressor) remains out of scope.

---

## 1. What the model is

This is a **PropNet/DPI-Net-style interaction network** (Li et al., cited
directly in the paper's §III-B), reused here as-is
(`model/gnn_dyn.py::PropNetDiffDenModel` / `PropModuleDiffDen`).

- **Node** = one object particle. In the paper's own data this is a
  farthest-point-sampled subset of a segmented point cloud (variable count,
  ω-dependent). **In our scope, a node = one cube**, always N=20 — no
  sampling, no resolution choice.
- **Edge** = a directed sender→receiver relation between two nodes that are
  (a) within a distance threshold `adj_thresh` of each other **and** (b)
  among the receiver's `min(10, N-1)` nearest neighbours (a hard cap on
  in-degree). Both conditions must hold (AND, not OR) — see
  `model/gnn_dyn.py:229-241`. Critically, the distance used to decide edges
  is **not** the current position `s_cur` but the *anticipated* post-action
  position `s_cur + s_delta` (`model/gnn_dyn.py:224-225`) — the graph
  topology already "looks ahead" to where the action is about to push
  things.
- **Message passing**: 3 rounds (`pstep = 3`, hardcoded in
  `model/gnn_dyn.py:160`, **not** exposed in any config — this is a
  code constant, not a hyperparameter you can read out of a yaml). Each
  round: aggregate per-edge effects into each receiver, update edge
  ("relation") embeddings with a residual-free propagator, aggregate edge
  effects back into nodes, update node ("particle") embeddings with a
  *residual* propagator (`model/gnn_dyn.py:182-193`).
- **What it predicts**: a **per-node 3-D residual added to the current
  position**, i.e. `ŝ_{t+1} = particle_predictor(effect) + s_cur`
  (`model/gnn_dyn.py:196-198`) — **not** `s_cur + s_delta`, and **not**
  orientation. The model has no rotation head at all; it only ever
  predicts xyz. Frame: whatever frame `s_cur`/`s_delta` are expressed in
  when you call it (see §4 — for us this will be plain world-frame metres,
  not the paper's camera-normalized space).
- **Loss**: masked MSE between predicted and ground-truth next positions,
  summed over a multi-step rollout of length `n_rollout` and normalized by
  `n_rollout * batch_size` (`train/train_gnn_dyn.py:170-191`). The mask
  restricts the sum to the first `particle_num[j]` entries per sample (a
  variable-N artifact from their FPS sampling — irrelevant for us since
  N=20 always, see §5).

## 2. Exact architecture numbers (from code + checkpoint state_dict)

Checkpoint inspected:
`Baselines/GNN/data/gnn_dyn_model/2023-01-28-10-42-05-114323/net_epoch_0_iter_1000.pth`
(no `config.yaml` was saved alongside it — the config lives only in
`config/train/gnn_dyn.yaml`). All shapes below are confirmed by both
sources agreeing (`nf_effect = 64` throughout):

| module | shape (from checkpoint) | matches code | activation |
|---|---|---|---|
| `particle_encoder.model.0` | Linear(5 → 64) | `ParticleEncoder(3+1+1, 64, 64)`, `model/gnn_dyn.py:127-128` | ReLU |
| `particle_encoder.model.2` | Linear(64 → 64) | — | ReLU |
| `relation_encoder.model.0` | Linear(6 → 64) | `RelationEncoder(2+3+1, 64, 64)`, `:132-133` | ReLU |
| `relation_encoder.model.2` | Linear(64 → 64) | — | ReLU |
| `relation_encoder.model.4` | Linear(64 → 64) | — | ReLU |
| `particle_propagator.linear` | Linear(129 → 64) | `Propagator(2*64+1, 64)`, `:136-137` | ReLU (with residual add) |
| `relation_propagator.linear` | Linear(193 → 64) | `Propagator(64+2*64+1, 64)`, `:140-141` | ReLU (with residual add) |
| `particle_predictor.linear_0` | Linear(64 → 64) | `ParticlePredictor(64,64,3)`, `:144-145` | ReLU |
| `particle_predictor.linear_1` | Linear(64 → 3) | — | **none** (raw regression output) |

**Total trainable parameters: 38,403** (summed from the shapes above — this
is a tiny model; plan batch size and epoch budget accordingly, §7).

Input feature composition (why the numbers above are 5 / 6 / 129 / 193):
- particle encoder input (5) = `s_delta` (3) ⧺ `a_cur` (1, per-node scalar
  attribute) ⧺ `particle_dens` (1, broadcast scalar, divided by 5000 inside
  the model, `:158`). **Note: raw position `s_cur` is never fed to the
  encoder** — only the action-conditioning field `s_delta`.
- relation encoder input (6) = `a_cur` of receiver (1) ⧺ `a_cur` of sender
  (1) ⧺ `(s_cur_receiver − s_cur_sender)` (3) ⧺ `particle_dens` (1).
- particle propagator input (129) = particle encode (64) ⧺ aggregated edge
  effect (64) ⧺ density (1).
- relation propagator input (193) = relation encode (64) ⧺ sender effect
  (64) ⧺ receiver effect (64) ⧺ density (1).

`adj_thresh = 0.08` (config `train.particle.adj_thresh`,
`config/train/gnn_dyn.yaml:53`) is used as `threshold = adj_thresh**2` on
squared distance (`model/gnn_dyn.py:229-230`). **This number is tuned to
the paper's own coordinate scale and must not be reused verbatim on our
data — see hazard §5, item 2.**

## 3. How the pusher/action enters the graph

**There are no separate pusher nodes/particles in this codebase**, despite
what the class docstrings suggest ("indicating the type of the objects,
slider or pusher", `model/gnn_dyn.py:148`). Concretely:

- `a_cur` (the per-node type attribute) is **always identically zero**,
  both in the reference training data
  (`dataset/dataset_gnn_dyn.py:114`: `attrs = np.zeros(states.shape[:2])`,
  never written to afterward) and in the existing
  `simple_mpc/adapters.py::GNNAdapter.predict_step`
  (`simple_mpc/adapters.py:434-435`: `a_cur = torch.zeros(...)`). Treat
  `a_cur` as a permanently-zero placeholder input slot, not a real feature,
  for this checkpoint/recipe. (Do not "fix" this — it's how the vendored
  model is actually built and trained; changing it would be a model
  change, out of scope for a design doc that must not alter
  `model/gnn_dyn.py`.)
- The action instead enters as a **per-node 3-vector field `s_delta`**,
  computed *outside* the network, once per (state, action) pair, and fed
  into the particle encoder (§2) and into the graph-construction distance
  (§1). It is built like this
  (`dataset/dataset_gnn_dyn.py:120-194`, mirrored differentiably in
  `simple_mpc/adapters.py::_gen_s_delta`, lines 37-110):
  1. Compute the push direction `push_dir = (e−s)/|e−s|` and its 2-D
     orthogonal `push_ortho`, from the 2-D push start `s` and end `e`.
  2. For every node, project its position onto `push_dir` (`x_loc`) and
     `push_ortho` (`y_loc`), relative to `s`.
  3. **Longitudinal hard mask**: 1 if `0 < x_loc < |e−s|` (the node lies
     between the tool's start and end along the sweep line), else 0
     (`dataset/dataset_gnn_dyn.py:188`).
  4. **Lateral soft mask**: `exp(−max(0, |y_loc| − pusher_w)/0.01)` — ≈1
     inside the blade's swept lane (half-width `pusher_w`), decaying
     sharply (0.01-length softness) outside it (`:189-191`).
  5. `s_delta_i = (remaining distance from node i to e, projected onto
     push_dir) × push_dir × longitudinal_mask_i × lateral_mask_i` — i.e. a
     vector pointing toward the tool's end position, non-zero only for
     nodes currently inside the swept lane (`:192-194`).
- `s_delta` does double duty: it conditions the particle encoder (§2), and
  it is added to `s_cur` **only** to decide graph edges (§1) — the actual
  predicted output is a residual on top of the *un*-shifted `s_cur`
  (`particle_pred + s_cur`, not `+ (s_cur+s_delta)`). The network has to
  learn the real displacement; `s_delta` is a hint, not a shortcut.
- `particle_dens` is **not a measured density** — in the reference
  training loop it is drawn fresh, per `__getitem__` call, from
  `Uniform(15, 6500)` (`dataset/dataset_gnn_dyn.py:79-84`), independent of
  the sample's actual particle spacing. It exists so one set of weights
  generalizes across the resolutions the (now out-of-scope) resolution
  regressor might pick. See hazard §5, item 3, for what to do with it here.

## 4. Their data format vs ours

Reference sample data inspected:
`Baselines/GNN/data/gnn_dyn_data/0/{<t>_particles.npy, <t>_color.png,
<t>_depth.png, actions.p}` (11 timesteps, episode 0).

| their tensor | what it is | shape / dtype / units (measured) | our source | conversion |
|---|---|---|---|---|
| `<t>_particles.npy` | raw PyFlex per-frame particle positions, homogeneous (x,y,z,w) | `(N,4)` float32, `N≈78k` for the ball scene; **raw PyFlex world units**, not metres, not yet camera-transformed (col 3 gets overwritten to 1.0 by `read_particles`) | `states[...,:3]` / `states_[...,:3]` | our cubes are already the exact per-node states — no depth/color/FPS/KDTree pipeline needed at all (see §6) |
| `actions.p` | pickled `(n_timestep, 4)` float64 array of `[sx, sy, ex, ey]` push segments **in raw PyFlex world 2-D units** (range observed ≈ ±5, matches `wkspc_w: 5.0`) | measured min/max ≈ (−4.9, −4.4) / (6.0, 4.9) | `p_starts[:, :2]`, `p_stops[:, :2]` | ours are **world-frame metres**, not PyFlex units — see hazard §5, item 5 |
| `<t>_depth.png` / `<t>_color.png` | RGBD render, used only to re-derive/track particles | `(720,720)` uint16 / `(720,720,3)` uint8 | — | **not needed**: we already have exact per-cube state, not an image we must segment |
| `global_scale = 24`, `cam_extrinsic` | normalize raw PyFlex coords into "camera-normalized" space (`read_particles`, `opengl2cam`) | — | — | **not needed**: skip this transform chain entirely, we're already in world metres (see §5, item 5) |
| n/a | `angles` (our field, blade yaw) | `(128,)` float32, radians | `angles` | **not required as a model input.** Verified numerically (4-sample check): `angles == atan2(push_dir.y, push_dir.x) + π/2` exactly, i.e. it is the blade's *face* orientation (perpendicular to travel), not the travel heading. See hazard §5, item 1. |

Our sample cell inspected:
`Genesis/data/slates_multistep/n20_L20mm/_0_data.pt` — dict of
`states (128,20,7)`, `states_ (128,20,7)` (xyz + quaternion wxyz, metres,
world frame), `p_starts (128,3)`, `p_stops (128,3)` (metres, world frame,
z constant = table height, ≈0.0175 m in the sample checked), `angles
(128,)` (radians). Measured facts that matter for the recipe:

- All 128 rows share **the same** `states` (verified: `std across batch =
  4.8e-12`) — one file = one fixed starting configuration ("slate") ×
  128 different candidate single-step pushes. This is a **slate**, not a
  trajectory.
- Push length `|p_stops − p_starts|` measured at 0.020009 ± 0.000001 m for
  every row of `n20_L20mm` (matches the cell name), z-component of the
  push vector is ≤ 9.4e-7 m (fully planar in xy).
- Box workspace = 0.128 × 0.128 × 0.04 m
  (`Genesis/data/slates_multistep/n20_L20mm/_0_config.yaml: box.vol`),
  cube edge = 0.005 m (`data_collection.cube_size`), `n_cubes: 20`.
- Blade/plate geometry, from the same config's `plate.size:
  [0.04, 0.002, 0.01]` → **length 0.04 m (lateral, i.e. the swept-lane
  full width), thickness 0.002 m (along travel), height 0.01 m.** This
  gives a sourced, physically-real value for `pusher_w`: **0.02 m**
  (half of the 0.04 m lateral length) — use this instead of the paper's
  `0.8/global_scale` (§5, item 2).
- Per-cell dataset layout (from `scripts/probes/prepare_slate_multistep_split.py`
  docstring + `manifest.json`): **50 slates × 3 steps × 128 envs** per
  cell; `_<cell>_train`/`_<cell>_eval` symlink dirs already exist with a
  documented, seeded 30/20 slate-level split (30×3=90 train files, 20×3=60
  eval files per cell). Pooling `n20_L20mm` + `n20_L40mm` per
  `ORCHESTRATION_LOG.md` gives **90+90 = 180 train files × 128 = 23,040
  training transitions** — this is exactly the "~23k transitions" figure
  in this agent's brief; use the existing `_train`/`_eval` split, do not
  re-split.
- The "3 steps" per slate are **not** freely chainable across the 128
  per-step candidates: `manifest.json`'s `batches` entries carry
  `slate_idx`, `step_idx`, `state_library_index` but no record of *which*
  one of the 128 step-0 candidates was actually executed to produce
  step 1's starting state. **Multi-step rollout supervision (`n_rollout >
  1`) is not reliably reconstructable from this layout without further
  digging** — see Risks §8, item 1.

## 5. Hazards — unit, frame, and convention mismatches (ranked)

This repo has a history of frame/unit-convention bugs — an axis silently
swapped between world and pixel space, a coordinate scale reused from a
different simulator unexamined, a heading read from the wrong field. Treat
every hazard below as load-bearing, not cosmetic.

1. **Blade-yaw vs travel-heading confusion (confirmed, not hypothetical).**
   `angles` is the blade's face orientation, exactly `heading + 90°`
   (numerically verified across 4 samples, offset = π/2 in every case, to
   float precision). If a re-implementer feeds `angles` directly as the
   travel heading (the natural-seeming thing to do, since it's the only
   "direction" field visible), the entire `s_delta` swept-lane mask (§3)
   comes out rotated 90° from correct — the model would be trained on
   completely wrong action-conditioning inputs while silently producing a
   loss curve that looks fine (no crash, just wrong geometry). **Fix:**
   derive `push_dir` purely from `p_stops − p_starts`; never read `angles`
   for the mask geometry. `angles` is not needed as a model input at all
   under this recipe.

2. **`adj_thresh` numeric mismatch.** The paper's `0.08` is tuned in *their*
   normalized-camera space: raw PyFlex world half-width `wkspc_w=5.0`,
   divided by `global_scale=24` → effective half-width ≈0.2 in that space.
   Our box half-width is 0.064 m. The magnitudes are coincidentally
   similar order, but `0.08` is **larger than our entire box half-width**
   and close to the box's full diagonal-ish scale — combined with the
   hard `min(10, N-1)`-neighbour cap (§1), for N=20 this makes `adj_thresh`
   nearly a no-op: almost every node ends up connected to its 10 nearest
   of the other 19, regardless of the literal value. That may be an
   acceptable graph (effectively "always top-10-NN") but it should be a
   **deliberate** choice, not an accident inherited from a different
   simulator's units. **Recommendation:** pick `adj_thresh` in metres
   directly — start around 2-3× cube edge (0.010-0.015 m) for a
   locally-meaningful contact radius, and sanity-check the resulting edge
   count/degree distribution before trusting it; do not reuse `0.08`
   un-examined.

3. **`particle_dens` is domain-randomization noise, not a measurement**
   (§3). Since this is a from-scratch retrain (see item 4 below), there is
   no pretrained-checkpoint distribution to stay compatible with, so this
   is lower-stakes than it looks — but a re-implementer who tries to
   "compute" density from our cube spacing (e.g. `1/mean_NN_dist²`) would
   silently be inventing a physically-motivated-looking number that has no
   counterpart in how the reference model was actually trained.
   **Recommendation:** since our scope has no resolution axis at all,
   fix `particle_dens` to a single constant for both training and
   inference (e.g. 1000.0, arbitrary but consistent) rather than sampling
   or "measuring" it — simplest, and removes one source of train/inference
   mismatch.

4. **Do not warm-start from the vendored checkpoint.** The checkpoint
   (`data/gnn_dyn_model/2023-01-28-10-42-05-114323/net_epoch_0_iter_1000.pth`)
   was trained on a different simulator (PyFlex), different objects
   (granular "carrots"/capsules, not rigid cubes), a different coordinate
   scale (§ above), and a different, variable N. Train from random
   initialization on our data; reuse only the architecture
   (`model/gnn_dyn.py`), not the weights.

5. **World-frame/metres vs their camera-normalized/OpenGL pipeline.** Their
   entire dataset and adapter code (`read_particles`, `opengl2cam`,
   `_gen_s_delta`'s `cam_extrinsic`/`opencv_T_opengl` block) exists to map
   PyFlex's raw-world → camera-normalized coordinates. **None of that
   applies to us** — our `states`/`p_starts`/`p_stops` are already
   plain world-frame metres (x,y = table plane, z = height — confirmed:
   z spans only ≈0.0125-0.0175 m across the whole sample batch, i.e. a
   tight, near-constant height band consistent with resting/lightly-
   stacked 5 mm cubes, while x,y spans the wider workspace extent).
   **Do not port `read_particles`/`opengl2cam`/the `cam_extrinsic`
   machinery at all** — building it "to be safe" is the likeliest way to
   silently reintroduce a bogus rotation/axis-swap on data that was
   already correct. Work directly in world (x,y) for the push-mask
   arithmetic in §3, and carry `z` through unchanged (it is not part of
   `s_delta`'s support, since the push is planar — measured push
   z-component ≤ 9.4e-7 m).

6. **Orientation is dropped entirely, not modeled.** The model's output is
   xyz-only; our state additionally carries a quaternion. This is a real
   capability gap, not a bug — but it does mean this baseline can only ever
   be credited for position, not orientation: confirm that
   occupancy/position-only scoring is genuinely what `slateK_exact`/
   `accuracy` measure (both operate on rasterised occupancy grids, so this
   should already be the case, but this agent did not re-derive the
   metrics code itself end-to-end; flagged as an assumption, not verified
   — see Risks §8).

## 6. Conversion recipe: our `.pt` batch → `forward()` inputs

```python
# batch: states(B,20,7), states_(B,20,7), p_starts(B,3), p_stops(B,3)
# (angles is NOT used — see hazard #1)

s_cur  = batch['states'][...,  :3]     # (B, 20, 3) metres, world xyz — drop quaternion
s_next = batch['states_'][..., :3]     # (B, 20, 3) ground-truth target, same convention

p0 = batch['p_starts'][:, :2]          # (B, 2) world xy, metres
p1 = batch['p_stops'][:, :2]           # (B, 2) world xy, metres
d       = p1 - p0
push_l  = d.norm(dim=-1, keepdim=True)                 # (B, 1)
push_dir   = d / push_l                                 # (B, 2)
push_ortho = torch.stack([-push_dir[:,1], push_dir[:,0]], -1)  # (B, 2)

PUSHER_W = 0.02   # metres — half the blade's 0.04 m lateral length,
                  # sourced from `plate.size` in the cell's own _<i>_config.yaml
SOFTNESS = 0.01   # metres — matches dataset_gnn_dyn.py:189-191's constant verbatim
                  # (unitless there since their coords were pre-normalized;
                  # here it is a physical length, so re-check this is still a
                  # sensible softness at metre scale before trusting it)

pos_xy     = s_cur[..., :2]                                    # (B, 20, 2)
rel        = pos_xy - p0[:, None, :]
x_loc      = (rel * push_dir[:, None, :]).sum(-1)              # (B, 20) along-push
y_loc      = (rel * push_ortho[:, None, :]).sum(-1)            # (B, 20) lateral

l_mask = ((x_loc > 0) & (x_loc < push_l)).float()                       # hard
w_mask = torch.exp(-(y_loc.abs() - PUSHER_W).clamp(min=0) / SOFTNESS)   # soft

to_end       = p1[:, None, :] - pos_xy                          # (B, 20, 2)
dist_to_end  = (to_end * push_dir[:, None, :]).sum(-1)          # (B, 20)

s_delta_xy = dist_to_end[..., None] * push_dir[:, None, :] * l_mask[..., None] * w_mask[..., None]
s_delta    = torch.cat([s_delta_xy, torch.zeros_like(s_delta_xy[..., :1])], dim=-1)  # (B,20,3), z=0 (planar)

a_cur         = torch.zeros(B, 20)                    # matches reference exactly (§3)
particle_dens = torch.full((B,), 1000.0)              # fixed constant (§5 item 3)

# Adjacency: build on (s_cur + s_delta), exactly as
# PropNetDiffDenModel.predict_one_step does (model/gnn_dyn.py:209-254) —
# call that method directly rather than reimplementing the top-k/threshold
# logic; just pass adj_thresh calibrated per §5 item 2 via the model's config.

s_pred = model.predict_one_step(a_cur, s_cur, s_delta, particle_dens)  # (B, 20, 3)
loss   = F.mse_loss(s_pred, s_next)   # no masking needed — N=20 always, no padding
```

**Output → next-step cube positions:** `s_pred` already *is* the predicted
next xyz for all 20 cubes, in world-frame metres — no inverse transform
needed (we never left world-frame). If downstream code (rasterizer,
`slateK_exact`, etc.) needs a full 7-vector, reattach the *input* frame's
quaternion unchanged (the model does not predict orientation — see hazard
§5 item 6) rather than inventing one.

## 7. What must change in the training code

**Recommendation: write a new dataset class + new training script; do not
patch `train_gnn_dyn.py`/`dataset_gnn_dyn.py` in place.** The two are built
around a fundamentally different I/O shape (PyFlex-live-sim `FlexEnv`,
per-frame `.npy`/`.png` directories, pickled action sequences, KDTree
particle tracking, FPS-variable N) that has no counterpart in our batched
`.pt` slates. Patching would mean deleting most of the file's logic anyway
and touching a vendored file the instructions say not to modify.

- **New:** `Baselines/GNN/dataset/dataset_genesis_gnn.py` — a `Dataset`
  that globs `_*_data.pt` under one or more of the existing
  `..._train`/`..._eval` symlink directories (reuse the
  `prepare_slate_multistep_split.py` split as-is — do not re-split at file
  granularity, since a naive re-split could leak a slate's 3 steps across
  train/eval, which is exactly the leak that script exists to avoid),
  flattens each file's 128 rows into 128 single-step examples, and returns
  the five tensors used in §6 (`s_cur, s_next, p0, p1` — `angles` can be
  loaded and ignored). No color/depth images, no KDTree/FPS, no
  `particle_num`/padding logic — none of it applies when N is always 20.
- **New:** `Baselines/GNN/train/train_genesis_gnn_dyn.py` — the same
  Adam/StepLR training loop as `train_gnn_dyn.py`, minus the `FlexEnv`
  bootstrap (no camera params/extrinsics needed at all, §5 item 5), with
  `n_history=1, n_rollout=1` (single-step; see Risks §8 item 1 for why not
  more), and the simplified `collate_fn` implied by fixed N=20 (a plain
  `torch.stack`, no per-sample padding).
- **Unchanged:** `model/gnn_dyn.py` (`PropNetDiffDenModel`/
  `PropModuleDiffDen`) — reused exactly as vendored. Only the *config
  dict* passed to its constructor changes: `adj_thresh` recalibrated per
  §5 item 2, `nf_effect=64`/pstep=3 stay as-is (§2).

## 8. Training recipe

- **Optimizer:** Adam, `betas=(0.9, 0.999)`, matching
  `train/train_gnn_dyn.py:117-119` exactly (no reason to change this).
- **LR:** start at `1e-3` (paper's own default,
  `config/train/gnn_dyn.yaml:24`); a `StepLR` decay is optional at this
  scale — 23k single-step examples and a 38k-parameter model converge fast
  — but keep the schedule hook for parity if useful.
- **Batch size:** the original's `batch_size: 4` was constrained by
  `FlexEnv`'s live-simulation dataset generation cost, not by GPU memory —
  irrelevant here. Each of our files already holds 128 same-state
  examples; **batch size 128-256** is a natural, GPU-cheap choice given
  ≤10×20=200 edges per graph and a 38k-parameter model.
- **Epochs:** with 23,040 training transitions and batch 128 (~180
  steps/epoch), and a model this small, expect each epoch to run in
  well under a second of pure compute on an 8 GB RTX 4070 Laptop (data
  loading, not the model, will dominate wall-clock). **Recommend 300-500
  epochs** (~55-90k gradient steps) as a realistic overnight-scale budget,
  checkpointing best-validation-loss (mirroring `train_gnn_dyn.py`'s
  `net_best.pth` pattern) evaluated on the existing `_eval` slate split.
- Before committing to a full run: **visualize `s_delta` for a handful of
  samples** (heatmap of its magnitude over the 20 cube positions, or just
  print which cubes get non-zero mask) to confirm the swept-lane geometry
  actually lights up along the real push line before trusting the
  `PUSHER_W`/`adj_thresh`/`SOFTNESS` constants in §6 — this is the
  cheapest possible check against hazard §5 items 1-2 recurring in a new
  form.

## 9. Risks / open questions, ranked

1. **Multi-step (`n_rollout > 1`) supervision is not reconstructable from
   this data layout** without further work: `manifest.json` records
   `slate_idx`/`step_idx`/`state_library_index` but not which of a slate's
   128 step-0 candidates was actually executed to reach step 1's starting
   state (§4). This spec recommends `n_rollout=1` (single-step) training
   as the safe, data-matching choice; multi-step rollout training (which
   the paper's own recipe uses, `n_rollout=5`) is left as a genuinely open
   question for whoever wants to extend this — it would need either a new
   manifest field or a separate investigation of the collection script
   (`Genesis/same_state_slate_collection.py`) to recover which candidate
   was chosen.
2. **`adj_thresh`, `PUSHER_W`, and the mask softness constant are
   plausibility-argued from first principles (§5, §6), not empirically
   tuned or validated against a rollout.** They are my best sourced
   estimate (blade geometry read directly from a cell's own
   `_0_config.yaml`), but nobody has yet run the swept-lane visualization
   check recommended in §8. Do that before a full training run.
3. **`particle_dens` handling (fixed constant vs. the paper's random
   `Uniform(15,6500)`) is an untested design choice** (§5 item 3) — cheap
   to ablate either way since the training loop is new regardless; not
   worth blocking on, but flagged so it isn't silently forgotten.
4. **Orientation/quaternion is entirely unmodeled** (§5 item 6) — I did
   not re-derive `slateK_exact`/`accuracy`'s computation myself to confirm
   it only needs positions/occupancy; this is an assumption (every sibling
   baseline in this project is scored on rasterised occupancy, which is
   itself orientation-blind past a cube's rendered silhouette, so this
   should hold), not something this agent verified end-to-end.
5. **Honest confidence this trains successfully on our data: moderate-
   high.** The architecture is tiny (38k params), well-understood now
   (every shape traced to code + checkpoint), and the data pipeline is
   simpler than the reference one in every respect (no images, no FPS, no
   camera transforms, fixed N). The actual risk is concentrated entirely
   in getting the three geometric constants in §6 (`PUSHER_W`,
   `adj_thresh`, `SOFTNESS`) right for metre-scale coordinates rather than
   in anything architectural — which is exactly why §8 recommends a cheap
   visualization check before committing to a full run.

---

## ORCHESTRATOR AMENDMENT — 2026-09-08, after numerical verification

Two of this spec's ranked hazards were checked directly against the data. One
is confirmed and sharpened; one is **wrong and is retracted**.

### RETRACTED — "multi-step rollout is not reconstructable" (was hazard 3)

The spec claims the 50x3x128 layout does not record which of a slate's 128
candidate actions was executed to produce the next step. It does, implicitly,
and the rule is trivial: **env `i` at step `k+1` starts from env `i`'s own
outcome at step `k`, so the action executed by env `i` at step `k` is simply
row `i` of that step's batch.** Each of the 128 envs runs its own independent
rollout; the slate structure exists only at step 0, where all 128 envs share
one start state.

Measured on `n20_L20mm`, slate 0 (`_0/_1/_2_data.pt`):

```
step0 states_ vs step1 states   POSITION  max 1.4e-4 m,  mean 4e-7 m
step0 states_ vs step1 states   QUATERNION max 2.2e-2   (sign/settle only)
step0 state spread across the 128 envs   1.5e-11   <- same-state slate
step1 state spread across the 128 envs   9.5e-1    <- fully diverged
```

The 0.14 mm position gap is inter-step settling, not a different state.

**Consequence for the implementer:** the paper's own multi-step rollout recipe
IS available — chain `(states[i], action[i]) -> states_[i]` across the three
step files of a slate at fixed `i`. Single-step training (`n_rollout=1`) is
still the recommended *starting* point on time-budget grounds, but treat
multi-step as an available upgrade rather than as blocked. This is also exactly
why control utility is scored at **step 0 only** — steps 1-2 are diverged
rollouts, not same-state candidate slates.

### CONFIRMED AND SHARPENED — the `angles` convention (was hazard 1)

Verified exactly, sd 1.9e-7 over 128 candidates:

```
angles - atan2(p_stop - p_start)  ==  90 deg   (mod 180 deg)
```

So `angles` is the **blade face orientation**, perpendicular to the push
heading, as the spec says. Sharpening, because it matters:

- The relation holds **mod 180 deg**, not mod 360 — the raw difference is
  90 deg for some candidates and 270 deg for others. A blade is symmetric
  under a 180 deg rotation, so this is harmless for constructing blade
  geometry.
- But it means **the heading cannot be recovered from `angles`** — the sign is
  not there. **Always derive the push heading from `p_stop - p_start`, never
  from `angles`.** Use `angles` only where a face orientation is wanted.

This is the same class of frame-convention bug flagged at the top of this
section (§5) waiting to happen again; treat it as a hard rule, not a
preference.

---

## CORRECTION (2026-09-10) — node positions were privileged; fixed

**What was wrong.** Both this spec (§1, §4, §6) and the implementation it
described (the original `Baselines/GNN/dataset/dataset_genesis_gnn.py` and
`Baselines/GNN/predictor.py`) took the position "our cubes are already the
exact per-node states — no depth/color/FPS/KDTree pipeline needed at all"
(§4, as originally written) and read `states[...,:3]`/`states_[...,:3]`
directly out of the raw `_*_data.pt` files as graph node positions, both
for training and at inference. That is **privileged 3D simulator state** —
exact per-cube centroid and, via the reattached quaternion, exact
per-cube orientation. A real deployment has no such sensor; it has a
top-down camera. Every other baseline in this project (NFD, the
persistence/mean-delta/linear reference operators) already runs on the
occupancy raster alone — the GNN was the one baseline quietly cheating,
and this was flagged as a mistake, not defended as a reasonable
simplification, once raised.

**What changed.** `Baselines/GNN/perception.py` (new module) implements
the paper's own recipe (§1's RGB-D → foreground point cloud → FPS,
faithfully adapted to our raster instead of a real RGB-D frame — see that
module's docstring for the full reasoning and terminology, including why
"foreground extraction" here means reading already-pure-foreground
occupied pixels, not segmenting a cluttered scene):

1. **Node INPUT positions are raster-derived, never privileged.** The
   top-down occupancy grid (`occ0` — the SAME rasteriser
   (`Baselines/common/data.py::rasterize_particles`) that produces every
   ground-truth occ0/occ1 elsewhere in this repo) has its occupied pixels
   read out as a world-xy point cloud, Farthest-Point-Sampled down to a
   fixed `n_particles`, then locally recentred (mirrors
   `utils.py::recenter`/`dataset_gnn_dyn.py:101`). These xy values ARE the
   node positions fed to the network — nothing here reads `states`.
2. **Height and orientation are dropped, not estimated.** A top-down
   occupancy raster carries no depth/height channel and no per-object
   orientation once reduced to an anonymous point cloud, so every node
   gets a single fixed `z` (`perception.py::Z_CONST = 0.0`) and, when a
   predicted particle needs rendering back to occupancy for scoring, a
   fixed identity quaternion (`DEFAULT_QUAT`) — never a privileged
   per-cube value. This sharpens hazard §5 item 6 (orientation was already
   unmodeled by the network) to also cover *position* height and the
   predictor's former practice of reattaching the INPUT's real quaternion.
3. **Training labels still use privileged state — that's fine, correctly
   scoped.** To know how far a given raster-observed point should move
   (the supervised target), `dataset_genesis_gnn.py` tracks each
   FPS-sampled point to its nearest real cube (KDTree on ground-truth
   `states`, mirroring `dataset_gnn_dyn.py:108-109`'s own tracking step
   exactly) and applies THAT cube's true displacement to the
   raster-derived point. This is the standard sim-training pattern —
   privileged label, realistic input — not a re-introduction of the same
   bug: `predictor.py` never runs this tracking step at inference; only
   `dataset_genesis_gnn.py` (training) does.
4. **Node count is now decoupled from true particle count.** Because
   nodes are FPS-sampled from an occupancy grid rather than one-per-cube,
   `n_particles` is a free hyperparameter — the graph does not need to
   match a cell's true cube count. This is what let a single GNN pool
   training data across cells with different true particle counts (e.g.
   `Genesis/data/overnight_randlen`'s n20 and n50 groups) with one fixed
   node count (e.g. 30), something the original one-node-per-cube design
   could never have supported. See §7 below for the loader change this
   required.

**§7 update — dataset loader no longer goes through `CellData`.**
`Baselines.common.data.load_cell`/`CellData` preallocates a fixed
`(n, 20, 7)` tensor per config and therefore cannot represent a pool that
mixes true particle counts (confirmed: it crashes on `overnight_randlen`'s
n50 groups). `dataset_genesis_gnn.py::_load_rows` instead reads occupancy
+ each row's own per-file `states`/`states_`/`p_starts`/`p_stops` directly
off `registry.dataset_registry.build_dataset`'s raw dataset (reusing
`Baselines.common.data._resolve_sample`'s index arithmetic, not
re-deriving it) — safe because, per point 4 above, the OUTPUT is always
`n_particles`-shaped regardless of a row's true particle count. This does
not touch `Baselines/common/data.py` itself (still used as-is by
`predictor.py`/`eval_baseline.py` for occupancy and by every config that
only needs `states` for evaluation-side ground truth, e.g. fitting the
linear/mean-delta reference operators or computing a pile centroid — those
uses are legitimate, they never feed `states` to a model's input path).

**Performance note.** `sample_nodes_xy`/`track_displacement` are seeded
deterministically per row (`seed=<flat row index>`) and therefore
epoch-invariant — `GenesisGNNDataset` now runs this computation ONCE at
construction time and caches the result as plain tensors, rather than
recomputing it in every `__getitem__` call across every epoch. An early
smoke-training attempt under the pre-fix design (computing it lazily, per
epoch) was killed after exceeding a 10-minute wall-clock budget; moving it
to `__init__` reduced the pooled-corpus (~98k row) one-time cost to a few
minutes, paid once regardless of epoch count.

**What did NOT change.** `model/gnn_dyn.py` (`PropNetDiffDenModel`) is
still reused exactly as vendored — it has no N-dependent parameters, so a
node-count change is purely a runtime choice (now saved into the
checkpoint dict as `n_particles` and read back by `predictor.py`, so a
model trained with e.g. `--n-particles 30` is always scored with 30
nodes). `geometry.py::compute_s_delta` is unchanged and unaffected — it
only ever consumed whatever `s_cur_xyz` was handed to it, never assumed
where those positions came from.

**Config note.** Every `configs/dataset/genesis_*.yaml` this baseline (or
any other) reads now carries a header note: `states`/`states_` are
privileged and must not be fed to a model's input path. Two new pooled
configs were added for the n20+n50 case specifically:
`genesis_overnight_randlen_train_all.yaml` /
`genesis_overnight_randlen_test_all.yaml` (all 5 spawn-mode/particle-count
groups, consumed via `_load_rows`, NOT via `load_cell`/`CellData`).
