# Sand Manipulation — a continuum material in the same scene

**Status:** implemented on branch `VisualForesight`. First dataset collected and
fitted — **the linear operator works on sand** (§6), unlike on cubes.
**Purpose:** run the visual-foresight comparison on a *continuum* instead of
discrete cubes, without changing anything else about the scene.

The motivating result: on rigid cubes the paper's per-pixel linear operator never
beat "predict nothing moved", and the leading explanation
([`docs/linear_foresight_findings.md`](linear_foresight_findings.md) §4, H1) is
that its carrots formed a near-continuous mass while our 30–80 cubes are a
discrete set that moves in threshold events. Sand tests that directly. The cube
route to a continuum — more, smaller particles — was blocked by GPU memory and
by the fact that dropped cubes refuse to form anything but a monolayer; MPM sand
sidesteps both.

Everything is in **new files**. Nothing in the rigid path changed.

| File | Role |
|---|---|
| `Genesis/sand_manipulation.py` | `SandManipulation`, a subclass of `SandboxManipulation` that swaps only the material |
| `Genesis/configs/sand.yaml` | copy of `basic.yaml` plus `sand:` / `mpm_options:` blocks |
| `Genesis/sand_data_collection.py` | collection driver: episodes of N sequential pushes |
| `transforms/sand_occupancy.py` | continuum → grid projections (`sand_to_density`, `sand_to_heightmap`, `sand_mass`) |
| `tests/test_sand_occupancy.py` | 13 GPU-free tests for the projections |

---

## 1. What is identical, and why it is a subclass

The tray, the plate geometry and actuator model, the trapezoidal sweep,
`execute_action`, and the pile-aware action sampling are all inherited
**unmodified**. `SandManipulation` subclasses rather than reimplements precisely
so that stays true as the rigid path evolves — a copy would drift the moment
either side is tuned, and "the rest of the scene is identical" is the whole
premise of the comparison.

Two places use deliberate interception rather than duplication:

- **`_init_scene`** temporarily swaps `gs.Scene` for a capture object, records the
  kwargs the parent passes, then rebuilds the scene with `mpm_options` and a
  coupler added. The sand scene therefore inherits every rigid/sim/vis option the
  parent chooses, including options added later, with nothing copied.
- **`_add_entities`** stubs `random_sequential_addition` for the duration of the
  parent call, so the plane, tray walls and plate are built by the parent's own
  code, and only the granular block is replaced.

## 2. What necessarily differs

**Particle count is an output.** MPM samples the pile volume at
`mpm_options.particle_size`, so the count is read back after the entity is added.
The default 20 mm × 12 mm cylinder at 2 mm grains gives **1912 particles**.

**No orientations.** State keeps the parent's `(B, N, 7)` layout with identity
quaternions rather than introducing a `(B, N, 3)` one, so the dataset loader,
`variance_decomposition.py` (which reads `states[..., :3]`) and every other
consumer work unchanged. Sand grains have no meaningful orientation, so the
identity quat is honest padding.

**Reset is a pose write.** The as-sampled pile is captured once at build time and
restored, which is both valid and far cheaper than re-settling.

**Plate rides on the floor.** The cube path parks the blade half a cube above the
floor because that is where a cube's centre sits. Sand needs the blade
essentially on the floor — anything it rides over is material it does not push —
so `_operation_height` puts the blade *bottom* one grain-radius up (1.0 mm at
2 mm grains). Override with `plate.sweep_clearance`.

**Material properties are fixed at build.** Genesis compiles the constitutive
model when the entity is added, so `set_material_properties` cannot re-set
friction/density per batch the way a rigid body's can. It is accepted and ignored
(with a log line) so the shared collection loop still works; vary sand by
building a new scene from a different `sand:` block.

---

## 3. Three things that were silently wrong, and how they were found

Each of these produced a run that *completed successfully* with plausible-looking
data. They are recorded because the same traps apply to any future MPM work here.

### The settle never ran

The parent's rest test reads `rigid_solver.get_dofs_velocity` over per-grain dof
indices and **short-circuits to `(0.0, 0.0)` when there are none**. Sand has none,
so `_pile_is_at_rest()` was true on the first check and the settle loop exited
after a single step — every recorded `s'` would have been material still in
motion. Fixed by overriding `_pile_motion` to read `sand.get_particles_vel()`.

### Fixed rigid geometry does not couple to MPM

Measured, with both the default and the Legacy coupler: sand falls straight
through the tray floor and settles on the MPM domain boundary. The symptom was
grains at z = −8 mm with the tray floor at +10 mm. A *moving* rigid body **does**
couple — the plate displaced grains 13 mm over a 100 mm sweep — so the exclusion
appears specific to fixed geoms.

The fix is not to fight it: the MPM domain is sized so its own boundary, after the
solver's `3·dx` padding, coincides exactly with the tray's inner walls and floor.
Sand is contained by the boundary condition at the same physical location. The
rigid tray entities are still built unchanged — they are what the plate collides
with.

**Consequence to keep in mind:** the walls the sand feels are an MPM boundary
condition, not the frictional rigid walls the cubes feel. For a centred pile that
never reaches a wall this is immaterial; for wall-adjacent work it is a real
difference between the two setups.

### `grid_density` is cells per metre, and the timestep was 10× too large

Genesis' default `grid_density=64` means `dx = 15.6 mm` — coarser than the entire
pile and eight times the grain size. It is now derived from the grain size at the
standard ~2 particles per cell (`dx = 2 × particle_size`).

Separately, MPM needs a far smaller substep than the rigid solver. The rigid
path's 5 substeps left `substep_dt` ~10× over Genesis' CFL-style suggestion.
`simulation.substeps_mpm` (default 30) raises it. **Substeps are raised rather
than `dt` lowered on purpose:** `dt` sets the plate's control cadence, so lowering
it would change the sweep trajectory and the scene would no longer be identical.
`sand.E` was also dropped to 1e5, which lowers the wave speed — 1e6 needs ~6×
more substeps for no visible difference in how a pile pushes.

---

## 4. Occupancy: do **not** use `particles_to_occupancy`

The rigid path's projection scatter-adds a count and then **clamps to [0, 1]**,
producing a binary silhouette. For sand that discards the only thing a continuum
has: a 1-grain-deep smear and a 40-grain-deep dune become identical. That is the
same information loss that made the cube datasets uninformative about depth
(`linear_foresight_findings.md` §3.2).

`transforms/sand_occupancy.py` provides three **unclamped, mass-preserving**
functions:

| Function | What it gives |
|---|---|
| `sand_to_density` | top-down column mass per cell — the direct analogue of the paper's greyscale image, and the quantity a transport operator conserves |
| `sand_to_heightmap` | max grain height per cell — says how *tall* the material is, which column mass cannot separate from how *wide* |
| `sand_mass` | fraction of grains still inside the tray, the conservation check |

`normalize="mean"` (the default) scales so a typical occupied cell reads ~1.0,
making sand maps directly comparable to the cube silhouettes whose occupied cells
are 1.0 by construction. Out-of-bounds grains are **dropped, not clamped** —
clamping would pile escaped sand into a false ridge along the wall, exactly the
artefact a transport model would then learn.

`sigma` exposes the Gaussian smoothing that the linear-foresight work found to be
a *precondition* of the SE(2) warp, not a hyperparameter.

---

## 5. Collecting

```bash
python -m Genesis.sand_data_collection \
    --episodes 40 --pushes 5 --n-envs 8 --push-length 0.02 \
    --output-root data/sand/pile20
```

One episode = pile restored, then N sequential pile-aware pushes of a **single
fixed length**, so the dataset supports one switched-linear operator without
binning. Output is the same on-disk schema as every cube dataset.

Measured on the first run: **1912 grains, ~1.7 s/transition, mass conserved
exactly (1.0000 → 1.0000)**, grain displacement 13.8 mm mean / 23.2 mm max for a
20 mm push, pile resting at z = 10.0–17.5 mm on a 10 mm floor. That is roughly
**70× cheaper per transition than piled cubes** (~120 s), which makes sand the
practical route to a continuum-scale dataset.

### Known limitation: start-state diversity

MPM sampling happens once, so every episode restarts from the *same* pile.
Diversity within an episode comes from the 5 sequential pushes; across episodes,
only the actions differ. That is good for fitting an operator (many actions from
one state) and weak for state coverage. If broader coverage is needed, run a few
random warm-up pushes before recording, or jitter the pile centre — neither is
implemented yet.

---

## 6. First result: the operator works on sand

**On a continuum the paper's per-pixel linear operator beats persistence by a
wide margin, which it never did on cubes.** `sand_foresight.py`, 1465 full-length
transitions, validation split by *episode* (never by transition — five sequential
pushes on one pile are strongly correlated, so a transition split leaks):

| model | rms | **% of the change** | explained |
|---|---|---|---|
| **linear-nonneg** | 0.398 | **54.8%** | **0.452** |
| linear-ridge→I | 0.399 | 54.9% | 0.451 |
| heur-cumulative | 0.537 | 73.9% | 0.261 |
| identity (warp only) | 0.574 | 79.0% | 0.210 |
| persistence | 0.727 | 100.0% | 0.000 |

Swept region; "% of the change" is error relative to persistence, whose error
*is* the change that occurred, so 100% means no better than predicting nothing
moved. **The operator explains 45% of what happens.** On cubes the best
configuration ever managed was 96–99% — under 5% explained
(`linear_foresight_findings.md` §2.3).

Robust across four episode-level validation splits: **53.0 / 54.0 / 54.9 /
54.8%**. A ~2-point spread against a ~45-point effect, which is a different
situation entirely from the cube work, where a 0.004 fold standard deviation
swamped every effect being compared.

### The controls matter, and they hold

Two baselines separate "the model learned transport" from cheaper explanations:

- **`identity (warp only)`** is the SE(2) round trip with `A = I` — i.e. the
  blur the warp inflicts, with no model at all. At 79% it *does* beat
  persistence, because the post-push field is smoother than the pre-push one, so
  blurring alone helps. The operator's genuine contribution is therefore
  79% → 55%, not 100% → 55%. Without this control that gain would have been
  overstated.
- **`heur-cumulative`**, the hand-written transport heuristic, reaches 74%. The
  fitted operator beats it — the inversion seen on cubes, where the heuristic
  ranked better, does not occur here.

### Caveat

`M/D = 1.08` at 32×32: the fit is only barely overdetermined, so these numbers
should tighten (or not) with more data. A larger collection is under way.
## 7. Open questions

1. **Density or height as the model input?** They carry different information and
   neither dominates. Two channels is the obvious answer and is untested.
2. **Is the MPM-boundary tray a problem?** It is frictionless-ish where the rigid
   walls are frictional. Only matters once material reaches a wall.
3. **Is `substeps_mpm = 30` enough?** It is ~1.6× over Genesis' suggested bound
   and ran stably in probing, but that is not a proof. If a pile ever explodes,
   this is the first knob.
