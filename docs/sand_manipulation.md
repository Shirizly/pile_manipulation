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
| `transforms/sand_occupancy.py` | continuum → grid projections (`sand_to_density`, `sand_to_heightmap`, `sand_to_mask`, `mask_threshold_for_mass`, `sand_mass`) |
| `tests/test_sand_occupancy.py` | 19 GPU-free tests for the projections |
| `sand_foresight.py` / `sand_model_zoo.py` | fit + validate the operator, and the model-family comparison |

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

### Configuration sweep — and blur *hurts* here

| crop / res | M/D | blur 0 | blur 1 |
|---|---|---|---|
| 1.0 / 64 | 0.27 | **50.9%** | 54.5% |
| 0.5 / 32 | 1.08 | 51.0% | 54.8% |
| 0.375 / 24 | 1.91 | 54.4% | 56.8% |
| 0.25 / 16 | 4.30 | 63.6% | 63.9% |

Two things invert relative to the cube work. A **bigger** canonical window is
better, even at `M/D = 0.27` — capacity-limited, not overfitting. And **blur
hurts**: the σ≈1 smoothing that was a *precondition* on cubes only destroys
information here, because a sand density map is already smooth. That
precondition was specific to pixel-scale binary silhouettes, not to the method.

---

## 7. What the operator is actually contributing (`sand_model_zoo.py`)

Fitting several same-complexity families on the combined 4745-transition set
(`M/D = 3.48`) reframes the headline number:

| model | % of change | explained | params |
|---|---|---|---|
| linear-nonneg | **51.2%** | 0.488 | ~1M |
| reduced-rank r=16 | 51.4% | 0.486 | 33K |
| **reduced-rank r=4** | **52.2%** | 0.478 | **8K** |
| affine (`Ay+b`) | 52.4% | 0.476 | ~1M |
| col-stochastic | 55.9% | 0.441 | ~1M |
| knn retrieval (k=1) | 60.2% | 0.398 | — |
| **mean-delta** | **61.8%** | **0.382** | **0** |
| persistence | 100.0% | 0.000 | — |

**A zero-parameter constant explains 38% of the change.** `mean-delta` predicts
`y' = y + mean(Δ)` in the canonical frame — no state dependence at all. Since the
full operator explains 48.8%, *all* of that machinery is worth **~10 points** over
"apply the average displacement". Most of the headline result is the **canonical
frame** normalising the action away, not the operator learning transport. Any
future claim about this operator has to be stated against mean-delta, not against
persistence.

**The operator is effectively rank 4.** Rank 4 (8K parameters) scores 52.2%
against the full 51.2%. Sand transport in this frame has very low effective
dimensionality, which is also why more data barely helped: going from `M/D` 1.08
to 3.48 moved the linear model 51.0% → 51.2%. It was never data-limited.

**Mass conservation hurts** (55.9% vs 51.2%), despite sand conserving mass to
1.0000 in the *tray*. The constraint is imposed in the **canonical window**,
where material genuinely leaves the crop, so forcing column sums to 1 fights the
data at the boundary rather than encoding physics. This was the variant Suh &
Tedrake flagged as unsolvable with their per-row decomposition; it is solvable
(per-column simplex projection, column sums 1.0000 ± 0.0000) and it is not worth
solving.

## 8. The overhead-camera view (binary mask) — and it is BETTER

Everything above used `sand_to_density`, i.e. column mass, which assumes a
depth-sensing observation. A plain camera above or below the tray does not see
depth; it sees a silhouette. `sand_to_mask` thresholds the continuum back to
binary, which is both the realistic sensor model and the like-for-like
comparison with the cube datasets (binary throughout).

**Choosing the threshold from the data, not by taste.** On the collected piles
(64×64, 2 mm cells, mean 2.24 grains per occupied cell):

| threshold | cells kept | % of grid | **% of mass kept** |
|---|---|---|---|
| ≥1 grain | 855 | 20.9% | 100.0% |
| **≥2 grains** | **538** | **13.1%** | **83.4%** |
| ≥3 grains | 297 | 7.3% | 58.3% |

≥2 discards a sparse single-grain fringe holding a sixth of the material — the
"capture most of the mass, ignore very low density" operating point.
`mask_threshold_for_mass()` picks it automatically from a target mass fraction.

### Result: the silhouette is not a handicap, and with blur it is a large gain

| view | blur | linear | identity | heuristic |
|---|---|---|---|---|
| density | 0 | 51.2% | 74.6% | 72.6% |
| mask ≥1 | 0 | 53.5% | 89.9% | 89.5% |
| mask ≥2 | 0 | 51.4% | 83.7% | 91.7% |
| mask ≥3 | 0 | 49.2% | 77.4% | 94.2% |
| **mask ≥2** | **1** | **22.3%** | 98.6% | 92.1% |
| mask, h≥2 mm | 0 | 39.2% | 81.3% | 93.2% |

Two things follow.

**Depth information was not essential.** Unblurred, the binary mask matches the
density map (51.4% vs 51.2%). The operator was never relying on knowing how deep
the sand was — which is good news for using a real camera.

**Blur is a precondition for sharp fields, exactly as the cube work found.** On a
binary mask it takes the operator from 51.4% to **22.3%**; on the naturally
smooth density map the same blur *hurts*. The rule is not "sand likes blur" or
"cubes like blur" — it is that the SE(2) warp destroys pixel-scale features, so
any field with sharp edges needs smoothing first and any field already smooth
does not.

### Full model comparison on the camera view

| model | % of change | explained |
|---|---|---|
| affine (`Ay+b`) | 21.8% | 0.782 |
| **linear-nonneg** | **22.3%** | **0.777** |
| reduced-rank r=16 | 22.6% | 0.774 |
| reduced-rank r=4 | 25.3% | 0.747 |
| knn retrieval | 34.5% | 0.655 |
| **mean-delta (0 params)** | **50.0%** | **0.500** |
| col-stochastic | 50.3% | 0.497 |
| persistence | 100.0% | 0.000 |

**This is where the operator earns its keep.** Against the mean-delta baseline
that matters, its state-dependent contribution is **+0.277** here versus +0.106
on the density view — it roughly doubles what a zero-parameter constant achieves,
instead of adding a tenth. Effective rank rises too (~16 vs ~4 on density), i.e.
the operator is using more genuine structure.

Stable across five episode-level validation splits: **22.3 / 22.4 / 22.6 / 22.7 /
22.4%**.

Why the silhouette suits a *transport* operator better than density is worth
stating: a blurred binary mask is a smoothed indicator, and its motion under a
push is exactly what a linear transport map represents. A density map carries
extra degrees of freedom — how much mass sits in each cell — which the same
operator must also predict.

Mass conservation fails again for the same reason as before (50.3%): the
constraint is imposed in the canonical window, where material legitimately leaves
the crop.

### The cube/sand comparison, on identical code

Running the same `mean-delta` and linear fits on the cube dataset:

| | mean-delta (0 params) | linear operator | linear's marginal gain |
|---|---|---|---|
| **cubes** (scattered+blind) | +0.013 | +0.010 | **−0.003** |
| **sand** | **+0.382** | **+0.488** | **+0.106** |

This is H1 confirmed in a sharper form than anticipated. A continuum responds to
a push in a **stereotyped, repeatable** way, so even a constant explains 38%; with
discrete cubes, whether a given cube is caught, tumbles or is missed is a
threshold event, so the average response carries almost nothing (1.3%). And *on
top of that*, sand alone has learnable state-dependence — the linear operator
adds 10.6 points on sand and nothing at all on cubes.

Both components are properties of the material, not of the method: the code,
canonical frame, solver and metric are identical across the two rows.

### But two explanations fit that table equally well

The cube row is **scattered and blind**, the sand row is a **centred pile**, so
the comparison confounds two things:

* **granularity** — a few large cubes move as individuals, and whether one is
  caught, tumbles or is missed is a threshold event, so the average response
  carries little; a continuum averages into something a linear map can capture.
* **pile depth** — the cube datasets were monolayers and sand was a heap, so the
  difference may be in what the field IS rather than what it is made of.

  Though "sand was a heap" holds only at the *start* of an episode. Measured
  over the five pushes of an episode on the pile20 set (5th-95th percentile
  extent, 98th percentile height above the floor):

  | after push | extent | height | grain layers |
  |---|---|---|---|
  | 1 | 53.1 mm | 4.7 mm | ~2.4 |
  | 2 | 61.4 mm | 3.7 mm | ~1.9 |
  | 3 | 67.6 mm | 3.1 mm | ~1.6 |
  | 4 | 73.1 mm | 2.7 mm | ~1.4 |
  | 5 | 77.2 mm | 2.4 mm | **~1.2** |

  The sand pile spreads to 77 mm and its height halves, ending at little over
  one grain layer. So four fifths of the sand transitions are on a thin spread
  sheet rather than a heap, which makes the depth explanation weaker than it
  first looks — and means any cube spreading over an episode is a *matched*
  behaviour rather than a defect, as long as it is measured rather than
  assumed.

`Genesis/cube_spectrum_collection.py` is built to separate them: the same amount
of data (5200 transitions, matching the first sand set) and the same action
sampling at **n = 20, 50, 80** small cubes, all *piled*, with sand as the
continuum limit at the far end. If the linear operator's margin over mean-delta
grows monotonically as n rises, granularity is the axis; if all three cube counts
sit together near zero and only sand differs, depth or continuum-ness is.

Two things had to be right for that to be a fair comparison. The piles are
irregular two-layer heaps rather than monolayers (`--spawn-mode heap`; see
`docs/piled_collection.md` §1.6 for the eight setups that established a cube
pyramid will not collapse on its own), and every row runs through the **same**
loader and projection as sand — `sand_model_zoo.py --cube-size` rasterises each
cube's real footprint. The cube numbers above came from the dataset registry and
the sand numbers from `sand_foresight.py`; running a spectrum through two
projection paths would put a code difference inside the comparison.

Analysis: `scripts/cube_spectrum_analysis.sh`, summarised across n by
`scripts/cube_spectrum_summary.py`.

## 7. Open questions

1. **Density or height as the model input?** They carry different information and
   neither dominates. Two channels is the obvious answer and is untested.
2. **Is the MPM-boundary tray a problem?** It is frictionless-ish where the rigid
   walls are frictional. Only matters once material reaches a wall.
3. **Does the operator's margin scale with particle count?** The spectrum
   collection above exists to answer this; unrun as of this writing.
4. **Is `substeps_mpm = 30` enough?** It is ~1.6× over Genesis' suggested bound
   and ran stably in probing, but that is not a proof. If a pile ever explodes,
   this is the first knob.
