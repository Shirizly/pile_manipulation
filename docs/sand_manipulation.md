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
one state) and weak for state coverage.

**Resolved — see §9.** `Genesis/sand_state_library.py` builds varied starts from
states already collected mid-episode, expanded through the tray's D4 symmetry
group. That took the episode-start pile centroid spread from 0.00 mm to
15.77 mm, and it materially changed the rank result (§9.3), so the numbers in
§6-8 should be read as the single-pile case.

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

### RESOLVED — but NOT by depth: the old cube row was a rasteriser artifact

The cube-count spectrum was collected to separate these two explanations. It
did, and the answer is neither: **the margin barely depends on regime at all.**

Run through ONE code path (`sand_to_mask`), every regime lands in the same band:

| regime | M | mean-delta | linear | margin |
|---|---|---|---|---|
| cubes n50, **scattered monolayer** | 2560 | 0.120 | 0.411 | **+0.291** |
| cubes n50, scattered, contact-sampled | 2108 | 0.182 | 0.437 | +0.255 |
| cubes n20, piled 2 layers | 5120 | 0.345 | 0.645 | **+0.300** |
| cubes n30, piled 2 layers | 2560 | 0.348 | 0.643 | **+0.295** |
| sand, size-matched | 5120 | 0.328 | 0.591 | +0.263 |

A **scattered monolayer** gives +0.291, statistically the same as a two-layer
heap's +0.300. Depth is not the operative variable.

**The -0.003 in the table above is not a measurement of cubes.** It came through
the `PileSweepData` raster, whose occupancy and plate channels are mutually
transposed — worth more than 40 points of "% of change" (EXP-0001). Comparing a
new `sand_to_mask` number against it attributes a *code-path difference* to
physics. §8 of this document warned about exactly that and then did it; this
section did it a second time before EXP-0002 caught it. See
`docs/experiments/REGISTER.md` C-019.

What DOES vary with regime is **mean-delta** (0.12 -> 0.35) and total
predictability (0.31 -> 0.64). Depth and continuum-ness make the response more
*stereotyped* — easier for a constant to predict — not more *linear*. The
operator's marginal contribution over that constant is roughly flat.

Evidence: EXP-0002 (five regimes, one path), EXP-0006 (n=20/n=30 at 2-5x the
transitions, size-matched sand).

### The two explanations this replaced

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


## 9. The varied-start dataset (48 000 transitions)

Everything above was fitted on 5 200 transitions whose episodes all began from
the *identical* as-sampled pile (§5, "Known limitation"). This dataset removes
that, at 10x the size, so the fits can be overconstrained and the rank results
separated from the data that produced them.

**Build.** No new physics. States collected mid-episode are already valid
settled piles -- asymmetric, off-centre, variously spread -- so
`Genesis/sand_state_library.py` draws from the existing dataset skipping each
file's first push (those are all the same untouched pile), assigns each draw one
of the tray's 8 D4 symmetries, and jitters grains 0.3 mm. The symmetry expansion
is exact rather than approximate: a rotated or mirrored settled pile in a square
tray is still a settled pile.

300 episodes x 5 pushes x 32 envs = **48 000 transitions** in 12 654 s (3.5 h),
zero failed episodes.

### 9.1 It is sane, and it is diverse

| check | value |
|---|---|
| mass in tray | 1.0000 -> 1.0000 (min 1.0000) |
| z range | 10.0 .. 16.8 mm (floor 10.0) |
| grain displacement | mean 5.36, p95 18.55, max 49.13 mm |
| push length | mean 18.15 sd 5.63 mm, 89.1% full-length |
| **episode-start centroid spread** | **15.77 mm** (single-pile: 0.00) |
| episode-start pile radius | 28.6 +- 2.9 mm |

The centroid spread is the whole point: starts differ in *where* the pile is,
and the radius spread says they differ in how spread out it is too.

### 9.2 Model comparison, both views

Explained variance on the swept region, 10 754 validation transitions,
episode-level split:

| model | density | mask (blur 1) |
|---|---|---|
| affine (Ay+b) | **0.3942** | **0.6022** |
| linear-nonneg | 0.3914 | 0.5890 |
| reduced-rank r=64 | 0.3862 | 0.5900 |
| reduced-rank r=16 | 0.3838 | 0.5836 |
| reduced-rank r=4 | 0.3761 | 0.5352 |
| col-stochastic | 0.3767 | 0.4535 |
| mean-delta (0 params) | 0.3051 | 0.3285 |
| knn (k=1) | 0.2864 | 0.3856 |

Three results from §6-8 replicate: the mask view beats density, `affine` edges
out plain linear (a free bias is worth a little), and **column-stochastic hurts**
-- mass conservation imposed in a window material legitimately leaves is still
the wrong constraint.

**The margin over mean-delta holds.**

| | single-pile | varied |
|---|---|---|
| density | +0.106 | +0.086 |
| mask | +0.277 | +0.260 |

Mean-delta itself degraded exactly as predicted (mask 0.500 -> 0.329): a single
stereotyped displacement cannot serve piles in different places. The operator
degraded too, and the *margin* barely moved -- so the state-dependence result is
a property of granular transport, not an artefact of the stereotyped start.

### 9.3 The rank result was partly about the data, after all

| | single-pile | varied |
|---|---|---|
| density input dims (90% var) | 312 | **573** |
| mask input dims (90% var) | 14 | **25** |
| mask rank knee | ~4 | **~16** |

Start diversity roughly doubled the state dimensionality on both views, and the
required rank rose with it: rank-4 matched the full operator on single-pile data
and now falls 9% short, while rank-16 recovers 99%. So the operator's rank
**tracks the dimensionality of its input space** rather than being a fixed
property of granular transport. The earlier caveat was right, and the surviving
claim is the weaker one: 16 modes of a possible 1024, against a 573-dimensional
input space.

### 9.4 Why the blurred mask wins, in one line

The blurred mask spans **25 dimensions** where density spans **573**, and yet
predicts far better (0.589 vs 0.391). Blurring removes exactly the
high-frequency detail that is unpredictable, leaving a low-dimensional manifold
a small operator captures well. That is the mechanism behind the "sharp fields
need smoothing" rule of §6.

The `identity (warp only)` control makes the gap wider than the error figures
suggest. Passing the state through the same warp with `A = I`:

| view | identity | linear | so the OPERATOR contributes |
|---|---|---|---|
| density | 72.6% | 60.9% | ~12 of 39 points |
| mask | 97.4% | 41.1% | ~56 of 59 points |

On density most of the apparent gain is resampling, not learning. On the mask
view the warp is nearly free, so almost all of it is learned. Without this
control the density view would look far better than it is.

### 9.5 The 10x data bought a bigger operator

| view | crop | res | M/D | linear |
|---|---|---|---|---|
| density | 1.0 | 64 | 7.81 | **60.7%** |
| density | 0.5 | 32 | 31.25 | 60.9% |
| density | 0.25 | 16 | 125.00 | 71.1% |
| mask | 1.0 | 64 | 7.81 | **39.4%** |
| mask | 0.5 | 32 | 31.25 | 41.1% |
| mask | 0.25 | 16 | 125.00 | 68.1% |

The full-image 64x64 operator is now the best configuration on both views. On
5 200 transitions it was hopelessly underdetermined; at 48 000 it is fittable and
slightly better, which is precisely what the larger collection was for. Tight
crops are much worse -- a push moves mass ~12 px, a large fraction of the
window, so cropping in throws away the material that moved.

Reproduce all of the above with
`bash scripts/sand_full_analysis.sh 'Genesis/data/sand/varied/**/*_data.pt' varied`.

## 9.7 SUPERSEDING CAVEAT: the sand was resting on a frictionless floor

Everything above describes a material that could not hold an angle of repose,
because its base had no friction. Genesis'
`engine/boundaries/boundaries.py::CubeBoundary.impose_pos_vel` reflects only the
NORMAL velocity component (`vel[i] *= -restitution`, restitution 0) and leaves
both tangential components untouched -- a perfect free-slip plane. Section 3
had deliberately placed the MPM domain boundary at the tray floor, so the sand
rested on that plane.

A frictional medium on a frictionless base cannot hold a slope: its bottom layer
has no shear resistance, so the heap spreads until lateral stress vanishes.
Measured, on a column too tall to stand (EXP-0008):

| | as shipped (coup_friction 0.1) | fixed (0.8) |
|---|---|---|
| angle of repose | 9.5 deg | **29.0 deg** (real dry sand 30-35) |
| settle to rest | **never** | 25 steps |
| pile extent after 5 pushes | 59.9-82.0 mm | 40.5-50.5 mm |
| pile top | 13.0-14.5 mm | 20.3-22.1 mm |

Stiffness was the wrong suspect and points the *wrong way*: raising E from 1e5
to 3e6 pancaked the pile from 5.8 mm to 0.3 mm tall, because a stiffer continuum
transmits its weight to a frictionless floor more efficiently. 0.9 mm is exactly
the pile's volume spread over the whole tray floor.

**Fix:** `box.coup_friction: 0.8` (and `plate.coup_friction: 0.3`) in
`configs/sand.yaml`. `coup_friction` is friction against a *continuum* and is a
different parameter from `friction`, which is rigid-rigid; Genesis defaults it
to 0.1. It saturates above ~0.4.

**Also learned:** fixed rigid geoms DO couple to MPM in Genesis 1.3.3. Section
3's claim that they do not -- the reason the MPM domain was made to stand in for
the tray -- is stale. Dropping the domain floor 30 mm below the tray floor left
the sand resting at 9.0 mm, i.e. caught by the rigid floor, not by the boundary
20 mm lower. The domain-as-container arrangement is no longer necessary, though
it is harmless once `coup_friction` is set.

**Consequence for every sand number in this document:** they describe a
spreading puddle, not sand. This is not a data-quality caveat that a re-run
would tighten -- it is a different material, so results before and after today
are not comparable. Re-collection is required before the sand rows can be set
beside anything else.

## 9.6 CAVEAT on every number above: the settle was capped at 100 steps

`configs/sand.yaml` carried **two top-level `simulation:` blocks**. YAML keeps
the last, so the whole first block was silently discarded and `settle_steps` ran
at the code default of **100** rather than the declared 2500. Every sand dataset
on disk was collected that way (EXP-0007, fixed 2026-09-04, guarded by
`tests/test_config_no_duplicate_keys.py`).

Consequence, measured: after a push the q=0.995 grain speed is 1.905 mm/s at
step 100 against 0.811 at step 3000, and the pile moves a further **1.31 mm** of
mean grain displacement after step 100 — about **11%** of the 10-15 mm a push
moves. Each transition's `s` is the previous `s'`, so it accumulates along an
episode. Mass conservation and floor containment were unaffected (1.0000 and
exactly 10.0 mm throughout).

The sections above therefore stand *as measured* but describe a pile recorded
slightly early. Whether that moves the fitted margins is untested; re-collecting
the 48 000-transition set with the fix is 3.5 h and would settle it.

Two things the same investigation established, which outlive the bug:

- **Sand creeps and never stops.** With the cap fixed, post-*push* settles do
  converge (0.98-1.00 mm/s), but post-*spawn* settles never pass the criterion
  at all (2.12-2.22 mm/s over three episodes). The median grain is at rest
  (0.046 mm/s) while the top 0.5% keeps moving and drift/step flattens at
  ~0.4 um/step rather than reaching zero. `q=0.995 < 1 mm/s` was inherited from
  the rigid path and tests the moving tail, not the pile.
- **Late-episode sand pushes barely do anything.** Mean displacement per push
  fell to 0.25-1.42 mm once pile extent reached 63-82 mm, against 8-15 mm early
  in an episode. With the separately measured flattening to ~1.2 grain layers by
  push 5, a 5-push episode is roughly 2-3 informative transitions plus 2-3
  near-no-ops.

Videos: `outputs/sand_physicality/` (oblique | overhead, 3 episodes x 5 sweeps).

## 10. Open questions

1. **Density or height as the model input?** They carry different information and
   neither dominates. Two channels is the obvious answer and is untested.
2. **Is the MPM-boundary tray a problem?** It is frictionless-ish where the rigid
   walls are frictional. Only matters once material reaches a wall.
3. **Does the operator's margin scale with particle count?** The spectrum
   collection above exists to answer this; unrun as of this writing.
4. **Is `substeps_mpm = 30` enough?** It is ~1.6× over Genesis' suggested bound
   and ran stably in probing, but that is not a proof. If a pile ever explodes,
   this is the first knob.
