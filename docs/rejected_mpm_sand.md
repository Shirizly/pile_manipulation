# Rejected: MPM sand as a granular medium

**Status: abandoned 2026-09-05. Do not re-attempt with MPM.** The code was
removed in the same commit that added this note; recover it from
`e4fa9947` (`Genesis/sand_manipulation.py` and friends) if you need the
measurements rather than the conclusion.

## Why it was tried

The visual-foresight programme needed a continuum end of the granularity
spectrum, to answer whether the switched-linear operator works better on sand
than on rigid cubes. Genesis ships `gs.materials.MPM.Sand`, so the cheapest
route was to swap the material and keep the tray, plate, sweep and action
sampling identical.

## Why it was dropped

**It does not look like sand, and the reason is structural rather than a
mis-set parameter.** Every specific defect found was fixable, and fixing all of
them still left a medium that behaves like a soft deformable body rather than a
collection of grains.

The parameters were audited end to end and the material model is *correct*:

- **Cohesion is exactly zero, by construction.** Genesis implements the Klár et
  al. Drucker-Prager return mapping; its tensile branch (`tr >= 0`) leaves `S`
  at the identity, releasing all elastic deformation. There is no cohesion
  parameter because there is no cohesion term.
- **Measured adhesion is zero.** Lowering the blade into a settled pile and
  lifting it straight out (a zero-length push) raised **0.0%** of grains by even
  one grain diameter; mean `dz` was −0.07 mm.

So the "sand sticks to the blade" impression was not adhesion. It was two
separate numerical artifacts, both diagnosed and both fixed — and the result
still was not sand:

| defect | cause | fix | outcome |
|---|---|---|---|
| pile never settled; recorded mid-motion | duplicate `simulation:` key in `configs/sand.yaml` reverted `settle_steps` 2500 → 100 | de-duplicate; guarded by `tests/test_config_no_duplicate_keys.py` | settles, but see below |
| pile spread to a 0.2–9° puddle and never stopped creeping | Genesis' `CubeBoundary.impose_pos_vel` reflects only the **normal** velocity and leaves tangential untouched — a perfectly free-slip floor | `box.coup_friction` 0.1 → 0.8 | 29° angle of repose, settles in 25 steps |
| material towed behind the blade | MPM grid transfer is blind to a thin body: particles on opposite sides share grid nodes, so momentum crosses it | `MPMOptions.enable_CPIC=True` (Genesis default is False) | near-wake drag 3.5% → 1.1% of the push |

**What remained is not fixable within MPM.** The pile is **three grid cells
tall** — 1885 particles at `particle_size` 2 mm, `dx` 4 mm, on a 12 mm heap. At
that resolution a continuum cannot express grain-scale discreteness, and
refining it does not change the kind of object being simulated:

| particle_size | particles | pile height in cells | relative cost |
|---|---|---|---|
| 2 mm (as used) | 1 885 | 3.0 | 1× |
| 1 mm | 15 080 | 6.0 | 8× |
| 0.5 mm | 120 637 | 12.0 | 64× |

Real dry sand grains are 0.1–1 mm, so even 0.5 mm "grains" are coarse — and MPM
particles are quadrature points of a continuum, not grains. More of them make
the surface smoother, not more granular. Stiffness does not help either, and
goes the wrong way: raising `E` from 1e5 to 3e6 **pancaked** the pile from
5.8 mm to 0.3 mm tall, because a stiffer continuum delivers its weight to the
floor more efficiently.

## If a granular medium is needed again

Use **DEM — many small rigid bodies** — not MPM. In this repo that is the
existing cube path (`Genesis/cube_spectrum_collection.py`, `--spawn-mode heap`),
which simulates actual rigid grains with real contact. The cost is known and
steep: Newton factorizes a dense Hessian per contact island at
`island_size^2.64`, so a *connected* pile runs 0.36 s/transition at n=20 and
9.93 at n=30 (`docs/scaling_to_200_objects.md`, EXP-0006). That is the honest
price of grain-level contact, and it buys physics MPM cannot represent at any
resolution.

## What survived

The spectrum result does not depend on the sand row. Run through one code path,
the linear operator's margin over mean-delta is flat across regimes — scattered
monolayer cubes +0.291, piled n=20 +0.300, piled n=30 +0.295 — so the original
question ("does the operator do more state-dependent work on a continuum?") is
answered by the cube counts alone (EXP-0002, EXP-0006, REGISTER C-019).

Every sand number ever recorded was collected before the floor-friction fix, so
it describes a frictionless-based spreading puddle. Those rows are void, not
merely caveated (REGISTER C-029), and nothing downstream should cite them.

## Evidence

`archive/2026-09-10_pre-reset/docs/experiments/EXP-0007` (settle cap, and the ~1.3 mm bias it caused; archived in the 2026-09-10 register reset — see `experiments/README.md`),
`EXP-0008` (frictionless floor, angle of repose, CPIC), and REGISTER claims
C-023 through C-029. The probes that produced those numbers are in commit
`e4fa9947`.
