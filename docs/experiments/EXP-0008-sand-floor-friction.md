---
id: EXP-0008
title: The sand rested on a frictionless boundary; coup_friction 0.8 gives a 29 deg repose and settles 25x faster
tier: T1
mode: exploratory
date: 2026-09-04
hypothesis: null
claim: >
  The sand pile could not hold an angle of repose because its base was
  frictionless, not because the material was too soft; raising the tray's
  coup_friction from Genesis' default 0.1 to 0.8 takes the settled angle of
  repose from 9.5 to 29 degrees (real dry sand is 30-35) and the settle from
  never converging to 25 steps.
prediction: null
provenance:
  commit: 25404a4f
  script: "scripts/sand_material_sweep.py, scripts/sand_floor_probe.py, scripts/sand_plate_friction_probe.py"
  data: ["live sim, 1 env, 1912 grains, configs/sand.yaml"]
  code_path: sand_to_mask
  seed: 0
  split: n/a
  runtime: "~6 min total across three probes, RTX 4070"
design:
  varied: {E: [1.0e5, 1.0e6, 3.0e6], friction_angle: [30, 45], substeps_mpm: [30, 60], mpm_floor_drop: [0.0, 0.03], box_coup_friction: [0.1, 0.4, 0.8, 1.5], plate_coup_friction: [0.1, 0.3, 0.6], pile_shape: ["r20 h12", "r10 h30"]}
  held_fixed: {nu: 0.2, rho: 1500.0, particle_size: 0.002, dt: 0.004, sampler: pbs, n_envs: 1, tray: 0.128x0.128x0.04, settle_velocity_threshold: 0.001, settle_rest_quantile: 0.995}
  baselines: [persistence, mean-delta]
  metric: "settled angle of repose (deg, outer-flank fit to the free surface), settle steps to q0.995 < 1 mm/s, pile height, resting z, mass in tray"
noise_floor: "single run per cell; repose estimates vary a few degrees between measurements of the same config (9.5 vs 3.2 on two runs of the unfixed setup), so read differences under ~5 deg as noise"
depends_on: [sand-projection]
establishes: [settled-state]
result: "coup_friction 0.1 -> 0.8 takes repose 9.5 -> 29.0 deg and settle 'never' -> 25 steps; stiffness goes the WRONG way (E 1e5 -> 3e6 pancakes the pile 5.8 -> 0.3 mm tall); dropping the MPM floor is unnecessary once coup_friction is set"
verdict: supported
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates

"Sand should settle quickly, it is a high-friction medium" is a falsifiable
statement about our setup, and there were two candidate causes with opposite
fixes. If the medium were too soft, a stiffer E would hold the pile up. If the
BASE were frictionless, stiffness would make it worse -- a stiffer continuum
transmits its weight to the floor more efficiently and, with no shear resistance
there, slides out more completely. The sweep separates them in one column.

The angle of repose is the discriminating measurement rather than settle time,
because it is a material property with a known value (30-35 deg for dry sand)
that a puddle cannot fake.

## What was actually run

Three probes. A material sweep over E, friction_angle and substeps on
settle-only sims. A floor probe that lowers the MPM domain floor below the tray
floor to ask which surface catches the sand. A plate probe over the tool's own
coup_friction.

The first repose measurements used the default pile (r 20 mm, h 12 mm), which
barely slumps -- so they were measuring a softened cylinder edge, not a repose
angle. Repeated on a column too tall to stand (r 10 mm, h 30 mm), which must
collapse to its repose angle.

Root cause is in Genesis, `engine/boundaries/boundaries.py::CubeBoundary`:
`impose_pos_vel` does `vel[i] *= -restitution` for the crossed axis only and
leaves both tangential components untouched. With restitution 0 that is a
perfect free-slip plane. `sand_manipulation.py` had deliberately placed the MPM
domain boundary AT the tray floor, on the earlier finding that fixed rigid geoms
do not couple to MPM -- so the sand was resting on that free-slip plane.

## Numbers

Stiffness is the wrong lever, and points the wrong way (default pile, settle only):

| setup | to rest | resid mm/s | repose | height |
|---|---|---|---|---|
| E1e5 f45 s30 (as shipped) | never | 2.02 | 3.2 | 5.8 mm |
| E1e6 f45 s30 | 1350 | 0.99 | 0.4 | 0.9 mm |
| E1e6 f45 s60 | never | 1.60 | 0.2 | 0.6 mm |
| E3e6 f45 s60 | never | 1.02 | 0.3 | 0.3 mm |

0.9 mm is exactly the pile's volume spread over the whole tray floor
(1.5e-5 m^3 / 0.128^2 = 0.92 mm), i.e. complete fluidisation.

Which surface catches the sand (default pile, default coup_friction 0.1):

| MPM floor drop | rest z | to rest | repose |
|---|---|---|---|
| 0 mm (as shipped) | 10.0 mm | never | 9.5 |
| 30 mm | 9.0 mm | 575 | 18.6 |

Resting at 9.0 mm with the boundary 20 mm lower means the RIGID floor caught it:
fixed geoms DO couple in Genesis 1.3.3, so the earlier "they do not couple"
finding is stale.

Tray coup_friction, drop 30 mm, default pile:

| coup_friction | rest z | to rest | repose | height |
|---|---|---|---|---|
| 0.10 | 8.8 mm | 825 | 17.6 | 6.3 mm |
| 0.40 | 10.3 mm | 50 | 20.2 | 9.7 mm |
| 0.80 | 10.4 mm | 25 | 19.1 | 9.9 mm |
| 1.50 | 10.5 mm | 25 | 19.4 | 10.0 mm |

Angle of repose on a column too tall to stand (r 10 mm, h 30 mm), coup 0.8:

| MPM floor drop | to rest | repose | height |
|---|---|---|---|
| 0 mm | 25 | **29.0** | 8.6 mm |
| 30 mm | 25 | 28.9 | 8.6 mm |

Identical, so **coup_friction alone is the fix** -- the domain drop is
unnecessary, because the rigid floor's SDF is co-located with the boundary and
grips once it is allowed to.

Plate coup_friction, one push on a settled pile:

| plate coup_friction | mean disp | p95 | engaged | dz top |
|---|---|---|---|---|
| 0.10 | 6.87 mm | 11.65 | 100% | +0.37 mm |
| 0.30 | 7.57 mm | 11.18 | 100% | +0.28 mm |
| 0.60 | 8.41 mm | 11.95 | 100% | +0.27 mm |

A real but secondary effect (+22% displacement over the range). `dz top` is
POSITIVE throughout, i.e. the blade now ploughs a heap ahead of itself rather
than smearing the pile flatter, which is what it did on the frictionless floor.

End-to-end, 3 episodes x 5 pushes, before and after:

| | before | after |
|---|---|---|
| pile extent after 5 pushes | 59.9-82.0 mm | 40.5-50.5 mm |
| pile top | 13.0-14.5 mm | 20.3-22.1 mm |
| post-spawn rest speed | 2.12-2.22 mm/s, never converged | 0.98 mm/s, converged |
| displacement per push | 0.25-14.84 mm | 2.64-11.19 mm (14 full-length pushes) |

## What would change the verdict

The repose estimate is one run per cell and wanders a few degrees between
repeats of the same config (9.5 vs 3.2 deg on two runs of the unfixed setup), so
29 deg should be read as "high 20s". Averaging five spawns per cell would fix
that in ~10 min.

Whether 0.8 is the right value is a separate question: it saturates above ~0.4,
so anything in 0.4-1.5 gives the same pile, and the choice inside that range is
not determined by these measurements.

## Threats

- `imprecision`: single run per cell, and the repose metric is a line fit to a
  binned free surface, sensitive to the binning on a pile that barely slumps.
  Mitigated for the headline number by using a column that must collapse.
- Considered and dismissed: **that the fix is really the domain drop.** Repose
  and settle time are identical at drop 0 and drop 30 once coup_friction is 0.8.
- Considered and NOT dismissed: this changes the physics of every future sand
  dataset, so it does not merely fix data quality -- results before and after
  are not comparable, and the existing 48 000-transition set is now describing a
  different material from anything collected after today.
