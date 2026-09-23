# DS-0003 — synthetic particle states (compaction generator, 2026-09-17 rewrite)

**Status:** active
**Payload:** `datasets/DS-0003-synthetic-states/data/states.pt` (gitignored; regenerate with the command below)
**Generator:** `Baselines/common/pile_compaction.py::sample_compacted_state` (reusable capability)
**Legality test:** `Baselines/common/cube_overlap.py::state_is_legal` (exact separating-axis test, tol=0.0)
**Driver:** `datasets/DS-0003-synthetic-states/generate.py`

## What this is

**~30,000 synthetic pile states** (10,000 each at `n_objects` = 20 / 50 /
100), same schema as real corpus states: `(n_objects, 7)` = xyz (m) + wxyz
quaternion, z fixed at `REST_Z = 0.012482` m, yaw-only rotation, bounds
`DEFAULT_BOUNDS = +-0.064 m`. Generated to give the encoder/value-function
corpus (DS-0002) breadth and density variation cheaply, without simulation.

## Why this generator replaces the old one (the point of this rewrite)

The previous generator (`goal_configs.py::sample_synthetic_state`, B1
scattered + B2 sequential clumps) checked legality with
`goal_configs.py::assert_no_penetration`: an axis-aligned circumscribed-box
test using the diameter `CUBE_SIZE*sqrt(2)` per axis. That bound is a
sufficient-but-far-from-necessary condition for non-penetration — two cubes
sitting diagonally at `dx=dy=CUBE_SIZE` (centre distance `CUBE_SIZE*sqrt(2)`,
genuinely legal at ANY yaw) are refused by it. Measured directly against
`DS-0002` (real simulated states): **this bound rejects 91-99% of real
states outright.** It cannot express "contact" at all, so no generator built
on top of it could ever be made to produce a compacted pile — the old
generator's `clump_prob`/`clump_size` machinery grew dendritic filaments
instead (measured: synthetic piles averaged 0.66-0.79 touching neighbours
per cube versus 1.08-2.05 for real, and that count did **not** rise with
`n_objects` — adding material made piles bigger, never denser).

The new generator (`pile_compaction.py::sample_compacted_state`) is a grid
seed followed by per-object relaxation toward the pile centroid, with every
candidate move checked against the exact separating-axis test
(`cube_overlap.py::overlaps_pairs`, which handles yaw exactly, no
approximation) at a small negative-`tol` safety margin. This is what makes
compaction possible at all: rigid row/column translation does not work
(measured: with random yaws one near-contact pair per row blocks the whole
row, spread barely moves), but relaxing cubes individually lets them slip
into gaps their neighbours leave.

`assert_no_penetration`/`_penetrates`/`mask_to_configuration` in
`goal_configs.py` are **unchanged** — goal generation (and claim C-021)
depends on their exact current behaviour. The new exact test lives
alongside the old one; migrating goal generation to it is a separate,
un-taken decision.

## Generation recipe

Per state:
- **`compaction` ~ Uniform(0.05, 1.0)** — the single dispersed<->compact
  diversity axis, replacing the old `clump_prob`/`clump_size` machinery
  entirely. Sets the effective relaxation sweep count
  (`n_eff = round(compaction * n_sweeps)`, `n_sweeps=26`).
- **`yaw_kappa` ~ Uniform(0, 6)** — von Mises concentration of yaws around a
  random common heading (real cubes partially align and pack tighter than
  uniform-random yaws allow). Capped at 6: at kappa=40 the seed lattice
  starts too tight and legality dropped to 4/8 in the coordinator's probe.
- `sample_compacted_state` returns a **centred** pile (mean-subtracted xy).
  This driver then places it at a **uniform random offset inside
  `DEFAULT_BOUNDS`**, rejecting offsets that would push any cube outside the
  workspace (conservative any-yaw half-extent pad,
  `CUBE_SIZE*sqrt(2)/2`). This offset step — not the compaction routine
  itself, which has no notion of the workspace — is what produces
  centre-of-mass spread.
- Legality asserted with `state_is_legal(xy, yaw, CUBE_SIZE, tol=0.0)` (the
  exact test) before a state is kept; any failure is counted and reported,
  not silently dropped (see Generation run below — zero occurred).

Schema is unchanged from the previous generator: `xyz`, `quat`, `n_objects`,
`group`, `cube_size`, `rest_z`, `bounds`. `n_clumps`/`n_scattered`/
`clump_sizes` are retired (nothing downstream reads them outside this
dataset's own generator); `compaction` and `yaw_kappa` are recorded per
state in their place.

## Measured packing performance (coordinator's pre-integration probe, 8 states/cell, tol=0.0)

| n | compaction | spread | legal | ms/state |
|---|---|---|---|---|
| 20 | 0.15 | 0.0122 | 8/8 | 14 |
| 20 | 1.0 | 0.0106 | 8/8 | 87 |
| 50 | 0.15 | 0.0196 | 8/8 | 35 |
| 50 | 1.0 | 0.0171 | 8/8 | 215 |
| 100 | 0.15 | 0.0276 | 8/8 | 72 |
| 100 | 1.0 | 0.0242 | 8/8 | 423 |

Real spread minima for reference: n=20 0.0074, n=50 0.0156, n=100 0.0209.
The old generator's minima were 0.0155 / 0.0280 / 0.0375 — the new generator
reaches substantially closer to real at every count, and beats the old
generator's floor outright at n=50/100.

**Known residual, not tuned away:** at n=100 the tightest synthetic pile
(~0.0242) is still ~15% short of real's tightest (0.0209). Cause:
per-object relaxation converges to a local packing minimum, not the global
one. Real piles also (a) tilt out of plane (mean 5-13 degrees, measured) and
(b) ~2% of cubes sit in a second layer — neither reproducible by a
single-layer, yaw-only generator. This is a structural limitation of the
approach, not a tuning problem.

## Generation run (this regeneration)

See the numbers filled in by the run itself, and
`data/generate_summary.json`. Regeneration command:

```
OMP_NUM_THREADS=4 /home/alon/anaconda3/envs/pme/bin/python -u \
    datasets/DS-0003-synthetic-states/generate.py \
    --n-per-count 10000 --counts 20 50 100 --workers 4 --seed 0
```

To generate more states beyond this run, rerun with a larger
`--n-per-count` (or additional runs at a new `--seed`, since states are not
resumable/appendable — the RNG stream is reseeded per-worker via
`numpy.random.SeedSequence.spawn`, so a different seed draws an independent,
non-overlapping stream).

## Validation against DS-0002 (real states)

`datasets/DS-0003-synthetic-states/validate.py` rasterises a **matched
n_objects mix** (1/3 each of 20/50/100) of both corpora through the fast,
bit-exact rasteriser (`experiments/EXP-0019-*/code/raster.py::rasterise`)
and compares `dmdc_baseline.py::occupancy_descriptors`. Full numbers:
`datasets/DS-0003-synthetic-states/validation/results.json`; figure:
`datasets/DS-0003-synthetic-states/validation/real_vs_synthetic.png`.

**Result of this rewrite (full before/after table, per-`n_objects` spread/COM/
neighbour-count comparison, and root cause): see
`datasets/DS-0003-synthetic-states/validation/RESULTS.md`.** Headline: `com`
stays fixed at 1.00 coverage and the packing floor at n=50/100 beats the old
generator's, but `moments2`/`dft_real`/`dft_imag` coverage **regressed**
(0.67/0.40/0.38 -> 0.00/0.00/0.10) because this generator's dispersed end is
still a cube-scale grid seed, not a large placement region — it produces
tight-to-very-tight piles across its whole `compaction` range, never the
loosely scattered states the old generator's region mechanism reached. Mean
touching-neighbour count (thr=0.00813 m) is now ~2x real at every
`n_objects` (e.g. n=20: synthetic 3.64 vs real 1.04); spread range at n=20
is `[0.0102, 0.0133]` against real's `[0.0079, 0.0629]`. This is a reported,
not hidden, limitation — the new generator solves the old one's inability to
compact, but at the cost of no longer covering the dispersed tail.

## Reuse, not reimplementation

This generator does not re-derive the SAT overlap maths or the relaxation
routine: `Baselines/common/cube_overlap.py` and
`Baselines/common/pile_compaction.py` are pre-existing, unit-tested modules,
imported unchanged. `generate.py` is only the sweep/config/multiprocessing/
save driver, plus the workspace-placement step described above.
