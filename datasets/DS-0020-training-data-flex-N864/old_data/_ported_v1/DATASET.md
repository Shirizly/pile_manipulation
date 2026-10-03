---
id: DS-0020
title: FleX carrot-pile TRAIN chains -- 2000 trajectories x 10 pushes from ONE uniformly spread 864-piece start state, native FleX units
status: superseded (archived 2026-10-02 -> old_data/_ported_v1/; replaced by the DS-0020 v2 payload, see ../../DATASET.md)
date: 2026-10-01
path: datasets/DS-0020-training-data-flex-N864/old_data/<traj 0..1999>/ (raw, gitignored; was data/<traj>/); old_data/_ported_v1/cache/ (compact, gitignored); old_data/_ported_v1/{config.yaml,splits.json,DATASET.md} (tracked)
producer: ported from an external FleX project (carrot-pile pushing; collector gnn_dyn data_gen, the same family as collect_true_action_results*_parallel.py) -- not collected in this repo, no collector config shipped
---

> **ARCHIVED (2026-10-02).** This describes DS-0020 **v1**, the first ported payload (2000
> trajectories from ONE start state). The user replaced the dataset's payload with a new collection
> (v2, `../../DATASET.md`); the v1 raw dirs were moved to `old_data/<traj>/` and this file, its
> `config.yaml` (cache_dir/split_file repointed), `splits.json` and `cache/` to `old_data/_ported_v1/`.
> `cache/image_paths.json` entries were rewritten `data/<t>/...` -> `../<t>/...`. Everything below
> (paths `data/`, `cache/`) refers to that archived layout. **EXP-0061 and MODEL-0004..0007 were made on
> THIS v1 data** -- load them with `old_data/_ported_v1/config.yaml`.

## What it is

Training corpus for EXP-0061 (rerun of EXP-0001 on FleX data: train here, test on
DS-0019). 2000 trajectories x 10 pushes = **20,000 transitions**; push `k` takes
state `k` to state `k+1`, so each trajectory is a 10-push CHAIN (rows within a
trajectory are correlated -- split by trajectory, never by row).

Raw layout (unchanged, `data/<t>/`): `{0..10}_particles.npy` (flat float32,
reshape `(19513, 4)` = `x, y, z, inv_mass`; y is UP, table plane is XZ; `inv_mass`
= 0.2 for every particle), `{0..10}_color.png` / `_depth.png` (720x720, camera
`cam_idx` 0; not used by the loader), `actions.p` (pickle, `(10, 4)`).
19513 particles in every state of every trajectory; 864 carrot pieces (per-piece
ids / rigid state are NOT stored, so no piece-level data). No config file shipped;
the user confirmed the scene is DS-0019's `run_config.json` scene
(`global_scale 24, wkspc_w 5.0, particle_r 0.125, cam_idx 0`).

## Units and frame (READ THIS before using the raw files)

* **Units: native FleX workspace units** (global_scale 24), NOT metres; the user
  chose not to rescale to Genesis.
* **The stored actions are NOT `[x0, z0, x1, z1]` of the particle frame: their 2nd
  and 4th components are `-z`.** Measured (EXP-0061 `plate_width_check.py` +
  hypothesis scan): taking actions literally, only ~7 % of particles in the
  plate's path move; with `a1 = -z` ~100 % do (all 9 axis swaps/negations tested;
  only this one works, on both DS-0019 and DS-0020). The loader therefore uses the
  **table frame `X = x_flex, Y = -z_flex`** (= the actions' own frame, as stored;
  right-handed with y up, same handedness as Genesis x/y/z-up) and converts
  particles (`FlexData.dataset.flex_xz_to_table`).
* Grid: **row (dim 0) = X = x_flex, col (dim 1) = Y = -z_flex** (the repo's
  row = world x, col = world y convention); confirmed visually in
  `experiments/EXP-0061-flex-cross-corpus-rerun/figures/data_check/DS-0020_triples.png`.

## Grid choice (resolved in config.yaml)

`half_extent 7.2`, 64x64 -> **0.225 units/px (to_pxl 4.444)**, centred at the
origin (PileSweepData's `ctr = W/2` convention). Chosen from the WHOLE corpus
(220k states, 4.3e9 particle-states; histogram in `cache/manifest.json`): the
start state spans ~x [-5.3, 6.04], z [-5.33, 6.0]; pushed material reaches
further. Fraction of particle-states outside the grid: +-6.0: 1.6 %, +-6.4:
0.29 % (8 % of states lose > 1 %), +-6.8: 0.024 %, **+-7.2: 0.002 % (0.9 % of states
lose > 0.1 %, none > 1 %)**. 64 px kept so every 64x64 consumer (NFD UNet,
goals, eval harness) works unchanged. Occupancy: hard disk raster, radius 1 px
(particle diameter 0.25 = 1.1 px; a 1-px radius fills a piece solidly).
Plate: width **2.4** (0.1 x global_scale, validated below), thickness 0.225
(= 1 px, a rendering choice matching Genesis's 1-px plate channel), sigma 0.75 px.

## Plate geometry, validated from the data (coordinator request)

`experiments/EXP-0061-*/code/plate_width_check.py` ->
`figures/data_check/plate_width.{png,json}`. Push 0 of every trajectory (1891
with L >= 1.5), each in its own push frame; particles initially at
0.15L < u0 < 0.85L. Moved fraction (du > 0.25) is 1.000 for |v0| < ~1.05 and
drops with a logistic edge at **h = 1.337 (95 % bootstrap CI 1.334-1.341,
width 0.10)**, symmetric (+v 1.336, -v 1.339). Expected for a 1.2 half-width
plate plus one particle radius: 1.2 + 0.125 = **1.325 -> consistent with 2.4**.
Particles that RODE with the plate (du > 0.5 (L - u0)) have an edge at 0.96
(material near the plate ends slides off sideways). Stop: carried central
particles end at u1 - L median 0.58 (p25 0.35), i.e. piled just ahead of the
push end point -- consistent with the plate centre stopping at the action end
point; the plate thickness itself is not resolved (<~0.45).

## Audit (whole corpus; `FlexData/build_cache.py` + `FlexData.dataset.FlexPileData` flags)

| check | result |
|---|---|
| start states | **all 2000 trajectories start from the IDENTICAL state 0** (max abs difference 0.0): a uniform single spread of 864 pieces over the workspace. Push 0 rows are therefore same-state branches; diversity comes from pushes 1-9 |
| particle count | 19513 in every state |
| particle ordering | consistent across states: moved fraction is ~0 for |v0| > 2.5 from the push (a permutation would show displacement everywhere) |
| actions | uniform over the FULL `[-5, 5]^2` (|a| max 4.9999) -- **no 1.6 action margin, unlike DS-0019 (+-3.4)**; not object-biased |
| push length | min 0.04, p1 0.56, p5 1.34, p25 3.3, median 5.14, p75 7.08, p95 9.4, p99 10.8, max 13.21 |
| NaN | 2 states (traj 46 state 10, traj 1132 state 10) -> 2 rows |
| escaped (any |coord| > 10) | 49 rows in 9 trajectories (9-51 particles each; escaped pieces stay out for the rest of that trajectory); one state has a particle at y = 21 |
| out of grid (> 1 % of particles outside +-7.2) | 0 rows |
| null (max XZ displacement < 0.1) | 121 rows (0.61 %); < 0.05: 113, < 0.25: 160 |
| float16 cache error | max 0.0039 units (0.017 px) |
| **kept** (`exclude_flagged=True`) | **19,828 / 20,000**; train 17,837 / 18,000, val 1,991 / 2,000 |

## Split

`splits.json`: **by trajectory**, `np.random.default_rng(0).permutation(2000)`,
first 200 -> val (200 trajectories, 2,000 rows), 1800 -> train (18,000 rows);
`test` empty (the test corpus is DS-0019). Made by `python -u -m
FlexData.build_cache splits`. Caveat: because every trajectory shares state 0,
val push-0 rows have the same pre-push state as train push-0 rows (different
actions).

## Cache and loader

`python -u -m FlexData.build_cache ds0020` -> `cache/chunk_{000..019}.npz`
(100 trajectories each: `xz (T,11,19513,2) float16` raw flex (x, z),
`actions`, `traj_ids`, `max_disp`/`p99_disp`/`n_moved_*`, `bbox`, `n_nan`,
`y_max`) + `cache/manifest.json` (per-chunk audit + corpus histograms),
atomic per chunk. ~1.7 GB vs 14 GB raw. Read with `FlexData.dataset.FlexPileData`
(registered dataset type `flex`; training config
`configs/dataset/flex_ds0020_train_ds0019_test.yaml`).

Occupancy source is pluggable (`occupancy.source` in `config.yaml`, or
`occ_source` on the dataset/factory): `particles` (default, the disk raster
above) or `image_mask` (precomputed top-down masks from the colour PNGs, loaded by
`FlexData.dataset.ImageMaskSource`). **Image-mask cache (2026-10-01):**
`python -u -m FlexData.image_mask ds0020` (owner `FlexData/image_mask.py`;
colour PNG segmented, derived camera f = 869.12, table-plane warp, binary =
area fraction > 0; depth PNGs NOT used, for parity with DS-0019) ->
`cache/image_masks.npz` (`traj_ids (2000,)`, `masks (2000, 11, 64, 64) uint8`,
`frac` float16 area fraction), `cache/image_masks_parts/chunk_*.npz`,
`cache/image_masks_manifest.json`, `cache/image_masks.DONE`; all 22,000 states,
69 s on 14 workers. Mean occupied fraction 0.418. Validated
(`experiments/EXP-0061-*/code/image_mask_cache_check.py` ->
`figures/data_check/image_mask_check.json`, `image_mask_triples_DS-0020.png`):
IoU vs the particle raster 0.946 mean (min 0.919, 200 states, identical to the
prototype), loader-frame shift scan peak at (0, 0), 99.9 % of removed mask
pixels inside the action's swept region (37 % with the 2nd/4th action
components negated) -- the -z action convention holds in the image frame. Every PNG
stays in `data/`; `cache/image_paths.json` indexes `[color, depth]` per
(trajectory, state) (all 44,000 present). Scoring truth is always the particle
soft splat. Mean occupied fraction of the grid (particle raster, val rows):
0.445 (p5-p95 0.37-0.53) -- vs 0.12 on DS-0019; see DS-0019's DATASET.md
"Known train/test mismatch".
