---
id: DS-0020
version: 2
title: FleX carrot-pile TRAIN chains v2 -- 1997 trajectories x 10 obj-biased pushes from varied rand_blob / rand_spread piles (46-766 pieces), DS-0019's scene, native FleX units
status: active
date: 2026-10-02
path: datasets/DS-0020-training-data-flex-N864/data/true_action_transitions_carrots/ (raw, gitignored); cache/ (compact, gitignored); config.yaml, splits.json, DATASET.md (tracked)
producer: external FleX project, same collector family as DS-0019 (collect_true_action_results*_parallel.py, "transitions" mode, 2 shards); run_config.json shipped in the raw dir. Collected and dropped in by the user 2026-10-02.
supersedes: DS-0020 v1 (2000 trajectories from ONE uniform-spread state) -- archived at old_data/<traj>/ (raw) + old_data/_ported_v1/ (config.yaml, splits.json, cache/, DATASET.md). EXP-0061 and MODEL-0004..0007 used v1.
---

## What it is

Training corpus in-domain with the DS-0019 test slates: same scene (`run_config.json`:
global_scale 24, wkspc_w 5.0, **action_margin 1.6 -> actions in +-3.4**, particle_r 0.125,
cam_idx 0, carrots), same pile generator (`init_pos_mix [rand_blob, rand_spread]`), same
`obj_biased` action sampler and the same six push-length bins (edges 0.96 / 2.40 / 3.85 /
5.29 / 6.73 / 8.17 / 9.62). `n_states 2000, n_steps 10, seed 0`, `max_retries 20`.
Each trajectory is a 10-push CHAIN (rows correlated -- split by trajectory, never by row).

Raw layout (`data/true_action_transitions_carrots/`, read through `manifest.jsonl`):
`state_init` x1997 (`n_particles`, `positions_path`, `color_path`, `init_pos`),
`transition` x19,970 (`state_idx, step_idx, action [x0, z0, x1, z1], valid, push_length,
bin, retries, after_positions_path, after_color_path`), `state_failed` x3 (802, 982, 1893:
`repeated_reset_crash`, no data). Per trajectory `<t>/initial_particles.npy`,
`initial_state.npz` (positions == initial_particles.npy for all 1997; `rigid_rotations`
gives the piece count), `initial_color.png`, `<k>_after_particles.npy` (flat float32,
reshape (P, 4) = x, y_up, z, inv_mass), `<k>_after_color.png` (720x720). **No depth PNGs.**
`manifest_shard{0,1}.jsonl` / `reset_crash_attempts_shard*.json` are collector bookkeeping:
trajectory 821 crashed after 2 pushes in shard 1 and was re-run (identical actions); the
merged `manifest.jsonl` holds only the complete re-run, which is what is on disk.

## Units, frame, chaining (verified on THIS data)

Native FleX units (NOT metres). Table frame **X = x_flex, Y = -z_flex**, grid row = X,
col = Y -- unchanged from v1 / DS-0019. Re-verified here
(`experiments/EXP-0061-*/code/data_check_v2.py` -> `figures/data_check_v2/data_check_v2.json`):

| check | as stored (Y = -z) | 2nd/4th negated (Y = +z) |
|---|---|---|
| removed mask pixels inside the swept region (3,000 rows) | **0.997** | 0.472 |
| changed mask pixels inside the swept region | 0.791 | 0.373 |
| in-path particles that move (~1,200 rows, L > 1.5) | **0.993** | 0.477 |

**Chaining (state k+1 before-state = push k's after-state):** no before-state file exists,
so tested by locality on all 1997 trajectories: of the particles that move > 0.25 in push
k >= 1, the fraction lying OUTSIDE push k's padded swept box is **0.008 mean (median 0,
p99 0.053)** with before = after-state of push k-1, vs **0.33 (median 0.25)** with
before = initial state (the "reset each push" alternative); 0.0015 for push 0 (true before
= initial). Same on the 1,441 steps that needed action retries (0.002 vs 0.32), so a retry
did not perturb the state. The cache/loader therefore takes **state 0 = initial, state
k+1 = after-state of push k**. Particle count is constant along every trajectory (asserted).

## Counts

| | count |
|---|---|
| trajectories with data (init + 10 valid transitions) | 1997 (3 `state_failed`, 0 invalid transitions, 0 incomplete) |
| **excluded: trajectories 0-99** (leakage, below) | 100 (1000 rows) |
| kept trajectories | **1897** -> train 1707 / val 190 |
| rows (train / val) | 17,070 / 1,900 |
| **kept after flags** (`exclude_flagged=True`) | **train 15,093 / val 1,665** (all 1997: 17,670 / 19,970) |
| flagged NaN / escaped / out_of_grid / null (all 1997 traj) | 0 / 1,937 / 126 / 271 (train 0/1,657/109/245, val 0/208/10/17) |

**Escaped is large here (9.7 % of rows, v1: 0.25 %)**: 325 trajectories have a particle
flung beyond |x|,|z| > 10 (3-270 particles, median 51, at the first escaped push; first
escape spread evenly over pushes 0-9; 135 blob / 190 spread). The escaped-particle count never
decreases along a trajectory (measured, 0 returns), so **1,612 of the 1,937 flagged rows are
carry-over rows** whose before-state already holds the escaped particles (outside the camera's
view, +-7.46 at the table, so before AND after images both lack them); only **325 rows are the
escape event itself** (kept trajectories: 315 events / 1,550 carry-over). The default flag drops
all of them -- a "first escape only" filter would keep ~1,550 more train/val rows (not implemented;
a decision for the training run).

## Leakage exclusion (user decision 2026-10-02)

The collection used DS-0019's seed, so **trajectories 0-99 start from near-copies of
DS-0019's 100 test piles**: same piece count for all 100 indices, initial image-mask IoU vs
the same-index DS-0019 state median **0.906**, min 0.720 (reproduced here). All steps of
trajectories 0-99 are excluded from train and val (`splits.json["excluded"]`).

Scan of the remaining 1897 initial states against all 100 DS-0019 initial states
(`code/data_check_v2.py` + `code/leakage_scan_v2.py` -> `figures/data_check_v2/leakage*.{json,png}`):

* **Mask IoU > 0.7 is NOT a usable near-duplicate test on this data**: 1192/1897 trajectories
  exceed it (max-over-100 IoU median 0.766, p99 0.874, max 0.907), because two unrelated
  rand_spread piles both fill the same workspace square. Piece+particle counts equal for 10.
* Piece level (the discriminative test; `initial_state.npz` `rigid_translations`, piece
  order is spatial): mean per-piece-index centroid distance for the known leaked pairs
  0.025-0.107 (median 0.069); for trajectories >= 100, best match over every same-count
  DS-0019 state **>= 0.154** (p1 0.245, median 0.49) -- **no further near-duplicates**.
  Closest (listed, NOT dropped): 1776~DS-0019 84 (0.154, IoU 0.78), 252~12 (0.159, IoU 0.73),
  570~42 (0.173, IoU 0.79), 1668~0 (0.187, IoU 0.82), 108~68 (0.197, IoU 0.70).
  4 trajectories (1017, 1111, 1743, 1759) have one NaN rigid translation (particles finite).

## Distributions vs DS-0019 (`figures/data_check_v2/distributions.png`)

| | DS-0020 v2 kept (1897 traj) | DS-0019 (100 states) |
|---|---|---|
| blob / spread | 948 / 949 | 50 / 50 |
| pieces p5 / median / p95 (range) | 106 / 298 / 586 (46-766) | 106 / 298 / 586 (46-766) -- same 7 values |
| particles p5 / median / p95 | 1,921 / 7,936 / 18,462 (1,249-23,054) | 2,109 / 7,334 / 18,521 (1,621-20,560) |
| initial mask occupied fraction median (p5-p95) | 0.112 (0.027-0.238) | 0.093 (0.028-0.239) |
| push length p5 / p25 / median / p75 / p95 (range) | 1.03 / 2.41 / **3.77** / 5.16 / 6.75 (0.02-9.41) | 1.52 / 3.12 / **5.33** / 7.10 / 8.57 (0.96-9.60) |
| realized length bins 0-5 (< bin 0) | 3,889 / 5,053 / 4,870 / 3,357 / **894 / 78** (829 < 0.96) | ~2,800-2,940 each |

**Push length is the remaining mismatch**: the manifest `bin` is the REQUESTED bin
(realized bin agrees on 95.6 % of rows); the +-3.4 action clamp shortens long requests, so
bins 4-5 (6.7-9.6) are 5 % of v2 rows vs 1/3 of DS-0019's, and 4.4 % of v2 rows are shorter
than DS-0019's minimum. Manifest `push_length` = |action end - start| (to 1e-6).

## Split

`splits.json`: by trajectory over the 1897 complete trajectories with state_idx >= 100;
`np.random.default_rng(0).permutation(1897)`, first round(0.1 x 1897) = 190 -> val, 1707 ->
train, `test` empty (test corpus = DS-0019). Holds `excluded` (0-99 + reason) and `absent`
(802, 982, 1893). Made by `python -u -m FlexData.build_cache splits`.

## Cache and loader

* `python -u -m FlexData.build_cache ds0020` -> `cache/v2_chunk_{000..019}.npz` (100
  state_idx slots each; particles concatenated with offsets because P varies; format in
  `FlexData/build_cache.py`'s docstring) + `cache/manifest.json` (per-chunk audit, XZ
  histograms, failed list); float16 error <= 0.0039 inside |coord| < 10. 714 MB. ~15 s.
* `python -u -m FlexData.build_cache paths` -> `cache/image_paths.json` (state 0 =
  `initial_color.png`, state k = `<k-1>_after_color.png`; depth null; 0 missing).
* `python -u -m FlexData.image_mask ds0020` -> `cache/image_masks.npz` (`traj_ids (1997,)`,
  `masks (1997, 11, 64, 64) uint8`, `frac` float16 area fraction), per-chunk
  `image_masks_parts/`, `image_masks_manifest.json`, `image_masks.DONE`; 21,967 states, 85 s
  on 14 workers; mean occupied fraction 0.134. Same segmentation/camera/warp as v1 and
  DS-0019 (the `code_sha256` differs from v1's only by docstring/registry edits).
* Loader: `FlexData.dataset.FlexPileData` with `config.yaml` (`cache_format:
  traj_manifest_v2` -> `_TrajManifestSource`). **`occupancy.source: image_mask` -- model
  inputs are visual only**; particles are cached only for the flags, these audits and the
  particle-splat scoring truth, never as model input. Training config:
  `configs/dataset/flex_ds0020v2_train_ds0019_test.yaml`.
* **GNN (`Baselines/GNN/flex_predictor.py`)** needs nothing extra: it perceives each state
  from the colour PNG via `cache/image_paths.json` (constant-plane depth; `depth_mode="true"`
  is impossible on v2, no depth PNGs); a GNN *trainer* working from masks can read
  `cache/image_masks.npz` (`frac` gives sub-pixel area) directly.
* Data-check figure (before / plate / after masks, val rows):
  `experiments/EXP-0061-flex-cross-corpus-rerun/figures/data_check_v2/triples.png`.
