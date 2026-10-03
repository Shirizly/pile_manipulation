---
id: DS-0019
title: FleX carrot-pile TEST slates -- 100 same-state slates x 200 obj-biased pushes, variable pile size (46-766 rigids, 1.6k-20.6k particles), native FleX units
status: active
date: 2026-10-01
path: datasets/DS-0019-slates-flex-pile-varN/ (renamed 2026-10-01 from DS-0019-slates-Flex-pile-N864 -- the N864 label was wrong, pile size varies); data/ (raw, gitignored), cache/ (gitignored), config.yaml + splits.json (tracked)
producer: ported from an external FleX project -- collect_true_action_results_parallel.py (true_action_slates_objbiased run; run_config.json and true_action_slates_FORMAT.md shipped alongside and left unchanged)
---

## What it is

Test-only same-state slate corpus for EXP-0061 (models trained on DS-0020).
100 initial states x 200 independently sampled pushes, each replayed from the
SAME settled state (a slate per state). Format fully documented in
`true_action_slates_FORMAT.md`; go through `manifest.jsonl`, never list files.
`run_config.json`: global_scale 24, wkspc_w 5.0, action_margin 1.6 (actions in
+-3.4), particle_r 0.125, cam_idx 0, `action_sampler: obj_biased`,
`init_pos_mix: [rand_blob, rand_spread]` (alternating by `state_idx % 2`),
6 equal-count push-length bins with edges 0.96 / 2.40 / 3.85 / 5.29 / 6.73 / 8.17 / 9.62.

## Units and frame

Same as DS-0020 (see its DATASET.md): **native FleX units; the stored actions'
2nd/4th components are -z of the particle files** (measured on this corpus too:
literal (x, z) reading -> 22 % of in-path particles move; `a1 = -z` -> 99.7 %).
Loader frame `X = x_flex, Y = -z_flex`, grid row = X, col = Y. Grid / plate /
flag thresholds are deliberately IDENTICAL to DS-0020 (`config.yaml`): +-7.2,
64 px (0.225 units/px), disk radius 1 px, plate 2.4 x 0.225.

## Audit

| check | result |
|---|---|
| actions | 20,000 recorded, **17,186 valid**; invalid: 2,598 `solver_explosion`, 216 `repeated_batch_abort` (14.1 %) |
| valid actions per slate | 97-199 (median ~175) |
| pile size | particles 1,621-20,560; `initial_state.npz['rigid_rotations']` count 46-766 (NOT 106-430 as first surveyed); particles per rigid 7.1-46.8. rand_blob median 3,572 particles / 190 rigids, rand_spread median 13,035 / 430 |
| particle ordering | `initial_particles.npy == initial_state.npz['positions']` asserted for all 100 states; displacement ~0 far from the push |
| action range / push length | +-3.4; length min 0.96, p5 1.52, p25 3.12, median 5.33, p75 7.1, p95 8.57, max 9.6; bins 0-5 have 2,792-2,937 valid rows each |
| NaN | 0 |
| **escaped** (any |coord| > 10) in rows marked `valid` | **397 rows (2.3 %)**, 2-364 particles each (median 60), up to |coord| ~ 129 -- solver blow-ups the collector did not catch; present in 95 of 100 slates |
| out of grid (> 1 % of particles outside +-7.2) | 47 rows |
| null (max XZ displacement < 0.1) | 159 rows (0.93 %); < 0.05: 143, < 0.25: 172 |
| float16 cache error | 0.0039 units inside the grid (0.057 only for escaped particles at |coord| ~ 100) |
| **kept** (`exclude_flagged=True`) | **16,583 / 17,186 valid rows, all 100 slates, 86-191 rows per slate (median 167.5)** |

Plate check on this corpus (16,360 transitions, `figures/data_check/plate_width.json`):
moved-fraction edge **h = 1.467 (CI 1.466-1.469)**, symmetric; wider than DS-0020's
1.337 because these piles are denser and multi-layer (pieces just outside the
plate are dragged by their neighbours); carried-fraction edge 1.016. Consistent
with a 2.4-wide plate (half-width 1.2 + particle radius + piece drag).

## Slate structure and split

`splits.json`: `splits.test` = all 100 `state_idx`; `slates` = `state_idx ->`
sorted valid `action_idx` list (17,186 rows, BEFORE the loader's flag filter --
`FlexPileData.flags` says which it drops). In the loader / eval cell every row is
a step-0 candidate (`step_idx = 0`) and `slate_idx = state_idx`, so
`eval_report._capture_report` scores exactly these 100 pools. **Pool size differs
from every Genesis corpus (86-191 vs 128/1000): slateN is not comparable across
corpora.**

## Cache and loader

`python -u -m FlexData.build_cache ds0019` -> `cache/state_{000..099}.npz`
(`init_xz (P,2) f16`, `after_xz (A,P,2) f16` raw flex (x, z), `action_idx`,
`actions`, `push_length`, `bin`, displacement stats, `n_particles`, `n_rigids`)
+ `cache/manifest.json`, atomic per state; ~400 MB vs 3.2 GB raw. Eval cell:
`FlexData.dataset.load_flex_cell` (corpus `flex_ds0019` in
`Baselines/common/eval_report.py`). Soft scoring truth:
`FlexData.dataset.truth_for_scoring_flex` (mass 1 per particle -- particle counts
differ 13x across slates, harmless within a slate).

## Known train/test mismatch (vs DS-0020) -- state it with every result

DS-0020 has ONE start state (a uniform full-workspace spread, 19,513 particles;
mean occupied fraction of the 64x64 grid 0.445, p5-p95 0.37-0.53 over its val
rows) and uniform full-workspace actions; DS-0019 has compact blob/spread piles
of very different size (1.6k-20.6k particles; mean occupied fraction 0.12) and
obj-biased actions inside +-3.4. Push-length ranges overlap (DS-0020 0.04-13.2
covers DS-0019 0.96-9.6).

## Harness smoke test (2026-10-01, eval_report functions on all 16,583 kept rows)

`persistence` accuracy 0.000 (by construction); `random` slateN (3 seeds,
default goal set, soft truth) lyapunov -0.005 / mass_in_region +0.013 /
signed_mass +0.016; truth-as-prediction 1.000 on all three (plumbing check).
**Ceiling note:** the hard disk-raster `occ1` used AS a prediction scores only
0.969 / 0.796 / 0.826 against the soft particle truth -- the binary occupancy
cannot see stacking in these multi-layer piles, so a perfect occupancy model is
capped near 0.8 on the mass value functions here. Report this ceiling next to
any FleX slateN.

Occupancy source is pluggable exactly as for DS-0020 (`particles` default /
`image_mask` hook, `FlexData.dataset.ImageMaskSource`). This corpus ships
**colour PNGs only (no depth)**; `cache/image_paths.json` indexes the initial
and every valid after-colour PNG (all present). PNGs are never moved.
**Image-mask cache (2026-10-01):** `python -u -m FlexData.image_mask ds0019` ->
`cache/image_masks.npz` (`state_ids (100,)`, `init (100,64,64) u8`, `init_frac`,
`after_keys (17186, 2) [state_idx, action_idx]`, `after (17186,64,64) u8`,
`after_frac` f16) + parts/manifest/`.DONE`; every initial state + every valid
after-state, 45 s. Mean occupied fraction 0.117 (init) / 0.126 (after).
IoU vs particle raster 0.946 mean (min 0.852; 200 states = prototype), frame
shift peak (0, 0), 99.8 % of removed mask pixels inside the swept region
(54 % under the negated-action reading) -- see DS-0020's DATASET.md.
