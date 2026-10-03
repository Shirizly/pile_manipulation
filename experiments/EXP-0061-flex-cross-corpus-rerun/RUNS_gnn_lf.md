# EXP-0061 -- GNN (original checkpoint) + linear visual foresight on FleX image masks: runs, choices, DS-0020 val numbers

> **Data-location note (2026-10-02).** Every "DS-0020" in this record is DS-0020 **v1** (2000
> trajectories from ONE uniform-spread start state). The user then replaced DS-0020's payload with a new
> blob/spread collection (v2). v1 was archived, not deleted: raw `datasets/DS-0020-training-data-flex-N864/old_data/<traj>/`,
> config/splits/cache (incl. `image_masks.npz`, `image_paths.json`) `old_data/_ported_v1/`; read it with
> `old_data/_ported_v1/config.yaml` (the code/ scripts and `configs/dataset/flex_ds0020_train_ds0019_test.yaml`
> were repointed there; logs and result files are left as written). The numbers here are unaffected.
> v2 data check: `figures/data_check_v2/`, `code/data_check_v2.py`, `code/leakage_scan_v2.py`.

2026-10-01, commit d72bb304 (dirty tree), python `/home/alon/anaconda3/envs/pme/bin/python -u`.
Everything here is on **DS-0020 val** (1,991 kept rows, 200 whole trajectories held out by
`splits.json`). **DS-0019 (test) was NOT scored** -- that is the next agent's job (commands at the end).
Input `occ0` and accuracy truth `occ1` are the **binary image masks** (`occ_source="image_mask"`,
`FlexData/image_mask.py`). `accuracy` = `fit_linear_foresight.metrics` swept-region accuracy
(plate 2.4 units = 10.67 px, half-width 0.5*plate+2, pad 0.5*plate); persistence = 0 by
construction (val persistence rms 0.563). No slateN here (DS-0020 has no same-state pools except
push 0); slateN comes from the DS-0019 run.

## Task A -- original dyn-res-pile-manip GNN, no training

Checkpoint `Baselines/GNN/data/gnn_dyn_model/2023-01-28-10-42-05-114323/net_epoch_0_iter_1000.pth`
(the folder/iter `config/mpc/config.yaml` of the source repo names). Loads strictly into
`model/gnn_dyn.py::PropNetDiffDenModel` (byte-identical to the source's `model/gnn_dyn.py`).

### Does the checkpoint match this data? -- yes
* Source training config `config/train/gnn_dyn.yaml`: carrots, global_scale 24, wkspc_w 5,
  particle_r 0.125, cam_idx 0, init_pos `spread`, **2000 episodes x 10 steps** -- exactly DS-0020's
  shape (one identical spread start state, 2000 x 10). Source train/valid = episodes 0-1799 / 1800-1999.
* Node-level check (the training loss's own quantity, renderer-independent; each node tracked to its
  nearest GT particle, table-XY DISPLACEMENT error), N=200, plane 0.24, 400 val rows / 9,722 moved
  nodes: model RMSE **1.10** vs persistence **3.02** FleX units on moved nodes (node accuracy 0.63
  moved / 0.61 all nodes). True depth: 1.10 vs 3.02 (0.64 / 0.62). Clearly better than persistence.
* Possible in-sample leak (our val split is random, the source held out episodes 1800-1999): mask
  accuracy 0.249 on our val rows with traj >= 1800 (208 rows) vs 0.252 on traj < 1800 -- no
  visible in-sample advantage.
* Caveat: the checkpoint is iteration 1000 of epoch 0 (batch 4 -> ~4k training samples seen); it is
  the one the source MPC config uses, not something we chose.

### Input pipeline as recovered (source: `env/flex_env.py::obs2ptcl_fixed_num_batch`, `utils.py`,
`dataset/dataset_gnn_dyn.py`)
1. Foreground = depth/24 < 0.599/0.8 (= height > 0.03 units). Equal pixel-for-pixel to
   "any RGB != 255" on the colour PNG (the collector whitens table pixels), so no depth is needed for
   the mask.
2. Back-project fg pixels to the OpenCV camera frame (camera (0,18,0) straight down, f 869.12,
   cx=cy=359.5; frame = (x, z, 18-y)) **divided by global_scale 24**.
3. Voxel-downsample 0.01 (scaled) -> FPS of `particle_num` points from a random start -> recenter
   (r = min(0.02, 0.5 r_fps)) -> particle_den = 1 / r_fps^2 (training range 15-6500).
4. Action: s/e at [a0, 0, -a1] FleX world -> camera frame; per-node `s_delta` = distance-to-end x
   push_dir x hard length gate x soft width gate exp(-excess/0.01), **pusher half-width 0.8/24**
   (training convention; the source planner uses 0.048, not what the weights saw). attrs = 0. One
   `predict_one_step` per push.
5. The source never renders a mask (its planner scores particles against the goal image). Our
   renderer: every fg PIXEL is carried by its nearest node's predicted XY displacement, re-projected
   on the table plane exactly like the masks, cell occupied iff any carried pixel lands in it. Zero
   motion reproduces the input mask exactly (0 px mismatch, asserted). k=4 inverse-distance blending
   was worse (n50 -0.05, n100 -0.01) -> k=1.

### particle_num (selected on 600 val rows, seed 0 subset; `results/gnn_sweep_val600.json`)
| N | mean den | acc true depth | acc plane 0 | acc plane 0.24 |
|---|---|---|---|---|
| 30 | 200 | 0.047 | 0.047 | -- |
| 50 | 390 | 0.097 | 0.104 | 0.102 |
| 100 | 900 | 0.211 | 0.204 | 0.209 |
| 200 | 2000 | 0.253 | 0.248 | 0.248 |
| 400 | 4500 | 0.263 | -- | 0.250 |

Plateau from 200; **N = 200** chosen (density mid-range of training's 15-6500; 400 is near the top).
Renderer bound (carry with the TRUE node displacements, 300 rows, `results/gnn_bound_val300.json`):
N=50 0.28, 100 0.38, 200 0.46 -- the node + nearest-carry representation itself caps mask accuracy
well below 1; the GNN reaches ~55 % of its bound at every N.

### Depth substitute (DS-0019 has no depth) -- measured on full DS-0020 val, N = 200
Options: (a) true depth PNG; (b) every fg pixel on a constant plane y = h. The source repo has no
depth-free path (it always used live rendered depth). Colour-from-height is weak (per-pixel
grey-vs-height r = 0.30-0.39), so no learned height map. h = 0.24 ~ median fg surface height over
100 DS-0020 train depth images (0.249, p5-p95 0.12-0.45).

| variant | val acc (1,991 rows) | 95 % CI | minus true depth (paired bootstrap) |
|---|---|---|---|
| true depth | 0.2549 | 0.2495-0.2603 | -- |
| **plane 0.24 (chosen, registered)** | **0.2517** | 0.2461-0.2571 | **-0.0033 [-0.0055, -0.0012]** |
| plane 0 (table) | 0.2486 | 0.2433-0.2539 | -0.0063 [-0.0085, -0.0041] |

Cost of dropping depth: ~0.003 accuracy (resolvable, small). Caveat for DS-0019: its piles are
multi-layer; a particle-based top-surface estimate gives median 0.40 (p95 0.68) vs 0.32 (p95 0.54)
on DS-0020 by the same estimator, so the 0.24 plane is ~0.1 low there -- the measured plane-0 vs
plane-0.24 slope (0.003 per 0.24) suggests a cost of the same order, but it is not measured.

Figures: `figures/gnn_check/pred_vs_true_n200_plane0.24.png` (registered config),
`pred_vs_true_n200_true.png`, `pred_vs_true_n100_plane0.24.png` -- input mask, truth, GNN, true-
node-displacement carry; 4 largest-change + 4 median rows. The GNN clears the swath only partly and
under-moves nodes (consistent with the node RMSE).

## Task B -- linear visual foresight (weights/MODEL-0004-linear-foresight-flex-mask)

`code/fit_lf_flex.py`: fit on DS-0020 train masks (17,837 rows), ridge toward identity, res 64
(D = 4096), lambda selected on DS-0020 val (trajectory split) from 0.1-3000.
EXP-0003 bin scheme adapted to FleX units: 6 equal-width bins over [0, max train length 13.21]:
edges 0 / 2.20 / 4.40 / 6.61 / 8.81 / 11.01 / 13.21; rows per bin train 2124 / 4867 / 5325 / 3961 /
1424 / **136**, val 234 / 565 / 606 / 433 / 139 / **14**. No bin below MIN_ROWS_PER_BIN = 50, but
the last bin is badly under-filled (M/D = 0.03) and its val number rests on 14 rows; every bin is
underdetermined (M < D), hence the large optimal lambda.

| lambda | 1 | 10 | 100 | **300** | **1000** | 3000 |
|---|---|---|---|---|---|---|
| switched val acc | 0.360 | 0.431 | 0.475 | **0.482** | 0.481 | 0.472 |
| single val acc | 0.320 | 0.326 | 0.344 | 0.351 | **0.354** | 0.352 |

Chosen: switched lambda 300 -> **0.482**, single lambda 1000 -> **0.354** (persistence 0). Per-bin
switched (lambda 300): 0.32 / 0.45 / 0.51 / 0.52 / 0.54 / 0.49; single (lambda 1000): -0.05 / 0.34 /
0.43 / 0.42 / 0.37 / 0.31. Selected on the same val split it is reported on (mildly optimistic;
curve flat within 0.01 over 100-1000). Closed-form fit equals `fit_operator` exactly (max abs diff 0).

For comparison on the same val rows: GNN (N 200, plane 0.24) 0.252 vs LF switched 0.482.

## Registered for eval_report (DS-0019 run NOT done here)
* `Baselines/common/eval_report.py` MODELS: `gnn_flex_drp` (flex_predictor, N = `GNN_FLEX_N` = 200,
  plane 0.24, ckpt env `GNN_FLEX_CKPT`, `is_gnn=False` -- truth is the image mask, the Genesis
  node-resampled truth path is not used), `lf_flex_switched`, `lf_flex_single` (MODEL-0004).
  `--ckpt MODEL=PATH` works for all three. New corpus `flex_ds0019_mask` (= `flex_ds0019` with
  `occ_source: image_mask`). Smoke-tested on 120 DS-0020 val rows through `_load_predictor` +
  `_accuracy`, cpu and cuda paths identical (GNN bit-identical; LF 4e-5).
* Command for the next agent:
  `python -u Baselines/common/eval_report.py --corpora flex_ds0019_mask --models gnn_flex_drp,lf_flex_switched,lf_flex_single --device cuda --out-prefix experiments/EXP-0061-flex-cross-corpus-rerun/results/ds0019_gnn_lf`
  (GNN perception reads the DS-0019 initial colour PNG once per slate; full DS-0020 val took ~10 min per config, dominated by one perception per row -- DS-0019 needs only 100).

## Commands run
    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/gnn_check.py --configs <sweep> --rows 600 --node-rows 200 --out results/gnn_sweep_val600.json
    python -u .../gnn_check.py --configs n200_plane0.24 n200_true n200_plane0 --node-rows 400 --save-pred all --out results/gnn_val_full.json
    python -u .../gnn_depth_bootstrap.py results/gnn_val_full.json n200_true n200_plane0.24 n200_plane0
    python -u .../gnn_bound_and_figs.py --cfgs n50_plane0.24 n100_plane0.24 n200_plane0.24 --rows 300 --fig-cfg ... --out results/gnn_bound_val300.json
    python -u .../fit_lf_flex.py --out weights/MODEL-0004-linear-foresight-flex-mask/operators.pt  (then renamed checkpoint.pt / fit.json)

## Open issues
1. **Node-check bug, fixed mid-run**: the first sweep's `node.*` fields (n30/n50/n100/n200_true/
   n200_plane0/k4) double-resolved the dataset index and are INVALID (flagged `_note` in the JSON);
   mask accuracies are unaffected. Valid node numbers: n400_*, n200_plane0.24 in the sweep JSON and
   all of `gnn_val_full.json`.
2. The GNN mask score depends on OUR renderer (source has none); its bound at N = 200 is 0.46, below
   LF's 0.48, so a GNN-vs-LF accuracy gap is partly representational. slateN on DS-0019 is the fairer
   comparison.
3. `actions_to_pixels` is 0.5 px off on the FleX grid (world 0 -> 31.5, grid/masks put it at 32);
   consistent between LF fit and predict and in every scorer; not corrected (CODEMAP trap).
4. `SingleLinearForesightPredictor` failed on `--device cuda` (operator left on CPU) -- fixed
   (`.to(batch.occ0.device)`), affects any earlier attempt to score `linear_single_*` on cuda.
5. No noise floor across FPS seeds (node sampling is fixed per state); the depth CIs are over rows only.
