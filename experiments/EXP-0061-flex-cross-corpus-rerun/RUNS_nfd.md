# EXP-0061 -- NFD on FleX binary image masks: run log

> **Data-location note (2026-10-02).** Every "DS-0020" in this record is DS-0020 **v1** (2000
> trajectories from ONE uniform-spread start state). The user then replaced DS-0020's payload with a new
> blob/spread collection (v2). v1 was archived, not deleted: raw `datasets/DS-0020-training-data-flex-N864/old_data/<traj>/`,
> config/splits/cache (incl. `image_masks.npz`, `image_paths.json`) `old_data/_ported_v1/`; read it with
> `old_data/_ported_v1/config.yaml` (the code/ scripts and `configs/dataset/flex_ds0020_train_ds0019_test.yaml`
> were repointed there; logs and result files are left as written). The numbers here are unaffected.
> v2 data check: `figures/data_check_v2/`, `code/data_check_v2.py`, `code/leakage_scan_v2.py`.

## 2026-10-01 -- image-mask cache, NFD x 3 seeds, sanity scores (training agent)

### 1. Image-mask cache (built first; other agents read it)

Owner moved to `FlexData/image_mask.py` (prototype `code/image_mask/image_mask.py` kept as the
validated record, with a pointer note). Colour PNG -> segment (any channel != 255) -> derived camera
(`FlexData/cam_params.json`, f = 869.12, cx = cy = 359.5) -> table-plane (y = 0) back-projection ->
per-cell area fraction on the instance grid (+-7.2, 64 px, row = X = x_flex, col = Y = -z_flex);
binary mask = frac > 0. Depth PNGs NOT used (parity with DS-0019, which has none).

    python -u -m FlexData.image_mask ds0019 --workers 14     # 45 s: 100 init + 17,186 after-states
    python -u -m FlexData.image_mask ds0020 --workers 14     # 69 s: 2000 x 11 = 22,000 states

Output per dataset: `cache/image_masks.npz` (uint8 mask + float16 `frac`; keys in the module
docstring), `cache/image_masks_parts/` (atomic per chunk/state), `cache/image_masks_manifest.json`,
`cache/image_masks.DONE` (counts, grid, camera, sha256 of image_mask.py = 6a243bb8...).
`FlexData.dataset.ImageMaskSource` now materialises the npz arrays once (indexing an NpzFile re-read
the whole array per item).

Validation (`code/image_mask_cache_check.py` -> `figures/data_check/image_mask_check.json`,
`image_mask_triples_DS-00{19,20}.png`):

| check | DS-0020 | DS-0019 |
|---|---|---|
| IoU(cache mask, particle disk raster), same 200 states as the prototype | 0.946 (min 0.919) | 0.946 (min 0.852) |
| max abs diff vs prototype IoU | 0.0 | 0.0 |
| loader frame: IoU(mask occ0, particle occ0) shift scan -3..3 px, peak at | (0, 0), 0.946 | (0, 0), 0.956 |
| removed mask pixels inside the swept region, actions as stored (Y = -z) | 0.999 | 0.998 |
| same, 2nd/4th action components negated (Y = +z) | 0.375 | 0.537 |
| mean occupied fraction mask / particle raster | 0.429 / 0.439 | 0.120 / 0.123 |

The -z action convention holds in the image frame; the mask cache agrees with the loader's frame.

### 2. Resumable training (user request) -- `training/trainer.py`

`training.save_full_state: true` -> `<log_dir>/last_state.pt` (model, optimizer, scheduler,
GradScaler, epoch, batches done, partial epoch sums, best-val bookkeeping, RNG states, sha1 of the
train/val row identity), atomic, every epoch end + every `full_state_every_min` (10) min; train order
from `_EpochPermSampler(shuffle_seed + epoch)` so a resume continues the same epoch at the same
batch. `Baselines/NFD/train_nfd.py --resume` (overrides are now applied BEFORE the resume lookup).
Weight checkpoints (`unet_best/last/epoch_*.pth`) are written atomically too.

Kill test (scratchpad, 3 epochs, `full_state_every_min 0.1`): kill -9 mid-epoch 2 (state at epoch 1
+ 499 of 4,460 batches) -> `--resume` printed "epoch 1 (+499 batches), best=1", row fingerprints
matched (same val split), epoch counter continued 2 -> 3, LR 1e-4. Resumed epoch-2 train/val loss
0.015713 / 0.012158 vs an uninterrupted same-seed run 0.015699 / 0.012183 -- equal within the GPU
nondeterminism floor (same-seed uninterrupted epoch 1 differs by the same ~2e-3 relative:
0.084107 / 0.084128 / 0.084136). The production runs were themselves killed and resumed twice
(epochs 3 and 8, below) with no visible discontinuity in the loss curve.

### 3. NFD training, seeds 0/1/2 (concurrent; ~214 MiB GPU each)

Config `Baselines/NFD/configs/nfd_3ch_flex_mask.yaml` (header lists every deviation from
`nfd_train_3ch_randlen.yaml`: flex dataset + image-mask occ, DS-0020 splits.json, no
min_push_length_m, epochs, StepLR step_size/patience kept non-firing, full-state checkpointing).

    PYTHONPATH=. python -u Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_3ch_flex_mask.yaml \
        --seed $s --override training.shuffle_seed=$s output.log_dir=Baselines/NFD/runs/nfd_3ch_flex_mask_seed$s
    # (+ --resume after each deliberate kill)   logs: logs/nfd_flex_mask_seed{0,1,2}.log

**Epochs: 80, not the presentation-matched 150.** Launched 08:24 at 150; with 3 seeds plus another
agent's job on the same GPU (at its 45 W cap) epochs took 145-175 s (50 s alone) -> 6-7 h. Cut to 100
at epoch 3, then 80 at epoch 8, via kill + `--resume` (LR constant, no early stopping, so only the stop
point moved). 80 epochs = 1.43M row presentations = 53 % of `nfd_randlen`'s 2.67M. All 3 finished
10:53-10:55 (wall ~2.5 h). **Not converged**: val loss still falling ~1 % per 10 epochs at epoch 80
(curves `figures/nfd/val_curves.png`, `results/nfd_val_curves.json`).

| seed | weights | best epoch | best val loss (MSE) | DS-0020 val swept `accuracy` | Trainer test pass hard_iou (DS-0019) |
|---|---|---|---|---|---|
| 0 | MODEL-0005 | 80 | 0.008696 | 0.6095 | 0.758 |
| 1 | MODEL-0006 | 80 | 0.008653 | 0.6117 | 0.759 |
| 2 | MODEL-0007 | 80 | 0.008708 | 0.6098 | 0.775 |
| mean +- sd | | | 0.008686 +- 0.000029 | **0.610 +- 0.001** | |
| persistence | | | 0.04915 (whole-grid MSE) | 0 | |

(`code/score_nfd_flex.py ds0020_val` -> `results/nfd_val_accuracy.json`; plate 2.4 units = 10.67 px;
truth = binary mask.) NFD beats persistence by a wide margin on val; the seed spread (sd 0.001) is
~600x smaller than the gap to persistence. Seed 2 started from a much worse init (epoch-1 val 0.148 vs
0.024/0.020) and caught up by epoch ~12.

### 4. DS-0019 test (user request: seed 0 first, then 1-2) -- binary image-mask truth

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/score_nfd_flex.py ds0019 \
        --ckpt Baselines/NFD/runs/nfd_3ch_flex_mask_seed$s/unet_best.pth --name nfd_flex_mask_seed$s \
        --out experiments/EXP-0061-flex-cross-corpus-rerun/results/nfd_seed${s}_ds0019.json   (+ .md table)

Composes `eval_report`'s own functions on corpus `flex_ds0019_mask` (occ0 = mask) with
`_capture_report(truth_s0=None)` -> truth = `cell.occ1` = the binary mask of the true after-state. This
is what `eval_report.py --corpora flex_ds0019_mask --truth-scoring image` does; **the command in
RUNS_gnn_lf.md omits `--truth-scoring image`, so it would score slateN against the soft particle splat
(the default), not the user-decided mask truth** -- the final-eval agent should pass it. No new
eval_report option was needed, so eval_report.py was not edited here.

slateN averaged over the 3 goals (100 slates; per-slate sem 0.008-0.018 per goal x vf):

| row | lyapunov | mass_in_region | signed_mass | accuracy |
|---|---|---|---|---|
| seed 0 | 0.948 | 0.878 | 0.896 | 0.4895 |
| seed 1 | 0.955 | 0.874 | 0.902 | 0.4809 |
| seed 2 | 0.953 | 0.878 | 0.904 | 0.4842 |
| mean +- sd | 0.952 +- 0.003 | 0.876 +- 0.002 | 0.901 +- 0.004 | 0.485 +- 0.004 |
| random | 0.006 | 0.002 | 0.003 | n/a |
| persistence (DEGENERATE slateN) | 0.123 | 0.011 | 0.093 | 0 |

Goal degeneracy on mask truth: frac(dv_true == 0) <= 0.001 (lyapunov), 0.006-0.008 (signed_mass),
0.047-0.187 (mass_in_region; worst random_quadrant); no slate has a flat pool. Per-goal tables,
soft-particle-truth comparison (seed 0: 0.961 / 0.849 / 0.905) and predictor parity (max |diff|
<1e-4) in `results/nfd_seed{s}_ds0019.md`. Caveats: slateN near ceiling on lyapunov (0.95) leaves
little headroom to separate models; pool sizes 86-191 are not comparable to Genesis corpora; known
DS-0020 -> DS-0019 distribution shift (DATASET.md).

### Side note (pre-existing, not fixed)

`fit_linear_foresight.actions_to_pixels` maps world -> pixel as `x * to_pxl + 31.5` while
`FlexPileData` (occupancy, plate channels, image masks) uses `x * to_pxl + 32`: the swept-region
mask used by `accuracy` is offset 0.5 px from the image frame on FleX. Small vs the 10.67 px plate,
but it is a frame inconsistency in the metric region.
