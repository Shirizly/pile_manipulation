# EXP-0062 / RUN-0001 -- NFD seed 0 trained on DS-0020 v2, tested on DS-0019, timed (2026-10-02, 14:20-15:50 CEST)

- commit d72bb304, **dirty** (EXP-0061's uncommitted files, plus in this run: `training/trainer.py`
  `patience_min_rel_delta` knob, new `Baselines/NFD/configs/nfd_3ch_flex_mask_v2.yaml`, `docs/CODEMAP.md`,
  this experiment's `code/`); python `/home/alon/anaconda3/envs/pme/bin/python -u`, OMP_NUM_THREADS=4.
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB, 35 W cap); `nvidia-smi` before launch: no other compute
  process (15 MiB used). Nobody else used the GPU during training (checked at epoch ~110: only this PID).
  Two short dev checks of the scoring/timing code ran concurrently with training epochs 4-10 (slowed those
  epochs; no other effect).
- data: DS-0020 **v2** (train 15,093 / val 1,665 kept rows, trajectories 0-99 excluded), DS-0019 (100
  slates, 16,583 kept rows). Model input AND truth = binary 64x64 top-down image masks (`occ_source=image_mask`).
- model: MODEL-0008 (`weights/MODEL-0008-nfd-flex-mask-v2-seed0/`). Reference: MODEL-0005 (v1-trained NFD seed 0).
- exact argv: `COMMAND_{train,eval_report,score,analyze,timing}.txt` (this dir) and `experiments/COMMANDS.jsonl`
  run ids `EXP-0062/RUN-0001/*`. Logs: `../../logs/`.

## Training

Config `Baselines/NFD/configs/nfd_3ch_flex_mask_v2.yaml` = `nfd_3ch_flex_mask.yaml` with (header lists them):
dataset -> `configs/dataset/flex_ds0020v2_train_ds0019_test.yaml`; epochs cap 250; plateau stop patience 20
with `patience_min_rel_delta 0.005` (new opt-in trainer knob: the early-stop counter resets only on a > 0.5 %
relative val improvement over the val loss at the last reset; `unet_best.pth` still saved on any new best;
default 0 = the old behaviour, bit-identical logic); StepLR step 1000 (never fires: constant LR 1e-4).
Full-state checkpointing on (`save_full_state: true`, `last_state.pt` atomic each epoch + every 10 min;
confirmed written after epoch 1). Never interrupted.

- **Epochs run: 129 (plateau stop), best epoch 128, best val MSE 0.010244**; wall 80 min (~37 s/epoch).
- Convergence (`figures/nfd_v2_seed0_convergence.png`, `results/nfd_v2_seed0_curve.json`): best-val
  improvement over the trailing 20 epochs 3.6 % at ep 40, 1.3 % at 60, 0.87 % at 80-100, 0.55 % at 119,
  **0.30-0.39 % at the stop** -- below the 0.5 % rule, comparable to the epoch-to-epoch val sd (0.5 % over
  the last 20 epochs), but still drifting down: near-converged, not flat. Train 0.01003, val 0.01024 (no
  overfitting gap). Trainer test pass on DS-0019 (whole-grid, binary truth): prob_mse 0.0150, hard_iou 0.807.

## DS-0019 test (binary image-mask truth, `--truth-scoring image`)

Truth verified again: `_capture_report(cell, pred, "default", None)` scores against `cell.occ1`, which for
corpus `flex_ds0019_mask` is `ImageMaskSource` -> `image_masks.npz['after']`; asserted in `eval_nfd_v2.py`
that `cell.occ1` takes exactly the values {0, 1}. `eval_report.py` CLI (official row,
`results/ds0019_eval_report_nfd_v2_s0.json`) and `code/eval_nfd_v2.py` (per-row/per-slate, same functions)
give identical numbers (accuracy 0.5490; averaged slateN 0.9610 / 0.9249 / 0.9330).

**Reuse check:** EXP-0061's per-slate file (`results/final_eval/ds0019.json` + `_rows.npz`, row `nfd_s0`) is
the same code path and flags (`final_eval.py`: `flex_ds0019_mask`, `_capture_report(..., None)`, same
`region_of`/`row_rms`); MODEL-0005 re-scored here reproduces it exactly (max |per-slate diff| 0, max
|row rms diff| 0, accuracy 0.4895), and the row order and persistence errors are asserted equal. So the
paired comparison uses EXP-0061's file.

slateN, mean of 3 goals [slate-bootstrap 95 % CI] (100 slates):

| model | lyapunov | mass_in_region | signed_mass | all 9 cells |
|---|---|---|---|---|
| **NFD v2 seed 0 (MODEL-0008)** | **0.961** [0.953, 0.968] | **0.925** [0.914, 0.935] | **0.933** [0.922, 0.944] | 0.940 |
| NFD v1 seed 0 (MODEL-0005) | 0.948 [0.937, 0.958] | 0.878 [0.861, 0.893] | 0.896 [0.878, 0.913] | 0.907 |
| random | 0.006 | 0.002 | 0.003 | 0.004 |
| persistence (DEGENERATE ranker) | 0.123 | 0.011 | 0.093 | 0.076 |

Per goal x vf (v2 / v1): rq 0.948/0.926, 0.875/0.832, 0.922/0.898; ring_O 0.957/0.952, 0.937/0.871,
0.930/0.885; T 0.977/0.967, 0.963/0.929, 0.947/0.906 (lyapunov, mass_in_region, signed_mass). Every v2 cell
>= the v1 cell. Goal degeneracy frac(dv_true == 0): lyapunov <= 0.001, signed_mass 0.006-0.008,
mass_in_region 0.047-0.187 (worst random_quadrant); no flat slate.

**Paired slateN v2 - v1** (`paired_stats.paired_comparison`, per-slate mean of cells, 10k bootstrap):
all 9 cells **+0.032 [+0.022, +0.043]**, 68/31 slates, Holm p < 1e-4; lyapunov +0.013 [+0.004, +0.023]
(p 0.011); mass_in_region +0.047 [+0.030, +0.065]; signed_mass +0.037 [+0.020, +0.054]; per goal rq +0.030,
ring_O +0.039, T +0.028 (all CIs exclude 0). Noise reference: EXP-0061's v1 3-seed spread is sd
0.002-0.004 per vf; v2 has ONE seed, so its own seed noise is not measured here.

**Accuracy** (swept region, ratio of population means): DS-0019 **0.549 [0.536, 0.562]** (slate-cluster) vs
v1 0.4895 [0.471, 0.508]; paired delta **+0.060 [+0.052, +0.067]**. DS-0020 **v2** val (1,665 rows,
trajectory-cluster CI): v2 **0.505 [0.496, 0.514]**, v1 0.429 [0.419, 0.439]; persistence 0. (v2's DS-0019
accuracy is higher than its own val accuracy; cause not isolated -- DS-0019's longer pushes are one
untested candidate, DATASET.md push-length table.)

**Strata** (slateN = per-slate mean of 9 cells; delta = v2 - v1 [slate-bootstrap CI]; accuracy within stratum):

| stratum (slates) | slateN v2 | slateN v1 | delta | acc v2 | acc v1 | acc delta CI |
|---|---|---|---|---|---|---|
| rand_blob (50) | 0.942 | 0.906 | +0.036 [+0.023, +0.051] | 0.535 | 0.467 | [+0.063, +0.072] |
| rand_spread (50) | 0.937 | 0.909 | +0.028 [+0.012, +0.044] | 0.561 | 0.508 | [+0.040, +0.065] |
| pieces small 46-106 (21) | 0.931 | 0.888 | +0.044 [+0.020, +0.070] | 0.482 | 0.425 | [+0.052, +0.063] |
| pieces mid 190-298 (39) | 0.945 | 0.918 | +0.026 [+0.011, +0.043] | 0.559 | 0.489 | [+0.062, +0.079] |
| pieces large 430-766 (40) | 0.939 | 0.907 | +0.032 [+0.017, +0.048] | 0.570 | 0.519 | [+0.037, +0.066] |

(Tercile edges as EXP-0061: 190 / 430 pieces -> 21/39/40 slates.) v2 > v1 in every stratum, both metrics.

Outputs: `results/nfd_seed0_ds0019.{json,md}` (analysis), `results/nfd_seed0_ds0019_raw.json` +
`_raw_rows.npz` (per-slate captures, per-row rms), `results/ds0020v2_val_rows.npz`,
`results/ds0019_eval_report_nfd_v2_s0.json`.

## Inference timing (`code/time_inference.py`, `results/timing_nfd.json`)

GPU idle (contention check passed in all 3 workers), RTX 4070 Laptop, torch on cuda:0; tensors fed to
the UNet forward asserted on `cuda:0`. Each slate's full pool (86-191 candidates, median 168) as ONE
`predict_occ` batch; 2 warm-up passes excluded; 3 timed passes, per-slate median; 3 separate processes.
Timed region = two `draw_plate_soft` plate channels + stack + UNet forward + sigmoid + `.cpu()` of the
(B, 64, 64) mask, inputs already on GPU.

| quantity | process 0 / 1 / 2 | median | cross-process spread |
|---|---|---|---|
| ms per slate, median | 4.83 / 4.88 / 4.90 | **4.88** | 1.4 % |
| ms per slate, p90 | 5.30 / 5.28 / 5.33 | 5.30 | 1.0 % |
| us per candidate (slate batches) | 28.9 / 29.1 / 29.3 | **29.1** | 1.3 % |
| total over all 100 slates (s) | 0.479 / 0.483 / 0.486 | 0.483 | 1.3 % |
| same incl. host->device copy, ms per slate | 5.57 / 5.55 / 5.54 | 5.55 | 0.6 % |
| fixed batch 128: ms per batch / us per candidate | 3.79-3.81 / 29.6-29.8 | 3.81 / 29.8 | 0.6 % |
| PNG -> binary mask per image (CPU): imread + segment/warp/threshold | 3.25 + 7.0-7.1 | **10.3 ms** | 1 % |

**No per-sample Python loop:** aten-op count per call identical at B=16 and B=128 (ratio 1.00), CUDA
kernel count ratio 1.05. Where the time goes (`code/profile_nfd_breakdown.py` at B=168, and the profiler's top
ops at B=128): UNet forward + sigmoid 4.38 ms (~92 %; cuDNN convolutions are the top op), plate rendering
+ stack 0.43 ms, device->host copy 0.34 ms. Forward cost is close to linear in B above ~128 (2.96 ms at
128, 16.5 ms at 512), i.e. the network itself, not overhead, dominates. bf16 autocast was slower
(6.1 ms). The shared perception step (one PNG -> mask per MPC step, CPU) costs ~2x a whole slate's
model call.

## Status

complete. Not done here (out of scope): seeds 1-2 for v2 (so v2's own seed noise is unmeasured),
visual-foresight and GNN reruns (next agents; reuse `code/time_inference.py`).
