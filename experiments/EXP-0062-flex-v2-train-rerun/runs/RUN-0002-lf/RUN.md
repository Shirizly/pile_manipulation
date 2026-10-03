# EXP-0062 / RUN-0002 -- switched linear visual foresight fit on DS-0020 v2, tested on DS-0019, timed (2026-10-02, 15:48-16:15 CEST)

- commit d72bb304, **dirty** (EXP-0061/RUN-0001 uncommitted files, plus in this run: new
  `code/{fit_lf_v2,eval_lf_v2,profile_lf_breakdown,aggregate_timing_lf}.py`; `code/time_inference.py` now also
  records + asserts the OUTPUT device for predictors with no `nn.Module` (LF has none, so its forward hooks
  saw nothing); **`Baselines/LinearForesight/predictor.py` operator device cache (below)**; `.gitignore`
  `weights/*/*/checkpoint.*`; `docs/CODEMAP.md`). python `/home/alon/anaconda3/envs/pme/bin/python -u`,
  OMP_NUM_THREADS=4. GPU RTX 4070 Laptop (8 GB), idle (15 MiB) before every GPU step.
- data: DS-0020 **v2** (train 15,093 rows / 1,669 trajectories, val 1,665 / 188, trajectories 0-99 excluded;
  disjointness and the exclusion asserted in `fit_lf_v2.py`), DS-0019 (100 slates, 16,583 kept rows). Input AND
  truth = binary 64x64 image masks (`occ_source=image_mask`; truth values asserted {0,1}).
- model: **MODEL-0009** (`weights/MODEL-0009-linear-foresight-flex-mask-v2/`), alt bundle `alt_collection_bins/`.
  References: MODEL-0004 (v1 LF, EXP-0061), MODEL-0008 (v2 NFD, RUN-0001).
- argv: `COMMAND_{fit,score,analyze,eval_report,timing}.txt` (this dir), `experiments/COMMANDS.jsonl` run ids
  `EXP-0062/RUN-0002/*`. Logs `../../logs/{fit_lf_v2*,score_lf_ds0019_val,analyze_lf,eval_report_lf_ds0019,timing_lf_*}.log`.

## Fit (`code/fit_lf_v2.py`)

= EXP-0061 `fit_lf_flex.py` recipe (ridge toward identity, res 64, D = 4096, MIN_ROWS_PER_BIN 50), v2 data,
(bin scheme x lambda) chosen on v2 val (whole trajectories held out -- no row-random CV anywhere). Sweep via
per-bin Gram matrices; final operators re-fit with `Baselines/LinearForesight/fit_switched.py::fit_bins`
(max |diff| vs Gram solve 9e-6). Lambda grid 1-10000 (added 10000 vs v1).

| scheme | edges | train rows/bin | val rows/bin | DS-0019 rows/bin |
|---|---|---|---|---|
| **(a) equal-width [0, 9.41] (chosen)** | 0 / 1.57 / 3.14 / 4.71 / 6.27 / 7.84 / 9.41 | 1722 / 3973 / 4530 / 3435 / 1323 / **110** | 177 / 441 / 472 / 416 / 143 / **16** | 895 / 3205 / 3153 / 3132 / 3259 / 2939 |
| (b) collection `edges_abs` (ends merged) | 0.96 / 2.40 / 3.85 / 5.29 / 6.73 / 8.17 / 9.62 | 3712 / 4057 / 3940 / 2630 / 692 / **62** | 409 / 431 / 399 / 338 / 79 / **9** | 2688 / 2754 / 2707 / 2817 / 2867 / 2750 |

**Flags:** no bin is below MIN_ROWS_PER_BIN = 50, but every bin is underdetermined (M < D = 4096: M/D 0.03-1.1),
which is why lambda is large. The last bin is badly under-filled in both schemes (110 / 62 train rows, M/D 0.027 /
0.015) while DS-0019 puts **2,939 / 2,750 rows (18 %) in it**; scheme (b) bin 4 also thin (692). Val numbers for
bin 5 rest on 16 / 9 rows.

Val accuracy (swept region, ratio of population means), lambda sweep 1/10/30/100/300/1000/3000/10000:
equal 0.267/0.348/0.372/0.387/**0.390**/0.384/0.370/0.346; collection 0.260/0.340/0.365/0.380/**0.383**/0.377/0.364/0.339;
single 0.244/0.261/0.271/0.280/0.285/**0.285**/0.281/0.270. Chosen: **equal, lambda 300 -> val 0.3903
[0.382, 0.398]** (trajectory-cluster CI); single lambda 1000 -> **0.2852 [0.277, 0.293]**. Equal - collection on val
+0.0070 [+0.0043, +0.0097] (paired, trajectory bootstrap) -- small but resolved; picked on val only. Per-bin val
(equal, lambda 300): 0.20 / 0.36 / 0.41 / 0.43 / 0.44 / 0.37; single (1000): -0.17 / 0.28 / 0.36 / 0.34 / 0.30 / 0.23.
Selected on the same val split it is reported on (mildly optimistic; curve flat within 0.006 over 100-1000).
For reference on the same v2 val rows: v1 LF (MODEL-0004) switched 0.267, single 0.163; NFD v2 (MODEL-0008) 0.505.

## DS-0019 test (binary image-mask truth, `--truth-scoring image`)

Truth = `cell.occ1` of corpus `flex_ds0019_mask` = `ImageMaskSource` after-mask, asserted values {0, 1}. Same
code path/flags as RUN-0001 and EXP-0061 `final_eval.py` (`_accuracy(..., "cuda")`, `_capture_report(cell, pred,
"default", None)`, `region_of`/`row_rms`). Official CLI row `results/ds0019_eval_report_lf_v2.json`
(`eval_report.py --ckpt lf_flex_{switched,single}=MODEL-0009 --truth-scoring image`) gives identical numbers.
**Reuse check:** MODEL-0004 re-scored here reproduces EXP-0061's per-slate file exactly (switched and single: max
|per-slate diff| 0, max |row rms diff| 0, accuracy 0.3284 / 0.2394), row order + persistence asserted equal -> the
paired comparison uses EXP-0061's file; MODEL-0008 from RUN-0001's per-slate file (same row order asserted).

slateN, mean of 3 goals [slate-bootstrap 95 % CI]:

| model | lyapunov | mass_in_region | signed_mass | all 9 |
|---|---|---|---|---|
| **LF v2 switched (MODEL-0009)** | **0.935** [0.925, 0.946] | **0.884** [0.867, 0.900] | **0.897** [0.882, 0.912] | **0.906** [0.894, 0.916] |
| LF v2 single (MODEL-0009) | 0.676 [0.642, 0.709] | 0.683 [0.650, 0.717] | 0.674 [0.644, 0.703] | 0.678 |
| LF v2 switched, collection bins (alt) | 0.934 | 0.886 | 0.876 | 0.899 |
| LF v1 switched (MODEL-0004) | 0.843 | 0.711 | 0.626 | 0.726 |
| LF v1 single (MODEL-0004) | 0.677 | 0.539 | 0.516 | 0.578 |
| NFD v2 (MODEL-0008) | 0.961 | 0.925 | 0.933 | 0.940 |
| random | 0.006 | 0.002 | 0.003 | 0.004 |
| persistence (DEGENERATE ranker) | 0.123 | 0.011 | 0.093 | 0.076 |

Per goal (lyapunov): switched v2 rq 0.925 / ring_O 0.926 / T 0.955; full 9-cell table in `results/lf_ds0019.md`.
Goal degeneracy as RUN-0001 (same rows/truth).

**Paired slateN** (`paired_stats.paired_comparison`, per-slate mean of 9 cells, 10k bootstrap, Holm over 10 pairs):
- LF v2 sw - LF v1 sw **+0.179 [+0.156, +0.203]**, 93/7 slates; every vf and goal positive (lyapunov +0.093, mass +0.174, signed +0.271; rq +0.295, ring_O +0.121, T +0.122).
- LF v2 single - LF v1 single +0.100 [+0.076, +0.125], 77/23 (lyapunov -0.002 [-0.038, +0.035] unresolved; ring_O -0.048 [-0.096, +0.001] unresolved).
- LF v2 sw - LF v2 single **+0.228 [+0.205, +0.252]**, 99/1.
- LF v2 sw - NFD v2 **-0.034 [-0.044, -0.025]**, 21/79 (every vf and goal negative, all CIs exclude 0).
- equal - collection bins on DS-0019 +0.007 [-0.002, +0.016], 57/41, unresolved.

**Accuracy** (DS-0019, slate-cluster CI): LF v2 sw **0.469 [0.455, 0.482]**, single 0.305 [0.295, 0.314], collection
bins 0.460, LF v1 sw 0.328, v1 single 0.239, NFD v2 0.549; persistence 0. Paired: v2 sw - v1 sw +0.141 [+0.136,
+0.146]; v2 sw - v2 single +0.164; v2 sw - NFD v2 -0.080 [-0.089, -0.071]. DS-0019 accuracy > v2 val accuracy
(0.469 vs 0.390), as for NFD; per-bin DS-0019 accuracy rises with push length (0.26 bin 0 -> 0.53 bin 4, 0.49 bin
5), and DS-0019's pushes are longer -- consistent with, not proof of, the length-mix explanation.

**Strata** (slateN all 9 cells; delta [slate CI]):

| stratum (slates) | LF v2 sw / single / LF v1 sw / NFD v2 | v2sw - v1sw | v2sw - v2single | v2sw - NFD v2 |
|---|---|---|---|---|
| rand_blob (50) | 0.900 / 0.732 / 0.807 / 0.942 | +0.092 [+0.069, +0.117] | +0.168 | -0.042 [-0.057, -0.029] |
| rand_spread (50) | 0.912 / 0.624 / 0.645 / 0.937 | +0.266 [+0.246, +0.287] | +0.288 | -0.025 [-0.038, -0.013] |
| pieces small 46-106 (21) | 0.899 / 0.715 / 0.812 / 0.931 | +0.087 [+0.052, +0.125] | +0.184 | -0.032 [-0.057, -0.008] |
| pieces mid 190-298 (39) | 0.905 / 0.693 / 0.745 / 0.945 | +0.160 [+0.124, +0.198] | +0.212 | -0.040 [-0.054, -0.027] |
| pieces large 430-766 (40) | 0.910 / 0.643 / 0.664 / 0.939 | +0.246 [+0.217, +0.277] | +0.267 | -0.029 [-0.044, -0.015] |

v1 LF was weakest on spread/large piles (it was fit on one uniform-spread start state, yet scores worst there --
cause not isolated); v2 closes most of that. NFD v2 beats LF v2 in every stratum.

## Timing

`code/time_inference.py` (same settings as RUN-0001: every DS-0019 slate's full pool, 86-191 candidates, as ONE
`predict_occ` batch, inputs pre-moved; + with-H2D variant; + fixed batch 128; 2 warm-up passes excluded, 3 timed
passes, per-slate median; cuda sync around every timed region; 3 separate processes; GPU idle, contention check
passed in every worker). LF has no `nn.Module`, so the forward-hook device probe sees nothing; the harness now also
asserts the **output** device (`cuda:0` / `cpu` as requested -- `predict_world` runs on `occ0`'s device).
`results/timing_lf.json` (merged by `code/aggregate_timing_lf.py`), per-config files in `results/timing_lf_parts/`.

**Bug found and fixed:** `LinearForesight/predictor.py` kept the operators on the CPU and `predict_switched`
copies each bin's 64 MB operator to the GPU on every call (`A.to(occ.device)`, 6 per call). That was **78 %** of the
CUDA time (45.9 ms/slate as-is vs 10.4 ms resident). The predictors now cache device copies (`_ops_on`);
predictions bit-identical (eval_report CLI JSON identical before/after). Any earlier LF CUDA timing is inflated
by this (EXP-0061 did not time LF).

| config (3 processes) | ms / slate median (p90) | us / candidate | cross-proc spread | incl. H2D ms / slate | batch 128: ms / us per cand |
|---|---|---|---|---|---|
| **switched, CUDA (headline)** | **10.4** (11.3) | **63.7** | 2.2 % | 12.7 | 9.67 / 74.5 |
| switched, CUDA, before fix | 45.9 (46.5) | 277 | 0.1 % | 48.1 | 45.3 / 349 |
| single, CUDA | 4.23 (4.52) | 25.2 | 0.7 % | 4.85 | 3.18 / 24.8 |
| switched, CPU (4 threads) | 55.7 (59.9) | 333 | 1.5 % | 55.9 | 47.0 / 366 |
| single, CPU (4 threads) | 39.7 (43.8) | 237 | 1.7 % | 40.2 | 29.6 / 233 |
| *ref: NFD v2 CUDA (RUN-0001)* | *4.88 (5.30)* | *29.1* | | *5.55* | *3.81 / 29.8* |

Headline = CUDA (the model is used on the GPU in MPC; CPU is 5x slower). Switched LF is **~2.1x slower than NFD**
per slate on CUDA after the fix.

**Per-sample-loop check:** no per-SAMPLE loop. Single: aten-op / kernel count ratio B=128 vs B=16 = 1.00. Switched:
2.84 -- this is the per-BIN loop in `predict_switched` (B = 16 occupies fewer of the 6 bins, so fewer passes);
the harness's flag fires but is not a per-sample loop. Per call the switched path pays one `bool(m.any())` host
sync per bin (6) plus boolean-mask indexing syncs.

**Where the time goes** (`code/profile_lf_breakdown.py`, the same stages as `predict_switched`, sync after every
stage -- inflates the CUDA total to 12.0 ms vs 10.2 ms; staged output asserted == `predict_occ`), ms per slate:

| stage | CUDA | share | CPU | share |
|---|---|---|---|---|
| push-frame warp (`to_push_frame`) | 1.88 | 16 % | 3.18 | 6 % |
| operator matmul (4096 x 4096 @ x) | 2.07 | 17 % | 36.9 | 67 % |
| unwarp (`from_push_frame`) | 2.62 | 22 % | 4.83 | 9 % |
| validity mask + blend + clamp | 4.10 | 34 % | 8.43 | 15 % |
| per-bin mask / `.any()` sync | 0.78 | 6 % | 0.64 | 1 % |
| scatter into output | 0.38 | 3 % | 0.46 | 1 % |
| pixels + bin index | 0.22 | 2 % | 0.39 | 1 % |

On CUDA the warp / unwarp / validity-blend round trip (72 %) dominates, not the operator: it is run once PER
OCCUPIED BIN on a small sub-batch (~28 rows), so it is launch-bound -- single LF (one round trip per call) takes
4.2 ms. A warp-once implementation (warp the whole pool once, apply the per-row bin operator, unwarp once -- as
`predict_switched_soft` already does for the soft gate) would plausibly bring switched close to single; not
implemented or measured here. On CPU the 4096^2 matmuls dominate (67 %).

## Outputs

`results/lf_ds0019.{json,md}` (analysis), `results/lf_ds0019_raw.json` + `_raw_rows.npz` (captures, per-row rms),
`results/lf_ds0020v2_val_rows.npz`, `results/ds0019_eval_report_lf_v2.json` (official CLI row),
`results/timing_lf.json` (+ `results/timing_lf_parts/`), `weights/MODEL-0009-*/fit.json` (sweep).

## Status

complete. Not done (out of scope): nonneg-constrained operators (the paper's best variant), res 32, a
warp-once switched implementation (cf. `predict_switched_soft`), GNN (next agent).
