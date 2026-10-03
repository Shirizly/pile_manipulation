---
# ---- identity -------------------------------------------------------------
id: EXP-0062
title: Trained on DS-0020 v2 (blob/spread piles) and tested on DS-0019, slateN ranks NFD > switched LF > GNN (paired, per slate), each v2 model beats its v1 / original counterpart, and on CUDA NFD is also the fastest (4.9 vs 10.4 vs 77 ms per slate)
tier: T1
mode: exploratory
date: 2026-10-03
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  For three model families, ONE training seed each -- NFD (MODEL-0008), switched
  linear visual foresight (MODEL-0009; single-operator variant as a reference row)
  and the dyn-res GNN architecture trained from scratch at a constant 30 nodes on a
  fully visual Chamfer target (MODEL-0010) -- fit on DS-0020 v2 (trajectories 0-99
  excluded, flagged rows dropped) and tested on all 100 DS-0019 same-state slates
  with model input AND ground truth = the binary 64x64 top-down colour-image mask
  (+-7.2 FleX units), goals random_quadrant / ring_O / T x value functions lyapunov /
  mass_in_region / signed_mass: (a) per-slate paired slateN (mean of 9 cells) orders
  NFD > LF switched > GNN with every pairwise 95 % slate-bootstrap CI excluding 0,
  and swept-region accuracy gives the same order; (b) each v2 model's paired slateN
  exceeds its v1 / original counterpart (MODEL-0005 NFD seed 0, MODEL-0004 LF
  switched, the original dyn-res-pile-manip GNN checkpoint at N 200 and at N 30) with
  CI excluding 0; (c) on the RTX 4070 Laptop GPU, ms per slate (whole candidate pool
  as one batch) orders NFD < LF switched < GNN.

# prediction: omitted -- exploratory. No threshold was committed in git before the
# runs; each run agent saw its own model's DS-0019 numbers as they were produced.

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: d72bb304
  dirty: true                     # see "What was actually run" for the file list
  data_commit: "unrecorded (DS-0019/DS-0020 v2 collected by an external FleX project; run_config.json shipped, no collector sha)"
  script: "per family: experiments/EXP-0062-flex-v2-train-rerun/code/{eval_nfd_v2,eval_lf_v2,eval_gnn_v2}.py (compose Baselines/common/eval_report.py functions; official CLI rows from eval_report.py itself), code/time_inference.py (timing), code/combined_summary.py (cross-family table + figure, no new scoring)"
  data: [DS-0019, DS-0020]
  code_path: "eval_report corpus flex_ds0019_mask, --truth-scoring image (_capture_report(truth_s0=None) -> cell.occ1 = FlexData.ImageMaskSource binary after-mask, asserted {0,1}); accuracy = fit_linear_foresight.metrics swept region; GNN masks via Baselines/GNN/flex_predictor.py nearest-node carry (default, non-vectorised path)"
  seed: 0
  split: "train/fit: DS-0020 v2 splits.json train (1707 trajectories, 15,093 kept rows; GNN: 8,289 5-step windows); val: DS-0020 v2 val (190 trajectories, 1,665 kept rows; GNN 911 windows); excluded: trajectories 0-99 (DS-0019 near-copies); test: all 100 DS-0019 slates, 16,583 kept rows"
  runtime: "NFD train 80 min GPU; LF fit+sweep ~15 min; GNN train 82 min CPU (GPU failed mid-run); DS-0019 scoring 3-30 s per pass; RTX 4070 Laptop 8 GB / 4 CPU threads"
  runs: [RUN-0001, RUN-0002, RUN-0003]

budget:
  declared: "per-run budgets in the three run agents' prompts (not recorded in the RUN files); this write-up 1.5 h HARD"
  spent: "runs 2026-10-02 14:20-19:30 CEST (~5 h wall); write-up ~45 min"
  outcome: within

design:
  varied:
    model_family: [NFD (MODEL-0008), LF switched + single (MODEL-0009), GNN N30 visual target (MODEL-0010)]
    training_data: [DS-0020 v2 (this record), DS-0020 v1 / source-repo training (MODEL-0005, MODEL-0004, original GNN ckpt -- EXP-0061 per-slate files reused after exact re-scoring)]
    goal: [random_quadrant, ring_O, T]
    value_function: [lyapunov, mass_in_region, signed_mass]
    device_for_timing: [cuda, cpu]
    sensitivity_rows: [LF collection-edge bins, GNN particle-supervised target, GNN original ckpt at N30, GNN rendering cap at N30]
  held_fixed:
    representation: "binary top-down colour-image mask, 64x64, +-7.2 FleX units, visual-only model inputs (user decision)"
    truth: "binary image mask of the true after-state (user decision); not the particle splat"
    training_seed: "0 for every model; one seed per model (user decision)"
    gnn_nodes: "constant 30 nodes, FPS on the colour-image foreground cloud, plane y 0.24 (user decision; plane checked in RUN-0003)"
    leakage: "DS-0020 v2 trajectories 0-99 dropped from train and val (user decision)"
    flagged_rows: "exclude_flagged=True (drops escaped / out_of_grid / null rows, 9.7 % escaped on v2)"
    slateN_pool: "every kept candidate of the slate (86-191), no replacement"
    lf: "ridge-to-identity, res 64; bin scheme (equal-width) + lambda 300 (single 1000) chosen on v2 val"
    nfd_recipe: "Baselines/NFD/configs/nfd_3ch_flex_mask_v2.yaml, plateau stop (patience 20, 0.5 % rel), stopped at epoch 129"
    gnn_recipe: "Baselines/GNN/flex_train.py, chamfer_carry target, Adam 1e-3, batch 4, rollout 5, plateau stop (patience 10, 0.5 % rel)"
    timing: "code/time_inference.py: whole slate pool as one batch, 2 warm-up + 3 timed passes, per-slate median, 3 processes, idle GPU"
  baselines: [persistence, random]
  metric: "slateN (3 value functions, experiments/METRICS.md), accuracy"

noise_floor: "v2 training-seed noise NOT measured (one seed per model). Only references: EXP-0061 NFD v1 3-seed slateN goal-avg sd 0.002-0.004 per vf (accuracy sd 0.004); EXP-0061 original-GNN FPS-sampling spread 0.005-0.016 slateN. Per-slate paired sd of a cross-family difference 0.069-0.071 (100 slates -> CI half-width ~0.014)."

depends_on: [flex-action-frame-neg-z, goal-mask-axis-convention-row-y-col-x, flex-ds0020v2-test-leak-excluded]
establishes: []

# ---- outcome --------------------------------------------------------------
result: "DS-0019 slateN all-9-cell mean [slate CI]: NFD v2 0.940 [0.932, 0.946], LF switched v2 0.906 [0.894, 0.916], GNN v2 0.850 [0.835, 0.863]; paired NFD-LF +0.034 [+0.025, +0.044] 79/21, LF-GNN +0.056 [+0.043, +0.070] 79/21, NFD-GNN +0.090 [+0.077, +0.104] 97/3, every vf the same sign. Accuracy 0.549 / 0.469 / 0.254 (GNN render cap at N30 0.363). v2 - v1 slateN: NFD +0.032 [+0.022, +0.043], LF +0.179 [+0.156, +0.203], GNN +0.168 [+0.137, +0.204] vs N200 (+0.103 vs N30). CUDA ms/slate: NFD 4.9, LF switched 10.4, GNN 77.2 incl. perception / 43.6 cached; CPU 80.7 / 55.7 / 259 / 207 (LF faster than NFD on CPU); PNG->mask 10.3 ms/image shared."
verdict: supported
downgrades: [imprecision, indirectness, selection, provenance, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

**Lead (slateN first).** On DS-0019, with every model trained on DS-0020 v2 and fed only the
binary colour-image mask, the control ranking is **NFD > switched LF > GNN**, resolved for every
pair and every value function; accuracy agrees; and each v2 model clearly beats the v1 / original
model it replaces. On the GPU the NFD is also the cheapest model to query; on the CPU the switched
LF is. One training seed per model -- the seed noise of these v2 rows is not measured.

Combined table + figure: `results/combined_v2.{md,json}`, `figures/combined_v2.png`
(`code/combined_summary.py`, derived from the per-family files, no new scoring).

## Why this test discriminates

EXP-0061 ranked NFD far above the GNN and LF, but every model there was trained on DS-0020 v1
(one uniform-spread start state) and tested on compact DS-0019 piles, so the ranking could have
been a train/test-shift effect that hits the models unequally (LF v1 was weakest on exactly the
spread/large piles; the original GNN never saw this data). DS-0020 v2 uses DS-0019's pile
generator, piece counts and action sampler, so if the shift had been driving the order, retraining
on v2 should compress or reorder it. It compressed it (NFD-LF from +0.183 to +0.034, NFD-GNN from
+0.227 to +0.090) but did not reorder it: all three pairwise per-slate differences stay resolved,
with the NFD ahead in every stratum (blob/spread, three pile-size terciles), every goal and every
value function. The "do nothing" baselines are far below every model (random 0.004, persistence
0.076 slateN all-9; persistence accuracy 0), so no model is near a degenerate ranker.

## What was actually run

Three runs, each authoritative for its own detail (exact argv in each run dir's `COMMAND*.txt` and
in `experiments/COMMANDS.jsonl`, run ids `EXP-0062/RUN-000{1,2,3}/*`):

- **RUN-0001 (NFD, MODEL-0008).** `train_nfd.py Baselines/NFD/configs/nfd_3ch_flex_mask_v2.yaml
  --seed 0` (dataset `configs/dataset/flex_ds0020v2_train_ds0019_test.yaml`; new opt-in trainer knob
  `patience_min_rel_delta 0.005`), 129 epochs, best epoch 128, GPU. Scored with
  `code/eval_nfd_v2.py score/analyze` and the official `eval_report.py --corpora flex_ds0019_mask
  --models nfd_randlen --ckpt nfd_randlen=MODEL-0008 --truth-scoring image` row (identical numbers).
- **RUN-0002 (LF, MODEL-0009).** `code/fit_lf_v2.py` (EXP-0061's ridge-to-identity recipe on v2;
  bin scheme x lambda swept on v2 val; alt collection-edge bins kept as `alt_collection_bins/`),
  `code/eval_lf_v2.py score/analyze`, official `eval_report.py` row. **Bug fixed here:**
  `Baselines/LinearForesight/predictor.py` re-copied each 64 MB bin operator to the GPU on every
  call (78 % of CUDA time); now cached on device, predictions bit-identical.
- **RUN-0003 (GNN, MODEL-0010).** `python -m Baselines.GNN.flex_train cache --particle-num 30`, then
  `train --target chamfer_carry` (and a `--target particle` sensitivity sibling) as a 25-min pilot
  plus `--resume` continuation, CPU, plateau stop at 23 epochs (best epoch 12). Scored with
  `code/eval_gnn_v2.py` and `eval_report.py --models gnn_flex_v2_n30 --device cpu --truth-scoring image`
  (identical). The GPU failed at ~16:50 ("GPU requires reset"), so GNN training and scoring ran on
  the CPU; CUDA timing was added after the reset.
- **Reuse of EXP-0061 per-slate files** for MODEL-0005 / MODEL-0004 / the original GNN at N 200 was
  done only after re-scoring each model here through the same code path: max |per-slate diff| 0 for
  all three. Row rms reproduced exactly for MODEL-0005 and MODEL-0004; for the original GNN, 18 of
  16,583 rows differ (max 0.0067) because EXP-0061 scored it on GPU and RUN-0003 on CPU. MODEL-0010
  itself was scored on the CPU only; no GPU re-score exists (slateN is expected identical on the
  evidence of the original checkpoint, not shown for MODEL-0010).
- **Timing** (`code/time_inference.py`): every DS-0019 slate's full pool as one `predict_occ` batch,
  inputs pre-moved, 2 warm-up + 3 timed passes, per-slate median, 3 separate processes, contention
  check on an idle GPU. The first GNN CUDA attempt failed the harness's device assertion (the GNN
  calls `predict_one_step`, bypassing the top-level module's `__call__`, so the forward-pre-hook saw
  nothing); the harness now makes one extra untimed call with hooks on every submodule; timed passes
  carry the same hook overhead as NFD/LF. LF has no `nn.Module`, so its output device is asserted.
- **Combined summary** (this write-up): `code/combined_summary.py` reads the three families'
  analysis/timing JSONs; the cross-family pairs are the ones those files computed (NFD-LF from
  `lf_ds0019.json`, both GNN pairs from `gnn_ds0019.json`; Holm families differ by file, every
  cross-family p < 1e-4).
- **Dirty tree** at d72bb304 (nothing committed): EXP-0061's uncommitted files (`FlexData/`,
  `Baselines/GNN/flex_predictor.py`, `Baselines/NFD/configs/nfd_3ch_flex_mask.yaml`,
  `configs/dataset/flex_*.yaml`, `datasets/DS-0019-*/`, `datasets/DS-0020-*/`, modified
  `Baselines/common/eval_report.py`, `Baselines/LinearForesight/{fit_switched,predictor}.py`,
  `fit_linear_foresight.py`, `registry/dataset_registry.py`, `Baselines/NFD/train_nfd.py`,
  `docs/{ARCHITECTURE,CODEMAP,INTERFACES}.md`, `.gitignore`) plus this experiment's:
  `training/trainer.py` (`patience_min_rel_delta`), `Baselines/NFD/configs/nfd_3ch_flex_mask_v2.yaml`,
  `Baselines/LinearForesight/predictor.py` (device cache), new `Baselines/GNN/flex_train.py`,
  `flex_predictor.py` additions (`predict_one_step_vec`, opt-in `fast_graph` / `vector_render` /
  `cache_states`; defaults = EXP-0061 path), `eval_report.py` MODELS `gnn_flex_v2_n30`,
  `gnn_flex_drp_n30`, and every file under `experiments/EXP-0062-*/code/`.
- **Weights / configs:** MODEL-0008 (`weights/MODEL-0008-nfd-flex-mask-v2-seed0/`), MODEL-0009
  (`weights/MODEL-0009-linear-foresight-flex-mask-v2/`), MODEL-0010
  (`weights/MODEL-0010-gnn-flex-mask-v2-n30/`); configs `Baselines/NFD/configs/nfd_3ch_flex_mask_v2.yaml`,
  `configs/dataset/flex_ds0020v2_train_ds0019_test.yaml`, `Baselines/GNN/flex_train.py` (recipe in its
  docstring and RUN-0003).

**User decisions recorded as design, not threats:** visual-only model inputs via the colour-image
mask; binary-mask ground truth; 64x64 grid; one seed per model; GNN fixed at 30 nodes; DS-0020 v2
trajectories 0-99 dropped (they start from near-copies of DS-0019's test piles, mask IoU median 0.906;
piece-level scan found no further near-duplicates, closest 0.154 vs leaked pairs <= 0.107).

## Numbers

**slateN, mean of 3 goals [slate-bootstrap 95 % CI]** (100 slates; from `results/combined_v2.md`):

| model | lyapunov | mass_in_region | signed_mass | all 9 | accuracy [slate CI] (suspect across families) |
|---|---|---|---|---|---|
| **NFD v2 (MODEL-0008)** | **0.961** [0.953, 0.968] | **0.925** [0.914, 0.935] | **0.933** [0.922, 0.944] | **0.940** [0.932, 0.946] | **0.549** [0.536, 0.562] |
| **LF switched v2 (MODEL-0009)** | 0.935 [0.925, 0.946] | 0.884 [0.867, 0.900] | 0.897 [0.882, 0.912] | 0.906 [0.894, 0.916] | 0.469 [0.455, 0.482] |
| **GNN v2 N30 (MODEL-0010)** | 0.900 [0.887, 0.914] | 0.813 [0.786, 0.838] | 0.835 [0.814, 0.856] | 0.850 [0.835, 0.863] | 0.254 [0.240, 0.267] |
| LF single v2 (MODEL-0009) | 0.676 | 0.683 | 0.674 | 0.678 | 0.305 |
| NFD v1 seed 0 (MODEL-0005) | 0.948 | 0.878 | 0.896 | 0.907 | 0.489 |
| LF switched v1 (MODEL-0004) | 0.843 | 0.711 | 0.626 | 0.726 | 0.328 |
| GNN original ckpt N 200 (EXP-0061) | 0.764 | 0.644 | 0.636 | 0.681 | 0.191 |
| GNN original ckpt N 30 | 0.841 | 0.692 | 0.705 | 0.746 | 0.188 |
| GNN rendering cap N 30 (true node motion) | 0.945 | 0.888 | 0.906 | 0.913 | 0.363 |
| random | 0.006 | 0.002 | 0.003 | 0.004 | -- |
| persistence (DEGENERATE ranker) | 0.123 | 0.011 | 0.093 | 0.076 | 0 |

**Paired slateN across families** (per-slate mean of cells, 10k bootstrap; better model first; wins/losses):

| pair | all 9 | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| NFD v2 - LF switched v2 | +0.034 [+0.025, +0.044] 79/21 | +0.026 [+0.015, +0.036] | +0.040 [+0.025, +0.057] | +0.036 [+0.021, +0.051] |
| LF switched v2 - GNN v2 | +0.056 [+0.043, +0.070] 79/21 | +0.035 [+0.019, +0.051] | +0.071 [+0.050, +0.094] | +0.062 [+0.039, +0.086] |
| NFD v2 - GNN v2 | +0.090 [+0.077, +0.104] 97/3 | +0.061 [+0.046, +0.076] | +0.112 [+0.087, +0.138] | +0.098 [+0.078, +0.118] |

Per goal, LF v2 - GNN v2 on random_quadrant is unresolved (+0.001); NFD v2 - GNN v2 on random_quadrant
is +0.020 (p 0.036); every other goal-level cross-family difference favours the higher-ranked model
with CI excluding 0 (RUN-0002/0003). Strata: NFD v2 > LF v2 > GNN v2 in all five strata
(rand_blob / rand_spread / small / mid / large piles).

**v2 vs v1 / original, paired slateN all 9 cells:** NFD +0.032 [+0.022, +0.043] 68/31; LF switched
+0.179 [+0.156, +0.203] 93/7; LF single +0.100 [+0.076, +0.125]; GNN vs original N 200 +0.168
[+0.137, +0.204] 91/9, vs original N 30 +0.103 [+0.087, +0.120] 91/9. Every vf positive for NFD, LF
switched and GNN. Accuracy deltas (slate CI): NFD +0.060 [+0.052, +0.067], LF switched +0.141
[+0.136, +0.146], GNN +0.063 [+0.045, +0.082] vs N 200.

**DS-0020 v2 val accuracy** (1,665 rows, trajectory-cluster CI): NFD 0.505, LF switched 0.390 (selected
on this split), GNN 0.190, original GNN N 200 -0.004; DS-0019 accuracy exceeds v2 val accuracy for
all three families (NFD +0.044, LF +0.079, GNN +0.064).

**Inference, ms per DS-0019 slate** (whole pool, 86-191 candidates, median 168, one batch; median of 3
processes, spread <= 2.7 %):

| model | CUDA (RTX 4070 Laptop) | CPU (4 threads) | us / candidate (CUDA) |
|---|---|---|---|
| **NFD v2** | **4.9** | 80.7 | 29.1 |
| **LF switched v2** | **10.4** | **55.7** | 63.7 |
| LF single v2 | 4.2 | 39.7 | 25.2 |
| **GNN v2, incl. its own perception** | **77.2** | 259 | 498 |
| GNN v2, perception cached | 43.6 | 207 | 259 |
| shared: PNG -> binary mask (CPU, per image, NFD/LF input) | 10.3 | 10.3 | -- |

The CUDA order NFD < LF < GNN does **not** hold on CPU (LF switched 55.7 < NFD 80.7). The GNN is
launch-bound on CUDA (kernel-count ratio B=128/B=16 = 6.40) by the per-candidate render loop and the
per-sample graph-build loop; NFD has no per-sample loop (ratio 1.00-1.05) and is ~92 % UNet forward.
Switched LF on CUDA is dominated by the per-occupied-bin warp/unwarp/blend round trip (72 %), not the
operator matmul.

## What would change the verdict

- **More seeds (cheapest decisive check for (a)).** NFD v2 seeds 1-2 (~80 min GPU each) and GNN v2
  seeds 1-2 (~80 min CPU / less on GPU) would measure the v2 seed noise. The smallest claimed gap,
  NFD - LF +0.034, is ~8x EXP-0061's v1 NFD seed sd; it would fall only if v2 seed noise were several
  times larger than v1's. LF has no seed (closed-form fit).
- **GNN vectorised render on CUDA.** `vector_render` / `fast_graph` exist and are bit-identical but
  were timed only on CPU (where the vector render is slower). On CUDA they could plausibly remove most
  of the 43.6 ms launch-bound cost; even a 5x speed-up (~9 ms model + ~34 ms perception) would leave
  the GNN slower than NFD incl. perception, but could move "GNN cached" below LF. ~15 min to time.
- **LF warp-once switched implementation** (cf. `predict_switched_soft`) could bring switched LF near
  single LF (4.2 ms) and reorder NFD vs LF on CUDA timing (claim (c)). ~1 h to write + time.
- **LF lambda / bin scheme chosen on a separate split** (e.g. split v2 val by trajectory into select /
  report halves): removes the selection on the reported val number; the DS-0019 test numbers are not
  selected on, and the val curve is flat within 0.006 over lambda 100-1000, so a large change is not
  expected. ~10 min.
- **DS-0019-matched push lengths in training** (re-collect or re-weight DS-0020 v2 toward bins 4-5,
  5 % of v2 rows vs 33 % of DS-0019): the most likely change to move LF (last bin 110 train rows vs
  18 % of test) and to explain DS-0019 > val accuracy. A test-side check is cheap: score accuracy per
  push-length bin for NFD and GNN as already done for LF (~5 min), or reweight val by DS-0019's
  length mix.
- **GPU re-score of MODEL-0010** (~30 s) to close the CPU/GPU scoring path difference.
- **Soft particle-splat truth** as a secondary row (EXP-0061 found NFD close under both); would show
  whether the binary truth's blindness to stacked material favours one family.

## Threats

- **imprecision**: one training seed per model; v2 seed noise is unmeasured. The only references are
  EXP-0061's v1 NFD 3-seed sd 0.002-0.004 slateN per vf and the original GNN's FPS-sampling spread
  0.005-0.016 -- different recipes and data. The claimed gaps (0.034-0.090 cross-family, 0.032-0.179
  v2-v1) sit well above those references, but the reference is borrowed, not measured.
- **indirectness**: (1) the GNN's mask, and therefore its slateN and accuracy, goes through OUR
  nearest-node carry renderer; at N = 30 even perfect node dynamics cap it at slateN 0.913 / accuracy
  0.363, and the cap alone is below NFD v2 (0.940) -- so "GNN < NFD" here is partly a statement about
  30-node mask rendering, not only about learned dynamics (the GNN reaches 70 % of its accuracy cap).
  (2) Truth is the binary top-down mask, blind to stacked material: EXP-0061's image-mask probe
  (`figures/image_mask/agreement.jsonl`, `hidden_frac`) measured 1.6-38 % of particles hidden
  (median 15 %) on DS-0019 and 4-15 % on DS-0020 v1. (3) Accuracy is reported but is suspect across
  families; slateN decides.
- **selection**: LF bin scheme and lambda were chosen on the same v2 val split whose LF val accuracy
  is reported (equal-width beat collection bins by +0.007 on val; on DS-0019 that difference is
  unresolved, +0.007 [-0.002, +0.016]). The GNN target (visual vs particle) was chosen on a 25-min
  val pilot -- the two finished runs are unresolved on DS-0019 slateN (+0.003 [-0.011, +0.017]), so
  the choice does not drive the ranking.
- **provenance**: dirty tree (above); the cross-family comparison crosses representations
  (grid-native NFD/LF vs node-based GNN) and devices (MODEL-0010 scored on CPU, NFD/LF on GPU;
  the original GNN's CPU re-score matched EXP-0061's GPU slateN exactly but 18/16,583 row rms
  differ by <= 0.0067). The "v2 beats v1" comparisons also change more than data: NFD v1 was
  budget-stopped at 80 epochs, v2 plateau-stopped at 129; LF's lambda grid gained 10000 (not chosen);
  the original GNN used the source repo's particle-supervised, variable-N, true-depth recipe. So (b)
  says the v2 MODELS are better, not that the v2 DATA alone made them so.
- **untested-dependency**: `flex-action-frame-neg-z` is `unchecked` (probe evidence on v1 and v2; the
  pytest checks only the as-stored reading).
- **Train/test push-length mismatch**: DS-0020 v2's ±3.4 action clamp shortens long requests, so
  bins 4-5 (6.7-9.6 units) are 5 % of v2 rows vs 1/3 of DS-0019's; median push 3.77 vs 5.33. LF's last
  bin has 110 train rows (16 val) but carries 2,939 DS-0019 rows (18 %).
- **Unexplained: DS-0019 accuracy above own-val accuracy** for all three families (NFD 0.549 vs 0.505,
  LF 0.469 vs 0.390, GNN 0.254 vs 0.190), the opposite of EXP-0061's v1 pattern. Not isolated. LF's
  per-bin DS-0019 accuracy rises with push length (0.26 -> 0.53), and DS-0019's pushes are longer --
  consistent with a length-mix effect, not proof of it.
- **NFD not fully converged**: stopped by the 0.5 %/20-epoch plateau rule while best-val was still
  improving 0.30-0.39 % per 20 epochs (comparable to epoch-to-epoch val sd 0.5 %); more training could
  only plausibly widen its lead. GNN val loss is noisy (single-epoch spikes to 2x); "plateau" is coarse.
- **Flagged-row drop**: `exclude_flagged=True` removes 9.7 % of v2 rows as escaped (1,612 of 1,937 are
  carry-over rows whose before- and after-images both lack the escaped particles; only 325 are the
  escape event). Training saw ~1,550 fewer train/val rows than a "first escape only" filter would give.
- **Timing caveats**: a 35 W laptop GPU, one machine; the GNN is launch-bound by per-sample Python
  loops (vectorised versions exist, timed only on CPU); the first CUDA GNN attempt failed the harness
  assertion and was fixed (not a timing change: timed passes carry the same hooks as NFD/LF); LF CUDA
  numbers are after RUN-0002's operator-cache fix (45.9 ms before); NFD/LF exclude their 10.3 ms CPU
  perception step while the GNN "incl." row includes its own (~34 ms); no per-stage CUDA breakdown
  for the GNN. CPU and CUDA rankings differ (above), so (c) is a CUDA-only claim.
- Considered, not taken -- **inconsistency**: the order holds in every vf and every stratum for every pair,
  and in every goal except LF - GNN on random_quadrant (+0.001, unresolved, not reversed).

## Unrelated findings

- `Baselines/LinearForesight/predictor.py` host->device operator copy on every switched call (fixed in
  RUN-0002): any earlier LF CUDA timing in the repo is inflated ~4.4x (EXP-0061 did not time LF).
- MODEL-0010's MODEL.md says "plateau-stopped at epoch 22"; RUN-0003 says 23 epochs run -- the same
  stop counted 0-based vs 1-based.
- RUN-0003's header still says "everything below ran on the CPU ... timing is CPU-only", which the
  CUDA timing section added later supersedes (training and scoring did run on CPU).
