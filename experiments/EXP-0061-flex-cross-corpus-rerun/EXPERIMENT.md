---
# ---- identity -------------------------------------------------------------
id: EXP-0061
title: On FleX carrot piles (fit on DS-0020, tested on DS-0019, binary image-mask truth) EXP-0001's pattern replicates -- NFD beats the dyn-res GNN on accuracy AND slateN, and every model's slateN is far above random
tier: T1
mode: exploratory
date: 2026-10-01
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  Re-test of EXP-0001's claim ("NFD's swept-region accuracy exceeds the GNN's;
  every slateN cell is positive") on FleX data: models fit on DS-0020 (2000 x 10
  pushes from one spread state) and scored on DS-0019 (100 same-state slates of
  compact piles, 16,583 kept rows), with model input AND ground truth = the binary
  64x64 top-down image mask (+-7.2 units), goals random_quadrant / ring_O / T x
  value functions lyapunov / mass_in_region / signed_mass. Under these conditions
  (a) the 3-seed 80-epoch NFD (MODEL-0005/6/7) has higher DS-0019 accuracy than
  gnn_flex_drp (original dyn-res-pile-manip checkpoint, N = 200, constant-plane
  depth, our nearest-node mask renderer), with non-overlapping slate-bootstrap
  CIs, and (b) every (model x goal x value-function) slateN cell of NFD, GNN and
  both linear-foresight models is > 0.

# prediction: omitted -- exploratory. EXP-0001's claim is the thing re-tested,
# but no threshold was committed in git before this run, and the NFD seed-0
# DS-0019 score had already been seen by the training agent (RUNS_nfd.md).

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: d72bb304
  dirty: true                     # see "What was actually run" for the file list
  data_commit: "unrecorded (DS-0019/DS-0020 ported from an external FleX project; no collector config)"
  script: Baselines/common/eval_report.py (official GNN+LF rows) + experiments/EXP-0061-flex-cross-corpus-rerun/code/final_eval.py (per-row/per-slate outputs, composes eval_report's functions) + code/analyze_final.py
  data: [DS-0019, DS-0020]
  code_path: "eval_report corpus flex_ds0019_mask, --truth-scoring image (= _capture_report(truth_s0=None) -> cell.occ1 = FlexData.ImageMaskSource binary after-mask); accuracy = fit_linear_foresight.metrics swept region, plate 10.67 px"
  seed: 0
  split: "train/fit: DS-0020 splits.json train (1800 trajectories); val: DS-0020 val (200 trajectories, 1,991 kept rows); test: all 100 DS-0019 slates (exclude_flagged=True)"
  runtime: "GNN 17-30 s per DS-0019 pass, NFD/LF 3-5 s; DS-0020 val GNN 6 min; single 8 GB GPU, shared machine"
  runs: [RUN-0001]

budget:
  declared: "3 h wall clock, HARD (final-eval agent)"
  spent: "~1 h 15 min wall clock"
  outcome: within

design:
  varied:
    model: [nfd_s0 (MODEL-0005), nfd_s1 (MODEL-0006), nfd_s2 (MODEL-0007), gnn_flex_drp (N 200, FPS sampling seeds 0/1/2), lf_flex_switched (MODEL-0004), lf_flex_single (MODEL-0004)]
    goal: [random_quadrant, ring_O, T]
    value_function: [lyapunov, mass_in_region, signed_mass]
    stratum: [init_pos rand_blob / rand_spread, piece-count tercile, GNN fallback / FPS states]
  held_fixed:
    representation: "binary top-down image mask, 64x64, +-7.2 FleX units (user decision)"
    truth: "binary image mask of the true after-state (--truth-scoring image), NOT the soft particle splat"
    slateN_pool: "every kept candidate of the slate (86-191), no replacement"
    gnn: "particle_num 200 (chosen on DS-0020 val accuracy), plane_y 0.24, carry_k 1, training-convention pusher half-width; small-pile fallback added in this run (see What was actually run)"
    lf_lambda: "switched 300 / single 1000, chosen on the DS-0020 val split it is also reported on"
    nfd_recipe: "Baselines/NFD/configs/nfd_3ch_flex_mask.yaml, 80 epochs (budget-forced, not converged)"
  baselines: [persistence, random]
  metric: "slateN (3 value functions, experiments/METRICS.md), accuracy"

noise_floor: "NFD training-seed spread (3 seeds): slateN goal-avg sd 0.002-0.004 per vf, every seed pair unresolved (|diff| <= 0.004, CIs span 0); accuracy sd 0.004 on DS-0019, 0.001 on val. GNN node-sampling spread (3 FPS seeds): slateN 0.759-0.764 / 0.640-0.648 / 0.636-0.652, every pair unresolved; accuracy 0.1908-0.1913. Per-slate paired sd of a model difference ~0.08-0.19 (slate is the replication unit, 100 slates)."

depends_on: [flex-action-frame-neg-z, goal-mask-axis-convention-row-y-col-x]
establishes: []

# ---- outcome --------------------------------------------------------------
result: "DS-0019 accuracy NFD 0.485 (seeds 0.481-0.490; slate CI ~[0.46, 0.51]) vs GNN 0.191 (CI [0.167, 0.213]); LF switched 0.328, single 0.239. slateN (mean of 3 goals, lyapunov/mass_in_region/signed_mass): NFD 0.952/0.876/0.901, GNN 0.762/0.644/0.642, LF switched 0.843/0.711/0.626, LF single 0.677/0.539/0.516, random 0.006/0.002/0.003; minimum model cell 0.362 (LF single, random_quadrant/signed_mass). NFD-GNN paired slateN +0.227 [+0.194, +0.263], 98/2 slates, Holm p < 1e-4. EXP-0001's claim replicates on this data."
verdict: supported
downgrades: [indirectness, provenance, selection, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

> **Data-location note (2026-10-02).** Every "DS-0020" in this record is DS-0020 **v1** (2000
> trajectories from ONE uniform-spread start state). The user then replaced DS-0020's payload with a new
> blob/spread collection (v2). v1 was archived, not deleted: raw `datasets/DS-0020-training-data-flex-N864/old_data/<traj>/`,
> config/splits/cache (incl. `image_masks.npz`, `image_paths.json`) `old_data/_ported_v1/`; read it with
> `old_data/_ported_v1/config.yaml` (the code/ scripts and `configs/dataset/flex_ds0020_train_ds0019_test.yaml`
> were repointed there; logs and result files are left as written). The numbers here are unaffected.
> v2 data check: `figures/data_check_v2/`, `code/data_check_v2.py`, `code/leakage_scan_v2.py`.

> **Follow-up (2026-10-03): EXP-0062** retrained all three families on DS-0020 **v2** (DS-0019's
> pile generator; trajectories 0-99 excluded) and re-tested on DS-0019 under this record's protocol
> (MODEL-0008 NFD, MODEL-0009 LF, MODEL-0010 GNN at 30 nodes). The NFD lead survives but narrows
> (NFD - LF switched +0.034, NFD - GNN +0.090 slateN), and LF switched now beats the GNN. This record's
> claim and numbers (v1-trained models) are unchanged; MODEL-0004/0005 and the original GNN checkpoint
> were re-scored there and reproduce this record's per-slate files exactly.

## Why this test discriminates

EXP-0001 found NFD > GNN on accuracy and all slateN positive on Genesis cube
corpora, with a GNN retrained in this repo. Here the GNN is the original
dyn-res-pile-manip checkpoint on the data family it was trained on (FleX carrots,
same scene and shape as DS-0020), so if EXP-0001's gap were only "our GNN port
is badly trained", the GNN should close or reverse it here. If the image-mask
pipeline broke ranking (frame error, wrong truth), some slateN cells would sit
near random (0.00-0.01 measured). Neither happened: the gap is 30 accuracy
points and 0.19-0.26 slateN, outside every noise floor measured, and the weakest
model cell is 0.36.

## What was actually run

- **Truth verified in code before scoring.** `eval_report.main` passes
  `truth_s0=None` for `--truth-scoring image`; `_capture_report` then scores
  against `cell.occ1`; for corpus `flex_ds0019_mask` that is
  `FlexData.dataset.ImageMaskSource.pair` -> `image_masks.npz['after']`, uint8
  with values exactly {0, 1} (checked). Not the area fraction, not the splat.
  RUNS_gnn_lf.md's suggested command omits the flag (default `soft` = particle
  splat); it was added here.
- **NFD rows reused, after reproduction.** `results/nfd_seed{0,1,2}_ds0019.json`
  came from `code/score_nfd_flex.py`, which composes the same eval_report
  functions with the same truth path. `final_eval.py` re-scored all three seeds:
  accuracy and every capture identical to the existing JSONs (0.4895 / 0.4809 /
  0.4842). The GNN seed-0 row from `final_eval.py` is identical to the
  `eval_report.py` CLI row (0.1908; same per-goal captures).
- **GNN small-pile fallback (code change in this run).** The first DS-0019 GNN pass
  crashed (`ZeroDivisionError`): 43/100 DS-0019 initial states (nearly all
  `rand_blob`, 73-200 voxels; `results/final_eval/ds0019_gnn_nvox.json`) have no
  more voxels than particle_num = 200, so FPS takes every voxel and the covering
  radius is 0 (particle_den = inf). The source repo (`env/flex_env.py::
  obs2ptcl_fixed_num_batch`, `utils.py::fps`) has no path for this either. Added
  to `Baselines/GNN/flex_predictor.py::perceive`: when n_voxels <= particle_num,
  every voxel is a node, the covering radius is measured against the dense fg
  pixel cloud, and particle_den is clamped to the source training max 6,500.
  States with more voxels are bit-identical (FPS path untouched). Sensitivity:
  `gnn_flex_drp_n50` (N = 50; min voxels 73, so no state uses the fallback).
- **GNN sampling noise** = the registered predictor with its per-state FPS start
  seed shifted (`final_eval.py::gnn_with_sampling_seed`; offset 0 = registered).
  Fallback states have no sampling, so on 43 slates the three seeds are identical
  by construction; the measured spread is from the other 57.
- Random row: per-slate mean over eval_report's 20 noise seeds; persistence slateN
  is degenerate (constant prediction) and is shown only as a reference.
- **Dirty tree** at d72bb304 (relevant files, all from the EXP-0061 session, none
  committed): untracked `Baselines/GNN/flex_predictor.py` (edited in this run),
  `FlexData/`, `Baselines/NFD/configs/nfd_3ch_flex_mask.yaml`,
  `Baselines/NFD/runs/nfd_3ch_flex_mask_seed{0,1,2}/`, `configs/dataset/flex_*.yaml`,
  `datasets/DS-0019-*/`, `datasets/DS-0020-*/`; modified `Baselines/common/eval_report.py`,
  `Baselines/LinearForesight/{fit_switched,predictor}.py`, `fit_linear_foresight.py`,
  `registry/dataset_registry.py`, `training/trainer.py`, `Baselines/NFD/train_nfd.py`,
  `docs/{ARCHITECTURE,CODEMAP,INTERFACES}.md`, `.gitignore`.
- Commands (exact argv in `runs/RUN-0001-final-eval/COMMAND_*.txt` and
  `experiments/COMMANDS.jsonl`, run ids `EXP-0061/RUN-0001/*`):

      PYTHONPATH=. python -u Baselines/common/eval_report.py --corpora flex_ds0019_mask \
          --models gnn_flex_drp,lf_flex_switched,lf_flex_single --device cuda \
          --truth-scoring image --out-prefix experiments/EXP-0061-flex-cross-corpus-rerun/results/ds0019_gnn_lf
      python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/final_eval.py ds0019
      python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/final_eval.py ds0019 --only gnn_flex_drp_n50
      python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/final_eval.py ds0020_val
      python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/analyze_final.py
      python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/plot_final.py

  Earlier attempts that failed and were re-run: eval_report without `PYTHONPATH=.`
  (import error), both DS-0019 jobs before the fallback (ZeroDivisionError), and
  one final_eval pass that OOM'd while sharing the GPU with eval_report.
- Training/fit records: RUNS_nfd.md (NFD, image-mask cache), RUNS_gnn_lf.md (GNN
  checks, N/depth choice, LF fit).

## Numbers

Outputs: `results/final_eval/headline.json` (compact), `summary.json` /
`summary.md` (every cell), `ds0019.json` (per-slate captures),
`ds0019_rows.npz` / `ds0020_val_rows.npz` (per-row errors), `results/ds0019_gnn_lf.json`
(eval_report CLI). Figure `figures/final_eval/slateN_and_accuracy.png`.

**slateN, mean of the 3 goals** (DS-0019, 100 slates; family = mean over 3 seeds,
[seed range], slate-bootstrap 95 % CI of the family mean):

| model | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| NFD (3 training seeds) | 0.952 [0.948-0.955] (0.944-0.960) | 0.876 [0.874-0.878] (0.862-0.891) | 0.901 [0.896-0.904] (0.885-0.916) |
| GNN N 200 (3 sampling seeds) | 0.762 [0.759-0.764] (0.715-0.800) | 0.644 [0.640-0.648] (0.604-0.682) | 0.642 [0.636-0.652] (0.599-0.683) |
| LF switched | 0.843 (0.818-0.867) | 0.711 (0.685-0.735) | 0.626 (0.582-0.667) |
| LF single | 0.677 (0.642-0.715) | 0.539 (0.500-0.576) | 0.516 (0.475-0.553) |
| GNN N 50 (sensitivity) | 0.797 | 0.696 | 0.691 |
| random | 0.006 (-0.007-0.018) | 0.002 (-0.008-0.011) | 0.003 (-0.007-0.012) |
| persistence (DEGENERATE ranker) | 0.123 | 0.011 | 0.093 |

**slateN per goal x value function** (mean +- slate sem):

| model | rq/lyap | rq/mass | rq/signed | O/lyap | O/mass | O/signed | T/lyap | T/mass | T/signed |
|---|---|---|---|---|---|---|---|---|---|
| nfd_s0 | 0.926+-0.010 | 0.832+-0.018 | 0.898+-0.012 | 0.952+-0.008 | 0.871+-0.015 | 0.885+-0.014 | 0.967+-0.008 | 0.929+-0.012 | 0.906+-0.014 |
| nfd_s1 | 0.937+-0.009 | 0.838+-0.018 | 0.903+-0.012 | 0.955+-0.008 | 0.873+-0.016 | 0.887+-0.014 | 0.972+-0.006 | 0.910+-0.014 | 0.915+-0.017 |
| nfd_s2 | 0.939+-0.009 | 0.831+-0.017 | 0.902+-0.012 | 0.950+-0.008 | 0.877+-0.016 | 0.883+-0.015 | 0.969+-0.007 | 0.925+-0.013 | 0.927+-0.012 |
| gnn samp0 | 0.863+-0.024 | 0.719+-0.025 | 0.818+-0.023 | 0.589+-0.037 | 0.562+-0.035 | 0.509+-0.037 | 0.839+-0.025 | 0.650+-0.028 | 0.582+-0.037 |
| gnn samp1 | 0.866+-0.023 | 0.719+-0.028 | 0.831+-0.023 | 0.603+-0.034 | 0.562+-0.033 | 0.515+-0.037 | 0.822+-0.029 | 0.640+-0.030 | 0.611+-0.036 |
| gnn samp2 | 0.864+-0.024 | 0.722+-0.027 | 0.817+-0.023 | 0.596+-0.036 | 0.580+-0.034 | 0.518+-0.033 | 0.817+-0.031 | 0.642+-0.028 | 0.574+-0.036 |
| gnn N 50 | 0.898+-0.013 | 0.760+-0.017 | 0.849+-0.013 | 0.678+-0.037 | 0.637+-0.030 | 0.612+-0.032 | 0.814+-0.029 | 0.691+-0.032 | 0.611+-0.040 |
| lf switched | 0.742+-0.030 | 0.561+-0.031 | 0.495+-0.038 | 0.880+-0.017 | 0.738+-0.027 | 0.695+-0.031 | 0.906+-0.013 | 0.833+-0.021 | 0.688+-0.029 |
| lf single | 0.602+-0.030 | 0.366+-0.034 | 0.362+-0.032 | 0.648+-0.037 | 0.549+-0.037 | 0.509+-0.037 | 0.781+-0.029 | 0.703+-0.031 | 0.678+-0.025 |
| random | 0.008+-0.013 | 0.001+-0.009 | 0.011+-0.008 | 0.006+-0.010 | 0.009+-0.009 | 0.001+-0.010 | 0.003+-0.012 | -0.004+-0.008 | -0.004+-0.007 |
| persistence | 0.184+-0.047 | -0.031+-0.028 | 0.004+-0.029 | -0.117+-0.038 | -0.093+-0.037 | 0.061+-0.032 | 0.303+-0.042 | 0.158+-0.034 | 0.215+-0.030 |

**Goal degeneracy, frac(dv_true == 0)** on the binary-mask truth (no slate has a
flat pool for any cell):

| goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| random_quadrant | 0.0010 | 0.1874 | 0.0061 |
| ring_O | 0.0005 | 0.0471 | 0.0083 |
| T | 0.0005 | 0.0674 | 0.0076 |

**Paired slateN** (`paired_stats.paired_comparison`, per-slate mean of the 9
cells, family seed-means, 100 slates, 10 pairs Holm-corrected; diff [95 % CI],
wins/losses):

| pair | all 9 cells | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| NFD - GNN | +0.227 [+0.194, +0.263] 98/2 | +0.190 [+0.150, +0.236] | +0.233 [+0.193, +0.273] | +0.259 [+0.218, +0.303] |
| NFD - LF switched | +0.183 [+0.160, +0.207] 94/6 | +0.109 [+0.088, +0.132] | +0.166 [+0.140, +0.192] | +0.275 [+0.232, +0.317] |
| NFD - LF single | +0.332 [+0.305, +0.360] 100/0 | +0.275 | +0.337 | +0.384 |
| GNN - LF switched | -0.044 [-0.086, -0.003] 46/54, Holm p 0.041 | -0.081 [-0.129, -0.035] | -0.067 [-0.112, -0.022] | +0.016 [-0.046, +0.075] (unresolved) |
| GNN - LF single | +0.105 [+0.067, +0.142] 73/27 | +0.085 | +0.105 | +0.125 |
| each model - random | +0.57 to +0.91, all Holm p < 1e-4 | | | |

Every individual NFD seed beats every GNN sampling seed and both LF models
(Holm p < 1e-4). Within-family pairs (NFD seeds; GNN sampling seeds) are all
unresolved (|diff| <= 0.004). Individual GNN seed vs LF switched: -0.041 to
-0.045, CI excludes 0 but Holm p 0.29-0.42 over the 28-pair family -- **GNN vs
LF switched is unresolved** at the individual-model level and flips sign by
stratum (below). GNN N 50 - GNN N 200: +0.045 [+0.009, +0.083], 62/38.

**Accuracy** (ratio of population means; DS-0019 row-bootstrap CI and
slate-cluster CI; DS-0020 val trajectory-cluster CI):

| model | DS-0020 val | DS-0019 test | row CI | slate CI | test - val |
|---|---|---|---|---|---|
| nfd_s0 | 0.6095 (0.606-0.613) | 0.4895 | 0.487-0.492 | 0.472-0.508 | -0.120 |
| nfd_s1 | 0.6117 (0.608-0.615) | 0.4809 | 0.478-0.483 | 0.463-0.498 | -0.131 |
| nfd_s2 | 0.6098 (0.606-0.613) | 0.4842 | 0.482-0.487 | 0.466-0.503 | -0.126 |
| gnn samp0 / 1 / 2 | 0.2517 (0.246-0.257) | 0.1908 / 0.1913 / 0.1913 | 0.188-0.194 | 0.167-0.214 | -0.061 |
| gnn N 50 | 0.102 (600-row sweep, RUNS_gnn_lf.md) | 0.1936 | 0.191-0.196 | 0.171-0.216 | +0.09 |
| lf switched | 0.4824 (0.478-0.487) | 0.3284 | 0.327-0.330 | 0.312-0.344 | -0.154 |
| lf single | 0.3543 (0.347-0.361) | 0.2394 | 0.237-0.242 | 0.226-0.252 | -0.115 |
| persistence | 0 | 0 | | | |

Paired accuracy deltas (slate-cluster CI): NFD s0 - GNN +0.299 [+0.280, +0.318];
NFD s0 - LF sw +0.161 [+0.150, +0.172]; LF sw - GNN +0.138 [+0.124, +0.153];
LF single - GNN +0.049 [+0.033, +0.065]. Accuracy ranking NFD > LF sw > LF single
> GNN is identical on val and test. **Accuracy and slateN disagree on GNN vs LF
single**: GNN is lower on accuracy (-0.049, resolved) but higher on slateN
(+0.105, resolved).

**Strata** (slateN = family mean over the 9 cells [slate-bootstrap CI]; accuracy
= ratio of means within the stratum, family mean):

| stratum (slates) | NFD slateN / acc | GNN slateN / acc | LF sw slateN / acc | LF single slateN / acc | GNN - LF sw slateN |
|---|---|---|---|---|---|
| rand_blob (50) | 0.908 [0.893, 0.922] / 0.468 | 0.630 [0.567, 0.688] / 0.113 | 0.807 [0.777, 0.834] / 0.263 | 0.617 / 0.193 | -0.177 [-0.237, -0.125] |
| rand_spread (50) | 0.911 [0.898, 0.923] / 0.499 | 0.735 [0.706, 0.762] / 0.255 | 0.645 [0.621, 0.669] / 0.382 | 0.538 / 0.277 | +0.089 [+0.056, +0.124] |
| pieces small: 46-106 (21) | 0.893 / 0.424 | 0.688 / 0.107 | 0.812 / 0.243 | 0.627 / 0.178 | -0.124 [-0.168, -0.072] |
| pieces mid: 190-298 (39) | 0.920 / 0.487 | 0.599 / 0.126 | 0.745 / 0.301 | 0.565 / 0.220 | -0.146 [-0.222, -0.075] |
| pieces large: 430-766 (40) | 0.908 / 0.511 | 0.761 / 0.284 | 0.664 / 0.390 | 0.564 / 0.284 | +0.097 [+0.059, +0.134] |
| GNN fallback states, <= 200 voxels (43) | 0.910 / 0.475 | 0.702 / 0.145 | 0.823 / 0.266 | 0.637 / 0.198 | -0.121 [-0.154, -0.086] |
| GNN FPS states, > 200 voxels (57) | 0.910 / 0.491 | 0.667 / 0.221 | 0.653 / 0.369 | 0.533 / 0.266 | +0.014 [-0.049, +0.073] |

Piece counts take only 7 values (46/106/190/298/430/586/766), so the terciles
are 21/39/40 slates, not equal. NFD - GNN is positive and resolved in every
stratum (+0.147 to +0.321). NFD's slateN barely moves across strata
(0.893-0.920); its accuracy drops on small piles (0.424 vs 0.511). The GNN
- LF switched sign is decided entirely by the stratum: LF switched wins on
blobs / small-mid piles, GNN wins on spreads / large piles.

**Train -> test shift:** every model loses accuracy from DS-0020 val to DS-0019
(NFD -0.12 to -0.13, LF -0.12 to -0.15, GNN -0.06); none falls to persistence.

## What would change the verdict

- **NFD trained to convergence** (val loss still falling ~1 %/10 epochs at 80):
  could only widen the NFD lead on accuracy; slateN is near ceiling on lyapunov
  (0.95), so the change would show on mass_in_region/signed_mass. ~4-6 h GPU for 3
  seeds at 150 epochs.
- **A GNN renderer that is not ours.** The nearest-node carry caps val accuracy
  at 0.46 even with the true node motion (RUNS_gnn_lf.md); a GNN-vs-NFD accuracy
  gap of 0.30 cannot be closed by a better renderer alone (0.46 < 0.49 is the
  cap, against NFD's 0.61 on val), but a particle-space value readout for slateN
  (what the source planner does) might narrow the slateN gap. ~1 h to write.
- **Choosing GNN N by slateN instead of accuracy**: N 50 already ranks better
  (+0.045). An N sweep on DS-0019 slateN is cheap (~20 s per N) but would be
  selection on the test set; a slate-bearing FleX validation corpus would be
  needed to do it honestly.
- **Soft-particle-truth slateN** (`--truth-scoring soft`, the repo default)
  as a secondary row: the NFD seed-0 soft numbers 0.961/0.849/0.905 are close
  (`results/nfd_seed0_ds0019.md`); not run for GNN/LF. ~2 min.
- **A train set that matches the test distribution** (compact piles, obj-biased
  actions in +-3.4) is the change most likely to reorder GNN vs LF; it would not
  plausibly reverse NFD's lead, which holds in every stratum.

## Threats

- **indirectness**: the GNN is scored through OUR design -- nearest-node pixel
  carry (caps val accuracy 0.46 with true node motion) and a constant 0.24
  plane for depth (DS-0019 has no depth; its multi-layer piles sit ~0.1 higher
  than the plane; cost measured only on DS-0020, 0.003); the small-pile fallback
  (43 % of test states) is also ours, and its density is clamped. The source
  planner never renders a mask. GNN's numbers are therefore a lower bound on
  what the checkpoint "knows". Also, truth is the binary image mask (user
  decision), which aliases (the trap METRICS.md's soft splat fixes); and
  EXP-0001's GNN accuracy used a node-resampled truth, so "accuracy" here is not
  the identical quantity EXP-0001 compared.
- **provenance**: dirty tree with uncommitted model code, loaders and the
  fallback edited during this run; DS-0019/DS-0020 have no collector config; the
  comparison crosses representations (grid-native NFD/LF vs node-based GNN).
  `actions_to_pixels` maps world 0 to pixel 31.5 while the FleX grid/masks use
  32 -> the swept-region mask that `accuracy` reads is 0.5 px off (4.7 % of the
  10.67 px plate); shared by every model and by the LF fit, not corrected.
- **selection**: LF lambda (300 / 1000) and GNN N (200) were chosen on DS-0020
  val, the split whose numbers are reported as "val"; the N choice by val
  accuracy picked the N that ranks worse on test (N 50 +0.045 slateN).
- **untested-dependency**: `flex-action-frame-neg-z` is `unchecked` (probe
  evidence, no pytest).
- Considered, not taken as a downgrade -- **imprecision**: the two claimed effects
  sit far outside every measured noise floor (NFD-GNN accuracy +0.30 vs seed sd
  0.004; slateN +0.19-0.26 vs seed/sampling spread <= 0.016; weakest slateN cell
  0.36 vs random 0.01). But: the training noise floor is 3 seeds of a
  non-converged recipe; lyapunov slateN near ceiling (NFD 0.95) leaves no
  headroom to separate NFD variants; and every comparison NOT in the claim (GNN
  vs LF switched) is unresolved / stratum-dependent and must not be ranked.
- Considered, not taken -- **inconsistency**: the claim holds in every stratum,
  every goal x vf cell, every seed. The GNN-vs-LF reversal across strata is not
  part of the claim.
- **LF's under-filled last bin** (136 train rows, M/D 0.03) affects only pushes
  > 11.0 units, which DS-0019 (max 9.6) never contains -- irrelevant on test.
- **Train/test shift** is large (one uniform spread start vs compact piles; actions
  anywhere in +-5 vs object-biased in +-3.4; occupied fraction 0.44 vs 0.12):
  quantified above (accuracy drops 0.06-0.15; NFD's slateN does not vary by
  stratum). It is the reason not to read the GNN-vs-LF order as general.
- pool sizes 86-191: slateN here is not comparable to Genesis-corpus slateN
  (METRICS.md), so EXP-0001's numbers and these replicate in sign/order only.

## Unrelated findings

- `Baselines/common/eval_report.py` run as a script needs `PYTHONPATH=.`
  (`from control_utility_test import lyapunov` fails otherwise); the
  RUNS_gnn_lf.md command omits it.
- `Baselines/GNN/flex_predictor.py`: non-fallback DS-0019 states also exceed the
  training density range (e.g. state 1: particle_den 7,317 > 6,500); not clamped.
- The skill-bundled dataviz palette validator does not run on this machine's
  Node (no ES-module / `??=` support).
