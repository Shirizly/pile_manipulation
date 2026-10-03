# EXP-0062 / RUN-0003 -- dyn-res GNN architecture trained from scratch on DS-0020 v2, constant 30 nodes, tested on DS-0019, timed (2026-10-02, 16:15-19:20 CEST)

- commit d72bb304, **dirty** (EXP-0061 / RUN-0001 / RUN-0002 uncommitted files, plus in this run: new
  `Baselines/GNN/flex_train.py`; `Baselines/GNN/flex_predictor.py` gained `predict_one_step_vec` and the opt-in
  `fast_graph` / `vector_render` / `cache_states` flags (defaults = the EXP-0061 path); `Baselines/common/eval_report.py`
  MODELS `gnn_flex_v2_n30`, `gnn_flex_drp_n30`; this experiment's `code/{plane_check,eval_gnn_v2,gnn_val_acc,
  profile_gnn_breakdown,aggregate_timing_gnn,plot_gnn_convergence}.py`; `docs/CODEMAP.md`). python
  `/home/alon/anaconda3/envs/pme/bin/python -u`.
- **GPU FAILURE.** The RTX 4070 Laptop GPU was idle (15 MiB) at the start, then at ~16:50 went into
  `nvidia-smi: GPU requires reset` (`torch.cuda.is_available() == False`); it never came back during the run and
  needs a reset/reboot by the user. **Everything below ran on the CPU** (torch 4 threads each): training, scoring,
  timing. The first timing pass was therefore CPU-only (same-device references given). **Update 2026-10-02 ~19:30:**
  after the GPU reset, CUDA timing was added (section "Timing on CUDA" below); training and scoring remain CPU-only.
- data: DS-0020 **v2** (splits.json, trajectories 0-99 excluded; 5-step windows of kept transitions: train 8,289 /
  val 911), DS-0019 (100 slates, 16,583 kept rows). **Model input = colour image only**; truth for scoring = binary
  image mask of the true after-state (`--truth-scoring image` path, `cell.occ1` asserted {0, 1}).
- model: **MODEL-0010** (`weights/MODEL-0010-gnn-flex-mask-v2-n30/`). Sensitivity sibling (not promoted):
  particle-supervised target, `artifacts/RUN-0003-gnn/alt_particle_target_best.pth`.
- argv: `experiments/COMMANDS.jsonl` run ids `EXP-0062/RUN-0003/*`; logs `../../logs/{gnn_cache_n30,train_gnn_*,
  valacc_pilot_*,score_gnn_*,analyze_gnn,profile_gnn_*,timing_gnn_*}.log`.

## Source recipe recovered (`/home/alon/Code/dyn-res-pile-manip`, read-only)

`train/train_gnn_dyn.py` + `config/train/gnn_dyn.yaml` + `dataset/dataset_gnn_dyn.py`:
Adam lr 1e-3 (betas 0.9/0.999), **batch 4**, n_history 1, **n_rollout 5** (chained, s_cur <- s_pred), loss = sum of
`F.mse_loss(s_pred, s_nxt)` over rows and steps / (n_rollout * B) on 3-D node positions in the camera frame divided
by global_scale 24, StepLR(1000 epochs, 0.1), best-val checkpoint, nominal 2000 epochs. Nodes: `fps_rad` of the first
frame's DEPTH point cloud at a random density U(15, 6500) (so the node count varies per sample), recentered; **each node
is tied to its nearest simulator particle, and both the step-0 input and every target are those particles' positions**
(particle correspondence); `s_delta` from the true tracked positions, pusher half-width 0.8/24; attrs 0; particle_den an input.
No EMD/Chamfer loss exists in the source.

What changed (user rules: input always visual; constant 30 nodes):
- input nodes = `flex_predictor.perceive` on the colour PNG (constant plane, voxel 0.01, FPS 30 from a per-state seeded
  start, recenter, den = 1/r_fps^2) -- the same function and seed as the test-time predictor (node cache built with it);
- rollout steps j > 0 use `s_delta` from the predicted nodes; den fixed per window;
- stop by val-loss plateau (patience 10 epochs, > 0.5 % relative improvement) instead of 2000 epochs; batch 4 kept.

**Plane y = 0.24 kept** (`code/plane_check.py` -> `results/gnn_plane_check.json`): a particle top-surface estimator
calibrated on v1 (true depth 0.249 vs estimator 0.201) gives v2 0.32 (p10-p90 0.25-0.37) and DS-0019 0.45. A plane error
only rescales node XY by (18 - p)/(18 - h): at radius 5 units that is 0.022 (v2) / 0.060 (DS-0019) units = 0.1 / 0.27 px
of the 0.225-unit grid -- negligible, and train and test use the same constant. Kept for comparability with EXP-0061.

## Training target: visual (Chamfer) chosen

Two targets, identical otherwise (`Baselines/GNN/flex_train.py --target`):
- **`chamfer_carry` (fully visual, PRIMARY):** every voxel of the input cloud is carried by its nearest input node's
  predicted (cumulative) displacement -- exactly what the mask renderer does -- and compared with the voxel cloud of the
  TRUE next frame's colour image (symmetric squared Chamfer; training clouds subsampled to <= 384 points per row for
  speed, val on full clouds) + an MSE pinning node z to the plane.
- **`particle` (particle-SUPERVISED, visually-input):** target = node + displacement of its nearest particle (table XY).

Choice made on a 25-min-each pilot (same seed, concurrent, CPU): DS-0020 v2 val mask accuracy through the real
predictor 0.1805 (visual, 6 epochs) vs 0.1940 (particle, 11 epochs) (`results/gnn_target_choice_*.json`). Both far above
persistence (val loss / persistence 0.16 visual, 0.34 particle). The visual target was **workable**, so it is primary,
as instructed; the particle run was continued to convergence as a sensitivity row. Final result: **no resolvable
difference in slateN** (visual - particle +0.003 [-0.011, +0.017], 57/43 slates, Holm p 0.68); visual +0.008 [+0.004,
+0.013] accuracy on DS-0019, val 0.190 vs 0.193 (CIs overlap). Goal-level differences of opposite sign (T +0.035, ring_O -0.028).

## Training (both CPU, 4 threads, concurrent with each other)

| | epochs run | best epoch | best val / persistence val loss | wall |
|---|---|---|---|---|
| visual (MODEL-0010) | 23 (plateau stop) | 12 | 2.98e-4 / 2.16e-3 (0.14) | 82 min |
| particle (alt) | 27 (plateau stop) | 16 | 9.18e-4 / 2.82e-3 (0.33) | 63 min |

Curves: `figures/gnn_v2_n30_convergence.png` (train loss is sampled every 20 iterations and, for Chamfer, on subsampled
clouds, so it sits above val). Visual val loss is noisy (single-epoch spikes to 2x) but flat at 0.14-0.15 x persistence
over the last 11 epochs; particle flat at 0.33-0.36 from epoch 8. Val mask accuracy barely moved after the pilot
(visual 0.1805 -> 0.1903, particle 0.1940 -> 0.1926): converged for this architecture/recipe.

**Interruptibility:** full state (`last_state.pt`: model, Adam, StepLR, epoch, position in the epoch's permutation, every
RNG, best, plateau counter, history) atomic after every epoch and every `--save-min` minutes. **kill -9 test:** the
particle run (save-min 0.5) was killed at epoch 1 / window 1332 and resumed from exactly that mid-epoch position
(log `[resume] epoch 1 pos 1332`); both final runs are themselves `--resume` continuations of the pilots.
Node cache `datasets/DS-0020-*/cache/gnn_nodes_n30_plane0.24/` (19 atomic chunks + manifest + DONE; 89 MB; 9 min, 12 workers).
`check-vec`: `predict_one_step_vec` == source `predict_one_step`, max |diff| 0.

## DS-0019 test (`code/eval_gnn_v2.py`, `results/gnn_ds0019.{json,md}`)

Same code path and flags as RUN-0001/0002 and EXP-0061 `final_eval.py` (`_accuracy`, `_capture_report(cell, pred,
"default", None)`, `region_of`/`row_rms`; row order and persistence errors asserted equal to EXP-0061's file).
Official CLI row `results/ds0019_eval_report_gnn_v2_n30.json` (`eval_report.py --models gnn_flex_v2_n30 --device cpu
--truth-scoring image`) gives identical numbers (accuracy 0.2541; 0.9003 / 0.8131 / 0.8352).
**Reuse check:** the original checkpoint at N 200 re-scored here (CPU) reproduces EXP-0061's per-slate slateN exactly
(max |per-slate diff| 0) and accuracy 0.1908; 18 / 16,583 rows differ in row rms (max 0.0067) -- a few mask pixels from
CPU (here) vs GPU (EXP-0061) float differences -- so the reused per-slate file is bit-identical for slateN, not for every row.

slateN, mean of 3 goals [slate-bootstrap 95 % CI] (100 slates):

| model | lyapunov | mass_in_region | signed_mass | all 9 |
|---|---|---|---|---|
| **GNN v2, N 30, visual target (MODEL-0010)** | **0.900** [0.887, 0.914] | **0.813** [0.786, 0.838] | **0.835** [0.814, 0.856] | **0.850** [0.835, 0.863] |
| GNN v2, N 30, particle target (alt) | 0.895 | 0.813 | 0.833 | 0.847 |
| original ckpt, N 30 | 0.841 | 0.692 | 0.705 | 0.746 |
| original ckpt, N 200 (EXP-0061 file) | 0.764 | 0.644 | 0.636 | 0.681 |
| rendering cap, N 30 (true node motion) | 0.945 | 0.888 | 0.906 | 0.913 |
| NFD v2 (MODEL-0008) | 0.961 | 0.925 | 0.933 | 0.940 |
| LF v2 switched (MODEL-0009) | 0.935 | 0.884 | 0.897 | 0.906 |
| random | 0.006 | 0.002 | 0.003 | 0.004 |
| persistence (degenerate ranker) | 0.123 | 0.011 | 0.093 | 0.076 |

**Paired slateN (all 9 cells, Holm):** GNN v2 - original N 30 **+0.103** [+0.087, +0.120] 91/9; - original N 200
**+0.168** [+0.137, +0.204] 91/9; - NFD v2 **-0.090** [-0.104, -0.077] 3/97; - LF v2 **-0.056** [-0.070, -0.043] 21/79
(all p < 1e-4); per vf the signs are the same. Per goal: vs LF v2 random_quadrant +0.001 (unresolved), ring_O -0.109,
T -0.060; vs NFD v2 random_quadrant -0.020 (p 0.036), ring_O -0.158, T -0.092.

**Accuracy** (suspect across model types; slate-cluster CI): GNN v2 **0.254** [0.240, 0.267]; particle alt 0.246;
original N 30 0.188, N 200 0.191; NFD v2 0.549; LF v2 0.469; persistence 0. Paired: v2 - original N 30 +0.066
[+0.060, +0.072]; v2 - N 200 +0.063 [+0.045, +0.082]. **DS-0020 v2 val** (1,665 rows, trajectory CI): GNN v2 0.190
[0.180, 0.200], particle alt 0.193, original N 30 0.108, **original N 200 -0.004 [-0.026, +0.018] (no better than
persistence; 224 / 1,665 val states hit its small-pile fallback)**, NFD v2 0.505, LF v2 0.390.

**Rendering cap at N = 30** (every fg pixel carried by its node's TRUE motion = the displacement of the node's
nearest particle): accuracy **0.363** [0.345, 0.380] on DS-0019, 0.312 on v2 val; slateN 0.913. GNN v2 reaches
**70 %** of the cap in accuracy (cap - GNN +0.109 [+0.095, +0.123]) and the cap is +0.064 [+0.049, +0.078] above it in
slateN. So of the GNN's 0.746 accuracy shortfall from 1, ~0.64 is the 30-node nearest-carry representation and ~0.11
the learned dynamics. **Even the cap (perfect node dynamics) is below NFD v2 in slateN (0.913 vs 0.940).** Cap at
N 100 / 200 on every 10th slate (sanity check, not a reported row): 0.55 / 0.57 vs 0.43 at N 30 on the same subset.

**Small-pile fallback at N = 30: 0 states** -- 0/100 DS-0019 states, 0 of the 20,867 DS-0020 v2 train+val states
(min 67 voxels). (Original ckpt at N 200: 43/100 DS-0019, 224/1,665 val.)

**Strata** (slateN all 9 cells; GNN v2 - LF v2 [CI]): rand_blob 0.840 (-0.060 [-0.083, -0.037]); rand_spread 0.859
(-0.053); pieces small 0.826 (-0.073), mid 0.851 (-0.054), large 0.861 (-0.049). GNN v2 beats both original-checkpoint
rows in every stratum (+0.088 to +0.254) and loses to NFD v2 and LF v2 in every stratum. Accuracy / cap: blob 0.226 /
0.400, spread 0.277 / 0.332 -- the GNN is furthest from its cap on compact blob and small piles.

## Timing (CPU ONLY -- GPU unavailable; `results/timing_gnn.json`, parts in `results/timing_gnn_parts/`)

`code/time_inference.py` unchanged settings (each slate's full pool as one batch, 2 warm-up passes, 3 timed, per-slate
median, 3 separate processes, + batch 128), `--device cpu`, 4 threads, machine otherwise idle.

| config | ms / slate (3 procs) | median | us / cand | batch 128 us / cand |
|---|---|---|---|---|
| **GNN v2 incl. its perception** (`cache_states=False`) | 249 / 270 / 259 | **259** | 1607 | 2029 |
| GNN v2, perception cached (model + render) | 212 / 206 / 207 | 207 | 1278 | 1316 |
| NFD v2 (MODEL-0008), same CPU | 80.8 / 80.6 / 80.7 | 80.7 | 485 | 460 |
| LF v2 switched, CPU (RUN-0002) | | 55.7 | 333 | 366 |

Stage breakdown (`code/profile_gnn_breakdown.py`, sync per stage, staged output asserted == `predict_occ`), ms per slate:
imread 4.8 | segment + back-project 11.6 | voxel downsample (np.unique) 28.9 | FPS 0.7 | recenter 0.8 | s_delta 1.2 |
graph build 4.2 | GNN forward 43.6 | **render 133.0 (55 %)**; total 240. Perception 46.8 ms (19 %), model 49.0 ms (20 %).

**Per-sample Python loops: yes, flagged** (aten-op count ratio B=128 / B=16 = 5.35). Two of them: (1) the renderer's
`for b in range(B)` (one carry + scatter over all fg pixels per candidate) -- the dominant cost; (2) the source model's
`predict_one_step` builds `rels_idx` with `[torch.arange(n_rels[i]) for i in range(B)]`. FPS and recenter loop over the 30
nodes only (1.5 ms, not a problem at N 30). Vectorised replacements exist and are bit-identical (`fast_graph`,
`vector_render`; asserted): graph build 4.2 -> 3.4 ms, but the one-scatter render is SLOWER on CPU (210 ms vs 133 ms:
it materialises a (B, n_pixels, 2) tensor), so on CPU the loop is not the bottleneck -- the per-candidate work over every
foreground pixel is. On a GPU the loop would be launch-bound; not measurable here. The registered spec keeps the default path.

## Timing on CUDA (added 2026-10-02 ~19:30, after the GPU reset; `results/timing_gnn.json`, parts `gnn_cuda_*.json`)

Same `code/time_inference.py` settings as RUN-0001/0002 (each slate's full pool as one batch, 2 warm-up, 3 timed passes,
3 separate processes, + batch 128), RTX 4070 Laptop, inputs verified on `cuda:0`. First attempt failed the harness's
device assertion: the GNN calls `predict_one_step` (bypassing the top-level module's `__call__`), so the forward-pre-hook
saw nothing. Fixed in the harness: if the top-level hooks see no input, ONE extra untimed call is made with hooks on every
submodule to check the device; the timed passes carry the same hook overhead as the NFD/LF runs.

| config | ms / slate (3 procs) | median | us / cand | batch 128 us / cand |
|---|---|---|---|---|
| **GNN v2 incl. its perception** (`cache_states=False`) | 77.2 / 76.9 / 77.7 | **77.2** | 498 | 799 |
| GNN v2, perception cached (model + render) | 42.2 / 43.7 / 43.6 | 43.6 | 259 | 286 |
| NFD v2 (RUN-0001, CUDA) | | 4.88 | 29.1 | 29.8 |
| LF v2 switched (RUN-0002, CUDA) | | 10.4 | 63.7 | |

Per-sample loops confirmed on CUDA (CUDA-kernel count ratio B=128/B=16 = 6.40, aten-op ratio 5.49): the GNN is
launch-bound by its per-candidate render loop and per-sample graph-build loop, not by the network. Perception costs ~34 ms
per slate on CUDA (77.2 - 43.6), most of it CPU-side (imread, segmentation, voxel np.unique). No per-stage CUDA breakdown
was run.

## Status

Complete (GPU timing added after the reset, section above). Not done: seeds > 1 (single seed as instructed, so seed noise of
the GNN-v2 rows is unmeasured; EXP-0061's NFD seed sd 0.002-0.004 slateN and the original-checkpoint FPS-seed spread 0.005-0.016 are
the only references), N other than 30, true-depth inputs (impossible on v2/DS-0019).
