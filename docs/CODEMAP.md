# Code map — where things already are

Derived from an audit of 26 subagent transcripts (949 tool calls, 831k tokens) in which
**74 % of all tool calls were Bash and most of those were navigation**: 216 `grep`, 142
`sed -n`, 125 `cat`, 94 `ls`, 78 `find`, against 180 `python`. Ten separate agents
independently located `transforms/functional.py`; ten found `fit_linear_foresight.py`; eight
each found `METRICS.md`, `check_register.py`, `control_utility_test.py`,
`eval_slaten_broad.py`.

**Check here before grepping.** If what you need is not listed, add it when you find it.
This file is an index, not documentation — it says *where*, not *why*. For why, see
`ARCHITECTURE.md`, `INTERFACES.md`, and the subsystem design docs.

---

## Python environment

**Use `/home/alon/anaconda3/envs/pme/bin/python`** (torch 2.11.0+cu130, CUDA available).
`envs/cge` also has torch but its `cv2` is built against a newer `libstdc++`
(`CXXABI_1.3.15` not found), so `import Baselines.common.goals` — and anything
downstream of it, including every scorer — fails outright there. The system
`python3`/`anaconda3/bin/python` have no torch at all.

## Representation conversions — `transforms/functional.py`

| need | function |
|---|---|
| particles → occupancy grid | `particles_to_occupancy` (use `footprint_radius_voxels` for cube datasets) |
| action → pose `(sx,sy,ex,ey,angle)` | `action_to_pose` |
| render the plate into a channel | `draw_plate_soft` (NFD/Schenck build 2 of these per candidate) |
| warp occupancy into the push frame | `to_push_frame` / `from_push_frame` / `push_frame_transform` / `warp_affine_occ` |
| mask the corners lost to rotation | `push_frame_validity_mask` |
| blend a push-frame prediction back | `blend_push_prediction` |
| Genesis action → camera coords | `genesis_action_to_cam3d`, `genesis_particles_to_cam3d` |

`to_push_frame`/`from_push_frame` compose into the **analytic pose-conditioned transform
`T_a(X)`** — do not write a new warp.

## Linear visual foresight — `fit_linear_foresight.py`

| need | function |
|---|---|
| world action → pixel endpoints | `actions_to_pixels` |
| build the 32×32 canonical visual state | `canonicalise` |
| fit a ridge operator (**toward identity**) | `fit_operator`, `fit_operator_nonneg` |
| warp → operator → unwarp → blend | `predict_world` |
| **the swept-region mask for `accuracy`** | `swept_region_mask` |
| paper's metrics | `metrics`, `contact_score` |

Note the ridge here regularises **toward identity**, unlike `dmdc_baseline`'s toward-zero —
λ values are not comparable across the two.

## Descriptor / DMDc operators — `dmdc_baseline.py`

| need | function |
|---|---|
| analytic occupancy descriptors | `occupancy_descriptors`, `descriptor_slices` |
| per-bin ridge fit (closed form, toward zero) | `fit_per_action_operators` |
| apply per-bin operators | `apply_operators` — **`broken`: allocates `[N,D,D]`, ~138 GB at D=631** |
| multi-step rollout in descriptor space | `rollout`, `multistep_report` |
| is this data actually contiguous? | `diagnose_contiguity`, `build_episodes`, `split_by_episode` |

## Switching by push length — `Baselines/LinearForesight/model.py`

`push_length_m` · `bin_index` · `predict_switched`. Bin scheme (EXP-0003): 6 equal-width bins
over `[0, max]`, `MIN_ROWS_PER_BIN = 50`, in `fit_switched.py`. Bin assignment is recomputed
per call, so a multi-step rollout naturally uses a different operator per step.

## Goals and control scoring

| need | where |
|---|---|
| `mass_in_region`, `signed_mass_in_region` | `Baselines/common/goals.py` |
| region masks (quadrant, letter, random) | `Baselines/common/goals.py` — `quadrant_mask`, `letter_mask`, `random_quadrant_mask`. `letter_mask` does **not** rotate — pass a pre-rotated mask if you need that. |
| mask → legal (non-penetrating) material configuration | `Baselines/common/goal_configs.py::mask_to_configuration` — grid-then-jitter placement of `n_objects` cubes with an exact non-penetration bound (`assert_no_penetration`); falls back to rejection sampling, then to dilating the placement mask, for masks too thin/small at the target object count (e.g. font-glyph strokes) |
| distance field from a mask | `goals.py::dist_field_from_mask` |
| `slateN` capture | `goals.py::slate_n_capture` |
| fixed-K scoring (the K=32 reference every record must report) | `score_pool_kfixed`, in `experiments/EXP-0012-*/code/` |
| Lyapunov value + rank metrics | `control_utility_test.py` — `lyapunov`, `lyapunov_weights`, `rank_metrics` |
| point-mass value readout (COM → world → sample a field) | `com_world_pixel` / `bilinear_sample`, in `experiments/EXP-0006-*/code/stage3-slaten__eval_slaten_latent.py`. **Hand-copied into at least 3 experiments** — reuse it rather than rewriting; the copies have each re-introduced device-placement bugs |
| learned value readout `h(embedding, goal) -> scalar`, any embedding dim | `experiments/EXP-0017-value-readout-instrument/code/value_readout.py::ValueReadout` — fit/predict/save/load + `dv(z_now, z_pred, goal)` for action ranking; capacities `linear|poly2|mlp`, feature modes `diff|concat` (`concat` is the one that works when the state and goal embeddings differ in dimension, e.g. a learned latent). **Computes the three mandatory `value_readout_r2` baselines (`mean_only`/`state_only`/`goal_only`) on every fit** |
| goal dataset: 500 masks + legal same-count configurations | `experiments/temp/goal-states/dataset.pt` (`masks` [500,64,64], `configs` [500,3,20,7]); descriptor features/targets already computed from it in `experiments/temp/desc-value-readout/stage3_features_and_targets.pt` (`phi_state` [1000,87], `phi_goal` [500,87], `slate_of_state`, `values` for 3 value fns) — reuse rather than re-deriving |
| **TRAP:** inner CV folds for ridge-lambda selection must be SLATE-AWARE | EXP-0015's `stage0_upper_bound.py`/`stage3b_regression.py`::`fit_eval_cv` and `persist_final_models.py::cv_lambda` use `np.array_split(rng.permutation(n),3)` — row-random. Post-push states within a slate are near-duplicates, so this selects lambda 100-1000x too small and drove EXP-0015's Fourier-order reversal (confirmed, EXP-0017). Group folds by `slate_of_state`: see `experiments/EXP-0017-*/code/run0001b_cv_leak.py::folds_slate_aware` |
| **grouped** inner CV for the ridge lambda (the fix for `readout-cv-folds-slate-aware`) | `experiments/EXP-0017-*/code/value_readout.py::GroupRidgeCV` + `ValueReadout.fit(..., state_groups=, goal_groups=, cv_group_by="state"\|"goal")` (added by EXP-0018). `RidgeCV`'s default is LOO-GCV, i.e. row-random — never use it on near-duplicate rows |
| 2000-goal library (masks + 3 legal 20-cube configurations each; rectangles, 9 letters, ellipses, two-blob) | `experiments/temp/goal-states/dataset_v2.pt`, generator `experiments/EXP-0018-*/code/stage1_generate_goals_v2.py`. The 500-goal `dataset.pt` is the EXP-0015 original and is untouched |
| diverse after-sweep state cache: 23 520 states from 5 source corpora, 87-dim descriptors, 2000 goal descriptors, all 3 value matrices, per-state corpus/group/push-length labels and the reproducible state index | `experiments/temp/exp0018-value-readout/cache.pt`, built by `experiments/EXP-0018-*/code/stage2_build_cache.py` |
| a worked end-to-end slate scorer | `experiments/EXP-0008-*/code/slaten-broad__eval_slaten_broad.py` — **read it, do not import it**: its import chain pulls in encoders/decoders and sibling temp dirs |

**AXIS CONVENTION (fixed 2026-09-17).** Goal masks, Lyapunov weight fields and
`goal_configs`'s world<->grid mapping are all in the SAME convention as
`transforms/functional.py::particles_to_occupancy`: **row = world x, col = world y**.
Until 2026-09-17 the mask side was row=y/col=x, so every value function multiplied an
occupancy by a TRANSPOSED mask; symmetric targets (`corner`, `center`, quadrants 0/3)
hid it. Only `lyapunov_weights("stripe")` and `quadrant_mask(1)`/`(2)` change output.
Pinned by `tests/test_goal_axis_convention.py`; invariant
`goal-mask-axis-convention-row-y-col-x` is `fixed`. Do NOT "fix" a mask with a `.T` at
a call site.

**Metric definitions are owned by `experiments/METRICS.md`.** Do not redefine a value
function that already has an entry there.

## Fitted objects — `weights/`

Reusable fitted instances, each with `MODEL.md` (what it is) and `tests.md` (what has been
tested on it). **Look here before refitting anything.**

| id | what |
|---|---|
| `MODEL-0001-stage2-visual-switched` | switched + global linear operators on the 32×32 canonical visual state |
| `MODEL-0002-descriptor-only-D-all-local` | switched linear operator on the **94-dim D-all-local push-frame descriptor basis** |
| `MODEL-0003-nfd-multistep-finetuned` | NFD fine-tuned through the 3-step closed-loop rollout (λ=0.7; λ unresolved) |

The 94-dim basis MODEL-0002 was fit on is defined in
`experiments/EXP-0004-*/code/dmdc-lenbins__descriptors_d.py` (`push_frame_full_descriptors`).
A descriptor model is only comparable to MODEL-0002 if it predicts **that** basis — see the
comparability rule in `experiment-log`'s Metrics section. Pixel-space linear operators from
the LinearForesight line live under `Baselines/LinearForesight/runs/`, not in `weights/`.

## LeJEPA-style latent encoder + switched-linear dynamics — `experiments/EXP-0016-*/code/`

The design doc is `docs/experimental_design/jepa_based_encoder.md`. A working,
experiment-local implementation of its §3.1/§5/§8-10 already exists — **reuse it
rather than rewriting the architecture**:

| need | where |
|---|---|
| residual-CNN occupancy encoder (configurable `input_resolution`/`latent_dim`/`n_res_blocks`) | `EXP-0016-*/code/model.py::ResCNNEncoder` |
| action encoding `[x,y,sin,cos,dx,dy]` + small MLP | `model.py::encode_action`, `ActionEncoder` |
| residual switched-linear latent dynamics with a soft softmax gate | `model.py::SwitchedLinearDynamics` (K=1 reduces exactly to single-linear) |
| post-hoc `z -> 64x64` occupancy decoder | `model.py::OccDecoder` |
| fast corpus -> frozen-latent cache | `code/encode_corpus.py` (98 k transitions in 88 s) |
| K x seed dynamics sweep + `Delta z = 0` baseline + effective rank | `code/fit_dynamics.py` |
| slateN on DS-0001 through an image readout, EXP-0014's path | `code/score_slaten.py` — reproduces EXP-0014's `persistence`/`random` rows exactly, so it is the cheapest way to check a new scorer is on the same scale |

**Known floor (EXP-0016): a FROZEN RANDOMLY-INITIALISED encoder already gives
K=8 > K=1 in latent R^2 (+0.373 vs +0.248, cross-seed sd ~0.002) and slateN
+0.42 on DS-0001.** Any "switched beats single" result from a learned encoder
must be reported against this floor or it says nothing.

## Baseline predictors

`Baselines/NFD/predictor.py` (`NFDPredictor`, `build_predictor`), `Baselines/GNN/predictor.py`,
`Baselines/SchenckCNN/`. All expose `predict_occ(batch) -> (B,H,W)` — an **offline
slate-scoring** interface, not the live MPC adapter contract.

`simple_mpc/adapters.py::make_adapter` supports **only** Eulerian wrappers and
`PropNetDiffDenModel`; it raises `NotImplementedError` for NFD, Schenck, and the fitted linear
operators. Offline scoring works for those; live MPC does not.

## Timing

`Baselines/common/benchmark_time.py` — correct GPU methodology (warm-up, `cuda.synchronize()`
either side, median+IQR, device introspected, contention gate). `simple_mpc/benchmark.py`
times adapters only.

**Known trap:** `Baselines/common/eval_baseline.py::_predictor_batch` never moves
`PredictorBatch` off CPU, so NFD/Schenck time on CPU while GNN forces CUDA — assert the device
of the tensors actually fed to each forward.

## Datasets

| corpus | shape |
|---|---|
| `Genesis/data/overnight_randlen` | 213 `.pt`, 512 transitions each; keys `states,states_,p_starts,p_stops,angles`; push length = `‖p_stop−p_start‖`, 0.5–79.6 mm. **Independent transitions, not chains.** |
| `Genesis/data/slates_multistep/{n20_L10mm,n20_L20mm,n20_L40mm}` | 50 slates × 3 steps × 128 envs, one file per (slate, step) + `manifest.json`. Sim is **not** reset between steps, so each row is a real 3-step trajectory; row index is the trajectory id. **Single push length per corpus** → 4 of 6 length bins get zero data. `n20_L10mm` is ruled problematic. |

| `Genesis/data/slates_binned/*` | `binned_slate_collection.py` output: `step{k}.pt`, rows indexed by chain id `c = slate_idx*n_actions + action_idx`, same row order in every step file. Spawn style (`scatter` = `drop`, one layer; `pile` = `heap`/`pyramid`) is on `BinnedSlateCorpus.spawn_style`. Push length drawn from 5 bins over 20-70 mm and **mixed along each chain**, so all length bins get data. Extra keys `len_target, len_realized, bin_requested, bin_realized` (the last is `-1` = `UNDERFLOW_BIN` for a wall-shortened push below 20 mm, kept but excluded from bin 0). |

| `Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm` | **n20 and scatter-spawn only** — no piled variant. 20 slates × 1000-candidate pools, `n_steps=1` (so "first step only" is automatic). Pool size differs from `slates_multistep`'s 128, and `slateN` is **not comparable across pool sizes** — report a fixed K reference. |
| `Genesis/data/Sean/<mode>-<timestamp>/<mode>/cube/...` | Three spawn modes (`inbetween`, `piled`, `scattered`), **1230 `*_data.pt` files, 96 rows each, 206,240 transitions**. Mixes **n20 / n50 / n100** particle counts (n50 and n100 dominate) — every slate corpus is n20, so this is a distribution shift. **Banded** push lengths: 33 distinct per-file bands, near-uniform coverage across 20-30/30-40/40-50/50-60/60-70 mm (~38-42k rows each), light tails below 20 mm, full range 0-70 mm. Directory also holds `.zip` archives — glob `*_data.pt` only. |
| `Genesis/data/overnight_randlen_{train,test}` | pre-split corpora (192/… train files, 15 test). Use these rather than re-splitting `overnight_randlen`. **`train` is five subdirectories** — `mixed_n20` (45 files), `piled_n20` (45), `piled_n50` (12), `scattered_n20` (45), `scattered_n50` (45), 512 transitions each — so glob `*/ *_data.pt`, and the spawn mode / particle count is in the directory name, not the file. |
| `Genesis/data/slates_multistep/*/manifest.json` | `batches` is a list of `{batch_idx, slate_idx, step_idx, env_count}` — the ONLY mapping from `_{batch_idx}_data.pt` to (slate, step). The 3 steps of one slate are one trajectory, i.e. near-duplicates: group by `slate_idx` in any split or CV fold. |

**Two rasterisers exist.** `Baselines/common/data.py::load_cell` returns `slates_multistep`'s
**official** occupancy, built by `PileSweepData._draw_particle_grid` (box/quaternion-aware);
`transforms.functional.particles_to_occupancy` is a point-splat. `data.py` warns they are not
interchangeable. **Measured**: they correlate r=0.998 and move pooled slateN by 0.002
(0.822 vs 0.820) — so the choice is not a plausible cause of a large scoring discrepancy, but
state which you used. `slates_binned` ships particle states only and has no official
rasteriser to cross-check against.

**Slate difficulty is not comparable across corpora.** `slates_multistep`'s `dv_true` spread
is std 0.134 — it contains both very easy and very flat slates, and the easy ones drive high
capture scores. `slates_binned` is uniformly moderately hard (std 0.041) with no easy slates,
and every model measured on it so far scores near zero. A `slateN` from one corpus should not
be read as the same quantity as a `slateN` from the other.

**LOADER TRAP — `load_randlen_cell()` is slow.** It materialises transitions one at a time
and takes **>10 min per call** on `overnight_randlen_train` (24.5k transitions), which is fatal
for any job that loads more than once. For bulk fitting, glob `*_data.pt` and `torch.load`
directly — see `experiments/EXP-0004-*/code/dmdc-lenbins__descriptors.py` for the fast path
(full 213-file corpus rasterised in ~40 s). **Exclude `*_failed.pt`** (failed simulations) and
the `.zip` archives.

Check `manifest.json` for `n_steps` and pool size before assuming a corpus's shape.
Load chains from the raw `_{batch}_data.pt` files. The `*_eval` configs apply
`min_push_length_m` filtering that drops rows and breaks row↔env alignment across steps.

## Reusable states-only datasets (no transitions) — `datasets/DS-0002`, `DS-0003`

The encoder (self-supervised on single-state views) and the geometric value
function need only STATES, never transitions — but every corpus above is
either slate-shaped (near-duplicate rows) or unchecked for cross-file
duplication. These two datasets exist so those two components can be trained
on genuinely distinct states, grouped so a CV/train-val split never leaks a
near-duplicate across the boundary (the `readout-cv-folds-slate-aware` bug).

| need | where |
|---|---|
| 304,655 distinct real states pooled/deduplicated across every corpus above, with `n_objects`/corpus/source-file/spawn-mode/role/**group** on every row | `datasets/DS-0002-real-distinct-states/` (builder: `datasets/DS-0002-real-distinct-states/build.py`; 631,612 raw candidate rows collapsed 51.8% under a 1mm-tolerance sorted-position hash) |
| exact yaw-aware overlap/legality test for cube footprints (SAT, replaces the old axis-aligned bound for placement work) | `Baselines/common/cube_overlap.py::overlaps_pairs`, `any_overlap_matrix`, `state_is_legal(xy, yaw, size, tol)` |
| synthetic pile-state generator (2026-09-17, current): grid seed + per-object relaxation toward the centroid, `compaction` in [0,1] sets sweep count, `yaw_kappa` von-Mises-concentrates yaws, legality via `cube_overlap.state_is_legal` | `Baselines/common/pile_compaction.py::sample_compacted_state` — returns a CENTRED pile; placement inside the workspace is the caller's job (see `generate.py::_place_in_bounds`) |
| synthetic pile-state generator (RETIRED for DS-0003, function still importable): scattered (grid+jitter) mixed with sequential touching clumps, legality via the old axis-aligned bound | `Baselines/common/goal_configs.py::sample_synthetic_state` (+ `_place_clump`, `_place_scattered_point`, `_penetrates`, `_sample_placement_region`) — that bound rejects 91-99% of real DS-0002 states and cannot express contact, so nothing could ever be compacted against it; see `cube_overlap.py`'s module docstring |
| synthetic states corpus + regeneration command | `datasets/DS-0003-synthetic-states/` (driver: `datasets/DS-0003-synthetic-states/generate.py`) |
| real-vs-synthetic occupancy validation (fast rasteriser + `dmdc_baseline.occupancy_descriptors`) | `datasets/DS-0003-synthetic-states/validate.py` |

## Collecting new data

Which driver produces which corpus shape, the `SandboxManipulation` seams they
compose, and the traps (shortened pushes, cuda default device, batch-index
numbering): the `data-collection` skill. Reading a binned slate corpus back —
trajectories, per-slate sequences, per-bin sweeps, integrity checks — is
`Genesis/binned_slate_dataset.py::BinnedSlateCorpus` (Genesis-free; do not
re-derive the grouping inline). `binned_slate_collection.py` is the one
that separates action sampling from simulation so each simulated batch is
length-homogeneous; `_pile_aware_stops` accepts a per-env `push_length` tensor
for it.

## Pool diagnostics / figures

| need | where |
|---|---|
| the standard per-slate figure (dv histogram, action pool on occ0, chosen actions, \|R_K\| vs K, rank-vs-rank, true-percentile bars) | `scripts/probes/pool_inspect.py` |
| which slates to plot (typical / worst / near-tie per model) | `scripts/probes/pool_survey.py` |
| cache loading, `rk_curve`, `action_geometry`, `swept_rectangle_corners`, `MODEL_COLORS` | `scripts/probes/pool_common.py` (reads an embedded `occ0`/`ws_min`/`ws_max` when the cache carries them) |
| a dV cache for `Genesis/data/slates_binned/*` (persistence, random, MODEL-0001/2/3) | `scripts/probes/binned_pool_cache.py` |

**Known trap:** `persistence` predicts `dv=0` for every candidate, so as a *ranker* it is
degenerate — `argmin` always returns row 0 and the induced order is row order. Use `random`
as the ranking floor. Separately, MODEL-0002's point-mass readout saturates on
`slates_binned`: 26 % of predictions are exactly 0 (predicted COM inside the target region,
where the distance field is 0), so ~26 % of a slate ties at the minimum.

## The evidence layer

`experiments/REGISTER.md` (claims) · `INVARIANTS.md` (tags) · `METRICS.md` (metric formulas) ·
`COMMANDS.jsonl` · `TEMP_LOG.md` · `scripts/check_register.py` (must exit 0) ·
`scripts/run_probe.py` (**`broken`: writes to `runs/COMMANDS.jsonl`, not the documented
`experiments/COMMANDS.jsonl`**).

Skills: `experiment-log` (storage, tiers, budgets), `register-validator` (frontmatter and
grades), `project-overview` (module ownership), `subagent-experimenter` (running a contained
experiment).

## LeJEPA Stage 1 (actually training the encoder) — `experiments/EXP-0019-*/code/`

EXP-0016 froze a random encoder; **EXP-0019 is where Stage 1 of
`docs/experimental_design/jepa_based_encoder.md` is actually run.**

| need | where |
|---|---|
| the SIGReg regulariser | `le-wm/module.py::SIGReg` — vendored, **dependency-free** (torch + einops only). `stable_pretraining` is NOT installed and is NOT needed. `le-wm/jepa.py`/`train.py` around it are a ViT/CLS design — read for the loss structure, do not copy the architecture |
| LeJEPA Stage-1 training loop (alignment + SIGReg, projector, per-epoch collapse monitoring) | `EXP-0019-*/code/train_encoder.py`. `--sigreg-on-z` applies SIGReg to `z` as well as to the projection `p` (implemented, **not yet run**) |
| physically-meaningful views (particle dropout, footprint-radius jitter, occupancy noise; NO rotations/flips/crops) | `EXP-0019-*/code/raster.py::make_views` |
| **fast rasteriser** — batch-vectorised disk splat, per-sample radius | `EXP-0019-*/code/raster.py::rasterise`. **Bit-exact** against `transforms.functional.particles_to_occupancy` (asserted by `check_raster.py`) and **77x faster** at B=64 — the project one loops `for b in range(B)` in Python in both its `footprint_radius` and `sigma>0` branches |
| corpus -> padded particle tensors, fast path, handles Sean's mixed n20/n50/n100 | `EXP-0019-*/code/data.py` (`randlen_files`, `sean_files`, `load_transitions`, `load_states`) |
| encode a corpus with a TRAINED encoder checkpoint | `EXP-0019-*/code/encode_states.py` |
| **HARD push-length gate** switched-linear latent dynamics, closed-form ridge per bin | `EXP-0019-*/code/fit_switched_hard.py` — replaces the design doc's §10 soft softmax gate; bins via `Baselines/LinearForesight/model.py::bin_index` |
| trained encoder checkpoint (the reusable deliverable) | `EXP-0019-*/artifacts/RUN-0001/encoder_lejepa.pt` — `state_dict` + `config` + projector + full training curves |
| fitted operators (global + 6 hard bins) with their config | `EXP-0019-*/artifacts/RUN-0003/operators.pt` |

**Known result (EXP-0019): LeJEPA Stage 1 does NOT variance-collapse (per-dim
std rises 0.045 -> 0.162) but the EFFECTIVE RANK of z falls 44.6 -> 7.5 of 256,
because SIGReg is applied to the projection `p`, not to `z`.** The trained
encoder's latent also transfers much worse across held-out files than a random
one (train-test latent-R^2 gap 0.28 vs 0.03 under an identical fit).

## LeJEPA SIGReg placement — `experiments/EXP-0020-*/`

EXP-0020 ran EXP-0019's never-used `--sigreg-on-z` flag with everything else held
fixed (same 252 files, 12 epochs x 250 steps, bs 192, lr 1e-3, lambda 0.02, seed 0).

| need | where |
|---|---|
| encoder trained with SIGReg on BOTH `p` and `z` | `EXP-0020-*/artifacts/RUN-0001/encoder_lejepa.pt` (state_dict + config + projector + `train_args` + curves) |
| its latent cache / operators | `EXP-0020-*/artifacts/RUN-0002/latents_{train,test}.pt`, `RUN-0003/operators.pt` (row-random lambda), `RUN-0004/operators_grouped.pt` (file-disjoint lambda) |
| **file-disjoint grouped inner CV for the latent ridge lambda** | `EXP-0020-*/code/fit_switched_grouped_lam.py` — imports `fit_switched_hard.py` unchanged and replaces only the inner split (fold of row i = `(i // 512) % 5`; `overnight_randlen` has exactly 512 rows per `_data.pt` file, so `i // 512` is the file id). Use this, not `fit_switched_hard.py`'s row-random split |

**Known result (EXP-0020): putting SIGReg on `z` does NOT fix the rank collapse
— effective rank of z still ends at 7.3 of 256 (EXP-0019: 7.5; 44.6 at init),
so the design doc's SIGReg *placement* is NOT the cause.** It nonetheless raises
latent R^2 from +0.033/+0.060 to **+0.292 global / +0.413 hard-6-bin switched**,
past EXP-0016's frozen-random floor of +0.209/+0.269, and cuts the train-test
latent-R^2 gap 0.30 -> 0.05. **Effective rank and latent R^2 therefore disagree
about which encoder is better — do not use effective rank alone as a proxy for
representation quality here.** Unresolved anomaly: effective rank measured on
the 98304-row train latent cache is 18.6 (EXP-0020) vs 41.7 (EXP-0019), the
reverse of the 8192-state training-monitor ordering.

**Validator quirk:** `scripts/check_register.py` line 136 warns "ran with a dirty
tree" unless `downgrades` contains the substring `dirty`, but `dirty` is not in
its own `DOWNGRADES` vocabulary (line 33) — so the warning cannot be cleared by
any valid record. Every record in the repo carries it. It is a warning, not a
failure; `check_register.py` still exits 0.
