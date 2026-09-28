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
| model-agnostic warp→predict→unwarp→blend, for any `fn` on the canonical frame | `push_frame_roundtrip` |
| render start/stop plate channels directly in the canonical frame (no world render + warp) | `canonical_plate_channels` |
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

**HARD gate carries no length gradient.** `bin_index` is `torch.bucketize`: fine for offline
scoring, useless as a gradient source (d(bin)/d(length) = 0 a.e., undefined at the edges). For
gradient-descent MPC use `soft_bin_weights` / `predict_switched_soft` (EXP-0023): triangular
interpolation between the two ADJACENT BIN CENTRES, i.e. linear interpolation of the per-bin
operators in push length. It agrees EXACTLY with the hard gate at every bin centre.
`predict_switched_soft` also pays the warp ONCE for the whole batch (all operators applied to
the same canonical occupancy, then length-weighted) instead of once per bin, and has no masked
in-place assignment for autograd to survive. **It is a RELAXATION of the fitted model, not the
fitted model** — the hard gate backs every existing register row and is left untouched; anything
scored through the soft gate must say so.

## Goals and control scoring

| need | where |
|---|---|
| `mass_in_region`, `signed_mass_in_region` | `Baselines/common/goals.py` |
| region masks (quadrant, letter, random) | `Baselines/common/goals.py` — `quadrant_mask`, `letter_mask`, `random_quadrant_mask`. `letter_mask` does **not** rotate — pass a pre-rotated mask if you need that. |
| mask → legal (non-penetrating) material configuration | `Baselines/common/goal_configs.py::mask_to_configuration` — grid-then-jitter placement of `n_objects` cubes with an exact non-penetration bound (`assert_no_penetration`); falls back to rejection sampling, then to dilating the placement mask, for masks too thin/small at the target object count (e.g. font-glyph strokes) |
| two-disconnected-squares goal (`two_squares`, opt-in; not in the 30-goal `many` set — use `--goal-set many_plus`) | `goals.py::two_squares_mask`; resolved by `eval_report.py::_fixed_goal` (EXP-0046) |
| distance field from a mask | `goals.py::dist_field_from_mask` |
| `slateN` capture | `goals.py::slate_n_capture` |
| value-function SENSE (cost vs value) by name; sense-safe `improvement`, `best_of`, and EXP-0023's `gradient_gain`/`pool_escape`/`regret_vs_oracle`/`capture_vs_oracle` | `goals.py::higher_is_better_for`, `improvement`, `gradient_benchmark_metrics` — `higher_is_better` is a REQUIRED keyword (TODO M4, EXP-0026) |
| **paired model comparison over states** — per-pair bootstrap CI, sign-flip p + Holm, Friedman/ANOVA, rank stability, `required_n` for sizing a benchmark | `Baselines/common/paired_stats.py` (EXP-0026). Input `(models, states)`, higher = better; the STATE is the replication unit |
| run the cross-model harness on GPU | `Baselines/common/eval_report.py --device cuda` (`_predict`, chunked; identical to CPU for NFD/linear; GNN always runs unchunked because its node sampling is seeded by batch position -- `gnn-node-sampling-consistent-within-state`, broken) |
| score a model with a DIFFERENT checkpoint in the cross-model harness (e.g. a training seed) | `Baselines/common/eval_report.py --ckpt MODEL=PATH` (repeatable). **TRAP:** setting the spec's env var (NFD_CKPT, GNN_CKPT, ...) does nothing -- `_load_predictor` overwrites it with `spec['ckpt']` (EXP-0036) |
| per-slate `slateN` (not just the mean) from the cross-model harness | `Baselines/common/eval_report.py::_capture_report` output keys `per_slate[goal][vf]` + `slate_ids` (EXP-0026) |
| closed-loop MPC with a learned model (rank / gd / cem / mppi under a wall-clock budget), and the push legality projection | `simple_mpc/learned_mpc.py` -- `plan`, `run_episode`, `project_push` (tests: `tests/test_learned_mpc.py`). `mpc.py`/`make_adapter` still cannot drive NFD/linear models; use this |
| **capacity-aware value functions** (lyapunov is linear in the image = transport to the goal set with UNLIMITED capacity) and uniform-coverage metrics | `simple_mpc/value_functions.py` -- `SlicedEMD` (V1), `linearised_emd_field` (V2: OT dual potential from the current state, a drop-in distance field), `CrowdingPenalty` (V3), `coverage_metrics` + `uniform_layout` (ceiling); selected by `ModelObjective(value="emd"/"emd_lin"/"crowd", mask=...)` or the batched CLI cell key `value` (EXP-0055; tests `tests/test_value_functions.py`) |
| **goal-aware candidate selection** from a pile-aware bank (geometric carry model, no learned model) | `simple_mpc/goal_aware_sampling.py` -- `carry`, `select_misplaced`, `select_deposit`, `ot_proposals`; batched CLI cell key `sampler` = pile / misplaced / deposit / ot_mix (EXP-0056). **TRAP:** Genesis sets torch's default device to cuda; the selectors pin themselves to the CPU |
| **human demonstrations on the benchmark tasks** (pure human: drawn push executed after `project_push`, no refinement/undo; oracle.py sim setup; per-task JSON) | `human_benchmark_gui.py` + `experiments/EXP-0058-human-demonstrations/code/analyse.py` (compares with EXP-0057 oracle cells). Not `human_mpc_gui.py`, which refines the push in the simulator |
| batched closed-loop runner CLI (cells = JSON overrides: planner params, `mass_weight`, `value`, `sampler`, `push_len`) | `experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py`. Logs cumulative sample / select / execute seconds per step. Every chunk plans for all 32 envs (padding included), so a partial chunk costs as much planning time as a full one |
| **scoring ground truth from particle states** (mass-conserving soft splat; use for every TRUE dv) | `transforms/functional.py::splat_particles_mass`; slate grid: `simple_mpc.adapters.occ_for_scoring`; corpus frame: `Baselines/common/eval_report.py::truth_for_scoring` (default `--truth-scoring soft`). **TRAP:** the hard rasterisers (`occ_from_particles`, `PileSweepData._draw_particle_grid`) alias -- a particle covers 4-6 px by sub-pixel position -- and swing image-based dv by up to ~40% of a push's effect under ~1 mm of motion. See METRICS.md "Ground-truth scoring" |
| every simulated push outcome, reusable across goals/value fns (final particle states, keyed by execution path + state hash + action) | `simple_mpc/gt_bank.py::GroundTruthBank`, payload DS-0004. Never mixes `sim_path`s (corpus collection vs `rollout_candidates` disagree, r=0.959) |
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

**TRAP -- `accuracy` is a ratio of population means, never a mean of per-row
ratios.** `accuracy = 1 - mean(err_pred) / mean(err_pers)`
(`fit_linear_foresight.py::metrics`). Computing `err_pred_i / err_pers_i`
per row and then averaging blows up silently into the millions whenever a
single row's swept-region persistence error is near zero (a push that barely
moved anything in the scored band) -- hit twice independently (EXP-0022
RUN-0012's wall-proximity stratification, RUN-0013's length stratification
under canonical-frame scoring). The fix used both times: aggregate
`err_pred` and `err_pers` SEPARATELY within whatever group you are
stratifying by (a stratum, a bootstrap resample, the whole population), and
only then form the ratio -- exactly the order of operations `metrics()`
already uses at the whole-population level. Use a bootstrap over rows for
uncertainty on a per-stratum `accuracy` or its delta between two models, not
a naive SEM of a per-row ratio. See
`experiments/EXP-0022-warped-nfd-push-frame/code/wall_proximity_stratified_eval.py`
(`per_transition_errors`, `accuracy_from_errors`, `bootstrap_delta`) for the
worked, reusable implementation.

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

**Warped NFD, Residual NFD, and Flow NFD moved out of `Baselines/NFD/` to
`model/warped_nfd/`, `model/residual_nfd/`, `model/flow_nfd/` respectively**
(each is a new model family with its own multi-file code, not a minor
parameter change over the NFD baseline, so it belongs under `model/` per
project convention — `Baselines/NFD/` now holds only the plain NFD baseline:
`nfd_lib.py`, `predictor.py`'s plain `NFDPredictor`, `train_nfd.py`, its
`configs/`, `SPEC.md`, `LOG.md`). Module paths below are current as of that
move; anything in `experiments/EXP-0022-*` or `experiments/EXP-0025-*`
citing the old `Baselines/NFD/*_lib.py` paths predates it (each record notes
this).

**Warped NFD** (`model/warped_nfd/lib.py`): the same NFD UNet, predicting in the
CANONICAL PUSH FRAME (`transforms.functional.push_frame_roundtrip`) instead of the world
frame. Registers dataset `nfd-genesis-3ch-warped` (`PileSweepData3ChWarped` + a
`push_px`-carrying `EulerianDatasetWrapper` subclass) and model `nfd-unet-warped`
(`WarpedNFDWrapper`; config knobs `plate_mode: canonical|warp`, `wall_channel: bool`,
`canon_res`, `scale`). The canonical-frame input-stack builder
(`model/warped_nfd/predictor.py::build_canonical_stack`) is shared verbatim between
`WarpedNFDWrapper` (training) and `WarpedNFDPredictor`/`build_predictor_warped[_walls]`
(eval), so the two paths cannot drift apart — training/eval parity measured at ~1e-6 world-
frame prob diff. See `model/warped_nfd/WARPED_NFD_NOTES.md` for the coordinate-convention traps
(push_frame's (col,row) vs draw_plate_soft's (row,col) point order) and the plate_mode timing
derivation. `Baselines/NFD/train_nfd.py` imports both `Baselines.NFD.nfd_lib` and
`model.warped_nfd.lib`. `model/warped_nfd/predictor.py` stays importable without
`Genesis.training.dataset`/the dataset registry, same as `Baselines/NFD/predictor.py` — `model/`
as a whole is Genesis-free, so this property is load-bearing, not incidental.

**Residual NFD** (`model/residual_nfd/lib.py`, EXP-0022 R1/R2): image-space RESIDUAL
PARAMETERISATION, not a residual objective — the loss stays `eulerian_combined` world-frame MSE
against the absolute `occ1` target, unchanged. Registers model `nfd-unet3ch-residual` (R1,
unwarped control; reuses the existing `nfd-genesis-3ch` dataset unchanged) and
`nfd-unet-warped-residual` (R2, warped; reuses the existing `nfd-genesis-3ch-warped` dataset
and `model/warped_nfd/predictor.py::build_canonical_stack`, unchanged). Both wrap a `UNetModels_modular.UNet`
with `residual: false` (its OWN occ0-into-logit skip OFF — asserted, not silently coerced, to
avoid double-counting) and instead apply an EXPLICIT `tanh` head (bounded, signed [-1,1] —
`sigmoid` cannot express a negative delta) to get a residual, add it to `occ0`, clamp to [0,1],
then convert back to a logit via the same safe-inverse-sigmoid trick `WarpedNFDWrapper` already
uses, so the unmodified loss can still score it. R2's mechanism: warp the PRISTINE `occ0` into
the canonical frame, predict the residual THERE, unwarp the RESIDUAL FIELD (not a reconstructed
occupancy) back to world, add to the SAME pristine `occ0`, clamp — **no validity-mask blend**,
because `grid_sample`'s zero-padding already means "no change" for a residual field (unlike for
an absolute-occupancy field, where the same zero-pad would wrongly read as "no material").
**Verified, not just argued** (`results/residual_pilot.md`): this bit-exact preservation holds
ONLY where `push_frame_validity_mask` is EXACTLY 0 (~21-23% of pixels); a thin partial-coverage
boundary band (~1.3% of pixels, mask strictly between 0 and 0.5) is NOT bit-exact and can carry
substantial deviation — bilinear interpolation weights are non-negative, so a zero row-sum
(mask=0) forces every individual weight to zero, but a small positive row-sum does not.
Eval-time predictors: `model/residual_nfd/predictor.py` (`build_predictor_residual_unwarped`,
`build_predictor_residual_warped`), registered in `Baselines/common/eval_report.py`'s `MODELS`
as `nfd_residual_unwarped_L20mm_pilot`/`nfd_residual_warped_L20mm_pilot`.
`Baselines/NFD/train_nfd.py` also imports `model.residual_nfd.lib`.

**Flow NFD** (`model/flow_nfd/lib.py`, EXP-0025): predicts a per-pixel BACKWARD displacement
field (dx, dy) and warps the current occupancy through it via `grid_sample` — mass-conserving
and zero-flow-is-identity by construction. Registers model `nfd-flow-warp`; reuses the
existing `nfd-genesis-3ch` dataset unchanged. Eval-time predictor in
`model/flow_nfd/predictor.py`, registered in `Baselines/common/eval_report.py`'s `MODELS` as
`flow_baseline`/`flow_coarse16`/`flow_smalldisp4`/`flow_largedisp24`/`flow_srcsink`/
`flow_supervised*`. `model/flow_nfd/supervised.py` adds direct particle-correspondence
supervision for the flow head (`build_flow_targets_for_split`, `augment_flow_batch_x8` — the
x8 flip/rotation augmentation extended to rotate the flow field's vector components
correctly, not just permute array layout). `Baselines/NFD/train_nfd.py` also imports
`model.flow_nfd.lib` (not `model.flow_nfd.supervised`, which is driven by
`experiments/EXP-0025-*/code/train_supervised_flow.py` directly).

`simple_mpc/adapters.py::make_adapter` supports **only** Eulerian wrappers and
`PropNetDiffDenModel`; it raises `NotImplementedError` for NFD, Schenck, and the fitted linear
operators, because those are not driven from a rendered observation at all.

**Occupancy-grid GRADIENT adapters (EXP-0023)** — `simple_mpc/adapters.py`, second half.
`make_occ_adapter(model_id, device, goal_shape)` + the `OCC_ADAPTERS` registry are the
LIVE-GRADIENT counterpart of `Baselines/common/eval_report.py`'s `MODELS` dict: **adding a
model is one entry.** They close the gap the paragraph above describes for the 64×64 /
±64 mm slate world: `predict_step(occ, act) -> occ1` is **differentiable w.r.t. the world-metre
action `[sx,sy,ex,ey]`** (yaw derived by `action_to_pose`, exactly the corpus's own `angles`
convention), and `dv(occ0, act)` is the Lyapunov cost the optimiser minimises.

| need | where |
|---|---|
| a grad-carrying adapter for a registered model | `make_occ_adapter` / `OCC_ADAPTERS` |
| the slate occupancy convention, shared with `binned_pool_cache.py` | `OCC_BOUNDS`, `OCC_GRID`, `OCC_FOOTPRINT_RADIUS`, `occ_from_particles` |
| the `PileSweepData` geometry stub a predictor needs without a dataset | `SlateRawStub` (promoted from `experiments/temp/multistep-rollout/rollout.py::RawStub`) |
| assert the `dv` sign convention in code | `assert_dv_convention` — checks the adapter's declared `value_fn`/`higher_is_better` against `goals.higher_is_better_for` AND goal geometry |

**No prediction math is duplicated for the NFD family.** `PredictorGradientAdapter` calls the
existing predictor's OWN `predict_occ` through `__wrapped__` — the function `@torch.no_grad()`
decorated — so the adapter runs the identical forward offline eval runs, with grad mode simply
left on. If a predictor's math changes, its adapter changes with it. (`.cpu()` inside those
predictors is differentiable, so it does not break the chain.)

**TRAP:** `predict_occ` is `@torch.no_grad()`. Calling it directly gives a detached tensor and
an optimiser that silently does nothing. `experiments/EXP-0023-*/code/verify_gradients.py` is
the reusable gate: it asserts finite, non-zero gradients in EVERY action component and 100 %
sign agreement with a directional finite difference, for every registered arm.

**Canonical-frame scoring (EXP-0022 A1).** `Baselines/common/eval_report.py --canonical-frame`
adds swept-region `accuracy` scored in the CANONICAL push frame as an ADDITIONAL column
(`accuracy_canonical`), alongside the unchanged world-frame `accuracy` -- never a replacement.
Not symmetric: a native-canonical predictor (`model/warped_nfd/predictor.py::WarpedNFDPredictor
.predict_occ_canonical`, added for this) is scored on its own canonical output with ZERO extra
resamplings; every other model's already-computed world-frame prediction is warped ONCE
(`to_push_frame`) to be scored here -- see `_accuracy_canonical`'s docstring. Any predictor
that wants the fair (zero-extra-resample) treatment in this frame should expose a
`predict_occ_canonical(batch) -> (pred_canon, start_px, end_px, canon_res, scale)` method;
`eval_report.py` uses it via `hasattr` and falls back to warping the world prediction otherwise.

## Retrieval-based transition model — `model/retrieval/` (EXP-0059)

Groundwork for `docs/experimental_design/retrieval_based_modeling.md`. Genesis-free
(torch/numpy/scipy only). Particle-in/particle-out contract
(`predict_particles(states0 (B,n,7), p_start, p_stop) -> states1`) — deliberately NOT
`simple_mpc.adapters.OCC_ADAPTERS`'s occupancy-in/out contract, since a retrieval model
needs the query's raw cube positions to retrieve against.

| need | where |
|---|---|
| metre-space push-frame transform (origin at push START, +u along push direction, +v lateral) + inverse, tray corners in that frame | `model/retrieval/frame.py::world_to_push_frame`/`push_frame_to_world`/`tray_corners_push_frame` (NOT `transforms.functional.push_frame_transform` — that one is image-centred/pixel-indexed, origin at the push MIDPOINT, and only warps rasterised occupancy) |
| yaw <-> quaternion (this repo's `(w,x,y,z)` order) | `model/retrieval/frame.py::yaw_from_quat`/`yaw_to_quat` |
| flat per-object transition bank over DS-0008+DS-0010, canonicalised into each transition's own push frame; `moved` mask (threshold parameter) + `moved_count_histogram`; a soft `MAX_RECOMMENDED_BANK_SIZE=50_000` warning (2026-09-28: the retrieval SET is designed for <=~50k transitions — more data may be collected, but a subset is chosen for the actual bank) | `model/retrieval/bank.py::TransitionBank.build`/`.from_states`, persisted by `.save`/`.load` |
| configurable push-frame distance (window, optional cap + separate mismatch penalty, corridor upweighting for cubes the plate actually sweeps, optional wall-distance feature, mean/sum reduction) + an EXHAUSTIVE GPU-batched top-k search over the whole bank (chunked over bank size, whole query batch at once — no per-sample Python loop for the search itself) | `model/retrieval/distance.py::DistanceConfig`, `topk_search` |
| reference/configurable predictors: identity (accuracy=0 by construction) and whole-bank k-NN retrieval (Hungarian per-cube correspondence transfers each neighbour's push-frame displacement/yaw-delta, gated by the donor cube's own `moved` flag), aggregations `nn1`/`cube_median`/`cube_weighted_mean`/`occ_mean`/`occ_weighted_mean` | `model/retrieval/predictor.py::PersistencePredictor`/`RetrievalPredictor` (`NearestTransitionPredictor` = k=1/nn1 alias) |
| **hedged occupancy is scoring-only** (2026-09-28 user guidance): `predict_particles` always returns ONE clean, committed-to state (top-1 for `occ_*` aggregations); the k-NN occupancy blend is exposed only via the optional `predict_occ` (same `hasattr` pattern as `WarpedNFDPredictor.predict_occ_canonical`), and `eval_retrieval.py` calls it ONLY for `accuracy_1` — rollout/`slateN` always call `predict_particles`. Per-cube overlap is NOT resolved (documented limitation, not fixed) | `model/retrieval/predictor.py::RetrievalPredictor.predict_particles`/`.predict_occ`; `experiments/EXP-0059-*/code/eval_retrieval.py::evaluate`/`_predict_occ_chunked` |
| per-query confidence FOR FREE alongside the clean prediction (no 2nd bank search): `top1_dist` (nearest neighbour's distance) + `knn_disagreement` (std across k neighbours' matched displacement, k>1 only) — for finding actions the bank predicts least well, not for scoring | `model/retrieval/predictor.py::RetrievalPredictor.predict_particles_with_confidence` |
| leakage-safe DS-0008 train/validation split for hyperparameter tuning, grouped by CHAIN (`(file_index, chain_env)`, never row-random) so tuning never touches DS-0009; DS-0008 has no same-state action pools, so only `accuracy_1`/`rollout_accuracy_4` validate this way — `slateN` is a DS-0009 TEST-only number for the shortlisted best configs | `model/retrieval/val_split.py::chain_holdout_split`/`build_tuning_bank_and_val_chains` |
| build/persist the bank; scorer that runs a `predict_particles`/`predict_occ` predictor through (as closely as the input-kind difference allows) `eval_narrow.py`'s own `acc`/`swept_region`/`GOALS`/slateN code, rasterising with the same `occ_from_particles` every OCC_ADAPTERS model is scored with; a small named-config sweep (window/cap/corridor/k/aggregation) tuned on the validation split, re-scored on DS-0009 for the shortlist only | `experiments/EXP-0059-retrieval-transition-model/code/build_bank.py`, `eval_retrieval.py --predictors persistence retrieval_1nn`, `sweep_retrieval.py --phase val\|test` |
| **visual debug figure** (2026-09-28, coordinator request): one 6-panel PNG/PDF per DS-0009 (state, push) query -- global-frame query+push, then 5 panels in the query's OWN push frame (identical axes): query vs 1-NN donor + Hungarian correspondence, query vs k=5 neighbours (translucent, ranked), 1-NN transfer vs truth, each neighbour's own transferred prediction, and the COMBINED (`k5_cube_median`) prediction vs truth with per-cube mm error + `top1_dist`/`knn_disagreement`. Reuses `predictor._search`/`._transfer_one`/`.predict_particles_with_confidence` verbatim (no transform/search/transfer math reimplemented) so the figure is faithful to what `eval_retrieval.py` scores. Checked (numeric + visual, 2026-09-28): `frame.py`'s push-frame round-trip and pairwise-cube-distance rigidity hold to float32 precision on real DS-0009 rows; no axis mirror/transpose or yaw-sign bug found. **Confirmed by eye, not a bug**: `predictor.py::_transfer_one`'s Hungarian match is over ALL 20 query/donor cubes unconditionally (no window/corridor mask), so a query cube far outside the corridor can occasionally draw a long-range, physically-implausible donor match (a real, visible failure mode of the design, not the code) — `cube_median` (k=5) visibly damps single-neighbour cases like this relative to raw 1-NN, consistent with the R1 sweep result above | `experiments/EXP-0059-retrieval-transition-model/code/retrieval_debug.py::plot_retrieval_debug` + CLI; figures in `experiments/EXP-0059-*/figures/retrieval_debug/` (indexed by that dir's `README.md`); shared rotated-cube/arrow/blade-drawing primitives factored into `scripts/probes/cube_viz.py` (also now used by `transition_panel.py` and `EXP-0051-*/code/demo_gifs.py`, replacing 2 near-duplicate inline copies) |

**SUPERSEDED 2026-09-28 (ISS-010 + the `_transfer_one` fix below — see `experiments/EXP-0059-*/LOG.md`'s post-fix rerun): every number in the next two paragraphs was measured through a predictor with a real transfer bug (unconditional full Hungarian match, no distance gate — see "particle-in/out reference predictors" row's fix note above) and, for the v0/R1 numbers specifically, an unaudited bank (~44% touchdown-illegal rows, ISS-010). Fixed + curated: `retrieval_1nn` DS-0009 `accuracy_1 = 0.337-0.452`/`slateN = 0.759-0.827` (all-rows/legal-only) and `k5_cube_median` `accuracy_1 = 0.330-0.493`/`slateN = 0.754-0.823`, both now in the narrow NFD's own range (`accuracy_1 0.506-0.638`/`slateN 0.750-0.774` on the SAME rows) — kept below for the historical record, not because they are still believed.**

**Known result (EXP-0059 v0, naive uncapped/unweighted 1-NN reference only):** on the DS-0009 narrow-domain test set, `retrieval_1nn` (`DistanceConfig()` defaults) scores `accuracy_1 = 0.172` (scatter -0.003, clump 0.219) and `slateN = 0.413`, well below the narrow NFD's `accuracy_1 ~0.51`/`slateN ~0.77-0.80` (EXP-0053) — expected, since it is the simplest possible member of the design doc's experimental matrix (k=1, no interaction-support hierarchy, no kernel/local-linear correction). `persistence` scores `accuracy_1 = 0.000` exactly and `rollout_accuracy_{1..4} = 0.000` exactly, validating the particle-in scoring plumbing against EXP-0053's own "persistence is 0 by construction" invariant.

**Known result (EXP-0059 R1, an 11-config sweep tuned on a DS-0008-chain validation split, leakage rule enforced, never touching DS-0009 during tuning -- `experiments/EXP-0059-*/code/sweep_retrieval.py`):** capping+corridor weighting alone lift validation `accuracy_1` only modestly (0.183 -> ~0.20); k=5 aggregations are where the real movement is, and the two aggregation FAMILIES trade off exactly as the "hedging is scoring-only" design predicts -- **hedged occupancy (`occ_mean`/`occ_weighted_mean`) wins accuracy** (DS-0009 test `accuracy_1 = 0.393`, more than double v0, still below narrow NFD) **but a CLEAN single state (`cube_median`) wins ranking** (DS-0009 test `slateN = 0.513` vs the hedged pair's `0.451`, and the best `rollout_accuracy_4 = 0.075`) -- consistent with `accuracy`'s known bias toward hedged/blurred predictions (see this file's "TRAP -- `accuracy` is a ratio of population means" and the subagent-experimenter skill's blur-accuracy trap). **Negative result, reported plainly:** `cube_weighted_mean` (distance-softmax over k=5) was the WORST config tried, val `accuracy_1 = 0.080` and NEGATIVE `rollout_accuracy_4 = -0.040`, below even the naive v0 baseline -- the default `weight_temperature=0.01` is likely miscalibrated against this distance's actual scale (cm), not yet diagnosed further. Transferring yaw made no measurable difference (`capped_corridor` vs `capped_corridor_no_yaw`, identical to 3 decimals) -- expected, since every scorer here is orientation-agnostic (rasterises cube CENTRES only). None of these tuned configs yet reach the narrow NFD on any headline metric. Full tables: `experiments/EXP-0059-*/results/sweep_val.json` (validation, all 11 configs) / `offline_eval_retrieval_bestconfigs.json` (DS-0009 test, top-3 by validation `accuracy_1` only).

**Diagnostics decomposing the gap (`ceiling` labels, never headline numbers -- `experiments/EXP-0059-*/code/diagnostics_r2.py`, results `diagnostics_r2.json`):** an oracle search over the WHOLE bank (not just the metric's own top candidates) finds a donor giving moved-cube mm-error ~1.18mm (N=36 DS-0009 rows) -- a real donor generally EXISTS; restricting the oracle to the metric's own top-50/top-500 ranks gives 1.80mm/1.39mm, so the RANKING METRIC leaves real, but not catastrophic, headroom on the table. Transfer-rule comparison under the SAME oracle donor: `displacement` 1.18mm vs `paste` (copy the donor's absolute post-push position) 4.74mm -- much worse, ignoring the query's own pre-push offset -- vs `transport + 5mm non-overlap projection` (`model/retrieval/overlap.py::resolve_overlaps`, a small reusable Jacobi-style relaxation) 1.17mm, near-identical to plain displacement. Same-state leave-one-out ceiling (retrieve only from the OTHER 63 pushes of an identical DS-0009 pool start state, action-nearest match): `accuracy_1 = 0.366`, mm-error 2.89mm (n=2048) -- even with a PERFECT state match, this simple per-cube-independent transfer rule alone caps out well below the narrow NFD, an intrinsic limit of the rule, not the retrieval index. 9.6% of DS-0009 chain rows have zero truly-moved cubes inside the query window (close to the designer's own 9-12% probe figure); the frozen `k5_cube_median` predictor correctly predicts "nothing moves" there 80.6% of the time.

### NFD with a retrieved reference — `model/retrieval_nfd/` (EXP-0059 section 8)

An image-space model, NOT the particle-in/out `model/retrieval/` package above: the narrow
NFD's UNet (features [4,8,16], residual head) with 2 extra input channels carrying a
retrieved donor transition, rendered into the QUERY's own world frame. 5 input channels
`[occ0, r_start, r_stop, donor_occ0, donor_occ1]` (the optional 6th "retrieval clean
prediction" channel from the design doc is NOT implemented -- out of the coder's time
budget). Donor mapping: the top-1/top-3 bank row's push-frame `(uv0, uv1, yaw0, yaw1)`
(`model/retrieval/bank.py::TransitionBank`) is treated AS the query's own push-frame
representation (`model.retrieval.frame.push_frame_to_world` with the QUERY's `p_start`/
`p_stop`), then rasterised.

| need | where |
|---|---|
| channels 0-2/target: the REAL `Baselines.NFD.nfd_lib.PileSweepData3Ch` (bit-exact with the narrow NFD's own training data, not an approximation); donor channels 3/4: `model/retrieval_nfd/render.py::render_cube_boxes_batch`, reimplementing `_draw_particle_grid`'s cv2-box path as a free function (same rasteriser at train AND eval time) -- an earlier vectorised soft-cube approximation was tried and reverted (2026-09-28, coordinator feedback) once the donor SEARCH was fixed to be fast enough (see below) that the exact cv2 rasteriser fit the time budget after all (~37s to render all ~12k rows' donor channels) | `model/retrieval_nfd/render.py`, `donors.py::build_query_rows_from_pilesweepdata` |
| donor top-k search: fully vectorised chain exclusion (chain keys -> integer ids, then a boolean mask + stable `argsort` per query chunk) -- the ORIGINAL version had a per-row Python loop calling `.tolist()` (a CUDA device-sync) ~12k times, which was the actual ~70-minute bottleneck, not the chamfer search itself; fixed, the full 11921x11921 search takes ~5.4 minutes | `model/retrieval_nfd/donors.py::topk_donors_excluding_chain` |
| chain-aware donor bank: reloads DS-0008 (valid rows; all 11921 rows here are valid, verified) + DS-0010 with a PARALLEL per-row chain-key array (`ds0008::{file}::{chain_env}` / `ds0010::{source_file}`) that `model/retrieval/bank.py::TransitionBank` itself does not keep, so training-time donor search can exclude the query's own chain; asserts its bank matches `TransitionBank.build()` row-for-row | `model/retrieval_nfd/donors.py::load_all_rows`/`build_bank_with_keys` |
| the frozen retrieval key (EXP-0059 R1 sweep-chosen: capped+corridor-weighted chamfer, cap=0.02, corridor_weight=3.0 -- see `offline_eval_retrieval_bestconfigs.json`), chain-excluding top-k donor search (train: top-3, random-of-3 picked per `__getitem__` call), and a corridor-cube-count-bucketed random donor for the control model | `model/retrieval_nfd/donors.py::frozen_distance_config`/`topk_donors_excluding_chain`/`random_donor_excluding_chain` |
| ONE-TIME precompute: renders every row's occ0/r_start/r_stop/target + top-3 donor pairs + 1 random-donor pair into a flat uint8/fp16 cache (`model/retrieval_nfd/cache/retrieval_ref_cache.pt`), carrying the narrow-NFD split tag (`Genesis.training.dataset.PileSweepData._filter_split`, val_pct=5/test_pct=5, SAME two directories `nfd_3ch_narrow_l20.yaml` trains on) per row. Runs the MANDATORY unit tests inline: rendering a query's own cubes through the donor-render path reproduces occ0 (max abs err asserted < 1e-5), and the push-frame v-mirror round-trips against an independently-derived world-space reflection | `model/retrieval_nfd/precompute.py` |
| flat cache-backed `Dataset` (`((input, physics), target)` contract, plugs into `registry.dataset_registry.EulerianDatasetWrapper` exactly like `Baselines.NFD.nfd_lib.PileSweepData3Ch` does) + its dataset-registry entry, reusing `Baselines.NFD.nfd_lib`'s EXISTING generic `nfd-unet3ch` model factory unchanged (`in_channels: 5` in config) so `training.trainer.Trainer` (AMP, augmentation -- channel-agnostic rot90/flip, so donor channels 3/4 get the SAME spatial augmentation as everything else for free, patience/checkpointing) needs no new code | `model/retrieval_nfd/dataset.py::RetrievalRefFlatDataset`, `lib.py`, `train_retrieval_nfd.py`, `configs/retrieval_nfd_ref.yaml` (main) / `configs/retrieval_nfd_random_donor.yaml` (control i, `donor_field: random`) |
| eval-time predictor matching `Baselines/NFD/predictor.py::NFDPredictor`'s `predict_occ(batch)` contract (so it plugs into `simple_mpc.adapters.OCC_ADAPTERS`/`eval_extended.py --models` like every other NFD variant) AND a particle-aware `predict_step_particles(occ0, act, states0)` entry point that `eval_extended.py::eval_occ_model` now dispatches to (a small, targeted patch, 3 call sites) whenever the harness has the TRUE particle state (1-step scoring, slateN candidates) -- retrieval runs on those real cubes, not pseudo-cubes; pseudo-cube extraction from `occ0` via connected-components centroids is now ONLY a fallback for rollout steps after the first (no true particles exist there, matching the design doc's own rollout spec). `zero_ref=True` is control (iii), the test-time zeroed reference, on the SAME checkpoint | `model/retrieval_nfd/predictor.py::RetrievalRefPredictor`; registered as `retrieval_nfd_ref`/`retrieval_nfd_ref_zeroed`/`retrieval_nfd_random_donor` in `simple_mpc/adapters.py::OCC_ADAPTERS` (once each checkpoint exists) |

**Known result (2026-09-28, full DS-0008+DS-0010 corpus, 20 epochs):** `retrieval_nfd_ref` beats
`retrieval_nfd_random_donor` on every metric (DS-0009 `slateN_tough` 0.686 vs 0.597) -- the
retrieved reference IS informative, not noise (G6). But `retrieval_nfd_ref_zeroed` (same
checkpoint, reference zeroed at test time) beats `retrieval_nfd_ref` WITH retrieval on (0.715 vs
0.686) -- the network learned to use good donors, but the single top-1-by-frozen-key donor
retrieved at eval time is not, on net, good enough to beat ignoring it, consistent with pure
retrieval's own known ranking-metric headroom (R1.5: oracle-top-50 `slateN ~0.53` vs pure
retrieval's `~0.51`, both well below the narrow NFD's `~0.77-0.80`). Full numbers/diagnosis:
`experiments/EXP-0059-retrieval-transition-model/results/coder_status.md`.

Results (once trained/evaluated): see `experiments/EXP-0059-retrieval-transition-model/results/coder_status.md`.

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
| `Genesis/data/narrow_l20_n20/{train,test_pools,test_chains,extra_18_22}` | DS-0008 (train chains) / DS-0009 (clean test pools+chains) / DS-0010 (extra matched-physics 18-22mm rows) — n20 single-layer, exact 20mm perpendicular pushes, training physics. Keys `states, states_, p_starts, p_stops, angles` (7-dim per-cube state: `[x,y,z,qw,qx,qy,qz]`, quaternion in `(w,x,y,z)` order) plus `valid`/`chain_env`/`chain_step`/`start_kind` (train/test chains) or `pool_idx`/`valid` (test pools); DS-0010 carries `source_file`/`source_rows` instead of `valid` (already filtered by its own `extract.py`). This is EXP-0053's register corpus (best NFD `accuracy_1 ~0.51`, `slateN ~0.77-0.80`) and EXP-0059's retrieval bank source. Produced by `Genesis/chain_collection.py` (`--mode chains`/`pools`) — see the data-collection skill. DS-A (val pools) and DS-B (multi-step test pools, `--mode seqpools`) are its EXP-0059 siblings, same recipe, new seeds. **ISS-010 (found 2026-09-28, sampler CLOSED same day): ~44-56% of these rows (DS-0010: ~0%) have the tool overlapping a cube at touchdown** — see `experiments/EXP-0059-*/code/audit_tool_placement.py`, per-row `_legality.pt` flags next to every source file. These payloads are UNCHANGED; DS-0008/9/11 have clean `_v2` replacements (see the row above, DS-0015/16/17) collected with the fixed sampler — use those for new work. |
| `datasets/DS-0014-retrieval-curated-interaction-sets/data/curated_bank.pt` | DS-0014 — the retrieval bank's ONLY supported source going forward (`TransitionBank.load_curated`): DS-0008+DS-0010 (`TransitionBank.from_states` schema) plus a geometric interaction-set mask (`in_set`, `model.retrieval.interaction`) and the ISS-010 touchdown-legality flag (`legal`) per row, plus provenance (`source_file`/`source_row`). Illegal rows are KEPT and flagged, excluded from the bank by default. Built by `datasets/DS-0014-*/build_curated_bank.py`. |
| `Genesis/data/narrow_l20_n20/{train_v2,test_chains_v2,test_pools_v2,val_pools_v2}` | DS-0015 (train, replaces DS-0008+DS-0010) / DS-0016 (test chains+pools, replaces DS-0009) / DS-0017 (val pools, replaces DS-0011) — **ISS-010 CLOSED (2026-09-28)**: collected with the fixed sampler (`Genesis/action_sampling.py::pile_aware_action_batch`, exact-SAT redraw + `start_gap_range`), verified 0% illegal / 0% `gap_out_of_window` at full scale (see each DS's own DATASET.md and `experiments/EXP-0059-*/results/clean_data_status.md`). Same schema as their predecessors plus a `gap_out_of_window` column (last-resort-accept flag, all-False here). DS-0008/9/11's own payloads are UNCHANGED and still carry the defect; use these `_v2` sets for any new work in this domain. DS-0012/0013 were NOT recollected (out of scope) and still carry the original defect. |

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
| **synthetic pile-state generator (CURRENT — this is what DS-0003 is built from)**: a state is a mix of scattered singletons and compact clumps. `clump_frac` sets CONTACTS (local density), `region_frac` sets SPREAD (global dispersion); the two are near-independent, which is what makes them calibratable separately | `Baselines/common/pile_synth.py::synthesize_state` — places its own units inside a sampled workspace region, so callers do NOT add a placement step |
| the compact-clump COMPONENT that generator calls: grid seed + per-object relaxation toward the centroid; `compaction` in [0,1] is the fraction of available travel closed, `yaw_kappa` von-Mises-concentrates yaws | `Baselines/common/pile_compaction.py::sample_compacted_state` — returns a CENTRED pile. Rigid row/column translation does NOT compact (one pair per row always blocks: measured spread 0.0289 → 0.0283 at n=100, i.e. nothing); per-object relaxation does |
| whole-state variant: 1..k independently-compacted clusters placed in the workspace | `pile_compaction.py::sample_multicluster_state` — **written but NOT used by the DS-0003 driver**; `synthesize_state`'s scatter+clump mixture superseded it. Kept because it reaches the dispersed extreme differently |
| synthetic pile-state generator (RETIRED for DS-0003, function still importable): scattered (grid+jitter) mixed with sequential touching clumps, legality via the old axis-aligned bound | `Baselines/common/goal_configs.py::sample_synthetic_state` (+ `_place_clump`, `_place_scattered_point`, `_penetrates`, `_sample_placement_region`) — that bound rejects 91-99% of real DS-0002 states and cannot express contact, so nothing could ever be compacted against it; see `cube_overlap.py`'s module docstring |
| synthetic states corpus + regeneration command | `datasets/DS-0003-synthetic-states/` (driver: `datasets/DS-0003-synthetic-states/generate.py`) |
| real-vs-synthetic occupancy validation (fast rasteriser + `dmdc_baseline.occupancy_descriptors`) | `datasets/DS-0003-synthetic-states/validate.py` |

**Calibration targets, and why these four statistics.** `synthesize_state`'s
defaults are fitted to real **POST-SWEEP** states (DS-0002 `role=post_sweep`),
not to initial states — post-sweep is what a dynamics or value model actually
meets. Targets, per `n_objects`, as spread p5/p50/p95 (RMS distance of cubes
from their own centre of mass) and contacts p50 (mean cubes within 0.00813 m):

| n | spread p5/p50/p95 | contacts p50 | \|COM\| p50/p95 |
|---|---|---|---|
| 20 | .0136 / .0453 / .0554 | 0.90 | .0135 / .0412 |
| 50 | .0196 / .0481 / .0557 | 1.64 | .0108 / .0387 |
| 100 | .0371 / .0510 / .0591 | 2.64 | .0089 / .0281 |

Real post-sweep piles are mostly DISPERSED with a compact tail — a generator
tuned to make tidy compact piles is wrong by default. Three knobs needed
MIXTURES rather than single distributions, each because a unimodal choice was
measured to fail: region extent (a compact branch under a dispersed bulk — no
power curve gives a long compact tail beneath a wide median), placement
(centred AND edge-hugging — insetting the region pins wide states to the middle
so material can never pile against a wall, but letting the boundary clip every
state compresses spread instead), and clump share (scales with `n_objects` —
real dense piles are many clumps spread out, not one blob).

**Do not re-derive the packing floor.** Real n=100 states reach spread 0.0209,
essentially the aligned-lattice limit at `CUBE_SIZE` pitch (0.0204). A
single-layer yaw-only generator cannot go below ~0.024: real cubes TILT (mean
5-13°, measured) and ~2% sit in a second layer, so real 2D compactness is
partly a 3D effect.

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
| **one (state, push) transition through one model**, 2x3: input raster, the model's own action channels (captured from the UNet input by a forward hook; switched-linear models, which have no action input raster: the same raster captured from the reference NFD for that action), true next state, RMS/accuracy/changed-cell IoU (whole + swept region), prediction, signed error map | `scripts/probes/transition_panel.py --model <OCC_ADAPTERS id> --source narrow\|<binned corpus dir> --index i [j ...]` |
| **push-by-push GIF of a recorded closed-loop episode** (goal mask + cubes + blade/arrow; side-by-side panels) | `experiments/EXP-0051-*/code/demo_gifs.py::draw_frame`, `write_gif`; EXP-0055/0056 cells: `experiments/EXP-0055-*/code/demo_gifs.py --cells <c> [--vs lyap --top 3]`. **DISPLAY ORIENTATION: world x right, world y DOWN** -- `letter_mask`'s glyph stands in the world with its top at low y, so a y-up plot shows every letter upside down (the first EXP-0051 GIFs did) |
| which slates to plot (typical / worst / near-tie per model) | `scripts/probes/pool_survey.py` |
| cache loading, `rk_curve`, `action_geometry`, `swept_rectangle_corners`, `MODEL_COLORS` | `scripts/probes/pool_common.py` (reads an embedded `occ0`/`ws_min`/`ws_max` when the cache carries them) |
| a dV cache for `Genesis/data/slates_binned/*` (persistence, random, MODEL-0001/2/3) | `scripts/probes/binned_pool_cache.py` |
| score a model as a SOURCE OF GRADIENTS (optimise an action against it, then execute in Genesis) rather than as a ranker | `experiments/EXP-0023-model-as-gradient-source/code/` — `verify_gradients.py` (the differentiability gate; run it before any optimisation), `stage1_optimise.py` (model side: rank pick + Adam on the model's own predicted `dv`, projected constraints), `stage2_genesis.py` (ground truth + Genesis-CEM oracle from a restored slate snapshot), `stage3_analyse.py` (`gradient_gain`, `pool_escape`, `capture_vs_oracle`, and the between-arm-vs-between-state power check) |
| **retrieval-model visual debug** (6-panel: global frame + 5 push-frame panels, k-NN neighbours, transfer, combined prediction vs truth) — see `model/retrieval/` section above for the full description | `experiments/EXP-0059-retrieval-transition-model/code/retrieval_debug.py` |
| shared rotated-cube-footprint / push-arrow / blade-line matplotlib primitives (metre- or mm-space; factored out of `transition_panel.py`, `EXP-0051-*/demo_gifs.py`, and `retrieval_debug.py`, which had each re-derived them) | `scripts/probes/cube_viz.py::cube_patch`/`push_arrow`/`blade_endpoints` |

**Known trap:** `persistence` predicts `dv=0` for every candidate, so as a *ranker* it is
degenerate — `argmin` always returns row 0 and the induced order is row order. Use `random`
as the ranking floor. Separately, MODEL-0002's point-mass readout saturates on
`slates_binned`: 26 % of predictions are exactly 0 (predicted COM inside the target region,
where the distance field is 0), so ~26 % of a slate ties at the minimum.

## Oracle-as-model MPC budget/objective ablation — EXP-0057

`experiments/EXP-0050-oracle-ceiling/code/oracle.py` (shared with EXP-0050) now also takes
`--value {lyap,crowd_floor}` (crowd_floor = lyapunov + 0.1 x `value_functions.CrowdingPenalty`,
floor_single=True), `--n-envs` (was hardcoded `K=32`), `--l-min`/`--l-max` (push-length-range
ablation), `--out-dir`, and records `sims_used`/`plan_time_s`/`signed_frac`/`obj_values`/`cell`
per episode per push. Its `sample()`/`simulate()` already flatten every episode's candidates
into one list and chunk by `n_envs` before this change — the efficient any-episode-count /
any-split batching pattern was already there, just needed `K` exposed as a CLI flag.
`experiments/EXP-0057-oracle-budget-ablation/code/run_queue.py` is a resumable multi-cell job
queue over `scripts/run_probe.py` (skip-if-complete by reading each cell's checkpoint JSON,
copies `runs/COMMANDS.jsonl` into `experiments/COMMANDS.jsonl` by a LINE-COUNT offset, not a
timestamp — two jobs finishing in the same second must not duplicate/drop a ledger line);
`code/analyse.py` computes the completion-push metric `oracle_completion_pushes`
(METRICS.md) with paired bootstrap deltas.

## The evidence layer

`experiments/REGISTER.md` (claims) · `INVARIANTS.md` (tags) · `METRICS.md` (metric formulas) ·
**`OPEN_ISSUES.md` (code defects and evidence debt awaiting an owner)** ·
`COMMANDS.jsonl` · `TEMP_LOG.md` · `scripts/check_register.py` (must exit 0) ·
`scripts/run_probe.py` (**`broken`: writes to `runs/COMMANDS.jsonl`, not the documented
`experiments/COMMANDS.jsonl`**).

**Found a real problem you cannot fix in this task? Append an entry to
`experiments/OPEN_ISSUES.md`** — a defect too big to fix inline, a test that
must be re-run because an input changed, or a record asserting something since
disproved. It is the one place such findings survive a run report nobody
re-reads. A claim goes in `REGISTER.md`, a property results rest on goes in
`INVARIANTS.md`, a scratch probe goes in `TEMP_LOG.md`.

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
