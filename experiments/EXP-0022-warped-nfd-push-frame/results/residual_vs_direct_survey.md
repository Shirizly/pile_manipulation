# Survey: residual vs. direct-state prediction, all dynamics models in the repo

NOTE (post-run refactor, pure code move, no behaviour change): the
`Baselines/NFD/warped_nfd_lib.py`, `Baselines/NFD/residual_nfd_lib.py` and
`Baselines/NFD/predictor.py::build_canonical_stack`/`WarpedNFDPredictor`
file:line citations below moved to `model/warped_nfd/lib.py` / `model/
residual_nfd/lib.py` / `model/warped_nfd/predictor.py` after this survey was
written -- the file:line pairs no longer resolve as written; the analysis
and verdicts are unaffected.

Read-only survey. For every model, the decisive test applied is: **what tensor does the
loss compare against what target** (not what the config/docstring calls the model).

## Table

| Model | Where (file:line) | Direct / Residual / Other | Evidence |
|---|---|---|---|
| `unetfilm` / `NFDUNetFiLM` (`model/NFDUNetFilm.py`) | `model/NFDUNetFilm.py:256-257` returns `self.head(d_features) + current_state`; loss `training/losses.py:161,168` compares `sigmoid(logits)` to `batch["target"]` (absolute occ1) | **Direct**, with an architectural residual/skip add of `occ0` into the pre-sigmoid logit | `NFDUNetFiLM.forward` adds `current_state` (raw 0/1 occupancy, not a logit) to the conv head output before that sum is treated as a logit; `EulerianCombinedLoss.__call__` (`training/losses.py:157-168`) always compares `sigmoid(logits)` against `targets = batch["target_occupancy"] or batch["target"]`, which is the absolute post-push occupancy (`Genesis/training/dataset.py:576-580`, `registry/dataset_registry.py:121-122`). Target is never `occ1-occ0`. |
| `unet-modular` / `UNetModels_modular.UNet` with `residual: true` (used verbatim by `Baselines/NFD/predictor.py`'s `nfd-unet3ch` and `nfd-unet-warped` via `_STRUCTURE_3CH`) | `model/UNetModels_modular.py:371-406`, esp. `394-399` | **Direct**, with the same input-occ0-skip pattern | `residual_mode` (`registry/model_registry.py:319` default `True`) only controls whether `base = x[:,0:1,:,:]` (occ0) is added to the raw conv output (`UNetModels_modular.py:396-399`) before that sum is passed on as a logit — **this is the config-collision trap named in the brief**: `residual: true` here means "add occ0 into the logit," not "predict occ1-occ0." Same `EulerianCombinedLoss` target as above. |
| Warped NFD (`nfd-unet-warped`, `Baselines/NFD/warped_nfd_lib.py::WarpedNFDWrapper`) | `Baselines/NFD/warped_nfd_lib.py:145-170` | **Direct** | `forward()` computes an absolute canonical-frame probability via `push_frame_roundtrip(fn, occ0, ...)` (blends toward occ0 only in the corners the warp cannot cover — validity mask, not a training target), converts back to a logit (`:167-169`), and that logit is scored by the same `EulerianCombinedLoss` against absolute `target`. Confirmed also by `experiments/EXP-0022-*/results/pilot_training_curves.md`: "world-frame MSE against the same target representation." |
| GNN / `PropNetDiffDenModel` (`model/gnn_dyn.py`) | `model/gnn_dyn.py:196-198` (`return particle_pred + s_cur`), `252-254`; loss `training/losses.py:228-248` (`LagrangianMSELoss`) | **Direct**, with an internal residual/skip add | `PropModuleDiffDen.forward` returns `particle_pred + s_cur` — the network head genuinely predicts a delta that is added to current particle position. But `LagrangianMSELoss` compares the returned `s_pred` (absolute position) against `batch["target_particles"]` (absolute next position), never against a delta. **`s_delta` in the batch (`training/types.py:48,62`) is an INPUT — "impulses of the objects" (`model/gnn_dyn.py:150,212`), consumed as `s_cur + s_delta` when building the receiver/sender graph (`model/gnn_dyn.py:224-225`) — not the model's output target.** Confirms trap #3: `s_delta` is input, not the residual target. |
| SchenckCNN (`Baselines/SchenckCNN/model.py`) | `Baselines/SchenckCNN/model.py:47-50`; training `Baselines/SchenckCNN/train_schenck.py:74,83-88` | **Direct**, with the same input-occ0-skip pattern | `forward` returns `(occ0 + delta).squeeze(1)` — architecturally residual — but `train_schenck.py` calls `loss_fn(pred, yb)` (plain `MSELoss`) where `yb` comes from the same dataset whose target is absolute occ1, not a delta. No sigmoid (paper convention: plain L2 on a continuous field), but still absolute-vs-absolute. |
| NCA (`model/NCAModels.py`, registered `"nca"`) | `model/NCAModels.py:184-198`, esp. `196` (`next_state = state_after + delta_corr`) | **Direct**, with the strongest architectural residual bias in the repo | Own docstring says it outright (`model/NCAModels.py:221-224`): "the NCA starts from the input state and accumulates delta updates, so the output is anchored near the input occupancy at initialisation." Still wrapped by `EulerianTrainingWrapper` and scored by `EulerianCombinedLoss` against absolute `target` — never trained on a delta target. |
| Spatial-transformer (`model/SpatTransNet.py`, registered `"spatial-transformer"`) | `model/SpatTransNet.py:109-160`, esp. `146-152` (`grid_sample` on `current_density`) | **Direct**, but mechanically an advection/warp of the input (cannot invent mass) | Output is a `grid_sample` warp of `current_density` itself (not of any learned free-standing field), converted to a logit (`:157-159`) and scored against absolute `target` by the same loss. Structurally incapable of producing occupancy the input didn't already have somewhere, which is a different kind of "anchored to current state" than a literal residual add. |
| Heuristic push models — `SplatPushModel`, `SpreadPushModel`, `SplatPushModel2`, `CumulativePushModel`, `FluidPushModel` (`model/eulerian_wrapper.py:593-1024`) | `model/eulerian_wrapper.py:644-679` etc. | **Other** — no learned weights, no loss/target at all | These have zero trainable parameters (`register_push_model`, `model/eulerian_wrapper.py:1024-1145`) and are never fit against a target; they redistribute mass geometrically (splat/spread/blur along the push direction) and return the resulting full occupancy grid directly. Not comparable to "direct vs. residual" in the trained-model sense — there is no loss line to cite. |
| `UNetFiLMPushModel` (MPC-facing wrapper around a trained `NFDUNetFiLM`, `model/eulerian_wrapper.py:1145-1400`) | `model/eulerian_wrapper.py:1375-1400` | **Direct** (deployment wrapper; training already covered above) | Comment at `:1375-1381` states explicitly: "model was trained with MSE(sigmoid(logit), target)" — i.e. absolute target, confirming the training-time claim from the vantage of the eval/MPC path. |
| Linear visual foresight, `fit_operator`/`fit_operator_nonneg` (`fit_linear_foresight.py:123-208`) | `fit_linear_foresight.py:123-152`, `metrics` at `:313-…` | **Direct, but regularised TOWARD IDENTITY** (own category per the brief, not forced into yes/no) | `A = argmin ||Y1 - A Y0||_F + ridge||A - I||_F` when `toward_identity=True` (default) — `A` still maps `occ0 -> occ1` directly (`fit_linear_foresight.py:124`); only the ridge SHRINKAGE TARGET is `I`, not the output being fit. `metrics()` (`fit_linear_foresight.py:313-335`) computes `d = (pred - truth) * region`, i.e. absolute prediction vs. absolute truth. MODEL-0001 (`weights/MODEL-0001-stage2-visual-switched/MODEL.md`) is this fit, persisted. |
| Descriptor DMDc operators, `fit_per_action_operators`/`apply_operators` (`dmdc_baseline.py:163-219`) | `dmdc_baseline.py:163-214`, `one_step_report` at `:227-246` | **Direct, but regularised TOWARD ZERO (or toward a supplied prior, e.g. identity)** | `A_b = argmin ||A*phi_t - phi_t1||^2 + lam||A - A0_b||^2`; when `prior_A is None` the ridge target is `0` (plain ridge, "the original behaviour" per the docstring, `dmdc_baseline.py:179`); when a prior is supplied it can be `I` (transfer-learning case). Either way `A` maps `phi_t -> phi_t1` directly: `one_step_report` (`dmdc_baseline.py:243-244`) computes `mse = (pred[:, sl] - phi_t1[:, sl]).pow(2).mean()` — absolute-vs-absolute. This is the "contrast with `dmdc_baseline`, which regularises toward zero" case named in the brief; MODEL-0002 is this fit (`lam=1e-4`, `weights/MODEL-0002-descriptor-only-D-all-local/MODEL.md`). |
| Switched-by-length wrapper (`Baselines/LinearForesight/model.py::predict_switched`) | not separately re-derived; it dispatches to the same `fit_operator`-style bin operators above | **Direct** (inherits its bin operators' category) | No independent loss/target of its own — it is a bin-selection layer over MODEL-0001-style operators. |
| MODEL-0003 (`weights/MODEL-0003-nfd-multistep-finetuned/`) | `weights/MODEL-0003-nfd-multistep-finetuned/MODEL.md` | **Direct** | Fine-tuned NFD checkpoint (same `nfd-unet3ch` architecture as row 2/3 above) through a 3-step closed-loop rollout objective `L = lam*L1 + lam^2*L2 + lam^3*L3`, explicitly "full-image, unmasked per-step MSE" against the absolute image at each of the 3 rollout steps (per its own MODEL.md) — never a step-wise delta. |
| LeJEPA latent switched-linear dynamics (`experiments/EXP-0016-lejepa-random-encoder-floor/code/model.py::SwitchedLinearDynamics`) | `experiments/EXP-0016-lejepa-random-encoder-floor/code/model.py:106-145`; fit loop `experiments/EXP-0016-lejepa-random-encoder-floor/code/fit_dynamics.py:65-67` | **Residual — the one confirmed residual-target model in the repo** | `forward()` returns `(z + dz, dz)` (`model.py:144-145`); the fit loop calls `dyn(zi, atr[idx])` and takes `_, dz = ...`, then `loss = ((dz - dztr[idx]) ** 2).mean()` where `dztr = tr["z1"] - tr["z0"]` (`fit_dynamics.py:65-66,109`) — the loss is computed **directly against the true latent delta**, never against absolute `z1`. This is a LATENT-space residual, not an image-space one. |
| EXP-0019/EXP-0020 hard-gate switched-linear latent operators (`fit_switched_hard.py`, `fit_switched_grouped_lam.py`) | `experiments/EXP-0019-lejepa-encoder-pushlen-switched/code/fit_switched_hard.py:51-146` | **Residual** (same latent-delta convention as EXP-0016) | `dztr = (tr["z1"] - tr["z0"])` (`:89-90`); `ridge_fit(feats(...), dztr[...], lam)` fits directly against the delta, ridge toward **zero** (`"Fit is closed-form ridge (toward zero) of dz"`, module docstring line 13); `r2 = 1 - mse(dz_pred - dz_true)/mse(dz_true)` (docstring line 17). Same pattern reused unchanged by EXP-0020's `fit_switched_grouped_lam.py`. |
| Dataset/loss target construction (`Genesis/training/dataset.py`, `training/losses.py`, `training/types.py`) | `Genesis/training/dataset.py:552,576,580`; `registry/dataset_registry.py:121-122,199-200,313`; `training/losses.py:161,232-236` | **Always constructs/consumes an ABSOLUTE target for image-space (Eulerian) and particle-space (Lagrangian) models** | `PileSweepData.__getitem__` returns `(self._input_grid.clone(), ...), self._output_grid.clone()` where `_output_grid` is built by `_draw_particle_grid(particles_, ...)` — the absolute post-push raster, never a difference. `EulerianDatasetWrapper`/`LagrangianDatasetWrapper` pass this straight through as `"target"`/`"target_particles"`. No dataset or loss in `training/` ever constructs `occ1 - occ0` as a training target. |
| `chg_mse` / `changed_*` in eval logs (`compare_model_emd.py:209-333`; e.g. `experiments/EXP-0022-*/runs/RUN-000*/RUN.md`) | `compare_model_emd.py:228,246-248,303` | **Diagnostic metric only, not evidence of residual training** | `changed_mask = (outputs - current_state).abs() > CHANGE_THRESHOLD` selects PIXELS that changed; `changed_sse` still accumulates `((probs - outputs)**2 * changed_mask)` — absolute prediction vs. absolute target, just restricted to a masked subset of pixels. Confirms trap #4: this is a masked-region MSE, not a residual-target loss. |

## Does any image-space occupancy model predict a residual?

**No.** Every image-space (Eulerian/occupancy) model in the repo — NFD (both plain and
warped), the modular UNet family, SchenckCNN, NCA, the spatial-transformer, and the
`UNetFiLMPushModel` MPC wrapper — is trained with `sigmoid(logit)` (or, for SchenckCNN,
a raw linear head) compared against the **absolute** post-push occupancy via
`EulerianCombinedLoss` (or an equivalent plain-MSE loop for SchenckCNN). The dataset
target construction (`Genesis/training/dataset.py`, `registry/dataset_registry.py`)
never builds `occ1 - occ0`. Several of these networks (NFDUNetFiLM, the modular UNet
with `residual: true`, SchenckCNN, NCA) do internally add the current occupancy back
into the pre-activation output — but that is an **architectural skip connection**, not
a change to what the loss is computed against (trap #1, confirmed repo-wide).

The only confirmed residual-TARGET training in the repo is in **latent space**:
EXP-0016's `SwitchedLinearDynamics` and EXP-0019/EXP-0020's hard-gate ridge operators,
both of which fit directly against `dz = z1 - z0`.

Linear pixel/descriptor operators (`fit_linear_foresight.py`, `dmdc_baseline.py`) are
direct maps `A: state_t -> state_t1`; only their ridge SHRINKAGE TARGET (`I` or `0`)
differs, which is a distinct axis from what they predict (trap #2, confirmed).

## (a) Cheapest train+test loop to test image-space residual prediction for NFD

**Cheapest existing config to copy:** `Baselines/NFD/configs/nfd_train_3ch_L20mm_pilot.yaml`
— trains `nfd-unet3ch` (features `[4,8,16]`, ~30k params) on
`slates_multistep/n20_L20mm_train` at `resolution_scale: 0.5`, batch 32, 20 epochs.
Measured cost (`experiments/EXP-0022-warped-nfd-push-frame/results/pilot_training_curves.md`,
RUN-0001-unwarped-control): **~32 s/epoch** (39 s cold start decaying to ~29 s), so **~11
minutes** for a full 20-epoch run; RUN-0002/RUN-0003 (warped variants, same corpus) ran
~37-38 s/epoch (~12-13 min). A 2-3 epoch smoke test (as `train_nfd.py`'s own `--override
training.epochs=2` pattern is used elsewhere) would cost **~1-2 minutes** and is enough to
confirm the loss trains down and the eval harness runs end-to-end before committing to
the full 20 epochs.

**What would have to change to predict `occ1 - occ0` instead of `occ1`:**

1. **Target construction.** `Genesis/training/dataset.py::PileSweepData.__getitem__`
   (`:562-580`) would need a residual variant — either a new dataset subclass (mirroring
   `PileSweepData3Ch`'s pattern in `Baselines/NFD/nfd_lib.py:43-91`) that returns
   `output_grid - input_grid[0]` instead of `output_grid`, or a transform-layer change in
   `registry/dataset_registry.py::EulerianDatasetWrapper.__getitem__` (`:120-122`) that
   subtracts `input[:,0]` from `target` after the fact. The dataset-subclass route is
   cleaner and matches the existing `PileSweepData3Ch`/`PileSweepData3ChWarped`
   subclassing convention this repo already uses.
2. **Loss.** `training/losses.py::EulerianCombinedLoss` (`:103-221`) currently always
   applies `probs = torch.sigmoid(logits)` (`:168`) and compares against `targets` in
   `[0,1]`. A residual target `occ1-occ0` lives in **`[-1,1]`, is signed, and a sigmoid
   head is architecturally wrong for it** — sigmoid can only output `(0,1)`, so it cannot
   represent a negative delta (material leaving a cell) at all. This is a real problem
   with the repo-wide "model emits a logit, loss applies sigmoid" convention (confirmed
   at `training/types.py:79-99`, `Baselines/NFD/predictor.py:24-27`, and the "residual"
   architectural skip-connections surveyed above, all of which still end in a sigmoid).
   Clean options, in order of least invasive:
   - Add a `target_mode: residual` config path to `EulerianCombinedLoss` that skips the
     sigmoid and instead compares the raw logit (or a `tanh(logit)` head, which is bounded
     to `(-1,1)` and signed) directly against `occ1-occ0` with plain MSE. `tanh` is the
     closer drop-in replacement for `sigmoid` since it keeps a bounded, saturating output
     head; plain unbounded linear + MSE (SchenckCNN's convention) is the other clean
     option and is already precedented in this repo.
   - This also breaks the existing `mass`/`add`/`remove`/`dice`/`bce` sub-terms in
     `EulerianCombinedLoss` (`:174-208`), all of which assume `probs in [0,1]` — they would
     need to be disabled (`mse: 1.0` only, everything else `0.0`, which is already the
     pilot config's setting, `nfd_train_3ch_L20mm_pilot.yaml`) or rewritten for a
     signed range.
3. **Model output.** The UNet itself (`model/UNetModels_modular.py:371-406`) needs
   `residual_mode` set so its existing `base + raw` skip becomes irrelevant/removed (it
   would otherwise double up with the new residual target — the network already adds
   `occ0` back in when `residual: true`, which is exactly the wrong behavior if the loss
   target is now `occ1-occ0` rather than `occ1`; `residual_mode` should be set `False` for
   this experiment) or reinterpreted as "predict delta directly, no add-back" by returning
   `raw` unchanged and letting the *loss* (not the model) hold the discussion of
   sigmoid-vs-tanh above.

**Do the existing metrics/eval harness work unchanged?**

- **`accuracy`** (`fit_linear_foresight.py::metrics`, used by
  `Baselines/common/eval_report.py`) compares `pred` against `truth` as **absolute
  occupancy fields** and needs `occ0 + pred_delta` reconstructed before it can be handed
  to `metrics()` at all — it would NOT work unchanged; the eval script would need a
  one-line "add back `occ0`" step before scoring, analogous to how `predict_occ` already
  applies `sigmoid` before returning (`Baselines/NFD/predictor.py:110-111`).
- **`slateN`** (`Baselines/common/goals.py::slate_n_capture`) also expects an absolute
  occupancy/value-function input, so the same "add back `occ0`, then proceed exactly as
  today" adapter is needed — no change to `slateN` itself, but a required predictor-level
  reconstruction step (mirroring `predict_occ`'s existing sigmoid application).
- **`hard_iou`/`hard_dice`** (seen in the pilot RUN.md logs) are already thresholded
  binarizations of an absolute occupancy prediction; they'd need the same reconstruction
  step before they mean anything, or be dropped for this experiment in favor of a
  delta-space MSE/sign-accuracy metric reported alongside.

**Estimated wall-clock for one train+test cycle:** ~15-20 minutes total — ~11-13 min for
the 20-epoch training run (unchanged cost; the residual target/tanh-head change does not
add meaningful compute), plus a few minutes for eval (`Baselines/common/eval_report.py`
is already the harness used for the 3 existing pilot runs, so its own runtime is already
measured/bounded by those RUN.md records) plus the reconstruction-step glue code. A
2-epoch smoke test first (~1-2 min) is recommended given trap #1/#2's config-collision
history in this repo.

## (b) Models that are *effectively* residual without being labelled as such

- **`NFDUNetFiLM`** (`model/NFDUNetFilm.py:256-257`) and **`UNetModels_modular.UNet` with
  `residual: true`** (`model/UNetModels_modular.py:394-399`, the config actually used by
  `Baselines/NFD/predictor.py`'s `nfd-unet3ch`/`nfd-unet-warped`) both add the raw input
  occupancy channel directly into the pre-sigmoid logit. This is a literal instance of
  the brief's own example — "a skip connection from the input occupancy straight to the
  output" — even though the training TARGET stays absolute. It biases every NFD-family
  checkpoint toward reproducing `occ0` unless the head learns a large enough correction
  to overcome it.
- **SchenckCNN** (`Baselines/SchenckCNN/model.py:47-50`) does the identical
  `occ0 + delta` skip, explicitly described in its own docstring as "residual output"
  (matching the paper's Fig. 3 "+"), while still trained against an absolute target.
- **NCA** (`model/NCAModels.py:196`, `next_state = state_after + delta_corr`) is the
  strongest case in the repo: its own docstring (`:221-224`) states the output "is
  anchored near the input occupancy at initialisation" by construction.
- **`EulerianSTN`/spatial-transformer** (`model/SpatTransNet.py:146-152`) is a different
  flavor of the same effect: its output is a `grid_sample` warp of the CURRENT density
  field, so it is mechanically incapable of predicting occupancy that wasn't already
  present somewhere in the input — an even stronger structural anchoring to the input
  than an additive skip, though not literally "residual."
- **`PropNetDiffDenModel`** (`model/gnn_dyn.py:198`, `return particle_pred + s_cur`) is
  the Lagrangian analogue: the network head genuinely predicts a delta added to current
  particle position, though the LOSS still scores the summed absolute position.
- **`fit_operator(..., toward_identity=True)`** (`fit_linear_foresight.py:123-152`,
  MODEL-0001) is identity-biased in its ridge SHRINKAGE, not its output — but the
  practical effect is the same qualitative story as the skip-connection cases above: an
  underdetermined fit degrades toward `A=I` (persistence), not toward erasing the pile.
  This changes what a "does it help to go residual" experiment is actually testing: since
  several of the strongest existing baselines already behave like `occ0 + (small
  correction)` by construction, an explicit residual-target reformulation may show a much
  smaller apparent gain than on a model with no such bias (e.g. `dmdc_baseline`'s
  toward-zero descriptor operators, or a residual_mode=False plain UNet), because those
  models are already partly solving the "start from `occ0`" problem via architecture
  rather than via the loss.

## Contradictions / notes vs. the task brief

- The brief cites `model/gnn_dyn.py` and `model/NFDUNetFilm.py` — both matched as
  described. `docs/ARCHITECTURE.md`'s module map lists `NCAModels.py`/`SpatTransNet.py`
  under `model/futureintegration/`, but they actually live at the top level
  (`model/NCAModels.py`, `model/SpatTransNet.py`) and are imported from there by
  `registry/model_registry.py:264,275` — the doc's path is stale (consistent with the
  project-overview skill's note that `docs/ARCHITECTURE.md` citations aren't always
  live-tree-accurate). `model/futureintegration/` itself contains different files
  (`MultiExitUnet.py`, `UNetModels_conditioned.py`, `Diff_Renderer.py`, `UNetModels.py`) not
  covered by this survey's scope (not named in the task brief and not currently
  registered under `model_registry.py`).
- No contradiction found with any of the five named traps; all five were hit and
  confirmed exactly as described in the brief (residual-connection-vs-residual-target
  collision, toward-identity-vs-residual, `s_delta`-is-input, `chg_mse` is diagnostic-only,
  and `blend_push_prediction`/`push_frame_roundtrip` blending is a validity-mask fallback,
  not a training target — confirmed by `transforms/functional.py` usage in
  `WarpedNFDWrapper.forward`, `Baselines/NFD/predictor.py::WarpedNFDPredictor.predict_occ`,
  and `fit_linear_foresight.py::predict_world`, none of which feed the blended result back
  into a loss).
