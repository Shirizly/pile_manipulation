# Baselines/NFD — LOG

Agent: A2-grid-specs (design), then B2-nfd-impl (implementation/training/
scoring). Branch `baselines/overnight`. Task: non-FiLM NFD baseline, built on
`model/UNetModels_modular.py::UNet` (`unet-modular` registry entry), per the
user's explicit scope correction (originally briefed against "the UNet
files" generically, then pinned to `UNetModels_modular.UNet`, not
`NFDUNetFiLM`).

---

## B2-nfd-impl HANDOFF (implementation phase, read this first)

**Status: implementing.** Files added, all under `Baselines/NFD/` (nothing
outside touched): `nfd_lib.py` (dataset+model registrations), `train_nfd.py`
(thin Trainer driver), `predictor.py` (eval-harness plug-in),
`configs/nfd_train_3ch.yaml` (primary), `configs/nfd_train_2ch_ablation.yaml`
(ablation, second in line per task priority).

**Design decisions made while implementing (not in SPEC.md, discovered by
reading code directly):**

1. **The existing `unet-modular` factory in `registry/model_registry.py`
   never forwards `final_kernel_size`** from its config dict — its
   `structure_parameters` dict literally has no such key, so
   `final_kernel_size: 1` (needed for Fig.7's 1x1 final conv) is
   unreachable through that path no matter what the training config says.
   Fix: `nfd_lib.py` registers a NEW model type `nfd-unet3ch` whose factory
   builds `UNetModels_modular.UNet` directly with the full structure dict
   (features/in_channels/final_kernel_size/residual/bottleneck_type all
   forwarded), wrapped in the existing `EulerianTrainingWrapper`. This is a
   new registration under `Baselines/NFD/`, not an edit to
   `registry/model_registry.py`. Verified: `UNet({"features":[4,8,16],
   "in_channels":3, "final_kernel_size":1, "bottleneck_type":"None",
   "residual":True})` instantiated through this factory gives exactly
   30,541 params (matches SPEC.md's hand-instantiated count) and
   `final_conv` is `Conv2d(4,1,kernel_size=(1,1))`, confirmed by printing
   the module.
2. **3-channel input, without touching `Genesis/training/dataset.py`:**
   `PileSweepData3Ch(PileSweepData)` in `nfd_lib.py` overrides only
   `_create_grids` (allocates 3 channels instead of 2) and `_draw_plate`
   (writes `draw_plate_soft(..., intensity=1.0)` into channels 1 and 2
   separately, no `1-(1-occ1)*(1-occ2)` union). `__getitem__`,
   `_extract_sample_in_pxl`, `_draw_particle_grid` are inherited
   unmodified, so occupancy rasterisation/axis convention/physics
   normalisation stay byte-identical to the register's own numbers.
   Registered as dataset type `nfd-genesis-3ch`.
3. **Checkpoint gotcha for the pooled config:** the pooled train dataset
   config has `val_pct: 0, test_pct: 0` (per `ORCHESTRATION_LOG.md`'s scope
   decision). `training/trainer.py`'s "best" checkpoint tracking uses an
   EMPTY val loader → `val_loss` is identically `0.0` from epoch 1 onward →
   `unet_best.pth` freezes at epoch-1 weights and never updates again
   (`0.0 < 0.0` is False every subsequent epoch). The actually-trained
   weights are in `unet.pth` (written unconditionally at the very end of
   `Trainer.run()`) or the last periodic `unet_epoch_N.pth`. Both
   `predictor.py` factories default to `unet.pth` — **do not use
   `unet_best.pth` for this baseline's scoring.**
4. **`EulerianCombinedLoss` (the only loss machinery `training/losses.py`
   offers) always computes `mse` against `sigmoid(logit)`**, never the raw
   field — this is a real, unavoidable (read-only file) deviation from
   SPEC.md §1.5's "plain regression on a continuous field, linear output,
   no sigmoid" recommendation. Trained with `loss.type: eulerian_combined,
   mse: 1.0`, everything else 0 (closest reachable approximation to "plain
   MSE, no mass term"), and `predictor.py` applies `torch.sigmoid` to the
   model's raw output before returning it to the eval harness (same
   convention every other UNet-family predictor in this repo already
   uses, per `Baselines/common/LOG.md` point 2). Flagged, not silently
   absorbed.
5. **Channel-convention check (task step 2) PASSED.** Verified two ways
   against `Baselines/NFD/configs/nfd_train_3ch.yaml`'s dataset, `pme` env:
   - `occ0` (channel 0) is `torch.allclose`-identical between the new
     3-channel dataset and the existing validated `genesis` (2-channel)
     dataset for the same indices — confirms nothing about occupancy
     construction changed.
   - Per-channel **centroid** of channel 1 vs the world→pixel-converted
     `p_start`, and of channel 2 vs `p_stop` (`PileSweepData.get_raw_action`
     + `raw.to_pxl`/`raw.ctr_in_PXL`, dim0=world_x/dim1=world_y convention,
     the same one `_draw_particle_grid`'s docstring documents for occ0),
     agree to **<0.006 px** across 7 sampled transitions spanning the pool.
     (An earlier attempt using footprint-IoU-at-a-fixed-threshold gave a
     misleading ~0.5 IoU — a red herring caused by comparing a
     full-intensity render against the old code's intentionally
     *asymmetric* 0.5/1.0-intensity union at one fixed absolute threshold,
     not a location bug; centroid comparison, which is invariant to a
     uniform intensity rescaling, resolved it cleanly.)

**Not yet done at time of this note:** training run, eval-harness scoring,
`exp0026_kcurve_exact.py`, ablation. See "Running notes" below for
what's in flight.

---

## SUMMARY FOR HANDOFF (read this first)

**Status: DONE.** Deliverable is `Baselines/NFD/SPEC.md`.

**One-line takeaway:** the paper's action encoding is two *separate*
rendered-pusher-pose channels (`aₜ := [r(xₜ), r(xₜ₊₁)]`, in_channels=3
total with state); this repo's existing pipeline (both
`Genesis/training/dataset.py` and `model/eulerian_wrapper.py`) instead
unions the two renders into one channel via an intensity trick
(in_channels=2). That's the single biggest fidelity gap, and it's cheap to
fix — `transforms/functional.py::draw_plate_soft` already exists and is
already called twice per sample; the fix is to stop combining its two
outputs.

**Key findings (see SPEC.md for full evidence/citations):**
- Prediction target: absolute next density field, not a delta or a flow
  field (contra `model/SpatTransNet.py`'s warp-field approach, which is a
  different paper's idea).
- Loss: plain Frobenius-norm-squared (~MSE), no BCE/dice/mass/sharpness
  terms in the paper. The repo's current best model already adds a `mass`
  term beyond what the paper describes — flagged as a repo addition.
- FiLM is confirmed **not** from the paper (no conditioning vector
  anywhere in the paper's method), and confirmed to be a deliberate
  in-house extension citing a different reference (Perez et al. AAAI
  2018) in `model/NFDUNetFilm.py`'s own docstring. Additionally verified
  the physics vector FiLM conditions on is dataset-wide **constant**, not
  per-sample, in the actual `.pt` files — so dropping it for `unet-modular`
  is argued to cost ~nothing here, not just "the paper didn't have it."
- Paper's Fig.7 architecture reproduces exactly via `unet-modular`'s
  `features=[4,8,16]` (bottleneck auto-doubles to 32, matching the figure)
  + `final_kernel_size=1` (paper's last conv is 1×1, not 3×3 like the
  rest) + `bottleneck_type="None"`. Verified by instantiating the class
  directly: 30,541 params, forward pass `(B,3,64,64)→(B,1,64,64)` runs.
- Paper gives **no** optimizer/lr/batch/epoch numbers for the dynamics
  network anywhere (checked full extracted text) — training recipe in the
  spec is explicitly inference, anchored to this repo's own trainer
  defaults and the existing best-model's recipe.
- Multi-step/recurrent training is ambiguous in the paper text (single
  sentence, no accompanying loss equation) — treated as rollout-time-only
  chaining of a single-step-trained model, medium confidence, flagged as
  R1.

**Files delivered:** `Baselines/NFD/SPEC.md`, this log.
**Not delivered (out of scope for this agent):** any code, any training
run, any modification to `model/UNetModels_modular.py`,
`model/NFDUNetFilm.py`, or `registry/model_registry.py`.

---

## Running notes

- 2026-09-08, start. Read `Baselines/ORCHESTRATION_LOG.md` first per
  instructions. Confirmed scope: pooled L20mm+L40mm training data, per-cell
  eval, comparison target is the existing UNet-FiLM (`accuracy` 0.419/0.504).
- PDF tooling: `pdftoppm`/poppler-utils not installed and no sudo available
  in this sandbox. Used `pypdf` (already in the `pme` conda env) for text
  extraction, and installed `pymupdf` via pip (writable, no sudo needed) to
  rasterize page 12 (Fig.7, the architecture diagram) to PNG for direct
  visual reading, since the channel-count numbers are embedded in the
  figure, not extractable as text. Noted for auditability in SPEC.md §5 R6.
- Gotcha: the paper's actual filename on disk is
  `Neural Field Dynamics Model for\nGranular Object Piles Manipulation.pdf`
  — a **literal embedded newline** between "for" and "Granular" (confirmed
  via `os.listdir`, not a display artifact of `ls`). Any script opening this
  file needs the literal `\n` in the path string, not a space.
- Read, in order: `model/NFDUNetFilm.py`, `model/UNetModels_modular.py`,
  `registry/model_registry.py` (factories for `unetfilm`, `unetfilm-shallow`,
  `unet-modular`, plus the other Eulerian factories for contrast — `nca`,
  `spatial-transformer`, to make sure NFD's "predict the field directly"
  claim wasn't confused with the STN's warp-field approach),
  `transforms/functional.py` (`build_action_delta` — confirmed this is a
  Lagrangian/PropNet-only function, unrelated to the Eulerian action
  channel; `draw_plate_soft` — confirmed this IS the same kind of
  differentiable rasterizer the paper describes as `r(x)`),
  `transforms/representation.py` (`eulerian_aliases` — confirmed this just
  aliases `input`/`target` for metric code, does not itself build the
  input tensor), `Genesis/training/dataset.py::_draw_plate` (found the
  actual union-of-two-renders construction: `1-(1-occ1)*(1-occ2)` with
  intensities 0.5/1.0), `model/eulerian_wrapper.py` (~line 1231-1370,
  confirmed the deployment-time path does the same union via
  `torch.maximum`), `configs/model/unetfilm.yaml`,
  `configs/dataset/genesis_slates_multistep_n20_L20mm_train.yaml`,
  `configs/training/expB_unetfilm_slates_multistep_n20_L20mm.yaml` (the
  actual recipe for the number-to-beat), `training/losses.py`
  (`EulerianCombinedLoss`, confirmed `mass` term is a repo addition beyond
  plain MSE), `training/trainer.py::_build_optimizer` (confirmed
  Adam+StepLR is the only supported combo).
- Checked directly (not just read) whether physics varies per-sample:
  loaded `Genesis/data/slates_multistep/n20_L20mm_train/_0_data.pt` and
  four others — keys are only `states, states_, p_starts, p_stops, angles`,
  no physics fields, confirming physics is a global constant for this
  dataset, not something FiLM could be meaningfully learning from per
  sample.
- Instantiated `UNetModels_modular.UNet` directly with the recommended
  config (`features=[4,8,16]`, `in_channels=3`, `final_kernel_size=1`,
  `bottleneck_type="None"`) to get an exact parameter count (30,541) and
  confirm the forward pass shape at 64×64 — no training, just construction
  + one forward call, consistent with "design doc, not implementation."
- Received mid-task correction from orchestrator: base spec on
  `UNetModels_modular.UNet`/`unet-modular`, not `NFDUNetFilm.py`; keep
  `NFDUNetFiLM` as a comparison row only; make explicit that
  `unet-modular`'s `forward(x)` takes no physics so the action must be
  channel-only; still resolve whether FiLM is paper or repo. Applied to the
  spec before finishing — see SPEC.md's "Scope correction" note at the top.
