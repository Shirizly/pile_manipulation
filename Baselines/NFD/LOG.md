# Baselines/NFD — LOG

Agent: A2-grid-specs (design), then B2-nfd-impl (implementation/training/
scoring). Branch `baselines/overnight`. Task: non-FiLM NFD baseline, built on
`model/UNetModels_modular.py::UNet` (`unet-modular` registry entry), per the
user's explicit scope correction (originally briefed against "the UNet
files" generically, then pinned to `UNetModels_modular.UNet`, not
`NFDUNetFiLM`).

---

## B2-nfd-impl HANDOFF (implementation phase, read this first)

**Status: DONE.** Working model trained and scored on both eval cells
(primary 3-channel run), plus the 2-channel ablation trained and scored.
Files added, all under `Baselines/NFD/` (nothing outside touched):
`nfd_lib.py` (dataset+model registrations), `train_nfd.py` (thin Trainer
driver), `predictor.py` (eval-harness plug-in), `configs/nfd_train_3ch.yaml`
(primary), `configs/nfd_train_2ch_ablation.yaml` (ablation). Headline
numbers and the ablation result are in the tables further down this file.

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

**Training + scoring: DONE.** Primary 3-channel run trained 100/100 epochs
(`Baselines/NFD/runs/nfd_3ch/`, ~2h02m wall clock, shared GPU with two other
agents' jobs queued behind/around it via `gpu_lock.sh`): train loss
0.0658→0.00658, val loss →0.00650 (best 0.006445 at epoch 86, `unet_best.pth`
— this is what `predictor.py` loads), test `hard_iou`=0.706, `hard_dice`=0.815.
Loss curve genuinely monotonic-ish (not a lock-timeout no-op) — confirmed by
reading the full epoch-by-epoch log, not just the exit code.

**Scored results (`Baselines/common/eval_baseline.py`, pooled-train-fit
reference operators, per-cell eval):**

| metric | L20mm | L40mm | reference (per-cell UNet-FiLM) |
|---|---|---|---|
| `accuracy` (all 3 steps) | **0.4158** | **0.5124** | 0.419 / 0.504 |
| `accuracy` step0 only | 0.4166 | 0.5594 | — |
| `slateK_exact`, K=32 | 0.9751 | 0.9893 | per-cell linear 0.9670/0.9790 (K=128) |
| `slateK_exact`, K=128 | 0.9646 | 0.9925 | GNN 0.966 (K=128, L20mm); Schenck 0.9455 (K=128, L20mm) |
| `worstK`, K=32 | 0.4564 | 0.4495 | (lower is worse; linear 0.7074/0.6601 at K=32) |
| `worstK`, K=128 | -0.0157 | -0.0000 | (near-0/negative = near-oracle at this K, same as linear/oracle) |
| `regret_dv` (sampled), K=32 | 0.0010 | 0.0014 | linear 0.0020/0.0028 |
| `regret_dv` (sampled), K=128 | 0.0017 | 0.0010 | linear 0.0025/0.0016 |

All numbers `goal=corner` (per coordinator: `goal=center` is C-040's known
`dv_true≈0` degeneracy on this eval split — confirmed here too, `helpful 0%`
on both cells — not a useful discriminator, not reported as a finding).

**Honest read: this pooled, non-FiLM, 30.5K-parameter NFD MATCHES/slightly
EXCEEDS the per-cell UNet-FiLM reference on raw `accuracy`** (0.4158 vs 0.419
on L20mm — within noise; 0.5124 vs 0.504 on L40mm — a genuine small win),
and its `slateK_exact` at K=128 is competitive with or better than every
other peer baseline scored so far (GNN 0.966, Schenck 0.9455, per-cell
switched-linear 0.9670/0.9790) despite NFD being pooled (harder regime) and
non-FiLM (repo's own convention, dropped per SPEC.md's argument that FiLM's
conditioning vector is dataset-wide constant here). Do not over-read this as
"NFD beats FiLM" in general — same caveat the orchestration log already
states: pooled-vs-per-cell is a confound, not isolated here. Schenck's CNN
has a notably higher raw `accuracy` (0.512/0.559) but a WORSE `slateK_exact`
at K=128 on L20mm (0.9455 < this run's 0.9646) — the same
kind of accuracy/ranking-quality dissociation the register already
documents for L10mm (C-045); flagged for the reader, not re-litigated here
since it isn't this baseline's claim to make.

**Two things stated explicitly per coordinator's request:**
1. **Device during scoring: CPU.** `Baselines/common/eval_baseline.py` never
   calls `.cuda()`/`.to("cuda")` anywhere in its pipeline, and
   `Baselines.common.data.load_cell` builds all tensors via
   `torch.stack`/`torch.tensor` with no device argument -- so `batch.occ0`
   (and everything else in `PredictorBatch`) is on CPU, and
   `NFDPredictor.predict_occ`'s `self.model.to(device)` follows that batch's
   device, i.e. also CPU. Scoring both cells (23,040-transition operator fit
   + 7,680-row scoring, each) took well under a minute of wall time on CPU
   for the 30.5K-parameter UNet -- no GPU/`gpu_lock.sh` needed for scoring,
   consistent with `Baselines/common/LOG.md` point 3's own note that this
   whole harness is CPU work.
2. **3-channel input convention check: PASSED** (see the earlier section of
   this log for the two checks: `occ0` byte-identical to the existing
   validated 2-channel dataset; per-channel centroid of channels 1/2 matches
   world-to-pixel-converted `p_start`/`p_stop` to <0.006 px across 7 sampled
   transitions spanning the pool).

**Gotcha hit and fixed (worth flagging for other agents using the pooled
config pattern):** the shared pooled dataset config style
(`configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml`, val_pct=0/
test_pct=0) crashes `training.trainer.Trainer.from_config` outright
(`PileSweepData` raises `ValueError("No configs found for dataset.")` on
the empty val/test split) because `Trainer.from_config` unconditionally
builds train/val/test datasets, unlike `Baselines.common.data.load_cell`
(which only ever requests "train"). This baseline's own dataset config
blocks (`Baselines/NFD/configs/nfd_train_3ch.yaml` /
`nfd_train_2ch_ablation.yaml`) use `val_pct: 5, test_pct: 5` instead — fixes
the crash and gives a real (if small) validation slice for early-stopping/
best-checkpoint tracking, matching SPEC.md's own recommendation. Any other
agent driving `training/trainer.py::Trainer` (not `Baselines.common.data`)
off a val_pct=0/test_pct=0 dataset config will hit the same crash.

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

## B2-nfd-impl running notes (implementation/training/scoring phase)

- Read `Baselines/ORCHESTRATION_LOG.md`, `SPEC.md`, `Baselines/common/LOG.md`
  first per brief. Then read `registry/model_registry.py` (confirmed
  `unet-modular`'s factory never forwards `final_kernel_size`),
  `model/UNetModels_modular.py` (full `UNet`/`DoubleConv` implementation),
  `registry/dataset_registry.py` (`_build_genesis_dataset`,
  `EulerianDatasetWrapper`), `Genesis/training/dataset.py`
  (`PileSweepData.__init__`/`_create_grids`/`_draw_plate`/`__getitem__`,
  confirmed the exact union line `1 - (1-occ1)*(1-occ2)`),
  `Baselines/common/data.py` and `eval_baseline.py` (predictor contract,
  `PredictorBatch` fields, confirmed `p_start`/`p_stop` are WORLD metres not
  pixels), `training/trainer.py` (confirmed checkpoint naming, `state_dict()`
  has no "model." prefix, augmentation is channel-count-agnostic,
  `_get_log_dir`/`_try_resume` semantics), `training/losses.py`
  (confirmed `EulerianCombinedLoss` always sigmoids before computing `mse`),
  `transforms/functional.py::draw_plate_soft` (signature/convention),
  `transforms/representation.py` (confirmed channel-count-agnostic).
- Wrote `nfd_lib.py` (`PileSweepData3Ch` + both registrations),
  `train_nfd.py`, `predictor.py`, both configs. Verified by direct
  instantiation: 30,541 params (matches SPEC.md exactly), forward pass
  `(B,3,64,64)→(B,1,64,64)`, `final_conv` is `Conv2d(4,1,kernel_size=(1,1))`.
- Channel-convention check (task step 2): built the 3-channel dataset,
  compared `occ0` against the existing validated `genesis` (2-channel)
  dataset for the same indices (`torch.allclose`, exact match) and compared
  channel 1/2 centroids against `p_start`/`p_stop` converted to pixels via
  `raw.get_raw_action`/`raw.to_pxl`/`raw.ctr_in_PXL` — <0.006 px error across
  7 samples. An earlier footprint-IoU-at-fixed-threshold attempt gave a
  misleadingly low ~0.5 IoU; diagnosed as an artifact of comparing a
  full-intensity render against the old code's intentionally asymmetric
  0.5/1.0-intensity union at one absolute threshold, not a location bug —
  resolved by switching to a threshold-free centroid comparison.
- Dry-ran the full `eval_baseline.py` harness (train-cfg fit + L20mm score)
  against a randomly-initialized (untrained) checkpoint before spending any
  GPU time, to catch predictor-plumbing bugs cheaply on CPU — ran clean
  (accuracy came out very negative, as expected for random weights; no
  crash). This is what caught nothing wrong with the predictor itself.
- **Bug found and fixed on the first real smoke-test attempt (2 epochs):**
  `Trainer.from_config` unconditionally builds train/val/test datasets;
  with `val_pct: 0, test_pct: 0` (copied from the shared pooled dataset
  config), `PileSweepData` raises `ValueError("No configs found for
  dataset.")` building the empty val split. Fixed by setting `val_pct: 5,
  test_pct: 5` in this baseline's own dataset config blocks (not the shared
  file) — also switched `predictor.py`'s default checkpoint from `unet.pth`
  to `unet_best.pth` now that validation is real. Re-ran the 2-epoch smoke
  test after the fix — trained cleanly (loss 0.062→0.018, val 0.024→0.016).
- Hit one transient `gpu_lock.sh: syntax error` / `unexpected EOF` on two
  separate invocations — both traced to another agent concurrently editing
  that shared file while my process read it mid-write (confirmed: the file
  is syntactically valid both times when checked with `bash -n` right after,
  and one of the concurrent edits was itself a comment addressed to
  "B2-nfd" fixing a PYTHONPATH issue). Not a bug in this baseline's own
  code; retried and it went through.
- Launched the real 100-epoch run against `nfd_train_3ch.yaml` (default
  `output.log_dir: Baselines/NFD/runs/nfd_3ch`); it queued behind another
  agent's GNN training on `gpu_lock.sh` for ~2 min then ran for ~2h02m wall
  clock (per-epoch time varied 51s-205s depending on GPU contention from
  concurrently-queued Schenck/GNN-eval jobs). Confirmed genuine training
  (not a lock-timeout no-op) by reading the full per-epoch log: loss
  monotonically dropped 0.0617→0.00658, val IoU rose to 0.74, `unet_best.pth`
  saved at epoch 86.
- Scored both eval cells (`eval_baseline.py`, CPU, no `gpu_lock.sh` needed —
  see "device" note above) and ran BOTH `scripts/probes/exp0026_kcurve_exact.py`
  (closed-form `slateK_exact`/`worstK`/`rank_profile`) and
  `scripts/probes/exp0026_kcurve.py` (sampled `regret_dv`/`pick_pctile`) on
  each dV cache, `--models persistence,mean-delta,linear,nfd_unet3ch,oracle`,
  `--ks 2,4,8,16,32,64,128`, `--goal corner`. See numbers table above.
  Outputs: `Baselines/NFD/runs/nfd_{L20mm,L40mm}_{accuracy.json,
  dv_cache.pt,kcurve_exact.json,kcurve_sampled.json}`.
- 2-channel ablation (`nfd_train_2ch_ablation.yaml`) trained after the
  primary run's commit, per task priority order: 100 epochs, GPU uncontended
  this time (other agents' jobs had finished), ~1h27m wall clock (vs ~2h02m
  for the primary run, entirely explained by GPU contention, not the
  1-channel-narrower input). Best val loss 0.006459 (epoch 99), test
  `hard_iou`=0.706 — essentially identical to the 3-channel run's 0.706.
  Scored both eval cells (`Baselines.NFD.predictor:build_predictor_2ch_ablation`)
  and ran `exp0026_kcurve_exact.py`. See ablation table below.

## Ablation result: 3-channel (faithful) vs 2-channel (repo's own union)

Same architecture/recipe/loss throughout (`features=[4,8,16]`,
`final_kernel_size=1`, `bottleneck_type=None`, `residual=true`, plain MSE,
100 epochs, pooled L20mm+L40mm train, `val_pct/test_pct=5/5`) — only the
action-channel encoding differs (`nfd-genesis-3ch` dataset, in_channels=3
vs the existing `genesis` dataset, in_channels=2).

| metric | L20mm 3ch | L20mm 2ch | L40mm 3ch | L40mm 2ch |
|---|---|---|---|---|
| `accuracy` | 0.4158 | **0.4211** | 0.5124 | **0.5145** |
| `slateK_exact` K=32 | 0.9751 | **0.9767** | 0.9893 | **0.9900** |
| `slateK_exact` K=128 | **0.9646** | 0.9580 | **0.9925** | 0.9904 |
| test `hard_iou` | 0.706 | 0.706 | — | — |

**Honest finding: at this model scale (30.5K params) and dataset size
(~23K pooled transitions), the 3-channel split action encoding does NOT show
a measurable, consistent advantage over the repo's existing 2-channel
0.5/1.0-intensity union.** Differences are small in both directions (2ch
slightly ahead on raw `accuracy` and `slateK_exact` at K=32; 3ch slightly
ahead at K=128) and well within what a single training seed's noise could
produce — this is a null result, not a "3ch wins" or "2ch wins" result, and
it should be reported as such rather than rounded toward either direction.

This does NOT contradict SPEC.md's Appendix-C.4 evidence (DVF vs
"DVF-Improved", a ~3-4x error reduction) — that ablation compares a
**vector-based** action representation against a **field-based** one; both
arms trained here are already field-based (`draw_plate_soft` renders in
both), so this ablation tests a narrower, different question (union vs.
split within an already-field-based encoding) and the paper gives no
evidence either way on that specific question. Do not cite this result as
evidence about the paper's actual headline claim.
