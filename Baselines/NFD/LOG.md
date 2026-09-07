# Baselines/NFD — LOG

Agent: A2-grid-specs. Branch `baselines/overnight`. Task: design-only spec
for a non-FiLM NFD baseline, built on `model/UNetModels_modular.py::UNet`
(`unet-modular` registry entry), per the user's explicit scope correction
(originally briefed against "the UNet files" generically, then pinned to
`UNetModels_modular.UNet`, not `NFDUNetFiLM`). No training performed, no
existing repo file modified.

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
