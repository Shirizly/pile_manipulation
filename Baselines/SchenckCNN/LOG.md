# Schenck CNN baseline — LOG

Agent: B3-schenck-impl. Branch `baselines/overnight`. Implements, trains,
and scores the Schenck CNN baseline per `Baselines/SchenckCNN/SPEC.md`
(written by A2-grid-specs) and the scope in `Baselines/ORCHESTRATION_LOG.md`.

---

## SUMMARY FOR HANDOFF (read this first)

**Status: IN PROGRESS.** (updated as the run proceeds; see bottom for the
latest state.)

**Scope, per SPEC.md and the task brief: single-tower ablation ONLY.** A
fully-convolutional stack of 16x [Conv 32@3x3, ReLU], no pooling/upsampling,
ending in Conv 1@1x1, residual output (predicted delta added onto occ0),
plain L2 loss. The paper's headline two-tower scoop&dump-net (with its
inter-tower mass-conservation channel) is explicitly OUT OF SCOPE: it exists
to separate a lift-and-carry phase (scoop) from a pour phase (dump); our
task is one continuous plate push with nothing picked up, carried, or
poured, so there is no scoop/dump split to build an architecture for.

**Two adaptations applied, per the task brief:**
1. **Occupancy substitutes for height-map.** We have a 64x64 soft/binary
   occupancy field (`Baselines/common/data.py`'s `occ0`/`occ1`), not a real
   depth-camera height-map. No height synthesis is attempted; occupancy is
   used as-is as the single state channel the residual is added onto.
2. **Heading comes from `p_stop - p_start`, never from `angle`.** `angle`
   (the blade FACE orientation, recoverable only mod 180 deg per
   ORCHESTRATION_LOG's verified data fact 3 / claim C-018) is used ONLY to
   orient the rendered plate FOOTPRINT shape in the action channels (exactly
   how `PileSweepData._draw_plate`/`draw_plate_soft` already use it to build
   every ground-truth grid in this dataset) — never to derive a travel
   direction. Direction of motion enters only through the two literal
   endpoint positions `p_start`/`p_stop`, rendered as two SEPARATE channels.

**What was reused from B2 (NFD), per the task brief's explicit instruction
not to invent a third action-rasterisation approach:**
`Baselines/SchenckCNN/action_encoding.py` copies, verbatim, the
world-to-pixel conversion and `draw_plate_soft` call pattern from
`Baselines/NFD/predictor.py::NFDPredictor.predict_occ`'s 3-channel branch
(`raw.to_pxl`/`raw.ctr_in_PXL` for the world->pixel map, `raw.configs[0]
["plate"]["size"]` for plate geometry, `max(0.5, 1.5*raw.resolution_scale)`
for sigma) — same primitive, same formula, same plate-geometry lookup. Like
NFD's faithful (non-ablation) arm, `p_start`/`p_stop` are rendered as two
SEPARATE full-intensity channels (not the repo's asymmetric 0.5/1.0 union).
On top of that shared piece, one thing was ADDED that NFD does not need:
Schenck's paper tiles each action angle across a constant-valued channel
(SPEC.md sec 2); we tile the single blade `angle` (no separate start/end/
roll angles for a rigid push) as one more constant channel. Total input:
occ0 (1) + r_start (1) + r_stop (1) + angle (1) = **4 channels**.

**Files:**
- `model.py` — `SchenckCNN`: 16x[Conv32@3x3+ReLU] (no pooling) -> Conv1@1x1
  -> residual add of channel-0 (occ0). Linear (non-logit) output head,
  matching the paper's plain-L2-regression framing (contrast the repo's
  UNet-family predictors, which are trained against a BCE-style logit and
  need a `sigmoid` at eval time — Schenck's single-net does not).
- `action_encoding.py` — `build_action_channels`/`build_input`, the reused
  action-map construction described above.
- `train_schenck.py` — standalone training loop (NOT routed through
  `training.trainer.Trainer`/the shared registries, unlike NFD — a
  from-scratch architecture with plain MSE and no physics conditioning is
  simpler as a ~100-line manual loop than as new registry entries). Adam,
  lr 5e-4 (the paper's one stated hyperparameter, SPEC.md sec 3), batch 32,
  plain `nn.MSELoss()`. Trains on the pooled
  `configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml` split
  (23,040 transitions), action channels precomputed once (a single batched
  `draw_plate_soft` call over the whole pool, not re-rasterised per epoch).
- `predictor.py` — `SchenckPredictor`/`build_predictor()`, the
  `Baselines/common/eval_baseline.py` predictor contract. No sigmoid (see
  above — `SchenckCNN.forward` already returns the residual-added linear
  value).
- `check_axes.py` — axis-convention self-check (see below), run BEFORE any
  training per the task brief's step 2.

---

## Running notes

- **Read first, in order:** `Baselines/ORCHESTRATION_LOG.md`,
  `Baselines/SchenckCNN/SPEC.md`, `Baselines/common/LOG.md` (+
  `Baselines/common/data.py`, `Baselines/common/eval_baseline.py` source),
  `Baselines/NFD/SPEC.md` sec 1.2-1.3 + `Baselines/NFD/nfd_lib.py` +
  `Baselines/NFD/predictor.py` (for the reuse target), and
  `fit_linear_foresight.py::actions_to_pixels` (to cross-check the
  world->pixel convention independently of NFD's own code, since both
  ultimately derive from the same `PileSweepData._draw_plate`/`to_pxl`/
  `ctr_in_PXL` construction — confirms `plate_pos = p_starts * to_pxl +
  ctr_in_PXL` is literally the dataset's OWN internal formula, not a
  re-derivation, so the action-encoding module's correctness is by
  construction, not just by empirical check).
- Conda: `~/miniconda3` does not exist; the real root is `~/anaconda3`
  (`source ~/anaconda3/etc/profile.d/conda.sh && conda activate pme`).
  Confirmed `torch==2.11.0+cu130`, CUDA available.
- **Axis-convention check (task brief step 2), run BEFORE training:**
  `PYTHONPATH=. python Baselines/SchenckCNN/check_axes.py`. Picked the most
  x-dominant and most y-dominant pushes in the pooled train pool, built this
  module's action channels, and compared the pixel-space displacement of
  the rendered footprint (r_stop argmax - r_start argmax) against the
  world-displacement's dominant axis, per `actions_to_pixels`' own
  documented convention (`dim0 <- world_x, dim1 <- world_y`).
  **PASSED both cases**: x-dominant push (world dx=+0.040m, dy=~0) ->
  pixel argmax shift (drow=+20, dcol=0), row/dim0-dominant, matching;
  y-dominant push (world dy=-0.040m, dx=~0) -> pixel argmax shift (drow=0,
  dcol=-20), col/dim1-dominant, matching. No axis flip/transpose hazard
  (the C-018 failure mode) found in this module's construction.
- Wrote `model.py`, `action_encoding.py`, `train_schenck.py`,
  `predictor.py`, `check_axes.py`.
- Launched a 2-epoch smoke test through `Baselines/common/gpu_lock.sh` on
  the pooled train config, per the task brief's step 3 (must go through
  scoring before any long run). GPU lock queued behind other agents'
  concurrent training as expected — this is normal per
  `ORCHESTRATION_LOG.md`'s hardware-constraint note, not a hang.

_(continued below as the run progresses)_
