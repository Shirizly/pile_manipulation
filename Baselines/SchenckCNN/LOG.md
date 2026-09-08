# Schenck CNN baseline — LOG

Agent: B3-schenck-impl. Branch `baselines/overnight`. Implements, trains,
and scores the Schenck CNN baseline per `Baselines/SchenckCNN/SPEC.md`
(written by A2-grid-specs) and the scope in `Baselines/ORCHESTRATION_LOG.md`.

---

## SUMMARY FOR HANDOFF (read this first)

**Status: DONE.** Trained 100 epochs on the pooled L20mm+L40mm train set,
scored on both eval cells through the shared harness, K-curve metrics run
on both dV caches. Checkpoint `runs/schenck.pth` (571 KB, well under the
50 MB budget).

**Headline numbers** (goal=corner for all control-utility numbers per the
coordinator: `center` is claim C-040's known dV≈0 degeneracy on these
slates, not a model failure):

| | L20mm | L40mm |
|---|---|---|
| `accuracy` (pooled-fit ref: mean-delta/linear) | 0.0544 / 0.2004 | 0.1182 / 0.3811 |
| `accuracy` **schenck_singlenet** | **0.5117** | **0.5593** |
| `slateK_exact` K=32 | 0.9826 | 0.9900 |
| `slateK_exact` K=128 | 0.9455 | 0.9938 |
| `regret_dv` K=32 (Lyapunov units) | 0.0007 | 0.0013 |
| `regret_dv` K=128 | 0.0027 | 0.0009 |
| `worstK` K=32 / K=128 | 0.484 / 0.014 | 0.533 / -0.0001 |

Against `ORCHESTRATION_LOG.md`'s per-cell reference table (mean-delta
0.086/0.138, linear 0.301/0.447, **UNet-FiLM 0.419/0.504**): the single-net
Schenck ablation's `accuracy` (0.5117/0.5593) **exceeds UNet-FiLM on both
cells**, despite being trained on the pooled set (both push lengths, one
model) rather than per-cell. Against the per-cell switched-linear at its
own intended operating point (`slateK_exact` K=128: L20mm 0.9670, L40mm
0.9790): Schenck is **slightly below** on L20mm (0.9455 vs 0.9670) and
**above** on L40mm (0.9938 vs 0.9790). Against the GNN (accuracy 0.253/0.399,
`slateK_exact` K=128 0.966 on L20mm): Schenck's `accuracy` is much higher
(0.51 vs 0.25) but its `slateK_exact` K=128 on L20mm is slightly LOWER
(0.9455 vs 0.966) — **a real accuracy/slateK_exact dissociation, the same
qualitative pattern flagged for L10mm in EXP-0029** (high image-accuracy
does not guarantee the best K=128 ranking behaviour; these are different
things the model can be good/bad at somewhat independently). Reported
honestly, not tuned toward any target — no hyperparameter search was run,
the paper's own lr (5e-4) and this repo's plate rasteriser were used as-is.

**Device note (coordinator flagged this — checked directly):**
`Baselines/common/eval_baseline.py` never calls `.cuda()`/`.to(device)` on
`PredictorBatch`'s tensors, so `batch.occ0.device` is CPU, and
`SchenckPredictor.predict_occ`'s `device = batch.occ0.device` line means
**scoring ran on CPU**, not the GPU used for training. This does not affect
correctness (same weights, same forward pass, float32 both places) — only
noted so the timing story stays honest: both eval runs (7,680 transitions
each) completed in well under a minute on CPU for this small (139,937-
param) model; a bigger model would need this checked before assuming
CPU-eval is free.

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
  `ORCHESTRATION_LOG.md`'s hardware-constraint note, not a hang. Completed:
  23,040 transitions loaded in 96s, inputs built in 2.2s, model
  139,937 params, ~20.3s/epoch on the shared GPU, loss 0.0186 -> 0.0146
  over 2 epochs. Checkpoint written to `runs/schenck_smoketest.pth`.
- **Coordinator update received mid-task**: `Baselines/common/gpu_lock.sh`
  was silently dropping `PYTHONPATH` (a fresh `bash` invocation does not
  inherit a caller's prefix-assignment `PYTHONPATH=. cmd`), which crashed
  B2/NFD's first training with `ModuleNotFoundError: No module named
  'Baselines'`. Coordinator patched the wrapper to export `PYTHONPATH` to
  the repo root itself. Verified the patched file directly (now exports
  `REPO_ROOT` computed from `BASH_SOURCE`) before relying on it. My own
  smoke test had set `PYTHONPATH=.` in the same shell as the `gpu_lock.sh`
  call, so it was unaffected either way, but the full run below goes
  through the patched wrapper's own export, per the coordinator's
  instruction.
- Ran the smoke-test checkpoint (`runs/schenck.pth`, copied from
  `schenck_smoketest.pth`) through `Baselines/common/eval_baseline.py` on
  L20mm eval (CPU-only, `CUDA_VISIBLE_DEVICES=""`, no GPU lock needed for a
  139k-param model at this batch size) to confirm the FULL scoring
  pipeline (predictor load -> accuracy.json -> dv_cache.pt) works before
  committing GPU time to a long run: **accuracy=0.1247 after just 2
  epochs** (persistence 0.0, mean-delta 0.0544, linear 0.2004, oracle 1.0 —
  note these reference numbers are POOLED-fit, not the per-cell numbers in
  ORCHESTRATION_LOG.md's table, so a direct comparison isn't apples-to-
  apples yet; this run is purely a plumbing check). Confirms predictor
  contract, checkpoint loading, and dv_cache writing all work end-to-end.
- Launched the FULL 100-epoch run through the patched `gpu_lock.sh`
  (queued behind B2/NFD's concurrent 100-epoch run, expected ~45-80 min
  wait, per the coordinator).

- **Full training run completed**: 100 epochs on the pooled train config
  through the (coordinator-patched) `Baselines/common/gpu_lock.sh`, queued
  behind B2/NFD's concurrent run as expected. Loss curve monotone-ish:
  0.017176 (epoch 1) -> 0.006570 (epoch 10) -> 0.005961 (epoch 20) ->
  0.005292 (epoch 50) -> 0.004888 (epoch 100), ~27-30s/epoch, checkpointed
  every 10 epochs to `runs/schenck.pth` (final file 571 KB). Verified from
  the actual epoch-by-epoch log (`runs/train_full.log`), not just the
  process exit code, per the task brief's instruction to verify the loss
  history.
- **Scored both eval cells** via `Baselines/common/eval_baseline.py
  --predictor Baselines.SchenckCNN.predictor:build_predictor` (no
  `gpu_lock.sh` needed for scoring, per the coordinator -- confirmed the
  harness never moves predictor-batch tensors to CUDA, so this predictor
  ran on CPU during scoring; see device note above). Wrote
  `runs/schenck_L20mm_{accuracy.json,dv_cache.pt}` and the L40mm
  equivalents.
- **Ran both K-curve scripts** (`exp0026_kcurve_exact.py` for
  `slateK_exact`/`worstK`/`rank_profile`, `exp0026_kcurve.py` for
  `regret_dv`/`pick_pctile`/the bivariate-normal extrapolation sanity
  check) on both dV caches, `--goal corner`, `--ks 32,128`, `--models
  persistence,mean-delta,linear,schenck_singlenet,oracle`. See the
  headline table above for the numbers; full output JSON in `runs/`.
- Deleted the temporary `Baselines/SchenckCNN/runs/schenck_smoketest.pth`
  reasoning: kept it (571 KB, harmless) as a record of the plumbing-check
  run; the real, final checkpoint is `runs/schenck.pth` (overwritten by
  the full 100-epoch run, confirmed by its final-epoch mtime and loss
  value matching `train_full.log`'s last line).

## Hazards for the next agent

- **`accuracy` vs `slateK_exact` are not the same axis.** This baseline is
  the clearest evidence yet (after EXP-0029's L10mm finding) that a model
  can win decisively on pixel-level `accuracy` (beating UNet-FiLM on both
  cells) while still landing slightly BELOW a much simpler per-cell linear
  operator on `slateK_exact` K=128 (L20mm: 0.9455 vs 0.9670). Do not use
  `accuracy` alone to declare a ranking-quality winner; always check both.
- **This model was trained pooled (both push lengths, one model) while the
  linear/UNet-FiLM reference numbers in `ORCHESTRATION_LOG.md`'s main table
  are per-cell.** The pooled mean-delta/linear rows printed inside this
  baseline's own `_accuracy.json` files are the correct like-for-like
  comparison; the per-cell numbers are a different (favourable-to-them)
  operating point, not an apples-to-apples baseline.
- **Two-tower scoop&dump-net was NOT attempted** (out of scope, per
  SPEC.md and the task brief) -- if a future agent is asked to revisit
  Schenck fidelity, that architecture (and its inter-tower mass-
  conservation channel) is the one piece of the paper still unreproduced
  here, and SPEC.md sec 4 explains in detail why it doesn't map onto a
  plate-push task.
- **No hyperparameter search was run.** lr=5e-4 (paper's stated value),
  batch=32, 16 layers @ width 32, 100 epochs -- all first-guess/paper/repo-
  convention values. There may be easy headroom left on `slateK_exact`
  K=128 specifically (where this model is weakest relative to its own
  `accuracy` strength) via more epochs, a lower final-lr, or an explicit
  ranking-aware auxiliary loss -- none of that was tried.
