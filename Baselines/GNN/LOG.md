# GNN baseline — LOG

## SUMMARY (2026-09-10): camera-only correction, node-count generalisation, cross-corpus report

**The node-construction pipeline was corrected.** Every entry below this
one describes a version of the GNN baseline that fed privileged
ground-truth simulator state (`states`/`states_`) directly to the graph
nodes -- a real deployment has no such sensor, only a top-down camera. See
`SPEC.md`'s "CORRECTION" section for the full rationale and
`Baselines/GNN/perception.py` for the fix: node positions are now
constructed via foreground-extraction + Farthest Point Sampling on the
occupancy raster alone (training labels still use privileged state, which
is fine -- see that section for why). `predictor.py`, `dataset/
dataset_genesis_gnn.py`, and `train/train_genesis_gnn_dyn.py` were all
rewritten accordingly; every accuracy/slateK_exact number in the older
entries below was computed under the OLD (incorrect) pipeline and should
not be cited going forward.

Two models were retrained under the corrected pipeline: `runs/ckpt_best.pth`
(pooled L20mm+L40mm slates, `n_particles=20`) and
`runs/randlen_train_all_n30/ckpt_best.pth` (pooled overnight_randlen TRAIN
corpus, all 5 groups n20+n50, `n_particles=30` -- node count is now a free
hyperparameter decoupled from any cell's true particle count, which is
what let one model pool across n20 and n50 cells at all).

Both models, plus `Baselines/NFD/runs/nfd_3ch_randlen`, were then scored
across L20mm, L40mm, and the overnight_randlen held-out test in one
cross-corpus, multi-metric report (`Baselines/common/eval_report.py`) --
image `accuracy` (GNN compared against a node-resampled ground truth, see
`docs/experiments/METRICS.md`) and `slateN` capture across 3 goal shapes x
3 value functions. Full numbers, caveats, and threats:
**`docs/experiments/EXP-0001-cross-corpus-gnn-nfd-report.md`**
(claim C-001, `docs/experiments/REGISTER.md`); raw JSON at
`Baselines/common/runs/cross_corpus_report.json`.

**Follow-up (EXP-0002):** the two overnight_randlen-trained models
(`gnn_randlen_n30`, `nfd_randlen` -- `gnn_l20l40` excluded, it never
trained on this corpus) were re-scored with the SAME code, this time
stratified by spawn mode (piled/scattered/mixed) instead of pooled.
Both models' accuracy is stable across modes (<0.04 spread) and NFD leads
GNN in every mode. See
**`docs/experiments/EXP-0002-randlen-spawnmode-stratified-report.md`**
(claim C-002); raw JSON at
`Baselines/common/runs/cross_corpus_report_spawnmode.json`.

---

## SUMMARY (agent E1-gnn-improve, rasteriser-batching pass, 2026-09-08)

**Task 1 (rasteriser batching): DONE.** `Baselines/GNN/predictor.py`'s
`predict_occ` looped in Python calling
`Baselines.common.data.rasterize_particles` once per candidate — the
cause of the previously-measured flat-in-K cost (866→832→820 us/candidate,
K=32→1024). `Baselines/common/data.py` is read-only, so the fix lives
entirely in `Baselines/GNN/predictor.py`:

- The world→pixel coordinate transform and the quaternion→yaw computation
  are now vectorised once for the whole (B,20,7) batch (plain elementwise
  ops, identical formula to `Genesis/training/dataset.py`'s
  `quaternion_to_yaw`) instead of recomputed per row.
- Each candidate's actual rasterisation (`_draw_particle_grid_fast`, same
  per-particle `cv2.boxPoints`/`cv2.fillPoly`/`cv2.circle` calls as
  `PileSweepData._draw_particle_grid`, unchanged) is dispatched across a
  `ThreadPoolExecutor` (`cv2` releases the GIL) instead of a serial loop,
  so B independent candidates amortise across CPU cores.
- **Tried and reverted:** collapsing the 20 per-particle `cv2.fillPoly`
  calls into one `cv2.fillPoly(grid, boxes, 1)` call per candidate. This
  is bit-identical to the per-particle loop ONLY for non-overlapping
  polygons; our cube piles have adjacent/overlapping boxes, and
  `cv2.fillPoly` given a list of contours in one call fills them under an
  even-odd winding rule, XORing out overlaps instead of setting them to 1.
  Caught by `Baselines/GNN/scripts/verify_rasterizer.py` (batched grid had
  strictly fewer filled pixels than the per-row ground truth on the very
  first real candidate tried, e.g. 117 vs 122). Reverted to one
  `cv2.fillPoly` call per particle; documented in `predictor.py`'s module
  docstring as a trap for the next person who's tempted to retry it.

**Equivalence proof:** `Baselines/GNN/scripts/verify_rasterizer.py` runs
both the batched path and the harness's own per-row `rasterize_particles`
on 200 real candidates from each eval cell (400 total) and asserts
max-abs-diff on the output grids. Result: **max abs diff = 0.0 (exact
match)** on all 400, after the fillPoly revert above.

**Re-scored both cells** (`eval_baseline.py`, same command as before):
`accuracy["gnn"]` unchanged to full float precision — L20mm
0.25271278619766235 (identical, verified against the pre-change
`gnn_L20mm_accuracy.json`), L40mm 0.3991153836250305→0.3991 (matches to
displayed precision; step-by-step and dv-cache diagnostics also
unchanged). No `accuracy` drift on either cell.

**Timing (`Baselines/common/benchmark_time.py`, re-run on the now-idle
GPU, same harness, 20 CPU cores available):**

| K | old us/candidate | new us/candidate |
|---|---|---|
| 1 | 2952 | 2296 |
| 32 | 866.6 | 850.9 |
| 128 | **832.4** | **685.3** |
| 1024 | 820.2 | 671.9 |

**~18% faster at K=128/1024, but still flat in K and still far from
NFD's 38.7 us.** This is a real, modest speedup, not the dramatic
amortisation the K=1→K=128 shape of NFD/mean-delta shows — thread-pool
parallelism helps (candidates run concurrently instead of serially) but
each candidate's own per-particle `cv2` work (20× `boxPoints`+`fillPoly`,
inherently serial per candidate since each needs its own pixel buffer) is
unchanged and still dominates. A genuinely NFD-scale fix would need to
either (a) rewrite the rasteriser as a native vectorised/batched routine
across particles too (blocked here: proven above to be non-trivial —
naive multi-polygon batching silently changes overlap semantics — and
`Baselines/common/data.py` is read-only besides), or (b) move
rasterisation onto the GPU entirely. Neither attempted this pass; reported
as a null-ish result on the "make it NFD-fast" framing but a real,
verified win on the "batch what can be safely batched" framing.

**Task 2 (per-node rotation ablation): NOT ATTEMPTED** — tool-call budget
spent on task 1 (equivalence proof + the fillPoly false start + two
re-scores + a re-timed benchmark). Left for a follow-up agent; see
`Baselines/GaussianSplatting/ASSESSMENT.md` for the recipe pointer (model
has no orientation head today; `predictor.py` reattaches the input
quaternion unchanged, `model/gnn_dyn.py` would need an added rotation
head and retraining, ~500 epochs / ~20 min per the original training log
below).

**Files touched:** `Baselines/GNN/predictor.py` (rewritten, see above),
`Baselines/GNN/scripts/verify_rasterizer.py` (new, the equivalence proof).
`Baselines/common/TIMING.md`/`timing_results.json` re-generated by the
benchmark re-run (committed per the task's explicit allowance, despite
`Baselines/common/` otherwise being read-only to this agent).

---

## SUMMARY (agent B1-gnn-impl, implementation phase, 2026-09-08)

**Status: DONE. Trained, scored on both eval cells, beats the linear
operator (its closest pooled-fit competitor) on `accuracy` on both cells
and on `slateK_exact`/`regret_dv`/`worstK` at nearly every K.**

### Headline numbers

All rows below are **pooled-train fits** (mean-delta/linear refit on the
same L20+L40 pool the GNN trains on, via `Baselines/common/eval_baseline.py`)
— NOT a per-cell-only fit (fitting the reference operators separately on
each cell's own train split rather than on the shared pool), which
produces a systematically easier fit and `linear`/`mean-delta` scores
~5-10 points higher on this account alone, with no relation to how well
the operator actually generalises. Always compare a model's numbers
against THIS file's own `mean-delta`/`linear` rows — refit on the exact
same pool the model being judged trained on — never a per-cell fit
reported elsewhere, which would misread a win as a loss purely from the
fitting-procedure difference.

**`accuracy`** (`Baselines/GNN/runs/gnn_{L20mm,L40mm}_accuracy.json`):

| model | L20mm | L40mm |
|---|---|---|
| persistence | 0.000 | 0.000 |
| mean-delta (pooled fit) | 0.054 | 0.118 |
| linear (pooled fit) | 0.200 | 0.381 |
| **gnn** | **0.253** | **0.399** |
| oracle | 1.000 | 1.000 |

GNN beats the pooled linear operator by +0.053 (L20mm) and +0.018 (L40mm).
Also notable: GNN's per-step accuracy **rises** across the 3 rollout steps
(L20mm: 0.239→0.251→0.270; L40mm: 0.455→0.376→0.322 — L40mm falls like
every other model, but less steeply than linear's 0.465→0.354→0.252) while
every reference row on L20mm falls. Plausible reading: the GNN's
message-passing graph re-derives local structure from `s_cur` at every
step rather than extrapolating a single global operator, so it degrades
more gracefully off-distribution (steps 1-2 are diverged rollouts, not
same-state slates, per `ORCHESTRATOR AMENDMENT`) — not independently
verified, flagged as a hypothesis.

**`slateK_exact` / `regret_dv` / `worstK`** at K=32 and K=128, goal=`corner`
only (goal=`center` is a known degeneracy for these cells — centred
target, centred pile, `dV=0` identically, "helpful 0%" is not a model
failure, confirmed in both cells' `eval_baseline.py` output):

| metric | model | L20mm K=32 | L20mm K=128 | L40mm K=32 | L40mm K=128 |
|---|---|---|---|---|---|
| slateK_exact ↑ | linear | 0.9524 | 0.9521 | 0.9787 | **0.9887** |
| slateK_exact ↑ | **gnn** | **0.9729** | **0.9658** | **0.9892** | 0.9870 |
| regret_dv ↓ | linear | 0.0020 | 0.0025 | 0.0028 | **0.0016** |
| regret_dv ↓ | **gnn** | **0.0012** | **0.0018** | **0.0015** | 0.0031 |
| worstK ↓ | linear | 0.7074 | 0.0099 | 0.6601 | **0.0048** |
| worstK ↓ | **gnn** | **0.5279** | **-0.0223** | **0.4236** | 0.0060 |

**Honest caveat — a real crossover, not a bug.** GNN wins clearly at every
K on L20mm and at K≤64 on L40mm. But at **K=128 on L40mm specifically**,
linear edges narrowly ahead on all three control-ranking metrics
(slateK_exact 0.9887 vs 0.9870; regret_dv 0.0016 vs 0.0031; worstK 0.0048
vs 0.0060) even though GNN still leads that same cell's `accuracy` (0.399
vs 0.381) and every smaller-K ranking metric. K=128 is the full 128-
candidate pool (the "pick literally the single best of everything" limit,
where `slateK`'s own denominator structure per METRICS.md gets noisy at
the tail) — investigated only enough to confirm both dV caches' step-0
subsets have the correct 20 slates x 128 candidates structure and the
crossover is not a plumbing artifact; not chased further given the
overnight time budget. Reported as-is rather than cherry-picking K.

**Ranking correlation (rho, within-slate, from `exp0026_kcurve.py`'s
bivariate-normal diagnostic):** gnn 0.948 (L20mm) vs linear 0.877 — GNN's
predicted-dV ranking tracks true dV noticeably more tightly than the
linear operator's.

### Geometry/training choices (recap; see full detail further down)

- `adj_thresh = 0.012` m, `pusher_w = 0.02` m (sourced from the blade's own
  `plate.size` in a cell's `_0_config.yaml`), `softness = 0.01` m (kept
  from the reference recipe), `particle_dens` fixed at 1000.0 (not
  measured/sampled) — all chosen/verified by `check_geometry.py` BEFORE
  training (see below), not guessed.
- Heading is derived ONLY from `p_stop - p_start`; `angles` is never read
  (confirmed hazard: it's the blade face orientation, recoverable only mod
  180°, not the travel heading — see SPEC.md and ORCHESTRATION_LOG.md).
- Trained single-step (`n_rollout=1`) on the pooled 23,040-transition
  L20mm+L40mm train split, 500 epochs, Adam lr=1e-3, batch 128,
  StepLR(100,0.5); best checkpoint at epoch 400, val_mse 3.05e-6 (mild,
  unremarkable overfit — final train_mse 2.38e-6 vs val_mse 3.07e-6).
  ~1206s (20 min) total wall-clock on the shared GPU.
- Model has no orientation head (by construction, `model/gnn_dyn.py`only
  predicts xyz) — `predictor.py` reattaches the INPUT frame's quaternion
  unchanged before rasterising, rather than inventing one. This is a
  known capability gap flagged in SPEC.md hazard/§6, not a bug.

### Hazards the next agent should know

1. **Per-cell vs pooled reference numbers are NOT directly comparable** —
   see the "Headline numbers" note above. Always compare against
   `Baselines/GNN/runs/gnn_*_accuracy.json`'s own `mean-delta`/`linear`
   rows (refit on the same pool the model being judged trained on), never
   a per-cell fit reported elsewhere.
2. **`Baselines/common/gpu_lock.sh` changed mid-run** (from a single
   exclusive flock to a 3-slot semaphore) while this agent was mid-flight
   — a transient race during that edit caused one early invocation to
   fail with a bash syntax error (file caught mid-write); retrying a
   moment later worked fine. Not this baseline's bug; flagged in case
   another agent hits the same transient during a concurrent edit.
3. **goal=`center`'s `dv_true` is identically ~0 (helpful 0%)** for both
   cells — a known degeneracy of a centred goal against an already
   roughly-centred pile (the target and the starting state coincide, so
   there is no distance left to close), not something wrong with the GNN
   or the harness. Score/report `goal=corner` for control-ranking metrics.

**Files delivered (all new, per SPEC.md section 7 — none of the vendored
`Baselines/GNN/{dataset,train,model}/*` files were touched):**
- `Baselines/GNN/geometry.py` — `compute_s_delta` (SPEC.md section 6 recipe,
  heading from `p_stop - p_start` only, never `angles`), and the three
  chosen constants: `PUSHER_W=0.02` (sourced from `plate.size`),
  `SOFTNESS=0.01` (kept from the reference), `ADJ_THRESH=0.012` (chosen
  after the geometry check below), `PARTICLE_DENS=1000.0` (fixed constant,
  per SPEC.md hazard 3).
- `Baselines/GNN/scripts/check_geometry.py` — the pre-training sanity check
  SPEC.md section 8 demands. Results below.
- `Baselines/GNN/dataset/dataset_genesis_gnn.py` — `GenesisGNNDataset`,
  reads raw `_*_data.pt` files directly (NOT `dataset/dataset_gnn_dyn.py`),
  flattens each file's 128 rows to independent single-step examples.
- `Baselines/GNN/train/train_genesis_gnn_dyn.py` — Adam(lr=1e-3,
  betas=(0.9,0.999)) + StepLR(100, 0.5), single-step (`n_rollout=1`) MSE
  loss on xyz only, batch 128, checkpoints every 10 epochs to
  `Baselines/GNN/runs/{ckpt_best,ckpt_last}.pth` + `history.json`.
- `Baselines/GNN/predictor.py` — `build_predictor()` for
  `Baselines/common/eval_baseline.py`; loads a checkpoint, runs
  `model.predict_one_step`, reattaches the INPUT quaternion (model has no
  orientation head), rasterises per-row via
  `Baselines.common.data.rasterize_particles` (harness's own routine, not
  reimplemented).

**Geometry sanity check (`check_geometry.py`, run BEFORE training):**
- s_delta mask lights up sensibly, not saturated/empty: for `n20_L20mm`
  (20mm push), ~5-16 of 20 cubes get non-trivial `s_delta` per candidate
  (mean 7.1), matching the short push's narrow longitudinal window
  (`x_loc` in `(0, 0.02m)`); for `n20_L40mm` (40mm push), 16-20 of 20 cubes
  are active (mean 19.1) — the whole pile falls inside the longer window,
  which is the expected behaviour of a hard longitudinal gate whose length
  scales with push length, not a bug.
- Printed the per-cube `(x_loc, y_loc, l_mask, w_mask)` breakdown directly
  (not just a lumped "distance to segment", which conflates the hard
  longitudinal gate with the soft lateral one and is misleading — caught
  this myself mid-check and rewrote it): cubes with `|y_loc| < PUSHER_W
  (0.02m)` correctly get `w_mask≈1.0`; cubes outside decay smoothly over
  the `SOFTNESS=0.01m` scale. No axis swap, no all-or-nothing collapse.
- Adjacency degree distribution swept over `adj_thresh ∈
  {0.008,0.010,0.012,0.015,0.020}` m (all built on `s_cur + s_delta`, exactly
  `PropNetDiffDenModel.predict_one_step`'s own construction, not
  reimplemented — `check_geometry.py` mirrors only the adjacency half to
  avoid a full forward pass): min in-degree is 2-4 at every value tested
  (never an isolated node), and `adj_thresh=0.012` is NOT saturated at the
  hard top-10 cap (`frac_at_cap` 0.47 for L20mm, 0.81 for L40mm — a real
  distance-threshold effect, not "always top-10-NN regardless of the
  literal value", the failure mode SPEC.md hazard 2 warned about).
  **Chose `adj_thresh=0.012` m** — matches SPEC.md's own recommended range
  (2-3x cube edge = 0.010-0.015m) and sits in the middle of a healthy,
  non-degenerate part of the sweep.

**Training recipe chosen:** 500 epochs, batch 128, Adam lr=1e-3, pooled
train pool (23,040 transitions), StepLR(100,0.5). Measured ~2.5s/epoch on
the shared GPU including a full validation pass on the pooled eval split
(15,360 transitions) — a 500-epoch run is ~20 min, well inside budget (the
38k-parameter model trains far faster than SPEC.md's own conservative
estimate). Smoke-tested first (`--smoke`, 2 epochs) before committing to
the full run, per the task's priority-order rule.

---

# GNN baseline — agent A1-gnn-spec log (design phase, superseded by the
# implementation above where the two disagree — SPEC.md's own
# ORCHESTRATOR AMENDMENT already documents the one place they did)

## Summary

Task: write `SPEC.md` (design doc, no training) for adapting the vendored
`PropNetDiffDenModel` GNN (Wang/Li et al., "Dynamic-Resolution Model
Learning for Object Pile Manipulation") to train on our Genesis
`slates_multistep` cube-pile data, at a **constant** N=20 node count
(resolution regressor explicitly out of scope per user decision recorded
in `Baselines/ORCHESTRATION_LOG.md`).

**Delivered:** `Baselines/GNN/SPEC.md`. No training run, no model code
changes — as instructed.

**Headline findings:**
- Graph = one node per cube (N=20 fixed), edges = distance-threshold AND
  top-10-nearest-neighbour, built on the *anticipated* post-action
  position `s_cur + s_delta`, not the current position.
- The action never becomes extra "pusher" nodes — despite the code's
  comments, `a_cur` (the node-type attribute) is always zero in both the
  reference dataset and the existing `simple_mpc/adapters.py::GNNAdapter`.
  The action instead becomes a per-node vector field `s_delta`
  (direction-to-tool-end, masked to the swept lane) fed into the particle
  encoder and used (only) to shape the graph topology.
- Architecture confirmed from the vendored checkpoint's `state_dict`
  (not guessed): `nf_effect=64` throughout, `pstep=3` message-passing
  rounds (hardcoded in `model/gnn_dyn.py:160`, not a config value), total
  **38,403 trainable parameters** — a tiny model.
- Three numeric constants from the reference recipe (`adj_thresh=0.08`,
  `pusher_w=0.8/24`, `particle_dens` sampled `Uniform(15,6500)`) are all
  tuned to a different simulator's own arbitrary coordinate scale and must
  not be reused verbatim on our metre-scale data. Found a sourced
  replacement for `pusher_w` (0.02 m, from the blade's own
  `plate.size: [0.04, 0.002, 0.01]` in a cell's `_0_config.yaml`);
  `adj_thresh` and the mask softness are argued from first principles but
  not empirically validated — flagged as the top open risk.
- Found and numerically **confirmed** a genuine frame hazard: our
  `angles` field is not the push heading, it's `heading + 90°` (blade face
  orientation), verified exactly across 4 samples. Not needed as a model
  input at all under this recipe, but a serious footgun if fed naively.
- Our data is much simpler to consume than theirs: no RGBD, no FPS/KDTree
  particle tracking, no camera/OpenGL transform chain — states are exact
  per-cube positions already. Recommended a new dataset+train script
  rather than patching the vendored ones.
- `manifest.json`'s 50-slates×3-steps×128-envs structure does not record
  which of a slate's 128 step-0 candidates was executed to reach step 1 —
  so multi-step (`n_rollout>1`) supervision, which the paper's own recipe
  uses, is not reconstructable without more digging. Recommended
  `n_rollout=1` (single-step) training as the safe, data-matching choice;
  flagged multi-step as an open question.
- Confirmed the "~23k transitions" figure in the task brief exactly:
  90 train files/cell × 2 cells (L20mm+L40mm pooled) × 128 rows = 23,040,
  using the existing `prepare_slate_multistep_split.py` 30/20 slate-level
  split (which must be reused, not re-derived, to avoid leaking a slate's
  3 steps across train/eval).

## Running notes (chronological)

- Read `Baselines/ORCHESTRATION_LOG.md` first, as instructed. Confirmed
  scope: GNN only, no resolution regressor, N=20 fixed. Confirmed
  `model/gnn_dyn.py` at repo root is byte-identical to
  `Baselines/GNN/model/gnn_dyn.py` (already known from the orchestration
  log; not re-verified by hand, took the log's word for it — cheap and
  low-risk to trust here since both are read-only references).
- Read `model/gnn_dyn.py`, `model/eulerian_wrapper.py`,
  `model/diff_mass_push.py` in full. Noted `eulerian_wrapper.py` and
  `diff_mass_push.py` are **not used by the GNN path at all** — they
  belong to the sibling Eulerian/NFD baseline (`SplatPushModel` etc.).
  They were in this agent's source list but turned out irrelevant to the
  GNN spec beyond confirming that `_gen_s_delta`-style differentiable
  action-conditioning exists elsewhere in the repo too (parallel
  construction, different model family).
- Read `train/train_gnn_dyn.py`, `dataset/dataset_gnn_dyn.py`,
  `data_gen/gnn_dyn_data.py`, both yaml configs. Found the collate
  function pads to the batch's max `particle_num` — a variable-N
  accommodation that's moot for our fixed N=20.
- Loaded the vendored checkpoint's `state_dict` directly (no conda
  activation needed — the plain shell python already had torch on path)
  and printed every tensor's shape. This is the ground truth for §2 of
  the spec; matched every shape back to the corresponding `nn.Module`
  constructor call in `model/gnn_dyn.py` by hand.
- Loaded a reference sample episode
  (`Baselines/GNN/data/gnn_dyn_data/0/`): particle npy is raw un-normalized
  PyFlex world coordinates (4 columns, 4th overwritten to 1.0 for the
  homogeneous transform); `actions.p` is a plain pickled
  `(10,4)` float64 array, range ≈ ±5, matching `wkspc_w=5.0`.
- Loaded our sample cell
  (`Genesis/data/slates_multistep/n20_L20mm/_0_data.pt`) and did the
  measurements in SPEC.md §4: confirmed all 128 rows share one starting
  state (std ≈ 4.8e-12), push length is exactly 0.02 ± 1e-6 m (matches
  cell name), push is planar (z-component of the push vector ≤ 9.4e-7 m),
  and — the one non-obvious finding — `angles` is exactly
  `atan2(dy,dx) + π/2`, not the heading itself, checked across the first
  4 rows with all 4 differences landing on π/2 to float precision.
- Read `simple_mpc/adapters.py::GNNAdapter` and its `_gen_s_delta` helper
  in full (as instructed — "someone previously intended to drive this
  model"). Confirmed it independently re-derives the same `s_delta`
  construction as `dataset_gnn_dyn.py`, and confirms `a_cur` is zeroed
  there too — good cross-check that the "no real pusher-node attribute"
  finding isn't a one-off dataset quirk.
- Extracted the paper's method section (PDF pages 3-4, via `pymupdf`
  since `pdftoppm`/`poppler-utils` was not installed and the `Read` tool's
  page-render path failed — used `fitz`/`pymupdf`'s text extraction
  instead, which was available). Confirmed the paper's high-level
  description (node/edge encoders, multi-step message passing "following
  Li et al.", MSE loss over a rollout horizon) matches the code's actual
  implementation; the paper itself carries no appendix with layer-size
  hyperparameters in these 11 pages (no separate supplementary-material
  PDF was found in `docs/reference_papers/`) — all architecture numbers
  in the spec come from the code/checkpoint, exactly as the task
  requested ("say which came from which").
- Grepped `Genesis/*.py` for the box/cube/blade physical constants needed
  to replace the reference recipe's now-inapplicable numeric constants.
  Found cube edge (0.005 m) and box volume (0.128×0.128×0.04 m) in a
  cell's own `_0_config.yaml` (`data_collection.cube_size`, `box.vol`) —
  matches the task brief's stated "0.128 m across" exactly. Found the
  blade/plate geometry the same way (`plate.size: [0.04, 0.002, 0.01]`),
  giving a sourced value for the pusher half-width (0.02 m) instead of
  guessing. Did **not** find an explicit named "adj_thresh"-equivalent
  contact radius anywhere in the Genesis code — that constant has no
  physical-simulator counterpart to source from, so §5/§9 of the spec
  correctly labels it as argued-from-first-principles, not measured.
- Read `scripts/probes/prepare_slate_multistep_split.py` in full — this
  is where the "50 slates × 3 steps × 128 envs, 30/20 slate-level split"
  structure and the exact 23,040-transition arithmetic came from. Also
  where the "which of the 128 step-0 candidates continues to step 1"
  gap was noticed (checked `manifest.json`'s `batches` list directly:
  fields are `batch_idx, slate_idx, state_library_index, step_idx,
  env_count, unresolved_after_resampling` — no executed-candidate index).
- Did not read `docs/experiments/METRICS.md` or re-derive
  `slateK_exact`/`accuracy` end-to-end — out of this agent's declared
  scope (design doc for the *model*, not the eval harness) and flagged
  explicitly as an unverified assumption in SPEC.md §9 item 4, rather than
  silently assumed.

## Tool-call budget

Used roughly 40 tool calls (reads, greps, small python snippets via Bash,
two Write calls). Stayed within the 40-55 budget given in the brief.
