---
# ---- identity -------------------------------------------------------------
id: EXP-0009
title: >
  Equalised-wall-clock candidate budgets differ sharply by model (7-128 of
  128) -- the gate for the larger control-comparison experiment is non-vacuous
tier: T1
mode: exploratory
date: 2026-09-13
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On an idle RTX 4070 Laptop 8GB, for the 8 timed model configurations
  (NFD-3ch, SchenckCNN, GNN, MODEL-0001 switched/global, MODEL-0002
  descriptor-only, hybrid14, hybrid94) scored against one real state from
  configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml with K real
  cycled candidate actions (K in {1,2,4,8,16,24,32,48,64,96,128}), the
  per-candidate end-to-end (preprocessing+forward) wall-clock cost is NOT
  flat across this range for every model: at least one model's N_i (the
  interpolated K at which its own end2end median crosses the fastest
  model's K=128 end2end median, 1.714ms for model0001_global) is below 128,
  making the planned equalised-wall-clock-budget control comparison
  non-vacuous.

prediction:
  supports: "at least one of the 8 models has N_i < 128 (interpolated on its own measured curve against the 1.714ms reference)"
  refutes: "every model's end2end median at K=128 is within measurement noise of its end2end median at K=1 (i.e. all 8 have N_i >= 128, scaling genuinely flat)"
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 0ddab20f
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0009-time-budget-bench/code/bench.py
  data: ["configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml"]
  code_path: experiments/EXP-0009-time-budget-bench/code/bench.py
  seed: 0
  split: "train split of the L20mm eval cfg (7680 transitions loaded; only row 0's state + K cycled real actions used, per benchmark_time.py's own MPC-inner-loop shape)"
  runtime: "~35s wall clock, cuda:0 (RTX 4070 Laptop 8GB), idle at start"
  runs: [RUN-0001, RUN-0002, RUN-0003]

budget:
  declared: "45 min wall-clock, ~110k tokens (RUN-0001); RUN-0002 amendment: separate 45min/~110k budget"
  spent: "~40 min, ~150k tokens (RUN-0001, over token budget, under wall-clock budget); RUN-0002 amendment: ~45 min wall-clock, close to token budget"
  outcome: exceeded

design:
  varied:
    model: [nfd, schenck, gnn, model0001_switched, model0001_global, model0002_descriptor_only, hybrid14, hybrid94]
    K: [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
  held_fixed:
    state: "row 0 of the loaded L20mm eval cell, rasterised once, outside every timed region"
    warmup: 5
    repeats: 15
    device: cuda (explicitly, for every model -- see Threats/device-bug fix)
  baselines: [persistence]
  metric: "candidate_throughput_ms"

noise_floor: >
  RUN-0001: IQR (q1/q3) of the 15 timed repeats per (model,K) cell, no
  formal threshold pre-declared. RUN-0002 amendment: a formal resolution
  floor WAS measured (defect 3) by re-running the SAME (model,K) config in
  3 SEPARATE process invocations -- schenck@K=128 end2end medians were
  46.09/47.84/48.11 ms across the 3 runs (~2ms / ~4.3% relative spread),
  and model0001_global@K=128 (fwd-only) gave 2.433/2.440 ms across 2 runs
  (~0.3%) but a DIFFERENT absolute value (~2.43ms) than the same logical
  quantity measured inside the main sweep process (~1.75ms end2end) --
  i.e. cross-process absolute drift can be much larger (~30-40%) than
  within-process repeat noise for models whose true cost is a few ms.
  RESOLUTION FLOOR APPLIED: differences smaller than ~5% relative (or
  ~2ms absolute at the ~45ms scale) between two numbers from DIFFERENT
  timed regions or DIFFERENT process invocations are not resolvable and
  must not be used to rank models or explain "impossible" orderings (this
  is exactly what RUN-0001's schenck fwd_ms@128=48.67 > end2end_ms@128=46.27
  was: a ~2.4ms/5% inversion, fully inside this floor, not a computation
  bug).

depends_on: [push-frame-warp-roundtrip]
establishes: [eval-baseline-scorer-batch-on-requested-device]

# ---- outcome --------------------------------------------------------------
result: >
  RUN-0001 (superseded in scope, not invalidated -- see Amendment):
  fastest model @ K=128 is model0001_global (1.714ms end2end). Derived N_i:
  model0001_global=128 (ref), model0002_descriptor_only=74.5, nfd=69.6,
  model0001_switched=54.2, schenck=7.0, gnn/hybrid14/hybrid94 all <1.
  RUN-0002 (timing-corrected, see Amendment): sync-free fix changed
  model0001_switched's K=128 end2end from 2.83ms to 2.17ms (~23% real
  reduction; the sync artifact was concentrated here, NOT uniform across
  all `.any()`-guarded models -- hybrid14/94/desc-only barely moved).
  Timing the goal-value function (lyapunov+mass_in_region) in-region makes
  descriptor-only's advantage EVAPORATE at tight budgets, not grow: its
  own N_i(T=2.5ms) goes from 128 (no value) to unresolvable/<1 (with
  value, its point-mass readout costs ~0.6ms against a ~0.03ms fwd, a much
  larger relative hit than model0001_global's image-space value cost,
  which barely moves that model's N_i). Descriptor-only does NOT become
  the fastest once scoring is included -- if anything it becomes worse
  relative to model0001_global. Budget sweep: T=1.714ms is degenerate (5/8
  models cannot clear even K=1); T>=25ms is nearly vacuous (schenck/gnn
  are the last to saturate at 128); T~=2.5ms is the most informative
  single budget (spreads N_i from ~3 to 128 across the field without every
  model collapsing to the same value).
  RUN-0003 (control comparison, see Amendment 2): scored slateN under
  N_i(T) budgets (T in {2.5,5,10}ms, with_value sweep), full-128-pool
  normalisation, corner shape, lyapunov+mass_in_region, n20_L20mm+
  n20_L40mm (n20_L10mm excluded). Nfd is the best budgeted control model
  at both trustworthy budgets (T=5,10ms), closely followed by the
  switched-linear/hybrid family; model0001_global (cheapest) trails by
  ~0.02-0.05 slateN once T>=5ms. No CNN/GNN model overtakes the linear
  family at any T tested (gnn's own T=10ms mean is still below
  model0001_global's T=2.5ms number). model0002_descriptor_only and both
  hybrids are INFEASIBLE at T=2.5ms once the value function's own cost is
  included. The 50-draw distribution shows the mandatory 5-draw headline
  is representative for N_i>=10 (schenck) but can be misleading (5-draw
  range exceeding the 50-draw's own [p5,p95]) for gnn's N_i in {1,8}.
  T=2.5ms results are flagged noise-floor and not used to rank beyond the
  hard INFEASIBLE flag.
verdict: supported
downgrades: [indirectness, imprecision, provenance, incomplete-design]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If every model's per-candidate cost were latency-bound (GPU launch
overhead dominates, batch size barely matters -- the failure mode the task
brief worried about), every model's end2end median at K=128 would sit
within noise of its K=1 median, and N_i would be >=128 for all 8, making an
equalised-wall-clock budget rank everyone on the full pool -- i.e. no bite.
Measuring the full curve per model and interpolating N_i against a shared
reference directly distinguishes "flat" from "has bite," per model, rather
than asserting it from a single batch size.

## What was actually run

- **Dirty tree at run time** (`provenance.dirty: true`): `git status`
  showed an unrelated, already-in-progress docs/skills reorganisation
  (renames/moves of `docs/experiments/*` -> `experiments/*`,
  `.claude/skills/_draft/*` -> `.claude/skills/*`, plus `.gitignore` and
  `experiments/COMMANDS.jsonl`) predating this session -- listed in full in
  `provenance.commit`'s companion `dirty_files` (see
  `utils.git_provenance()` output captured at run start). None of those
  files were read or written by `bench.py`, `make_budgets.py`, or any
  checkpoint/config this run loaded. This is a T1 record, so a dirty tree
  is not a hard error (that rule is T2-only) and is not counted as a
  downgrade domain here since nothing in the dirty diff overlaps this
  run's actual inputs/outputs.
- **Device-bug fix** (see task brief and `Baselines/common/benchmark_time.py`'s
  own module docstring, which first documented the trap): as invoked by
  `Baselines.common.eval_baseline`'s scorer, NFD/Schenck derive their
  device from `batch.occ0.device`, and that batch is never moved off CPU
  by `_predictor_batch` -- so both silently run on CPU while GNN forces
  cuda-if-available internally, an apples-to-oranges comparison.
  `bench.py::build_batch` fixes this by explicitly `.to(DEVICE)`-ing every
  tensor (`occ0`, `actions`, `p_start`, `p_stop`, `angle`) for every one of
  the 8 models before any forward call, and every model wrapper asserts
  (`assert_all_on`, plus a `next(model.parameters()).device` check for
  `nn.Module`-backed models) the ACTUAL device of what it fed forward,
  rather than assuming the `--device` flag was honoured. Verified: all 8
  rows report `device_actual: cuda`/`cuda:0` in `results_timing.json`.
- **Timing scope**: state rasterisation (particle->occupancy) is done ONCE,
  before any timed region. All 8 models' preprocessing (draw_plate_soft x2
  for NFD/Schenck, canonicalise/push-frame-warp for MODEL-0001/hybrids,
  push-frame descriptor computation for MODEL-0002/hybrids, and the GNN's
  `sample_nodes_xy`) is per-candidate and INSIDE the timed `end2end` region;
  `fwd_ms` isolates the forward call alone on the same preprocessed input.
- **GNN Python-loop caveat**: `sample_nodes_xy` loops per-sample in Python
  (`for i in range(B)`, `Baselines/GNN/predictor.py`). At K=128 this loop
  is ~40ms of GNN's ~44ms end2end cost (fwd-only is 3.9ms) -- i.e. >90% of
  GNN's measured wall-clock at large K is this implementation artifact, not
  the graph net. Reported as such (see `results/RESULTS.md`), with a cheap
  loop-excluded single-sample estimate (~0.31ms/sample) also recorded in
  `results_timing.json` (`gnn_one_sample_s_estimate`) so architecture and
  implementation are distinguishable.
- MODEL-0001's "switched" and "global/unswitched" readings both load the
  SAME checkpoint (`weights/MODEL-0001-stage2-visual-switched/checkpoint.pt`)
  -- `switched_ops` (6 length-bin operators) vs. `global_op` (one operator)
  -- exactly the operator-variant ablation the task named.
- hybrid14/hybrid94 operators were loaded from
  `experiments/temp/stage2-slaten/operators/{hybrid14,hybrid94}_lam1.0.pt`
  (still `experiments/temp/`, not promoted -- these are a DIFFERENT
  experiment's fitted objects, reused here read-only for timing; promoting
  THEM is out of scope for this record).
- No control/slateN evaluation was run. This record is timing only, per
  the task's explicit scope.

## Numbers

End-to-end (preprocessing+forward) median ms by K, fwd-only at K=128, and
derived N_i (candidates affordable in the fastest model's own K=128 time,
1.714ms):

| model | device | K=1 | K=8 | K=32 | K=64 | K=128 | fwd-only@128 | N_i |
|---|---|---|---|---|---|---|---|---|
| model0001_global | cuda | 1.25 | 1.22 | 1.21 | 1.31 | 1.71 | 1.24 | 128.0 (ref) |
| model0002_descriptor_only | cuda | 1.67 | 1.66 | 1.63 | 1.65 | 2.06 | 0.18 | 74.5 |
| nfd | cuda:0 | 1.12 | 1.12 | 1.32 | 1.57 | 4.03 | 3.62 | 69.6 |
| model0001_switched | cuda | 1.49 | 1.49 | 1.49 | 1.99 | 2.83 | 2.36 | 54.2 |
| schenck | cuda:0 | 0.77 | 1.89 | 6.36 | 20.71 | 46.27 | 48.67 | 7.0 |
| gnn | cuda:0 | 1.88 | 4.39 | 12.06 | 22.92 | 44.11 | 3.92 | <1 |
| hybrid14 | cuda | 2.77 | 2.67 | 2.72 | 2.89 | 3.64 | 1.57 | <1 |
| hybrid94 | cuda | 2.77 | 2.72 | 2.70 | 2.83 | 3.64 | 1.58 | <1 |

Full curve (all 11 K, IQR, pre/fwd/end2end separately) in
`artifacts/RUN-0001-timing-sweep/results_timing.json`. No baseline model
(`persistence`) was timed since persistence has no per-candidate cost (it
returns `occ0` unchanged) -- `design.baselines` names it as the trivial
zero-cost floor rather than a timed row; this is noted as a design choice,
not an omission.

## What would change the verdict

If a re-run on a genuinely idle GPU (this one WAS idle, contention gate
passed) showed schenck/gnn's steep scaling was actually queueing/thermal
noise rather than real compute, N_i for those two would rise and weaken
the "3 of 8 can't clear even 1 candidate" headline -- cheapest check: rerun
just those two models' K-sweep (~10s) on a freshly-idle card and diff
medians. Would not flip `verdict: supported` (model0001_global/nfd/etc.
already show non-flat behavior on their own), only change which models the
"< 1 candidate" claim applies to.

## Threats

- **indirectness** (the only downgrade claimed): `candidate_throughput_ms`
  measures wall-clock on ONE (state, K-cycled-actions) construction on ONE
  GPU under one contention state -- it is a proxy for "cost under the
  actual MPC deployment loop," which may batch/pipeline differently, run on
  different hardware, or hit different cudnn/cuBLAS algorithm selection at
  different batch sizes than this sweep exercised. The relative ordering
  (schenck/gnn/hybrids throughput-bound, model0001_global/desc-only/nfd
  closer to latency-bound) is expected to be robust to this; absolute N_i
  numbers are not guaranteed to transfer to different hardware.
- Considered and dismissed: **provenance** -- all 8 models were scored
  against the SAME state/action construction (`build_batch`), the SAME
  device-forcing fix, the SAME warmup/repeat/sync methodology; no model
  used a different code path for its own timing plumbing than another.
- Considered and dismissed: **selection** -- the reported table is the
  full 8x11 sweep, not a cherry-picked subset; `results_timing.json`
  carries every cell.
- Considered and dismissed: **inconsistency** -- only one seed/state was
  used (no per-seed variation was part of this design), so "does it hold
  across seeds" does not apply; this is a `held_fixed` design choice
  (single real state, matching `benchmark_time.py`'s own convention), not
  an inconsistency found across cells that were run.

## Amendment (RUN-0002, timing-corrected) -- what changed and why

A review of RUN-0001 found three defects that bias its numbers. RUN-0001's
numbers stand as measured (see "Numbers" above) -- they are correct for the
SCOPE they measured (sync-stalled switched-operator fwd, no value function,
no repeatability check). RUN-0002 is a superseding-in-scope, not
invalidating, re-run: same 8 models, same state, narrower K set
(`{1,32,128}` vs RUN-0001's 11-point curve, to keep RUN-0002 inside its own
time budget alongside the extra value-function/repeat-check work).

**Defect 1 -- GPU sync stalls in the switched-operator `fwd`.**
`bench.py`'s switched-operator loop (`model0001_switched`, `hybrid14`,
`hybrid94`) and the descriptor-only `fwd` looped over 6 length-bins and
guarded each masked assignment with `if bool(m.any()):` -- `bool(...)`
forces a GPU->CPU sync, 6x per call. First fix attempt: gather each
sample's own operator via `index_select` + a single batched `bmm` (fully
sync-free). Measured, then DISCARDED: it was sync-free but SLOWER at
K=128 than the original (e.g. model0001_switched's fwd went from 2.36ms to
6.02ms) because a per-sample `bmm` pays a full GEMM per sample instead of
one shared-operator GEMM per bin -- trading a host-sync cost for a larger
device-compute/memory-bandwidth cost. ACTUAL fix: keep the original
grouped-GEMM shape (cheap: one shared operator per bin) and simply DROP the
`bool(m.any())` guard -- `pred_c[m] = ...` with an all-False boolean mask is
already a legal no-op, so the guard bought nothing but a sync. Verified
numerically identical to the old loop (`verify_sync_free_equivalence` in
`bench.py`): max abs diff = 0.0 exactly (bit-identical; RES=32 canon
predictions), for all 4 affected models, at K=128.
Effect on timings: model0001_switched's K=128 end2end dropped from 2.83ms
(RUN-0001) to 2.17ms (RUN-0002) -- a real ~23% reduction, i.e. the sync
artifact was real and non-trivial for this model. hybrid14/hybrid94/
model0002_descriptor_only did NOT meaningfully change (RUN-0001 vs
RUN-0002 K=128 end2end: hybrid14 3.64->3.86ms, hybrid94 3.64->3.90ms,
desc-only 2.06->2.26ms -- all within/near the measured resolution floor,
see Defect 3 below). **So the sync artifact was concentrated in
model0001_switched, not spread evenly across every `.any()`-guarded
model** -- the task brief's prediction that it inflated "every
switched/descriptor row" was only partly right.

**Defect 2 -- the value function was never timed.** Added goal-value
computation (both `lyapunov` and `mass_in_region`, per
`experiments/temp/slaten-broad/eval_slaten_broad.py`'s own
`value_true_and_pred`) inside the timed region for every model, against a
fixed top-right-quadrant goal (timing-only, not accuracy-tuned). Image
models (nfd, schenck, model0001 switched/global, hybrid14/94) score their
own predicted occupancy directly. GNN produces particles, not an image --
its value closure rasterises `s_pred` back to occupancy via
`rasterize_nodes_as_cubes_batch` (already imported by its predictor
machinery, previously unused in this bench) and scores that -- a real,
now-honestly-timed cost (GNN's K=128 end2end+value jumps from 44.5ms to
55.3ms). Descriptor-only cannot produce an image at all -- it uses the
documented point-mass (monopole) approximation from
`eval_slaten_latent.py`: `V_lyapunov_hat = d(world_COM_hat)` (mass cancels
in the true `sum(occ*d)/sum(occ)` formula under a point-mass assumption)
and `V_mass_in_region_hat = mask(world_COM_hat) * mass_hat`, both via
`com_world_pixel`/`bilinear_sample`. Full pre/fwd/end2end-with-and-without-
value table: `artifacts/RUN-0002-timing-corrected/results_timing_v2.json`,
summarised in `results/RESULTS.md`.
**Does descriptor-only become fastest once the value function is timed?
NO -- the opposite.** Its point-mass readout costs ~0.6ms against its own
~0.03ms fwd (a >20x relative hit), while model0001_global's value cost is
two cheap reductions over an already-computed 32x32-ish image (~0.06ms
against a ~0.9-1.2ms fwd, <10% relative hit). Descriptor-only's N_i at
T=2.5ms falls from 128 (no value) to unresolvable (<1, with value) --
model0001_global remains the most robust model once scoring is included.

**Defect 3 -- precision was never characterised.** RUN-0001 reported
schenck's fwd_ms@128=48.67 exceeding its own end2end_ms@128=46.27 --
impossible if both numbers were noise-free (end2end is pre+fwd, so
end2end >= fwd always, in expectation). RUN-0002 re-ran ONE fixed config
(schenck @ K=128, 15 reps) in 3 SEPARATE process invocations
(`code/repeat_check.py`): end2end medians 46.09 / 47.84 / 48.11 ms --
a ~2.0ms / ~4.3% between-run spread, i.e. RUN-0001's 2.4ms/5% "impossible"
inversion is FULLY inside ordinary between-run measurement noise, not a
computation bug. A second check (model0001_global @ K=128, fwd-only) gave
2.433/2.440ms across 2 runs (~0.3% spread) but a markedly DIFFERENT
absolute value than the same logical quantity measured inside the main
sweep's own process (~1.75ms end2end there) -- cross-process absolute
drift can be much larger than within-process repeat noise for models whose
true cost is only a few ms. **Resolution floor applied**: a difference
smaller than ~5% relative (or ~2ms absolute at the ~45ms scale) between
numbers from different timed regions or different process invocations is
NOT resolvable and must not be used to rank models. `repeats` was not
raised (15 was already used); the fix here was measuring between-RUN
spread, which `repeats` alone (within one run) cannot expose.

**Budget re-derivation as a sweep** (`results/budgets.json`, RUN-0002):
T in {1.714ms (original), 2.5, 5, 10, 25, 50}, both without and with the
value function, capped at 128. T=1.714ms is degenerate (5 of 8 models
cannot clear even K=1). T=25/50ms is nearly/fully vacuous (everyone
saturates near/at 128; schenck and gnn are the last two to saturate).
**T~=2.5ms is the most informative single budget**: it spreads N_i from
~3 (hybrid14/94, below-floor at this T) up to 128 (model0001_switched/
global, model0002_descriptor_only) without collapsing the whole field to
one value -- the original run's T=1.714ms reference was, per the task's
own diagnosis, below several models' fixed overhead and therefore harsher
and less informative than it needed to be.

## Unrelated findings

- `Baselines/common/eval_baseline.py::_predictor_batch` (the actual scorer
  used by other experiments in this repo, not just this timing harness)
  still has the device bug described above -- it builds `PredictorBatch`
  from CPU tensors and never moves them to CUDA. This bench's own
  `build_batch` works around it locally but does NOT fix
  `eval_baseline.py` itself; any existing `accuracy`/`slateN` numbers for
  NFD/Schenck produced via that scorer were very likely computed on CPU
  (functionally fine for correctness, since both derive dtype/shape from
  the input regardless of device, but worth flagging if anyone later times
  `eval_baseline.py` runs themselves rather than using this dedicated
  bench).
- `experiments/temp/stage2-slaten/operators/{hybrid14,hybrid94}_lam1.0.pt`
  are read-only inputs from a different (uncited-here) experiment's fitted
  objects, still living in `experiments/temp/` rather than `weights/`; not
  promoted as part of this record since promoting them is a different
  experiment's business.

## Amendment 2 (RUN-0003, budgeted-control) -- the control half

RUN-0001/RUN-0002 measured TIME only. RUN-0003 measures slateN CONTROL
performance under the derived `N_i(T)` budgets, closing the loop this
record's title promised ("the gate for the larger control-comparison
experiment").

**What ran**: `code/eval_budgeted_control.py` reuses `bench.py`'s own
per-model `pre()`/`fwd()` wrappers (same device-bug-fixed code RUN-0001/
RUN-0002's timing numbers came from) to get one prediction per candidate
over the FULL 128-candidate, same-state pools from
`experiments/temp/slaten-broad/eval_slaten_broad.py`'s multistep datasets
(`n20_L20mm`+`n20_L40mm`, `n20_L10mm` excluded, 20 slates x 128 rows each,
verified). Every model's raw prediction is then routed through
`eval_slaten_broad.py`'s OWN `build_goal`/`value_true_and_pred`/
`desc_pointmass_value` -- i.e. downstream of "model produced a prediction,"
there is exactly ONE scoring path shared by the linear family and by
NFD/GNN/Schenck, not two independently-written scorers. `budgeted_slate_
capture()` (new, in the eval script) is `Baselines/common/goals.py::
slate_n_capture`'s formula with one change: the model only RANKS within a
without-replacement subset of size `N_i(T)`, but oracle-best/random-floor
normalisation is ALWAYS computed over the full 128 -- the user's explicit
design choice, so a budget-limited model is penalised for possibly missing
the true best action, not just scored on an easier smaller pool.

**Result** (full numbers: `results/RESULTS.md`'s RUN-0003 section,
`artifacts/RUN-0003-budgeted-control/results_control.json`): nfd is the
best control model at both trustworthy budgets (T=5,10ms); the switched-
linear/hybrid family is close behind; model0001_global (RUN-0002's
cheapest/most-budget-robust model) trails by ~0.02-0.05 slateN once T>=5ms
-- i.e. being cheap does not make it the BEST controller once the budget is
loose enough for everyone else to also reach N_i=128. No CNN/GNN model
overtakes the linear family at any T tested (gnn's own T=10ms mean, 0.832
pooled lyapunov, is still below model0001_global's T=2.5ms number, 0.952).
`model0002_descriptor_only`/`hybrid14`/`hybrid94` are INFEASIBLE at T=2.5ms
(their value-inclusive per-candidate cost already exceeds the budget for
even 1 candidate) -- a hard `null` in `budgets.json`, not a close call.

**Why this downgrades the record further** (frontmatter `downgrades` grew
from `[indirectness, imprecision]` to `[indirectness, imprecision,
provenance, incomplete-design]`, grade `low` -> `very-low`):
- `provenance`: the per-candidate PREDICTION step is still model-specific
  code (NFD/Schenck/GNN predictor classes vs. the linear family's
  switched/global operators) -- unification stops at "produced an
  occupancy/descriptor prediction," not at the forward pass itself. This
  was flagged as an acceptable downgrade in the task brief rather than a
  blocker.
- `incomplete-design`: EXP-0008's own design called for 5 goal shapes x 3
  value functions; RUN-0003 ran 1 shape (corner) x 2 value functions
  (lyapunov, mass_in_region) under the 60-minute budget. This is a real
  incomplete-design gap, not hidden in a footnote -- see `RESULTS.md` and
  the eval script's own docstring for the explicit scope-reduction list.
- The existing `indirectness`/`imprecision` downgrades from RUN-0001/
  RUN-0002 still apply unchanged (candidate_throughput_ms is still a proxy,
  and T=2.5ms is still inside the measured noise floor -- RUN-0003's own
  T=2.5ms cells are flagged and not used to rank beyond the hard
  INFEASIBLE signal).

**50-draw vs 5-draw**: representative for schenck (N_i>=10); for gnn
(N_i in {1,8}) the 5-draw top/bottom occasionally exceeds the 50-draw's
own [p5,p95] band (2 of ~12 checked cells) -- gnn-vs-schenck at T=5ms
should be read as noisy given n=20 slates/dataset and gnn's low N_i, not
as a settled ranking. This is a `imprecision`-flavoured caveat on that one
comparison specifically, already covered by the record-level downgrade.

**What was NOT re-litigated**: RUN-0001/RUN-0002's timing numbers and
`N_i(T)` derivations are taken as given inputs here, not re-measured or
re-derived.
