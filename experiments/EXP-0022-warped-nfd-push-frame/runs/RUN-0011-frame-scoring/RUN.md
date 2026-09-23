# RUN-0011 -- A1 (canonical-frame scoring) + A2 (warped-goal control scoring)

`PLAN.md`'s Phase A items A1 and A2, both scoring-only (no training), run
concurrently with RUN-0010 (GPU) on CPU as instructed.

## A1: canonical-frame accuracy, additive

- `Baselines/common/eval_report.py` gained `--canonical-frame` (off by
  default) and `_accuracy_canonical`. It reports `accuracy_canonical`
  alongside the unchanged `accuracy` for every non-GNN model.
- `Baselines/NFD/predictor.py::WarpedNFDPredictor` gained
  `predict_occ_canonical`, sharing pixel/`fn` construction with the existing
  `predict_occ` via a new `_build_fn` helper -- so the two paths cannot drift
  apart, matching this module's existing parity discipline.
- Scored: `nfd_randlen`, `nfd_warped_randlen` (RUN-0005), and all 4
  LinearForesight entries already in `MODELS`
  (`linear_switched_res64`/`linear_single_res64`/`linear_switched_res32`/
  `linear_single_res32`) -- all present, none skipped for A1 except GNN
  (out of scope, see below).
- Corpora: L20mm eval cell, L40mm eval cell, `overnight_randlen_test` --
  the same 3 EXP-0001/RUN-0009 used, loaded through `load_cell`/
  `load_randlen_cell` with their manifests (non-trivial `slate_idx`/
  `step_idx`).
- GNN models (`gnn_l20l40`, `gnn_randlen_n30`) **skipped**: their accuracy
  path already routes through a node-count resampling bottleneck
  (`resample_occupancy_through_nodes`), and composing that with a SECOND,
  per-model canonical frame was judged not worth the budget here --
  `eval_report.py --canonical-frame` prints an explicit SKIPPED line for any
  `is_gnn=True` model rather than silently omitting it.

**World-frame `accuracy` verified unchanged**: `nfd_randlen` reproduces
RUN-0009/EXP-0001's published 0.4071/0.5088/0.4564 to 4-5 decimals in this
run, with `--canonical-frame` turned ON -- confirming the new code path is
additive, not a modification of the existing one.

**Headline finding: canonical-frame scoring does NOT reorder the models.**
The ranking by `accuracy` is identical in both frames, in all 3 corpora, for
all 6 models scored. See `results/frame_scoring.md` for the full table and
the asymmetry discussion (a native-canonical model pays zero extra
resamplings in the canonical column; every other model pays one).

## A2: warped-goal ranking-robustness check

New: `code/ranking_robustness_check_warped_goal_n40.py`, mirroring
`code/ranking_robustness_check_n40.py`'s 42-pool design (14 pools x 3
corpora, `lyapunov` to a per-slate random-quadrant goal, canon_res=64) but
warping the GOAL into each candidate's OWN canonical push frame instead of
warping the state through a shared round-trip.

**Result: REFUTED, decisively.** Top-1 flip rate 20/42 = 47.6%, vs. the
world-frame baseline's 2/42 = 4.76% (RUN-0009) -- an order of magnitude
worse, and dominated by `randlen_test` (13/14 = 92.9% flips), the corpus
whose pools mix the widest range of push lengths/directions per slate. See
`results/frame_scoring.md` for the full breakdown and the mechanism.

## Anomalies

- **`Baselines/LinearForesight/runs/operators_res64.pt` missing from disk**
  (never git-tracked, only its `_accuracy.json` sibling was committed).
  Worked around by copying in a same-shaped file found at
  `experiments/temp/res32-vs-64/strays/operators_res64.pt`
  (`res=64`, matching `train_cfg`, all expected keys) -- NOT independently
  re-derived/re-fit. `OPEN_ISSUES.md` entry added; owner should confirm or
  regenerate via `fit_switched.py`.
- Device: all models ran on CPU (`PredictorBatch.occ0.device`, printed per
  model by the harness) -- the known `eval_baseline.py::_predictor_batch`
  trap. GPU was left to RUN-0010 throughout; no CUDA calls were made by this
  run's processes.
- No training was started; RUN-0010 was left running undisturbed
  (confirmed via `nvidia-smi`/`ps aux` before, during, and after).

## Timing

Two `eval_report.py` invocations (L20mm alone, then L40mm+randlen_test) plus
the warped-goal script, run concurrently with RUN-0010 (GPU training) and
RUN-0012 (a concurrent A3 subagent, CPU) -- heavy CPU contention (load
average ~9-10 on a 20-core machine) slowed wall-clock but did not affect
correctness. Total wall-clock for this run's 3 processes: ~12 minutes.

## Outputs

- Raw JSON: `../../artifacts/RUN-0011-frame-scoring/frame_scoring_merged.json`
  (+ the two un-merged halves, + logs).
- Analysis: `../../results/frame_scoring.md`.
- Code: `../../code/ranking_robustness_check_warped_goal_n40.py`.
