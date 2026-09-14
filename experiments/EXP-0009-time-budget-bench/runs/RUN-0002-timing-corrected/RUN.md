# RUN-0002-timing-corrected

Fixes 3 defects found in RUN-0001 (see `EXPERIMENT.md`'s "Amendment"
section) and re-times all 8 model configurations at a reduced K set
(`{1,32,128}`, vs RUN-0001's 11-point curve -- kept narrow to fit this
run's own time budget alongside the new value-function and
repeatability work).

- Command: see `COMMAND.txt`.
- Commit: `0ddab20f` (dirty -- same pre-existing docs/skills reorg as
  RUN-0001, unrelated to this run; no file this run reads or writes is
  among the dirty files).
- Device: cuda:0 (RTX 4070 Laptop, 8GB), idle at start, contention gate
  passed, not `--force`d.
- Dataset: same as RUN-0001 (`configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml`,
  row 0 + K cycled real actions).
- What ran, in order:
  1. `code/bench.py` -- loads all 8 models, runs
     `verify_sync_free_equivalence` (defect 1 check: old `.any()`-guarded
     loop vs the sync-free fix, on the SAME `pre_out`, at K=128) for the
     4 affected models, then times pre/fwd/end2end AND
     fwd_with_value/end2end_with_value (defect 2) at K in {1,32,128},
     15 repeats, >=5 warmup, per (model,K) cell.
  2. `code/repeat_check.py schenck 128` and
     `code/repeat_check.py model0001_global 128`, each invoked 3x (schenck)
     / 2x (model0001_global, time-constrained) as SEPARATE `python -u`
     processes -- defect 3's between-run repeatability check.
  3. `code/make_budgets_v2.py` -- N_i(T) sweep over
     T in {1.714 (RUN-0001's original reference), 2.5, 5, 10, 25, 50} ms,
     both without and with the value function, capped at 128; writes
     `results/budgets.json` (overwriting RUN-0001's single-T budgets.json,
     which is preserved verbatim inside the new file under
     `run0001_reference_budgets`).
- Total wall clock: `bench.py` itself ~37s; `repeat_check.py` x5 invocations
  (model loading dominates each) ~2-3 min; `make_budgets_v2.py` <1s.
- Status: completed, no errors, no `--force` needed.
- Outputs: `artifacts/RUN-0002-timing-corrected/results_timing_v2.json`
  (includes `sync_free_equivalence_max_abs_diff` per model, and
  `end2end_with_value_ms`/`fwd_with_value_ms` alongside the original
  `end2end_ms`/`fwd_ms`/`pre_ms`), `results/{RESULTS.md,budgets.json}`
  (both updated in place with a RUN-0002 section/sweep, RUN-0001's
  numbers kept alongside, not overwritten), `EXPERIMENT.md`'s new
  "Amendment" section.

## Numerical-equivalence check (defect 1)

Max abs diff, old `.any()`-guarded loop vs sync-free fix, same `pre_out`,
K=128 (see `artifacts/RUN-0002-timing-corrected/results_timing_v2.json`
`sync_free_equivalence_max_abs_diff`):

| model | max abs diff |
|---|---|
| model0001_switched | 0.0 |
| model0002_descriptor_only | 0.0 |
| hybrid14 | 0.0 |
| hybrid94 | 0.0 |

Exactly 0 -- the fix (drop `bool(m.any())`, keep the grouped-GEMM shape)
is bit-identical to the original, not merely close.

## Repeatability check (defect 3)

3 separate process invocations, schenck @ K=128, end2end median:
46.09 / 47.84 / 48.11 ms (spread ~2.0ms, ~4.3% of the mean).
2 separate process invocations, model0001_global @ K=128, fwd-only median:
2.433 / 2.440 ms (spread ~0.3%) -- but ~2.43ms here vs ~1.75ms end2end for
the same model measured inside `bench.py`'s own process, i.e. cross-process
absolute drift exceeds within-process repeat noise for small-cost models.
