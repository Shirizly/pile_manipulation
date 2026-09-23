# RUN-0009 -- main-run eval: warped NFD (RUN-0005) vs. world-frame NFD baseline, 3 corpora

Scores `nfd_randlen` (world-frame baseline, `Baselines/NFD/runs/nfd_3ch_randlen/
unet_best.pth`, EXP-0001's checkpoint) and `nfd_warped_randlen` (RUN-0005,
`Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth`) through the SAME
`Baselines/common/eval_report.py` harness EXP-0001 used, on the same 3 corpora
(L20mm eval cell, L40mm eval cell, `overnight_randlen_test`). Both models are scored
**in the same process, side by side** so the comparison is sound regardless of
EXP-0001's own dirty-tree/no-commit provenance gap.

## Code changes made for this run
- `Baselines/common/eval_report.py`: added one `MODELS` entry, `nfd_warped_randlen`,
  pointing at `Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth` via
  `build_predictor_warped`, with explicit `kwargs=dict(canon_res=None,
  plate_mode="canonical", scale=1.0)` -- these match the checkpoint's own
  training-time defaults per `model_card.yaml`, and `_assert_matches_model_card`
  checks this on every load (same pattern as the existing L20mm-pilot entries). No
  other `MODELS`/`CORPORA` entries were touched or removed.
- New code: `experiments/EXP-0022-warped-nfd-push-frame/code/
  warp_accuracy_ceiling_multi.py` (extends `warp_accuracy_ceiling.py`, L20mm-only, to
  also cover L40mm and randlen_test) and `.../ranking_robustness_check_n40.py`
  (extends `ranking_robustness_check.py`'s 3-pool L20mm-only check to 42 pools across
  all 3 corpora).

## Exact argv
See `COMMAND.txt` (three commands: the main harness, the multi-corpus ceiling script,
the N=40+ ranking-robustness script -- all read-only against existing checkpoints, run
sequentially in the same session).

## Timing
- Start: 2026-09-22 ~11:03 UTC (GPU idle beforehand: 15 MiB / 0%, `uptime` load
  average ~1.6-3.3 from unrelated processes).
- Main harness (`eval_report.py`): ~189 s total (corpus loads: L20mm 26.9s cached[^1],
  L40mm 26.9s, randlen_test 48.0s; each model ~8-15s per corpus).
- `warp_accuracy_ceiling_multi.py`: ~104 s (3 corpus loads + round-trip scoring).
- `ranking_robustness_check_n40.py`: ~97 s.
- Exit status: 0 for all three.

[^1]: `load_cell` for L20mm/L40mm each report their own ~27-49s load time in the
  script's own stdout; not deduplicated across the 3 scripts run in this session
  (each script re-loads its own cells independently).

## Anomalies
- **Device**: both models ran on **CPU** for every corpus, confirmed via
  `PredictorBatch.occ0.device` (the harness prints this per model). This is the
  documented `eval_baseline.py::_predictor_batch` trap (never moves the batch off
  GPU). GPU was idle and unused throughout. Does not bias the comparison (both models
  pay the same cost), only wall-clock, which was cheap enough (single-digit to
  low-teens seconds per model per corpus) that it did not matter here.
- **Baseline reproduction**: re-scoring `nfd_randlen` in this run reproduced EXP-0001's
  published accuracy (0.4071/0.5088/0.4564) to 4-5 decimal places in all 3 corpora --
  see `results/randlen_eval.md`'s "Baseline reproduction check" section. No
  discrepancy found, so no `OPEN_ISSUES.md` entry was added on this point.
- No other anomalies.

## Outputs
- Raw JSON: `../../artifacts/RUN-0009-randlen-eval/randlen_eval.json`
- Logs: `stdout.log` (main harness), `warp_ceiling_multi.log`,
  `ranking_robustness_n40.log`
- Analysis: `../../results/randlen_eval.md`
