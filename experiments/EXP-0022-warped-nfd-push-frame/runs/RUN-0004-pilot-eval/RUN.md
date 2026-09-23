# RUN-0004 -- pilot eval: score all three L20mm pilot arms with `Baselines/common/eval_report.py`

Scores RUN-0001 (unwarped control), RUN-0002 (warped), RUN-0003 (warped+walls) with the SAME
harness (`Baselines/common/eval_report.py`) that produced EXP-0001's NFD numbers, restricted to
the held-out `L20mm` eval cell (`configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml`,
manifest `Genesis/data/slates_multistep/n20_L20mm/manifest.json`).

## Code changes made for this run
- `Baselines/NFD/predictor.py`: `build_predictor_warped`/`build_predictor_warped_walls` now
  accept `canon_res`/`plate_mode`/`scale` (previously hardcoded) and assert them against the
  `model_card.yaml` written next to the checkpoint by `train_nfd.py`
  (`_assert_matches_model_card`) -- so a silently mismatched setting at eval time raises rather
  than producing a wrong-but-plausible number.
- `Baselines/common/eval_report.py`:
  - Added three `MODELS` entries: `nfd_unwarped_L20mm_pilot`, `nfd_warped_L20mm_pilot`,
    `nfd_warped_walls_L20mm_pilot`, pointing at the three pilot checkpoints, with explicit
    `kwargs=dict(canon_res=None, plate_mode="canonical", scale=1.0)` for the warped two (these
    are also the checkpoints' own training-time defaults, per each `model_card.yaml`; the new
    assertion checks this on every load).
  - `_load_predictor` now forwards `spec.get("kwargs", {})` to the factory.
  - Added `persistence` and `random` reference rows, computed once per corpus (not per model):
    `persistence` reuses `cell.occ0` directly (accuracy = 0 by construction; its slateN row is
    flagged DEGENERATE since it predicts dv=0 for every candidate). `random` is estimated by
    feeding i.i.d. noise images through the existing `_capture_report` machinery so its own
    argmax reduces to a uniform-random pick (averaged over 20 seeds) -- a Monte-Carlo check of
    the exact analytic fact that `E[slate_n_capture(random pick)] = 0`.
  - Added `device` introspection/printing per model (the `eval_baseline.py::_predictor_batch`
    CPU trap noted in the task brief) -- `PredictorBatch.occ0.device` is what the predictor's
    own `predict_occ` forward actually runs on.
  - No entries removed from `MODELS`/`CORPORA`.

## Verifying the mismatch-assertion actually fires
Not separately tested with a deliberately wrong setting (budget); the assertion did not raise
on the real run, which only shows it did not trip on the (correct) values used -- see
`results/pilot_eval.md` for the caveat.

## Exact argv
See `COMMAND.txt`.

## Timing
- Start: 2026-09-22 08:11:20 UTC (GPU idle beforehand, confirmed with `nvidia-smi`: 15 MiB / 0%)
- End: 2026-09-22 08:13 UTC
- `eval_report.py` main run: ~53s total (26.9s corpus load + ~8-11s per model)
- `ranking_robustness_check.py`: a few seconds
- Exit status: 0 for both

## Anomalies
- All three models (and the reference rows) ran on **CPU**, not GPU -- confirmed via
  `PredictorBatch.occ0.device`. This is the documented `eval_baseline.py::_predictor_batch`
  trap: it never moves the batch off CPU. GPU was idle throughout and not used. This does not
  affect the numbers, only wall-clock (a training run this size on CPU vs GPU would matter more,
  but eval over 7680 transitions in single-digit seconds per model was cheap enough that no one
  noticed).
- No other anomalies. GPU/CPU load otherwise idle (`uptime` load average ~2.9-3.3 from other,
  unrelated processes on the box, not this job).

## Outputs
- Raw JSON: `../../artifacts/RUN-0004-pilot-eval/pilot_eval.json`
- stdout: `stdout.log` (main harness), `ranking_robustness.log` (the warp-cancels-in-ranking
  check)
- Analysis: `../../results/pilot_eval.md`
