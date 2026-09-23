# RUN-0012 -- A3 wall-proximity stratification (PLAN.md gate on the wall-channel run)

Compares the world-frame baseline (`nfd_randlen`, `Baselines/NFD/runs/
nfd_3ch_randlen/unet_best.pth`) against the warped NFD (RUN-0005,
`Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth`) on the SAME transitions,
per-transition swept-region `accuracy`, stratified by a wall-proximity
statistic (see `results/wall_proximity.md` for the definition, tables and
verdict). No training. Both models scored on CPU throughout (asserted
in-script; GPU was left untouched for the concurrent training job).

## Code
New: `code/wall_proximity_stratified_eval.py`. Reuses
`Baselines.common.randlen_data.load_randlen_cell`, `Baselines.common.data.
load_cell`, `Baselines.NFD.predictor.NFDPredictor`/`WarpedNFDPredictor`,
`fit_linear_foresight.actions_to_pixels`/`swept_region_mask`. Does not touch
`Baselines/common/eval_report.py` or any scoring path another agent might be
editing concurrently.

## Exact argv
See `COMMAND.txt` for the first (randlen_test) invocation. The L20mm/L40mm
follow-up used the same script with `--corpora L20mm,L40mm --out-prefix
.../wall_proximity_L20L40`.

## Timing
- randlen_test: corpus load 110.8s, baseline predict 15.6s, warped predict
  17.8s -- total ~146s.
- L20mm+L40mm: ~95s+~75s -- total ~170s.
- Both runs on CPU; GPU was running an unrelated training job throughout and
  was never touched by this script (no `.cuda()` call anywhere in it).

## Anomalies
- **A metric bug was found and fixed mid-run** (see `results/wall_proximity.md`
  "Metric note"): a first per-row `accuracy` implementation divided per-row
  numerator by per-row denominator before aggregating, which blows up whenever
  a single row's persistence error is near zero. Produced accuracy values in
  the millions on the first randlen_test pass. Fixed by keeping per-row RMS
  errors un-ratioed and only forming the ratio at the stratum-aggregation
  step (matching `fit_linear_foresight.metrics`'s own order of operations),
  with bootstrap CIs instead of a naive per-row-ratio SEM. The corrected
  aggregate baseline accuracy (0.4564) reproduces RUN-0009's published number
  exactly, which is the check that the fix is right.
- L40mm's push-length confound-check table hit one empty quantile bin (length
  is a near-delta distribution in that cell, so two quantile edges collided)
  -- handled by falling back to the most-populated length band for the
  within-band wall-distance check, not silently skipped.
- Corpus-dependent result, reported plainly rather than smoothed over: the
  wall-proximity effect is clear and monotonic (surviving a push-length
  control) in L20mm/L40mm, but goes flat/non-monotonic once push length is
  controlled for in `overnight_randlen_test`, the corpus the decision should
  be based on. See `results/wall_proximity.md`'s Interpretation/Verdict.

## Outputs
- Raw per-transition JSON: `../../artifacts/RUN-0012-wall-proximity/
  wall_proximity.json` (randlen_test), `wall_proximity_L20L40.json` (L20mm+L40mm).
- Logs: `stdout.log` (randlen_test), `stdout_L20L40.log` (L20mm+L40mm). The
  empty-bin traceback mentioned above happened on an earlier invocation of
  this same log path (overwritten by the corrected re-run); not preserved on
  disk, reported here from the terminal transcript instead.
- Analysis: `../../results/wall_proximity.md`.
