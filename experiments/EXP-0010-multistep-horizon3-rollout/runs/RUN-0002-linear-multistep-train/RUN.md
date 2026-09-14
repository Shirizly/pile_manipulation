# RUN-0002-linear-multistep-train

Gradient-trains the linear pixel-space operator (both `switched`, 6
push-length bins, and `global`, 1 operator) through the closed-loop 3-step
rollout objective `L = lam*L1 + lam^2*L2 + lam^3*L3` (swept-region-masked
per-step MSE), lambda in {0.3, 0.5, 0.7, 0.9}, initialised from
`MODEL-0001`'s closed-form ridge-toward-identity solution. 150 Adam
iterations x 4 lambdas x 2 operator kinds = 8 training runs, full-batch,
lr=2e-3. **Negative result**: held-out terminal slateN collapses from
+0.937/+0.789 (init) to negative at every lambda; spectral radius blows up
3.5-4.7x in the only bins that receive gradient (both corpora are
single-push-length, so 4 of 6 switched bins get zero training rows).

- **Command**: see `COMMAND.txt` (reconstructed — predates
  `run_probe.py`-style logging; original invocation
  `python -u train_multistep.py` from `experiments/temp/multistep-train/`,
  followed by a separate `python recompute_wlt.py` post-hoc bugfix pass
  that re-scores the saved operators without retraining).
- **Commit**: `0ddab20f` (dirty tree — same pre-existing, unrelated
  docs/skills reorganisation as RUN-0001).
- **Device**: single GPU run, 765s wall clock (`run.log`).
- **Data / split**: raw `_{batch}_data.pt` files (same loader as
  RUN-0001), `n20_L20mm` + `n20_L40mm` (`n20_L10mm` excluded). Split by
  `slate_idx`, seed 0, 35 train / 15 test, the SAME split index set
  applied to both datasets; disjointness asserted in-script
  (`set(train) & set(test) == {}`), passes.
- **Status**: completed, no errors. A post-hoc bugfix pass
  (`recompute_wlt.py`) was run afterward to fix a sign error in the
  win/loss/tie table for the lyapunov (cost) metric — it re-scores the
  already-saved operators, no retraining.
- **Outputs**: `artifacts/RUN-0002-linear-multistep-train/
  {results_multistep_train.json,wlt_vs_init_fixed.json,run.log}` (raw),
  8 fitted operator checkpoints (`operator_{switched,global}_lam{0.3,0.5,
  0.7,0.9}.pt`, kept in `experiments/temp/multistep-train/` per the
  experiment-log skill's "keep every fitted object" rule — not promoted to
  `weights/` since this is a refuted, unstable result, not a reusable
  instance), `results/RUN-0002-linear-multistep-train-RESULTS.md` (curated,
  copied verbatim from the source `experiments/temp/multistep-train/
  RESULTS.md`).
