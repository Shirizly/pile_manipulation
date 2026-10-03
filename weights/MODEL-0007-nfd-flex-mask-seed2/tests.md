# MODEL-0007 test history

- DS-0020 val (1,991 rows, binary image-mask truth, swept-region `accuracy`, plate 10.67 px):
  **0.6098** (persistence 0) -> EXP-0061 -> `experiments/EXP-0061-flex-cross-corpus-rerun/results/nfd_val_accuracy.json`
- DS-0019 (100 same-state slates, 16,583 rows, binary image-mask truth, default goal set), sanity
  pass by the training agent, NOT the final eval: slateN averaged over goals lyapunov 0.953 /
  mass_in_region 0.878 / signed_mass 0.904; accuracy 0.4842
  -> `experiments/EXP-0061-flex-cross-corpus-rerun/results/nfd_seed2_ds0019.{json,md}`
- **final DS-0019 test eval** (binary image-mask truth, 100 slates, per-slate + per-row outputs, paired tests, strata) -> EXP-0061 / RUN-0001 -> `experiments/EXP-0061-flex-cross-corpus-rerun/results/final_eval/` (row `nfd_s2`)
