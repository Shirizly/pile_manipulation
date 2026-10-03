# MODEL-0005 test history

- DS-0020 val (1,991 rows, binary image-mask truth, swept-region `accuracy`, plate 10.67 px):
  **0.6095** (persistence 0) -> EXP-0061 -> `experiments/EXP-0061-flex-cross-corpus-rerun/results/nfd_val_accuracy.json`
- DS-0019 (100 same-state slates, 16,583 rows, binary image-mask truth, default goal set), sanity
  pass by the training agent, NOT the final eval: slateN averaged over goals lyapunov 0.948 /
  mass_in_region 0.878 / signed_mass 0.896; accuracy 0.4895
  -> `experiments/EXP-0061-flex-cross-corpus-rerun/results/nfd_seed0_ds0019.{json,md}`
- **final DS-0019 test eval** (binary image-mask truth, 100 slates, per-slate + per-row outputs, paired tests, strata) -> EXP-0061 / RUN-0001 -> `experiments/EXP-0061-flex-cross-corpus-rerun/results/final_eval/` (row `nfd_s0`)
- DS-0020 **v2** val accuracy (1,665 rows, binary mask truth): 0.4290; DS-0019 re-scored as the v1 reference for the v2-trained MODEL-0008 (bit-identical to EXP-0061) -> EXP-0062 / RUN-0001 -> `experiments/EXP-0062-flex-v2-train-rerun/results/nfd_seed0_ds0019.json`
