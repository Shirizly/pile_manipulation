# MODEL-0004 test history

- DS-0020 val (1,991 rows, image-mask truth, swept-region `accuracy`, plate 2.4 units = 10.67 px):
  switched 0.482 (lambda 300), single 0.354 (lambda 1000), persistence 0 -- lambda was SELECTED on
  this split, so these are mildly optimistic (curve flat: switched 0.475-0.482 over lambda 100-1000)
  -> EXP-0061 -> `experiments/EXP-0061-flex-cross-corpus-rerun/RUNS_gnn_lf.md`, `fit.json`
- DS-0019 (test, slateN + accuracy): NOT YET RUN (later EXP-0061 agent; `eval_report.py
  --corpora flex_ds0019_mask --models lf_flex_switched,lf_flex_single`)
- **final DS-0019 test eval** (lf_flex_switched / lf_flex_single, binary image-mask truth, slateN + accuracy + strata) -> EXP-0061 / RUN-0001 -> `experiments/EXP-0061-flex-cross-corpus-rerun/results/final_eval/`
- DS-0019 per-slate reuse check (re-scored, max |per-slate diff| 0, max |row rms diff| 0) + paired slateN vs v2 refit MODEL-0009 -> EXP-0062 / RUN-0002 -> `experiments/EXP-0062-*/results/lf_ds0019.{json,md}`
