# RUN-0002 — A5: time-matched slateN ranking (DESIGN.md addenda 3-4)

- Code: code/budget_matched.py; commit 3bae8cd7, dirty (uncommitted working tree).
- Data: DS-0006 (160 states x 128 pushes), A3's cached predictions and soft truth
  (artifacts/RUN-0001/pred_*.pt, truth.pt). No new predictions.
- Attempt 1 (2026-09-24 12:25, 25 s, tag exp0030_budget_matched): per-call timing;
  degenerate (addendum 4). Outputs kept: artifacts/RUN-0002/timings_percall_run1.json,
  results/budget_matched_run1_degenerate.json.
- Attempt 2 (12:26, 30 s, tag exp0030_budget_matched_v2): throughput timing, the
  reported run. Outputs: artifacts/RUN-0002/timings.json, results/budget_matched.json.
- Timing was on the shared GPU while EXP-0036 seed training ran (relative costs are
  what is used; absolute throughput is contention-affected).
- Command: `PYTHONPATH=. python scripts/run_probe.py --tag exp0030_budget_matched_v2
  --exp EXP-0030 --out-dir experiments/EXP-0030-state-superiority-ds0005/runs/RUN-0002-budget-matched
  --threads 4 --timeout 1400 -- python experiments/EXP-0030-state-superiority-ds0005/code/budget_matched.py`
