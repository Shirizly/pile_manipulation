# RUN-0002 — A1: paired re-analysis of EXP-0023

- **Commit:** 3bae8cd7, dirty (uncommitted: this session's edits to `Baselines/common/eval_report.py` (per-slate output, additive), `Baselines/common/goals.py`, `simple_mpc/adapters.py`, new `Baselines/common/paired_stats.py`, `simple_mpc/gt_bank.py`; plus pre-existing user edits to eval_report.py's MODELS dict and unrelated EXP-0022 logs). The exact run-time git state is in the `.json` meta beside the log.
- **Python:** /home/alon/anaconda3/envs/pme/bin/python via `scripts/run_probe.py` (python -u, 4 threads). Ledger entries: `runs/COMMANDS.jsonl` (run_probe's actual path, invariant run-probe-ledger-path-matches-doc is broken), copied into experiments/COMMANDS.jsonl.
- **Command:** `experiments/EXP-0026-benchmark-power/code/a1_exp0023_paired.py --out experiments/EXP-0026-benchmark-power/results/a1_exp0023_paired.json`
- **Input:** EXP-0023 results/metrics.json (60 rows). No simulation. Seconds.
- **Check:** two-way residual sd reproduces EXP-0023's 0.01826 after df rescaling (59 -> 45): 0.02091. (A first attempt asserted equality without the rescaling and stopped; that is the only difference.)
