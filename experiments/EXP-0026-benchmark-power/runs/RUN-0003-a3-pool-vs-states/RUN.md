# RUN-0003 — A3: pool size vs states on DS-0001's cached pools

- **Commit:** 3bae8cd7, dirty (uncommitted: this session's edits to `Baselines/common/eval_report.py` (per-slate output, additive), `Baselines/common/goals.py`, `simple_mpc/adapters.py`, new `Baselines/common/paired_stats.py`, `simple_mpc/gt_bank.py`; plus pre-existing user edits to eval_report.py's MODELS dict and unrelated EXP-0022 logs). The exact run-time git state is in the `.json` meta beside the log.
- **Python:** /home/alon/anaconda3/envs/pme/bin/python via `scripts/run_probe.py` (python -u, 4 threads). Ledger entries: `runs/COMMANDS.jsonl` (run_probe's actual path, invariant run-probe-ledger-path-matches-doc is broken), copied into experiments/COMMANDS.jsonl.
- **Command:** `experiments/EXP-0026-benchmark-power/code/a3_pool_vs_states.py --out experiments/EXP-0026-benchmark-power/results/a3_pool_vs_states.json` (reps 200, seed 0)
- **Input:** experiments/temp/binned-pools/dv_cache_corner.pt (DS-0001, corner/lyapunov, cached nfd=MODEL-0003, visual-switched=MODEL-0001, descriptor=MODEL-0002). CPU, ~1 min.
