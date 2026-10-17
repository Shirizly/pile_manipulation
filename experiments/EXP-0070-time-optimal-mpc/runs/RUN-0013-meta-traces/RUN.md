# RUN-0013 — incumbent vs deliberation time, prediction and truth

- `python scripts/run_probe.py --tag exp0070_meta_traces --exp EXP-0070 -- python experiments/EXP-0070-time-optimal-mpc/code/meta_traces.py` -> `results/meta_traces.json`
- 270 states (RUN-0008's), CEM on model lyap dV (mirror of learned_mpc.plan's CEM branch, pool 1024 from diag_small.bank, pop 1024, elite 0.25) for 1.0 s, running top-64 snapshot at 0.03/0.06/0.1/0.2/0.35/0.5/1.0 s; `lyap` and `mass` (blendnt1.0 rule, etc_map_m1 offsets) incumbents at each; 2233 distinct incumbents executed in Genesis (32 envs/batch)
- commit a5ccb680 dirty; GPU ~25 min
