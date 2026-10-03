# RUN-0005 — retrain with the corrected action frame → MODEL-0012 (this repo)

- what: RUN-0003 repeated with ONE change, `train.action_z_sign: -1` (z = -a1). Same loop (`code/train_gnn_dyn_grouped.py`, import paths only changed), same adapter, seed 42, batch 16, lr 1e-3, n_rollout 5, node_budget 30; `n_epoch: 100` to match where RUN-0003 was stopped (RUN-0003 was configured for 300). Escaped rows NOT filtered (kept identical to RUN-0003, issues.md I-2).
- resolved config: `config.yaml` (this dir). Data: DS-0021.
- commit d72bb304, **dirty** (other sessions' uncommitted work + this port's files); RTX 4070 Laptop 8 GB; started 2026-10-03 21:19 CEST.
- command: `PYTHONPATH=. python scripts/run_probe.py --tag exp0064_run0005_train_fixed --out-dir experiments/EXP-0064-obj-count-effect-study/artifacts/RUN-0005-train-fixed-frame --exp EXP-0064 -- python experiments/EXP-0064-obj-count-effect-study/code/train_gnn_dyn_grouped.py --config experiments/EXP-0064-obj-count-effect-study/runs/RUN-0005-train-fixed-frame/config.yaml --out-root experiments/EXP-0064-obj-count-effect-study/artifacts/RUN-0005-train-fixed-frame` (exact argv + git provenance in `artifacts/RUN-0005-train-fixed-frame/COMMAND.txt` and `*.json`).
- output: `artifacts/RUN-0005-train-fixed-frame/<timestamp>/` (net_best.pth, net_epoch_*.pth, log.txt) → promoted to MODEL-0012.
- finished 2026-10-03 22:50 CEST, all 100 epochs; best val loss 0.0453 (MODEL-0011 as run: 0.0615 on the same 200 val windows). Promoted to MODEL-0012 (sha256 570224e9...) by `code/finalize_model0012.sh`.
