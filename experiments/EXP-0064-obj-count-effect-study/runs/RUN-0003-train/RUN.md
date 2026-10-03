# RUN-0003 — train MODEL-0011 (as run; z-mirrored action), source repo

- where: `~/Code/dyn-res-pile-manip` @ 5c9eca9, dirty; 2026-10-03 04:53 → ~06:15 (run dir timestamp; source report "~1h20m"), GPU.
- command (RECONSTRUCTED): `python -m train.train_gnn_dyn_grouped` reading `config/train/gnn_dyn_grouped.yaml` (resolved copy as written by the run: `artifacts/RUN-0003-train/run_dir_config.yaml`; seed 42, batch 16, lr 1e-3, n_history 1, n_rollout 5, node_budget 30, adj_thresh 0.08, 90/10 per-group state split).
- code: `code/train_gnn_dyn_grouped.py` (loop verified identical to the source `train/train_gnn_dyn.py` by diff — `code/ported_reference/train_gnn_dyn_original.py`), `code/dataset_grouped_particles.py`.
- configured 300 epochs, **stopped by hand after epoch 100** (val flat 0.061-0.067 since epoch ~15-20). Best val 0.0615 at epoch 87.
- output: MODEL-0011 (`checkpoint.pth` = net_best; all epoch checkpoints in `epochs/`). Logs: `artifacts/RUN-0003-train/train_gnn_dyn_grouped.log`, `weights/MODEL-0011-*/epochs/train_log.txt`.
- known defect: action frame mirrored (issues.md I-1).
