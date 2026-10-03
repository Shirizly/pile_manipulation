# RUN-0004 — as-run eval of MODEL-0011 on DS-0022, source repo

- where: `~/Code/dyn-res-pile-manip` @ 5c9eca9, dirty; 2026-10-03.
- command (RECONSTRUCTED): `python eval_grouped_capture.py <run>/net_best.pth --config config/train/gnn_dyn_grouped.yaml --out test_outputs/eval_results_grouped.json`. Code copy: `code/eval_grouped_capture.py` (source-repo import paths; not runnable here as-is — RUN-0006 is the runnable re-score).
- output: `artifacts/RUN-0004-eval/eval_results_grouped.json` (overall, by group, per state).
- defects: I-1 (mirrored action), I-2 (escaped rows kept), I-5 (unseeded FPS), I-6 (one point goal, node truth), I-7 (pool size).
