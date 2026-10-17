# RUN-0005 — learned etc on state features vs f(V); TD fit

- `python scripts/run_probe.py --tag exp0070_etc_learned --exp EXP-0070 -- python experiments/EXP-0070-time-optimal-mpc/code/etc_learned.py` -> `results/etc_learned.json`
- data: the 1296 recorded episodes of RUN-0002 (states re-read from the source records); GroupKFold(5) by start state; sklearn 1.7.2 HistGradientBoosting / RidgeCV.
- run-to-run noise ceiling: inline script on EXP-0066 letters (recorded in EXPERIMENT.md body, not saved to a file).
- CPU, ~2 min. commit a5ccb680 dirty.
