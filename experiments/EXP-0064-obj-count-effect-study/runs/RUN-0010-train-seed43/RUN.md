# RUN-0010-train-seed43 -- overnight follow-up (2026-10-03/04), one change vs RUN-0005

- config: `config.yaml` here (diff vs RUN-0005 marked "ONLY change"). Same loop, adapter, 100 epochs.
- commit 3373e65b + uncommitted adapter knobs (committed next in f4a4cee6); RTX 4070 Laptop; run by `code/overnight_queue.sh` (exact argv + provenance: `artifacts/RUN-0010-train-seed43/COMMAND.txt`, `*.json`).
- result: all 100 epochs; best val loss valid [99/100][0/13] LR: 0.001000, Loss: 0.038204 (0.038204)
0.045945. Checkpoint (not promoted): `experiments/EXP-0064-obj-count-effect-study/artifacts/RUN-0010-train-seed43/2026-10-04-02-45-41-177841/net_best.pth` (all epoch checkpoints beside it).
- scored by RUN-0012 (`code/eval_extended.py`, 3 FPS reps, 17 point goals) -> `artifacts/RUN-0012-eval-overnight/<model>/`; summary `results/extended_summary.{md,json}`.
