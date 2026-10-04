# RUN-0011-train-seed44 -- overnight follow-up (2026-10-03/04), one change vs RUN-0005

- config: `config.yaml` here (diff vs RUN-0005 marked "ONLY change"). Same loop, adapter, 100 epochs.
- commit 3373e65b + uncommitted adapter knobs (committed next in f4a4cee6); RTX 4070 Laptop; run by `code/overnight_queue.sh` (exact argv + provenance: `artifacts/RUN-0011-train-seed44/COMMAND.txt`, `*.json`).
- result: all 100 epochs; best val loss valid [99/100][0/13] LR: 0.001000, Loss: 0.040866 (0.040866)
0.045936. Checkpoint (not promoted): `experiments/EXP-0064-obj-count-effect-study/artifacts/RUN-0011-train-seed44/2026-10-04-04-09-45-029467/net_best.pth` (all epoch checkpoints beside it).
- scored by RUN-0012 (`code/eval_extended.py`, 3 FPS reps, 17 point goals) -> `artifacts/RUN-0012-eval-overnight/<model>/`; summary `results/extended_summary.{md,json}`.
