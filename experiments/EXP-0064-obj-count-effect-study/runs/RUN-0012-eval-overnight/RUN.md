# RUN-0012 -- extended re-score of RUN-0008..0011 (same harness as RUN-0006)

- `code/overnight_queue.sh` ran `code/eval_extended.py --models <name>:-1:<ckpt>:<encoding> --fps-reps 3 --n-extra-goals 16` after each training; models gnn_dropesc (RUN-0008, tube), gnn_origenc (RUN-0009, orig), gnn_fixed_s43 / gnn_fixed_s44 (RUN-0010/0011, tube). The eval adds a second untrained baseline, `field_orig` (s0 + the ORIGINAL baseline action encoding).
- summary: `python code/summarize_extended.py --dirs artifacts/RUN-0006-eval-extended/{asrun,fixed} artifacts/RUN-0012-eval-overnight/*/ --out results/` (adds paired-vs-field_orig tables).
- queue log: `artifacts/RUN-0012-eval-overnight/queue_status.log` (2026-10-03 23:56 -> 2026-10-04 05:34).
