# RUN-0001 — 72 episodes x 24 pushes with states recorded
queue_S.sh -> run_probe (tag exp0051_success) -> EXP-0043 batched_closed_loop.py --record-states, 2 models x {gd, cem} tuned x 9 goals (incl. two_squares) x starts 40-41, 1 s, 24 pushes. 2026-09-25 01:59-03:23, exit 0; commit 3bae8cd7, dirty. GPU shared with EXP-0048/0049/0050 jobs. Analysis: code/analyse.py -> results/analysis.json.
