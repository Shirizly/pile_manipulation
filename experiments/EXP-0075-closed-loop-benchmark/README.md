# EXP-0075 closed-loop benchmark (started 2026-10-08; exploration -- no EXPERIMENT.md / register entry yet)
Closed-loop test of the EXP-0074 Sean-trained models (ensemble zoom128 + vanilla128, unrolled, val-selected mass balance in the loop) in legal Genesis episodes (batched driver of EXP-0043/0066, `--legalize --legalize-fallback`, seeded planners).
Design (user brief): single-push vs up-to-4-push planning; push length 20 mm only vs the full 20-70 mm range; mostly hard goals (letters at stroke width w20, EXP-0067 widths), <= 10 pushes per episode; planners from the 2026-10-07 strategic review
(horizon-H CEM with shift warm start and terminal objective, rank baseline); score = in-goal mass / placement optimum (EXP-0067 vstar_width), completion at 0.9 / 0.8 x optimum, steps to completion.
Code: code/wide_planner.py (models, objective, planners, runner), code/closed_loop.py (driver), code/queue.sh, code/analyse.py. Status and results: LOG.md.
