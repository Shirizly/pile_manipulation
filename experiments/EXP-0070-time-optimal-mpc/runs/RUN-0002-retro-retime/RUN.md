# RUN-0002 — re-time recorded closed-loop episodes; bootstrap etc(V)

- `code/retro_time.py` (via run_probe tag exp0070_retro_time, 2nd run adds raw V to the rows) -> `results/retro_time.json`, `results/retro_episodes.json`
- `code/fit_etc.py etc_map_v0 results/retro_episodes.json` -> `results/etc_map_v0.json` (binned PAV + power-law fits; the power law is what the planner uses)
- `code/time_vs_progress.py` -> `results/time_vs_progress.json`; `code/budget_retime.py` -> `results/budget_retime.json` (EXP-0045, lyapunov completion)
- inputs: EXP-0051 legal, EXP-0054 legal, EXP-0052, EXP-0055 all, EXP-0066 (6 files) recorded episodes; EXP-0046 vstar.json; THETA letter 0.7 (env THETA_LETTER), 0.9 quadrants / two_squares
- commit a5ccb680 dirty; CPU/GPU seconds
