# RUN-0001 — goal ceilings V*(g) for 31 goals (CPU, ~25 s)
`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONPATH=. python -u experiments/EXP-0046-goal-ceiling/code/vstar.py`
-> results/vstar.json (checkpointed per goal), stdout.log. Goals = eval_report `many_plus` (30 `many` + `two_squares`).
Final invocation after two re-runs on 2026-09-25 (added noisy-greedy restarts; padded EDT so quadrant widths are not
inflated by the array edge; mass optimiser also seeded from the lyap optimum). Commit 3bae8cd7, dirty (goals.py /
eval_report.py edits from this experiment uncommitted).

# RUN-0002 — achieved_fraction + goal redundancy (CPU, ~4 s)
`python -u experiments/EXP-0046-goal-ceiling/code/analyse.py` -> results/achieved_fraction.json,
results/redundancy.json, runs/RUN-0002-analyse/stdout.log. Reads EXP-0044 goal_breadth_tuned.json (768 episodes),
EXP-0045 time_vs_actions.json (512), EXP-0030 RUN-0001 truth.pt / pred_*.pt (8 models) with DS-0006 step0 slate_idx.
