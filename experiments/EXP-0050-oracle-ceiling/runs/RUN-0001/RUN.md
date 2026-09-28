# RUN-0001 — simulator-as-model planners
- exp0050_greedy16 (01:37-02:07): --mode greedy --n-cand 16, 8 goals x start 40, 8 pushes -> results/greedy16_s40.json.
- lookahead / greedy64: cancelled before running (addenda 1-2).
- exp0050_oracle_cem (02:00): crashed at push 1 (torch default device cuda in randn) -> fixed.
- exp0050_oracle_cem2 (02:09-03:59): --mode cem (32 pile-aware + 2 x 32 Gaussian refits, 96 simulated/push), 8 goals x start 40, 12 pushes -> results/cem_s40.json (final states only; per-push states/mass added to the code afterwards).
- exp0050_oracle_cem_mass: queued (addendum 3).
All: commit 3bae8cd7 dirty; GPU shared with other Genesis jobs.
