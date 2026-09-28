# RUN-0001

Via run_probe (exp0032_pilot.json): code/pilot.py (defaults: 3 models x 3 planners x 2 goals x 4 DS-0006 starts, 8 steps, 1.0 s budget, 64 pile-aware candidates, TRAINING_PHYSICS). results/episodes.json checkpointed per step; results/summary.json from code/analyse.py. ~70 min GPU. NOTE: this run's 'rank' planner scored only its 64 initial candidates (budget unused); learned_mpc.plan was changed afterwards so rank resamples until the budget is spent. Commit 3bae8cd7, dirty.
