# RUN-0003-budgeted-control

The control half of the time-budget experiment: measures slateN control
performance when each model is restricted to `N_i(T)` candidates (from
RUN-0002's `results/budgets.json`, `with_value` sweep), drawn without
replacement from the existing N=128-per-slate candidate pools.

- Command: see `COMMAND.txt`.
- Commit: `0ddab20f` (dirty -- same pre-existing docs/skills reorg as
  RUN-0001/RUN-0002, unrelated to this run's inputs/outputs).
- Device: cuda:0 (RTX 4070 Laptop, 8GB).
- Datasets: `n20_L20mm` + `n20_L40mm` (n20_L10mm excluded per task
  instruction), 20 slates x 128 step-0 candidates each, verified.
- Wall clock: ~58s.

## What ran

`code/eval_budgeted_control.py`:
1. Loads 8 model specs via `bench.py`'s own `make_nfd/make_schenck/make_gnn/
   make_switched_linear/make_desc_only` (the SAME code RUN-0001/RUN-0002's
   timing numbers came from -- device-bug-fixed, GPU-forced).
2. For every slate (20 per dataset x 2 datasets), builds one same-state
   128-candidate batch and runs every model's `pre()`+`fwd()` ONCE over the
   full pool (predictions are precomputed per candidate; budget-limited
   subsampling below is then free).
3. Routes every model's raw prediction (occupancy image for
   nfd/schenck/gnn/model0001_*/hybrid*; particle rollout, rasterised back to
   an occupancy grid, for gnn; predicted descriptor vector -> point-mass
   COM+mass readout for model0002_descriptor_only) through
   `eval_slaten_broad.py`'s OWN `build_goal`/`value_true_and_pred`/
   `desc_pointmass_value` -- the same value/scoring code EXP-0008 used, so
   the linear family and the NFD/GNN/Schenck family are compared through
   ONE scoring path downstream of "model produced a prediction."
4. For every (dataset, value_fn in {lyapunov, mass_in_region}, budget T in
   {2.5,5,10}ms): draws 50 independent without-replacement subsets of size
   `floor(N_i(T))` per slate (draws 0-4 are reported as the mandatory
   5-draw top/bottom/mean headline; all 50 as a secondary mean/p5/p95
   distribution), and scores each via `budgeted_slate_capture` --
   `slate_n_capture`'s formula but with the oracle-best/random-floor
   normalisation always computed over the FULL 128-pool regardless of the
   model's own subset size (the user's explicit design).
5. Also computes a K=32 fixed reference point (same machinery, n_i=32,
   budget-independent) for every model, per (dataset,value_fn).
6. Head-to-head wins/losses/ties + paired sem (draw 0) for 6 model pairs,
   per (dataset,value_fn,T).
7. Pools n20_L20mm+n20_L40mm (mean-of-means, both datasets ~20 slates).

## Scope reductions (documented, not hidden -- see script docstring)

- Goal shape: **corner only** (the task's own "lead with" shape); the other
  4 EXP-0008 shapes were not re-run here.
- Value functions: **lyapunov + mass_in_region**; `signed_mass_in_region`
  dropped.
- `N_i(T)` taken from budgets.json's `with_value` sweep (a real control
  loop pays for value computation too).
- persistence/random are NOT budget-limited (N=128 always) -- they are the
  floor the budgeted models are normalised against, not real models with a
  measured N_i.

## Output

`artifacts/RUN-0003-budgeted-control/results_control.json`.
