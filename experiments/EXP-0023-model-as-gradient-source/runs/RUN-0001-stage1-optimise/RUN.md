# RUN-0001 — stage 1, the model side (no Genesis)

Per state and per arm: pick the best action in the shared 100-action seed
pool by the model's own predicted `dv` (`a_rank`), then run gradient-descent
MPC through the model from that action (`a_grad`). Writes only actions; no
ground truth is touched here.

## Exact argv
```
OMP_NUM_THREADS=4 PYTHONPATH=. python -u \
  experiments/EXP-0023-model-as-gradient-source/code/stage1_optimise.py \
  --arms nfd_3ch_randlen nfd_3ch_finetuned nfd_warped_randlen \
         nfd_residual_warped linear_switched_soft linear_switched_hard \
  --out experiments/EXP-0023-model-as-gradient-source/artifacts/stage1_actions.pt
```
Defaults used: `--n-states 10 --pool 100 --steps 120 --lr 1.5e-3 --goal corner --seed 0`.

## Timing
188 s total (24–44 s per arm for 10 states x (100-action pool scoring + 120
Adam steps)). Under 3-way GPU contention (RUN-0010 training, plus two other
subagents' jobs) — **timings here are inflated and are not a benchmark.**

## Optimiser and constraints
- Adam, lr 1.5e-3, on the raw `[sx,sy,ex,ey]` in metres, 120 steps.
- Objective: the adapter's own `dv` — a COST, so `dv.sum().backward()` and
  Adam MINIMISES it.
- After every step the action is projected back into the legal set:
  endpoints clamped to ±0.060 m, then the push length rescaled into
  [0.020, 0.070] m along the current direction with the start held fixed.
- The returned action is the **best iterate by PREDICTED `dv`**, not the last
  — the only signal a real controller has.
- `bound_hit_frac` records, per (state, arm), the fraction of the 120 steps
  at which the box / length projection actually moved the action.

## Sanity of this stage in isolation
Every arm improved its OWN predicted `dv` on every one of the 10 states, by
roughly 1.2x to 3.5x. That is not a result — it is the tautology this
experiment exists to test against ground truth: a model asked to optimise
against itself always succeeds. Stage 2 asks Genesis what actually happened.
