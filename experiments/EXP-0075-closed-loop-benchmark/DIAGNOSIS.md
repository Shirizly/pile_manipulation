# EXP-0075 diagnosis: why do horizon-H (H = 2, 4) planners do worse than H = 1?
Written 2026-10-09 after the benchmark (results/analysis_all.txt). Model: zoom128 + vanilla128 ensemble with mass balance. 32 episodes per cell (letters O/T/S/X, w20, DS-0006 starts 40-47). Value = lyapunov - 1.0 x in-goal fraction (lower = better).
Mean rel over pushes 1-10 (full push range): H=1 .752, H=2 .544, H=4 .402; 20 mm: .567 / .479 / .375.

## D1 code audit and unit tests (code/diag_unit.py) -- no bug found
* The first output of an H=2 / H=4 prediction equals the H=1 prediction exactly (max abs diff 0.0), and the planner's model path equals the validated evaluation path (EXP-0074 Ens + balance) to 1e-7.
* The planned trajectory IS passed on as part of the pool: after a decision, steps 2..H of the plan become the first H-1 steps of 16 noisy copies (sigma 5 mm) in the next decision's initial pool; the last step is a random bank draw. On a real recorded state these rows cost far less than the pool
  median (-0.57 vs -0.06) and rank in the top ~10 % of the 256 rows. BUT the warm start is discarded whenever the executed push was changed by legalisation or fallback, which happens in 29-33 % of decisions (full range) and 37-56 % (20 mm).
* Later pushes of fresh pool sequences are random rows of the current-state bank, and the Gaussian refit over (H, 4) treats them like the first push (elite spread 20-28 mm per step on the stand-in bank).
* In the best plans the later pushes are partly inert: for an H=4 plan on a recorded T state the predicted change of the occupancy per push was [104, 114, 22, 133] pixels of mass vs 148 for the H=1 plan's single push.

## D2 on-policy replay (code/diag_replay.py, no simulator): the model is NOT bad on the paths the planner visits
Predicting the executed pushes from the true state, j = 1..4 pushes ahead: Spearman between predicted and true value change .93-.98 in every cell; bias at j = 4: -0.026 (H=4, full range; slightly optimistic), +0.051 (H=1, full range; pessimistic), -0.010..-0.017 (20 mm). RMSE grows from .02-.04 (j=1) to .04-.09 (j=4). So multi-push prediction quality along executed paths is good.

## D3 first-push sacrifice (same 32 start states): the first push of a longer-horizon plan makes much less progress
True value change of the executed first push, full range: H=1 -0.251, H=2 -0.129, H=4 -0.097 (39 % of greedy); 20 mm: -0.101 / -0.070 / -0.043. With a terminal-only objective, postponing progress is free, and re-planning after every push lets the plan postpone again
(figures/fig_first_push.png: H=4 first pushes are often short, barely touch cubes, dV about 0). Closed-loop true value change after 1/2/4 pushes: H=1 -0.251/-0.433/-0.683, H=2 -0.129/-0.244/-0.347, H=4 -0.097/-0.181/-0.293.

## D4 simulator test (code/diag_sim.py): execute the whole selected plan open loop; execute the model's top-20 sequences
| plan | predicted value change after 1 / 2 / 4 pushes | true | optimism at the end | shifted by legaliser |
|---|---|---|---|---|
| H=1 (1 push) | -0.280 | -0.262 | -0.019 | 12/32 |
| H=2 (2 pushes, open loop) | -0.150 / -0.439 | -0.162 / -0.388 | -0.050 | 18/64 |
| H=4 (4 pushes, open loop) | -0.103 / -0.226 / -0.330 / -0.547 | -0.098 / -0.213 / -0.293 / -0.495 | -0.051 (10 %) | 41/128 |
* The selected sequences are optimistic and the optimism grows with the push index (H=4: -0.004, -0.013, -0.037, -0.051).
* The decisive comparison: greedy CLOSED-LOOP H=1 reaches a TRUE value change of -0.433 after 2 pushes and -0.683 after 4; the best H=4 plan found under the model PREDICTS only -0.547 after 4 pushes (true -0.495) -- the sequence optimiser did not even find plans as good as greedy
  re-planning. => a SEARCH failure (1280 evaluations over 16 dimensions with random later pushes), not just model error.
* Ranking of the model's top-20 sequences (executed in the simulator): Spearman .19 (H=1), .32 (H=2), .36 (H=4); regret of the model's pick vs the best of the 20: .028 / .063 / .071 (in value units; the plans are 3x larger for H=4, so relative regret is similar).
* Legaliser shifts 32-37 % of pushes in all three cases (the CEM leaves the legal manifold), which also contributes to predicted-vs-true differences.

## Conclusion
No code bug. The horizon deficit has three causes, in order of size: (1) procrastination -- a terminal-only objective makes early progress optional and receding-horizon re-planning never collects it (D3: first push 39 % of greedy progress; closed-loop H=4 after 4 pushes -0.293 vs open-loop execution of its own plan -0.495 vs greedy -0.683);
(2) search failure -- sequence CEM over (H,4) with random later pushes does not reach even the greedy solution (D4); (3) model optimism on SELECTED sequences, growing with push index to ~10 % at 4 pushes (D4), while the model is accurate on executed paths (D2).
Proposed fixes (not yet run): (a) cumulative / discounted objective sum_k gamma^k V(s_k) (or prefix-min + a push-count penalty) so early progress is rewarded -- this is the benchmark score itself; (b) later-push proposals that are pile-aware w.r.t. the PREDICTED state, or a nested search
(top-k first pushes x best-of-n second pushes) instead of Gaussian refit over random later steps; (c) execute the first 2 pushes of a plan before re-planning; (d) do not discard the warm start after a legaliser shift (re-legalise the shifted plan instead).
