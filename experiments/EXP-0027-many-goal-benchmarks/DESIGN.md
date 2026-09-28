# EXP-0027 — many-goal ranking benchmarks: slateN (item 1) and gradient-descent action quality (item 2)

Written 2026-09-23 BEFORE either benchmark was run (plan gate). Follows
EXP-0026, which found (a) the 20-21-slate harness cannot order slateN margins
< 0.03, (b) goals behave as near-independent replicates of a slate for slateN
on L20mm/L40mm (not fully on randlen_test), (c) EXP-0023 cannot rank arms on
gradient_gain at n=10, though dv_grad separates them globally.

## Item 1 — eval_report with a many-goal set (TODO M6)

Goal set "many": the harness's OWN goal family, widened -- all 26
`helvetica_thin` letters (the harness uses O and T) + all 4 quadrants (the
harness uses one seeded random quadrant per slate). 30 fixed goals, same for
every slate and model; value fns lyapunov / mass_in_region / signed_mass as
before. Same 12 models, 3 corpora, slates as EXP-0026 RUN-0001.

C1. Difficulty is matched: goal-averaged lyapunov slateN of every model under
    "many" is within 0.05 of its 3-goal value on each corpus (checked, not
    assumed -- A4's random goals were easier).
C2. The number of model pairs whose paired slate-bootstrap 95% CI excludes 0
    (goal-averaged lyapunov) rises by >= 1.5x vs the 3-goal harness on L20mm
    AND L40mm. Refutes if < 1.2x on both. (Letters are all centred and
    overlap, so they may share more information than A4's random goals; this
    is the risk the prediction takes.)
C3. (descriptive) the reference table re-issued with per-model 95% CIs and
    rank intervals under both goal sets.

## Item 2 — gradient-descent action quality on DS-0001, many goals

States: all 20 DS-0001 slates. Pool: 100 corpus actions per state, drawn with
EXP-0023's exact generator (torch.randperm, seed 0, slates in order) -- slates
0-9 therefore reproduce EXP-0023's pools exactly (checked). Goals: 12 --
`corner` (lyapunov_weights, EXP-0023's goal), the 4 quadrants, and letters
O, T, A, C, H, L, S. Value fn: lyapunov (distance field of the goal mask)
only. Arms: EXP-0023's 6 OCC_ADAPTERS arms. Per (state, goal, arm):
a_rank = argmin predicted dv over the pool; a_grad = EXP-0023's projected
Adam (120 steps, lr 1.5e-3, 20-70 mm, 4 mm margin), started at a_rank, best
iterate by predicted dv. No CEM oracle (not needed to rank arms; cost).

Genesis: GenesisOracleEnv.rollout_candidates, full fidelity (EXP-0023 stage
2's final-batch path), for the 100 pool actions AND every distinct a_rank /
a_grad. Every outcome banked in DS-0004 (sim_path oracle_rollout_full). The
pool is scored under every goal from the same simulations, so all quantities
share one execution path.

Primary quantity, per (state, goal, arm), on slateN's scale (higher = better):
  grad_capture = (mean_pool_dv - dv_grad) / (mean_pool_dv - best_pool_dv)
  rank_capture = same with dv_rank (= slateN of the rank-only pick)
grad_capture > 1 means gradient descent beat every pool action. Raw dv_grad
also reported. (s, g) cells with pool range < 1e-4 are dropped (reported).

C4. Goals replicate states for dv_grad too: median ICC over goals of the
    per-(state, goal) paired arm difference in grad_capture < 0.3, and
    averaging over 12 goals shrinks the median per-pair sd of the per-state
    difference >= 2x vs a single goal (sqrt(12) = 3.5x is the independent
    limit). Refutes if the shrink is < 1.5x.
C5. With 20 states x 12 goals, >= 5 of 15 arm pairs are Holm-significant on
    grad_capture (EXP-0023 at 10 states x 1 goal: 0/15 on everything).
    Refutes if <= 1.
C6. (continuity) on corner / slates 0-9, a_rank equals EXP-0023's exactly
    and the simulated dv_rank / dv_grad match EXP-0023's stage-2 values to
    1e-4 (same path, deterministic) -- a pipeline check, not a claim.

Baselines: random pool pick (capture 0 by construction), pool best (capture 1).

## Cost, and the cheapest invalidating check

Item 1: ~25 min CPU (as RUN-0001). Item 2 stage 1: 20 x 6 GD runs batched
over 12 goals, ~10-20 min GPU. Stage 2: <= 100 + 144 actions per state,
32 envs per ~11 s batch -> ~1.5 min/state, ~30-35 min. Cheapest check first:
C6 on 2 slates (catches a pool / projection / path mismatch before the full run).

## The result that would most embarrass this

Goals do NOT replicate states for dv_grad (ICC high: an arm that optimises
well on a state does so for every goal), so item 2 is no more powerful than
EXP-0023 per state. The design catches it: C4 refutes and C5 likely too.
