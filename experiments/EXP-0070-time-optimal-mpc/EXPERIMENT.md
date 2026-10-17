---
id: EXP-0070
title: >
  Physical time per push is 0.8-1.2 s (41-47 % of it transit) and planning is as large; re-timed, a
  0.1 s CEM budget finishes tasks ~33 % faster than 1 s. Selecting pushes by t(a) + etc(V') does not beat
  lyapunov at equal wall time (CEM directly on J collapses into cheap do-little pushes; lyap search +
  time re-rank is a null), and greedy chained (rotate-in-place) continuations never beat a fresh push at
  states the lyapunov planner chose to stop at
tier: T1
mode: exploratory
date: 2026-10-05
hypothesis: null
claim: >
  (a) Under the gantry limits in Genesis/configs/basic.yaml (0.125 m/s, 2 m/s^2) with a 10 mm safe height,
  the physical time of a recorded closed-loop push (lift from the previous end, transit, descent, sweep,
  no final lift) is 0.8-1.2 s per push, 41-47 % of it transit, comparable to the 0.5-1 s planning budget;
  the analytic trapezoid matches Genesis to within 35 ms. (b) Re-timed with it (EXP-0045, lyapunov
  completion at 0.9), total task time is minimised at a 0.1 s (CEM) / 0.3 s (GD) planning budget, 6.3 /
  4.0 s (~33 / 25 %) below 1 s; in legal closed loop 0.1 s and 0.5 s tie at equal wall time (flat curve). (c) In legal closed loop (broad nfd_3ch_randlen, CEM 1024 / 0.25 / 0.5 s,
  16 pushes, DS-0006 scatter starts, 7 goals, planner seed 0), choosing pushes by J = t(a) + etc_g(V + dV)
  - etc_g(V), with etc_g a per-goal power law of the true lyapunov fitted on 1296 recorded episodes,
  does not raise in-goal mass at equal wall time over plain lyapunov: CEM run on J collapses into short
  cheap pushes (quadrants: -0.19 to -0.21 in-goal mass by 8-20 s); lyapunov search + re-ranking its top
  64 by J is within +-0.02 at every time (42 paired tasks), also after one policy-iteration refit. (d) At
  24 plate-down states visited by that lyapunov planner, no chained push (rotate about the blade centre
  by <= 60 deg, sweep 10-50 mm from the current pose; Genesis truth) beats the planner's own next push.
provenance:
  commit: a5ccb680
  dirty: true
  data_commit: "recorded episodes of EXP-0051 (legal), EXP-0054 (legal), EXP-0052, EXP-0055, EXP-0066 at their own commits; EXP-0046 vstar.json"
  script: experiments/EXP-0070-time-optimal-mpc/code/{sim_timing,retro_time,fit_etc,budget_retime,time_vs_progress,time_mpc,analyse_pilot,equal_time,refit_merge,etc_learned,chain_probe,chain_analyse,chain_model_eval}.py
  data: ["Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys (DS-0006 starts 40-47)", "EXP-0051/0052/0054/0055/0066 results/*.json (recorded states + actions)", "EXP-0045 results/time_vs_actions.json", "EXP-0046 results/vstar.json"]
  code_path: "action time: code/action_time.py (analytic trapezoid); closed loop: simple_mpc.learned_mpc.run_episodes_batched with ModelObjective / plan swapped in-process by code/time_mpc.py; truth: occ_for_scoring + mass_in_region; chained pushes: code/chain_probe.py primitives on SandboxManipulation (lower/sweep without lift, servo-driven rotation at operating height)"
  seed: 0
  split: "not applicable (closed loop; etc fits: GroupKFold by start state in RUN-0005)"
  runtime: "~3.5 h wall over two sessions (machine lost power 16:35-18:40; resumed)"
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004, RUN-0005, RUN-0006, RUN-0007, RUN-0008, RUN-0009, RUN-0010, RUN-0011, RUN-0012, RUN-0013, RUN-0014, RUN-0015]
budget:
  declared: "2 h (session 1) + 2-3 h (session 2), user"
  spent: "~1 h 45 min + ~1 h 45 min"
  outcome: within
design:
  varied:
    objective: "lyap (baseline) / time (CEM on J, power-law etc) / time_lin (linear etc) / rate (dV / (t_plan + t)) / time_rerank (lyap search, J re-rank of top 64) / time_rerank on the refitted etc_map_v1"
    planning_budget: "0.5 s (pilots), 0.1 s (pilot_b01); 0.03-1 s re-timed from EXP-0045"
    action_set: "free pushes (pile-aware + the planner's pick) vs chained pushes (dphi in -60..60 deg x L 10-50 mm from the plate pose), Genesis truth"
    etc_model: "per-goal power law of V / PAV-binned / ridge and gradient-boosted features / fitted value iteration (TD)"
  held_fixed:
    model: "nfd_3ch_randlen (broad-trained), CEM pop 1024 elite 0.25, n_cand 1024, bank 16384, legalize + legal fallback, planner seed 0"
    starts: "DS-0006 scatter starts 40-47 (v0) / 40-45 (v0b, v1, b01)"
    completion: "in-goal mass / EXP-0046 optimum >= 0.9 (quadrants, two_squares), >= 0.7 (letters; pilot convenience, 0.9 is reached in 0.5 % of recorded letter episodes)"
    gantry: "plate.speed 0.125 m/s, acceleration 2 m/s^2, safe height 10 mm, rotation assumed 3.14 rad/s / 20 rad/s^2 (sensitivity 1.57-6.28)"
  baselines: ["do-nothing: the start state (k = 0) in-goal mass, every table", "lyapunov MPC (the existing objective) paired by (goal, start, seed)", "old clock: flat 2 s per push (METRICS completion_time)", "f(V) power law as the etc baseline for learned etc", "replicate pushes as the simulator noise floor (test A)"]
  metric: "task_time_physical (METRICS.md) and in-goal mass at equal wall time (4-20 s); slateN / accuracy not applicable to closed-loop time (reference rows: none computed -- see Threats)"
noise_floor: "paired (goal, start) bootstrap over 42 tasks: equal-time in-goal mass CI half-width ~0.03-0.06 (ALL), ~0.05-0.1 per family; censored task time CI half-width ~1.5 s; Genesis replicate push: 0.13 mm mean per-cube, dV 3e-5"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  (a) Genesis: sweeps and free-air transits reach 1 mm of target 34-35 ms before the analytic T(d); per
  recorded push 0.77-1.18 s mean (p10-p90 0.58-1.53), transit 41-47 % of it (35 % for the linear model).
  (b) EXP-0045 re-timed: CEM 0.03/0.1/0.3/1.0 s -> 23.4/12.8/13.1/19.1 s; GD 26.6/14.7/12.1/16.1 s (old flat
  clock 39.3/20.1/19.6/25.5). (c) v0 (16 quadrant pairs): time cell completes 0.19 vs lyap 0.94, +6.3 s
  [+3.3, +8.8]; time_lin +4.7 s; rate +0.8 s n.s. v0b (42 pairs): time_rerank -0.10 s per push (sig.),
  censored task time -0.28 s [-1.7, +1.3], equal-time in-goal mass +-0.02 (all CIs span 0); rate +1.7 s
  [+0.1, +3.3] task time but equal-time +0.02..+0.03 (CIs touch 0). v1 (refit on 6 own episodes per goal):
  equal-time -0.003 at 16 s overall, quadrants -0.04..-0.06 (CI excludes 0). b01 (0.1 s): lyap 0.1 s vs 0.5 s
  equal-time +-0.02 (CIs span 0) despite -0.045 in-goal mass per push; time_rerank vs lyap at 0.1 s +-0.03. (d) Chained
  pushes: best chained dV mean +0.0009 vs planner -0.036 (0/24 states better), vs best of 31 random free
  -0.0093 (chained better in 21 %); NFD ranks chained pushes Spearman 0.71 vs 0.77 (1 step), 0.61 vs
  0.70 (2 steps), dV MAE +38 %. Learned etc (RUN-0005): features / boosting R2 0.44 vs f(V) 0.43 on
  letters; plate features +0.005; ~half of time-to-complete variance is run-to-run noise from the same
  start. No-lift vs standard transitions (RUN-0006 A): 0.6 mm mean per cube, dV 3.5e-4 (~3 % of a push);
  lifting a settled pile 0.002 mm. Centre-pivot rotation +-10/20/40/60 deg moves 2.4/4.6/5.5/6.9 cubes
  > 1 mm, in-goal mass 0.004/0.010/0.044/0.082. Loop (2026-10-06, RUN-0009..0012): offset-corrected TD maps remove
  the stall (endgame all-J>0 share 0.86 -> 0.06); re-ranking the lyap top-64 by J on the MASS-GAP readout raises
  in-goal mass at equal wall time by +0.03..+0.06 vs lyap (iteration 1, iteration 2, ablation); removing t(a) from J
  does not lose it (paired -0.03 at 20 s WITH the time term).
verdict: supported
downgrades: [imprecision, indirectness, incomplete-design]
grade: very-low
supersedes: []
invalidated_by: null
---

## The idea, refined (plan as executed)

**Objective.** The task's true cost is total wall time to completion: planning + physical execution,
where execution includes getting the plate to the next push. MPC so far ranks pushes by predicted value
change only. Time-optimal MPC picks

    a* = argmin_a  t(a | plate pose) + E[ etc(s'(s, a), g ; pi) ]

with etc the expected remaining time under the policy pi itself (a Bellman fixed point,
etc(s) = t_plan + min_a [t(a) + E etc(s')]). Pilot approximations, each a knob to revisit:
1. **t(a)** — analytic, checked in Genesis (RUN-0001): lift to safe height h (from the last push's end),
   transit at sweep speed (trapezoid, rotation concurrent and never limiting), descent h, sweep L, no lift
   after. `code/action_time.py`; METRICS `task_time_physical`.
2. **etc(s, g)** — a per-goal scalar map of the TRUE lyapunov V(s), bootstrapped by Monte-Carlo
   remaining time in 1296 recorded closed-loop episodes (RUN-0002), censoring handled by one imputation
   pass, then a smooth power law etc = c V^p (PAV-binned maps have plateaus -> zero marginal value of
   progress -> the planner stalls; seen in the smoke run).
3. **E[.]** — the learned model's single predicted dV (nfd_3ch_randlen), V' = V_soft(s) + dV_pred.
   No model-uncertainty term.
4. **Policy iteration** — refit etc on the episodes the time-aware policy itself produced, re-run.
5. **Search** — CEM directly on J collapsed; the working variant searches on lyapunov and re-ranks the
   64 best actions by J (`time_rerank`).

## Refined framing after session 1 (user notes, 2026-10-05 evening)

1. **Transitions end at the end of the sweep; the lift belongs to the NEXT action.** The state is
   (particles, plate pose, plate down). Datasets and the MPC runner instead end every transition with
   lift + settle and teleport the plate away. Test A: lifting a SETTLED pile changes nothing (0.002 mm),
   so in the simulator the two s' markings the user anticipated reduce to "settled, plate down" (lift or
   not is then free) vs "mid-curve, unsettled" (a chained segment started before the pile is at rest --
   not simulated here; that is where a separate head / small model would be needed). Lifting BEFORE the
   pile settles (current execute_action) differs from plate-down settling by 0.6 mm mean per cube, ~5x
   the replicate floor and ~3 % of a push's dV: existing data are a fair approximation for n20 scatter
   (user: likely small for small piles and scatter; untested for large piles).
2. **etc(s, g) should be a learned model, jump-started by f(V(s)).** What the pilot exposed: a one-step
   planner compares the seconds a push costs, t(a), with the seconds it buys, etc(V) - etc(V'). If the
   etc curve is too shallow for the dV pushes actually achieve, EVERY push looks not worth its time,
   min J > 0, and the planner picks the cheapest do-little push. A Monte-Carlo fit of f(V) on another
   policy's episodes has no reason to agree with the planner's own predicted dV; a value fitted to satisfy
   etc(s) = t + etc(s') on transitions is consistent by construction. Test D: cheap features do not beat
   f(V) (R2 0.44 vs 0.43), the plate pose adds nothing measurable, and roughly half of time-to-complete
   variance is run-to-run noise from the same start -- so the case for a learned etc is Bellman
   consistency with the planner's own model, not a better fit. A fitted-value-iteration (TD) booster scores R2 0.23 on the same
   completed-episode targets, but that test is biased against it: completed episodes are the fast ones, and TD
   (which also learns from the 75 % censored letter episodes) correctly predicts longer remaining times. A fair
   test of TD needs on-policy closed loop, not held-out Monte-Carlo targets.
3. **Rotation in place (about the blade centre), chained sweeps; explicit vs implicit "spline" models.**
   A chained action is (dphi, L) relative to the plate pose (2-D search, not 4-D) and costs
   t_rot(dphi) + T(L): no lift, transit or descent (~0.6 s of a ~1 s push). Test B: a centre-pivot
   rotation is a small, local transition for |dphi| <= 20 deg (2-5 cubes move > 1 mm, max 2-6 mm, in-goal
   mass change <= 0.014), comparable to a small push at 40-60 deg. Test C: greedily continuing from where
   the lyapunov planner stopped never beats the planner's fresh push (0/24) -- but those stops were
   chosen as stand-alone pushes, so this does not test PLANNED curves. The implicit-pose NFD ranks
   chained pushes almost as well as free ones one step ahead (Spearman 0.71 vs 0.77) and worse two steps
   ahead (0.61), consistent with the user's caveat that chaining pays only if multi-step prediction does.

## Why this test discriminates
If time per push were near-constant or unrelated to the action, t(a) would add a constant and every cell
would tie; transit share (41-47 %) and per-push spread (0.6-1.5 s) say it is not. If the bootstrapped etc
were a usable value function, the time-aware cell would match lyapunov's in-goal mass at equal wall time
while spending less per push; if not, it would trade progress for cheap pushes (v0 shows exactly that
when CEM optimises J). If chained pushes had value at visited states, the Genesis-truth best chained push
would beat a fresh one in a sizeable share of states.

## What was actually run
- RUN-0001: Genesis timing probe; two failed invocations before the reported one (see RUN.md).
- RUN-0002: re-timing of 1296 recorded episodes, etc bootstrap (`etc_map_v0`), time-vs-progress,
  EXP-0045 budget re-timing (lyapunov completion: EXP-0045 recorded no states).
- RUN-0003: pilot_v0 (lyap / time / time_lin / rate): **stopped after 2 of 7 chunks** (16 quadrant pairs)
  once CEM-on-J's collapse was clear; then pilot_v0b (lyap / rate / time_rerank, 7 goals x starts 40-45,
  126 episodes). The machine lost power at 16:35 during v0b chunk 1; chunk 0 survived on disk and the
  run was resumed (completed episodes skipped, the half chunk re-run) -- the ledger shows starts with no
  end events for the interrupted runs.
- RUN-0004: `etc_map_v1` refit from v0b's time_rerank episodes (6 per goal; two_squares and letter_O kept
  v0, too few completions) -> pilot_v1 (time_rerank only); pilot_b01 (lyap / time_rerank at 0.1 s).
  v1 ran alone (no lyap in its chunks), so its comparison with v0b's lyap crosses chunk composition / GPU
  load; it also overlapped ~3 min of test-C smoke runs.
- RUN-0005: learned etc on state features (CPU).
- RUN-0006: Genesis tests A (lift), B (rotation), C (chained vs free, 24 states, 32 + 32 candidates);
  test A overlapped v0b planning for ~1 min (cells interleaved, so all cells shared it).
- RUN-0007: NFD ranking of the test-C candidates, 1 step from the true plate-down state and 2 steps
  through its own prediction of it.
- Code is new and was untracked at run time; tree dirty with other agents' uncommitted edits
  (`simple_mpc/learned_mpc.py` is read as is). The objective swap replaces
  `simple_mpc.learned_mpc.ModelObjective` / `.plan` inside the experiment's own process only.

## Numbers
Results files: `results/sim_timing.json`, `retro_time.json`, `budget_retime.json`,
`time_vs_progress.json`, `etc_map_v0.json`, `etc_map_v1.json`, `analysis_pilot_{v0,v0b,v1,b01}.json`,
`equal_time_*.json`, `etc_learned.json`, `chain_probe_{lift,rotate,chain}.json`, `chain_analyse.json`,
`chain_model_eval_nfd_3ch_randlen.json`.

Equal-wall-time in-goal mass (fraction of optimum), paired minus lyap, v0b, 42 tasks:

| cell | 4 s | 8 s | 12 s | 16 s | 20 s |
|---|---|---|---|---|---|
| lyap (level) | 0.33 | 0.50 | 0.60 | 0.66 | 0.70 |
| time_rerank | +0.013 | +0.009 | +0.013 | -0.001 | -0.017 |
| rate | +0.007 | +0.023 | +0.031 | +0.028 | +0.021 |
| time_rerank, refit v1 | -0.020 | +0.011 | +0.021 | -0.003 | -0.028 |

(all CIs span 0 overall; v0 CEM-on-J, quadrants: -0.07 / -0.19 / -0.19 / -0.22 / -0.21, CIs exclude 0)

Per-push physical time by time quartile, letters (legal recorded episodes): longer pushes made MORE
progress per second of (plan + action): 0.0059 -> 0.0084 dV/s from the shortest to the longest quartile
(correlation only; the planner chose long pushes when they were worth it).

**Budget in legal closed loop (b01, 0.1 s, 42 tasks, paired with v0b's 0.5 s by (goal, start, seed);
`results/budget_b01_vs_v0b.txt`).** lyap: per PUSH the 0.1 s planner is worse (in-goal mass -0.043 / -0.048 /
-0.023 after 4 / 8 / 16 pushes, ~2048 vs ~12000 model evaluations per decision), but at equal WALL time the two
budgets tie: +0.002 / +0.021 / +0.014 / -0.003 / -0.023 at 4-20 s, every CI spans 0 (quadrants -0.05 at 16 s,
CI just excludes 0). time_rerank vs lyap at 0.1 s: equal-time +-0.03, CIs span 0 except two_squares at 8 s
(n = 6). So the budget curve is flat between 0.1 and 0.5 s; the re-timed EXP-0045 saving is mostly from not
spending 1 s.


## Why minimising J prefers short pushes (RUN-0008, offline diagnosis, 270 visited states)
Not because the model is optimistic about small pushes: across 5056 executed pushes the model's dV is
`pred = +0.015 + 0.96 x true` -- an almost purely ADDITIVE pessimistic offset, roughly the same for every
push length (+0.013..+0.018 per length bin), so short pushes are if anything predicted harmful (20-30 mm:
pred +0.014 vs true -0.003). The offset grows as the task nears completion (+0.009 at V > 0.15, +0.021 at
V < 0.03). Three mechanisms, in order of weight:
1. **Exchange rate.** Between the J-planner's push and the lyap planner's push the trade is a near-tie: the
   lyap push makes ~0.007-0.013 more (model) progress and costs ~0.3 s more (longer sweep + longer transit,
   61 vs 48 mm, 74 vs 56 mm). Whether progress wins depends only on etc's slope (seconds per unit V). The
   quadrant power law (p = 0.25, fit R2 0.09) gives 22-31 s/V where most quadrant decisions happen; the
   lyap planner's own push then has a Bellman residual of +0.67 s (t_plan + J > 0: the map says even the
   best push is not worth its time); letters / two_squares have negative residuals (-0.9 / -2.3 s) and
   choose productive pushes.
2. **Endgame stall (the closed-loop collapse).** Near completion (quadrant V < 0.08) <= 2 % of a broad bank
   is predicted to improve V at all and the best push's predicted gain (-0.016) is about the size of the
   offset (+0.017; implied true gain ~-0.033). J > 0 for 96-100 % of these states, so the argmin becomes
   the cheapest push (27-35 mm, short transit), which really does little, and the loop repeats. The lyap
   planner is immune: an additive offset never changes argmin dV, and it has no absolute-seconds term to
   lose to. Completion is a mass-threshold event invisible to a smooth etc(V), so the large time saving of
   FINISHING is never on offer.
3. **Search (secondary).** CEM on J returns a push with worse J than the lyap planner's push in 36-46 % of
   states (the smooth t(a) term pulls the elites toward nearby, short pushes), but the mean J gap is small
   (-0.49 vs -0.45 s).
Subtracting the calibrated offset b(V) makes the value of the lyap push exceed its time in the 0.03-0.08
band (J +0.64 -> ~-0.2) but flips the cemJ-vs-cemV ranking in only ~10 % more states: the offset sets
J's SIGN (stall vs no stall), the slope sets the trade-off.


## The loop as the user specified it (2026-10-06) -- RUN-0009..0011
etc_g(s) = f_g(v(s, g)), bootstrapped from recorded trajectories (v vs actual time to completion), then updated
from the trajectories each new policy produces (TD); MPC OPTIMISES v (lyapunov) and J only re-orders the selection,
softly: execute argmin over the planner's top-64 pool of (1 - lam) z(dV) + lam z(J), lam raised as J becomes
trustworthy. v for the map: lyapunov or the mass gap to completion, max(0, theta - mass_rel) (the session's
"completion-aware" point is just this choice of v). J reads v(s') through the model with a calibrated per-readout
offset (RUN-0008: the model's change is additively biased).

**Maps (RUN-0009, `etc_map_m0`).** MC power laws (the session-1 bootstrap) have a held-out mean Bellman residual
f(r_k) - dt_k - f(r_{k+1}) of -0.6..-0.9 s for every goal and both readouts: on the observed trajectories each push buys
less time-to-go than it costs -- the map is too shallow, the data-side view of RUN-0008's stall. TD fits are ~1.5x
steeper (c) and Bellman-consistent (-0.3..+0.1 s); their held-out MC R2 is negative, but that test only sees the
completed (fast) episodes and is biased against TD. The mass-gap readout fits noisier than lyapunov (Bellman MAE
1.3-2.4 vs 1.0-2.0 s). Model offsets (`etc_v2_calib.json`): lyapunov change +0.008..+0.027 (largest near completion);
mass-gap change +0.02..+0.04 near completion, ~0 to -0.016 far from it.

**One-step decision test (RUN-0009, 270 states, Genesis truth for every distinct pick).** Over the lyap planner's
top-64 pool: (i) the offset correction removes the stall -- share of quadrant endgame states (V < 0.08) where every
push has J > 0 falls 0.97 -> 0.20 (MC map) and 0.86 -> 0.06 (TD map); (ii) re-ranking by J on a LYAPUNOV readout picks
pushes ~5 mm shorter, saves ~0.06 s and loses proportionally more true dV -- progress per second equal or slightly
lower than the lyap pick in every band (lyap pick rate 0.0249 V/s overall vs 0.0243-0.0246); (iii) J on the MASS-GAP
readout (gap|td+off) keeps true dV (-0.0494 vs -0.0496) while true in-goal mass gain per push rises +27 % overall
(+0.131 vs +0.103) and +54 % on letters / two_squares (+0.099 vs +0.065), in-goal mass per second +30 %; blend
lam = 0.75 keeps most of it (+0.126), lam = 0.5 about a third. One step only -- a greedy mass choice can cost later
(EXP-0052), hence the closed loop.

**Pre-declared before iteration 1's results (2026-10-06 02:05):** iteration 2 refits gap|td on all data with
iteration-1 blend episodes weighted x3 (map m1) and runs blend<lam2> + a lyap control on the same 42 tasks, where
lam2 = 0.75 if BOTH (a) the better iteration-1 blend cell's mean equal-time in-goal mass >= lyap's at 12 s AND at
16 s (point estimates), and (b) |mean Bellman residual| of map m0 on that cell's own transitions < 0.5 s;
lam2 = 0.5 if exactly one holds; lam2 = 0.25 if neither.

**Iteration 1 (RUN-0010, map m0 = recorded-trajectory bootstrap, 42 paired tasks, lyap re-run in the same chunks).**
Equal-wall-time in-goal mass minus lyap (95 % CI):

| cell | 4 s | 8 s | 12 s | 16 s | 20 s |
|---|---|---|---|---|---|
| lyap (level) | 0.33 | 0.50 | 0.60 | 0.66 | 0.70 |
| blend lam 0.5 | +0.031 [+0.014, +0.048] | +0.041 [+0.011, +0.071] | +0.054 [+0.019, +0.091] | +0.028 [-0.014, +0.072] | +0.024 [-0.025, +0.073] |
| blend lam 1.0 (J only) | +0.043 [+0.020, +0.068] | +0.047 [+0.018, +0.075] | +0.050 [+0.009, +0.089] | +0.040 [-0.000, +0.080] | +0.023 [-0.004, +0.052] |

blend 1.0 censored task time -1.42 s [-2.72, -0.18] (letters -2.23 s [-3.57, -0.96], equal-time letters +0.05..+0.06 at
4-12 s with CIs excluding 0, so not only the censoring artefact); completion 0.57 vs 0.47; per push -0.04 s. two_squares
gains most at 12-16 s (+0.12..+0.20, n = 6), quadrants ~0. The pre-declared rule gave lam2 = 0.5: condition (a) held,
(b) failed narrowly (m0's mean Bellman residual on blend 1.0's own transitions +0.57 s > 0.5).

**Iteration 2 (RUN-0011, map m1 = m0 refit with iteration-1 blend 1.0 trajectories x3, lam2 = 0.5 by the rule, fresh lyap
control in the same chunks, 42 tasks).** Equal-time in-goal mass minus lyap: +0.008 / +0.031 [+0.006, +0.060] / +0.031
[-0.005, +0.068] / +0.033 [-0.003, +0.071] / +0.047 [+0.009, +0.088] at 4-20 s; letters +0.060 at 12 s and +0.062 at 20 s
(CIs exclude 0); completion +0.14 [0.00, +0.29]; censored task time -1.11 s [-2.46, +0.16]. The blend gain replicates in
a fresh run against a fresh baseline at about iteration 1's lam-0.5 size; one refit step (~42 new episodes) adds
nothing visible.

**Ablation (RUN-0012, exploratory: added after iteration 1 was seen; 42 tasks, lyap / blend 1.0 / blendnt 1.0 in the same
chunks, map m0).** blendnt = blend 1.0 with t(a) removed from J (mass-gap re-rank of the lyap pool, nothing else changed).
vs lyap at 4 / 8 / 12 / 16 / 20 s: blend 1.0 +0.055 / +0.050 / +0.042 / +0.024 / +0.026 (first two CIs exclude 0 --
iteration 1's early gain replicates); blendnt 1.0 +0.032 / +0.057 / +0.031 / +0.049 / +0.057 (8, 16, 20 s CIs exclude 0).
Paired blend 1.0 minus blendnt 1.0: +0.023 [0.000, +0.046] at 4 s (letters +0.035), then -0.007 / +0.011 / -0.026 /
-0.031 [-0.057, -0.006] at 20 s (quadrants -0.04 at 16-20 s, CI excludes 0); the time term saves 0.037 s per push.
**So the closed-loop gain of the loop's selection comes from re-ranking the lyap pool by the mass-gap time-to-go
(the readout), not from the action-time term** -- and with t(a) removed, argmin J over the pool IS argmax predicted in-goal mass gain (f is
monotone and the offset is constant within a state), so the time-to-go map contributes nothing beyond choosing in-goal
mass as the selection value. The control that was run is plain lyapunov; optimising mass directly (CEM on mass) was not: within a pool of 64 near-optimal pushes the time differences are
~0.04 s per push, and preferring the cheaper one helps marginally in the first seconds and costs progress later.

## Deliberate vs act: a value-of-computation meta-planner (user, 2026-10-06; RUN-0013..0015)
Everything in seconds of task time. w(s) = f'(r(s)) is the price of progress (TD time-to-go map on the mass gap);
the incumbent after deliberating tau is a*_tau with predicted gain D_tau (offset-corrected), true gain kappa(tau) D_tau
(kappa absorbs the winner's curse of longer search); continue iff  w kappa dD(tau) - dt_act(tau) > dtau,  i.e. one
more step must buy more task time than it costs. A constant budget is the state-blind special case. Checked per CEM
iteration it is one multiply-compare.

**Traces (RUN-0013, 270 states x checkpoints 0.03-1.0 s, 2233 incumbents simulated).** The mass-selected incumbent's
TRUE in-goal gain peaks after ONE CEM iteration (0.06 s: 0.150; 0.03 s pool only 0.132) and then FALLS (0.143 at 0.2 s,
0.130 at 1.0 s); its predicted (corrected) gain follows (0.157 at 0.1 s -> 0.140 at 1.0 s): as CEM converges, the
top-64 collapses onto near-copies of the lyapunov optimum and the mass re-rank loses its choice. The lyap incumbent's
true dV improves until ~0.2 s (-0.051) and is flat after (-0.052 at 1.0 s) -- EXP-0045's knee, reproduced.

**Replay (RUN-0014, seconds of task time per decision, N = f(r0) - f(r0 - true gain) - t_act - tau, cross-fitted).**
const 0.03 / 0.06 / 0.1 / 0.2 / 0.35 / 0.5 / 1.0 s: +1.72 / **+2.06** / +1.88 / +1.75 / +1.43 / +1.17 / +0.57; the
best constant (0.06 s) is the same chosen in hindsight or on the other fold. Rule A (r0-only schedule) +1.87 (-0.19
[-0.29, -0.09] vs best constant), rule B (online incumbent) +1.83 (-0.22 [-0.33, -0.12]); per-state oracle +2.88. The
rules lose because the improvement curve is not monotone and they sometimes continue on noise (two_squares); the
oracle's +0.8 s is per-state luck in the outcome that nothing observable predicts. For letters both rules equal the
0.06 s constant. Prediction for the closed loop: ~+0.9 s per decision from cutting the budget 0.5 -> 0.06 s; no
state-dependent rule beats the best constant.

**Closed loop (RUN-0015 / 0015b, 42 tasks, mass re-rank selection in every cell, only the budget differs).** First run
with 16 pushes per cell had an equal-push / unequal-time confound (fast cells finish at ~19 s vs ~25 s); re-run with 22
pushes for the fast cells so all cells span ~25 s. In-goal mass at equal wall time minus constant 0.5 s:

| cell | 4 s | 8 s | 12 s | 16 s | 20 s | 30 s | completion (vs 0.55) |
|---|---|---|---|---|---|---|---|
| const 0.06 s | +0.044 [+0.023, +0.066] | +0.038 [+0.007, +0.069] | +0.010 | +0.007 | +0.016 | +0.001 | 0.57 |
| rule-A meta (mean 0.061 s) | +0.039 | +0.040 | +0.035 [+0.003, +0.070] | -0.024 | -0.012 | -0.027 | 0.19 (quadrants 0.25 vs 0.92) |

Task time (censored) const 0.06 s -0.16 s [-1.85, +1.68]. **The replay's predicted +0.9 s per decision does not survive
the closed loop:** cutting planning to one CEM iteration is ahead in the first ~10 s and level after; the meta rule,
which gives near-complete states the shortest budget, fails to finish quadrants. Why the replay misled: its states were
drawn at pushes 1, 2, 4 (smallest-gap tercile still r0 = 0.21), so it never saw the endgame, where the closed loop says
search is worth more. RUN-0013's lyap incumbent shows the same direction within the sampled range (true gain keeps
improving to 0.35-1.0 s for the smallest-gap tercile, saturates by 0.2 s for the largest).

## Reading
- (2026-10-06, meta-planner) Value of computation is strongly state-dependent but in the OPPOSITE direction to a
  gap-proportional gain model: early in a task one CEM iteration (0.06 s) is the best use of time, near completion more
  search pays. A constant 0.06 s ties constant 0.5 s at equal wall time over the task (early lead, same finish); the
  rule-A schedule stalls near completion. The formal rule (continue iff w kappa dD > dtau) stands; what failed is its
  gain model, fitted on early-task states only.
- (2026-10-06) The user's loop works as a SELECTION layer: optimise lyapunov, re-rank its top-64 by a time-to-go map on
  the mass gap with the model's offset removed -> +0.03..+0.06 in-goal mass at equal wall time, replicated in three
  runs (iterations 1, 2, ablation). The action-time term inside J is not what produces it (ablation); at this pool
  width t(a) is nearly constant. One TD refit step changed nothing visible; it needs many more on-policy episodes.
- Physical time is real and action-dependent, but at this planner's quality the decision that moves
  total time is HOW LONG TO PLAN, not which push: the re-timed EXP-0045 says 0.1 s CEM finishes ~6 s
  sooner than 1 s, while re-ranking pushes by physical time moves total time by less than the noise.
- A time objective is only as good as its etc: a map from another policy, read through a model's own dV,
  is not Bellman-consistent, and a one-step planner then confuses "cheap" with "good". Searching on
  lyapunov and using time only to break near-ties avoids the collapse but gains nothing measurable.
- Removing repositioning by chaining needs PLANNED multi-segment sweeps (and a model that predicts the
  second segment); greedy continuation from a stand-alone push's stop is worthless.

## What would change the verdict
- A TD-trained etc (on the planner's own transitions, model-predicted dV) used in J: if min J ~ 0 and
  time-aware selection then beats lyap at equal time, (c) is narrowed. ~1 day.
- 2-segment planned chains (search over (a1, dphi, L2) jointly, Genesis truth, then the NFD): the only
  fair test of chaining. ~2-3 h for the oracle version.
- Larger piles / clumps (where lift-before-settle and rotation disturbance may matter more).

## Threats
- Completion threshold 0.7 for letters is a convenience; most letter episodes are censored, which makes
  censored task time reward cheap pushes mechanically -- hence the equal-time metric.
- One model, one planner, one seed; 42 tasks; quadrants only 12 pairs.
- The gantry's rotation limits are assumed; the transit model ignores acceleration blending between
  lift, transit and descent (conservative: real paths can blend corners).
- slateN / accuracy are not reported: this record measures control time, not model quality.

## Unrelated findings
- `scripts/run_probe.py` invocations killed by the power loss left ledger start events without ends
  (information, not an error, per the skill).
- `simple_mpc.learned_mpc.plan`'s CEM is captured by any smooth additive term in the objective (here
  t(a)); worth knowing for any future objective that mixes a smooth cost with the rugged model dV.
