---
id: EXP-0075
title: >
  Legal closed loop with the wide-trained ens128 (one model, one planner seed, letters O/T/S/X): the PLANNER dominates --
  full 20-70 mm pushes, fetch proposals + legalise-before-scoring, a cumulative objective for H>1 and OT-seeded pools move
  rel@10 by 0.08-0.25, the model choice by <= 0.03; at equal evaluations H=1 is never beaten by H=2/4
tier: T1
mode: exploratory
date: 2026-10-08
hypothesis: null
claim: >
  In legal Genesis closed loop (EXP-0043/0066 batched driver, --legalize --legalize-fallback, TRAINING_PHYSICS; DS-0006
  starts 40-47 or fresh 48-63; letters O/T/S/X at stroke width w20 (w15 / w10 where stated); 10 pushes; score rel =
  in-goal mass / EXP-0067 placement optimum; planner seed 0; model ens128 = MODEL-0025 unless stated; paired bootstrap over
  32-64 episodes): (1) model choice changes mean rel 1-10 by <= 0.03 (ens128 vs vanilla128 +0.032, vs zoom128 n.s.) while
  the planner changes it by 0.08-0.25; (2) at equal evaluations per decision (1,280 or 320) horizon H=2/4 never beats H=1;
  the original H>1 deficit (-0.09 / -0.19) is mostly a terminal-only objective plus a search failure, not model exploitation;
  (3) fetch proposals with legalisation before scoring raise H=1 rel@10 by +0.077 at full push range; (4) an OT-seeded pool
  beats the fetch pool by +0.04-0.05 mean rel at H=2/4 (not at H=1), reproduced on 64 fresh w20 episodes and at w15 / w10.
provenance:
  commit: 13b0886f
  dirty: true
  script: "code/closed_loop.py (driver), wide_planner.py (models, SeqObjective, plan_seq, run_seq_episodes), queue*.sh, analyse.py, mass_analyse.py, fixes / ot_pool / captrans / layout analysers, diag_*.py, bs_*.py, gd_budget.py, intensive.py, regret_decomp.py, layout_regret.py, time_clean.py"
  data: ["DS-0006 starts 40-47, 48-63", "EXP-0067 vstar_width.json (normaliser)", "MODEL-0016..0025 (EXP-0074 checkpoints)"]
  data_commit: "DS-0006 as registered"
  code_path: "WideEns(members, balance=True) -> SeqObjective (terminal / cumulative / cumspill / captrans / layout) -> plan_seq (rank / CEM H-step, pool fl / ot / mix, optional GD, legal scoring) -> run_seq_episodes -> Genesis execution with legalize_pushes + legal_fallback"
  seed: "planner seed 0 (seeded per (seed, step, env)); one model instance per arm"
  split: "not applicable (control episodes)"
  runs: ["results/main_L20.json", "main_Lfull.json", "members_L20.json", "sweep_*.json", "fixes_*.json", "ot_pool/", "mass/", "captrans/", "layout/", "layout_lam/", "layout_regret/", "budget_study/", "gd_budget*", "intensive*", "timing_clean.json"]
  runtime: "2026-10-08 to 2026-10-12 (+ 2026-10-15 timing table); shared GPU for most cells"
budget:
  declared: "per sub-study in LOG.md (e.g. 2.5 h layout weight, 6 h mass comparison, 1.75 h w10)"
  spent: "several sub-studies stopped at their budget (LOG: layout_lam 7 of 10 cells, layout_regret 50 of 96 states, mass w15 second block not run)"
  outcome: stopped-early
design:
  varied:
    planner: "rank (best of N) vs CEM H=1/2/4; evaluations 256-5,120; pool bank / fetch (fl) / OT / mix; GD refinement; legal scoring"
    objective: "terminal lyapunov-minus-in-goal, cumulative, cumspill, captrans, Sinkhorn layout (lam 0-20)"
    push_range: "20 mm only vs 20-70 mm"
    model: "ens128 vs zoom128 vs vanilla128 (members_L20); vanilla64 / zoom64 / LF in the budget study"
  held_fixed:
    episodes: "10 pushes (16 in layout_lam), letters O/T/S/X, 32 paired episodes per cell (64 in the w20 mass comparison)"
    model: "ens128 (MODEL-0025) with mass balance for every planner cell outside the model comparisons"
  baselines: "rank (best of 1,280 single pushes, no refinement); vanilla CEM (base1F / base2F configs); push-0 state (do-nothing level of rel)"
  metric: "closed-loop rel (in-goal mass / placement optimum): mean rel over pushes 1-10, rel@10, done@0.9; skeleton_coverage_r5 (METRICS.md)"
noise_floor: "paired bootstrap 95 % CI over 32-64 episodes; one planner seed and one model instance (planner-seed sd in EXP-0066 0.0065-0.0068 rel)"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  (LOG.md entries cited by time.) Model (H=1, 20 mm, 1,280 evals, 22:58): mean rel ens .567 / zoom128 .558 / vanilla128
  .536; ens - vanilla +0.032 [.013, .050], ens - zoom +0.009 [-.004, .023]. Push range (00:25): full range - 20 mm +0.185
  [.149, .217]. Horizon (22:20): cem2 - cem1 -0.088, cem4 - cem1 -0.192, cem1 - rank +0.165; diagnosis (DIAGNOSIS.md): no code
  bug; terminal-objective procrastination, search failure (GD-refined open-loop plans reach -1.037 vs CEM -0.544 true value),
  optimism grows to ~10 % at 4 pushes. Fixes (2026-10-11): fetch + legal scoring rel@10 .894 -> .971 (+0.077 [+0.047, +0.114]);
  legal scoring alone +0.042 full range, null at 20 mm (P2 refuted there); cumulative objective H=2 cubes leaving 12.0 -> 1.7 and
  +0.169 mean rel, still -0.038 below H=1. OT pool (2026-10-12, w20 64 fresh episodes, 320 evals + GD): ot - fl H1 +0.002
  [-0.003, +0.007], H2 +0.053 [+0.040, +0.066], H4 +0.041 [+0.029, +0.055]; w15 H2 +0.046, w10 H2 +0.034. Model regret at
  visited states +0.010 vs proposal gap 0.090 value units (regret_base1F). Drawn-letter task: skeleton coverage stays 0.5-0.7;
  the Sinkhorn layout term raises it by +0.03..+0.07 (saturating by lam 2-6) at -0.02..-0.05 mean rel; capacity-transport
  objective null by construction.
verdict: supported
downgrades: [imprecision, inconsistency, provenance]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
Paired episodes (same goal x start, seeded planner) isolate one factor per comparison; the rank arm and the vanilla CEM arm
are the do-little baselines; the model comparison (members_L20) is run inside the same harness so model and planner effects
are on one scale.

## What was actually run
LOG.md is authoritative (newest at bottom), DIAGNOSIS.md for the horizon deficit, README.md for scope. Several sub-studies
carry pre-registrations written in LOG.md before their cells finished (2026-10-11 planner fixes P1-P3, OT pool, mass
comparison, layout term, layout weight, layout regret); they were not committed to git before running (dirty tree), so the
record is filed as exploratory. Uncommitted at run time: most of code/ after 13b0886f, simple_mpc/learned_mpc.py fallback /
planner_seed (ISS-014), Genesis/action_sampling.py::legalize_pushes_torch.

**Bug found and fixed (2026-10-11 03:40):** `wide_planner.SeqObjective._bind` kept the state encoding in the shared model
object, so `legal_fallback` ranked env k's bank on env 31's state; every fallback push (7-32 of 320 decisions per cell) in
cells before 2026-10-11 was a near-random legal bank push. Baselines were re-run with the fix and reproduce the recorded
cells (base1F vs cem1_Lfull rel@10 -0.015 [-0.046, +0.019]; base1_20 vs cem1_L20 +0.004).

## What would change the verdict
More planner seeds and a second model instance (all cells: one of each); legal starts other than DS-0006 40-63; the
drawn-letter task needs an objective not tested here (partial / unbalanced transport, LOG 16:45 reading).

## Threats
Shared GPU: plan times are indicative only (clean timing table T0a, results/timing_clean.md, supersedes them); `inconsistency`:
legal scoring helps at full range and not at 20 mm, OT helps at H>=2 only and at 320 not 1,280 evaluations; `provenance`:
some baselines come from earlier cells re-run after the fallback fix.

## Unrelated findings
- `legalize_pushes_torch` (batched GPU legaliser): identical decisions to the CPU legaliser on 1,280 pushes, 27 ms vs 3.3 s.
- Sample-and-rank shows a winner's-curse optimism that grows with N (-0.013 -> -0.026) without losing performance; CEM's
  chosen pushes are slightly pessimistic (LOG 00:25).
