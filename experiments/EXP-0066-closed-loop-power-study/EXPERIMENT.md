---
id: EXP-0066
title: >
  Closed-loop power study: narrow v2 NFD (3 training seeds) beats linear narrow v2 res64 (3 instances) by +0.064
  [+0.045, +0.083] on 128 paired legal episodes, resolved with seeds fixed and random; training-seed spread (0.036)
  is 5x the seeded planner spread (0.0065); the gap is goal-specific (O, S yes; T, X no) and sits at the
  random-effects detection limit (MDG 0.063)
tier: T1
mode: confirmatory              # H1 / H1h / H2 / MDG registered in DESIGN.md before any scored cell ran
date: 2026-10-04
hypothesis: null
claim: >
  In legal-action closed loop (DS-0006 scatter starts 40-71, letters O/T/S/X, 16 pushes of 20 mm, CEM pop 1024 /
  elite 0.25 / 0.5 s, TRAINING_PHYSICS, legalize + legal fallback, score = in-goal mass over pushes 4-16 / optimum),
  the narrow v2 NFD family (3 training seeds) beats the linear narrow v2 res64 family (full-data fit + 2 data-bootstrap
  refits) by a gap whose paired-over-episodes 95 % bootstrap CI excludes 0 (H1), and that gap is larger than both the
  training-seed spread and the planner-seed spread (H2); the design's power is stated as the minimal detectable gap.
prediction:
  supports: "H1: paired bootstrap 95 % CI of G (NFD 3-seed mean - linear 3-instance mean, 128 paired episodes) excludes 0; H2: sd(training seeds) > sd(planner seeds) and both < |G|"
  refutes: "H1: CI includes 0; H2: planner-seed sd >= training-seed sd, or either sd >= |G|. Embarrassing: the hierarchical CI (H1h, seeds resampled) includes 0 -- the family gap sits inside the seed spread at 32 starts x 3 seeds"
  discriminating: true
provenance:
  commit: a5ccb680
  dirty: true                    # legal fallback + planner seeding + bootstrap refit flag uncommitted at run time (see body)
  data_commit: "DS-0006 config provenance; DS-0015 for the linear refits"
  script: code/queue.sh (driver experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py), code/analyse.py
  data: ["DS-0006 states 40-71 (starts)", "DS-0015 (linear bootstrap refits)", "EXP-0046 results/vstar.json (per-letter optimum)"]
  code_path: >
    simple_mpc.learned_mpc.run_episodes_batched (plan cem; legalize=Genesis.action_sampling.legalize_pushes;
    legalize_fallback=True -> learned_mpc.legal_fallback; planner_seed set) -> GenesisOracleEnv executor with
    TRAINING_PHYSICS; score from recorded states via occ_for_scoring + mass_in_region
  seed: "planner seeds 0 / 1 / 2 (torch global RNG per (seed, step, env)); NFD training seeds 0 / 1 / 2; linear bootstrap seeds 1 / 2"
  split: "not applicable"
  runtime: "5.3 h GPU (12 cells x 24-28 min, RTX 4070; 20:39-01:56 CEST) + 2 x 41 s linear refits + 2 min pilot"
  runs: [RUN-0001]
budget:
  declared: "9 h wall, <= 7.5 GPU-h (hard; report partial rather than overrun)"
  spent: "5.5 h wall of a 9 h budget; 5.3 GPU-h of 7.5 (all 12 cells incl. the 2 optional ones ran)"
  outcome: within
design:
  varied:
    family: "narrow v2 NFD (nfd_3ch_narrow_l20_v2, _seed1, _seed2) vs linear narrow v2 res64 (full fit, boot1, boot2)"
    planner_seed: "0 / 1 / 2 for nfd seed 0 and for the full-fit linear; 0 elsewhere"
    goal: "letter_O, letter_T, letter_S, letter_X"
    start: "DS-0006 states 40-71"
  held_fixed:
    planner: "cem, pop 1024, elite 0.25, 0.5 s per decision, n_cand 1024, bank 16384 per env"
    episode: "16 pushes of exactly 20 mm (project_push; wall clipping to >= ~7 mm as in RUN-0002), legalize + legal fallback on"
    executor: "GenesisOracleEnv with TRAINING_PHYSICS, 32 envs per chunk"
    n_episodes: "128 per cell (fixed from the cost pilot before running)"
  baselines: "the seed / instance replicate cells (closed-loop seed floor), the planner-seed replicate cells (planner floor); do-nothing = 0 pushes gives the start's own in-goal mass (score ~ the start level; reported as the push-0 value); optimum = 1"
  metric: "mass_in_region (in-goal mass fraction from occ_for_scoring states, mean over pushes 4-16, divided by the goal's mass_frac_best_placement; EXP-0039 RUN-0002 / EXP-0051 formula)"
noise_floor: >
  Pre-run, from EXP-0039 RUN-0002 on these 4 goals: per-model episode sd 0.10-0.12; paired sd of (NFD 3-seed mean minus
  linear) 0.118 at n = 64; NFD seed sd 0.033 over 3 cell means. Seeds fixed: SE(G) ~ 0.009 at n = 128 (MDG ~ 0.025);
  seeds random: SE(G) ~ 0.028 (MDG ~ 0.08). RUN-0002 gap on these goals +0.041.
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: "G = +0.064 [+0.045, +0.083] (paired, p < 1e-4), hierarchical [+0.024, +0.106] (p 0.001); sd training seed 0.036 (0.56 G), linear instance 0.015 (0.24 G), planner seed 0.0065 / 0.0068 (0.10 G); MDG 0.027 seeds fixed / 0.063 seeds random; per goal O +0.094, S +0.152, T -0.006, X +0.016; executed illegal 0 / 24,576"
verdict: supported
downgrades: [inconsistency, indirectness]
grade: low
---

## Why this test discriminates
RUN-0002 compared 8 models on 8 goals x 16 starts with ONE planner run and one checkpoint per recipe (3 seeds for one
recipe only); the narrow NFD vs linear res64 pair was resolved for 2 of 3 seeds. Here the pair gets 3 instances per
family, planner replication under a seeded planner (new), 2x the starts, and the gap is read against pre-stated
seed, instance and planner spreads with a pre-stated minimal detectable gap. The outcome that embarrasses the design
(H1h CI includes 0) is as informative as the one that supports it: it bounds what closed-loop comparison can resolve.

## What was actually run
Exactly DESIGN.md's 12 cells in its order (10 primary + 2 optional), 2026-10-04 20:39 -> 2026-10-05 01:56 CEST, every
episode complete (12 x 128 = 1536 episodes, 24,576 pushes), one GPU job at a time; `results/<tag>.json`, logs and pid
files under `logs/`, ledger `EXP-0066/RUN-0001/<tag>` in `experiments/COMMANDS.jsonl`. Cost pilot (no outcome read)
before the pre-registration. Uncommitted at run time (dirty): `simple_mpc/learned_mpc.py` (legal_fallback, legalize_fallback, planner_seed),
`experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py` (`--legalize-fallback`, `--planner-seed`,
legalize(envs=...)), `Baselines/LinearForesight/fit_switched.py` (`--bootstrap-seed`), `simple_mpc/adapters.py`
(boot1/boot2 registry entries), `tests/test_learned_mpc.py` (2 new tests), docs. Pre-registration: `DESIGN.md`.

## Numbers (`results/analysis.{json,md}`)
| cell | mean [CI] | median [q05, q95] | fallback rate |
|---|---|---|---|
| nfd seed 0 / 1 / 2 (ps 0) | 0.311 [0.292, 0.329] / 0.240 [0.222, 0.258] / 0.267 [0.248, 0.285] | 0.317 [0.10, 0.49] / 0.224 [0.09, 0.43] / 0.273 [0.09, 0.44] | 0.026 / 0.039 / 0.059 |
| linear full / boot1 / boot2 (ps 0) | 0.225 [0.204, 0.247] / 0.197 [0.179, 0.214] / 0.203 [0.185, 0.222] | 0.214 [0.06, 0.45] / 0.181 [0.06, 0.37] / 0.175 [0.07, 0.40] | 0.013 / 0.004 / 0.005 |
| nfd seed 0, planner seed 1 / 2 | 0.305 [0.287, 0.324] / 0.318 [0.300, 0.337] | 0.293 / 0.317 | 0.027 / 0.034 |
| linear full, planner seed 1 / 2 | 0.239 [0.218, 0.260] / 0.234 [0.213, 0.257] | 0.216 / 0.214 | 0.012 / 0.020 |
| broad nfd_3ch_randlen seed 0 / 1 (context) | 0.364 [0.347, 0.382] / 0.390 [0.373, 0.406] | 0.359 / 0.391 | 0.054 / 0.055 |

- **H1 supported:** G = +0.0639 [+0.0446, +0.0827], paired sd 0.110, SE 0.0098 (128 (goal, start) episodes; p < 1e-4).
  Full-fit linear only: +0.047 [+0.025, +0.069]. **H1h (seeds + instances resampled): [+0.024, +0.106], p 0.001** --
  the embarrassing outcome did not occur, but see MDG.
- **H2 supported:** training-seed sd 0.0358 (0.56 x G) > planner-seed sd 0.0065 NFD / 0.0068 linear (0.10 x G);
  linear data-bootstrap instance sd 0.0151 (0.24 x G). Planner replicate pairs: none resolved for NFD
  (+0.005, -0.008, -0.013), one nominal for linear (-0.013 [-0.025, -0.002]); seed pairs ALL resolved (s0 > s2 > s1).
- **MDG at this n** (80 % power, 2.8 SE): 0.027 with seeds fixed; 0.063 with seeds random (SE 0.0224 from the measured
  seed / instance sds over 3 + 3 instances). The observed gap equals the random-effects MDG.
- **Per goal** (NFD 3-mean minus linear 3-mean): O +0.094 [+0.053, +0.133], S +0.152 [+0.123, +0.178],
  T -0.006 [-0.035, +0.022], X +0.016 [-0.009, +0.041].
- **Variance (1280 primary episodes, sequential SS):** goal 7.8 %, start-within-goal 36.7 %, family 8.1 %,
  instance-within-family 4.0 %, planner seed 0.1 %, residual 43.2 %.
- **Legality:** executed-illegal 0 in every cell (24,576 pushes); legal fallback fired in 0.4-0.5 % of steps for the
  bootstrap linears, 1.2-2.0 % full-fit linear, 2.6-5.9 % narrow NFD seeds, 5.4-5.5 % broad NFD. Legalisation shift
  rate (planned push slid) 0.34-0.38 NFD narrow, 0.50-0.58 linear, 0.58-0.67 broad NFD (as in RUN-0002).
- Holm over the 15 ps-0 pairs: 13 resolved; unresolved: nfd seed 1 vs linear full (+0.014 [-0.010, +0.039]) and
  boot1 vs boot2.

## Reading
The pair RUN-0002 left marginal is resolved at this budget, in both the fixed- and random-seed readings, and planner
replication is a real factor only in the sense that it is now measurably small: a seeded CEM run differs from
another by ~0.007 in cell mean, 5x less than retraining the NFD. The family gap is NOT uniform: on T and X the
families tie, on O and S the NFD wins by 0.09-0.15, so the family claim is a 2-of-4-letters claim. The weakest NFD
seed (0.240) is statistically tied with the full-fit linear (0.225): one seed per family, as most records here have,
cannot resolve this pair. With seeds random the minimal detectable gap (0.063) equals the observed gap; a gap half
this size between two families would need ~4x the seeds or would be unresolvable at feasible budgets. Start identity
explains 37 % of episode variance vs 8 % for the family: pairing by start is what makes any of this resolvable.

## What would change the verdict
A 4th / 5th NFD seed landing at or below 0.225 (the H1h CI's lower edge is +0.024); a different planner (rank) or
budget reversing the pair (RUN-0002 saw planner-dependent winners); scoring T and X alone (gap ~0). Cells not run:
none of the design; not designed: rank arm, clump starts, more than 3 seeds (~27 min GPU per cell).

## Threats
- Linear bootstrap replicates are weaker fits than the full-data operator (held-out accuracy 0.280 / 0.279 vs 0.332,
  known before running): the linear "instance spread" carries a downward bias; the full-fit-only comparison
  (lin_orig vs the 3 NFD seeds) is reported beside the family version.
- Planner seeding is global-RNG seeding per (seed, step, env); any other consumer of the global RNG inside a step
  (none known in the CEM path) would share the stream but not break replication.
- `dirty: true`: the code fix needed for "never execute an illegal push" was not committed before the run (user: do
  not commit); every change is listed above and in the diff.

## Unrelated findings
- Data-bootstrap refits of the ridge linear operator are systematically WORSE than the full-data fit (held-out
  accuracy 0.280 / 0.279 vs 0.332; closed loop 0.197 / 0.203 vs 0.225, both resolved): a bootstrap sample has
  ~63 % distinct rows, and the ridge-toward-identity recipe loses accordingly. Bootstrap is not a seed proxy here.
- The broad nfd_3ch_randlen seeds (0.364 / 0.390) again lead every narrow model on scatter starts (RUN-0002,
  RUN-0003), and the broad seed-1 retrain beats seed 0 by +0.026 [resolved by CI]; broad models use the legalisation
  shift most (0.58-0.67) and the fallback most (5.4-5.5 %).
- RUN-0002's unfixed path executed 15-121 illegal pushes per 2048-step model cell; here the fallback replaced
  0.4-5.9 % of pushes per model and no cell executed an illegal push.

## Scope note (2026-10-05 policy)

Both families compared here (nfd_3ch_narrow_l20_v2 seeds; linear_narrow_l20_v2_res64 and its
bootstrap refits) are narrow-corpus pilot models, and the starts are the n20 scatter pilot
population. Under the standing policy (`experiments/README.md`, "Standing policies") C-075 is
pilot-grade: the resolved family gap, the three spreads and the minimal-detectable-gap figures
are the design calibration this record was run for, and they carry over; the model ordering
itself is a project conclusion only once replicated with broad-trained families (randlen seeds
vs a broad linear fit) on the same harness. The context cells (broad randlen seeds above every
narrow model) point the same way and are the natural first cell of that replication.
