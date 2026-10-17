---
id: EXP-0067
title: >
  Thin letter goals as a tool-geometry limit: completion vs stroke width (8 / 11 / 15 / 20 mm realised) for the broad
  NFD CEM in legal closed loop, with a perfect-model anchor
tier: T1
mode: confirmatory              # H1 / H2 and the analysis registered in DESIGN.md before any scored cell ran
date: 2026-10-05
hypothesis: null
claim: >
  In legal-action closed loop (DS-0006 scatter starts 40-71, letters O/T/S/X, 16 pushes of 20 mm, CEM pop 1024 /
  elite 0.25 / 0.5 s, TRAINING_PHYSICS, legalize + legal fallback, model nfd_3ch_randlen, planner seed 0), the
  completion rate at 0.9 x the per-goal placement optimum rises monotonically with the letter stroke width
  (realised 8.1-8.6 / 10.8 / 15.1 / 19.8 mm mean median-skeleton width), and the smallest width completing more than
  half the episodes lies between 10 and 20 mm nominal (2-4 cube widths, below the 40 mm blade).
prediction:
  supports: "H1: completion rate (max over k <= 16 of in-goal mass / optimum at that width >= 0.9) is non-decreasing over base -> w10 -> w15 -> w20 and the base-to-w20 difference's Newcombe 95 % CI excludes 0; H2: the first width with completion > 50 % is w10, w15 or w20"
  refutes: "H1: a decrease between successive widths outside the CI, or base-to-w20 CI includes 0; H2: completion < 50 % at every width (embarrassing: < 10 % even at w20 -> not a geometry limit; planner/objective at fault) or > 50 % already at the base width"
  discriminating: true
provenance:
  commit: a5ccb680
  dirty: true                    # EXP-0066's uncommitted learned_mpc / driver changes plus this record's goals.py / driver / oracle.py edits (see body)
  data_commit: "DS-0006 config provenance"
  script: code/queue.sh (driver experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py; anchor experiments/EXP-0050-oracle-ceiling/code/oracle.py), code/calibrate_width.py, code/vstar_width.py, code/analyse.py
  data: ["DS-0006 states 40-71 (starts)", "env/target_shapes/helvetica_thin/*.npy (720 px glyphs)", "EXP-0046 results/vstar.json (base-letter optimum)", "EXP-0066 results/broad_s0_ps0.json (base-width cell, reused)", "EXP-0057 results/oracle_A_default.json (base-width oracle anchor, reused)"]
  code_path: >
    Baselines.common.goals.widened_letter_mask (chamfer-field dilation before INTER_AREA resize) -> letter_goal_mask;
    simple_mpc.learned_mpc.run_episodes_batched (plan cem; legalize + legal_fallback; planner_seed 0) -> GenesisOracleEnv
    with TRAINING_PHYSICS; score from recorded states via occ_for_scoring + mass_in_region / vstar_width.json optimum
  seed: "planner seed 0; model nfd_3ch_randlen training seed 0; oracle seed-base 0"
  split: "not applicable"
  runtime: "learned sweep 3 x 28-29 min GPU (RTX 4070, 09:08-10:33 CEST) + 2 min pilot; oracle anchors 1.6 h + 1.9 h GPU (10:33-14:03 CEST); CPU: calibration + vstar < 2 min"
  runs: [RUN-0001]
budget:
  declared: "7 h wall, <= 5 GPU-h (hard; report partial rather than overrun)"
  spent: "5.1 h wall of 7 (08:58-14:50 CEST incl. the rate-limit gap); 4.95 GPU-h of 5 (sweep 1.43 h, oracle anchors 3.49 h, pilot 0.03 h)"
  outcome: within
design:
  varied:
    stroke_width: "base (d = 0; realised 8.1-8.6 mm) / w10 (d = 6; 10.8) / w15 (d = 18; 15.1) / w20 (d = 31; 19.8), nominal mm -> asset-px dilation per LETTER_WIDTH_DILATE_PX"
    goal: "letter_O, letter_T, letter_S, letter_X at each width"
    start: "DS-0006 states 40-71"
  held_fixed:
    model: "nfd_3ch_randlen (broad NFD; EXP-0066's best context cell), planner seed 0 -- one model, one seed: chosen before running, not swept"
    planner: "cem, pop 1024, elite 0.25, 0.5 s per decision, n_cand 1024, bank 16384 per env"
    episode: "16 pushes of 20 mm, legalize + legal fallback on, TRAINING_PHYSICS executor, 32 envs per chunk"
    n_episodes: "128 per width (4 letters x 32 starts; fixed from the cost pilot before running)"
    anchor: "oracle CEM 64 x 4, 256 sims per push, n_envs 128, stop-solved, 16 pushes, O and T, starts 40-43, widths 15 (A1) then 10 (A2) subject to the DESIGN.md budget rule; base width from EXP-0057 (starts 40, 41)"
  baselines: "base-width cell (EXP-0066 broad_s0_ps0, identical settings) = the thin letters as they are; do-nothing = the start's own in-goal mass (rel_0, reported); optimum = 1 at each width (placement optimum recomputed per width); perfect-model anchor where run"
  metric: "mass_in_region (in-goal mass fraction from occ_for_scoring states) / mass_frac_best_placement at that width; completion = max_k<=16 rel_k >= 0.9; also rel at k = 8, 16 (median, q05, q95), pushes-to-completion censored at 16, fallback rate, executed-illegal count"
noise_floor: >
  EXP-0066: planner-seed replicate sd of a 128-episode cell mean 0.0065 (score), per-episode score sd ~0.11; the two
  broad training seeds gave cell means 0.364 / 0.390. A completion-rate difference between two 128-episode cells
  needs ~0.12 to clear a Newcombe 95 % CI near 50 % and ~0.05 near 0-10 %. Same 32 starts in every cell.
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: "completion @0.9 x opt: base 0/128, w10 1/128, w15 20/128 (15.6 %), w20 40/128 (31.2 %); monotone, base->w20 +0.312 [+0.233, +0.397]; rel16 median 0.48 -> 0.57 -> 0.72 -> 0.77; per letter at w20 O 56 %, S 41 %, X 25 %, T 3 %; no width <= 20 mm exceeds 50 % pooled (H2 refuted); executed illegal 0; oracle anchor (perfect-model CEM, O and T, 4 starts) w15 3/8 complete (all T, pushes 7-12; O 0/4), rel16 mean 0.87; w10 1/8, rel16 0.81; base 0/4 (EXP-0057); learned model still climbing 0.013-0.031 rel/push over pushes 12-16 at every width"
verdict: supported
downgrades: [inconsistency]
grade: moderate
---

## Why this test discriminates
Every earlier closed-loop record scored thin letters and found no completion, for learned models and for the
simulator used as its own model; none varied the goal geometry. Holding the model, planner, objective, starts and
legality machinery fixed and widening only the stroke separates "the planner cannot shape a pile into a thin
stroke" from "the planner cannot do letters". The placement optimum is recomputed per width so a wider goal is not
credited for a looser denominator (it rises from 0.94-0.97 at the base width to 1.000 at 20 mm).

## What was actually run
Pre-registration `DESIGN.md` (09:10 CEST) after a mask calibration, the per-width placement optimum and a 2-push cost
pilot (timing only). Learned sweep `code/queue.sh`: cells w10 (09:08-09:36), w15 (09:36-10:04), w20 (10:04-10:33),
each 4 letters x 32 starts = 128 episodes, every episode complete; base width reused from EXP-0066 `broad_s0_ps0`
(same model, planner seed, starts, settings; the only config difference is the driver's later `record_quats` flag,
absent vs False, recording-only). Oracle anchors: oracle_w15 (10:33-12:11, 8 tasks, 16 pushes, 5837 s of planning) and oracle_w10 (12:11-14:03, 6684 s), both complete; every DESIGN.md cell ran. A first queue launch at 09:05 failed in every cell
within 30 s (a shell `local` one-liner expanded the width before assigning it, goal names `letter_O_w`); the
fail-status ledger lines at 09:06-09:08 are that launch, no outcome was produced. Uncommitted at run time: EXP-0066's
`simple_mpc/learned_mpc.py` / `simple_mpc/adapters.py` / driver flags, plus this record's `Baselines/common/goals.py`
(`widened_letter_mask`, `parse_letter_goal`, `letter_goal_mask`, `LETTER_WIDTH_DILATE_PX`), the EXP-0043 driver and
EXP-0050 `oracle.py` (`goal_mask` -> `letter_goal_mask`; `oracle.py --vstar-files`). Goal-convention tests pass.

## Numbers (`results/analysis.{json,md}`; rel = in-goal mass / optimum at that width)
| width (d asset px) | realised median stroke mm O/T/S/X | optimum O/T/S/X | completion @0.9 x opt [Wilson 95 %] | @0.8 | rel8 median | rel16 median [q05, q95] | score 4-16 | rel0 (start) median | fallback | illegal |
|---|---|---|---|---|---|---|---|---|---|---|
| base (0) | 8.1/8.1/8.1/8.6 | 0.956/0.965/0.938/0.949 | 0/128 = 0.000 [0, 0.029] | 0.008 | 0.315 | 0.480 [0.281, 0.715] | 0.364 | 0.081 | 0.054 | 0 |
| w10 (6) | 10.3/12.2/9.1/11.5 | 0.990/0.985/0.986/0.989 | 1/128 = 0.008 [0.001, 0.043] | 0.062 | 0.358 | 0.574 [0.360, 0.796] | 0.419 | 0.098 | 0.064 | 0 |
| w15 (18) | 14.7/16.3/14.7/14.7 | 1.000/0.999/1.000/1.000 | 20/128 = 0.156 [0.103, 0.229] | 0.344 | 0.469 | 0.720 [0.447, 0.926] | 0.535 | 0.149 | 0.053 | 0 |
| w20 (31) | 20.3/20.3/18.2/20.3 | 1.000 x 4 | 40/128 = 0.312 [0.239, 0.397] | 0.500 | 0.575 | 0.771 [0.492, 0.969] | 0.613 | 0.204 | 0.089 | 0 |

Per letter, completion @0.9 (n = 32 each) / rel16 median: O 0 / 0.59 -> 0.03 / 0.68 -> 0.31 / 0.79 -> **0.56 [0.39, 0.72]** / 0.90;
S 0 / 0.52 -> 0 / 0.54 -> 0.06 / 0.71 -> 0.41 / 0.83; X 0 / 0.50 -> 0 / 0.58 -> 0.22 / 0.77 -> 0.25 / 0.77;
**T 0 / 0.41 -> 0 / 0.51 -> 0.03 / 0.60 -> 0.03 / 0.67**. Pushes to completion: median > 16 in every pooled cell
(cumulative completion at k = 8 / 12 / 16: w15 0 / 0.03 / 0.16; w20 0.02 / 0.14 / 0.31); letter O at w20 reaches 50 %
at push 14. Executed-illegal 0 in all 8192 sweep pushes (+ 2048 base).
- **H1 supported:** successive differences base->w10 +0.008 [-0.022, +0.043], w10->w15 +0.148 [+0.085, +0.222],
  w15->w20 +0.156 [+0.053, +0.256]; monotone; base->w20 +0.312 [+0.233, +0.397] (Newcombe).
- **H2 refuted:** no width <= 20 mm nominal (19.8 realised) completes > 50 % pooled; only O does (56 % at w20).
  Neither the embarrassing (< 10 % at w20) nor the opposite (w10 > 50 %) outcome occurred.
- Oracle anchor (perfect-model CEM 64 x 4, 256 sims/push, O and T): base width (EXP-0057, starts 40/41, read at k = 16)
  0 / 4 complete, rel16 0.73 / 0.78 / 0.71 / 0.78. w15 (starts 40-43): **3 / 8 complete, all letter T** (T/40 done at push 12, T/42 at 7, T/43 at 10; T/41 0.84), O 0 / 4 (rel16 0.76 / 0.85 / 0.89 / 0.81), rel8 mean 0.79, rel16 mean 0.87. w10: 1 / 8 (T/42 at push 14), rel16 mean 0.81 (O 0.71-0.87, T 0.83-0.91). Per-letter rel slope of the learned model (`results/slopes_k8_k16.json`): mean (rel16 - rel8) / 8 = 0.019-0.037 per push at every width and letter, and over pushes 12-16 still 0.013-0.031 per push with 69-84 % of episodes still rising -- the broad model is still climbing at k = 16 everywhere, so completion within 16 pushes is horizon-censored.

## Reading
Stroke width is a real, monotone limit for this planner: completion goes from 0 of 128 at the native ~8 mm strokes
to 40 of 128 at ~20 mm, with half the episodes reaching 0.8 x optimum. But it is not the whole story. A 20 mm stroke
is four cube widths and half the blade, the optimum is 1.0 and the mask holds > 80 cubes, yet pooled completion is
31 % and the median episode still ends at 0.77 x optimum after 16 pushes -- and letter T does not move at all
(1 of 32 at every width >= 15 mm; T's rel16 median rises 0.41 -> 0.67 but never nears 0.9; WHY is not isolated
here -- a stroke-junction / mass-split explanation is an untested hypothesis, checkable from the recorded states). The 10 mm step, which is 1-2 px on the 64 px grid (S: +1 px), is inside the noise of the
completion rate (+0.8 %) though it lifts rel16 by ~0.09; the first resolvable gain needs ~15 mm (3 cube widths).
Reading for the goal library: widening the letters to 15-20 mm turns "never completed" into "completed in 16-31 %
of episodes within 16 pushes", with O/S the letters that respond and T the one that does not; a letter set at
15 mm gives a non-degenerate completion metric where the current one reads 0 for every model.
The perfect-model anchor confirms the direction (base 0 / 4 -> w10 1 / 8 -> w15 3 / 8 on O and T) and overturns the per-letter reading: the oracle completes T at 15 mm in 3 of 4 starts (by push 7-12) and never completes O, the learned model the reverse (O 31 %, T 3 %). So T is not a geometry limit at 15 mm; T's flatness is a property of the learned model / its objective, and O's responsiveness is partly that the learned planner fills a ring more easily than a stroke junction. With 4 starts per letter this is a direction, not an estimate.

## What would change the verdict
A planner-seed or training-seed replicate reversing one of the successive steps (planner-seed sd of a cell mean was
0.0065 in EXP-0066; the 0.15 steps here are 20x that, the 0.008 step is not resolved); a 20-push horizon (completion is
still rising at push 16: cumulative 0.14 -> 0.31 over pushes 12-16 at w20); other models (one model, one seed here);
clump starts. Cells not run: none (all 5 queue cells ran; 8-start oracle cells were pre-emptively reduced to 4 starts in DESIGN.md).

## Threats
- One model (nfd_3ch_randlen, seed 0) and one planner seed: the width effect is for this planner; the per-letter
  pattern (T flat) could be model-specific. The oracle anchor is the only cross-check and has 4 tasks per cell.
- The nominal widths are quantised by the 64 px grid (2.03 mm/px): "10 mm" is 9.1-12.2 mm by letter, so the w10 cell
  is heterogeneous (S barely widened, T widened to 12 mm); the realised widths are the ones to cite.
- Reused base cell: same settings and planner seed, but env-slot order within chunks is the same by construction
  (4 goals x 32 starts in the same order), so the base cell is as replicable as any other cell.
- Wider masks start with more mass inside (rel0 median 0.08 -> 0.20), so part of the rel gain is the start, not the
  pushing; the completion metric at 0.9 is far above every start (max rel0 < 0.5 in all cells).
- `dirty: true` (user rule: do not commit); every uncommitted file is listed above.

## Unrelated findings
- The glyph assets `env/target_shapes/helvetica_thin/*.npy` are chamfer distance fields (cv2 DIST_L2 5x5), not
  binary images; `letter_mask` thresholds the INTER_AREA-resized field at 0.5. Binarising before the resize gives a
  different (thicker, 389 vs 340 px for O) mask -- any future goal-generation code must dilate in the field.
- The realised base stroke is 8.1-8.6 mm (4 px) by the EXP-0046 skeleton measure, not the 7.5 mm quoted in
  earlier records.
- The legal fallback fires more often at w20 (8.9 % of steps) than at the other widths (5.3-6.4 %).
