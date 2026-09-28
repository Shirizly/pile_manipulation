---
id: EXP-0046
title: >
  Goal ceilings are ~0 for every many-set goal (V* <= 0.006 vs V0 ~ 0.3), so closed-loop runs
  already reach 0.80-0.96 of the achievable lyapunov improvement; the lyapunov optimum keeps only
  82-97% of mass inside letter masks under the soft splat; the 30 goals collapse to ~12 ranking-distinct ones
tier: T1
mode: exploratory
date: 2026-09-25
hypothesis: null
claim: >
  For 20 single-layer non-overlapping 5 mm cubes scored with occ_for_scoring (sigma 1 px) on the
  64 px / +-64 mm grid, the best flat placement reaches lyapunov V*(g) <= 0.01 for all 30 `many`
  goals and `two_squares`; at that optimum mass_in_region is < 100% of total mass for every letter
  (soft-splat spill across ~8 mm strokes), and closed-loop EXP-0044/0045 episodes (DS-0006 starts
  40-47) reach achieved_fraction >= 0.8 after 8 pushes at budgets >= 0.1 s.
provenance:
  commit: 3bae8cd7
  dirty: true
  data_commit: not applicable
  script: experiments/EXP-0046-goal-ceiling/code/vstar.py, code/analyse.py
  data: ["DS-0006 step0 (slate_idx, start states)", "EXP-0044 results/goal_breadth_tuned.json", "EXP-0045 results/time_vs_actions.json", "EXP-0030 artifacts/RUN-0001 truth.pt, pred_*.pt"]
  code_path: "per-cube separable Gaussian cost map (== occ_for_scoring + learned_mpc.lyap, checked to 1e-9) + greedy/lattice/coordinate-descent packing"
  seed: 0
  split: "not applicable"
  runtime: "~30 s CPU"
  runs: [RUN-0001, RUN-0002]
budget:
  declared: "45 min wall clock, CPU only"
  spent: "~35 min"
  outcome: within
design:
  varied: {goal: "30 many goals + two_squares"}
  held_fixed: {n_cubes: 20, cube: "5 mm, single layer z = 12.5 mm", scoring: "occ_for_scoring sigma 1 px", grid: "64 px, +-64 mm"}
  baselines: ["do-nothing (V0, achieved_fraction 0)", "theoretical max mass = 20 x pi r^2 (all mass inside)"]
  metric: "achieved_fraction (experiments/METRICS.md); lyapunov; mass_in_region / total mass"
noise_floor: "packer is heuristic: 14 restarts agree to <= 0.0002 lyapunov; EXP-0043 closed-loop repeat sd 0.01-0.04 at k = 8"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  V* = 0 (quadrants), 0.0001 (two_squares), 0.0010-0.0025 (letters) except letter_I 0.0063
  (capacity 13 < 20). Mass inside at the lyapunov optimum: letters mean 0.946 (0.817 I - 0.966),
  signed 0.893 (0.634 - 0.933); quadrants 1.000; the mass-maximising placement is no better
  (<= 0.001). achieved_fraction: EXP-0044 k=8 0.880; EXP-0045 k=8 0.802, k=24 0.865 (b >= 0.1 s:
  0.87-0.92 at k=8, 0.94-0.96 at k=24; b = 0.03 s 0.50 / 0.60). Push-ranking clustering gives 12
  groups; 18 goals are droppable.
verdict: supported
downgrades: [indirectness]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
If goals were hard to reach even in principle (V* comparable to V0), the EXP-0045 plateau at
~0.31 would be a ceiling rather than a planner/model shortfall; V* ~ 0 separates the two. If the
lyapunov optimum held all mass inside the mask, the three value functions would agree at the
optimum; the splat spill fraction measures how far they differ.

## What was actually run
Dirty tree: uncommitted at run time were this experiment's own edits (goals.py `two_squares_mask`,
eval_report.py `many_plus`/`EXTRA_GOALS`/`_fixed_goal` branch, tests/test_goals_two_squares.py, code/) plus
other agents' unrelated working-tree changes listed in git status (none in the scoring path used here).
RUN-0001 (`code/vstar.py`): lyapunov under occ_for_scoring is the mean of a per-cube cost map
c(x,y) = g(x)^T D g(y) (verified against the real occ_for_scoring + lyap, max diff 9e-10). 20 centres
chosen on a 0.5 mm grid inside the tray, pairwise Chebyshev spacing >= 5 mm (axis-aligned, no
overlap), by greedy + best of 100 lattice offsets + 12 noisy-greedy restarts, each followed by
coordinate descent. Capacity = max cubes whose footprint lies fully in mask pixels (lattice/greedy
lower bound). Stroke width = 2 x max internal EDT (tray edge counted as outside). Same packer used
to MAXIMISE mass_in_region, to test whether the lyapunov optimum is also the mass optimum.
RUN-0002 (`code/analyse.py`): achieved_fraction per episode for EXP-0044/0045 (V0 check: start 40 /
quadrant_0 recomputed from DS-0006 = 0.29389, identical to the episode's values[0]); goal
redundancy from EXP-0030 truth/pred lyapunov dv (160 states x 128 pushes): per-state Spearman of
true dv between goals (push-ranking agreement), mask IoU, distance-field correlation, per-goal
slateN of 8 models and model-ranking Kendall; average-linkage clustering on 1 - push-rank corr.
New goal `two_squares` (Baselines/common/goals.py::two_squares_mask; two 12x12 px squares at
world x ~ +-32 mm, y = 0; opt-in via eval_report goal set `many_plus`).

## Numbers
Cube 5 mm (diagonal 7.07 mm); pixel pitch 2.03 mm. Letter strokes (median skeleton width) 8.1 mm.

| goal | V* | capacity | cubes fully inside at V* | max stroke width mm | mass in / total at V* | signed / total at V* |
|---|---|---|---|---|---|---|
| letter_A | 0.0017 | 27 | 11 | 12.2 | 0.946 | 0.893 |
| letter_B | 0.0019 | 38 | 17 | 9.1 | 0.948 | 0.895 |
| letter_C | 0.0021 | 25 | 10 | 9.1 | 0.941 | 0.882 |
| letter_D | 0.0013 | 35 | 19 | 11.5 | 0.963 | 0.927 |
| letter_E | 0.0015 | 36 | 20 | 11.5 | 0.962 | 0.925 |
| letter_F | 0.0010 | 27 | 20 | 11.5 | 0.963 | 0.926 |
| letter_G | 0.0017 | 33 | 15 | 12.2 | 0.953 | 0.905 |
| letter_H | 0.0014 | 33 | 20 | 11.5 | 0.966 | 0.932 |
| letter_I | 0.0063 | 13 | 4 | 8.1 | 0.817 | 0.634 |
| letter_J | 0.0023 | 18 | 16 | 9.1 | 0.918 | 0.837 |
| letter_K | 0.0023 | 28 | 16 | 12.2 | 0.943 | 0.886 |
| letter_L | 0.0022 | 20 | 18 | 9.1 | 0.918 | 0.837 |
| letter_M | 0.0019 | 47 | 16 | 12.2 | 0.957 | 0.914 |
| letter_N | 0.0015 | 36 | 18 | 12.8 | 0.962 | 0.925 |
| letter_O | 0.0017 | 32 | 16 | 9.1 | 0.955 | 0.910 |
| letter_P | 0.0010 | 30 | 19 | 11.5 | 0.965 | 0.929 |
| letter_Q | 0.0015 | 33 | 19 | 12.2 | 0.960 | 0.920 |
| letter_R | 0.0012 | 36 | 20 | 11.5 | 0.966 | 0.933 |
| letter_S | 0.0022 | 30 | 13 | 9.1 | 0.938 | 0.876 |
| letter_T | 0.0011 | 22 | 20 | 11.5 | 0.965 | 0.930 |
| letter_U | 0.0014 | 28 | 20 | 9.1 | 0.961 | 0.921 |
| letter_V | 0.0018 | 22 | 12 | 12.2 | 0.941 | 0.882 |
| letter_W | 0.0016 | 41 | 14 | 12.2 | 0.960 | 0.920 |
| letter_X | 0.0022 | 21 | 16 | 12.2 | 0.949 | 0.897 |
| letter_Y | 0.0021 | 16 | 15 | 11.5 | 0.932 | 0.865 |
| letter_Z | 0.0019 | 26 | 15 | 12.2 | 0.953 | 0.906 |
| quadrant_0 | 0.0000 | 144 | 20 | 65.0 | 1.000 | 1.000 |
| quadrant_1 | 0.0000 | 144 | 20 | 65.0 | 1.000 | 1.000 |
| quadrant_2 | 0.0000 | 144 | 20 | 65.0 | 1.000 | 1.000 |
| quadrant_3 | 0.0000 | 144 | 20 | 65.0 | 1.000 | 1.000 |
| two_squares | 0.0001 | 32 | 20 | 24.4 | 0.998 | 0.995 |

The lyapunov optimum does not need cubes fully inside (it minimises distance, which is 0 anywhere
in the mask); e.g. letter_A holds only 11 of 20 fully inside although capacity is 27.

achieved_fraction (mean over episodes; V0 ~ 0.33, V* ~ 0.0015):

| source | n | impr k=8 | af k=8 | af k=24 |
|---|---|---|---|---|
| EXP-0044 all (tuned, 1 s) | 768 | 0.286 | 0.880 | - |
| EXP-0045 all | 512 | 0.262 | 0.802 | 0.865 |
| EXP-0045 b0.03 | 128 | | 0.497 | 0.599 |
| EXP-0045 b0.1 | 128 | | 0.874 | 0.944 |
| EXP-0045 b0.3 | 128 | | 0.914 | 0.957 |
| EXP-0045 b1.0 | 128 | | 0.924 | 0.959 |

EXP-0044 per goal af8: quadrants 0.93-0.96; letters I 0.90, X 0.90, Z 0.87, L 0.87, T 0.87,
S 0.87, H 0.84, C 0.82, O 0.81 (ring-like goals hardest). EXP-0045 per goal af24: quadrant_3 0.91,
quadrant_0 0.90, L 0.88, T 0.86, X 0.86, S 0.85, Z 0.84, O 0.82.

Goal redundancy (push-ranking corr, 12 clusters): {C,D,O,Q,U}, {I,T,V,Y,Z}, {F,P},
{A,B,E,G,H,K,N,R,S,X}, singletons M, J, W, L, quadrant_0..3. Top near-duplicate pairs:
B-S 0.967, F-P 0.956, C-O 0.946 (model-ranking Kendall 1.0), O-Q 0.936 (IoU 0.90), B-E 0.924,
V-Y 0.921, C-D 0.915, H-N 0.902, T-Y 0.901. Distance-field correlation is >= 0.95 for nearly all
letter pairs, so it does not discriminate; IoU and push-rank corr do. Quadrants are mutually
distinct (rank corr -0.41..0.10). Recommended 12: letter O, T, F, S, X, M, W, J, L, I +
quadrant_0, quadrant_1 (+ two_squares opt-in); max push-rank corr within the set 0.84, mean 0.42.

## What would change the verdict
An exact packer (MILP) lowering V* materially (would raise af only for letter_I/J/Y); allowing
stacking (V* could fall further; the flat ceiling is the physically sensible one for 20 cubes).
Redundancy on a second corpus (other start distribution) reordering the clusters.

## Threats
- indirectness: V* is a flat-layer ceiling from a heuristic packer, not the reachable set of the
  pusher; push-ranking redundancy is measured on EXP-0030 one-step pools, not closed loop.
- Considered: V0 scale mismatch between EXP-0044 and this scoring -- dismissed (start 40 V0 matches
  to 1e-9).

## Unrelated findings
- Distance-field correlation between letter goals is 0.95-0.99 for almost every pair: the
  normalised distance field is dominated by the tray-scale far field, which is why lyapunov ranks
  pushes similarly across very different letters.
