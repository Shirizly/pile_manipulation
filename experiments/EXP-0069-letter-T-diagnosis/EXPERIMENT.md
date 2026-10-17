---
id: EXP-0069
title: >
  letter_T is the hardest of the closed-loop letters for every model family, not a saturated or mis-rendered
  one: its crossbar (45 % of the optimum mass) is filled only by pushes along world y, which every narrow planner
  executes ~19 % of the time, while the linear model's centre-only, x-dominant pushes happen to land on T's
  stem/junction -- so the NFD-vs-linear gap closes on T (and X) and opens on O and S
tier: T1
mode: exploratory
date: 2026-10-05
hypothesis: null
claim: >
  In the legal closed-loop episodes already on disk (EXP-0066: DS-0006 scatter starts 40-71, letters O/T/S/X,
  16 x 20 mm CEM pushes, score = in-goal mass over pushes 4-16 / EXP-0046 optimum; EXP-0039 RUN-0002 cem/rank,
  8 letters x 16 starts), letter_T's near-zero NFD-minus-linear gap is NOT explained by headroom saturation
  (H-a), by a wall-adjacent or mis-rendered mask (H-b wall, H-e) or by legality fallbacks; it IS consistent with
  push-direction geometry (H-b direction) plus the linear model's degenerate centre-blob policy (H-d): on T,
  x-pushes add +0.8e-3 of total mass to the crossbar per push and +10e-3 to the stem, y-pushes add +12e-3
  (NFD) / +23e-3 (linear) to the crossbar; y-pushes are 18-22 % of executed pushes in every family; the linear
  model starts 8-17 % of its pushes outside |x|,|y| > 32 mm (NFD 49-69 %) and leaves the bands above the crossbar
  and below the stem untouched (0.229 -> 0.236, 0.248 -> 0.246 of total mass) while the NFD drains them
  (-> 0.207, 0.180). Capacity correlates with the 8-letter gap (Pearson +0.65) but letter_L (capacity 20,
  gap +0.052) shows it is not sufficient on its own.
provenance:
  commit: a5ccb680
  dirty: true
  data_commit: "EXP-0066 results written 2026-10-04/05 at a5ccb680 dirty; EXP-0039 RUN-0002 at 81f90250"
  script: experiments/EXP-0069-letter-T-diagnosis/code/diagnose_T.py, code/mechanism_8letters.py
  data: ["EXP-0066 results/{nfd_s0,nfd_s1,nfd_s2,lin_orig,lin_b1,lin_b2,broad_s0,broad_s1}_ps0.json (recorded particle states)", "EXP-0039 results/run0002/hm_{cem,rank}_*.json", "EXP-0046 results/vstar.json", "EXP-0051 results/analysis.json per_goal (read only)"]
  code_path: "occ_for_scoring (soft splat) + Baselines/common/goals.mass_in_region on the recorded states, the EXP-0066 analyse.py formula; masks from Baselines/common/goals.letter_mask (helvetica_thin, 64 px)"
  seed: 0
  split: "not applicable (re-analysis of recorded episodes)"
  runtime: "~4 min CPU total, no GPU, no simulation"
  runs: [RUN-0001]
budget:
  declared: "2.5 h wall clock, CPU only, report <= 500 words"
  spent: "~75 min"
  outcome: within
design:
  varied: {letter: "O T S X (EXP-0066; 3 families); O T S L X Z C H (EXP-0039 cem/rank; 8 models)", model_family: "narrow NFD (3 seeds), narrow linear res64 (3 instances), broad nfd_3ch_randlen (2 seeds)", push_direction_class: "along x / along y / diagonal (22.5 deg bands)"}
  held_fixed: {episodes: "exactly the recorded EXP-0066 / EXP-0039 RUN-0002 episodes; nothing re-simulated", scoring: "occ_for_scoring sigma 1 px, 64 px / +-64 mm, in-goal mass / mass_frac_best_placement", push_len: 0.02, steps: 16}
  baselines: ["do-nothing: in-goal mass at k=0 (same 32 starts for every letter)", "broad nfd_3ch_randlen family as the non-narrow reference", "EXP-0046 optimum placement (capacity, mass_frac_best_placement)"]
  metric: "achieved in-goal mass fraction / optimum (EXP-0066 score; see METRICS.md achieved_fraction family) -- reported as reference; slateN / accuracy not applicable to a re-analysis of closed-loop states"
noise_floor: "within-letter SE of a 32-start family mean ~0.02 (per-model 16-episode SE 0.022-0.028 in EXP-0039); per-start paired gap sd 0.07-0.12; per-push direction-class means pool 200-1000 pushes"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  All three families score lowest on T (NFD 0.209, linear 0.215, broad 0.314 vs 0.28-0.43 on O/S/X); T has the
  lowest k=0 in-goal mass (0.067 vs 0.083-0.094) and the smallest fraction of headroom closed by k=16 (NFD 0.21
  vs 0.28-0.33), so H-a (saturation) is refuted; the mask is 216 px, stroke 8 mm, >= 16 px from every tray edge,
  flip-x symmetric (IoU 1.00), centroid (0, -11 mm) because the crossbar is on one side -- H-e refuted; fallback
  rate on T 0.035 vs 0.038-0.049 (NFD), 0 illegal pushes -- legality refuted. Direction: x-pushes feed the stem
  (+10e-3/push), y-pushes feed the crossbar (+12 to +23e-3/push); y-pushes are 18-22 % of pushes in every
  family; crossbar fill at k=16 is 0.094 (NFD) / 0.116 (lin) / 0.168 (broad) of a 0.45 target vs stem 0.155 /
  0.112 / 0.260 of 0.55. Linear pushes start near the centre (8-17 % from the outer bands vs 49-69 % NFD) and
  pile mass at the T junction / X crossing (blob 0.08-0.10 vs 0.03-0.05 on O/S/C), never draining the bands above
  the crossbar or below the stem; its on-policy pred-true corr is best on T (+0.26) and the narrow NFD's worst
  (-0.17 vs -0.02..-0.04). 8-letter gap vs capacity r = +0.65 but L (cap 20) has gap +0.052.
verdict: supported
downgrades: [indirectness, imprecision]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
If T were saturated (H-a) it would have the highest k=0 in-goal mass and the largest fraction of headroom closed;
it has the lowest of both. If the mask were mis-rendered or wall-adjacent (H-b wall, H-e), the final occupancy
maps would not cluster on the drawn glyph and the mask statistics would differ from O/S/X; they cluster on it and
the stroke width / edge distance match. If legality drove it, fallback / shift / illegal counts would be elevated
on T; they are not. What remains is split by push direction: per-push gains to the crossbar vs the stem by
direction class separate "the crossbar needs y-pushes that planners rarely make" from "T is uniformly hard".

## What was actually run
Dirty tree at run time: uncommitted were other agents' edits to `simple_mpc/learned_mpc.py`, `simple_mpc/adapters.py`,
`Baselines/LinearForesight/fit_switched.py`, EXP-0039/0043 code and docs (listed in EXP-0066 RUN-0001), plus this record's own
`code/` -- none in the read-only scoring path used here.
`code/diagnose_T.py` recomputes in-goal mass per push from the recorded states of every EXP-0066 cell and of the
16 EXP-0039 RUN-0002 cem/rank files (same splat and formula as EXP-0066's analyse.py; recomputed cell scores agree
with EXP-0066's table to 3 decimals), then tabulates mask geometry, per-family x letter scores, k0/k4/k16,
headroom closed, legality fallback / shift rates, wall-touching push ends, push-direction classes, on-policy
pred_dv vs true_dv, final-mass distance-to-mask profiles, letter_T crossbar vs stem mass, and paired per-start
gaps. `code/mechanism_8letters.py` adds the 8-letter gap vs capacity / area / blob predictors, the mean-map blob
statistics, cube spread, and the letter_T region accounting (above crossbar / beside stem / below stem / on mask).
A third inline check (logged in RUN.md, output `results/push_direction_effect.json`) computes the per-push gain to
the mask / crossbar / stem by direction class and family. Exploratory: no prediction was committed; the hypotheses
H-a..H-e were listed in the brief before any number was read.

## Numbers
Full tables: `results/diagnosis.md`, `results/mechanism.md`, `results/push_direction_effect.json`. Figures:
`results/figures/f1_ingoal_vs_push.png` (in-goal mass vs push index, 4 letters x 3 families),
`f2_occupancy_maps.png` (mean occupancy at k=0 and k=16 over the mask outline), `f3_push_directions.png`
(direction roses), `f4_T_parts_and_gaps.png` (crossbar vs stem fill; per-start paired gaps),
`f5_blob_8letters.png` (linear vs NFD final maps, 8 letters).

| letter | px | capacity | optimum | k0 (all) | NFD score / k16 / headroom closed | linear score / k16 / closed | broad score / k16 / closed | NFD fallback | lin fallback | NFD y-push frac | lin y-push frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| O | 340 | 32 | 0.956 | 0.094 | 0.317 / 0.397 / 0.33 | 0.223 / 0.270 / 0.19 | 0.431 / 0.593 / 0.55 | 0.038 | 0.005 | 0.27 | 0.37 |
| T | 216 | 22 | 0.965 | 0.067 | 0.209 / 0.259 / 0.21 | 0.215 / 0.236 / 0.18 | 0.314 / 0.443 / 0.40 | 0.035 | 0.001 | 0.19 | 0.18 |
| S | 306 | 30 | 0.938 | 0.088 | 0.280 / 0.340 / 0.28 | 0.128 / 0.134 / 0.05 | 0.380 / 0.511 / 0.46 | 0.049 | 0.001 | 0.22 | 0.31 |
| X | 278 | 21 | 0.949 | 0.083 | 0.284 / 0.362 / 0.30 | 0.268 / 0.314 / 0.25 | 0.383 / 0.529 / 0.49 | 0.044 | 0.023 | 0.21 | 0.20 |

Per-push gain on letter_T (fraction of total mass, mean over pushes of that direction class):

| family | x-pushes: n, to crossbar, to stem | y-pushes: n, to crossbar, to stem | diagonal: n, to crossbar, to stem |
|---|---|---|---|
| NFD | 658, +0.8e-3, +10.0e-3 | 294, +11.8e-3, +2.6e-3 | 584, +3.2e-3, +7.8e-3 |
| linear | 1053, +0.8e-3, +9.8e-3 | 276, +22.9e-3, -14.5e-3 | 207, +3.6e-3, +6.6e-3 |
| broad | 329, +4.1e-3, +27.7e-3 | 230, +15.2e-3, +3.0e-3 | 465, +8.2e-3, +10.3e-3 |

letter_T mass by region, k=0 -> k=16 (tray fraction): above crossbar (0.25): NFD 0.229 -> 0.207, lin -> 0.236,
broad -> 0.174; lower quadrants beside the stem (0.64): NFD 0.649 -> 0.485, lin -> 0.477, broad -> 0.333; below
stem (0.25): NFD 0.248 -> 0.180, lin -> 0.246, broad -> 0.088; on mask: 0.065 -> 0.250 / 0.227 / 0.428.

EXP-0039 cem, 8 letters: NFD3 - linear res64 gap: O +0.069, T -0.020, S +0.133, L +0.052, X -0.017, Z +0.083,
C +0.049, H +0.068 (SE of a 16-episode model mean 0.022-0.028). Correlation of this gap with capacity +0.65,
mask px +0.45, linear blob fraction -0.59, linear score -0.76 (n = 8, descriptive). Rank arm: T's tau vs the
pooled order is +0.79 (above the median letter), so the T anomaly is CEM-specific.

## What would change the verdict
A direct test of the direction mechanism costs one GPU closed-loop cell: re-run EXP-0066's NFD and linear cells on
letter_T with the CEM candidate bank restricted to pushes along world y (and, as control, along x); if the
crossbar fill and the T score rise under the y-only bank for both families, the geometry story holds; if only
the NFD gains, the gap on T is model-specific after all. A rotated-T goal (crossbar along y) is the cheaper
variant: the x-dominant planners should then fill the crossbar and starve the stem, flipping the crossbar/stem
table above.

## Threats
- indirectness: the direction-class gains are observational (the planner chose the push given the state);
  a y-push's larger crossbar gain partly reflects that the planner picks y-pushes when crossbar mass is available.
- imprecision: per-letter family means have SE ~0.02 and the T gap's CI already spans 0; the 8-letter
  correlations are n = 8 and descriptive only.
- Considered and dismissed: the mask convention (the k=16 maps cluster on the drawn glyph in every family;
  flip-x IoU 1.00 for T); the legality path (fallback rate on T is at or below the other letters'; 0 illegal
  pushes executed); the start distribution (same 32 starts for every letter; k0 differs only through the mask).
- Why planners of both narrow families prefer x-pushes (38-53 % along x vs 22-26 % along y, with the candidate
  bank drawn isotropically by `pile_aware_action_batch::draw_headings`) is NOT isolated here; the broad family is
  nearly isotropic (31 % / 25 %). The training corpus DS-0015 draws headings uniformly, so it is not a data bias.

## Unrelated findings
- The linear narrow v2 res64 planner's executed pushes are almost exactly axis-aligned (4-lobed rose, 13-27 %
  diagonal) and start inside |x|,|y| < 32 mm 83-92 % of the time; the NFD's start outside that box 49-69 % of the
  time. The linear planner never collects peripheral mass, which is why its score collapses on S (0.128) and its
  k=16 map is a blob at the T junction / X crossing (`results/figures/f5_blob_8letters.png`).
- `experiments/EXP-0066-closed-loop-power-study/results/analysis.md`'s per-letter table reports medians; the
  per-letter MEANS (this record) put linear above NFD on T by 0.006, consistent with the paired gap.
