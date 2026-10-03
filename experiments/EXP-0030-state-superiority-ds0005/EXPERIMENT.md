---
id: EXP-0030
title: >
  On DS-0006 (160 states, two 64-push pools each, training-matched physics) the
  NFD-model ensemble again beats the best single model (+0.030 / +0.059), per-state
  advantage repeats only weakly across pools (r ~0.15), per-state switching loses,
  and no short descriptor survives correction
tier: T1
mode: confirmatory              # A1-A4 pre-registered (DESIGN.md + two addenda, all before outcomes)
date: 2026-09-24
hypothesis: null
claim: >
  On DS-0006 with 8 OCC_ADAPTERS models, 30 goals and soft truth scoring:
  (A1) median split-half r of per-state advantage over NFD-family pairs >= 0.2;
  (A2) cross-fitted per-state switching gain interval includes 0; (A3) the NFD
  ensemble beats the best single model (CI excluding 0); (A4) >= 1 descriptor link
  Holm-significant.
prediction:
  supports: "A1 r >= 0.2; A2 interval includes 0; A3 CI > 0; A4 >= 1 Holm-significant link"
  refutes: "A1 r < 0.2; A2 interval excludes 0; A3 CI includes 0; A4 none"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/analyse.py
  data: ["DS-0006"]
  code_path: "OCC_ADAPTERS predict_step -> per-goal lyapunov / mass_in_region -> slateN per 64-push pool; truth occ_for_scoring(states_)"
  seed: "re-split rng seed 0 (100 re-splits; primary = first); state bootstrap 5000"
  split: "each state's 128 pushes into two disjoint 64-push pools"
  data_commit: not applicable
  runs: [RUN-0001, RUN-0002]
  runtime: "~5 min"
budget:
  declared: "TODO G3a; not separately declared"
  spent: "~5 min compute (plus DS-0006 collection, 79 min)"
  outcome: within
design:
  varied:
    value_fn: "lyapunov, mass_in_region"
  held_fixed:
    models: "nfd_3ch_randlen, nfd_warped_randlen, nfd_warped_randlen_flipaug, nfd_warped_randlen_flipaug_epoch30, nfd_residual_warped_flipaug_randlen, nfd_residual_worldframe_noaug_ep43, nfd_3ch_finetuned, linear_switched_hard; ensemble = mean of the 7 models whose id starts with 'nfd' (the 6 randlen-trained variants AND the weak nfd_3ch_finetuned) -- corrected 2026-09-24, earlier text said 6"
    goals: "26 letters + 4 quadrants (64x64 slate grid)"
    truth_scoring: soft
  baselines: "best single model chosen on pool 1; random = capture 0"
  metric: slateN; split-half r; switching gain; ensemble gain
noise_floor: "state-bootstrap 95% intervals over 160 states; no training-seed floor (EXP-0036 pending)"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x,
             occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  A1 refuted: median r (NFD pairs) +0.16 lyapunov, +0.13 mass (max +0.25, pairs vs
  the weak nfd_3ch_finetuned). A2: lyapunov refuted in the unfavourable direction
  (-0.025 [-0.054, -0.018]); mass supported (-0.012 [-0.024, +0.002]). A3 supported:
  +0.030 [+0.018, +0.041] lyapunov, +0.059 [+0.045, +0.068] mass. A4 refuted: 0
  Holm-significant links; strongest raw links hull_area / radius_of_gyration with
  advantage over nfd_3ch_finetuned, rho ~+0.20-0.22, p ~0.007-0.013, Holm 0.21-0.34;
  ridge selector always chose the better model (gain 0). A5 (added, addenda 3-4) supported:
  at MATCHED wall-clock (throughput cost; the ensemble costs the sum of its members),
  the 7-member ensemble loses to the best single model at every budget, -0.20 to -0.53
  lyapunov and -0.26 to -0.49 mass; the 5-member one -0.14 to -0.34 / -0.20 to -0.34.
verdict: refuted                # the conjunctive claim fails on A1 and A4; A3 (ensemble) replicates
downgrades: [indirectness]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
Two disjoint 64-push pools per state, 160 states and training-matched physics give
a direct test of state-level (not pool-level) superiority, with power the 20-state
DS-0001 analysis lacked.

## What was actually run
RUN-0001 exactly as pre-registered (corpus switched from DS-0005 to DS-0006 before
any outcome, addendum 2; `n_clusters` replaced by `frac_with_neighbour_12mm` after a
DS-0001 smoke run, addendum 1). NaN-robust version of the code (the EXP-0035 fix);
0 undefined (state, pool) cells here.

## Numbers (results/analysis.json)
| value fn | A1 median r (NFD) | A2 switching gain | A3 ensemble - best single | best single model |
|---|---|---|---|---|
| lyapunov | +0.16 | -0.025 [-0.054, -0.018] | +0.030 [+0.018, +0.041] | nfd_residual_warped_flipaug_randlen 0.821 |
| mass_in_region | +0.13 | -0.012 [-0.024, +0.002] | +0.059 [+0.045, +0.068] | nfd_residual_worldframe_noaug_ep43 0.776 |

Model means (lyapunov): nfd_residual_warped_flipaug_randlen 0.821,
nfd_warped_randlen 0.785, epoch30 0.785, nfd_residual_worldframe_noaug_ep43 0.776,
nfd_warped_randlen_flipaug 0.775, nfd_3ch_randlen 0.760 (json for the rest).

## Reading
- Across three corpora now (DS-0006 here; DS-0007 n20 and n50 in EXP-0035) the
  seven-member NFD ensemble (incl. the weak nfd_3ch_finetuned) beats the best single model, +0.014 to +0.059.
- A state-level component of "which model wins" exists but is small (r ~0.15 at
  64-push pools; ~0.4-0.55 at 500-push pools on DS-0001, EXP-0029): most per-state
  advantage in a practical pool is pool noise, so per-state selection from one pool
  loses to simply using the best model -- and to the ensemble.
- No short descriptor explains per-state advantage after correction; the one
  suggestive pattern (spread-out states favour good models over the weak finetuned
  model) is about a weak model, not about choosing among good ones.

## What would change the verdict
Per-state evidence from much larger pools (A1 rises with K) or a selector using
model DISAGREEMENT rather than state geometry; an ensemble of seeds (EXP-0036) to
see whether diversity or averaging drives A3.

## Threats
n20 scatter only; one physics setting; single training run per model.

## A5: the ensemble at a matched time budget (RUN-0002; DESIGN.md addenda 3-4)
A3 gave every model the same candidates, so the ensemble spent 5-7x the compute.
Here each model gets as many of a state's 128 pushes (random subsets, 200 per state,
shared across models) as it can score in the time the reference nfd_3ch_randlen
needs for m pushes; capture is normalised by the full 128-push pool.

Cost (us per candidate, chunks of 128 as in the closed-loop planner; shared GPU):
nfd_3ch_randlen 57, residual_worldframe 57, 3ch_finetuned 56, residual_warped 76,
warped variants 84-89, linear_switched_hard 144; ensemble_nfd (7) 506, ensemble_nfd5 362.

slateN capture, lyapunov (pushes scored in brackets), reference budget m = 16 / 32 / 64 / 128:

| arm | m=16 | m=32 | m=64 | m=128 |
|---|---|---|---|---|
| nfd_residual_warped_flipaug_randlen | 0.524 (11) | 0.633 (23) | 0.718 (47) | 0.784 (95) |
| nfd_residual_worldframe_noaug_ep43 | 0.539 (15) | 0.633 (31) | 0.706 (63) | 0.761 (127) |
| nfd_3ch_randlen | 0.544 (16) | 0.628 (32) | 0.692 (64) | 0.745 (128) |
| nfd_warped_randlen | 0.492 (10) | 0.592 (20) | 0.674 (41) | 0.737 (82) |
| ensemble_nfd5 | 0.199 (2) | 0.400 (5) | 0.530 (10) | 0.640 (20) |
| linear_switched_hard | 0.382 (6) | 0.483 (12) | 0.566 (25) | 0.620 (50) |
| ensemble_nfd (7) | -0.001 (1) | 0.295 (3) | 0.464 (7) | 0.584 (14) |
| nfd_3ch_finetuned | 0.394 (16) | 0.445 (32) | 0.483 (65) | 0.512 (128) |

Ensemble minus A3's best single model (state-bootstrap 95% CI), m = 16 / 32 / 64 / 128:
- lyapunov, ensemble_nfd: -0.525, -0.338, -0.253, -0.200 [-0.211, -0.190];
  ensemble_nfd5: -0.324, -0.233, -0.188, -0.144 [-0.153, -0.134].
- mass_in_region, ensemble_nfd: -0.486, -0.368, -0.305, -0.260 [-0.271, -0.249];
  ensemble_nfd5: -0.338, -0.274, -0.242, -0.202 [-0.212, -0.192].
Every CI is entirely below 0 (full table: results/budget_matched.json). A5's
prediction holds.

Reading:
- When candidate count is what limits the planner, the ensemble's better ranking
  (+0.03-0.06 at equal count) is far smaller than what it loses by scoring 1/6-1/9
  as many pushes. Its A3 win is an equal-COUNT result, not an equal-TIME one.
- Speed reorders the single models too: at matched time the fast
  residual_worldframe and plain nfd_3ch_randlen tie the slower residual_warped at
  small budgets, and the warped variants drop behind nfd_3ch_randlen at every budget.
- Scope: the pool caps each model at 128 pushes, so only budgets up to the
  reference's cost for 128 pushes (~7 ms) are covered. At a real 1 s budget every
  model scores thousands of pushes, where capture saturates and a better ranker
  could win back; this test does not reach that regime (DS-0006 has 128 pushes per
  state). Timing ran on a shared GPU; the costs are relative.
- Deviation: the first A5 run timed a single call, which is launch overhead
  (~4-7 ms for any n <= 128) and gave every slower model 1 push -- degenerate, kept
  on disk, replaced by throughput costs before any informative number (addendum 4).

## Unrelated findings
linear_switched_hard is the SLOWEST model per candidate (144 us, 2.5x NFD) in this
harness; the "linear is fast" reading from the EXP-0032 pilot (linear_switched_soft,
~2x the evaluations) does not transfer to the hard variant here -- untested why.

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- EXP-0035 replicates the negative state-dependence result at ~800 states (C-036 narrowed).
- EXP-0065 / ISS-013: 54 % of DS-0006's candidate pushes put the blade on a cube at touchdown (pre-fix pile-aware sampler); this record's pool numbers were not re-scored on legal-only candidates.
- EXP-0065 RUN-0002 re-scored THIS record's cached predictions on legal-only candidates: the true best push is illegal in 72 % / 54 % of cells (lyapunov / mass); legal-only minus size-matched slateN is +0.113 for linear_switched_hard vs +0.01..+0.07 for NFDs, Kendall(all, legal) 0.83. The A3 ensemble gain persists (ensemble still first).
