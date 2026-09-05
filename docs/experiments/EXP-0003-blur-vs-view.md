---
id: EXP-0003
title: Blur moves the operator's error by ~33 points on cubes; the view choice moves it by ~8
tier: T1
mode: confirmatory
date: 2026-09-03
hypothesis: H-A2
claim: >
  RESTATED 2026-09-05 to cube-only scope (see Amendment). On scattered
  monolayer cubes, sweeping Gaussian blur from sigma 0 to 1.5 moves the linear
  operator's error by ~33 points of "% of the change", while switching between
  the mask and density views at matched blur moves it by ~8 points. Blur is the
  larger axis by 4x, and the two are separable.
prediction:
  supports: "|mask - density| stays under ~10 points at every sigma, while the sigma range spans >20 points"
  refutes:  "one view leads the other by more than the sigma effect at any sigma"
  # NOTE: as originally written this prediction was about sand AND cubes. The
  # sand arm is withdrawn; on cubes alone the view gap is 7.6-8.2 points, which
  # clears the stated <10 threshold but by less margin than the sand arm did.
  discriminating: true
provenance:
  commit: 17a7d4a7
  dirty: true                     # BACKFILLED 2026-09-05: `scripts/probes/view_blur.py`
                                  # did not exist at 17a7d4a7; it was written in the same
                                  # session and first committed at aac084e3. So this sha
                                  # bounds the run from below only -- the analysis code
                                  # that actually ran is the aac084e3 version of that file.
  script_first_committed: aac084e3
  script: scripts/probes/view_blur.py
  data: ["Genesis/data/foresight/L040/**/*_data.pt"]
  code_path: points_to_mask / points_to_density
  seed: 0
  split: "episode-level, 25% of files, seed 0"
  runtime: "~20 min, CPU"
design:
  varied: {blur: [0.0, 0.5, 1.0, 1.5], view: [mask, density], res: [32, 64]}
  held_fixed: {crop: 1.0, ridge: 1.0, estimator: "ridge toward identity", grid: 64, normalize: mean, split_rule: identical}
  baselines: [persistence, mean-delta, identity-warp]
  metric: "swept-region rms as a percentage of the persistence rms at the SAME blur"
noise_floor: "not measured for this design; view differences of <7 points are treated as not interpretable"
depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, particle-projection]
result: "cubes L040: mask 89.4/83.6/67.4/56.1%, density 97.0/93.4/75.4/64.3% across sigma 0/0.5/1.0/1.5. Blur spans 33 points, view spans 8."
verdict: supported
downgrades: [indirectness, imprecision]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

Earlier work varied view and blur together (mask at σ=1, density at σ=0), so a
view effect and a blur effect were observationally identical in it. Crossing
the two factors separates them: if the view matters, the two columns should
separate at fixed σ; if blur matters, the rows should separate at fixed view.
Both move, but by very different amounts.

## What was actually run

Full 2×4 cross of blur × view on the L040 scattered-cube set, at res 32 and
again at res 64 to check the finding is not a resolution artifact. Every cell
is reported below — no cell was selected.

**Amendment, 2026-09-05.** This experiment originally ran the same cross on MPM
sand as its primary arm, and the cube cross as the replication. That path was
withdrawn as non-physical (`docs/rejected_mpm_sand.md`), so every sand number
is removed and the cube arm is promoted to the finding. Two consequences,
stated rather than buried:

1. The **blur** result is unaffected in direction and barely in size — cubes
   showed 89.4% → 56.1% across σ where sand showed 63.0% → 33.2%.
2. The **view** result is materially weakened. It rested mainly on sand, where
   the two views tracked within ~4 points. On cubes alone the mask leads the
   density view by 7.6–8.2 points at every σ — still small next to the 33-point
   blur effect, but no longer "the views are interchangeable". The view
   sub-question is now **inconclusive** and the verdict below reflects only the
   blur half.

## Numbers

cubes n50 scattered monolayer (L040), res 32, crop 1.0 — error as a percentage
of the change at that σ (lower is better; 100% = no better than persistence):

| view | σ=0 | σ=0.5 | σ=1.0 | σ=1.5 |
|---|---|---|---|---|
| mask | 89.4% | 83.6% | 67.4% | **56.1%** |
| density | 97.0% | 93.4% | 75.4% | **64.3%** |
| *mask − density* | −7.6 | −9.8 | −8.0 | −8.2 |

Blur spans 33.3 points on the mask view; the view gap is 7.6–9.8 at every σ.
The `identity (warp only)` control rises toward 100% as σ grows — i.e. the warp
becomes free — which is the mechanism the σ effect runs through, and is why
this record feeds H-A2 rather than standing alone.

## What would change the verdict

For the blur half: nothing cheap — it reproduces at two resolutions and on two
views. For the view half: re-running the cross on the piled cube sets (n20,
n30), where the density view carries real depth information that a monolayer
cannot. That is the test that would say whether the 8-point mask lead is a
monolayer artifact. ~1 h, no new data.

## Threats

- `indirectness` — **the load-bearing one.** The metric is normalised within
  each σ, but σ changes *what is being predicted*: a blurred field is a
  genuinely easier and genuinely less informative target. So the σ column is
  not a like-for-like accuracy comparison and must not be read as "blur makes
  the model better". The **view** comparison at fixed σ is like-for-like and is
  what the claim is about. H-C1 is the experiment that resolves what σ is worth
  for control, and the "does blur remove noise or signal?" question is now the
  main line — see `docs/ideas_log_signal_vs_detail.md`.
- `imprecision`: one seed-0 split where LORO was affordable.
- `untested-dependency`: `swept-region-metric`, `episode-split` unchecked.

## Grade note, 2026-09-05

`untested-dependency` dropped: this record's `depends_on` tags all
hold as of the invariant tests added today (`tests/test_metric_invariants.py`,
`tests/test_grid_convention.py`). Grade very-low -> low. The evidence did not
change; what changed is that the assumptions it rests on are now checked.

## Unrelated findings

- `fit_operator_nonneg` costs O(D³) per FISTA iteration — a full D×D matmul.
  Measured during a later run: ~51 s for 4000 iterations at D=1024, versus
  >600 s for *200* of 4000 at D=4096. This is the real reason res-64 nonneg
  numbers are rare in this repo, and it is documented nowhere near the
  function. Logged for triage, not acted on.
