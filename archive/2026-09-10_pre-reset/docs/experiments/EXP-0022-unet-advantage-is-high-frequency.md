---
id: EXP-0022
title: The UNet's win over the linear operator is entirely high-frequency; on coarse structure they are equivalent
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: C-030
claim: >
  EXP-0021's finding that the UNet beats the linear operator in 14/14 cells was
  measured on a SHARP target, where C-002 says the linear method's SE(2) warp is
  destructive. Scored on a common BLURRED target -- each model preprocessing
  internally as it likes -- the UNet's 7-11 point advantage collapses to within
  +-5 points, and the ranking reverses in at least half the cells.
prediction:
  supports: "the UNet's advantage shrinks by more than half when both models are scored on a sigma=1 target"
  refutes:  "the advantage survives blurring roughly intact, i.e. the UNet is better at coarse structure too"
  discriminating: true
provenance:
  commit: c3f0104e
  dirty: False
  data_commit: "f196d657 (blind cells), 3d7db119 (contact cells) — read from each dataset's _N_config.yaml provenance block"
  script: scripts/probes/exp0022_blur_fair.py
  data: ["Genesis/data/granularity/{n20,c20,c5}/**", "Genesis/data/foresight/L040/**"]
  code_path: "PileSweepData raster (post-fix), fit_linear_foresight fit_operator + predict_world"
  seed: 0
  split: "registry file-level split; identical held-out transitions for both models"
  runtime: "~6 min CPU for four cells"
budget:
  declared: "45 min, 90k tokens"
  spent: "~40 min, ~60k tokens"
  outcome: within
design:
  varied: {scoring_target: [sharp, "blurred sigma=1"], cell: [blind_n50, blind_n20, contact_n20, contact_n5]}
  held_fixed: {res: 64, crop: 0.5, ridge: 1.0, estimator: "ridge toward identity", checkpoints: "EXP-0021's, unchanged", split: identical, metric: "swept-region rms as % of persistence AT THE SAME sigma"}
  baselines: [persistence]
  metric: "pct_persistence (swept region), computed at the SAME sigma for both models — see docs/experiments/METRICS.md. Originally: swept-region rms / persistence rms, both computed on whichever target is being s"
noise_floor: "not measured for this design; the effect is a 9-12 point swing that reverses a ranking, and it reproduces in 4/4 cells"
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend, swept-region-metric, episode-split, world-frame-alignment]
establishes: []
result: >
  Sharp -> blurred, linear vs UNet: blind_n50 76.2/66.4 -> 57.6/58.8;
  blind_n20 79.3/72.1 -> 60.7/65.5; contact_n20 74.7/65.7 -> 57.3/57.6;
  contact_n5 82.9/72.0 -> 67.6/66.9. Swing 9.3-12.0 points in every cell.
verdict: supported
downgrades: [imprecision, indirectness]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0021 scored both models at sigma=0. That is the one configuration a standing
claim (C-002) says is broken for the linear method: its SE(2) warp destroys
pixel-scale content, and on L040 the operator measures 77.6% of persistence
unblurred against 57.8% blurred. So the head-to-head was run with one model in
its worst configuration.

But the two numbers are **not comparable**, which is the trap: they score
different targets against different persistence baselines. "Linear gets 57.8%"
and "UNet gets 66.4%" cannot be set side by side. The fair test is one target
for both, and neither EXP-0018 (blurred, no UNet) nor EXP-0021 (sharp, both)
ran it.

## What was actually run

Each method preprocesses as it likes; both are scored identically:

- **linear** — fitted and predicting in whichever field is being scored.
- **UNet** — receives the sharp input it was trained on, predicts sharp, and its
  prediction is then blurred for scoring. The checkpoints are EXP-0021's,
  unmodified; nothing was retrained.
- **truth and persistence** — blurred the same way, so the denominator matches.

## Numbers

Swept-region rms as % of persistence *at the same sigma*:

| cell | sharp: linear | sharp: UNet | blurred: linear | blurred: UNet | swing |
|---|---|---|---|---|---|
| blind n=50 (L040) | 76.2 | **66.4** | **57.6** | 58.8 | 11.0 |
| blind n=20 | 79.3 | **72.1** | **60.7** | 65.5 | 12.0 |
| contact n=20 | 74.7 | **65.7** | **57.3** | 57.6 | 9.3 |
| contact n=5 | 82.9 | **72.0** | 67.6 | **66.9** | 10.2 |

The UNet leads by 7.2-10.9 points sharp, and by -4.8 to +0.7 blurred. The
swing is 9-12 points in all four cells, and the ranking reverses in two.

## What this means

**The UNet's entire advantage is high-frequency.** On the coarse structure that
survives a sigma=1 blur, the two model classes are equivalent to within a few
points, with the linear operator marginally ahead on average.

That is not a technicality, because of what the rest of the register says about
which component control consumes:

- EXP-0007: the operator's error is ~3x worse in the finest band (0.686 of the
  band signal) than at ~8 px (0.221). The UNet appears to win exactly that band.
- EXP-0008 and its reviewer amendment: at matched rms, high-frequency noise
  destroys action ranking while amplitude and blur errors cost it **nothing**.
- C-039 (EXP-0012, n=50 same-state slates): noise still costs 5.5x more ranking
  quality than displacement once the cross-state confound is removed.

So the UNet is better at predicting the component the control evidence says MPC
does not consume, and is **not** better at the component it does. Whether it is
the better model *for this project* is therefore open, not settled -- and it is
now answerable, because the same-state slates and the ranking metric both exist.

## What would change the verdict

The decisive follow-up is control utility, not more one-step error: score both
model classes with `rank_metrics` on the same-state slates
(`Genesis/data/slates/n20_heap_5mm`, 50 states x 32 actions, verified). If the
UNet's fine-detail advantage buys nothing there, C-041 should be restated as a
statement about sharp-target accuracy rather than about model quality.

Also worth one run: a sigma sweep rather than a single sigma=1, to locate where
the crossover sits.

## Threats

- `imprecision`: one split per cell, no seed repeats, no measured floor. The
  effect is large and reproduces 4/4, but the *sign* of the blurred comparison
  is within a few points in three of four cells and should not be read as
  "linear wins" -- only as "the advantage is gone".
- `indirectness`: still one-step image error. The claim this bears on (C-030)
  is about control, which this does not measure.
- Considered and dismissed: **that blurring the UNet's output handicaps it.**
  It is given the sharp input it was trained on and produces its native
  prediction; only the scoring is smoothed, identically for every model.
- Considered and NOT dismissed: sigma=1 is a choice. It comes from C-002, which
  derived it for the warp, not for this comparison.

## Unrelated findings

- `load_transition_arrays` takes a *dataset* config, but EXP-0021's eval script
  documents itself as taking a *training* config; both work only because the
  shell driver passes dataset configs. Confusing, unfixed.
