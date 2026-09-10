---
id: EXP-0020
title: The fixed grid convention is right against physics, not merely self-consistent
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: C-018
claim: >
  The convention EXP-0001 standardised on (dim0 = world_x) is the physically
  correct one: the direction in which occupancy moves, read out of the grid
  under that convention, aligns with the direction the blade travels in world
  metres. The competing pre-fix convention (dim0 = world_y) does not.
prediction:
  supports: "mean cosine between grid transport direction and world push direction > 0.5 under dim0=world_x, and at least 0.3 higher than under dim0=world_y"
  refutes:  "the two conventions score comparably, or dim0=world_y wins — in which case the whole system was standardised onto the wrong axis and MPC reasons in a mirrored frame"
  discriminating: true
provenance:
  commit: 284608fa
  dirty: false
  data_commit: "unrecorded (cube_spectrum/n20 predates dataset stamping; mtime places it near 901ba35e/299c9d6e)"
  script: tests/test_world_frame_ground_truth.py
  data: ["Genesis/data/cube_spectrum/n20/*_data.pt (1876 pushes over 6 files)"]
  code_path: particles_to_occupancy
  seed: n/a
  split: "none — this is a property of the data, not a fitted model"
  runtime: "<1 min, CPU"
budget:
  declared: "20 min, 40k tokens"
  spent: "~15 min, ~20k tokens"
  outcome: within
design:
  varied: {convention: ["dim0=world_x (post-fix)", "dim0=world_y (pre-fix)"]}
  held_fixed: {dataset: cube_spectrum/n20, rasteriser: particles_to_occupancy, sigma: 0.0, grid: 64, bounds: "±0.064 m"}
  baselines: ["the competing convention", "world-frame particle motion (ground truth, grid-free)"]
  metric: "world_alignment_cosine — see docs/experiments/METRICS.md. Originally: cosine between the grid's transport direction (centroid of arriving occupancy mi"
noise_floor: "not needed: the two hypotheses are separated by 0.98, and the competing one sits at 0.00 — i.e. exactly the no-information value"
depends_on: [grid-convention, rasteriser-identity]
establishes: [world-frame-alignment]
result: >
  dim0=world_x: mean cos +0.9792. dim0=world_y: mean cos -0.0012. 99.7% of
  pushes move material forward in world metres (grid-free ground truth).
verdict: supported
downgrades: []
grade: high
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0001 established that the dataset's occupancy channel, its plate channel
and `particles_to_occupancy` **disagreed**, and fixed them to agree on
`dim0 = world_x`. That is a consistency argument, and consistency is not
correctness: had all three been standardised onto the *wrong* axis, the system
would still be internally coherent, one-step prediction would be unaffected
(the model would simply learn transport along whichever axis it was shown), and
the error would surface only in the MPC — which renders and reasons about
pushes in the grid frame while commanding actions in the world frame.

Nothing in EXP-0001, and nothing in any record since, tested that. This does,
by going outside the code entirely: recorded particle positions in world metres
are ground truth, independent of every rasteriser.

## What was actually run

Two steps, the first of which has to pass for the second to mean anything.

1. **Grid-free ground truth.** Mean particle displacement per transition,
   in world metres, projected onto the world-frame push direction. This uses no
   grid, no convention, and no fitted anything.
2. **The convention read-out.** Rasterise pre- and post-push occupancy, take
   the centroid of the positive part of the delta (where material *arrived*)
   minus the centroid of the negative part (where it *left*). Under
   `dim0 = world_x` that difference is already a world-frame vector; compare it
   with the world push. Then repeat with the axes swapped, which is exactly the
   pre-fix hypothesis.

## Numbers

n = 1876 pushes, 6 files:

| quantity | value |
|---|---|
| pushes moving material forward in world metres (grid-free) | **99.7%** |
| mean cos(grid transport, world push), **dim0 = world_x** | **+0.9792** |
| mean cos(grid transport, world push), dim0 = world_y | **−0.0012** |

The competing convention does not merely score worse — it scores **zero**, the
exact no-information value. Swapping the axes makes the grid's account of where
material went statistically independent of where the blade actually pushed it.

## What this adds

It retroactively explains the magnitude of EXP-0001's effect. The pre-fix
operator was not learning a slightly-misaligned transport map; it was being
asked to predict transport along an axis **orthogonal to and uncorrelated
with** the one material actually moved along. That is why it lost to
persistence rather than merely underperforming, and why the fix was worth ~49
points rather than a few.

It also closes the one criticism of C-018 that internal evidence could never
answer, so C-018 moves to grade `high`.

## What would change the verdict

Almost nothing available offline — this is a property of recorded physics. The
residual assumptions are that `p_starts`/`p_stops` are in the same world frame
as `states` (they are both written by the same simulator step) and that the
tray is not symmetric in a way that hides a flip. The second is worth one
thought: the tray IS square and roughly symmetric, which is precisely why a
transpose survived undetected for so long — but a *transpose* is not a symmetry
of the push direction, which is what this measures, and the ±0.98 vs 0.00 split
shows it is not hidden here.

## Threats

- The mean displacement in step 1 averages over all particles including
  untouched ones, which dilutes but cannot reverse the signal; 99.7% forward
  makes dilution irrelevant.
- Single dataset. It is the constrained domain the programme cares about, and
  the effect is not marginal, but a second geometry would cost minutes.
- Considered and dismissed: **that the grid delta centroid is dominated by
  a few cells.** The measure is mass-weighted over the whole field, and the
  result holds at n=1876 with a near-unit cosine.

## Unrelated findings

none.
