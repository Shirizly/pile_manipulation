# EXP-0025 — flow-warp NFD pilot: results

**All cells: a subset of `slates_multistep/n20_L20mm_train`, ~40 epochs, one
shared recipe. Comparable to EACH OTHER and to the direct control, and NOT to
the fully-trained arms in EXP-0022.** Scored on the L20mm eval cell through
`Baselines/common/eval_report.py`, the same harness as everything else.

## The idea under test

Instead of emitting the next occupancy, predict a per-pixel **displacement
field** and warp the current occupancy through it (backward warping via
`grid_sample`). Motivation: mass-conserving by construction, residual by
construction, far lower-dimensional output — and only ~1.75% of pixels change
per push, so a direct model spends almost all its capacity re-emitting its
input.

## Numbers

`slateN`, goal-averaged (leads — the metric that decides); `accuracy` beside it,
flagged suspect per `experiments/METRICS.md`.

| cell | lyapunov | mass_in_region | signed_mass | accuracy |
|---|---|---|---|---|
| `random` (ranking floor) | −0.008 | −0.005 | −0.012 | n/a |
| **`flow_direct_control`** (direct prediction) | 0.608 | **0.738** | **0.487** | **0.294** |
| `flow_coarse16` (flow at 16x16, upsampled) | **0.754** | 0.517 | 0.447 | 0.213 |
| `flow_baseline` (flow at 64x64, max 12 px) | 0.586 | 0.430 | 0.472 | 0.141 |
| `flow_smalldisp4` (max 4 px) | 0.535 | 0.332 | 0.268 | 0.160 |
| `flow_srcsink` (+ source/sink term) | 0.656 | 0.316 | 0.433 | 0.089 |
| `flow_largedisp24` (max 24 px) | 0.374 | 0.151 | 0.232 | **−0.027** |

`persistence` is excluded as a ranking floor — it predicts dv=0 for every
candidate and is degenerate.

## Verdict: the flow head does NOT beat direct prediction

The direct-prediction control wins on `accuracy` by a wide margin (0.294 vs
0.213 for the best flow cell) and on 2 of 3 `slateN` value functions. This
reproduces the prior negative result this approach already had in this project,
now with a like-for-like control and a parameter sweep rather than a single
configuration.

**One genuine exception, worth not burying:** `flow_coarse16` **beats the
direct control on `slateN`/lyapunov**, 0.754 vs 0.608 — and `slateN` is the
metric that decides. It loses on the other two value functions, so this is not
a win, but it is the kind of single-value-function disagreement
`experiments/METRICS.md` warns is common and it is the one result here that
would justify looking again.

## Which design parameters mattered

- **Flow resolution is the biggest lever, and coarse wins.** 16x16 upsampled
  (0.213) clearly beats full 64x64 (0.141). A smooth, low-dimensional field is
  easier to learn than a per-pixel one, which is the whole argument for the
  approach and is the one place it holds up.
- **Displacement bound matters, and the failure is asymmetric.** 24 px is
  catastrophic — `accuracy` goes **negative**, i.e. worse than predicting no
  change at all, presumably because a large bound makes `grid_sample`'s
  gradients noisy and lets the field sample from far away. 4 px (0.160) and
  12 px (0.141) are close, so the useful range is narrow and low.
- **A source/sink term hurts** (0.089 vs 0.141 for the same configuration
  without it). The physical argument for adding one is real — material does
  leave the 2D view as cubes stack — but it evidently costs more in lost
  conservation than it buys in expressiveness at this scale.

## Caveats

- Short-epoch cells on a data SUBSET. This ranks design choices against each
  other; it does not establish any cell's absolute quality.
- `n20_L20mm` is a **single-push-length** corpus, so in particular it cannot
  show whether a flow head generalises across push lengths — the one thing a
  displacement field ought to be good at. A negative here is therefore weaker
  evidence than it looks, and `flow_coarse16`'s lyapunov win correspondingly
  more interesting.
- Single training run per cell; no seed-level noise floor, so the ordering
  among the middle cells should not be over-read.
