# A3 -- wall-proximity stratification: is the wall channel worth training?

RUN-0012. Compares `nfd_randlen` (world-frame baseline,
`Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth`) against `nfd_warped_randlen`
(RUN-0005, `Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth`) on the SAME
transitions, per-transition swept-region `accuracy`, stratified by a
wall-proximity statistic. Both models ran on **CPU** (asserted in-script:
`next(model.parameters()).device == cpu` for both, and every prediction batch
stayed on `cpu` -- no CUDA touched, per the task's "keep this on CPU" brief; no
training was run).

## The statistic

**Primary: `d_wall` = the Euclidean-axis distance, in mm, from the push END
point `(ex, ey)` (world metres, from the action tuple) to the nearest
workspace boundary**:

```
d_wall = min(ex - x_min, x_max - ex, ey - y_min, y_max - ey)
```

Chosen over the swept-rectangle-minimum-distance or margin-fraction
alternatives because it needs only the action tuple already loaded (no plate
half-width, no mask integration), and it is exactly the point most likely to
interact with a wall after a push directed toward one. It is a slightly
optimistic (upward-biased) proxy for the true swept-rectangle clearance: it
does not subtract the plate's own half-width. Not hidden -- stated here.

**Distribution (sanity check, so the stratification is not reporting a
degenerate split):**

| corpus | n | d_wall min/p10/p50/p90/max (mm) |
|---|---|---|
| randlen_test | 10751 | 6.5 / 14.1 / 31.3 / 51.6 / 63.8 |
| L20mm | 7680 | 21.5 / 37.7 / 49.5 / 57.7 / 63.6 |
| L40mm | 7680 | 21.6 / 32.4 / 43.6 / 56.0 / 63.7 |

All three spread over a wide, non-degenerate range with roughly equal-sized
quantile bins (workspace half-extent is 64 mm) -- the stratification
discriminates.

## Metric note (bug found and fixed during this run)

`accuracy = 1 - mean(err_pred) / mean(err_pers)` is a **ratio of population
means**, not a mean of a per-row ratio (see `fit_linear_foresight.metrics`).
A first attempt computed the ratio PER ROW before aggregating and produced
accuracy values in the millions, because some individual pushes have a
near-zero persistence error in the swept band (little material moved) and
the per-row division blows up. Fixed: per-row RMS errors
(`err_pred`/`err_pers`) are kept un-ratioed, and only the stratum-level
aggregation (mean of `err_pred` over mean of `err_pers`, per stratum) forms
the ratio -- the same order of operations `metrics()` uses at the whole-
population level. Uncertainty on the delta is a **bootstrap** over rows
within each stratum (2000 resamples, mean/SEM/95% percentile CI), not a naive
SEM of a per-row ratio, for the same reason.

## randlen_test (primary corpus)

Overall: baseline_acc=0.4564, warped_acc=0.4000, delta=**-0.0564** (boot SEM
0.0011, 95% CI [-0.0585,-0.0544]) -- reproduces RUN-0009's published baseline
number (0.4564) exactly.

**Stratified by wall distance** (5 quantile bins, closest -> farthest):

| bin | n | d_wall mean (mm) | base_acc | warp_acc | delta | 95% CI |
|---|---|---|---|---|---|---|
| 0 | 2151 | 13.7 | 0.456 | 0.384 | -0.0718 | [-0.078,-0.066] |
| 1 | 2243 | 25.1 | 0.436 | 0.379 | -0.0563 | [-0.061,-0.052] |
| 2 | 2057 | 31.5 | 0.456 | 0.414 | -0.0420 | [-0.046,-0.038] |
| 3 | 2150 | 41.1 | 0.463 | 0.409 | -0.0543 | [-0.058,-0.050] |
| 4 | 2150 | 52.3 | 0.471 | 0.413 | -0.0582 | [-0.063,-0.054] |

Shape: worst at the closest bin (-0.072), *best* in the middle (-0.042), then
worse again at the farthest bin (-0.058) -- **U-shaped, not a monotonic
"deficit grows near walls."**

**Confound check -- stratified by push length** (same 5 bins):

| bin | n | len mean (mm) | base_acc | warp_acc | delta |
|---|---|---|---|---|---|
| 0 | 2151 | 14.8 | 0.405 | 0.251 | **-0.1541** |
| 1 | 2150 | 28.3 | 0.451 | 0.380 | -0.0703 |
| 2 | 2150 | 38.8 | 0.463 | 0.416 | -0.0477 |
| 3 | 2150 | 50.4 | 0.472 | 0.437 | -0.0343 |
| 4 | 2150 | 65.0 | 0.469 | 0.450 | -0.0191 |

Push length produces a delta range of **-0.154 to -0.019** (8x the wall-distance
range of -0.072 to -0.042). `corr(d_wall, push_len) = 0.104` -- weak, so the
two are not strongly entangled by simple correlation, but length is by far the
bigger driver of the deficit's SHAPE in this corpus.

**Within a fixed push-length band** (band 2, 33.3-44.3 mm, n=2150),
re-stratified by wall distance:

| bin | n | d_wall mean (mm) | delta | 95% CI |
|---|---|---|---|---|
| 0 | 430 | 15.1 | -0.0525 | [-0.061,-0.045] |
| 1 | 430 | 26.2 | -0.0398 | [-0.048,-0.032] |
| 2 | 430 | 34.2 | -0.0407 | [-0.049,-0.033] |
| 3 | 430 | 44.9 | -0.0486 | [-0.057,-0.041] |
| 4 | 430 | 55.0 | -0.0568 | [-0.066,-0.048] |

With length held fixed, the wall-distance delta is **nearly flat** (-0.040 to
-0.057, non-monotonic, all CIs overlapping neighbours) -- no clean
length-independent wall signal survives in this corpus.

## L20mm / L40mm (single-nominal-push-length cells -- a cleaner natural control)

These cells hold push length near-constant by construction (single nominal
length per cell; PLAN.md's own caveat that this makes them a weak testbed for
length generalisation is exactly what makes them a CLEAN wall-proximity probe:
`corr(d_wall, push_len)` is -0.09 (L20mm) / 0.02 (L40mm), i.e. **wall distance
and length are naturally uncorrelated here**).

**L20mm**, stratified by wall distance (closest -> farthest):

| bin | n | d_wall mean (mm) | base_acc | warp_acc | delta |
|---|---|---|---|---|---|
| 0 | 1536 | 34.8 | 0.345 | 0.144 | **-0.2019** |
| 1 | 1536 | 45.3 | 0.383 | 0.287 | -0.0964 |
| 2 | 1536 | 49.5 | 0.369 | 0.307 | -0.0618 |
| 3 | 1536 | 52.9 | 0.410 | 0.376 | -0.0340 |
| 4 | 1536 | 58.1 | 0.459 | 0.447 | -0.0120 |

Monotonic, ~17x range (-0.202 -> -0.012). Within the most-populated
(near-constant) push-length band (n=1527, still all "20.0mm" nominal),
re-stratified by wall distance: **-0.182, -0.046, -0.061, -0.036, -0.019** --
still a strong, largely monotonic near-wall penalty, this time WITH length
held fixed (survives the confound control).

**L40mm**, stratified by wall distance:

| bin | n | d_wall mean (mm) | base_acc | warp_acc | delta |
|---|---|---|---|---|---|
| 0 | 1549 | 31.2 | 0.458 | 0.381 | -0.0774 |
| 1 | 1523 | 38.6 | 0.495 | 0.429 | -0.0659 |
| 2 | 1536 | 43.6 | 0.522 | 0.480 | -0.0419 |
| 3 | 1536 | 49.4 | 0.528 | 0.485 | -0.0434 |
| 4 | 1536 | 56.3 | 0.520 | 0.486 | -0.0334 |

Monotonic-ish, ~2.3x range. Within a fixed length band (n=2744), re-stratified
by wall distance: **-0.070, -0.044, -0.027, -0.027, -0.018** -- monotonic,
length held fixed, ~4x range.

One `push_len` bin in L40mm's confound-check table came back empty (2 of 5
quantile edges collided, since length is a near-delta distribution there) --
handled by falling back to the most-populated band for the within-band check,
not silently dropped.

## Interpretation

**The wall-proximity effect is real and replicates (2/2) in the cells where it
can be cleanly isolated from push length** (L20mm, L40mm: monotonic,
length-independent, 4-17x range surviving a length-band control). **But in
`overnight_randlen_test` -- the corpus this decision should actually be based
on, per PLAN.md's own note that L20/L40 are "a weak testbed for anything that
has to generalise across a length continuum" -- push length is by far the
dominant driver of the deficit's shape (8x the range of wall distance), and
once length is held fixed the wall-distance delta goes nearly flat and
non-monotonic.** These two findings are not contradictory: `randlen_test`'s
push lengths span 0-80 mm (vs. L20/L40's near-single value), so length
variance swamps whatever wall signal exists, and a single mid-length band
(n=2150) may simply not have enough near-wall extreme cases combined with that
length to resolve a smaller effect the way the wider L20/L40 samples could.

## Verdict

**Not a clean "yes."** On the corpus that matters for the training decision
(`overnight_randlen_test`), the deficit's shape is dominated by push length,
not wall proximity, and the wall-distance effect does not survive controlling
for length there. The wall-proximity effect that IS confirmed, twice, only
shows up in the single-length cells, where it cannot be separated from
"whatever else differs near the workspace edge in a slate-collection corpus at
one push length" (e.g. this could also be conflated with position within the
workspace more generally). **A wall channel is likely to help a MINORITY of
near-wall transitions at best, not rescue the deficit's dominant driver.**
Given the deficit's shape on the real corpus is mostly a length effect (already
flagged as a known, larger issue -- RUN-0005's own known handicap from the
degenerate x8 augmentation, and PLAN.md's Phase A4 fair re-run), spending 3h on
the wall channel before A4 (the fair re-run) is not well motivated by this
diagnostic. If A4's fair re-run still shows a length-independent, wall-linked
residual deficit on `randlen_test` specifically, that would be a stronger case
to revisit the wall channel -- this run does not find that residual on the
corpus it needs to be found on, though it does not rule it out either (a
single mid-length band at n=2150 is not a lot of statistical power to detect a
smaller wall effect once length is controlled).

## Caveats
- RUN-0005 (the warped checkpoint scored here) carries the documented 4x
  under-sampling handicap from the x8 augmentation collapsing to 2 distinct
  canonical inputs (PLAN.md). That handicap is argued to be roughly uniform
  across wall proximity, so it should not distort the delta's SHAPE across
  strata -- but it does inflate the absolute deficit magnitudes reported here.
- `d_wall` does not subtract plate half-width (stated above) -- a systematic,
  not random, upward bias on the true clearance; does not change bin ordering.
- Only ONE within-band control was tried per corpus (the most populated
  push-length band); a full 2D (length x wall-distance) grid was not built
  for budget reasons.
