# EXP-0016 results

Machine-readable: `RESULTS.json`, `slaten_corner*.json`, `batch_headroom.json`,
`../artifacts/RUN-0002/dynamics_metrics.json`,
`../artifacts/RUN-0003/decoder_metrics.json`.

## slateN — DS-0001, `corner`/`lyapunov`, step 0, n = 20 slates

Mean over 3 dynamics seeds; the bracket is the range of the per-run
between-slate sem.

| cell | slateN | sd over dyn seeds | between-slate sem |
|---|---|---|---|
| random-encoder K=1 | **+0.4191** | 0.0117 | 0.065–0.080 |
| random-encoder K=4 | +0.4117 | 0.0596 | 0.052–0.071 |
| random-encoder K=8 | +0.3765 | 0.0620 | 0.065–0.091 |
| decode-z0 (no dynamics) | −0.0042 | 0 | 0.089 |
| persistence (degenerate ranker) | −0.0042 | 0 | 0.089 |
| **random (ranking floor)** | −0.1409 | 0 | 0.101 |

EXP-0014 reference rows on the identical cell: MODEL-0001 +0.7399,
MODEL-0003 +0.5163, MODEL-0002 −0.0904.

**K=1 vs K=8 is UNRESOLVED** at this power.

## Latent — test R² against the `Δz = 0` baseline, same latent space

Comparable ONLY within this encoder; a later cell that fine-tunes the encoder
moves both the prediction and the denominator.

| cell | R² test (mean ± sd, 3 seeds) | R² test per-dim | R² train | gate entropy (nats) | params |
|---|---|---|---|---|---|
| Δz = 0 | 0.0000 (exact) | 0.0000 | — | — | 0 |
| K=1 | +0.2483 ± 0.0004 | +0.2358 | +0.2790 | 0 (single mode) | 113,633 |
| K=4 | +0.3683 ± 0.0024 | +0.3508 | +0.4588 | 1.028 (max 1.386) | 335,972 |
| K=8 | +0.3726 ± 0.0012 | +0.3544 | +0.4863 | 1.710 (max 2.079) | 632,424 |

**K=1 → K=8 = +0.1243, ≈50× the cross-seed sd. RESOLVED, and on a RANDOM
ENCODER.** K=4 vs K=8 (+0.0043) is marginal — unresolved.
0.0 % of K=8 test rows have a gate max > 0.9: soft blending, not switching.

## Latent statistics of the frozen random encoder (98304 train states)

| quantity | value |
|---|---|
| per-dimension std of z | mean 0.0531, min 0.0389, max 0.0951 |
| ‖mean(z)‖ | 5.52 (vs centred rms 0.054) |
| **effective rank** | **52.6 of 256** |
| rms(Δz) / rms(z centred) | 0.738 |

Not per-dimension degenerate, but spectrally so: ~80 % of the nominal
dimensionality is unused.

## Decoder (frozen random z)

| quantity | value |
|---|---|
| train mse / test mse | 0.00378 / 0.0331 |
| autoencoding accuracy vs empty frame | −0.111 |
| accuracy, swept region (8.1 % of frame), K=1 | +0.137 |
| accuracy, swept region, K=8 | +0.191 |
| accuracy, swept region, decode-TRUE-next-z ceiling | +0.174 |
| accuracy, full frame (every cell, ceiling included) | ≈ −1.00 |

K=8 exceeds its own oracle ceiling — these are noise around a decoder that
cannot resolve the swept band.

## Goal degeneracy on DS-0001

frac(dv_true == 0) = 0.031; dv_true mean +0.00066, sd 0.0257; 51.0 % of
candidates improving. Not degenerate.

## Cost (RTX 4070 Laptop, 8 GB)

See EXPERIMENT.md's cost table. Max batch at res64: encoder forward no-grad
**2048** (4160 MiB, 4096 OOMs); decoder training step **1024** (4013 MiB,
2048 OOMs).
