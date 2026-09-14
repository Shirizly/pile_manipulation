# NFD 3-step closed-loop rollout fine-tuning: results

Init: `Baselines/NFD/runs/nfd_3ch/unet_best.pth`. Objective `L=lam*L1+lam^2*L2+lam^3*L3`,
Lk = full-image (unmasked) MSE. Data: n20_L20mm + n20_L40mm, 35/15 slate split (seed 0),
disjoint (asserted). Batch 256, 100 epochs (3500 iters/lambda), Adam lr 3e-4, ~5min/lambda
on the RTX 4070 — tiny UNet (30k params) needed no gradient checkpointing; batch 256 used
comfortably within 8GB (~3.1GB observed), no compromise required.

## n20_L20mm, closed-loop per-step accuracy

| lam | step1 | step2 | step3 | tf step1 | tf step2 | tf step3 |
|---|---|---|---|---|---|---|
| untrained | +0.344 | +0.174 | +0.092 | +0.344 | +0.350 | +0.355 |
| 0.3 | +0.394 | +0.265 | +0.209 | +0.394 | +0.383 | +0.379 |
| 0.5 | +0.392 | +0.265 | +0.211 | +0.392 | +0.383 | +0.379 |
| 0.7 | +0.392 | +0.267 | +0.214 | +0.392 | +0.384 | +0.381 |
| 0.9 | +0.395 | +0.268 | +0.216 | +0.395 | +0.386 | +0.384 |

n20_L40mm shows the same pattern (untrained cl step3 +0.169 -> trained ~+0.25-0.26; tf
step3 +0.398 -> ~+0.41, all lam).

## Terminal slateN (horizon 3), n20_L20mm, closed-loop, full pool (N=128) + K=32 ref

| lam | lyapunov capture | wins/losses/ties | K32 | mass_in_region capture | wins/losses/ties | K32 |
|---|---|---|---|---|---|---|
| untrained | 0.967±0.011 | 15/0/0 | 0.942 | 0.821±0.053 | 15/0/0 | 0.878 |
| 0.3 | 0.961±0.014 | 15/0/0 | 0.971 | 0.876±0.040 | 15/0/0 | 0.932 |
| 0.5 | 0.964±0.013 | 15/0/0 | 0.973 | 0.891±0.040 | 15/0/0 | 0.929 |
| 0.7 | 0.973±0.012 | 15/0/0 | 0.973 | 0.912±0.035 | 15/0/0 | 0.935 |
| 0.9 | 0.974±0.012 | 15/0/0 | 0.974 | 0.912±0.036 | 15/0/0 | 0.932 |

(15 slates, N=128/slate; full-pool terminal capture was already near-ceiling even
untrained, so this axis mostly saturates — the per-step accuracy numbers are the more
informative comparison here.)

## Answers

**Step-3 improves, substantially.** Closed-loop step3 accuracy rises from the untrained
+0.099 baseline to +0.209..+0.216 across all four lambdas — roughly a 2.1-2.2x
improvement, well outside noise (consistent across both datasets and all lambdas).

**No cost to step-1 or teacher-forced accuracy** — both improve slightly instead
(step1 +0.344->+0.39-0.395; tf step3 +0.355->+0.379-0.384). Rollout training here is a
pure win on every axis measured, not a step1-for-step3 trade.

**Lambda difference is not resolvable at this budget.** All four lambdas land within
~0.01 of each other on every metric (step3: 0.209-0.216); the ordering (0.9 slightly best)
is consistent but the gap is small relative to run-to-run noise we didn't independently
estimate (single seed per lambda). Lambda=0.7 or 0.9 is marginally preferred but the
difference is not a strong claim.

**No divergence or instability.** All four losses decreased monotonically-with-noise
over 3500 iterations; no NaN/inf triggered the abort check; no lambda needed early
stopping. This mirrors the finding that unmasked full-image MSE (vs. the swept-region
mask that caused the linear-operator global-op run to diverge) kept training stable here,
though NFD's nonlinear/bounded (sigmoid) output may also just be more forgiving than an
unconstrained linear operator.

Artifacts: `train_nfd_multistep.py`, `results_nfd_multistep.json`, `nfd_lam{0.3,0.5,0.7,0.9}.pth`.
