# RUN-0000 — differentiability gate (must pass before any optimisation)

`DESIGN.md` blocker 3. A silently-zero gradient makes this whole benchmark
measure nothing while producing plausible numbers, so this runs first and
fails loudly.

## Exact argv
```
OMP_NUM_THREADS=4 PYTHONPATH=. python -u \
  experiments/EXP-0023-model-as-gradient-source/code/verify_gradients.py \
  --out experiments/EXP-0023-model-as-gradient-source/artifacts/grad_check.json
```

## What it asserts, per registered arm
1. `simple_mpc.adapters.assert_dv_convention` — the Lyapunov value is
   "lower = closer to goal" and `dv` of an improving push is negative.
2. `dv` is finite; `d(dv)/d(action)` is finite.
3. The gradient is non-zero in **every one of the four** action components
   (`sx, sy, ex, ey`) and no row has a zero gradient vector.
4. The per-row directional derivative along the analytic gradient agrees in
   SIGN with a central finite difference (h = 1e-4 m) for **100 %** of rows.

Anything less is reported as `DEGENERATE`, not silently passed.

## Result — all 7 registered arms PASS

| arm | mean ‖∇dv‖ | zero-grad rows | dir-FD sign agree | dir-FD median rel err |
|---|---|---|---|---|
| `linear_switched_hard` | 18.0 | 0.00 | 1.00 | 0.149 |
| `linear_switched_soft` | 17.5 | 0.00 | 1.00 | 0.148 |
| `nfd_3ch` | 5.95 | 0.00 | 1.00 | 0.045 |
| `nfd_3ch_finetuned` | 3.32 | 0.00 | 1.00 | 0.040 |
| `nfd_3ch_randlen` | 3.08 | 0.00 | 1.00 | 0.019 |
| `nfd_residual_warped` | 10.3 | 0.00 | 1.00 | 0.257 |
| `nfd_warped_randlen` | 21.3 | 0.00 | 1.00 | 0.134 |

Raw per-component numbers: `artifacts/grad_check.json`.

## Two things this run settled that the design left open

- **`predict_switched`'s masked in-place assignment into a clone**
  (`DESIGN.md` blocker 3) does survive autograd — `linear_switched_hard`
  passes with non-zero gradients in all four components. The hard gate's
  defect is narrower than "no gradient": the warp, the operator and the
  occupancy path all carry gradient, and only the *choice of operator* is a
  step function in push length, so the length gradient is missing exactly
  the term that says "a longer push would use a different operator". That is
  what `predict_switched_soft` restores.
- **The residual/warped arms have the loosest FD agreement** (median 15–26 %
  relative error at h = 1e-4 m). Expected: `grid_sample` bilinear warps and
  the sharp `draw_plate_soft` sigmoids make the objective strongly curved at
  that scale. Sign agreement is 1.00 for every arm, which is the property
  that matters for descent.
