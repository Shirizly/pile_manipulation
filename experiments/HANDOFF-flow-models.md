# Handoff — flow / advection dynamics models

For a session picking up the flow line. Read this before touching anything;
it exists so you do not re-derive what is already measured.

## Bottom line first

**The flow parameterisation as built is a DEAD END, and the evidence is a
ceiling, not a training result.** Do not start by training a better flow model.
Start by raising the ceiling, because the ceiling is model-free and cheap to
measure, and the current one says no amount of training can win.

## The ceiling measurement — the single most important number here

Take the TRUE next state's particle displacements, build the exact
ground-truth flow field, warp `occ0` through it, and score the result as if it
were a prediction. **A perfect flow model scores this and no higher.**

Measured on the **L20mm eval cell**:

| | accuracy | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| **ground-truth flow field (the ceiling)** | **0.167** | **0.813** | **0.770** | **0.512** |
| NFD baseline, *achieved* | 0.407 | 0.853 | 0.820 | 0.753 |
| LinearForesight switched res32, *achieved* | 0.262 | 0.902 | 0.830 | 0.705 |

**The ceiling is below what two existing baselines already achieve, on every
metric.** That is what makes it a dead end rather than an underperforming idea.

Code: `experiments/EXP-0025-flow-warp-nfd-pilot/code/`. Re-run the ceiling
before and after any change to the parameterisation — it takes minutes and
needs no training.

**Note a correction that was made mid-investigation:** the ceiling was first
compared against the flow sweep's own weak in-sweep control (0.608 lyapunov,
a subset-trained short-epoch model) and looked like headroom. It is not.
Always compare a ceiling to the best ACTUAL model on the same corpus and value
function, never to a weak control from inside your own sweep.

## Why it is capped — three separate mechanisms, do not conflate them

1. **Bilinear `grid_sample` blurs a near-binary occupancy field even at ground
   truth.** Stratification shows accuracy is WORST in the near-static bin and
   improves with displacement, i.e. the blur costs most exactly where
   persistence error is already tiny and `accuracy`'s denominator is smallest.
   This is the dominant cap on `accuracy`.
2. **Capture radius.** Photometric gradient descent can only align features
   that already nearly overlap — about one feature width. A cube is ~5 mm
   ~ 2.5 px at 2 mm/px; measured per-particle displacements reach **6.5 px at
   p95 and 20 px at max**. So a from-scratch photometric fit literally cannot
   find the large displacements. This caps LEARNING, not the ceiling.
3. **The target field is rigid translation only** — no rotation, and stacked
   cubes collide in the 2D projection. This caps the CEILING itself and is the
   most promising thing to attack.

## What was actually run

`experiments/EXP-0025-flow-warp-nfd-pilot/` (record: `EXPERIMENT.md`, verdict
`refuted`, register row C-028).

Photometric sweep, all on a subset of `slates_multistep/n20_L20mm_train`,
~40 epochs, one shared recipe, comparable only to each other:

| cell | lyapunov | mass_in_region | signed_mass | accuracy |
|---|---|---|---|---|
| direct-prediction control | 0.608 | **0.738** | **0.487** | **0.294** |
| flow, coarse 16x16 upsampled | **0.754** | 0.517 | 0.447 | 0.213 |
| flow, full 64x64 | 0.586 | 0.430 | 0.472 | 0.141 |
| flow + source/sink term | 0.656 | 0.316 | 0.433 | 0.089 |
| flow, max displacement 4 px | 0.535 | 0.332 | 0.268 | 0.160 |
| flow, max displacement 24 px | 0.374 | 0.151 | 0.232 | **-0.027** |

Then directly-supervised flow (particle correspondence gives an exact target):

| cell | lyapunov | mass_in_region | signed_mass | accuracy |
|---|---|---|---|---|
| supervised + masked magnitude penalty | 0.707 | 0.559 | 0.411 | 0.180 |
| supervised, no penalty | 0.576 | 0.454 | 0.256 | 0.150 |

## What those design parameters taught

- **Coarse beats full resolution** (0.213 vs 0.141). Not a capacity effect —
  parameter counts are IDENTICAL (30,546); the coarse cell is the same network
  with its output field average-pooled to 16x16 and bilinearly upsampled back.
  It is pure regularisation. The flow is never supervised directly in the
  photometric cells, so `dloss/dflow = grad(occ0)` at the sample point, which
  is ZERO wherever the image is locally flat — most of a sparse occupancy grid.
  Unconstrained vectors drift freely and are garbage at test time. Pooling ties
  each vector to a 4x4 block so it is more likely to touch material.
- **A 24 px displacement bound is catastrophic** (accuracy NEGATIVE, worse than
  predicting no change). Widening the search does not widen the basin of
  attraction. 4 px scored best of the unpooled cells even though it cannot
  physically express a 20 mm push — meaning the head was learning a small local
  correction, not transport.
- **A source/sink term hurts.** It removes the one hard constraint that was
  regularising the problem and gives the optimiser a cheaper way to cut loss
  (fabricate/delete mass) than learning a correct displacement.
- **Masked magnitude shrinkage beats no penalty**, but only once data is
  adequate — see the confound below.

## Two traps that already cost real time here

1. **The x8 augmentation is NOT valid for a vector field without extra work.**
   `training/trainer.py::_augment_eulerian_batch` transforms
   `input`/`target`/`physics`/`push_px` and silently DROPS anything else, and
   even for `push_px` it permutes point COORDINATES — it has no notion of
   rotating a vector's COMPONENTS. `model/flow_nfd/supervised.py::augment_flow_batch_x8`
   does it correctly; verified by equivariance
   (`warp(R(occ0), R_vec(R(flow))) == R(warp(occ0, flow))`, max abs diff
   3.8e-6) with a negative control that skips the component rotation and gives
   max abs diff 1.0, so the check has power. **Reuse that, do not re-derive.**
2. **A dropped-augmentation confound inverted a conclusion once.** The first
   supervised cells trained without augmentation, seeing 768 effective samples
   against the other cells' 6,144. Under that starvation the masked-penalty
   cell looked COLLAPSED (slateN below the random floor); with augmentation
   restored it became the BEST supervised cell (lyapunov -0.018 -> 0.707).
   Always match effective sample count before comparing.

## Target-field convention (verified, do not re-derive)

Backward warp: `out(x) = occ0(x + f(x))`. For material ending at pixel `x` and
coming from `x0`, the target is `f(x) = x0 - x` — the **NEGATIVE displacement,
indexed at the DESTINATION pixel**. Channel 0 is `dcol`, channel 1 is `drow`,
matching the wrapper's `[x/col, y/row]` order. `states`/`states_` in every
`_data.pt` are `(B, 20, 7)` with **consistent particle indexing**, so
correspondence is exact. Collisions resolved by nearest-destination-centre;
rotation is NOT modelled, which is a real ceiling limitation.

## Where the code lives

`model/flow_nfd/` — `lib.py` (registers `nfd-flow-warp`), `predictor.py`,
`supervised.py`. Moved out of `Baselines/` this session per the rule that a
non-trivial new model family gets its own directory under `model/`.
Configs stay in `Baselines/NFD/configs/nfd_train_flow_*.yaml`. Scoring is
`Baselines/common/eval_report.py` (six flow cells are registered in `MODELS`).

## If you continue — what to do first

**Raise the ceiling, then re-measure it, and only train if it clears the
baselines above.** Candidates in order of expected value:
1. **Hybrid: warp PLUS a small additive residual correction.** Breaks the pure
   advection constraint that caps the ceiling, and keeps mass-conservation as a
   soft prior rather than a hard one. Most likely to lift the ceiling above the
   baselines.
2. **Model rotation**, not just translation, in the target field.
3. **Coarse-to-fine on the IMAGE** (not just the flow field). The coarse-16
   cell pooled the flow but left the image at 64x64, so it never addressed
   capture radius at all. Downsampling the image turns a 20 px displacement
   into 2.5 px, inside the basin; solve there, refine upward. This only helps
   LEARNING, so it is worthless until the ceiling is fixed.

## One live caveat on all of it

Everything above is measured on `n20_L20mm`, a **single-push-length** corpus. A
displacement field's natural advantage is expressing how far material moves as
a function of push length — the one thing that corpus cannot show. So the
negative is weaker evidence than it looks, and re-measuring the ceiling on
`overnight_randlen` is cheap and worth doing before closing the line.
