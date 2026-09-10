# Prediction on pushed piles — a corrected baseline and a wide hypothesis set

**Date:** 2026-09-03 · **Branch:** `VisualForesight`
**Inputs:** [`reports/linear_foresight_report.md`](../reports/linear_foresight_report.md),
[`docs/linear_foresight_findings.md`](linear_foresight_findings.md),
[`docs/ideas_log.md`](ideas_log.md) — `docs/sand_manipulation.md` was also an
input as of 2026-09-03; it was deleted with the sand path (`docs/rejected_mpm_sand.md`)
**Evidence records:** the measurements below are recorded as `EXP-0001`…`EXP-0006`
in [`docs/experiments/`](experiments/README.md); the claims they support or
invalidate are tracked in [`REGISTER.md`](experiments/REGISTER.md).
**Scope:** one session, no data collection, no training. Everything below was
run on data already on disk, on CPU. Reproduction commands in §6.

**Amendment, 2026-09-05.** The MPM sand arm was withdrawn as non-physical (it
simulated a soft continuum, not granular material — see
`docs/rejected_mpm_sand.md`); the sand code and datasets were deleted in commit
`e9b83f99`, including `docs/sand_manipulation.md`, so links to it below are now
dead and are being replaced with `docs/experiments/EXP-####` citations. **What
changed:** the regime table in §1.4 and the evidence table in §9 have been
redone on cube-only data; H-A2's view-vs-blur comparison is restated to
cube-only scope with a weaker view finding; H-B7's cross-material prediction is
marked withdrawn; and a stale "mass conservation hurt" claim under B2 is
corrected against the cube-only measurement that superseded it. **What did not
change:** the frame-bug diagnosis (§1.1-§1.3), the regime-margin flatness claim
(§1.4, now resting on four cube rows spanning +0.226..+0.306 instead of five),
H-A2's core claim that bandwidth is the difficulty axis (holds on cubes alone),
and every hypothesis whose evidence was already cube-only.

**Bottom line up front:** while grounding the hypotheses I found that the
occupancy channel and the plate/action channel of every `PileSweepData` sample
are in **mutually transposed frames**. Fixing it turns the central negative
result of the linear-foresight work ("on cubes nothing beats persistence") into
a 41% error reduction, and collapses the "it was pile depth, not granularity"
conclusion. §1 documents that; §2–§5 are the hypotheses, re-derived on top of
the corrected picture.

---

## 1. The frame bug

### 1.1 Evidence

**Source level.** `Genesis/training/dataset.py:_draw_particle_grid` rasterises
cubes with OpenCV: `cv2.circle(grid_np, (round(center_x), round(center_y)), …)`
and `cv2.boxPoints(((center[0], center[1]), …))`. **cv2 takes a point as
(column, row)**, so `center_x = particle_state[0]` (world x) becomes **dim 1**
and world y becomes **dim 0**. The occupancy channel is `dim0 = world_y`.

`transforms/functional.py:draw_plate_soft` builds
`gx, gy = meshgrid(ix, iy, indexing="ij")` and compares `cx = center[:, 0]`
(world x) against `gx`, which indexes **dim 0**. The plate channel is
`dim0 = world_x` — the opposite. Its own docstring claims
`"dataset convention (dim0=world_y, dim1=world_x)"`, which contradicts its body.
`dmdc_baseline.py:45` repeats the same wrong claim.

**Measured.** On `genesis_foresight_L040` (2560 transitions, 50 cubes):

| check | result |
|---|---|
| registry occupancy vs particles re-rasterised at `dim0=world_x`, best IoU over ±4 px shifts | **0.103** |
| same, with the registry grid **transposed** | **0.573** |
| plate-channel centroid vs action midpoint under `dim0=world_x` | (34.5, 26.6) vs (34.1, 25.8) — **0.9 px** |
| plate-channel centroid under `dim0=world_y` | (25.8, 34.1) — **12 px off** |

So the two channels of a single training sample disagree, and the plate channel
is the one that agrees with `actions_to_pixels`.

### 1.2 What it costs

Same fit code, same metric, same seeded episode split, same actions, res 64 /
crop 0.5 / blur 1.0 / mask view / ridge-toward-identity. Error as a percentage
of the change that occurred (100% = no better than predicting nothing moved):

| occupancy source | linear operator | mean-delta | identity (warp only) |
|---|---|---|---|
| registry, as stored (**the path every cube result used**) | **107.1%** | 99.0% | 100.0% |
| registry, transposed | **57.9%** | 87.1% | 98.3% |
| re-rasterised from particles, area-matched | **54.1%** | 86.5% | 98.3% |

Footprint area is not the lever — sweeping the splat from 0.002 m to 0.012 m
moves the operator only 68.6% → 48.9%, monotonically, and the area-matched
row above (`occ_mean` 0.1169 vs the registry's 0.1194) is the fair comparison.
The transpose is the lever.

### 1.3 What this invalidates

- **`reports/linear_foresight_report.md` §1, §2, §2.1–§2.3, §2.7, and the Q2/Q3/Q8
  answers.** Every pixel-operator number on cube data came through
  `load_transition_arrays`. "Nothing beats persistence" does not survive.
  Q4 (mass), Q5 (non-negativity), Q6 (deposit structure) and §3 (the 5th DOF)
  are transpose-invariant or measured elsewhere, and stand.
- **The former `docs/sand_manipulation.md` §8, "RESOLVED: it was pile DEPTH,
  not granularity"** (that document was deleted along with the sand path,
  `docs/rejected_mpm_sand.md`, so this is now a historical pointer only). The
  scattered-monolayer row (mean-delta +0.013, linear +0.010, margin −0.003)
  came from the registry path; the piled and sand rows came from
  `sand_to_mask` (now `points_to_mask`). The comparison crosses the bug
  boundary. §1.4 below redoes it on one path and the conclusion does not hold
  — independent of, and prior to, the later sand withdrawal.
- **Everything trained through `PileSweepData`** — every UNet, NCA, spatial
  transformer and FiLM variant in `runs_cubes/`. See §3-A1.
- The scalar-level results (`variance_decomposition.py`,
  `deltav_predictability.py`, `density_stratified.py`) work from **particle
  positions**, not from the rasterised grid, so the contact-stratification
  findings (§2.6, §2.9 of the report) are unaffected.

### 1.4 The regime comparison, redone on one code path

Mask view, blur 1.0, ridge-toward-identity, occupancy re-rasterised from
particles, explained variance over the swept region, seed-0 episode split
(`docs/experiments/EXP-0002-regime-margin.md`).

**Amended 2026-09-05.** This table originally carried a fifth row, MPM sand
(margin +0.262 / +0.245), as the continuum end of the comparison. That path was
withdrawn as non-physical (`docs/rejected_mpm_sand.md`); the row is removed.
The claim below is unaffected — the four cube rows already spanned
+0.226..+0.306 and the sand row sat inside that range, so nothing about the
flatness result rested on it.

| regime | M | mean-delta | linear | **operator margin** |
|---|---|---|---|---|
| *res 64, crop 0.5* | | | | |
| cubes n50, scattered monolayer, blind pushes | 2560 | 0.120 | 0.411 | **+0.291** |
| cubes n50, scattered, contact-sampled | 2108 | 0.182 | 0.437 | **+0.255** |
| cubes n30, piled heap | 1381 | 0.322 | 0.548 | **+0.226** |
| cubes n20, piled 2 layers | 4840 | 0.332 | 0.638 | **+0.306** |
| *res 64, crop 1.0* | | | | |
| cubes n50, scattered monolayer, blind | 2560 | 0.141 | 0.308 | +0.167 |
| cubes n50, scattered, contact-sampled | 2108 | 0.204 | 0.352 | +0.148 |
| cubes n30, piled heap | 1381 | 0.345 | 0.548 | +0.202 |
| cubes n20, piled 2 layers | 4840 | 0.345 | 0.640 | +0.295 |

Read it as a decomposition. **The operator's marginal contribution over a
zero-parameter constant is roughly flat across every regime** (+0.15…+0.31) —
scattered monolayer cubes included. What the regime changes is how much of the
push's effect is *stereotyped*: mean-delta goes 0.12 → 0.33 from scattered
monolayer to pile or continuum, and total predictability goes 0.31 → 0.64.

So depth does not make the dynamics "more linear". It makes them
**more repeatable**, which raises the floor every model stands on and raises the
ceiling too, but it does not change how much state-dependence a linear map can
capture. (The original version of this claim also covered a continuum-vs-depth
axis via the sand row; with that row withdrawn, the claim is scoped to depth
across cube regimes only — see the amendment above.)

**Caveat:** single seed-0 split holding out 25% of files, not leave-one-run-out.
The report's own noise floor was ~0.004 rms on cube data; these effects are
0.15–0.30 in explained variance, i.e. far larger, but the ordering *within* the
table (n20 vs n30 vs the scattered regimes) is not resolved by one split.

---

## 2. Five axes of difficulty, revised

1. **Frame consistency between the state channel and the action channel.**
   Newly discovered, and it dominated everything else on cube data. Not a
   "difficulty" so much as a bug, but it belongs first because it silently
   masqueraded as one.
2. **Spatial bandwidth of the field.** The largest genuine dial (§3-A2). It
   controls both how much the SE(2) warp destroys and how much of the target is
   predictable at all.
3. **Stereotypy** — the share of the effect a constant canonical-frame
   displacement already explains. Set by pile depth / continuum-ness (§1.4).
4. **Contact variance** — blind action sampling injects one strongly nonlinear
   scalar (report §2.6, §2.9). Unaffected by the bug; still the best-supported
   finding in the repo.
5. **Fit conditioning** (M/D, shrinkage target) and **representation aliasing**
   (§3-A4). Several published negative results live here rather than in physics.

---

## 3. Hypotheses: sources of difficulty

### A1 — The frame bug is why UNet training on scattered monolayers failed
**Claim.** A network whose two input channels are mutually transposed can only
condition on the action by learning a reflection between them. It will mostly
learn to denoise the occupancy channel and largely ignore the action.

**Evidence already gathered** (inference only, `unetfilm_corl_limited_100e`,
200 held-out samples, whole-image rms, sigmoid applied):

| variant | rms | vs persistence |
|---|---|---|
| persistence (copy the input) | 0.12055 | 100.0% |
| UNet, true action channel | 0.11063 | **91.8%** |
| UNet, action channel shuffled across the batch | 0.11442 | 94.9% |
| UNet, action channel zeroed | 0.11633 | 96.5% |

The model beats persistence by 8.2 points, and **only 3.1 of those come from
knowing which push was applied** — the remaining 5.1 are smoothing the current
occupancy. That is the signature of a model that cannot use its action channel.

**Test (needs GPU, flagged not run).** Retrain the same config twice on
`corl_limited`: as-is, and with `_draw_particle_grid` fixed to write
`(center_y, center_x)`. Prediction: the action-shuffled gap widens from 3.1
points to most of the model's advantage, and absolute error drops substantially.
Cheapest possible version — one model, one small dataset, ~1 h.

**Second, independent mismatch worth checking in the same pass.** The MPC path
(`simple_mpc/adapters.py`, `oracle_mpc.py`, `occupancy_reward.py`) builds
occupancy with `particles_to_occupancy`, which is `dim0 = world_x`. Models
trained on the cv2 grid (`dim0 = world_y`) are therefore fed a transposed pile
at deployment as well. If so, every MPC number involving a learned model is
measuring a model outside its training distribution. **Cheap to check** — one
assertion comparing the two rasterisers on a single scene — and it should become
a permanent test.

### A2 — Difficulty is set by the field's spatial bandwidth, not by the material
**Claim, scope note added 2026-09-05.** This title was originally supported by
a cross-material (cube vs sand) invariance; with sand withdrawn, the surviving
evidence is the blur sweep on cubes alone, so read "not by the material" as "not
by depth/granularity" rather than as a tested material contrast. **Claim.**
Blur is not a preprocessing detail; it is the main axis. Sweeping σ
with everything else fixed (mask and density views, error as % of the change,
seed-0 split, `docs/experiments/EXP-0003-blur-vs-view.md`):

**Restated 2026-09-05 to cube-only scope.** This table originally led with an
MPM-sand row and used it as the primary evidence for point (a) below; that path
was withdrawn as non-physical (`docs/rejected_mpm_sand.md`) and every sand
number is removed.

| dataset | view | σ=0 | σ=0.5 | σ=1.0 | σ=1.5 |
|---|---|---|---|---|---|
| cubes n50 scattered monolayer | mask | 89.4% | 83.6% | 67.4% | **56.1%** |
| | density | 97.0% | 93.4% | 75.4% | **64.3%** |

Two consequences, one of them changed by the amendment.

**(a) WITHDRAWN — the view comparison is now inconclusive, not "mask ≈
density".** The original claim ("the two views track each other within a few
points at every level, on both materials") rested mainly on the sand rows,
where the gap was ~3-7 points. On cubes alone the mask view leads density by
**7.6-9.8 points at every σ** — small next to the 33-point blur effect, but not
the "interchangeable" reading this section used to give it. Whether that gap is
a monolayer artifact (no cube dataset here has a density-informative depth) is
untested; EXP-0003 flags re-running the cross on the piled n20/n30 sets as the
next cheap step. This also means the tension with §6's "blur *hurts* density"
(measured on `pile20`) is not resolved by this record — both claims should be
treated as open.

**(b) Blur changes the target, so these numbers are not a free lunch.** A
heavily blurred field is a genuinely easier thing to predict *and* a genuinely
less informative thing to predict. The percentage is normalised within each σ,
but the quantity being normalised shrinks. The honest statement is that σ is a
**bias–variance dial on the prediction target**, and nobody has asked what value
of it is right *for control* — which is H-C1. This half of the claim is
unaffected by the amendment: the blur effect is barely smaller on cubes alone
(89.4% → 56.1%, a 33.3-point span) than it looked with sand included.

**Test.** Plot explained variance against the field's own high-frequency ratio
`|∇²occ|₁/|occ|₁` rather than against σ, across every cube regime and both
views. Prediction: the curves collapse onto one. If they do, "depth" and
"granularity" are proxies for one scalar, and the difficulty axis is
bandwidth — material is no longer part of this claim since sand is withdrawn.
~30 min, no new data.

### A3 — Effect = stereotyped part + state-dependent part, and regimes move only the first
Stated in §1.4. **Test:** redo that table under leave-one-run-out, at three blur
levels, adding `corl` (the general random-action monolayer set) and the real
`chickpeas` data. If the operator margin stays in +0.15…+0.31 everywhere, this
becomes the organising fact of the whole project: model effort buys a roughly
constant increment, and regime choice buys the rest.

### A4 — At pixel level the data is too sparse for any nonparametric method
On piled cubes (4000 transitions, res 32, D=1024, blurred mask) the median
nearest-neighbour distance across episodes is **0.045 rms/px against a field
whose mean is 0.018** — neighbours are not neighbours. Outcome disagreement
between a point and its nearest neighbour is essentially independent of their
input distance over the whole observed range (regression slope ≈ 0). So the
nearest-neighbour Bayes-error estimator does not converge here, and `knn`
retrieval is structurally capped, exactly as `model_zoo.py` measures (renamed
from `sand_model_zoo.py` when the sand path was removed; the measurement itself
was always on piled-cube data, not sand).

**Consequence for hypothesis testing:** the aliasing floor — the ceiling any
model on this representation can reach — is **not measurable at the pixel
level** with the data volumes available. It *is* measurable on a low-dimensional
descriptor (§3-B3) or on a scalar target. **Test:** run the same estimator on
the 25-dimensional PCA coordinates of the blurred mask and on scalar targets
(band displacement, `dV`). ~1 h. This is the missing number in every comparison
table in the repo: everything is scored against persistence and mean-delta, and
nothing against "what is achievable".

### A5 — Several published negatives are conditioning artifacts, not physics
`reports/linear_foresight_report.md` §2.7 (contact-switched pixel operators do
not help) and the crop/ridge sweeps all ran on transposed data, where the
operator had almost no signal to condition. **Test:** re-run §2.7 verbatim on
the corrected grid. Prediction: with the operator now explaining ~0.3–0.6,
contact switching pays at the pixel level after all, and §2.6's prediction —
which §2.7 recorded as refuted — is reinstated. ~1 h, no new data. **This is
the highest-value single re-run in the backlog.**

### A6 — The half-pixel/scale mismatch is a separate, smaller defect
`particles_to_occupancy` normalises with `* (res - 1)`; `actions_to_pixels` uses
`* res - 0.5`. Best alignment between the two rasterisers sits at a (−1, −1) px
shift, not (0, 0). At 64×64 with a 20 px push that is a 5% systematic offset in
the canonical frame. Cheap to fix and cheap to unit-test; fold it into the same
change as A1.

---

## 4. Hypotheses: models that could be especially accurate

The goal explicitly allows narrow domains, so several of these trade generality
for accuracy on purpose.

### B1 — Band-decomposed operators (one linear map per spatial frequency)
**Why.** A2 says the low band is highly predictable and the high band is not,
and a single operator has to serve both with one shrinkage strength.
**Mechanism.** Laplacian pyramid of the canonical image; a separate
ridge-toward-identity operator per level, with per-level λ chosen on held-out
data; reconstruct. Costs one extra hyperparameter per level and nothing else.
**Prediction.** Beats the single full-resolution operator at every σ, and
recovers most of the σ=1.5 accuracy while still predicting a σ=0 field.
**Cost.** Half a day, no new data. This is the cheapest way to convert the blur
finding into an actual model rather than a metric artifact.

### B2 — Predict a displacement field, not an image
**Why.** A push *is* a transport map. Predicting pixel values forces the model
to represent transport as a large sparse matrix, which is why it needs blur to
be well-conditioned.
**WITHDRAWN claim removed 2026-09-05:** this section used to motivate itself
by saying mass conservation "had to be imposed as an awkward constraint... it
hurt" (sourced only to the now-deleted `sand_manipulation.md` §7). That
comparison depended on sand and does not survive
(`docs/rejected_mpm_sand.md`). The cube-only measurement that exists,
`docs/experiments/EXP-0006-spectrum-n20-n30.md`, found the opposite on cubes:
the column-stochastic (mass-conserving) constraint cost ~0.0005 explained
variance (0.6442 vs 0.6447) — nearly free, not harmful, because cubes largely
do not leave the crop. So the motivating claim for this hypothesis is weaker
than originally stated; the "why" now rests only on the representational
argument above (transport-as-sparse-matrix), not on a measured penalty.
**Mechanism.** Estimate the per-transition transport map with Sinkhorn between
the two canonical frames (`simple_mpc/ot_planner.py` already has the machinery),
project onto a small flow basis (8–16 smooth modes in the canonical frame),
regress the basis coefficients linearly on the input image and the action, and
apply the resulting warp. Mass conservation is then exact by construction, in
the *world* frame rather than in the crop.
**Prediction.** Matches or beats the pixel operator on sharp (σ=0) fields, where
the pixel operator is worst, and degrades gracefully at high contact instead of
smearing.
**Cost.** 1–2 days. The most promising structurally new model on this list.

### B3 — Fit the operator on 16–25 PCA modes, not on pixels
**Why.** The rank-16-out-of-25 dimensionality figure this section used to cite
came only from the now-deleted `sand_manipulation.md` §9.3 and cannot be
re-sourced — **flagging rather than keeping it**: it is not backed by any
surviving cube record. What does survive, cube-only, is the general M/D
argument already established in A4 above: cube pixel operators run at
`D = 1024`-4096 with `M` in the low thousands at best (A4's example: 4000
transitions at `D = 1024`), so most of the pixel space is being regularised
away for lack of data regardless of the exact useful rank.
**Mechanism.** PCA the canonical frames, fit a 25×25 operator (plus action
terms), reconstruct. M/D goes from ~1 to ~200.
**Prediction.** Beats the full pixel operator on every cube set, where M is
small relative to D. Also makes A4's aliasing-floor estimator tractable, makes
contact-switching affordable (A5), and makes per-regime specialisation (B6)
affordable.
**Cost.** Half a day. Highest ratio of expected value to effort on this list.

### B4 — Contact-switched linear, at the descriptor level
Report §2.6 established that essentially all the nonlinearity in scalar targets
is one variable. §2.7 said it does not transfer to pixels — measured on
transposed data (A5). Combined with B3 this becomes a well-posed fit: three
operators of 25×25 on ~1500 samples each.
**Prediction.** Recovers most of the linear/nonlinear gap; the paper's model
class, switched on state-action contact instead of on the action alone.

### B5 — Residual over the frozen geometric heuristic
`ideas_log.md` GP1, probe P4: the `cumulative` splat heuristic already makes the
residual 20–30% easier than persistence on every geometry descriptor group,
without being fitted at all. Nobody has run it at the pixel level.
**Mechanism.** Mean function = `differentiable_push_cumulative` with geometry-
frozen hyperparameters; fit the operator on `I₁ − heuristic(I₀, u)`.
**Watch out for** the circularity `ideas_log.md` §7 flags: never score a
heuristic-based model with a surrogate from the same heuristic family, and keep
the mass row out of it (P4 measured the heuristic's mass handling at ×11.6).
**Cost.** Half a day.

### B6 — Deliberate narrowness: one operator per operating envelope
The goal permits narrow domains, and §1.4 says the regime sets the ceiling. So
specify an envelope — fixed push length, contact score above a threshold, pile
depth class, distance from the wall — and fit inside it.
**Prediction.** Within a tight envelope on piled cubes, explained variance
exceeds 0.75, against 0.64 pooled. **Test:** stratify the existing n20 piled set
four ways and fit per stratum with B3's reduced representation so the splits
stay well-determined. Half a day, no new data. This also produces the thing MPC
actually needs: a stated envelope, plus a cheap runtime check for whether the
current state is inside it.

### B7 — Two-frame input
**Restated 2026-09-05.** This hypothesis was originally justified by sand's
five-sequential-pushes-per-episode structure, which made `(I_{k-1}, I_k, u)`
available at zero cost for 4/5 of every episode. That justification **survives
on cube data alone**: the piled cube collections also run 5 pushes per episode
(`pushes_per_episode: 5`, `docs/experiments/EXP-0006-spectrum-n20-n30.md`
provenance), so the same zero-cost previous-frame is available for n20/n30 with
no new data. What does **not** survive is the original cross-material
prediction below, which depended on contrasting sand against cubes — that
comparison is **WITHDRAWN** (`docs/rejected_mpm_sand.md`).

The mechanism argument itself ("a previous frame carries whether the material
was recently disturbed, hence loosely or densely packed") was a granular-medium
argument and has no established analogue for rigid cube stacks — a settled cube
configuration does not have a continuous "packing density" the way sand does.
So there is currently no positive case on record for this hypothesis, only the
original expectation that cubes "settle completely between pushes" and a
previous frame should therefore add little. **Prediction, cube-only:** near-zero
gain from `(I_{k-1}, I_k, u)` over `(I_k, u)` on piled cube data. A near-zero
result here is a clean negative result about hidden state and still worth
recording, but it is no longer contrasted against a positive sand case.
**Cost.** 2 hours.

### B8 — D4 symmetry augmentation as free data
The square tray has 8 geometric symmetries regardless of what is in it — a
sand-specific state-library module once exploited this to synthesise start
states, but it was removed with the sand path (`docs/rejected_mpm_sand.md`) and
has no cube-side equivalent on record. The tray symmetry itself is unaffected by
the material and remains free training augmentation for the operator: every
(state, action, next-state) triple has 7 exact partners.
**Prediction.** Largest effect where M/D is worst — the cube sets at res 64
(M/D ≈ 0.5) — potentially worth more than any model change there.
**Caveat.** The action must be transformed with the state, and reflections flip
the plate yaw sign. Given A1, this is exactly the kind of transformation that
needs a unit test before it is trusted.
**Cost.** 2–3 hours.

### B9 — Graded occupancy instead of binary-then-blur
Blur is a crude way to get a smooth field. A signed-distance or fractional-area
coverage rasterisation gets sub-pixel information *without* discarding
high-frequency content. **Prediction:** matches σ≈1 blur accuracy while
retaining the sharp field, which would make A2's dial a free choice rather than
a trade. **Cost:** 3 hours; `points_to_density` (renamed from `sand_to_density`
when the sand path was removed; it is material-agnostic and already used for
cube density views) is most of the machinery.

---

## 5. Hypotheses: what actually helps MPC

### C1 — The best blur for ranking is not the best blur for pixel accuracy
**Claim.** Error falls monotonically with σ (A2), but ranking must fall
eventually: at large σ every candidate action produces nearly the same predicted
image, so the differences MPC needs are erased. There should be an interior
optimum.
**Test.** Sweep σ and score with `control_utility_test.py`'s ranking metrics
(Spearman of predicted vs true `dV`, best-of-16 realised utility) rather than
rms. ~2 h, no new data. **This is the single most on-goal cheap experiment in
the document** — it answers "which model is good for MPC" with the repo's own
control metric.

### C2 — Predict `dV` directly instead of predicting an image
A greedy controller consumes one scalar per candidate. `deltav_predictability.py`
already shows `dV` on compact goals is 70–75% linearly reachable on scattered
data and 89–99% on piled — better-behaved than the image. A direct scalar
regressor sidesteps the warp, the crop, mass conservation and the blur choice
entirely.
**Cost against it:** goal-specific, and it cannot be rolled out multi-step.
`ideas_log.md` I1 has the fix — express `V` as a linear functional of the
descriptor basis, so the operator stays goal-agnostic and the cost is a
projection.
**Test.** Compare best-of-16 realised `dV` for (a) the pixel operator, (b) a
direct `dV` regressor, (c) I1's projected-cost descriptor model, on the same
candidate slates. ~1 day.

### C3 — Mean-delta may already be a competitive controller (the negative control that matters)
On piled cubes a zero-parameter constant displacement explains ~0.33 (n20 0.332,
n30 0.322, `docs/experiments/EXP-0002-regime-margin.md`) — re-sourced from an
earlier reading that also cited a sand row in the same range; the sand row is
withdrawn (`docs/rejected_mpm_sand.md`) but the cube numbers alone support the
same ~0.33 figure at unchanged strength. Ranking
needs only relative ordering, and a constant displacement in the *canonical*
frame is not constant in the world frame — it still differentiates actions by
where they point. Nobody has scored it as a controller.
**Prediction.** Mean-delta captures a substantial fraction of the operator's
control utility. If it captures most of it, the entire operator-fitting programme
is buying little at the level that matters, and effort should move to action
search and goal representation.
**Cost.** 2 h, reusing `control_utility_test.py`. Run this **before** B1–B9.

### C4 — Check the operator's spectrum before any multi-step rollout
Probe P17 found 10 of 55 eigenvalues of `I + A` outside the unit circle for the
descriptor operator — naive rollouts diverge. The pixel operator has never been
checked. **Test:** one eigendecomposition per fitted operator; report `max|λ|`
and the count outside the unit circle alongside every accuracy number. ~1 h, and
it should become a standing column in the model tables.

### C5 — Gate actions on predicted improvement versus residual noise
`ideas_log.md` MC5/OT5. With a per-regime residual variance (cheap, closed form)
an MPC can refuse actions whose predicted improvement does not clear ~1σ. Given
C3, the interesting measurement is what fraction of best-actions clear the bar
at all — probe P13, still unrun.

---

## 6. Needs new data — flagged, not run

- **D1. Same-state candidate slates.** Every ranking result in the repo draws
  candidates from *different* states, because the datasets hold one action per
  state (report §2.4 flags this). Ranking is the control-relevant metric, and it
  has never been measured cleanly. Collecting ~50 states × 16 actions is small
  and targeted, and it would make C1–C3 conclusive rather than suggestive.
- **D2. The A1 retrain.** Two runs of one config on `corl_limited`, ~1 h of GPU.
- **D3. Non-circular closed-loop rollouts.** `ideas_log.md` §7 calls this the
  one risk that outranks all ideas: the only closed-loop judge available is a
  surrogate built from the same heuristics some candidate models embed. ~50 real
  Genesis rollouts fixes it.
- **D4. Real data.** `Genesis/data/chickpeas*` (440 MB, two surfaces) has never
  been through this pipeline. Worth one pass once the frame convention is fixed —
  and worth checking whether the real-data path shares the bug.

---

## 7. Suggested order

1. **Fix the frame convention** (A1, A6) and add a test asserting that the
   occupancy channel, the plate channel and `particles_to_occupancy` agree.
   Check the MPC deployment path for the same mismatch. *Half a day, blocks
   everything.*
2. **Re-run the invalidated comparisons** on corrected grids: report §2, §2.7
   (A5), and §1.4's cube regime table here, under LORO instead of a single
   split. (The original item also named `sand_manipulation.md` §8's cube-vs-sand
   table; that comparison is withdrawn along with the sand path,
   `docs/rejected_mpm_sand.md`, and is no longer part of this item.) *One day.*
3. **C3, then C1** — establish what control utility the current models actually
   have over a zero-parameter baseline, and at what blur, before building more
   models. *Half a day.*
4. **B3** (PCA-space operator), which unlocks A4, B4 and B6 cheaply. *Half a day.*
5. **A2's bandwidth-collapse test** — the cheapest route to a single explanatory
   axis for the whole regime story. *Half a day.*
6. **B1, B5, B7, B8** in parallel; **B2** if a structurally different model is
   wanted. **D2** whenever GPU time is free.

---

## 8. Reproduction

Kept in `scripts/probes/`:
`ab_occ.py` (the transpose A/B), `regimes.py` (§1.4), `view_blur.py` (A2),
`channels.py` (§9 below) and `unet_action_ablation.py` (A1). All run on CPU in
minutes against data already on disk.

```bash
# the transpose, isolated
PYTHONPATH=. python scripts/probes/ab_occ.py --cube-size 0.007   # 107.1% -> 57.9%
# four cube regimes on one code path (originally five with a sand row,
# withdrawn — docs/rejected_mpm_sand.md)
PYTHONPATH=. python scripts/probes/regimes.py
# blur x view sweep (cube data; the sand dataset this originally pointed at
# was deleted with the sand path, docs/rejected_mpm_sand.md — see EXP-0003)
PYTHONPATH=. python scripts/probes/view_blur.py --glob 'Genesis/data/foresight/L040/**/*_data.pt'
# does the trained UNet use its action channel?
PYTHONPATH=. python scripts/probes/unet_action_ablation.py
```

## 9. One question answered in passing

The former `docs/sand_manipulation.md` §10 Q1 asked — "density or height as the
model input? Two channels is the obvious answer and is untested." That document
was deleted with the sand path (`docs/rejected_mpm_sand.md`).

**RESTATED 2026-09-05 to cube-only scope.** This table originally had three
columns — a mask target on cubes, plus mask *and* density targets on MPM sand —
and used the two sand columns to support a general claim about how every view
behaves. The sand columns are removed
(`docs/experiments/EXP-0005-input-channels.md`):

| input | mask target (cubes n20) |
|---|---|
| mean-delta (0 params) | 0.000 |
| mask only | **0.739** |
| height only | 0.631 |
| density only | 0.656 |
| mask + height | 0.733 |
| mask + density | 0.737 |
| mask + height + density | 0.731 |

Explained variance over mean-delta, res 32, σ=1, episode split.

On the surviving cube column, no multi-channel stack beats mask-only, and
height is the weakest single input (0.631 vs 0.656 for density and 0.739 for
mask) — that narrow finding survives unchanged.

**WITHDRAWN, not merely narrowed: "each view predicts itself best."** The
original generalisation required at least two different targets to say
anything about "each view" — only the sand columns ever supplied a second
target (a density target), and no cube run here used one. With the sand
columns gone, this record has only one target (mask) and cannot support a
claim about what happens under other targets. So: one channel, matched to
whatever the downstream cost consumes, is still the practical takeaway for
predicting a mask target — but "two channels is the obvious answer and is
wrong" is no longer established as a general property of the views. A
density-target cube run is what would restore it (also flagged in EXP-0005 as
the confound-breaking next step).
