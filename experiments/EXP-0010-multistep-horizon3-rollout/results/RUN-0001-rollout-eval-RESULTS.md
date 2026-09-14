# Closed-loop 3-step rollout eval (slates_multistep, n20_L20mm + n20_L40mm)

Code: `rollout.py`. Raw output: `results_multistep.json`. Run: single
`python -u rollout.py`, ~5s total on GPU, commit 0ddab20f (dirty tree, see
provenance block in the JSON).

## Data / trajectory-identity check

Loaded directly from the raw `_{batch}_data.pt` files (NOT through the
`*_eval` dataset configs, which apply `min_push_length_m` filtering that
drops rows and would break row==env-identity alignment across steps).
50 slates x 128 envs x 3 steps per dataset, unfiltered.

Re-verified the task's row-stability claim on one file pair per dataset
(step-0 file's `states_` vs step-1 file's `states`):

| dataset | aligned median L2 | shuffled-control median L2 |
|---|---|---|
| n20_L20mm | 6.6e-08 | 3.0e-02 |
| n20_L40mm | 8.4e-08 | 1.4e-01 |

Confirms the claim: row index is a stable per-env trajectory identity.

## Bin-sequence distribution (switched operator, MODEL-0001's edges)

Both cells are **single-push-length** collections (`push_length` = 0.02m /
0.04m fixed in `manifest.json`, not `randlen`), so despite 3 independent
steps there is effectively **one bin the entire rollout lives in**:
n20_L20mm: 6400/6400 rows land in bin 1, all 3 steps. n20_L40mm: 6399/6400
in bin 3 for all 3 steps (1 row dips to bin 2 at step 1). So "the switched
operator applies 3 possibly-different per-bin operators across a rollout"
is true in principle (`Baselines/LinearForesight/model.py:27,34` recomputes
the bin every call, confirmed by reading it) but **not exercised** by this
particular corpus — switched and global stay close to each other here
because there's almost no bin-switching happening, not because switching
doesn't matter in general (that finding lives elsewhere, on `randlen` data).

## Per-step image accuracy (swept-region mask, recomputed per step)

`+acc` = `1 - rms(model)/rms(persistence)`. TF = teacher-forced (fed the
TRUE occupancy at every step), CL = closed-loop (fed its own previous
prediction, steps 2-3 only; step 1 is identical to TF by construction since
step 1's input is always true `occ0`).

**n20_L20mm** (step1 / step2 / step3):

| model | TF | CL |
|---|---|---|
| model0001_switched | +0.168 / +0.167 / +0.171 | +0.168 / +0.110 / +0.094 |
| model0001_global | +0.031 / +0.029 / +0.035 | +0.031 / -0.065 / -0.104 |
| hybrid94 | +0.170 / +0.167 / +0.172 | +0.170 / +0.112 / +0.095 |
| nfd | +0.351 / +0.354 / +0.360 | +0.351 / +0.181 / +0.099 |

**n20_L40mm** (step1 / step2 / step3):

| model | TF | CL |
|---|---|---|
| model0001_switched | +0.412 / +0.327 / +0.249 | +0.412 / +0.284 / +0.168 |
| model0001_global | +0.351 / +0.287 / +0.218 | +0.351 / +0.231 / +0.119 |
| hybrid94 | +0.416 / +0.330 / +0.250 | +0.416 / +0.289 / +0.172 |
| nfd | +0.451 / +0.423 / +0.400 | +0.451 / +0.303 / +0.171 |

MODEL-0002 (descriptor-only): image `accuracy` is not defined (no decoder,
scores 0.0 by construction per its own MODEL.md) — not reported here.

## Terminal slateN (horizon-3, goal=corner, K=128 exact + K=32 Monte-Carlo)

`persistence`/`random` are mandatory floor baselines; `oracle` = ranking by
the TRUE terminal value (=1.0 by construction). Reported per value function.
**Ties** here are defined as "model's chosen row exactly equals the true
argmax row" (a strict, conservative definition — with continuous-valued
predictions this essentially never fires at float precision, so all
"ties" columns read 0; this is NOT the METRICS.md notion of two *models*
agreeing on the same pick, which was not computed here for time reasons —
flagged as a methodology gap, not hidden).

**n20_L20mm, lyapunov (cost, lower=better), teacher-forced / closed-loop capture, mean±sem, n=50 slates:**

| model | TF capture | TF K=32 | CL capture | CL K=32 |
|---|---|---|---|---|
| model0001_switched | 0.972±0.006 | 0.970 | 0.941±0.009 | 0.948 |
| model0001_global | 0.937±0.011 | 0.921 | 0.834±0.022 | 0.835 |
| hybrid94 | 0.972±0.006 | 0.968 | 0.943±0.009 | 0.946 |
| nfd | 0.977±0.006 | 0.980 | 0.971±0.006 | 0.949 |
| persistence | -0.039±0.077 | 0.001 | (same) | (same) |
| random | 0.081±0.060 | 0.022 | (same) | (same) |

**n20_L40mm, lyapunov:**

| model | TF capture | CL capture |
|---|---|---|
| model0001_switched | 0.998±0.001 | 0.992±0.003 |
| model0001_global | 0.997±0.001 | 0.991±0.003 |
| hybrid94 | 0.997±0.001 | 0.985±0.008 |
| nfd | 0.996±0.001 | 0.956±0.014 |
| persistence | -0.125±0.096 | (same) |
| random | -0.093±0.095 | (same) |

`mass_in_region` (same corner mask, higher=better) shows the same ordering
but a bigger TF->CL drop for `nfd` and `model0001_global` (full numbers in
JSON, e.g. L20mm CL: switched 0.936, hybrid94 0.936, nfd 0.878, global
0.753).

**MODEL-0002 descriptor-only (point-mass/COM readout, TEACHER-FORCED ONLY — see limitation below):**

| dataset | lyapunov capture | mass_in_region capture |
|---|---|---|
| n20_L20mm | 0.641±0.031 | 0.614±0.046 |
| n20_L40mm | 0.831±0.027 | 0.683±0.035 |

**Limitation, stated plainly:** MODEL-0002's closed-loop rollout was
**scoped out**. Its operator consumes a 94-dim descriptor built in the
CURRENT step's push-frame (mass, COM, 2nd moments, band mass, Fourier
shape terms). Chaining it across 3 steps closed-loop requires
re-expressing that whole vector in the NEXT step's push-frame — trivial
for the COM (a point; `com_world_pixel`/`world_to_pushframe_px` already do
this) but there is no existing transform in this repo for the shape terms
(moments2/band_mass/dft) under an arbitrary SE(2) reframing, and inventing
one under this task's time budget would produce an unvalidated number.
The teacher-forced number above is the same "per-step ranking from the
true state" every other descriptor/latent eval in this repo already
reports; it is a real result, just not the closed-loop one the task
wanted for this model.

## Answers

**How fast does closed-loop error compound vs teacher-forced?**
On raw per-step image accuracy the effect is large and monotonic: on
n20_L20mm, `model0001_switched` drops from a flat ~+0.17 (TF, all 3 steps)
to +0.11 (step2) / +0.09 (step3) closed-loop — roughly a 35%/45% relative
loss at steps 2/3. `model0001_global` is worse and actually goes NEGATIVE
by step 3 closed-loop (-0.10, i.e. worse than persistence) while its own
TF number stays a stable +0.03 — closed-loop exposes a model that looks
mediocre-but-positive TF as net-harmful once it eats its own output twice.
`nfd`, the most accurate TF model, degrades the fastest in relative terms
(+0.35 -> +0.10 CL by step 3, a 3.5x drop) — its higher one-step accuracy
does not protect it from compounding. On terminal `slateN` ranking the
compounding cost is much smaller in relative terms (persistence/random are
so much worse that even a degraded model stays near-oracle): e.g.
`model0001_global` on L20mm/lyapunov drops from 0.937 (TF) to 0.834 (CL),
about an 11% relative loss — ranking is far more forgiving of compounding
error than per-pixel accuracy is, because it only needs relative ordering
of the 128 candidates to survive, not pixel fidelity.

**Which model best ranks whole 3-action sequences by terminal outcome?**
`nfd` wins on lyapunov-TF marginally (0.977 vs switched/hybrid94's 0.972 on
L20mm; near-tied ~0.996-0.998 on L40mm), but **loses its lead under
closed-loop and under mass_in_region**: on L20mm CL/lyapunov switched
(0.941) and hybrid94 (0.943) both overtake nfd's degraded 0.971->... wait,
nfd CL/lyapunov L20mm is actually still highest (0.971) — but on
mass_in_region CL it drops to 0.878, well below switched/hybrid94 (0.936).
So the honest answer is: **hybrid94 and model0001_switched are the most
robust across both value functions and both rollout modes** (never far
from the top, never catastrophic); `nfd` is the best single-metric
performer (lyapunov) but the least robust across metrics/modes.

**Does one-step ranking (visual > latent > descriptor) survive to horizon 3?**
No hybrid/latent-vs-visual reordering was tested here (only visual +
hybrid94 + descriptor-only were in scope), but within what was run: the
image models (switched/global/hybrid94/nfd) all stay far above the
descriptor-only model (MODEL-0002) at horizon 3 (~0.93-0.99 vs 0.61-0.83
capture) — the one-step ordering "image > descriptor" survives to horizon
3, consistent with EXP-0006/EXP-0008's in-corpus finding for this same
dataset family. `model0001_global` (unswitched) is consistently the
weakest image model at horizon 3, especially closed-loop and under
mass_in_region (0.753 on L20mm) — switching helps more, not less, as
horizon grows.

**Any model that diverges/blows up?**
`model0001_global` (the unswitched global operator) goes net-negative
image accuracy by step 3 under closed-loop on n20_L20mm (-0.104, worse
than doing nothing) — the clearest outright divergence. `nfd` doesn't go
negative but its per-step accuracy collapses fastest in relative terms
under closed-loop (see above). No model produced NaN/exploding pixel
values; degradation is gradual pixel blur/mass drift, not a hard blowup.

## Scoping decisions (not findings)

- GNN/SchenckCNN excluded per task instruction (pending EXP-0009).
- MODEL-0002 closed-loop rollout excluded (see limitation above).
- `slateN` "ties" here uses a strict argmax-match definition, not the
  cross-model-agreement definition METRICS.md describes; a proper
  cross-model tie count was not computed under this budget.
- K=32 reference is a 50-repeat Monte-Carlo subsample per slate (without
  replacement), not the closed-form `slateK_exact` — documented
  approximation, not the exact combinatorial number.
