# Residual parameterisation pilot (R1/R2) — COMPLETE

NOTE (post-run refactor, pure code move, no behaviour/retrain change): the
`Baselines/NFD/residual_nfd_lib.py` / `Baselines/NFD/residual_predictor.py`
paths cited below moved to `model/residual_nfd/lib.py` / `model/residual_nfd/
predictor.py` after these runs -- registered type names and every number
below are unchanged.

RUN-0015 (R1, unwarped residual) and RUN-0016 (R2, warped residual) both
completed their full 20-epoch training runs (launched in parallel with each
other and with RUN-0010, which was confirmed alive before/during/after
every step in this task and was never touched — GPU contention inflated
both runs' wall-clock, see each RUN.md). RUN-0017 scored all four arms
(RUN-0001, RUN-0002, R1, R2) through the same harness used for the existing
pilot record.

## Headline: four-arm `slateN` and `accuracy`, L20mm eval cell (RUN-0017)

`Baselines/common/eval_report.py`, 60 step-0 slates x 128 candidates,
3 goals x 3 value functions. **`slateN` first (the metric to trust),
`accuracy` beside it flagged suspect** — both frames as swept-region.

| arm | lyapunov | mass_in_region | signed_mass | accuracy |
|---|---|---|---|---|
| **random** (ranking floor) | -0.008 | -0.005 | -0.012 | n/a |
| persistence (accuracy's own denominator; DEGENERATE ranker, see `pilot_eval.md`) | +0.067 | -0.018 | +0.010 | 0.000 |
| RUN-0001 nfd_unwarped (direct, control) | 0.8810 | 0.8233 | 0.7646 | 0.4018 |
| RUN-0002 nfd_warped (direct) | 0.8342 | 0.8366 | 0.8042 | 0.3596 |
| **R1 nfd_residual_unwarped** (RUN-0015) | 0.8720 | 0.7969 | 0.7824 | **0.4091** |
| **R2 nfd_residual_warped** (RUN-0016) | 0.8608 | 0.8586 | 0.8066 | **0.4006** |

RUN-0001/RUN-0002 numbers reproduce `pilot_eval.md`'s published values
exactly (methodology rule #2, checked before interpreting anything further)
— this eval run is on the same footing as the existing record.

### Reading it

- **On `slateN`, all four arms are close** (within ~0.03-0.06 of each other
  per value function, same band the existing RUN-0001-vs-RUN-0002 pilot
  already showed) — no arm dominates all three value functions. R2 beats
  RUN-0002 on 2 of 3 value functions (mass_in_region 0.859 vs 0.837,
  signed_mass 0.807 vs 0.804) and is close on the third (lyapunov 0.861 vs
  0.834, actually slightly ahead); R1 is close to RUN-0001 throughout, split
  across value functions the same noisy way RUN-0001-vs-RUN-0002 already
  was. At 60 slates/single push length this is a shakeout, not a
  generalisation verdict — same caveat `pilot_eval.md` already states for
  RUN-0001/RUN-0002.
- **On `accuracy` (suspect metric, report beside slateN, never instead):
  R2 clearly beats RUN-0002 (0.4006 vs 0.3596, ~11% relative) and nearly
  closes the warp's entire accuracy gap to the unwarped arms** (RUN-0001
  0.4018, R1 0.4091) — RUN-0002 sat visibly below both unwarped arms, R2
  does not. R1 also modestly beats RUN-0001 (0.4091 vs 0.4018).
- **This accuracy gain is the interesting part, and it does NOT come from
  where the design predicted.** Section 1 below shows the residual
  formulation's theoretical accuracy CEILING (0.6852) is barely above the
  direct formulation's (0.6640) — a ~3-5% relative gain, nowhere near
  enough on its own to explain R2 closing an 11%-relative gap to the
  unwarped arms while starting from a ceiling only 3% higher than
  RUN-0002's. **The likely explanation (not yet directly tested) is an
  optimisation/inductive-bias effect of the explicit tanh-residual
  parameterisation itself** — an easier objective for the network to fit,
  independent of the resampling-ceiling argument the design was originally
  justified by. This is a hypothesis, not a confirmed mechanism; isolating
  it would need e.g. comparing gradient norms or convergence speed between
  the direct and residual arms, not done here.
- Neither R1 nor R2 saturates its own accuracy ceiling (R1's world-frame
  arm has no ceiling to speak of beyond the metric's own scale; R2 at
  0.4006 is well under its own 0.6852 residual ceiling, same qualitative
  picture `pilot_eval.md` already reported for RUN-0002 under the direct
  ceiling — dynamics-prediction error, not the ceiling, remains the binding
  constraint for R2 too).

## Design recap (see the configs' own header comments for the full rationale)

Loss stays world-frame MSE against the absolute `occ1` target
(`eulerian_combined`, mse=1.0) — unchanged from RUN-0001/RUN-0002. The
network's `UNetModels_modular.UNet` has its own `residual` flag set `false`
(asserted, not silently coerced) so its architectural occ0-into-logit skip
does not double-count with an EXPLICIT residual this wrapper applies: raw
conv output → `tanh` (bounded, signed [-1,1] — a sigmoid head cannot express
a negative delta) → `occ_pred = clamp(occ0 + delta, 0, 1)` → converted back
to a logit via the same safe-inverse-sigmoid trick `WarpedNFDWrapper`
already uses, so the unmodified loss can score it. R1 is unwarped
(reuses `nfd-genesis-3ch` dataset, unchanged); R2 predicts the residual in
the canonical push frame, unwarps the RESIDUAL FIELD (not a reconstructed
occupancy), and adds it to the SAME pristine world-frame `occ0` — no
validity-mask blend (reuses `nfd-genesis-3ch-warped` dataset and
`predictor.py::build_canonical_stack`, both unchanged).

Code: `Baselines/NFD/residual_nfd_lib.py` (training-time registrations),
`Baselines/NFD/residual_predictor.py` (eval-time predictors, registered in
`Baselines/common/eval_report.py`'s `MODELS` as
`nfd_residual_unwarped_L20mm_pilot`/`nfd_residual_warped_L20mm_pilot`).
Configs: `Baselines/NFD/configs/nfd_train_residual_{unwarped,warped}_L20mm_pilot.yaml`.
Runs: `../runs/RUN-0015-residual-unwarped-L20mm/`, `../runs/RUN-0016-residual-warped-L20mm/`.

## Design recap (see the configs' own header comments for the full rationale)

Loss stays world-frame MSE against the absolute `occ1` target
(`eulerian_combined`, mse=1.0) — unchanged from RUN-0001/RUN-0002. The
network's `UNetModels_modular.UNet` has its own `residual` flag set `false`
(asserted, not silently coerced) so its architectural occ0-into-logit skip
does not double-count with an EXPLICIT residual this wrapper applies: raw
conv output → `tanh` (bounded, signed [-1,1] — a sigmoid head cannot express
a negative delta) → `occ_pred = clamp(occ0 + delta, 0, 1)` → converted back
to a logit via the same safe-inverse-sigmoid trick `WarpedNFDWrapper`
already uses, so the unmodified loss can score it. R1 is unwarped
(reuses `nfd-genesis-3ch` dataset, unchanged); R2 predicts the residual in
the canonical push frame, unwarps the RESIDUAL FIELD (not a reconstructed
occupancy), and adds it to the SAME pristine world-frame `occ0` — no
validity-mask blend (reuses `nfd-genesis-3ch-warped` dataset and
`predictor.py::build_canonical_stack`, both unchanged).

## 1. The residual-formulation warp ceiling — THE CLEAN RESULT, needs no trained model

Per the task brief: round-trip the TRUE residual (`occ1-occ0`) through
warp→identity→unwarp (no blend) and add it to pristine `occ0`, exactly
mirroring `results/warp_accuracy_ceiling.md`'s direct-formulation ceiling
(round-trip the true `occ1` through warp→identity→unwarp+blend). Code:
`../code/residual_warp_ceiling.py`.

| `canon_res` | DIRECT ceiling (existing, `warp_accuracy_ceiling.md`) | RESIDUAL ceiling (this pass) |
|---|---|---|
| 32 | 0.4534 | 0.4936 |
| **64** (matches R1/R2's `canon_res`) | **0.6640** | **0.6852** |
| 90 | 0.7490 | 0.7661 |
| 128 | 0.8183 | 0.8303 |
| 181 | 0.8681 | 0.8774 |

**The residual formulation's ceiling is barely above the direct formulation's
at every `canon_res` (+0.021 to +0.032, ~3-5% relative) — NOT the
substantial rise the mechanism predicts.** If the "preserve the unchanged
98% bit-exact" argument were the dominant effect, the residual ceiling
should have been much closer to 1.0 at `canon_res=64`, since the swept
region is a small fraction of the image and the rest should round-trip
losslessly. It does not.

### Why — verified, not argued

Checked directly (`push_frame_validity_mask` at the same `canon_res`,
compared pixel-by-pixel against the round-tripped residual's deviation from
zero):

| region | fraction of pixels | max \|occ_pred − occ0\| |
|---|---|---|
| mask exactly 0 (fully outside both warps' support) | 21.5% | **0.0 — bit-exact, as claimed** |
| mask strictly in (0, 0.5) (partial boundary coverage) | 1.3% | **0.389 — NOT bit-exact** |

`grid_sample`'s bilinear weights are non-negative, so a validity mask value
of exactly 0 (a zero row-sum of weights) forces every individual weight to
be exactly 0 — the "delta=0 there is guaranteed" argument in
`residual_nfd_lib.py`'s docstring holds, but **only for that ~21.5%, not
for the full ~23% (`1 - 0.664`-scale) region `warp_accuracy_ceiling.md`
calls "outside the validity mask" at the conventional 0.5 threshold.** The
remaining ~1.3% straddles a partial-coverage boundary band where the
round-tripped residual is a real but incomplete (non-renormalized) weighted
sum of the true residual — and since the true residual can be near ±1 right
at the swept region's edge, this thin band can carry large error. That
band is thin (1.3% of pixels) but sits exactly where the pusher acts, i.e.
exactly where the swept-region `accuracy` metric integrates, so its
contribution to the ceiling is disproportionate to its pixel count.

**Consequence for the design claim in `residual_nfd_lib.py`'s docstring**:
the "no blend needed, zero-padding already means no change" argument is
TRUE but narrower in scope than stated there — it holds for the fully
excluded ~21.5%, not for the full conventional "outside validity mask"
region. This is a real, verified partial refutement of the strongest
argument for R2's design, discovered by this ceiling check, and is worth
flagging to whoever reads the residual_nfd_lib.py docstring next. (The doc
comment there does already gate the "verified... in this experiment's
RUN.md" claim — that verification is this section.)

## 2. Dead-gradient check — training is not starved (2-epoch smoke checkpoints)

`clamp(occ0 + delta, 0, 1)` has zero local gradient wherever the pre-clamp
sum is outside (0, 1). Measured on a real held-out batch (64 rows) through
each arm's 2-epoch smoke checkpoint (`../code/residual_dead_gradient.py`):

| arm | dead-gradient fraction (all pixels) | dead-gradient fraction, of pixels needing change (\|Δ\|>0.05) |
|---|---|---|
| R1 unwarped | 96.81% | **12.92%** |
| R2 warped | 96.57% | **12.58%** |

The overwhelming majority of "dead" pixels are the true, static background
where `occ0` and `occ1` agree and clamping there is harmless (no
information is lost about pixels that don't need it). Only ~13% of the
pixels that actually need correction are gradient-starved by the clamp in
either arm — training is not stalled by this mechanism, and both arms'
losses decayed smoothly and monotonically over their 2-epoch smoke runs
(R1: trn 0.0076→0.0061, val 0.0061→0.0057; R2: trn 0.0069→0.0057, val
0.0056→0.0055) with healthy IoU (0.77-0.78) at epoch 2 already —
comparable to RUN-0001's own epoch-2 territory.

## 3. Full 20-epoch training — timing (both COMPLETE)

RUN-0015 and RUN-0016 were launched in parallel with each other and with
RUN-0010 (confirmed alive at every check made in this task, never touched)
to save wall-clock. Both completed cleanly, 20/20 epochs, no NaNs, no
divergence, no restarts. **Timing was inflated by up to 3-way GPU
contention and is not comparable to RUN-0001/RUN-0002's clean numbers**:

| run | clean baseline (idle GPU) | this run (contended) | slowdown |
|---|---|---|---|
| RUN-0015 (R1) | ~32.2 s/epoch (RUN-0001) | ~66.3 s/epoch | ~2.1x |
| RUN-0016 (R2) | ~37-38 s/epoch (RUN-0002) | ~74.7 s/epoch | ~2.0x |

Both checkpoints (`unet_best.pth`) are saved in
`Baselines/NFD/runs/nfd_residual_{unwarped,warped}_L20mm_pilot_2/` (the
trainer auto-suffixed `_2` because the earlier 2-epoch smoke test already
occupied the un-suffixed directory) and copied into their run's `artifacts/`
alongside `model_card.yaml`.

## Bottom line

- **The residual ceiling (no training needed) is essentially a NULL
  RESULT relative to its own prediction**: 0.6852 vs 0.6640, a ~3-5%
  relative gain, not the "substantial rise" the mechanism predicts — and
  the cause is understood precisely (a thin partial-coverage boundary band,
  not the bulk "outside validity mask" region, is where the bit-exact
  argument breaks down).
- **Yet the TRAINED R2 model clearly beats RUN-0002 on `accuracy`** (0.4006
  vs 0.3596) and matches or edges out RUN-0002 on 2 of 3 `slateN` value
  functions. **This is the contradiction worth flagging**: the theoretical
  argument this design was built on (bit-exact preservation via
  zero-padding) barely moves the ceiling, but the trained model improved by
  more than the ceiling moved. The gain is real (reproduced against a
  verified RUN-0001/RUN-0002 baseline) but its cause is NOT the mechanism
  named in the brief — most plausibly an optimisation/inductive-bias
  benefit of the explicit residual parameterisation itself, untested
  directly here.
- R1 (unwarped residual) also modestly beats RUN-0001 on `accuracy` (0.4091
  vs 0.4018), consistent with the brief's own prediction that R1 would gain
  "little" — RUN-0001 already has an occ0-into-logit skip, so R1's explicit
  residual adds less new capability than R2's does relative to RUN-0002.
- Both arms trained in a numerically healthy way throughout (smooth loss
  decay both in the 2-epoch smoke test and the full 20-epoch run;
  dead-gradient check found no starvation of pixels that matter).
- At pilot scale (60 slates, single push length, `pilot_eval.md`'s own
  "shakeout not verdict" caveat applies identically here) this is not yet a
  generalisation claim — the comparison that would decide that is the
  multi-length `overnight_randlen` run, per `PLAN.md`'s own standing
  caveat for the whole warped-NFD line of experiments.
