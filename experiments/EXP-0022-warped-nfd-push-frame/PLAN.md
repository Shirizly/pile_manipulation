# EXP-0022 — revised plan after the first warped arm

Written 2026-09-23, after RUN-0005 (warped NFD, `canon_res=64`, no wall
channel) was trained on `overnight_randlen_train` and scored against the
world-frame NFD baseline. This file supersedes the implicit four-arm plan in
`LOG.md`; it does not delete it, because RUN-0005 and its records stand.

## Where the first arm left us

RUN-0005 scored slightly BELOW the world-frame baseline: `slateN` lower by
0.01-0.03 on a 0.75-0.94 scale (consistently under `lyapunov`, roughly tied
under the mass-based value functions), `accuracy` lower in all three corpora.
See `results/randlen_eval.md`.

**That verdict does not stand, for a reason found afterwards.** The trainer's
x8 rotation/flip augmentation divides the loader batch by 8 and rebuilds a
batch from 8 views. For a warped model those 8 views collapse to **2 distinct
canonical inputs** — `push_frame_transform` is a pure rotation, so the 4
rotation views are exactly identical in the canonical frame, and the 4
flipped views are exactly identical to each other. RUN-0005 therefore drew
each gradient step from **4x fewer distinct transitions** than the baseline
did, at identical compute. It was the baseline's recipe applied to a model
that does not need it.

RUN-0005 is consequently a **lower bound on warped performance, not a
measurement of it**, and its numbers should not be cited as the answer to
this experiment's question.

## Why the warp plausibly lost anyway — four costs, one partly-redundant benefit

1. **Degenerate augmentation** (above). Rotation invariance is built into the
   warped model by construction, so only the flip carries information. This
   is a property to exploit, not a bug: at matched wall-clock, flip-only
   augmentation buys ~4x more distinct transitions per gradient step.
2. **The action encoding wastes the input.** In the canonical frame both
   plate channels are determined by push length alone — two 64x64 channels,
   8192 numbers, carrying one scalar.
3. **Corner loss and the blend.** Rotating a square inside a same-sized
   square loses the corners; outside the validity mask the model is forced to
   emit persistence. Any part of the SCORED swept region that falls outside
   that mask penalises the model for pixels it was never allowed to predict.
4. **The walls move.** Static in the world frame, so the world-frame model
   learns their constant effect for free; they land somewhere different in
   every canonical input. Still untested.

Against those, the intended benefit — sparing the network from learning
pose-equivariance — is partly redundant: a UNet is already
translation-equivariant by construction, and the x8 augmentation was already
teaching the baseline rotation.

## The reframe

The warp's distinctive payoff is not a tidier image. It is that **the action
collapses to a scalar**, which is the one thing the world frame cannot do,
since there the plate channels must carry pose. RUN-0005 kept the
world-frame action encoding, so it paid all of the warp's costs and collected
none of its distinctive benefit. That is the weakest version of the idea, and
it is the version that has been tested so far.

## Phase A — remeasure. Gates everything else; almost all of it is re-scoring.

- **A1. Warped-frame scoring.** Add prediction/target scoring in the
  canonical frame as an ADDITIONAL reporting frame, and re-score RUN-0005,
  the NFD baseline, the fitted linear operators and EXP-0001's models in
  BOTH frames. Scoring `accuracy` against the warped true next state removes
  one of the two resamplings and the corner/blend problem from both the
  objective and the metric. **If the framing reorders models that world-frame
  scoring ranked differently, that is a methodological finding about the
  whole register, not just about this experiment.** No training.
- **A2. Warped-goal control scoring, as a HYPOTHESIS to test, not an
  assumption.** `slateN` ranks candidates FROM THE SAME STATE, and every
  candidate has a DIFFERENT push frame. Warping the goal per candidate means
  each candidate's value is computed in its own rotated, resampled frame, so
  the resampling bias no longer cancels ACROSS the comparison — it varies
  across it. That is the same mechanism behind the measured 4.8% top-1 flip
  rate, and warping the goal could amplify it rather than remove it. So:
  implement it, rerun the ranking-robustness check under it, and let the
  measurement decide. No training.
- **A3. Wall-proximity stratification.** Group test actions by a
  wall-involvement statistic and compare the warped-vs-baseline deficit per
  stratum. Deficit concentrated in wall-proximate pushes -> the wall channel
  is motivated. Deficit flat -> it will not rescue anything. This is the
  cheap gate on a 3h training run. No training.
- **A4. DONE — and it is the experiment's answer.** RUN-0010 (flip-only
  augmentation, 120 epochs, 668,040 steps) reaches `slateN`/lyapunov parity
  or slightly above the baseline on 2 of 3 corpora, but does NOT recover
  swept-region accuracy. **The warp does not help: neutral on control,
  negative on image prediction.** See `EXPERIMENT.md` and C-027. Its
  epoch-30 checkpoint matches or beats its own epoch-108 on 7 of 9 `slateN`
  cells, so flip-only augmentation buys a cheaper model rather than a better
  one. The length-profile test of the standing hypothesis below was NOT run.
  Original plan text follows.

- **A4 (as planned). Fair re-run of the warped arm.** Flip-only augmentation, epoch/step
  accounting settled first, matched to the baseline on an axis chosen
  explicitly. **This is what replaces RUN-0005's verdict.** ~3h.

  Open item, to settle before sizing: `nfd_train_3ch_randlen.yaml`'s header
  claims ~2784 steps/epoch, which is `N/32` — but with `augment: true` the
  loader batch is `32//8 = 4`, implying `N/4`. One of those is wrong, and it
  decides what "matched gradient steps" means.

## Phase B — action representation, as a 2x2

`{world frame, push frame} x {2 plate channels, 1 swept-rectangle channel}`.
Pilot on L20mm+L40mm, decide on `overnight_randlen`.

The **swept-rectangle channel** is the primary candidate: one channel marking
the region the tool sweeps. It keeps the input format (so it is directly
comparable to the existing models), reduces representation complexity, works
UNWARPED as well as warped — which is what makes the 2x2 possible and
separates "does the frame help" from "does the action encoding help", two
questions entangled in everything run so far — and leaves the door open to
varying tool size, since the rectangle carries plate width as its own width.
`fit_linear_foresight.py::swept_region_mask` already exists, and is already
the mask `accuracy` is computed over.

**FiLM conditioning on push length is a secondary arm**, worth running but
with a principled reason to expect less: FiLM modulates channels uniformly
across space, whereas push length has an inherently SPATIAL effect — it sets
where the material ends up, not a global gain. `model/NFDUNetFilm.py` already
exists. The two are composable (rectangle channel plus FiLM on length) if the
encoding turns out to matter. FiLM also cannot express tool geometry, so it
does not serve the varying-tool-size goal the rectangle does.

Caveat on the pilot corpus: L20mm+L40mm is essentially two push lengths plus
early-contact spread, so it is a weak testbed for anything that has to
generalise across a length continuum. Fine as a cheap pilot at ~30 min/run
against ~3h on randlen; the result that counts should come from randlen.

## Phase C — the wall channel

Only if A3 justifies it. The design is unchanged: a 4th channel,
`to_push_frame(ones)`, marking the workspace extent in the transformed frame.

## Open question for Phase B

For the encoding comparison, is the reference arm the existing world-frame
2-channel model, or a push-frame model that keeps two plate channels? The
latter isolates the encoding with the frame held fixed; the former also
folds in the frame. Not yet decided.

---

## Phase A interim state (2026-09-23), and why diagnostics are PAUSED here

Four no-training diagnostics ran (RUN-0011..0014). Summary of what each
settled, and one decision that follows:

- **A1, canonical-frame `accuracy` (RUN-0011): no reordering.** Every model
  gains ~0.04 scored in the canonical frame, and the ranking is identical in
  both frames across all three corpora. The framing debate does not change
  any conclusion already drawn from world-frame accuracy anywhere in the
  register. Canonical scoring can be adopted for fairness without
  invalidating prior claims.
- **A2, warped-goal `slateN` (RUN-0011): REFUTED.** Top-1 flip rate 20/42
  (47.6%) against 2/42 (4.76%) for shared-frame scoring, worst on
  `randlen_test` (13/14) where candidates within a pool differ most in push
  direction. Evaluating each candidate's value in its own push frame
  destroys the cross-candidate comparison, as predicted. Measured at
  `canon_res=64`, where each candidate's frame loses different corners of
  the goal; a validity-masked or supersampled variant is untested, but at a
  10x gap it is not worth pursuing.
- **A3, wall proximity (RUN-0012): the wall hypothesis does not survive on
  the corpus that matters.** On `randlen_test` the deficit is U-shaped in
  wall distance and goes FLAT once push length is held fixed. It does
  replicate monotonically on L20mm/L40mm and survives a length control
  there, but those are single-push-length corpora. **The wall channel is not
  worth training yet.**
- **The real driver is PUSH LENGTH**: warped-minus-baseline accuracy runs
  -0.154 at the shortest pushes to -0.019 at the longest, an 8x range that
  dwarfs every other stratification tried.
- **RUN-0013 refuted the obvious explanation for that.** Action-channel
  overlap is not the driver: the two canonical plate channels are fully and
  identically separated from ~15mm upward, yet the deficit still shrinks
  3.3x across the 28-65mm bins where the geometry does not change at all.
  Metric artifact explains only 12-30%.
- **RUN-0014's input-resampling control is direction-right but
  magnitude-wrong**, and is confounded by its own design: it degrades the
  baseline with a warp-AND-unwarp (two resamplings) where the warped model
  pays one, and the baseline was never trained on resampled input where the
  warped model was. So it over-degrades, and cannot be made clean without
  training. Input resampling is a real partial contributor, not a calibrated
  explanation. `canon_res=96` removes only ~28% of the round-trip
  degradation, so it stays deprioritised.

### The decision

**Stop diagnosing and wait for RUN-0010.** Every one of RUN-0012/0013/0014
was measured on RUN-0005's checkpoint, which carries the known x8-augmentation
handicap (4x fewer distinct transitions per gradient step). There is a
specific reason to think that handicap is **not uniform across push length**,
which would make it a candidate for the unidentified third cause: sub-20mm
pushes are a rare tail of `overnight_randlen` (early-contact cutoffs against
a commanded 20-70mm range), and reduced data diversity per step hurts rare
regions of the input distribution disproportionately. That predicts the
deficit's length profile should FLATTEN under RUN-0010's flip-only
augmentation.

Continuing to characterise a handicapped checkpoint spends budget on a model
we have already decided to replace. The next action is to re-run the
length-stratified comparison on RUN-0010 and see whether the profile
flattens — which tests the hypothesis above for free.
