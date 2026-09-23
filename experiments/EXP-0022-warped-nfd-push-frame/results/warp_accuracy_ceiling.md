# The push-frame warp's own accuracy ceiling

**What this is.** A warped model predicts in the canonical push frame and is
unwarped back to the world frame, so its output passes through TWO
`grid_sample` resamplings that the world-frame NFD never pays. That
resampling destroys information on its own, before any question of dynamics.
This measures how much.

**How.** Feed the TRUE next occupancy `occ1` through the same
`push_frame_roundtrip` (warp -> identity -> unwarp -> blend, `scale=1.0`)
and score it with the ordinary swept-region `accuracy`, exactly as if it
were a prediction. **A perfect warped predictor scores this and no higher.**

Corpus: `configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml`,
n=7680, 64x64 grid. Code: `../code/warp_accuracy_ceiling.py`.

| `canon_res` | accuracy ceiling | persistence through the round trip | mean round-trip \|err\| |
|---|---|---|---|
| 32 | 0.4534 | 0.0020 | 0.01171 |
| **64** (= grid res, LinearForesight's own setting) | **0.6640** | 0.0254 | 0.00657 |
| 90 (~= 64*sqrt(2)) | 0.7490 | 0.0258 | 0.00482 |
| 128 | 0.8183 | 0.0234 | 0.00346 |
| 181 | 0.8681 | 0.0186 | 0.00248 |

Unwarped persistence scores 0.0000 by construction (it is the metric's own
denominator).

## Why it matters

At `canon_res=64` the warped arms cannot exceed **0.664** on this corpus no
matter how good their dynamics are. The world-frame NFD baseline scores
0.407-0.509 on comparable cells (EXP-0001), so the warped arms are boxed
into a narrow band immediately above the baseline. **Any accuracy comparison
between a warped and an unwarped arm must report this ceiling beside it**,
or it will read as a dynamics result when part of it is a resampling
artifact.

The ceiling rises steeply with `canon_res` because the loss is
undersampling: a rotated 64x64 square does not fit in a 64x64 grid without
it. `canon_res ~ 64*sqrt(2)` is where that stops.

## What it does NOT constrain

`slateN` is a RANKING metric over candidate actions from the same state. A
resampling loss that is close to uniform across candidates shifts every
candidate's score together and largely cancels in the ranking. So this
ceiling is a strong caveat on `accuracy` and a weak one on `slateN` — which
is a further reason to lead with `slateN`, as `experiments/METRICS.md`
already requires. This is an argument, not a measurement; it has not been
verified that the resampling loss really is near-uniform across a candidate
pool, and that check is cheap and worth doing before leaning on it.

## Consequence for the design

The main run adds a **`canon_res=96`** arm (~`64*sqrt(2)` rounded to a
multiple of 8) alongside the `canon_res=64` arm, so that "the warp hurts"
can be separated from "the resampling hurts". `canon_res=64` remains the arm
that matches the LinearForesight warp exactly.
