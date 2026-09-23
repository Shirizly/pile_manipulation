# Does flip-only augmentation flatten the warped arm's push-length deficit?

**No. It is REFUTED, and the profile is if anything slightly steeper.**

## The hypothesis

`PLAN.md` recorded a specific prediction. The warped arm's swept-region
`accuracy` deficit versus the world-frame baseline grows sharply at short push
lengths, and after metric artifact (12-30%), canonical action-channel overlap
(refuted) and input resampling (partial, uncalibrated) each failed to account
for it, the standing candidate was the **x8-augmentation handicap itself**:
sub-20mm pushes are a rare tail of `overnight_randlen` (early-contact cutoffs
against a commanded 20-70mm range), and 4x less data diversity per gradient
step should hurt rare regions of the input distribution most.

**That predicts the length profile should FLATTEN under flip-only
augmentation.** RUN-0010 is that arm; this is the test.

## Result

Warped-minus-baseline `accuracy` delta on `overnight_randlen_test`, world
frame, same 5 quantile bins as RUN-0012/RUN-0013.

| push length (mm, bin mean) | RUN-0005 (x8 aug) | RUN-0010 (flip aug) |
|---|---|---|
| 14.8 | −0.1541 | −0.1453 |
| 28.3 | −0.0703 | −0.0587 |
| 38.8 | −0.0477 | −0.0387 |
| 50.4 | −0.0343 | −0.0266 |
| 65.0 | −0.0191 | −0.0127 |
| **range (shortest/longest)** | **8.1x** | **11.4x** |

Bootstrap SEMs are 0.0011-0.0061, so every bin's improvement is real but small.

**Every bin improved slightly — and the ratio between the ends got LARGER, not
smaller.** The fair arm is uniformly a little better across the whole length
range, which is consistent with it simply being a better-trained model, and
carries no sign of the specific short-push rescue the hypothesis predicted.

## What this leaves

The length dependence is now **unexplained after four candidate causes**:

| candidate | status |
|---|---|
| metric artifact (small persistence denominator at short pushes) | real, but only 12-30% (RUN-0013) |
| canonical action-channel overlap at short push lengths | **refuted** — channels fully separated above ~15mm, yet the deficit still shrinks 3.3x across the 28-65mm bins (RUN-0013) |
| input-side resampling | real partial contributor, magnitude wrong by 1.6-5.7x and confounded by the control's own design (RUN-0014) |
| the x8-augmentation handicap | **refuted** — this record |

The wall-distance stratification in the same run reproduces RUN-0012's shape on
this arm too: non-monotonic, and roughly flat once push length is held fixed
(within band 33-44mm: −0.043, −0.030, −0.035, −0.040, −0.046). So the wall
hypothesis does not revive on the fairly-trained arm either, and the gate
against training a wall channel stands.

**A short push is simply harder for a push-frame model than for a world-frame
one, and none of the four mechanisms tested explains why.** That is the open
question this line of work ends on.
