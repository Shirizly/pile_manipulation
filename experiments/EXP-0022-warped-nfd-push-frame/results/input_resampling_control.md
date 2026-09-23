# RUN-0014 -- input-resampling control: does degrading the baseline's occ0
reproduce the warped model's length profile?

Follows RUN-0013 (ruled out the literal channel-overlap mechanism as the
primary cause of the length-dependent deficit; a metric artifact explains
only 12-30%). Hypothesis under test: the warped model's `occ0` passes through
one `grid_sample` before the network ever sees it (to enter the canonical
frame); that is a roughly CONSTANT absolute blur, so it costs relatively more
when the true state change (and hence `err_pers`, the denominator of
`accuracy`) is small -- i.e. at short pushes.

## The control

Feed the world-frame baseline (`Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth`,
unmodified) an `occ0` that has been round-tripped through
`transforms.functional.push_frame_roundtrip` with an **identity** `fn`,
`canon_res=64`, `scale=1.0`, using each transition's own push endpoints
(the SAME native pixel derivation `WarpedNFDPredictor._build_fn` uses for its
own canonical warp -- not the world `actions_to_pixels` derivation, so the
degradation applied is bit-for-bit what a single-warp entry into the
canonical frame would look like, glued back into the world frame by the
round trip's own blend). Only `occ0` is degraded; the two plate/action
channels the baseline builds from `p_start`/`p_stop` are untouched. Scored
world-frame, same swept-region `accuracy`, same aggregation-of-means +
bootstrap-CI methodology as RUN-0012/0013 (`per_transition_errors`,
`accuracy_from_errors`, `bootstrap_delta3` -- new, a 3-arm version of
`bootstrap_delta`).

Implementation: extended
`experiments/EXP-0022-warped-nfd-push-frame/code/wall_proximity_stratified_eval.py`
(`run_predictor_degraded_input`, `bootstrap_delta3`, `strat_table3`,
`--degraded-baseline` flag) rather than forking a new script. CPU only
(asserted in-script), no training. Run: `randlen_test`, 10751 transitions,
n_bins=5, same quantile-bin edges as RUN-0012/0013 (bins are computed from
the push-length distribution alone, which is frame-independent, so they
match exactly by construction -- same corpus, same `--n-bins 5`).

## Reproduction check (mandatory)

Clean-input baseline overall accuracy on `randlen_test`, this run: **0.4564**
-- reproduces RUN-0009/RUN-0012's published number exactly.

## Three-way per-bin table (push length, WORLD frame, same bins as RUN-0012)

| bin | len mean (mm) | base_acc (clean) | degr_acc (input-resampled) | warp_acc | delta_degraded | delta_warped | frac_of_deficit |
|---|---|---|---|---|---|---|---|
| 0 | 14.8 | 0.4050 | 0.1584 | 0.2510 | **-0.2467** [-0.259,-0.234] | -0.1541 [-0.166,-0.142] | +1.60 [+1.54,+1.68] |
| 1 | 28.3 | 0.4507 | 0.3026 | 0.3803 | **-0.1481** [-0.154,-0.142] | -0.0703 [-0.075,-0.065] | +2.11 [+2.00,+2.23] |
| 2 | 38.8 | 0.4634 | 0.3310 | 0.4157 | **-0.1324** [-0.137,-0.128] | -0.0477 [-0.052,-0.044] | +2.78 [+2.60,+2.97] |
| 3 | 50.4 | 0.4718 | 0.3531 | 0.4374 | **-0.1187** [-0.123,-0.115] | -0.0343 [-0.037,-0.032] | +3.46 [+3.23,+3.73] |
| 4 | 65.0 | 0.4691 | 0.3610 | 0.4500 | **-0.1080** [-0.111,-0.105] | -0.0191 [-0.021,-0.017] | +5.68 [+5.08,+6.34] |

Overall (whole corpus): baseline_clean=0.4564, baseline_degraded=0.3146,
warped=0.4000; delta_degraded=-0.1419, delta_warped=-0.0564,
frac_of_deficit=+2.52.

`frac_of_deficit = delta_degraded / delta_warped` per bin (bootstrapped
jointly with the two deltas, same resample indices) -- how much of the
warped model's own deficit the degraded-input baseline's delta would account
for if it were the same mechanism at the same scale.

## Verdict

**The shape is monotonic and shrinks with push length in both arms -- same
direction as the hypothesis -- but the match is not clean, and it fails in
both directions at once, not just one.**

1. **Direction confirmed.** `delta_degraded` shrinks monotonically from
   -0.247 at 14.8mm to -0.108 at 65mm, the same direction as the warped
   model's own -0.154 -> -0.019. Input resampling alone, with zero change to
   the push frame or the action encoding, reproduces a length-dependent
   penalty that gets smaller as pushes get longer -- consistent with the
   "constant absolute blur over a growing `err_pers` denominator" mechanism.

2. **Magnitude overshoots at every bin, and increasingly so.** The degraded
   baseline's delta is 1.6x to 5.7x LARGER (more negative) than the warped
   model's actual observed deficit at every single bin -- `frac_of_deficit`
   is never below 1.6 and grows with push length rather than sitting near 1.
   Two plausible, non-exclusive reasons, neither of which the brief asked to
   be resolved by training: (a) the round trip applied here is a
   warp-THEN-unwarp (two resamplings, since the control needed to hand the
   baseline a normal world-frame occ0), whereas the warped model's own occ0
   only ever suffers ONE resampling (into the canonical frame, never back);
   (b) the baseline was never trained on resampled input, so it has no
   opportunity to have partially compensated for it the way the warped model
   -- trained end-to-end on exactly this degradation -- necessarily has. Both
   make this a harsher, not milder, proxy than the true single-resampling
   cost, so the overshoot is not surprising, but it means the raw magnitude
   comparison cannot be read as "input resampling explains 250%+ of the
   deficit" -- it is not isolating the same quantity 1:1.

3. **The STEEPNESS does not match, even allowing for the magnitude
   offset.** Normalizing each arm's own bin-0 value to 1: the warped model's
   deficit falls to 12% of its bin-0 value by bin 4 (an 8.1x bin0/bin4
   range); the degraded-input baseline's delta only falls to 44% of its own
   bin-0 value (a 2.3x range). If input resampling of this kind were the
   dominant driver of the warped model's specific SHAPE, the two profiles,
   scaled to their own start, should decline at comparable rates -- they do
   not. The degraded-input baseline is far flatter across the length range,
   which is why `frac_of_deficit` grows monotonically with length rather
   than sitting at a roughly constant fraction.

**Read plainly: input resampling is a real, direction-consistent
contributor to a shrinking-with-length deficit shape, but this control does
not show it is the dominant, or even a well-calibrated, explanation of the
warped model's specific profile.** It is at least as consistent with "input
resampling adds its own smaller, shallower shrinking-with-length term, on
top of the ~12-30% metric-artifact term RUN-0013 already found, with a
still-unidentified third component making up the difference in steepness"
as with "input resampling is essentially the whole story." Given the
magnitude/steepness mismatch, I would not report the third cause as
identified. This is a partial, not a full, confirmation.

## Consequence for `canon_res=96`

Per the brief, this does NOT rise to "the control confirms the hypothesis"
cleanly enough to assert Phase B's `canon_res=96` decision should be
revisited on this evidence alone -- but since input resampling is at least a
real contributor, the cheap model-free measurement below is still worth
recording.

**Identity round-trip fidelity at canon_res=64 vs 96** (no model; real
`occ0` from `randlen_test`, all 10751 rows, native pixel derivation, same
`push_frame_roundtrip` used above):

| canon_res | full-image RMS(occ0_roundtrip - occ0) mean | swept-region RMS mean |
|---|---|---|
| 64 | 0.0732 | 0.0878 |
| 96 | 0.0528 | 0.0631 |

**canon_res=96 removes ~28% of the round-trip RMS degradation relative to
64** (27.9% full-image, 28.2% swept-region -- the two measures agree closely).
That is a real, non-trivial fidelity gain, but it is far from eliminating the
degradation (RMS at 96 is still ~72% of the RMS at 64) -- supersampling
would shrink, not remove, whatever share of the deficit is attributable to
input resampling. Given the mismatch found above, this bounds the plausible
benefit of the `canon_res=96` arm: it should not be expected to close most
of the remaining deficit, only to reduce the input-resampling component of
it by roughly a quarter to a third.

## Caveats

- The round trip used here (warp-then-unwarp-then-blend, to hand the
  baseline a normal world-frame image) is harsher than the single warp the
  warped model's own `occ0` actually suffers (see point 2 above) -- this
  control's absolute deltas are not directly comparable 1:1 to the warped
  model's deltas, only the qualitative shape is.
- The baseline was never trained on resampled input, which the warped model
  effectively was (its whole training run used this degraded representation)
  -- some of the overshoot may be "out of distribution for this baseline,"
  not "the true cost of the resampling," and this control cannot separate
  the two without training (out of scope, no training permitted this run).
- `frac_of_deficit`'s bootstrap CI is computed jointly with the two deltas
  (same resample indices) but its own sampling distribution is a ratio of
  two correlated bootstrapped quantities and is not itself validated against
  a closed-form check; treat the reported CI as indicative, not exact.
- The 64-vs-96 fidelity measurement uses `blend=True` (round-trip default),
  which restores everything outside the validity mask exactly (from the
  original `occ0`) -- the RMS numbers above are already averaged over both
  the always-perfect exterior and the imperfect interior, so the interior-only
  degradation is somewhat understated by the full-image column; the
  swept-region column (interior to the scored band) is the more relevant one
  and shows the same ~28% figure.
