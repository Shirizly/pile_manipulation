# RUN-0013 -- push-length stratification under canonical-frame scoring, plus a
# direct test of the action-encoding-degeneracy mechanism

Follows RUN-0012 (world-frame push-length stratification, found the deficit
8x stronger by length than by wall proximity) and RUN-0011 (canonical-frame
scoring, added `WarpedNFDPredictor.predict_occ_canonical`). Question: is
RUN-0012's short-push deficit (1) a metric artifact of scoring in the world
frame (small persistence-error denominator inflating the *fractional*
penalty of a roughly-constant absolute resampling blur), or (2) a real
failure of the canonical action encoding, which literally collapses toward
two identical blurs as push length -> 0?

Both models scored on CPU only (asserted in-script, same as RUN-0012); no
training; RUN-0010 (GPU) was left alone throughout.

## Reproduction check (mandatory before interpreting anything)

Overall canonical accuracy on `randlen_test`, this run: `nfd_randlen`
(baseline) = **0.4960**, `nfd_warped_randlen` = **0.4523**, vs. RUN-0011's
published 0.4954 / 0.4536. Close (within 0.001-0.0015) but not bit-exact --
traced to a genuine floating-point/file-order nondeterminism in
`load_randlen_cell`, not a scoring bug: a standalone full-corpus,
no-chunking replay of `WarpedNFDPredictor.predict_occ_canonical` against
`eval_report.py`'s own reference number reproduced 0.4523 (not 0.4536)
independently of this script, so the residual ~0.0013 gap is not attributable
to anything this run's code does differently from `eval_report.py`. Small
relative to every delta discussed below (0.03-0.15) -- does not affect the
verdict.

**A real bug WAS found and fixed while chasing this reproduction, per the
task brief's warning.** Debugged in isolation
(`/tmp/debug_chunk.py`, not committed): reassigning `start_px, end_px` to the
NATIVE pixel derivation `predict_occ_canonical` returns (rather than reusing
the WORLD `actions_to_pixels` pixel derivation used to build the swept-region
mask) is required before warping truth/prev/region into the canonical frame.
The two pixel conventions differ by a small but SYSTEMATIC ~0.5px offset
(not noise -- `max_px_diff` is ~0.500-0.501px for every row, not a spread
around 0). Using the wrong one silently dropped `nfd_warped_randlen`'s
canonical accuracy from 0.452 to **0.295** -- a huge, wrong number that
would have been read as "canonical scoring makes the warped model much worse
than world-frame scoring," the opposite of RUN-0011's finding, and entirely
an artifact of this script, not a property of the model.
`Baselines/common/eval_report.py::_accuracy_canonical` already does this
reassignment correctly; `wall_proximity_stratified_eval.py`'s new
`--frame canonical` path now matches it (see the inline comment at the
reassignment).

## Push-length-stratified deltas, world (RUN-0012) vs. canonical (this run)

Same 5 quantile bins on `overnight_randlen_test` (bins recomputed from the
push-length distribution, which is frame-independent, so edges match
RUN-0012 exactly -- confirmed: `corr(d_wall, push_len) = 0.104` reproduces
RUN-0012's value to 3 decimals).

| bin | len mean (mm) | world delta (RUN-0012) | canonical delta (RUN-0013) | canonical/world ratio | canonical 95% CI |
|---|---|---|---|---|---|
| 0 | 14.8 | -0.1541 | **-0.1081** | 0.70 | [-0.1177,-0.0988] |
| 1 | 28.3 | -0.0703 | **-0.0559** | 0.80 | [-0.0604,-0.0514] |
| 2 | 38.8 | -0.0477 | **-0.0383** | 0.80 | [-0.0421,-0.0345] |
| 3 | 50.4 | -0.0343 | **-0.0300** | 0.87 | [-0.0331,-0.0269] |
| 4 | 65.0 | -0.0191 | **-0.0167** | 0.87 | [-0.0194,-0.0140] |

Range (bin0/bin4 ratio): world 8.1x (-0.154/-0.019), canonical 6.5x
(-0.108/-0.017). Canonical-frame wall-distance delta (for reference, same
run): -0.056 (closest) to -0.049 (farthest), a 1.15x range -- push length
remains, by a wide margin, the dominant stratifying variable under canonical
scoring too, exactly as it was under world-frame scoring.

**Canonical scoring shrinks the deficit at every length bin (12-30%), most at
the shortest lengths, but does not remove it, and the strongly monotonic
shape survives intact.** This is a real, if partial, metric-artifact
component -- consistent with explanation (1) contributing MORE at short
lengths, where `err_pers` is smallest -- but the residual is large (bin0
still -0.108, not near zero) and still spans a 6.5x range across the corpus.
**Explanation (1) alone does not account for the deficit's shape or
magnitude.**

## Does the canonical action encoding actually degenerate? (model-free check)

Built the exact canonical two-plate-channel encoding `nfd_warped_randlen`
receives (`transforms.functional.canonical_plate_channels`, `plate_mode=
"canonical"`), using the corpus's own geometry (`to_pxl=500` px/m from
`resolution_scale=0.5`; plate size `[0.04, 0.002, 0.0175]` m from
`Genesis/data/overnight_randlen_test/piled_n20/_23_config.yaml`; `canon_res=
world_res=64`, `scale=1.0`, `sigma=0.75` world px -- all `nfd_warped_randlen`'s
own training-time defaults), and swept push length 0-80mm (script:
`experiments/EXP-0022-warped-nfd-push-frame/code/action_encoding_degeneracy.py`,
no model, no data loading, <1s).

Measured the two channels' Pearson correlation and relative L2 distance
(`||r_start - r_stop|| / ||r_start||`) as a function of L:

| L (mm) | corr | L2 / \|channel\| |
|---|---|---|
| 0.0 | 1.000 | 0.00 |
| 1.0 | 0.932 | 0.36 |
| 2.0 | 0.766 | 0.68 |
| 3.8 | 0.495 | 0.99 |
| 5.0 | 0.299 | 1.17 |
| 8.0 | 0.062 | 1.35 |
| 10.5 | -0.001 | 1.40 |
| 14.3 | -0.020 | 1.41 |
| 20-80 (flat) | ~ -0.022 | ~1.414 (= sqrt(2), plateau) |

**The channels are only meaningfully overlapping below ~5mm and are fully
geometrically separated (correlation ~0, L2 at its `sqrt(2)` plateau) by
~10-15mm.** The plateau is exact and stays flat from ~15mm to 80mm -- there is
no further separation to be had once the two plate renders stop overlapping
at all.

**This directly contradicts explanation (2)'s literal mechanism as the
driver of the observed deficit shape.** RUN-0012/0013's bin0 (mean push
length 14.8mm, range [0.1,23.5]mm) sits almost entirely PAST the point where
the channels are geometrically distinguishable, yet it carries by far the
largest deficit (-0.108 canonical, -0.154 world). More decisively: bins 1-4
(mean lengths 28.3-65.0mm) are ALL past the ~15mm plateau -- the encoding is
equally, maximally separated in every one of those four bins (correlation
flat at ~-0.02, L2 flat at ~1.414) -- yet the canonical deficit still shrinks
monotonically across them, from -0.056 to -0.017, a 3.3x range with **zero
change in geometric channel distinguishability to explain it**. Whatever
drives that residual monotonic trend across 28-65mm, it is not "the model
cannot tell the two plate channels apart" -- that mechanism has already
saturated by the time bin 1 starts.

## Verdict

**Neither competing explanation survives intact; the two effects do not
sum to the whole deficit, and a third, unidentified length-dependent cause is
needed for the bulk of it.**

- Explanation (1), metric artifact: real but partial. Canonical scoring
  recovers 12-30% of the world-frame deficit at every length bin (more at
  short lengths, consistent with a shrinking-denominator effect), but leaves
  a still-strongly-monotonic 6.5x-range residual. Not "largely disappears."
- Explanation (2), literal channel-overlap degeneracy: **refuted as the
  primary mechanism.** The channels are only geometrically indistinguishable
  below ~5-10mm, a narrow slice at the extreme low end of bin0's range, but
  the deficit is large and monotonically shrinking all the way out to 65mm,
  a range over which the geometric encoding is IDENTICALLY, maximally
  separated throughout (flat correlation/L2 from ~15mm on). The specific
  "two plates blur into one" story in `PLAN.md` does not fit this shape.

**What this does NOT rule out**: PLAN.md's Phase B action-representation
redesign (swept-rectangle channel, FiLM on length) could still help for
reasons other than the literal channel-overlap mechanism -- e.g. the network
may still be relying on the ABSOLUTE POSITIONS of two small, separated blobs
in a way that generalises poorly across length even once they no longer
overlap, or there may be a length-dependent training-data density effect
(shorter pushes rarer/harder in `overnight_randlen_train`, untested here).
This run only rules out the specific "channels look the same" causal story
as the account of the SHAPE seen in RUN-0012 -- it does not test Phase B's
proposed fix directly, nor does it identify what the true residual driver
is. That remains open.

## Caveats
- Both explanations' contributions were only separated additively/ordinally
  (compare deltas, compare thresholds); no attempt was made to fit a
  quantitative decomposition (e.g. regressing the deficit on corr(L) plus a
  metric-artifact term) -- the qualitative mismatch (deficit persists flat
  where geometry is flat) is strong enough on its own, but a joint model was
  not built.
- The encoding-degeneracy check uses ONE representative plate-size config
  (`piled_n20/_23_config.yaml`); `overnight_randlen` pools 5 spawn-mode
  groups and the plate config is very unlikely to vary across them (fixed
  hardware parameter, not sampled), but this was not independently verified
  across all 5.
- RUN-0005/RUN-0013's warped checkpoint still carries the x8-augmentation
  4x-undersampling handicap flagged throughout this experiment (PLAN.md);
  the ABSOLUTE deficit magnitudes here may shrink once RUN-0010's fair
  re-run (flip-only augmentation) lands, but the length-vs-geometry
  MISMATCH argued above is a property of the encoding and the metric, not of
  which checkpoint is scored, so the refutation of explanation (2) is not
  expected to depend on which checkpoint is used.
