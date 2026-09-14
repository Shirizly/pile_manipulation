---
# ---- identity -------------------------------------------------------------
id: EXP-0005
title: Analytic descriptors add ~nothing to a 32x32 visual switched-linear operator in image space; a reconstruction-only learned latent scores far below either
tier: T1
mode: exploratory
date: 2026-09-13
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On `Genesis/data/overnight_randlen` (same 213-file, 80/20 file-stratified,
  seed-0 split as EXP-0004; 87040 train / 22016 holdout transitions), for a
  switched-by-push-length-bin (EXP-0003 6-bin scheme) ridge operator scored
  by held-out swept-region image `accuracy`: (a) adding 14- or 94-dim
  analytic push-frame descriptors to a 32x32 canonical-visual state moves
  holdout accuracy by <=0.0016 absolute (0.1697 -> 0.1712/0.1713) -- inside
  noise of each other; (b) a 64-dim learned latent (trained only for
  reconstruction, frozen, +94-dim descriptors) scores 0.0602 holdout,
  roughly a third of the visual operator's 0.1697, i.e. a clear regression;
  (c) switching by push length continues to help in every representation
  tested (visual, hybrid, latent) by a similar relative margin to EXP-0003's
  64x64 pixel-space result.

prediction: null  # exploratory

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 0ddab20f
  dirty: true                      # see EXP-0004's identical note; unrelated
                                    # pre-existing dirty tree, this record's
                                    # own inputs are untracked temp scratch
  data_commit: "unrecorded (overnight_randlen has no provenance block, same gap as EXP-0001..EXP-0004)"
  script: "experiments/temp/hybrid-vis-desc/{build_data.py,fit_hybrid.py} (stage 2); experiments/temp/latent-switched/{train_encoder.py,fit_latent_switched.py} (stage 3)"
  data: ["Genesis/data/overnight_randlen (213 files, mixed/piled/scattered, cube)"]
  code_path: "fit_linear_foresight.swept_region_mask/metrics (accuracy, shared with Baselines/LinearForesight); stage-3 encoder/decoder is a small conv autoencoder defined in train_encoder.py, not a project module"
  seed: 0
  split: "same 170/43-file, 80/20 stratified-by-spawn_mode split as EXP-0004 (87040 train / 22016 holdout transitions); stage 3 reuses stage 2's cached tensors directly (`hybrid-vis-desc/cache/{train,test}_cache.pt`) so both stages score identical rows"
  runtime: "stage 2: not recorded precisely (flagged a budget overrun in its own RESULTS.md -- see Threats); stage 3: encoder training 16s (GPU), switched-linear fit+scoring 208s (CPU)"
  runs: []

budget:
  declared: "not separately declared for this promotion; stage 2's OWN session declared ~110k tokens/45min and exceeded it (see Threats); this record only promotes already-finished numbers, no new computation"
  spent: "promotion pass: ~15 min / ~20k tokens"
  outcome: within

design:
  varied:
    representation: ["32x32 visual only", "visual + 14-dim descriptors", "visual + 94-dim descriptors", "64-dim reconstruction latent only", "latent + 94-dim descriptors", "descriptors only (image-undefined, scored 0 by construction)"]
    model: ["switched (6 bins)", "global (unswitched)"]
    ridge_lam: [0.1, 1.0, 10.0]
  held_fixed:
    rasteriser_and_resolution: "32x32 canonical push-frame occupancy (fit_linear_foresight's own convention), same crop/plate-pixel-width as Baselines/LinearForesight"
    split: "identical 170/43-file split and cached tensors for stages 2 and 3"
    binning: "same EXP-0003 6-bin scheme as EXP-0004"
    fit_recipe: "ridge toward IDENTITY (not toward zero -- appropriate for an occupancy-valued or latent state, unlike EXP-0004's toward-zero z-scored descriptor recipe)"
  baselines: [persistence, "global (unswitched) operator of the same representation"]
  metric: "accuracy (image, swept region, per experiments/METRICS.md) -- the standard metric, not a proxy, unlike EXP-0004"

noise_floor: "not measured -- single file-level split (seed 0), no fold sweep. The hybrid-vs-visual gap (0.0016) is small enough that its sign should not be trusted without one; the latent-vs-visual gap (0.11 absolute, ~65% relative) is far larger than any lam-sensitivity observed (<=0.001 across the {0.1,1.0,10.0} sweep) and is not plausibly noise."

depends_on: [push-frame-warp-roundtrip, occ-rasteriser-consistency]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  Holdout switched accuracy: visual 0.1697, +14desc 0.1712, +94desc 0.1713,
  latent-only 0.0484, latent+94desc 0.0602, descriptors-only 0.0000
  (undefined in image space by construction), persistence 0.0000. Descriptors
  add at most +0.0016 absolute to the visual state (noise-level, given no
  floor was measured) -- the visual channel already contains everything the
  hand-designed descriptors summarise. The reconstruction-only latent is a
  clear, non-noise regression relative to visual (0.0602 vs 0.1697, roughly
  a third), attributed to lossy reconstruction (holdout MSE 0.0088, not 0)
  discarding fine spatial detail a pixel-space operator can still use.
  Switching beats the matching global/unswitched operator in every
  representation (visual 0.1697 vs 0.1010; latent+desc 0.0602 vs 0.0100;
  latent-only 0.0484 vs -0.0021, i.e. global is WORSE than persistence for
  the unswitched latent operator).
verdict: supported
downgrades: [imprecision, indirectness, incomplete-design, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If the descriptors carried information the 1024-pixel visual state lacked,
adding them would move holdout accuracy by a margin comparable to the
switching effect itself (~0.06-0.1 absolute, per EXP-0003/this record's own
switched-vs-global rows); instead the move is two orders of magnitude
smaller (0.0016). If the learned latent were preserving what a pixel
operator needs, it would approach the visual operator's accuracy; instead it
lands at roughly a third of it, with a measured non-zero reconstruction
error (0.0088) offered as the mechanism -- a design that discriminates
between "latent bottleneck loses information" and "latent representation
is fine" would need to test a dynamics-trained (not reconstruction-only)
latent, which this record explicitly does not attempt (see "What would
change the verdict").

## What was actually run

Stage 2 (`experiments/temp/hybrid-vis-desc/`) fit switched + global ridge
operators (toward-identity) for 5 cells (persistence, visual, hybrid14,
hybrid94, descriptor-only at two widths) x 3 ridge lam values, on the 32x32
canonical-frame cache. It ran over its own declared budget (~110k
tokens/45min) -- flagged honestly in its own `RESULTS.md`, not hidden; no
cell was dropped as a result, everything planned finished.

Stage 3 (`experiments/temp/latent-switched/`) reused stage 2's exact cached
tensors, trained a small conv autoencoder (64-dim latent, 6000 steps, BCE +
a VICReg-style variance hinge) purely for reconstruction, then fit the same
switched/global ridge recipe on `[z || 94-dim descriptors]` and decoded back
through the frozen decoder to score in image space. The variance hinge was
carried over from `latent-killprobe`'s finding (EXP-0007) that a naive
reconstruction objective alone does not prevent collapse; here it was
included from the start rather than discovered mid-run, and reconstruction
loss itself is also a strong anti-collapse pressure (a constant latent
cannot reconstruct varying occupancy grids) -- both are believed to have
contributed to the healthy `z_std` (4.37 final, vs 2e-6 in the un-regularised
killprobe run) but the two effects were not disentangled.

Nothing here re-derives the descriptor-basis question (that is EXP-0004);
this record only asks whether descriptors matter ONCE a visual channel is
present, and separately, how a learned latent compares to raw pixels.

## Numbers

**Holdout `accuracy`, lam=1.0 (flat across {0.1,1.0,10.0}, see source
RESULTS.md files for the full sweep):**

| representation | switched | global (unswitched) |
|---|---:|---:|
| persistence | 0.0000 | -- |
| visual (32x32) | 0.1697 | 0.1010 |
| visual + 14-dim descriptors | 0.1712 | 0.1066 |
| visual + 94-dim descriptors | 0.1713 | 0.1068 |
| descriptors only (14 or 94-dim) | 0.0000 | -- |
| latent only (64-dim, frozen) | 0.0484 | -0.0021 |
| latent + 94-dim descriptors | 0.0602 (best lam=0.1) | 0.0100 |

Per-bin breakdowns, the secondary (non-standard, non-z-scored)
descriptor-space accuracy numbers, and the full lam sweep are in
`experiments/temp/hybrid-vis-desc/results_stage2.json` and
`experiments/temp/latent-switched/results_stage3.json` -- cited by id, not
restated in full here.

## What would change the verdict

- **A dynamics-trained (not reconstruction-only) latent**: both stage-3
  RESULTS.md and this record's own reading agree the regression is most
  plausibly attributable to a reconstruction-only training objective, not to
  learned latents in general. A latent trained end-to-end against the
  prediction loss is the natural next probe and was explicitly NOT attempted
  here (declared as future work in the source RESULTS.md).
- **A latent-width sweep** -- only one width (64-dim) was tried; no
  evidence here on whether a wider latent closes the gap to visual.
- **A noise floor** for the hybrid-vs-visual comparison specifically -- the
  0.0016 gap could plausibly flip sign under a different split; not
  measured.

## Threats

- **`imprecision`**: no noise floor (single seed-0 split); the small
  hybrid-vs-visual gap (0.0016) should be read as "indistinguishable," not
  as a small positive effect.
- **`indirectness`**: per `experiments/METRICS.md`'s own standing caveat,
  `accuracy` is suspect for CROSS-model-type comparisons (exactly the
  comparison here: visual/hybrid/latent are different representations, not
  a parameter sweep within one model). This record reports `accuracy` only
  -- no `slateN` control-metric re-scoring of the SAME model family exists
  yet for stage 2's specific accuracy numbers beyond what EXP-0006 already
  covers for a subset of these cells (visual/hybrid on `slates_multistep`,
  and latent/descriptor-only separately) -- see EXP-0006 for the slateN
  picture, which partially reorders this one (e.g. descriptor-only scores
  0.0 here but a real positive slateN there).
- **`incomplete-design`**: no latent-width sweep (one 64-dim configuration
  only); no dynamics-fine-tuned latent variant; stage 2's own budget
  overrun (flagged, not hidden) means the full lam/lr sensitivity space for
  the encoder was not explored either.
- **`untested-dependency`**: `occ-rasteriser-consistency` is `unchecked` in
  `INVARIANTS.md`, inherited from EXP-0001..0004 for the same reason.
- Considered and dismissed: **provenance mismatch** between stage 2 and
  stage 3 -- stage 3 explicitly reuses stage 2's exact cached input tensors
  (`hybrid-vis-desc/cache/{train,test}_cache.pt`), so the two stages are
  scored on identical rows, not just a similar split.

## Unrelated findings

- Stage 2's `RESULTS.md` documents a real budget overrun (a background job
  plus a "Monitor-wait" mistake burned time before the coordinating agent
  intervened) -- worth noting as a process failure mode for anyone reusing
  this pattern: waiting on a background watcher for a long fit, rather than
  running it in the foreground with a bounded timeout, is exactly what
  `experiment-log`'s "Never wait on a background watcher" rule warns
  against.
- `experiments/temp/hybrid-vis-desc/cache/{train,test}_cache.pt` (2.2GB +
  176MB) uses a DIFFERENT pixel convention (`fit_linear_foresight.
  actions_to_pixels`) than `experiments/temp/dmdc-lenbins`'s own cache
  (`descriptors_b.world_to_pushframe_px`) -- both caches exist side by side
  in `experiments/temp/` under similar-looking descriptor code; anyone
  reusing one cache with the other's fitting code would silently mismatch
  pixel conventions. Flagged in stage 2's own RESULTS.md; repeated here
  since it is exactly the kind of cross-directory confusion this promotion
  pass is meant to prevent from just living in `temp/`.
