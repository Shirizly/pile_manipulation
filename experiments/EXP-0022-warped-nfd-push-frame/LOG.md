# EXP-0022 — Warped NFD (push-frame NFD) — running log

A running, chronological record of what has been done. Numbers live in
`results/`; this file says what was attempted, what was decided and why, and
what state the experiment is in. It is meant to be readable without any
other context.

**Post-run code move (pure refactor, no behaviour change, no retraining):**
the warped/residual NFD code this log describes as `Baselines/NFD/
warped_nfd_lib.py` / `Baselines/NFD/residual_nfd_lib.py` / `Baselines/NFD/
residual_predictor.py` later moved to `model/warped_nfd/` / `model/
residual_nfd/` (see `docs/CODEMAP.md`). Paths below are as they were at the
time of each run; registered type names (`nfd-unet-warped`,
`nfd-unet3ch-residual`, `nfd-unet-warped-residual`) are unchanged, and every
number in `results/` still reproduces from the checkpoints on disk.

## The question

`Baselines/NFD` predicts next-step occupancy in the WORLD frame from
`[occ0, r(p_start), r(p_stop)]`. The LinearForesight baseline instead warps
everything into a canonical PUSH FRAME (push midpoint at the origin, push
direction along +x) with an SE(2) affine warp, predicts there, and unwarps.
That warp collapses the action space from (start, angle, length) to (length)
alone.

This experiment asks: **does giving NFD the same push-frame warp help or
hurt it, against the otherwise-identical world-frame NFD?**

Then a follow-up: in the un-warped NFD the walls are always in the same
pixels, so the network can learn their constant effect from data. Under the
warp the walls land in a different place in every transformed input. So the
second model adds a **warped workspace-extent channel** telling the network
where the walls sit in the transformed frame, and asks whether that recovers
anything the warp cost.

## Arms

| arm | input channels | frame |
|---|---|---|
| `nfd` (existing baseline, not retrained) | `[occ0, r_start, r_stop]` | world |
| `nfd-warped` | same 3, warped | push frame |
| `nfd-warped-walls` | those 3 + warped workspace indicator | push frame |

## Design decisions (settled with the user before any code was written)

1. **Action channels.** Two implementations, both built. `plate_mode=warp`
   warps the world-frame plate renders like everything else (in_channels
   stays 3, identical to the baseline). `plate_mode=canonical` draws the two
   plate renders analytically at their exact canonical pose instead, which
   avoids resampling blur. **Default is chosen by measured cost**: canonical
   if it adds <10% to total step time (forward + all preprocessing),
   otherwise warp.
2. **Loss frame: WORLD.** The warp is differentiable, so the whole
   warp -> UNet -> unwarp -> blend path is trained end-to-end against the
   same world-frame target the baseline uses. The objective is then
   literally identical to the baseline's and only the internal
   representation differs, which is the strongest form of the comparison.
   (The alternative, loss in the canonical frame against a warped target, is
   what LinearForesight's fit does, but it optimises a different quantity,
   so an accuracy gap would partly reflect the objective rather than the
   representation.)
3. **Wall representation: warped workspace indicator.** A 4th channel,
   `to_push_frame(ones)`: 1 inside the workspace, 0 outside. This is exactly
   the information the un-warped NFD gets for free from the walls being
   static, and it is free to compute (it is already half of
   `push_frame_validity_mask`).
4. **Warp geometry: `scale=1.0`, canonical resolution = the grid
   resolution.** Matches what `Baselines/LinearForesight/fit_switched.py`
   uses (it never passes `scale`, so the default 1.0 applies), so the warp
   is the same one the linear baseline was fit under.

## Comparison baseline

`Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth`, whose test numbers are
recorded in **EXP-0001** (accuracy and slateN over 3 corpora x 3 goals x 3
value functions, produced by `Baselines/common/eval_report.py`). The warped
arms must be trained on the SAME corpus and split
(`overnight_randlen_train`, config `nfd_train_3ch_randlen.yaml`) and scored
through the SAME `eval_report.py` path, or the comparison is not one.

## Progress

- **2026-09-22** — Recon done. Confirmed the push-frame warp primitives
  already live in `transforms/functional.py`, but the warp -> apply ->
  unwarp -> blend COMPOSITION exists only as
  `fit_linear_foresight.py::predict_world`, hard-wired to a matrix operator
  and sitting in a root-level script with its own `main()`. Promoting that
  composition to a shared, model-agnostic utility is therefore the first
  task, per the user's requirement that the transformation be a shared
  utility rather than something reached through an otherwise-irrelevant
  class.
- **2026-09-22** — Identified an integration trap: `training/trainer.py`'s
  `_augment_eulerian_batch` applies 4 rotations x 2 flips to the image and
  silently drops every other batch key. A push-frame model needs the push
  endpoints, and rotating the image without rotating the endpoints would
  corrupt every augmented sample. The baseline trains with
  `augmentation: true`, so turning augmentation off would break recipe
  parity; the augmentation must instead learn to carry the endpoints.
- **2026-09-22** — Started the shared-utility refactor (subagent).
- **2026-09-22** — User direction: run a small pilot on
  `slates_multistep/n20_L20mm_train` alone before the full
  `overnight_randlen` training, to shake out the path cheaply.
- **2026-09-22** — Note on the pilot corpus: `n20_L20mm_train` is 90 batch
  files (~11.5k transitions) at a SINGLE nominal push length. In the
  canonical push frame every push in that corpus therefore looks nearly
  identical, which is the most favourable possible case for a push-frame
  model and the least informative about whether the warp generalises across
  lengths. The pilot is a feasibility/shakeout run, not evidence; the
  comparison that decides anything is the `overnight_randlen` one against
  EXP-0001's numbers.
- **2026-09-22** — Shared-utility refactor **done and independently
  verified**. `transforms/functional.py` gained `push_frame_roundtrip`
  (model-agnostic warp -> callable -> unwarp -> blend, differentiable) and
  `canonical_plate_channels` (plate renders drawn analytically in the
  canonical frame). `fit_linear_foresight.py::predict_world` now delegates
  to the shared composition instead of re-implementing it; checked
  bit-identical (max-abs-diff exactly 0.0) against the previous
  implementation across three warp configurations, so no existing
  LinearForesight register row is disturbed. `training/trainer.py`'s
  augmentation now carries an optional `push_px` key through the same
  rot90/flip map it applies to the images. Test suite 252 -> 255 passing,
  no failures.
  - **Convention trap found and documented** (`Baselines/NFD/WARPED_NFD_NOTES.md`):
    `draw_plate_soft` takes its centre in **(row, col)** order while the
    push-frame functions use **(col, row)**, and the two also disagree on
    the angle convention (`plate_angle = pi - phi`). Mixing them without
    the swap silently produces a plausible-looking but wrong render. This
    cost one wrong verification pass before it was caught.
- **2026-09-22** — Started building the warped NFD itself (subagent):
  `Baselines/NFD/warped_nfd_lib.py`, the three pilot configs, the eval
  predictor, the `plate_mode` timing measurement, and correctness checks
  (identity round-trip, gradient flow, train/eval parity, invariance of the
  canonical input across the x8 augmentation, 2-epoch smoke).
  - **Pilot arms decided**: the pilot trains THREE models on
    `n20_L20mm_train` alone — warped, warped+walls, and an UNWARPED control
    on the same data. The control is necessary because no existing NFD
    checkpoint was trained on L20mm alone (`nfd_3ch` is L20mm+L40mm
    pooled), so without it the pilot would have nothing to compare against.
- **2026-09-22** — **Measured the warp's own accuracy ceiling** (see
  `results/warp_accuracy_ceiling.md`). Feeding the TRUE next occupancy
  through the warp round trip and scoring it as a prediction caps
  swept-region `accuracy` at 0.664 at `canon_res=64`, purely from
  resampling. The world-frame NFD baseline scores 0.407-0.509 on comparable
  cells, so at the LinearForesight-matching setting the warped arms sit in a
  narrow band they cannot escape. The ceiling rises with canonical
  resolution (0.453 at 32, 0.749 at 90, 0.818 at 128) because the loss is
  undersampling of a rotated square.
  - **Design consequence, decided with the user**: the main run adds a
    `canon_res=96` arm alongside `canon_res=64`, to separate "the warp
    hurts" from "the resampling hurts". 64 remains the arm that matches the
    LinearForesight warp exactly. Every reported `accuracy` must carry the
    corresponding ceiling as a reference level.
- **2026-09-22** — **Pilot training done** (RUN-0001 unwarped control,
  RUN-0002 warped, RUN-0003 warped+walls; L20mm-only, 20 epochs, identical
  recipe, run sequentially on an idle GPU). All three completed cleanly, no
  NaNs, no divergence. Final world-frame val MSE lands within ~2e-4 across
  the three arms, which is the same order as within-arm epoch-to-epoch
  noise, so **the training curves do not rank the arms** and are not read as
  doing so. Curves and caveats in `results/pilot_training_curves.md`.
  - Flagged anomaly: the warped+walls arm's hard-IoU was much lower than the
    other two for its first two epochs (0.22/0.33 vs 0.63/0.54) before
    joining them from epoch 3 on, while its loss decayed smoothly
    throughout. Recorded rather than smoothed over; currently reads as a
    slow-to-threshold IoU early in optimisation for that arm, not a failure.
- **2026-09-22** — Started pilot evaluation (subagent): all three arms
  scored through `Baselines/common/eval_report.py`, the same harness that
  produced EXP-0001's numbers, on the held-out L20mm eval cell —
  `slateN` over 3 goals x 3 value functions, plus swept-region `accuracy`
  reported against the warp ceiling, plus the `random` ranking floor.
  - Also commissioned the check that `results/warp_accuracy_ceiling.md`
    currently only ARGUES for: whether the warp's resampling loss is near
    enough to uniform across a candidate pool that it cancels in a ranking
    metric. If it is not, `slateN` is damaged by the warp too and the whole
    comparison needs rethinking.
- **2026-09-22** — **Pilot evaluation done** (RUN-0004), scored through
  `Baselines/common/eval_report.py`, the same harness that produced
  EXP-0001's numbers. Full tables in `results/pilot_eval.md`. Headline, and
  it is deliberately not a verdict:
  - On **`slateN`** (the metric that decides) all three arms clear the
    `random` floor by a wide margin and land within ~0.03-0.05 of each
    other, with no arm ahead on all three value functions — the unwarped
    control leads under `lyapunov`, the warped arms lead under
    `mass_in_region` and `signed_mass`. **No clear win or loss for the warp
    at this scale.**
  - On `accuracy` (the suspect metric) the unwarped control is ahead of both
    warped arms, and — importantly — both warped arms score well BELOW their
    own 0.664 resampling ceiling, so on this corpus their prediction quality,
    not the warp ceiling, is what binds. The three accuracy numbers are still
    not a flat three-way comparison, because only the warped arms pay a
    ceiling at all.
  - **Ranking-robustness check**: pushing the true next occupancy of every
    candidate in a pool through the warp round trip preserves the induced
    ranking well in aggregate (mean Spearman 0.992 over 3 pools) but flipped
    the top-1 candidate in 1 of 3. So the claim in
    `results/warp_accuracy_ceiling.md` that resampling "cancels in ranking"
    holds on average but is **not airtight slate-by-slate**, and 3 pools is
    far too small to bound the flip rate. This needs a larger sample before
    anything leans on it — folded into the main run's evaluation.
  - Standing caveat, restated because it governs how all of the above reads:
    `n20_L20mm` is a SINGLE-push-length corpus, so in the canonical frame
    every push in it looks nearly identical. That is the most favourable
    possible case for a push-frame model. The pilot is a shakeout.
- **2026-09-22** — **Main run launched**: four warped arms on
  `overnight_randlen_train`, trained sequentially on an idle GPU
  (`code/run_randlen_arms.sh`), each with the SAME corpus, split, recipe and
  30-epoch gradient-step count as `nfd_train_3ch_randlen.yaml`, whose
  checkpoint is the world-frame NFD baseline scored in EXP-0001. The
  baseline is NOT retrained — it is the comparison point.

  | run | arm | canon_res | wall channel |
  |---|---|---|---|
  | RUN-0005 | warped | 64 (matches LinearForesight exactly) | no |
  | RUN-0006 | warped + walls | 64 | yes |
  | RUN-0007 | warped | 96 (supersampled) | no |
  | RUN-0008 | warped + walls | 96 | yes |

  Before launching, all four configs were checked to build and take a real
  training step with finite, non-zero gradients on every parameter, and the
  wall-channel arms were confirmed to reach the UNet with a 4-channel input
  conv (30,577 params vs 30,541). The epoch count is deliberately NOT reduced
  to compensate for the warp's extra cost: matching the baseline's update
  count is what keeps the comparison honest.
- **2026-09-22** — **User direction: pause after the first arm and test it
  before committing the rest of the GPU time.** The sequential chaining
  script (`code/run_randlen_arms.sh`) was killed while RUN-0005 was at epoch
  7/30, leaving RUN-0005's own training process running to completion;
  RUN-0006/0007/0008 were never started and their run directories are empty
  placeholders. The script is unchanged and can be re-run for the remaining
  arms (it uses `--no-resume`, so re-running it as-is would also retrain
  RUN-0005 — invoke the remaining `run_arm` lines only).
  - Measured epoch time for the warped `canon_res=64` arm on randlen: ~363
    s/epoch, so ~3h for its 30 epochs. That is ~1.75x the world-frame
    baseline's measured 207 s/epoch on the same corpus. The four-arm plan is
    therefore a ~10-12h GPU commitment in total, which is the reason the
    pause-and-test instruction matters.
  - Because the wrapper script was killed mid-run, the `end` event for
    RUN-0005 was NOT written to `experiments/COMMANDS.jsonl` by the script.
    A start event with no end is meaningful (interrupted), so the end event
    is written by hand once training finishes, and labelled as such.
- **2026-09-22** -- **RUN-0005 finished training** (2h43m, 30 epochs, best val
  0.009284 at epoch 28, test hard_iou 0.884540, no NaNs/divergence). Record
  completed in `runs/RUN-0005-warped-randlen-r64/RUN.md`; the reconstructed
  `end` event for `experiments/COMMANDS.jsonl` is labelled
  `"reconstructed": true` there, not presented as a logged timestamp.
  RUN-0006/0007/0008 remain unstarted, per the pause-and-test direction.
- **2026-09-22** -- **RUN-0009: scored RUN-0005 against the world-frame NFD
  baseline (`nfd_randlen`, EXP-0001's checkpoint) on all 3 of EXP-0001's
  corpora (L20mm, L40mm, randlen_test), through `eval_report.py`, both models
  in the SAME process** so the comparison does not depend on EXP-0001's own
  dirty-tree/no-anchoring-commit provenance. Full tables in
  `results/randlen_eval.md`. Headline:
  - **The re-scored baseline reproduces EXP-0001's published accuracy exactly**
    (0.4071/0.5088/0.4564, to 4-5 decimals) -- EXP-0001's numbers were not in
    doubt despite the provenance gap.
  - On `slateN` (the deciding metric): the baseline leads the warped arm on
    `lyapunov` in all 3 corpora and is roughly tied or slightly ahead on the
    mass-based value functions, every gap 0.01-0.03 against a ~0.75-0.94
    scale -- a small, consistent edge for the world-frame baseline, not a
    dramatic loss for the warp and not a wash either.
  - On `accuracy` (suspect metric): the warped arm trails the baseline in all
    3 corpora but sits well below its own corpus-specific resampling ceiling
    everywhere (re-measured per-corpus: L20mm 0.664, L40mm 0.780,
    randlen_test 0.647, via the new `warp_accuracy_ceiling_multi.py`) -- the
    gap reflects prediction quality, not the warp's unavoidable cost.
  - **Ranking-robustness check scaled from the pilot's 3 pools to 42** (14
    each of L20mm/L40mm/randlen_test): mean Spearman 0.9952, top-1 flip rate
    2/42 = 4.76% (Wilson CI wide enough to still contain the pilot's 1/3
    estimate, so that estimate wasn't wrong, just too small to trust). Reads
    as "mostly but not perfectly trustworthy" for `slateN` on a warped arm --
    a real, modest caveat, not a reason to distrust the comparison above.
  - Overall reading: giving NFD the push-frame warp, on the full randlen
    corpus (not the single-push-length pilot), is a small net negative to
    neutral for control-relevant ranking and a clearer negative on raw pixel
    accuracy -- neither a clean win nor a catastrophic loss for the warp.
  - `nfd_warped_randlen`'s test history: RUN-0005 (training) and RUN-0009
    (this eval) in this experiment's own `runs/`; it is not a `weights/`-
    registered reusable model, so no `tests.md` entry was created.
- **2026-09-23** — **Paused and re-planned with the user. See `PLAN.md`,
  which supersedes the four-arm plan.** The key finding is that RUN-0005 was
  handicapped: the x8 rotation/flip augmentation collapses to 2 distinct
  canonical inputs for a warped model, so it drew each gradient step from 4x
  fewer distinct transitions than the baseline at identical compute.
  **RUN-0005 is therefore a lower bound, not a measurement, and its "warp is
  a small negative" verdict does not stand.** RUN-0006/0007/0008 were never
  started and are dropped in favour of the revised plan.
- **2026-09-23** — Phase A started. Trainer gained an `augmentation: flip`
  mode (the x2 rotation-free subgroup) alongside the existing `true`/`false`;
  `true`/`false` are byte-identical to before, checked directly, and the
  flip mode was verified to reproduce exactly the first two views of the x8
  group. **Epoch/step accounting settled by measurement**: with
  `augmentation: true` the loader batch is 32//8 = 4, giving 22,270
  steps/epoch on the 89,081-transition corpus, so the baseline's 30 epochs is
  668,100 gradient steps -- not the ~83,500 its own config header claims.
  That header states the NO-augmentation figure and is wrong by 8x; logged in
  `experiments/OPEN_ISSUES.md`. It does not affect the EXP-0001 comparison
  (baseline and arms share the accounting), only the stated rationale for the
  epoch count and any future run sized by trusting it.
  - **RUN-0010 launched**: the fair re-run of RUN-0005 --
    `augmentation: flip`, 120 epochs = 668,040 steps, matched to the baseline
    on gradient steps and approximately on wall-clock, while seeing 4x more
    distinct transitions per step. `save_every_n_epochs: 10` also yields an
    epoch-30 checkpoint, which is the "same epochs, a quarter of the
    wall-clock" reading of the same run, for free.
  - Two no-training diagnostics started in parallel (CPU only, so they do not
    contend with the GPU run): A1/A2 canonical-frame scoring across the NFD
    and linear models, including whether the framing REORDERS models and
    whether warped-goal `slateN` helps or hurts (RUN-0011); and A3
    wall-proximity stratification of the warped-vs-baseline deficit, the
    cheap gate on whether the wall channel is worth training (RUN-0012).
- **2026-09-23** -- **RUN-0011 (A1+A2) done.** `Baselines/common/eval_report.py`
  gained an additive `--canonical-frame` flag (`_accuracy_canonical`) and
  `WarpedNFDPredictor` gained `predict_occ_canonical` (native canonical
  output, no unwarp); world-frame `accuracy` verified unchanged
  (`nfd_randlen` still reproduces 0.4071/0.5088/0.4564 with the flag on).
  **A1: canonical-frame scoring does NOT reorder the models** -- identical
  ranking to world-frame `accuracy` for all 6 models (`nfd_randlen`,
  `nfd_warped_randlen`, both LinearForesight resolutions x
  switched/single) across all 3 corpora. The framing debate does not change
  any conclusion drawn so far. **A2: warped-goal `slateN` is REFUTED, not
  merely unhelpful** -- warping the goal into each candidate's own push
  frame (instead of warping the state under a shared frame) raised the
  top-1 flip rate from the world-frame baseline's 2/42 = 4.76% to
  **20/42 = 47.6%**, worst on `randlen_test` (13/14 = 92.9%) where pools mix
  the widest range of push lengths/directions per slate -- exactly the
  mechanism the plan predicted (goal-warping doesn't cancel across a
  ranking the way state-warping does, because every candidate distorts the
  goal differently). Full tables and the asymmetry discussion:
  `results/frame_scoring.md`. Also found and worked around:
  `Baselines/LinearForesight/runs/operators_res64.pt` was missing from disk
  (never git-tracked); a same-shaped stray copy was restored to unblock
  scoring, provenance not independently re-verified --
  `experiments/OPEN_ISSUES.md` entry added for the owner to confirm or
  regenerate.
- **2026-09-23** -- **RUN-0012 (A3, wall-proximity gate) done.** Stratified
  per-transition swept-region `accuracy` by wall distance (mm, push END point
  to nearest workspace boundary) for `nfd_randlen` vs. `nfd_warped_randlen`
  (RUN-0005) on `overnight_randlen_test` (primary) plus L20mm/L40mm.
  **Corpus-dependent result, not a clean gate either way.** In L20mm/L40mm
  (single nominal push length, so wall distance and length are naturally
  near-uncorrelated there) the warped deficit grows monotonically toward
  walls and SURVIVES a push-length-band control (L20mm: delta -0.202 closest
  -> -0.012 farthest, still -0.182 -> -0.019 within one fixed-length band).
  But on `overnight_randlen_test` -- the corpus this decision should be based
  on -- push length is by far the dominant driver of the deficit's shape (8x
  the range wall distance produces: -0.154 to -0.019 vs. -0.072 to -0.042),
  and the wall-distance delta goes flat/non-monotonic once a push-length band
  is held fixed. **Verdict: not well motivated to spend 3h on the wall
  channel before Phase A4's fair re-run** -- the deficit on the corpus that
  matters is mostly a length effect, and the wall effect confirmed twice in
  the narrower cells does not clearly transfer to `randlen_test`, though a
  smaller residual there is not ruled out (limited power: one 2150-row
  length band). A metric bug was found and fixed en route: per-row
  `accuracy` must aggregate `err_pred`/`err_pers` separately before forming
  the ratio (it's a ratio of means, not a mean of ratios) -- an initial
  per-row-ratio version produced accuracy values in the millions. Full
  tables, the fix, and the verdict: `results/wall_proximity.md`.
- **2026-09-23** -- **RUN-0013 (length stratification under canonical-frame
  scoring + a model-free encoding-degeneracy check) done.** Followed up
  RUN-0012's finding that the warped deficit is 8x more sensitive to push
  length than to wall distance, to discriminate "metric artifact" (world-
  frame `accuracy`'s denominator is small at short lengths) from "real
  encoding failure" (canonical plate channels overlap as L->0).
  `wall_proximity_stratified_eval.py` gained a `--frame {world,canonical}`
  flag reusing its existing stratification/aggregation/bootstrap machinery.
  **A real scoring bug was found and fixed en route**: canonical-frame
  truth/region/prev must be warped using the NATIVE pixel derivation
  `predict_occ_canonical` returns, not the world `actions_to_pixels`
  derivation used for the swept-region mask -- the two differ by a small but
  SYSTEMATIC ~0.5px offset, and using the wrong one silently dropped
  `nfd_warped_randlen`'s canonical accuracy from 0.452 to 0.295 (caught only
  because the task brief mandated reproducing RUN-0011's published 0.4536
  first; `eval_report.py::_accuracy_canonical` already did this reassignment
  correctly). **Verdict: neither explanation survives alone.** Canonical
  scoring shrinks the world-frame deficit by 12-30% at every length bin (more
  at short lengths) but a strongly monotonic 6.5x-range residual survives --
  explanation (1) is real but partial. Directly measuring the canonical
  two-plate-channel encoding's geometry (no model: correlation/L2 between the
  two channels vs. push length) found the channels are only meaningfully
  overlapping below ~5-10mm and are FULLY separated (correlation flat at
  ~0, L2 at its exact `sqrt(2)` plateau) from ~15mm onward -- yet the deficit
  keeps shrinking monotonically all the way to 65mm with ZERO further change
  in geometric distinguishability across that whole range, refuting the
  literal "two plates blur together" mechanism as the shape's driver.
  **A third, unidentified length-dependent cause accounts for most of the
  deficit's shape**; PLAN.md's Phase B redesign is not thereby vindicated or
  ruled out, since it could still help for reasons other than the literal
  overlap story. Full tables and both checks: `results/length_stratified.md`.
- **2026-09-23** -- **RUN-0014 (input-resampling control) done.** Tested
  whether the warped model's `occ0` suffering one `grid_sample` before the
  network sees it (a roughly constant absolute blur, costing more when
  `err_pers` is small at short pushes) is RUN-0013's unidentified third
  cause. Fed the unmodified world-frame baseline an `occ0` round-tripped
  through `push_frame_roundtrip` (identity `fn`, `canon_res=64`, native
  pixel derivation matching `WarpedNFDPredictor._build_fn` exactly) and
  scored it world-frame against the same RUN-0012 length bins.
  `wall_proximity_stratified_eval.py` gained `--degraded-baseline`
  (`run_predictor_degraded_input`, `bootstrap_delta3`, `strat_table3`).
  Reproduced the published 0.4564 clean-input baseline exactly. **Partial,
  not clean, confirmation**: the degraded-input delta shrinks monotonically
  with push length (-0.247 at 14.8mm to -0.108 at 65mm), the same direction
  as the warped model's own deficit, but overshoots it in magnitude at every
  bin (1.6x-5.7x, growing with length) and is far shallower in its own
  relative decline (falls to only 44% of its bin-0 value by the longest bin,
  vs the warped model's 12%) -- the two profiles do not track once magnitude
  is accounted for. Likely confounds acknowledged, not resolved: this
  control's round trip is a harsher double-resampling (warp+unwarp) versus
  the warped model's single warp, and the baseline was never trained on
  resampled input the way the warped model implicitly was. Model-free
  identity-round-trip fidelity check (no model, real `occ0`): `canon_res=96`
  removes ~28% of the round-trip RMS degradation relative to 64 (full-image
  and swept-region measures agree) -- a real but modest gain, not enough on
  its own to justify expecting `canon_res=96` to close most of the residual
  deficit. Full tables, caveats, and the fidelity numbers:
  `results/input_resampling_control.md`.
- **2026-09-23** — Phase A diagnostics **paused by decision** after
  RUN-0011..0014; see the "Phase A interim state" section appended to
  `PLAN.md` for what each settled. In short: canonical-frame scoring does not
  reorder anything (so it is safe to adopt); warped-goal `slateN` is refuted
  (47.6% top-1 flip vs 4.76%); the wall hypothesis does not survive a
  length control on `randlen_test`, so the wall channel is not worth training
  yet; the real driver is PUSH LENGTH, and the obvious explanation for that
  (canonical action-channel overlap) is refuted, while input resampling is a
  real but uncalibrated partial contributor.
  - The pause is the point: RUN-0012/0013/0014 all measured RUN-0005's
    checkpoint, which carries the x8-augmentation handicap. **Hypothesis for
    the unidentified length-dependent cause: the handicap is not uniform
    across push length.** Short pushes are a rare tail of this corpus, and
    4x less data diversity per gradient step should hurt rare regions most.
    That predicts RUN-0010's flip-only augmentation will FLATTEN the length
    profile — testable for free once it lands, and a reason not to spend more
    budget characterising a checkpoint already slated for replacement.
- **2026-09-23** — R1/R2 (image-space residual parameterisation, per
  `results/residual_vs_direct_survey.md`): new registrations
  `Baselines/NFD/residual_nfd_lib.py` (`nfd-unet3ch-residual`,
  `nfd-unet-warped-residual`; loss unchanged, `eulerian_combined` MSE
  against absolute `occ1`; parameterisation is `clamp(occ0 + tanh(delta),
  0, 1)`, `residual: false` asserted on the wrapped UNet to avoid
  double-counting with its own occ0-into-logit skip) + eval-time
  `Baselines/NFD/residual_predictor.py`, both additive (no existing file's
  behaviour changed; `train_nfd.py` gained one import line).
  **The model-free ceiling check (no training needed) is the clean result
  so far**: round-tripping the TRUE residual `occ1-occ0` (warp → identity →
  unwarp, no blend) at `canon_res=64` scores accuracy-ceiling 0.6852, barely
  above the existing DIRECT-formulation ceiling of 0.6640
  (`results/warp_accuracy_ceiling.md`) — **not the substantial rise the
  mechanism predicts**. Root cause found and verified (not just argued):
  bit-exact preservation of the unchanged region (`occ_pred == occ0`) holds
  ONLY where `push_frame_validity_mask` is EXACTLY 0 (~21.5% of pixels,
  confirmed: max deviation there is 0.0) — a thin partial-coverage boundary
  band (~1.3% of pixels, mask strictly in (0, 0.5)) is NOT bit-exact and
  carries deviation up to 0.39, because bilinear `grid_sample` weights are
  non-negative so a zero row-sum forces every weight to zero, but a small
  positive row-sum does not. Dead-gradient check on 2-epoch smoke
  checkpoints (real batch, both arms): `clamp` is active on ~96.5-96.8% of
  pixels (expected — background pixels where occ0 and target agree), but
  only ~12.6-12.9% of pixels that actually need to change (`|occ1-occ0| >
  0.05`) fall in the dead zone — training is not starved. Both 2-epoch
  smoke runs showed healthy, monotonic loss decay. **Full 20-epoch
  RUN-0015/RUN-0016 training was launched (in parallel with each other and
  with RUN-0010, all three confirmed alive throughout) but did not finish
  within this task's budget** — 3-way GPU contention slowed epochs well
  below the ~30-38s/epoch measured for RUN-0001/RUN-0002 on an
  otherwise-idle GPU (R1 at 3/20 epochs, R2 at <1/20 epochs when this task's
  budget ran out). `slateN`/`accuracy` comparison against RUN-0001/RUN-0002
  is therefore NOT YET DONE — left for a follow-up once the checkpoints
  finish. See `results/residual_pilot.md` for the full writeup of what
  could be completed.
- **2026-09-23 (continued)** — RUN-0015/RUN-0016 both finished (20/20 epochs,
  clean, no NaNs; timing inflated ~2-2.1x by 3-way GPU contention with
  RUN-0010, which was confirmed alive throughout and never touched).
  RUN-0017 scored all four arms (RUN-0001/RUN-0002/R1/R2) through
  `eval_report.py` on L20mm; RUN-0001/RUN-0002 numbers reproduce
  `pilot_eval.md` exactly. **R2 (warped residual) beats RUN-0002 (warped
  direct) on `accuracy` (0.4006 vs 0.3596, ~11% relative) and on 2 of 3
  `slateN` value functions** — nearly closing the warp's whole accuracy
  gap to the unwarped arms (RUN-0001 0.4018). **This is a real gain the
  ceiling check does NOT explain**: the residual-formulation ceiling
  (0.6852) is barely above the direct ceiling (0.6640, ~3-5% relative), far
  too small on its own to account for an 11%-relative trained-model gain —
  the likely cause is an optimisation/inductive-bias effect of the tanh-
  residual parameterisation itself, not the "preserve 98% bit-exact"
  argument the design was built on (that argument is independently
  confirmed narrower than stated — see the R1/R2 bullet above). R1 also
  modestly beats RUN-0001 (0.4091 vs 0.4018), as predicted ("little" gain,
  since RUN-0001 already has an occ0-skip). Full table:
  `results/residual_pilot.md`.
- **2026-09-23** — **RUN-0010 finished and was scored (RUN-0018). This is the
  answer to the experiment's question.** Best val 0.009120 at epoch 108
  (RUN-0005: 0.009284). Scored against the world-frame baseline and RUN-0005
  on all three EXP-0001 corpora, in both frames.
  - **The augmentation fix removes the control deficit.** `slateN`/lyapunov
    goes from consistently-below the baseline (RUN-0005: .820/.908/.927 vs
    .853/.932/.942) to **parity or slightly above on 2 of 3 corpora**
    (RUN-0010: .858/.911/.943). So RUN-0005's "warp is a small negative"
    really was the handicap talking, on the control metric.
  - **The fix does NOT recover image accuracy.** RUN-0010 scores
    .350/.443/.409 against the baseline's .407/.509/.456 — essentially
    unchanged from RUN-0005, and still far below the warp's own resampling
    ceiling, so prediction quality is what binds, not the warp's unavoidable
    cost.
  - **Verdict: the warp does not help.** Roughly neutral on control, clearly
    negative on image prediction. Recorded as `refuted` in `EXPERIMENT.md`
    and as C-027 in `REGISTER.md`. No seed-level noise floor was measured, so
    the per-cell margins must not be read as ordering the arms — the verdict
    rests on the direction being consistent across three corpora.
  - **Unexpected and practically the most useful result here**: RUN-0010's
    **epoch-30** checkpoint — a QUARTER of the baseline's wall-clock and
    gradient steps — matches or beats its own epoch-108 checkpoint on 7 of 9
    `slateN` cells. Flip-only augmentation on a push-frame model buys a
    CHEAPER model at equal control quality, even though it does not buy a
    better one.
- **2026-09-23** — **`EXPERIMENT.md` written** (T1, verdict `refuted`, grade
  `very-low`: imprecision, incomplete-design, provenance,
  untested-dependency) and `REGISTER.md` row C-027 added.
  `scripts/check_register.py` exits 0. The record had been deferred while
  results were still arriving, which left a dangling EXP-0022 citation from
  EXP-0023 breaking the validator — that is now closed.
- **2026-09-23** — **RUN-0020: the standing length-profile hypothesis is
  REFUTED.** Flip-only augmentation does NOT flatten the warped arm's
  short-push accuracy deficit — every bin improved slightly but the ratio
  between the shortest and longest bins got *larger* (11.4x, from 8.1x). The
  fair arm is uniformly a bit better across the whole length range, which
  reads as "a better-trained model", not as the specific short-push rescue
  predicted. See `results/flipaug_length_profile.md`. The wall-distance
  stratification reproduces on this arm too (flat once length is fixed), so
  the gate against training a wall channel stands.
  - **The length dependence is now unexplained after four candidate causes**:
    metric artifact (12-30%), canonical channel overlap (refuted), input
    resampling (partial, uncalibrated), augmentation handicap (refuted). A
    short push is simply harder for a push-frame model than a world-frame
    one, and this line of work ends without knowing why.
- **2026-09-23** — **RUN-0019/RUN-0021: warped + flip augmentation + residual
  on randlen — the combination the record implied, never previously run.** Best
  val 0.005634 at epoch 112 (against 0.009120 for the same arm without the
  residual). **The residual recovers the warp's accuracy deficit**:
  .384/.469/.462 against the baseline's .407/.509/.456, i.e. parity on the
  held-out `randlen_test`. `slateN` remains a wash across all three arms.
  See `results/residual_flipaug_randlen.md`.
  - **Verdict unchanged but its shape changes.** The warp still does not help —
    `slateN` is a wash and accuracy only reaches parity. But it no longer
    *costs* anything, so the honest statement is "the warp is free once the
    model is parameterised and trained properly", not "the warp hurts".
  - **The residual, not the warp, is what transfers**: it replicates from the
    L20mm pilot (.360 -> .401) to full randlen scale (.409 -> .462). Whether an
    explicit residual head helps a WORLD-FRAME NFD at randlen scale was never
    tested and is the obvious next run.
- **2026-09-23/24** — **RUN-0022 (world-frame NFD + residual, no augmentation)
  was INTERRUPTED at epoch 43 of 240** when its host session restarted;
  RUN-0023 (flip-only) never started. Scored the surviving epoch-30 and
  epoch-43 checkpoints anyway (RUN-0024) — and **they are the best-scoring
  models on `accuracy` anywhere in EXP-0022**, beating the world-frame baseline
  in all three corpora (+0.025/+0.016/+0.028) and topping `slateN`/lyapunov on
  `randlen_test` (0.9529 vs 0.9417). See `results/residual_worldframe_partial.md`.
  - **"18% of the budget" is the wrong reading.** Gradient steps and data
    exposure diverge under different augmentation: this run is
    under-OPTIMISED (119,669 vs 668,100 steps, 0.18x) and simultaneously
    over-EXPOSED (43 vs 30 dataset passes, 1.43x), because with no
    augmentation one pass costs 2,783 steps instead of 22,270.
  - **Not a clean comparison** — it differs from the baseline in TWO ways at
    once (residual head AND augmentation), so the gain cannot be attributed to
    the residual alone, and the 0.016-0.028 accuracy margins are the size a
    seed-level noise floor might or might not cover.
  - **This makes TODO M1 (augmentation-matched controls) the blocking item.**
    The clean design is `{plain, residual} x {no aug, x8}`, step-matched;
    three of those four cells do not exist.
