---
# ---- identity -------------------------------------------------------------
id: EXP-0022
title: >
  Giving the NFD UNet the LinearForesight push-frame warp does NOT improve it:
  once the arm is trained fairly (flip-only augmentation), control-relevant
  ranking reaches parity with the world-frame baseline on 2 of 3 corpora while
  swept-region image accuracy stays clearly worse; an explicit residual
  parameterisation, not the warp, produced the only real gain
tier: T1
mode: exploratory
date: 2026-09-23
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  Warping occupancy into the canonical push frame (the SE(2) warp
  `Baselines/LinearForesight` is fit under: push midpoint at the origin, push
  direction along +x, canon_res = grid resolution, scale = 1.0) before an NFD
  UNet predicts, then unwarping and blending, improves that UNet's swept-region
  `accuracy` and its `slateN` action ranking relative to an otherwise-identical
  world-frame NFD trained on the same corpus, split, recipe and gradient-step
  count (`Baselines/NFD/runs/nfd_3ch_randlen`, scored in EXP-0001).

# prediction: omitted -- mode is exploratory. No threshold was committed before
# running; this record reports a design exploration and its measurements.

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: a175b981
  dirty: true                     # heavy concurrent multi-agent session; see
                                  # "What was actually run" for the file list
  data_commit: "unrecorded (overnight_randlen / slates_multistep carry no provenance block)"
  script: >
    Baselines/NFD/train_nfd.py (training, via the configs named in design.varied);
    Baselines/common/eval_report.py (all scoring, the same harness EXP-0001 used);
    experiments/EXP-0022-warped-nfd-push-frame/code/ (diagnostics)
  data:
    - "Genesis/data/overnight_randlen_train (5 groups, 89,081 transitions) -- training"
    - "configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml"
    - "configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml"
    - "configs/dataset/genesis_overnight_randlen_test_all.yaml"
    - "Genesis/data/slates_multistep/n20_L20mm_train -- pilot arms only"
  code_path: >
    transforms/functional.py::push_frame_roundtrip (the shared warp -> model ->
    unwarp -> blend composition promoted out of fit_linear_foresight.py for this
    experiment) -> Baselines/NFD/warped_nfd_lib.py::WarpedNFDWrapper (training)
    / Baselines/NFD/predictor.py::WarpedNFDPredictor (eval), which share one
    canonical-stack builder so the two paths cannot drift.
    NOTE (post-run refactor, pure code move, no behaviour change): this code
    moved to model/warped_nfd/lib.py::WarpedNFDWrapper (training) /
    model/warped_nfd/predictor.py::WarpedNFDPredictor (eval) after this
    experiment's runs completed -- the sha-plus-path above no longer resolves
    as written; see docs/CODEMAP.md's "Warped NFD" entry for the current
    paths. Registered type names (`nfd-unet-warped`, `nfd-genesis-3ch-warped`)
    are unchanged.
  seed: 0
  split: >
    Training: the pre-existing file-level split Genesis/data/overnight_randlen_{train,test}
    is REUSED, not re-derived, and is the same split the world-frame baseline was
    trained on -- this parity is the point of the record. val/test 5/5 inside the
    train files is a monitoring slice only, not a generalisation measure.
  runtime: >
    RUN-0005 2h43m (30 epochs); RUN-0010 ~4h45m (120 epochs, inflated by GPU
    contention with concurrent jobs); pilot arms ~11-13 min each; all diagnostics
    and all scoring CPU-only, seconds to minutes each.
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004, RUN-0005, RUN-0009, RUN-0010,
         RUN-0011, RUN-0012, RUN-0013, RUN-0014, RUN-0015, RUN-0016, RUN-0017,
         RUN-0018]
  env: "python 3.10, torch 2.11.0+cu130 (anaconda3/envs/pme), RTX 4070 Laptop 8 GB"

budget:
  declared: "not declared in advance as a single figure -- this record grew from one arm into a multi-phase investigation under user direction; each delegated sub-task carried its own declared budget"
  spent: "~1 day wall-clock, ~10 GPU-hours, ~2M subagent tokens across 11 delegated tasks"
  outcome: exceeded

design:
  varied:
    frame: [world, push-frame]
    augmentation: ["full (x8 rotations x flips)", "flip-only (x2)"]
    action_encoding_in_canonical_frame: [warped plate renders, analytic canonical plate renders]
    output_parameterisation: [direct occupancy, explicit tanh residual added to occ0]
    canon_res: [64]
    training_corpus: [overnight_randlen_train, "slates_multistep/n20_L20mm_train (pilot only)"]
  held_fixed:
    architecture: "UNetModels_modular.UNet, features [4,8,16], final_kernel_size 1, bottleneck None"
    loss: "eulerian_combined, mse 1.0, every other term 0 -- world-frame MSE against the same absolute occ1 target in EVERY arm, including the residual arms (the residual is a parameterisation change, not an objective change)"
    optimiser: "Adam lr 1e-4, StepLR, grad_clip 1.0, mixed precision, batch 32"
    gradient_steps: "668,100 (baseline) vs 668,040 (RUN-0010) -- matched by measurement, not by epoch count"
    grid_resolution: "64x64 (resolution_scale 0.5)"
    scoring: "Baselines/common/eval_report.py, the same harness and code path EXP-0001 used"
  baselines: [persistence, random, "world-frame NFD nfd_3ch_randlen (EXP-0001)", "the warp's own accuracy ceiling"]
  metric: "slateN (leads; 3 goals x 3 value functions), swept-region accuracy (reported beside it, flagged suspect per experiments/METRICS.md)"

noise_floor: >
  Not measured for slateN across training seeds -- every arm is a single training
  run, so the 0.01-0.03 slateN differences reported here are NOT benchmarked
  against run-to-run spread and should not be read as ordering the arms. This is
  the record's main imprecision and it is why the verdict rests on the direction
  being consistent across three corpora rather than on any single margin.
  Separately, EXP-0024 established that the GROUND TRUTH is not a noisy draw:
  re-simulating the same action from the same snapshot returns the same dv
  (within-action/between-action dv variance ratio <= 5.3e-5), so none of the
  spread here comes from the simulator.

depends_on: [randlen-train-test-file-disjoint, occ-rasteriser-consistency,
             goal-mask-axis-convention-row-y-col-x,
             genesis-snapshot-restore-repeat-determinism]
establishes: [push-frame-warp-roundtrip]

# ---- outcome --------------------------------------------------------------
result: >
  slateN/lyapunov (L20mm/L40mm/randlen_test): world-frame baseline
  0.853/0.932/0.942; warped trained with the BASELINE's x8 augmentation
  (RUN-0005) 0.820/0.908/0.927 -- consistently below; warped trained FAIRLY with
  flip-only augmentation (RUN-0010) 0.858/0.911/0.943 -- at or slightly above the
  baseline on 2 of 3 corpora. Swept-region accuracy is NOT recovered by the fix:
  baseline 0.407/0.509/0.456, RUN-0005 0.348/0.459/0.400, RUN-0010
  0.350/0.443/0.409, all far below the warp's own resampling ceiling
  (0.664/0.780/0.647), so prediction quality rather than the warp's unavoidable
  cost is what binds. RUN-0010's epoch-30 checkpoint (a QUARTER of the baseline's
  wall-clock and gradient steps) matches or beats its own epoch-108 checkpoint on
  7 of 9 slateN cells. On the L20mm pilot, an explicit tanh-residual
  parameterisation lifted the warped arm's accuracy 0.360 -> 0.401 (~11% relative,
  nearly closing the gap to unwarped) and took the top slateN spot on 2 of 3 value
  functions -- the largest single gain in this record, and NOT attributable to the
  resampling mechanism it was designed around (the residual formulation's ceiling
  is only 0.685 vs 0.664, far too small to explain it). RUN-0019 then ran the
  combination the record implies -- warped + flip augmentation + residual, on
  randlen, matched to the baseline -- and it RECOVERS the accuracy deficit:
  0.384/0.469/0.462 against the baseline's 0.407/0.509/0.456, i.e. parity on the
  held-out randlen_test corpus, with slateN a wash. The warp still does not
  help, but it no longer costs anything either; the residual parameterisation is
  the transferable finding, replicating from pilot (0.360->0.401) to full scale
  (0.409->0.462).
verdict: refuted
downgrades: [imprecision, incomplete-design, provenance, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

The push-frame warp collapses the action space from (start, angle, length) to
length alone, which is exactly what lets `Baselines/LinearForesight` cover the
whole action space with one operator per length bin. If that reduction in input
variability is what a learned model needs, an NFD UNet given the same warp
should beat an otherwise-identical world-frame NFD trained on the same data for
the same number of updates. Every other factor is held fixed (architecture,
loss, optimiser, corpus, split, gradient-step count), and the warped arm is
scored through the same harness that produced the baseline's own published
numbers — which were reproduced to 4-5 decimals inside this record before any
new number was trusted.

## The finding that reframed the record

**The x8 rotation/flip augmentation is degenerate for a push-frame model.**
`push_frame_transform` is a pure rotation, so all 4 rotated views of a sample
warp to the *same* canonical image, and the 4 flipped views to a second one —
2 distinct inputs, not 8. The first warped arm (RUN-0005) therefore drew every
gradient step from **4x fewer distinct transitions** than the baseline, at
identical compute. Its "the warp is a small negative" result was withdrawn and
re-run as RUN-0010 with `augmentation: flip`, matched to the baseline on
gradient steps (668,040 vs 668,100, measured rather than assumed).

This is the reason the record's headline is a *refutation* rather than a
negative: the fair arm reaches parity on control, so the honest statement is
"the warp does not help", not "the warp hurts".

## What was actually run

Working tree dirty throughout — a heavy concurrent multi-agent session touching
`transforms/functional.py`, `training/trainer.py`, `fit_linear_foresight.py`,
`Baselines/NFD/*`, `Baselines/common/eval_report.py`, `simple_mpc/adapters.py`
and the docs. Two changes to shared code were verified not to disturb existing
evidence: `fit_linear_foresight.py::predict_world` now delegates to the promoted
`push_frame_roundtrip` and was checked **bit-identical** (max-abs-diff exactly
0.0) across three warp configurations; `training/trainer.py`'s augmentation
gained a `flip` mode and an optional `push_px` key, with `true`/`false`
behaviour verified unchanged.

## Numbers

**slateN, goal-averaged per value function** (lyapunov / mass_in_region / signed_mass):

| corpus | baseline | RUN-0005 (x8 aug) | RUN-0010 (flip aug) | RUN-0010 @ epoch 30 |
|---|---|---|---|---|
| L20mm | .853 / .820 / .753 | .820 / .800 / .779 | **.858** / .819 / .745 | .860 / **.849** / **.786** |
| L40mm | **.932** / .783 / .841 | .908 / **.832** / .847 | .911 / .809 / .855 | .909 / .830 / **.859** |
| randlen_test | .942 / **.889** / **.866** | .927 / .888 / .841 | .943 / .866 / .844 | **.950** / .880 / .826 |

`random` ranking floor is within +/-0.04 of 0 in every cell. `persistence` is
excluded as a ranking floor — it predicts dv=0 for every candidate and is
degenerate.

**Swept-region accuracy**, world frame, with each corpus's warp ceiling:

| corpus | baseline (no ceiling) | RUN-0005 | RUN-0010 | RUN-0010 @30 | warp ceiling |
|---|---|---|---|---|---|
| L20mm | **.4071** | .3476 | .3496 | .3501 | .6640 |
| L40mm | **.5088** | .4589 | .4431 | .4617 | .7804 |
| randlen_test | **.4564** | .4000 | .4086 | .3879 | .6470 |

Canonical-frame accuracy is in `results/frame_scoring.md`; it lifts every model
by ~0.04 and **reorders nothing**.

Pilot-scale arms, ceilings, stratifications and the residual result are in
`results/` (`pilot_eval.md`, `warp_accuracy_ceiling.md`, `frame_scoring.md`,
`wall_proximity.md`, `length_stratified.md`, `input_resampling_control.md`,
`residual_pilot.md`, `residual_vs_direct_survey.md`). `PLAN.md` carries the
design rationale and the phase-by-phase state.

## What would change the verdict

- **A noise floor on slateN.** Every arm is one training run. The 0.01-0.03
  differences are not benchmarked against seed-to-seed spread. 3 seeds of
  RUN-0010 and of the baseline would settle whether the parity is real. This is
  the cheapest thing that would most change confidence.
- **The length stratification on RUN-0010**, which was set up but not run. A
  standing hypothesis (`PLAN.md`) predicts the warped arm's short-push deficit
  should FLATTEN under flip-only augmentation, because short pushes are a rare
  tail and reduced data diversity hurts rare regions most. Untested.
- **A residual arm on `overnight_randlen`.** The residual parameterisation is
  the largest gain here but was only measured at L20mm pilot scale, on a
  single-push-length corpus. The combination actually implied by this record —
  warped + flip augmentation + residual, on randlen — was never run.
- **The wall channel**, gated OUT by `results/wall_proximity.md` on the corpus
  that matters, remains untested at randlen scale.

## Threats

- **`imprecision`**: no noise floor on slateN (above). The verdict rests on
  direction being consistent across 3 corpora, not on any margin.
- **`incomplete-design`**: the decisive follow-up cells (seeds, RUN-0010 length
  profile, randlen residual arm) were not run. The record answers "does the warp
  as specified help" and not "is there a warped design that helps".
- **`untested-dependency`**: `occ-rasteriser-consistency` is `unchecked` -- two rasterisers exist in this repo and the eval path's use of one is believed-by-construction, not tested.
- **`provenance`**: dirty tree throughout a concurrent multi-agent session; the
  shared-code changes were verified non-disturbing (above) but the sha does not
  describe the code that ran.
- Considered and **dismissed**: ground-truth simulator noise, established by
  EXP-0024 to be negligible through this mechanism.

## Unrelated findings

- `Baselines/LinearForesight/runs/operators_res64.pt` was **missing from disk**
  (never git-tracked) and was restored from an untracked stray file to unblock
  scoring; its provenance is unverified. Logged in `OPEN_ISSUES.md`.
- `nfd_train_3ch_randlen.yaml`'s header justifies its epoch count with "~83,500
  gradient steps"; the real figure is 668,100 — it quotes the no-augmentation
  arithmetic. Comparisons are unaffected (all arms share the accounting); any run
  sized by trusting that comment is not. Logged in `OPEN_ISSUES.md`.

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- Most arm differences here are below the later-measured noise: slateN gaps < 0.03 are unresolvable at 20 slates x 3 goals (EXP-0026, C-029) and NFD training-seed sd is 0.01-0.04 (EXP-0036, C-044).
- `slates_multistep` (L20mm/L40mm) was simulated with friction 0.3 / density 1000 -- off-training-physics for randlen-trained models (`benchmark-physics-matches-training`, broken); EXP-0048 later found physics sets barely change 20 mm single-push rankings on scatter (C-053), untested for piles/longer pushes.
