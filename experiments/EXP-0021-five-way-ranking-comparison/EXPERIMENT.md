---
# ---- identity -------------------------------------------------------------
id: EXP-0021
title: >
  Five-way action-ranking comparison on one shared corpus and split:
  analytic descriptors vs switched-linear visual foresight vs NFD vs
  LeJEPA latent dynamics vs LeJEPA trained with a goal-conditioned value
  loss, against a frozen-random-encoder kill-gate, scored by slateN and
  top1_regret
tier: T1
mode: confirmatory
date: 2026-09-17

# ---- the claim ------------------------------------------------------------
claim: >
  PRE-REGISTERED, NOT YET MEASURED. On DS-0001 (`slates_binned`
  n20_scatter_s20a1000_L20-70mm, 20 slates x 1000 candidates, step 0), over
  3 goals (`corner`, `ring_O`, letter `X`) x 3 value functions (`lyapunov`,
  `mass_in_region`, `signed_mass_in_region`), with EVERY dynamics model fit
  on the SAME corpus and split (`overnight_randlen_train`, 192 files /
  98,304 transitions; held out on `overnight_randlen_test`, 21 files /
  10,752), the ranking of models by `slateN` is stated in `prediction`
  below. No number in this record is quoted from a prior experiment; the
  `random` floor is re-derived inside this run.

prediction:
  discriminating: true
  statement: >
    (P1, the kill-gate) The generic-LeJEPA arm's `slateN`, averaged over the
    9 cells, EXCEEDS the frozen-random-encoder arm's by more than the
    cross-encoder-seed sd measured here (3 seeds). If it does not, LeJEPA
    representation learning buys nothing this metric can see, and P2/P3 are
    reported but not interpreted as representation results.
    (P2, the goal-loss test) The LeJEPA+goal-loss arm EXCEEDS the generic
    LeJEPA arm under the LINEAR VALUE-HEAD readout by more than that same
    sd. The two arms differ in exactly one bit -- whether the head
    w^T(z_x - z_g) + b was trained jointly with the encoder or fitted
    post-hoc on a frozen one -- so this is the only cell that tests
    `docs/experimental_design/goal_loss.md`.
    (P3, the headline ordering) NO direction is pre-registered between the
    latent arms and `visual-switched`/`nfd`. Stated explicitly so that
    whichever way it falls is reported as measured rather than as expected:
    the frozen-random encoder alone reached slateN +0.42 in EXP-0016 while
    the visual operator reached +0.740, so the latent arms landing BETWEEN
    the random-encoder floor and the visual operator is a live and
    unembarrassing outcome.
    REFUTES condition for the programme: if the frozen-random-encoder arm is
    not separated from BOTH learned-encoder arms in either direction, the
    LeJEPA line is not paying for its GPU time and the record says so.
  outcome: "pending -- committed before any arm was trained or scored"

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 28271c09
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0021-five-way-ranking-comparison/code/
  code_path: experiments/EXP-0021-five-way-ranking-comparison/code/
  data: ["DS-0001", "Genesis/data/overnight_randlen_train",
         "Genesis/data/overnight_randlen_test", "DS-0002", "DS-0003"]
  seed: 0
  split: >
    Dynamics: the pre-existing file-level split
    `Genesis/data/overnight_randlen_{train,test}` is REUSED, not re-derived
    (invariant `randlen-train-test-file-disjoint` HOLDS). EVERY dynamics
    model in this record is fit on that same 192-file train set -- this is
    the point of the record. Two legacy checkpoints were EXCLUDED for
    failing it: `weights/MODEL-0001` and `weights/MODEL-0002` were fit on a
    legacy 170-file split whose train set contains 17 of the 21 files of
    `overnight_randlen_test` (81% leakage), so the descriptor arm was refit
    clean and the visual arm uses `Baselines/LinearForesight/runs/
    operators_res32.pt` (same recipe, correct split). `weights/MODEL-0003`
    was dropped outright: it is fit on `slates_multistep`, a different
    corpus. Encoders: trained on DS-0002/DS-0003 states, which are grouped
    so a split cannot leak a near-duplicate. DS-0001 is a fixed evaluation
    pool; nothing in this record is fitted on it.
  runtime: pending
  runs: []
  env: "python 3.10.20, torch 2.11.0+cu130 (anaconda3/envs/pme), RTX 4070 Laptop 8 GB"

budget:
  declared: "3 h GPU wall-clock for encoder training + ~1 h scoring"
  spent: pending
  outcome: within

design:
  varied:
    arm: [random, descriptor-94d-switched, visual-switched-res32, nfd-randlen,
          frozen-random-encoder, lejepa-generic, lejepa-goal-loss]
    frame: [world-64, push-32]
    readout: [decoder-to-occupancy, linear-value-head]
    encoder_seed: [0, 1, 2]
    goal: [corner, ring_O, letter_X]
    value_fn: [lyapunov, mass_in_region, signed_mass_in_region]
  held_fixed:
    dynamics_corpus: "overnight_randlen_train, 192 files / 98304 transitions"
    eval_pool: "DS-0001, 20 slates x 1000 candidates, step 0"
    grid: 64
    bounds: "+-0.064 m both axes"
    rasteriser: "transforms.functional.particles_to_occupancy, footprint_radius 1.25 vox"
    latent_dynamics: "closed-form ridge, residual, HARD 6-bin push-length gate (Baselines/LinearForesight bin_index), file-disjoint grouped inner CV for lambda"
    lejepa: "EXP-0020 recipe: align ||p1-p2||^2 + 0.02*SIGReg(p) + 0.02*SIGReg(z), 12 epochs x 250 steps, bs 192, lr 1e-3, AdamW+OneCycle, latent 256, proj 256"
    lambda_sigreg_choice: >
      0.02, NOT swept here. It was swept in experiments/temp/lejepa-debug:
      over 0.02/0.2/1.0 the switched-6 latent R^2 falls monotonically
      (+0.4124/+0.3956/+0.3568) while effective rank rises -- 0.02 is at or
      near this axis's optimum, so it is held fixed rather than re-swept.
    goal_loss_value_fn: "mass_in_region (per direction 2026-09-17; lyapunov is the fallback if it fails outright)"
    goal_library: "regenerated post-28271c09; letter X and the eval goals held out of goal-head training"
  baselines:
    - "random (the ranking floor; `persistence` is a DEGENERATE ranker here -- dv=0 for every candidate makes argmin return row order -- and is reported as a reference row only, never as the floor)"
    - "frozen-random-encoder (the kill-gate: representation learning must beat its own untrained initialisation)"
    - "dz = 0 / persistence-in-latent (the do-nothing baseline for every latent fit)"
  metric: slateN

noise_floor: >
  TO BE MEASURED, not assumed. Three components, all reported separately:
  (1) between-slate sem over the 20 slates within one run (EXP-0016 measured
  0.052-0.091 on this pool); (2) cross-encoder-seed sd over the 3 seeds --
  the component EXP-0016 explicitly left UNMEASURED and the reason 3 seeds
  are run here; (3) wins/losses/TIES per cell, because at K=N the metric is
  deterministic per pool and two decent models very often pick the same
  action, collapsing effective n. `top1_regret` in Lyapunov units is
  reported beside every slateN per METRICS.md, and the K=32 fixed reference
  is reported because slateN is not comparable across pool sizes.

depends_on:
  - randlen-train-test-file-disjoint
  - goal-mask-axis-convention-row-y-col-x
  - occ-rasteriser-consistency
  - slates-binned-uniform-difficulty
  - push-frame-warp-roundtrip

result: pending
verdict: inconclusive
downgrades: [incomplete-design, untested-dependency]
grade: low
---

# EXP-0021 — five-way action-ranking comparison

**Status: pre-registered. Nothing has been measured. `prediction` above was
committed to git before any arm was trained or scored.**

## Why this record exists

`REGISTER.md` C-020 records a model ranking on this pool from EXP-0014 and
explicitly asks for a pre-registered, multi-goal repeat before anything
leans on it. This is that repeat, widened from 1 goal x 1 value function to
9 cells, with two new arms (LeJEPA latent dynamics, with and without the
goal-conditioned value loss of `docs/experimental_design/goal_loss.md`) and
with every dynamics model moved onto one shared corpus and split.

## Why `slateN` and `top1_regret`, not `accuracy`

Per `experiment-log`, `accuracy` compares variants within one model type and
is suspect across types. This record compares five model *types*, two of
which (the descriptor arm, and the latent arms under the value-head readout)
have no image prediction at all, so image `accuracy` is undefined or zero by
construction for them. `slateN` is the verdict metric; `top1_regret` is
reported beside it in Lyapunov units because `slateN`'s ratio hides whether
a captured fraction is a large or trivial absolute gain.

## The readout confound, and how it is controlled

The image-space arms are scored directly: `value(occ1_hat) - value(occ0)`.
The latent arms have no image, so they need a readout, and a latent arm
losing could mean its representation is bad OR that its readout is. Readout
is therefore CROSSED, not fixed: every latent arm is scored both through a
post-hoc decoder to occupancy (identical scoring path to the image arms) and
through a linear value head on `dz = z_hat - z_g`. The goal-loss arm differs
from the generic arm in exactly one bit under the second readout -- whether
that head was trained jointly with the encoder or fitted post-hoc on a
frozen one.

## Known asymmetry between the frames, stated rather than buried

The world-frame latent arms see 64x64 occupancy; the push-frame arms see the
32x32 canonical push frame, matching the project's canonical visual-foresight
resolution. Those two differ in BOTH frame and resolution, so they are not a
clean ablation of each other. The push-frame arm's proper comparator is
`visual-switched-res32`, which sees the identical canonicalised input -- in
the push frame, position and orientation are removed and push length is the
only remaining action variable, which the 6-bin hard gate already carries.
The comparison there is therefore linear-in-latent vs linear-in-pixels on the
same input.

## Unrelated findings

None yet; this record has not been run. Findings that surfaced while setting
it up are filed where they belong rather than here: the goal-mask axis
convention bug and its fix (commit 28271c09, `INVARIANTS.md` tag
`goal-mask-axis-convention-row-y-col-x`, now `fixed`, with
`tests/test_goal_axis_convention.py`); the 81% train/test leak in the legacy
split behind MODEL-0001/MODEL-0002 (affects C-017's comparison, numerically
small: refitting the descriptor operator clean moves `descriptor_accuracy`
0.4350 -> 0.4319); and the SIGReg-paid-off-by-BatchNorm mechanism behind
EXP-0019's memorisation (`experiments/temp/lejepa-debug/RESULTS.md`).

## What would change the verdict

- A cross-encoder-seed sd large enough to swallow P1 or P2. Three seeds is
  the minimum that can measure it at all; if the sd is comparable to the
  effects, the honest answer is UNRESOLVED and more seeds, not a ranking.
- The push-frame arms are run at ONE seed and are exploratory. A push-frame
  result that looks decisive needs its seeds before it is believed.
- `occ-rasteriser-consistency` is `unchecked` and
  `readout-cv-folds-slate-aware` is `broken`. Neither is load-bearing here
  (one rasteriser is used throughout; every ridge lambda is selected with
  grouped folds), but both are cited so a later invalidation finds this
  record.

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- Never run. The project moved to the eval_report harness and closed-loop work (EXP-0026+); DS-0001, its planned corpus, is off-training-physics and 46 % illegal touchdowns (EXP-0065). On hiatus with the latent/embedding line (user, 2026-10-03), not abandoned; if resumed, re-register on a legal, training-physics corpus rather than DS-0001.
