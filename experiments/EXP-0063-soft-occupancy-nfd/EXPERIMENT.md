---
id: EXP-0063
title: >
  Soft-occupancy vs condensed-output NFD (narrow clean domain, pilot, 1 seed/arm): training the
  NFD on Gaussian-blurred occupancy (input + target) raises slateN_tough over the 3-seed baseline
  (sigma 2 px: +0.115 pooled 64 pools, +0.145 at 3 pushes); a condensing loss term (sharpness
  p(1-p)) does not help and at w=0.7 hurts (-0.047 pooled, -0.074 with degenerate cells dropped)
tier: T1
mode: exploratory
date: 2026-10-03
hypothesis: null
claim: >
  On the clean narrow domain (n20 single layer, 20 mm pushes; DS-0015 train, DS-0016 test pools,
  DS-0017 val pools, DS-0018 3-push sequences), an NFD (nfd_3ch_narrow_l20_v2 recipe) trained with
  a Gaussian-blurred occupancy representation (sigma 1-2 px, on input and target, input blurred
  identically at eval) ranks pushes better on slateN_tough than the same recipe trained on the
  hard raster, and the gain is not reproduced by blurring the hard model's output at eval; adding
  the existing `sharpness` (mean p(1-p)) term to the MSE loss does not improve ranking and at
  weight 0.7 degrades it. One training seed per arm.
provenance:
  commit: d72bb304
  dirty: true
  script: "code/eval_pilot.py (new; wraps EXP-0059 code/eval_extended.py::eval_occ_model and
    bootstrap_ci.py unchanged); EXP-0059 code/multistep_eval_v2.py (3-push, --out to this record)"
  data: ["DS-0015 (train, 10,653 rows after the NFD's own 5/5% carve-out)", "DS-0016 (test:
    test_chains_v2, test_chains_v2_clean, test_pools_v2 32 pools)", "DS-0017 (val_pools_v2, 32
    pools, used as a second held-out pool set -- nothing selected on it)", "DS-0018 (3-push
    seqpools)"]
  code_path: "train: Baselines/NFD/train_nfd.py + configs/nfd_3ch_narrow_l20_v2_{soft_s1,
    soft_s2,sharp_w03,sharp_w07}.yaml. eval: simple_mpc.adapters.make_occ_adapter ->
    encode_state(occ_from_particles(s0)) -> predict_step; truth via occ_for_scoring"
  seed: "training --seed 0 for every arm; baseline seeds 0(unseeded)/1/2 from EXP-0059"
  split: "PileSweepData file-hash split, identical to nfd_3ch_narrow_l20_v2"
  runs: [RUN-0001-train-arms, RUN-0002-eval]
budget:
  declared: "~3 h wall-clock (4 x 60-epoch trainings concurrently ~80 min + eval ~30 min)"
  spent: "~2.3 h wall-clock (13:00-15:15; trainings ~85 min concurrently at ~80 s/epoch)"
  outcome: within
design:
  varied:
    arm: "soft_s1 / soft_s2: occupancy_blur sigma 1 / 2 px (2 / 4 mm) on input occupancy AND
      target, eval input blurred identically; sharp_w03 / sharp_w07: loss.sharpness (mean
      p(1-p)) weight 0.3 / 0.7 added to MSE"
  held_fixed:
    recipe: "nfd_3ch_narrow_l20_v2.yaml byte-identical otherwise (UNet [4,8,16], MSE, 60 epochs,
      bs 32, lr 1e-4, aug, best-val-loss checkpoint)"
    seeds: "ONE training seed per arm (pilot); baseline has 3"
    scoring: "slateN truth = occ_for_scoring (soft mass splat) for every model; 13-goal and
      8-goal-tough lyapunov"
  baselines: "persistence (do-nothing); nfd_3ch_narrow_l20_v2 seeds 0/1/2; output-blur control
    nfd_v2_outblur{1,2} (baseline seed 0 with its prediction blurred at eval)"
  metric: "slateN_tough (primary, METRICS.md slateN over the 8 tough goals), slateN (13 goals),
    3-push terminal slateN_tough (DS-0018); accuracy_1 / rollout_accuracy reported but NOT
    comparable for the soft arms (scored against the hard raster)"
noise_floor: "baseline seed spread on DS-0016 test slateN_tough 0.700-0.731 (EXP-0059, sd
  0.016); 32-pool bootstrap CI half-width ~0.08 per model, ~0.04-0.07 paired"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  slateN_tough, paired pool-bootstrap delta vs the per-pool mean of the 3 baseline seeds
  (baseline test 0.731/0.708/0.700, val 0.721/0.700/0.567, 3-push 0.693/0.695/0.633).
  soft_s2: test 0.787 (+0.074 [+0.019,+0.135]), val 0.818 (+0.156 [+0.084,+0.266]), pooled
  +0.115 [+0.064,+0.174], 3-push terminal 0.819 (+0.145 [+0.098,+0.192]). soft_s1: test 0.820
  (+0.107 [+0.064,+0.153]), val 0.639 (one near-degenerate val cell gives capture -40; dropped:
  0.798), pooled +0.041 [-0.152,+0.164] / +0.096 [+0.067,+0.126] degenerate-dropped, 3-push 0.701
  (+0.027 [-0.015,+0.070], unresolved). Output-blur control (seed-0 model, prediction blurred at
  eval): +0.007 [-0.000,+0.015] (sigma 1) / -0.004 [-0.020,+0.011] (sigma 2) vs seed 0 pooled --
  eval-time smoothing does not reproduce the gain. sharp_w03: test 0.712, val 0.666, pooled
  +0.001 [-0.052,+0.069], 3-push 0.653 (-0.021 [-0.054,+0.012]). sharp_w07: test 0.665 (-0.048
  [-0.092,-0.002]), val 0.616, pooled -0.047 [-0.098,+0.019] / -0.074 [-0.108,-0.042]
  degenerate-dropped, 3-push 0.636 (-0.038 [-0.090,+0.007]). The sharpness term works as intended
  (mean p(1-p) 0.004 -> 0.001/0.000; mid-valued pixels 2.0% -> 0.8%/0.3%). accuracy_1 is not
  comparable for the soft arms (scored vs the hard raster): 0.469/0.396 vs 0.557-0.570.
verdict: supported
downgrades: [imprecision, inconsistency]
grade: low
supersedes: []
invalidated_by: null
---

## Plan (written while the arms trained, before any arm was scored)

**Question.** Does the NFD rank pushes better (slateN_tough) on the clean narrow domain when
(A) its state representation is soft occupancy (Gaussian-blurred raster in AND out) or (B) its
loss pushes outputs toward 0/1 (the existing `sharpness` term, mean p(1-p))? A and B pull in
opposite directions on output blur.

**Why `sharpness` is the condensation term.** Per pixel with hit-rate q, MSE + w p(1-p) has
optimum p* = (2q - w) / (2(1 - w)) for w < 1 -- a contrast stretch about 0.5 (w -> 1 is a hard
threshold), where plain MSE gives p* = q. w = 0.3 is mild; w = 0.7 maps q <= 0.35 to 0.

**What would count.** An arm is "better" only if its paired delta vs the per-pool 3-seed mean
excludes 0 on test AND points the same way on val (the pooled 64-pool delta is the summary).
With one seed per arm, a delta inside the seed spread (~0.03) is unresolved. The output-blur
control is what separates "trained on soft occupancy" from "ranks with a smoothed image": if
soft_s1 ~ outblur1, the training change bought nothing beyond eval-time smoothing.

**Embarrassing result to watch for.** The soft arms beating the baseline only because their
input is blurred at eval via a different path than training (train/eval mismatch would
instead show up as a collapse) -- checked by the unit tests in `tests/test_soft_occupancy.py`
and by the smoke eval reproducing EXP-0059's baseline 0.731 exactly.

## What was actually run

- **Plumbing (new, reusable):** `transforms/functional.py::gaussian_blur_occ`;
  `transforms/representation.py::OccupancyBlur` (config type `occupancy_blur`; re-points the
  `current_occupancy`/`target_occupancy` aliases, which the wrapper's default transforms set
  before config transforms -- without that, validation would score against the stale hard
  target); `Baselines/NFD/predictor.py::input_blur_sigma_from_run` (sigma read from the
  checkpoint's own `run_config.yaml`); `OccupancyGradientAdapter.encode_state` (blurs
  raster-built inputs only, never a fed-back prediction), wired into EXP-0059
  `eval_extended.py::_encode`, `multistep_eval.py::run_occ` and `learned_mpc.ModelObjective` (per
  ensemble member; identical behaviour for every existing model). Tests:
  `tests/test_soft_occupancy.py` (6 pass). Four `OCC_ADAPTERS` entries.
- **Checks before the real runs:** 1-epoch smoke trainings of soft_s1 and sharp_w07; smoke eval
  reproduced EXP-0059's baseline exactly (test slateN_tough 0.731, seeds 0.708/0.700; 3-push
  0.693/0.695/0.633). Real-data samples: sigma 1 / 2 lowers a cube's peak to 0.78 / 0.55, raster
  mass 199 -> 201 / 202 (reflect padding at walls).
- **RUN-0001:** 4 trainings concurrently, `--seed 0`, 60 epochs, best-val-loss checkpoints at
  epochs 51 / 53 / 57 / 52 (soft_s1, soft_s2, sharp_w03, sharp_w07). Checkpoints:
  `Baselines/NFD/runs/nfd_3ch_narrow_l20_v2_<arm>/`.
- **RUN-0002:** `code/eval_pilot.py` (1-step test + val, controls, binariness) and EXP-0059
  `multistep_eval_v2.py --out results/multistep.json` (CPU); `code/summarize.py` ->
  `results/summary.md` (full table: every arm, control and baseline).
- **Sensitivity, added after seeing results (stated as such):** dropping (pool, goal) cells
  whose true dv spread is < 5% of the split median. It removes 3 val `quadrant_0` cells (pools 17,
  22, 27; spread <= 1% of median) and no test cell. Because vt does not depend on the model, the
  same cells drop for every model. Headline numbers are the unfiltered ones; the filtered column is
  only there to show that soft_s1's val drop and seed 2's val drop each come from one such cell.
- Dirty tree: uncommitted EXP-0063 plumbing above plus other sessions' unrelated edits.

## Interpretation and limits

- Both soft arms beat the baseline on 1-step test, and sigma 2 is resolved on every pool set
  and at 3 pushes. Its 3-push 0.819 is about the same as EXP-0059's retrieval_k5 (0.821) on the
  same DS-0018 pools, which no single NFD seed had reached (0.633-0.695). That is a cross-record
  comparison on the same harness and pools, not a paired test, and it rests on one seed.
- **Why soft helps is not isolated.** Untested hypothesis: the hard training raster aliases
  (a particle covers 4-6 px depending on its sub-pixel position, invariant
  `score-occupancy-subpixel-stable` / EXP-0027), so the hard MSE target carries position noise
  that blurring averages out. A discriminating test: train on a mass-splat target
  (`splat_particles_mass`) with no extra blur, and compare it with soft_s1 / soft_s2.
- **sharpness** was weighted 0.3 and 0.7, fixed before any run. Its harm grows with the weight,
  which fits the analysis in the plan: forcing pixels with 0.35 < q < 0.65 toward 0/1 throws away
  the probability mass that the Lyapunov value integrates. So "condensed outputs" and "soft
  occupancy" are not two sides of one dial; sharpening was neutral to harmful and softening helped.
- One seed per arm. The baseline's seed spread (sd ~0.016 test) is smaller than the soft_s2
  effect, but arm seed noise was not measured.

## Unrelated findings

- **slateN capture is unbounded below on near-tie pools.** EXP-0059's harness skips a (pool,
  goal) cell only if its true dv spread is < 1e-9. DS-0017 val has 3 `quadrant_0` cells at
  0-1% of the median spread, where one wrong pick scores -1.3 to -40 and moves a model's 32-pool
  val mean by up to ~0.16. This already explains EXP-0059's seed-2 val value (0.567; 0.713 with
  those cells dropped). It affects every val_pools_v2 number. A relative degeneracy threshold, or
  reporting per-cell medians, would fix it; not changed here.
