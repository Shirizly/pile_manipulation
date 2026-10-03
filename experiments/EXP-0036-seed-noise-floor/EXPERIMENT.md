---
id: EXP-0036
title: >
  Training seed alone moves NFD slateN by an sd of 0.01-0.04 (4-5 of 6 seed pairs
  significantly different), about the size of most model gaps in the register;
  accuracy is 3-10x steadier across seeds; averaging 4 seeds gains +0.06-0.07, and
  4 different architectures gain more on lyapunov
tier: T1
mode: confirmatory              # P1, P2 pre-registered in DESIGN.md before training; P2's like-for-like re-scoring added before it ran (addendum)
date: 2026-09-24
hypothesis: null
claim: >
  For the world-frame NFD recipe (nfd_train_3ch_randlen.yaml, 4 training seeds; soft
  truth, 30 goals): (P1) the seed sd of lyapunov slateN on randlen_test is >= 0.005;
  (P2) the 4-seed ensemble beats the mean single seed (state-bootstrap CI > 0) on
  DS-0007, but gains less than an equally sized ensemble of different architectures.
prediction:
  supports: "P1 sd >= 0.005; P2 seed-ens - mean seed CI > 0 AND seed_ens4 < arch_ens4"
  refutes: "P1 sd < 0.002; P2 CI includes 0, or seed_ens4 >= arch_ens4"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/train_seeds.sh, code/score_seeds.py, code/accuracy_seeds.sh, code/ensemble_compare.py
  data: ["overnight_randlen (training)", "DS-0006", "DS-0007 scattered_n20, scattered_n50", "randlen_test (eval_report)"]
  code_path: >
    train_nfd.py --seed k; slateN: OCC_ADAPTERS predict_step (CPU) -> per-goal
    lyapunov / mass_in_region -> whole-pool capture vs soft truth (EXP-0030/0035
    truth files); accuracy + randlen_test slateN: eval_report nfd_randlen spec with
    --ckpt override, CPU, --goal-set many, soft truth
  seed: "training seeds 1, 2, 3 (+ the original unseeded run as seed 0); state bootstrap rng 0, 5000"
  split: "not applicable (whole pools)"
  data_commit: not applicable
  runs: [RUN-0001-train, RUN-0002-score, RUN-0003-accuracy]
  runtime: "training 3 x ~2.1 h GPU; scoring ~3 min CPU; accuracy ~6 min CPU"
budget:
  declared: "~6 h GPU training + minutes of scoring (DESIGN.md)"
  spent: "~6.4 h GPU + ~15 min CPU"
  outcome: within
design:
  varied:
    training_seed: "0 (original nfd_3ch_randlen), 1, 2, 3"
  held_fixed:
    recipe: "Baselines/NFD/configs/nfd_train_3ch_randlen.yaml except output.log_dir; unet_best.pth"
    goals: "26 letters + 4 quadrants; lyapunov and mass_in_region"
    truth_scoring: soft
  baselines: "random pick (capture 0); each single seed vs the seed ensemble; architecture ensembles (arch_ens4, ensemble_nfd5, ensemble_nfd) on the same pools"
  metric: slateN; accuracy
noise_floor: "this record IS the training-seed floor; state-bootstrap 95% CIs for paired differences"
depends_on: [score-occupancy-subpixel-stable, occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  P1 supported: randlen_test lyapunov slateN over 4 seeds 0.894 / 0.915 / 0.909 /
  0.906, sd 0.009 (range 0.021); accuracy 0.456-0.464, sd 0.003. On DS-0006 / DS-0007
  n20 / n50 the lyapunov slateN seed sd is 0.019 / 0.037 / 0.022 (mass 0.025 / 0.009 /
  0.010), with 3-5 of 6 seed pairs CI-separated. P2: first half supported (seed
  ensemble - mean seed +0.057 to +0.074, every CI > 0; - best seed +0.017 to +0.056);
  second half supported on lyapunov only (seed_ens4 - arch_ens4 -0.012 / -0.048 /
  -0.019, CIs < 0) and not on mass_in_region (-0.009 / -0.002 / +0.005).
verdict: supported              # P1 holds; P2 holds on lyapunov, not on mass_in_region -- see body
downgrades: [imprecision, inconsistency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
If seeds were near-identical (sd < 0.002), every single-run model comparison in the
register would stand as measured; if they differ by ~0.01-0.04, many of them are
within training noise. The same 4 seeds also separate "any averaging helps" from
"averaging DIFFERENT models helps" at a matched member count.

## What was actually run
- RUN-0001: 3 trainings, sequential, `train_nfd.py --seed {1,2,3}` (configs/).
- RUN-0002 (`code/score_seeds.py`): seeds 1-3 predicted on CPU through analyse.py's code
  path; seed 0 reused from the EXP-0030 / EXP-0035 cached predictions. Check: seed 0
  recomputed on CPU for 3 states matches the cache to 5e-5 on lyapunov; on
  mass_in_region (values ~170x larger, sd 2.6) to 0.018 absolute, with no pick changed.
  That is float rounding between GPU and CPU.
- RUN-0003 (`code/accuracy_seeds.sh`): the first attempt scored the ORIGINAL checkpoint
  four times. Setting NFD_CKPT has no effect because eval_report writes spec['ckpt']
  into it before building. Those outputs were moved to results/randlen_test_wrong_ckpt/.
  An explicit `--ckpt MODEL=PATH` flag was added to eval_report and the run redone.
- P2's like-for-like re-scoring (DESIGN.md addendum): written after the per-seed
  numbers were seen, but before `ensemble_compare.py` ran. The pre-registered
  reference (EXP-0035's 7-model gain) was measured on half pools, so it was not
  comparable with whole-pool seed scores.

## Numbers
Per-seed slateN (lyapunov), seeds 0 / 1 / 2 / 3, and the seed ensemble:

| corpus | seed 0 | seed 1 | seed 2 | seed 3 | sd | pairs CI != 0 | seed_ens4 |
|---|---|---|---|---|---|---|---|
| randlen_test (21 slates) | 0.894 | 0.915 | 0.909 | 0.906 | 0.009 | (not computed) | -- |
| DS-0006 (160 x 128) | 0.746 | 0.787 | 0.781 | 0.758 | 0.019 | 4/6 | 0.838 |
| DS-0007 n20 | 0.760 | 0.838 | 0.758 | 0.785 | 0.037 | 5/6 | 0.855 |
| DS-0007 n50 | 0.860 | 0.894 | 0.846 | 0.851 | 0.022 | 4/6 | 0.921 |

mass_in_region seed sd: DS-0006 0.025, n20 0.009, n50 0.010.
Accuracy (randlen_test): 0.4564 / 0.4615 / 0.4622 / 0.4642, sd 0.0033, range 0.0078.

Ensembles on the same whole pools (lyapunov; results/ensemble_compare.json):

| corpus | best single architecture | seed_ens4 | arch_ens4 | ensemble_nfd5 | ensemble_nfd (7) | seed4 - arch4 |
|---|---|---|---|---|---|---|
| DS-0006 | 0.810 (residual_warped) | 0.838 | 0.850 | 0.846 | 0.837 | -0.012 [-0.022, -0.002] |
| DS-0007 n20 | 0.867 (residual_warped) | 0.855 | 0.903 | 0.902 | 0.903 | -0.048 [-0.056, -0.040] |
| DS-0007 n50 | 0.914 (residual_warped) | 0.921 | 0.941 | 0.941 | 0.931 | -0.019 [-0.024, -0.015] |

mass_in_region seed4 - arch4: -0.009 [-0.017, -0.002], -0.002 [-0.009, +0.004],
+0.005 [0.000, +0.009].

## Reading
- **Seed noise is the size of the model gaps the register reads.** On DS-0006 the
  original nfd_3ch_randlen is the WORST of its 4 seeds (0.746, versus 0.787 for the
  best seed). Every "architecture X beats nfd_3ch_randlen by 0.02-0.04 slateN" from
  one run per model is within this range. Example: residual_worldframe's 0.761 sits
  inside the seed range 0.746-0.787.
- **Accuracy is much steadier across seeds.** Seed sd 0.003 against between-model
  gaps of 0.02-0.10. residual_worldframe's accuracy (0.484) is about 6 seed-sd
  above the best seed (0.464), so accuracy does separate it from the seed spread.
  slateN does not. EXP-0039 found this model is really better in closed loop.
- **Averaging helps even when the members are all seeds of one recipe** (+0.06-0.07
  over the mean seed). On lyapunov, different architectures add another 0.01-0.05
  at the same member count. At a matched TIME budget, any ensemble still loses
  (EXP-0030 A5).
- The seed floor from 4 runs is itself imprecise (an sd from n = 4).

## What would change the verdict
More seeds (sd from 4 runs has wide error); seeds of a second architecture (the
residual-warped arm, 5.4 h per run, is not seeded); closed-loop seed gaps for
seeds 2-3.

## Threats
- One architecture only.
- Seed 0 was trained unseeded, earlier, possibly with a different code state (train_nfd.py gained `--seed` for this record).
- n20/n50 scatter and randlen_test only.

## Unrelated findings
eval_report ignored NFD_CKPT (and every `ckpt_env` override), because `_load_predictor`
writes spec['ckpt'] into the variable before building. No command in COMMANDS.jsonl used
that route (Baselines/common/eval_randlen_indist.py documents NFD_CKPT/GNN_CKPT for its own
loader, which was not checked). Now fixed with `--ckpt MODEL=PATH`
(the env route still does not work and says so in `--help`).

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- DS-0006 rows: EXP-0065 / ISS-013: 54 % of DS-0006's candidate pushes put the blade on a cube at touchdown (pre-fix pile-aware sampler); this record's pool numbers were not re-scored on legal-only candidates.
