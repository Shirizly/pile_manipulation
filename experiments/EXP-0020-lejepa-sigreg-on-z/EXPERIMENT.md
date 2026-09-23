---
# ---- identity -------------------------------------------------------------
id: EXP-0020
title: >
  Adding SIGReg directly to z (EXP-0019's unrun `--sigreg-on-z`) does NOT stop
  the effective-rank collapse of z -- rank still ends at 7.3 of 256 vs 7.5 for
  the broken run and 44.6 at init -- yet it transforms latent R^2, which rises
  from +0.033/+0.060 to +0.292 global / +0.412 hard-6-bin switched, clearing
  the EXP-0016 frozen-random-encoder floor (+0.209/+0.269) and closing the
  train-test gap from 0.30 to 0.05
tier: T1
mode: exploratory
date: 2026-09-15
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  Applying SIGReg to the latent `z` in addition to the projection `p`
  (loss = ||p1-p2||^2 + 0.02*SIGReg(p) + 0.02*SIGReg(z), everything else held
  byte-identical to EXP-0019 RUN-0001: same 252 files =
  `overnight_randlen_train` + 60 strided Sean files, same 12 epochs x 250
  steps, batch 192, lr 1e-3, AdamW+OneCycle, latent_dim 256, proj_dim 256,
  seed 0, same `ResCNNEncoder(res=64, latent=256, n_res_blocks=2)`):
  (1) DOES regularise z -- the SIGReg statistic measured on z falls 499 -> 244
  where in EXP-0019 it stayed flat at 499 -> 479 -- and drives per-dimension
  std of z from 0.045 to 0.892 (min 0.393), i.e. onto the isotropic unit
  scale SIGReg targets;
  (2) does NOT prevent the effective-rank collapse of z: effective rank on the
  same 8192-state training monitor falls 44.6 -> 3.1 by epoch 0 (WORSE than
  EXP-0019's 10.2), then partially recovers to 7.3 at epoch 11, essentially
  indistinguishable from EXP-0019's 7.5 -- so the design doc's SIGReg
  PLACEMENT is NOT the cause of the rank collapse;
  (3) nevertheless raises latent R^2 on the file-disjoint
  `overnight_randlen_test` split, under EXP-0019's `fit_switched_hard.py`
  run unchanged, from +0.0325 -> +0.2923 (single global operator) and
  +0.0597 -> +0.4126 (HARD 6 equal-width push-length bins), clearing the
  EXP-0016 frozen-random-encoder floor of +0.2087/+0.2693 measured under the
  identical fit code, and shrinking the train-test latent-R^2 gap from 0.30
  to 0.05;
  (4) hard push-length gating still beats the single global operator
  (+0.4126 vs +0.2925), and does so in every one of the 6 bins.
  Scoped to: this encoder architecture and seed 0, sigreg_on_z weight 0.02
  (unswept), 12 epochs x 250 steps at batch 192, this view set, closed-form
  ridge latent dynamics, `overnight_randlen_{train,test}`. No `slateN` and no
  `accuracy` were measured.

prediction: null

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0020-lejepa-sigreg-on-z/code/
  data: ["Genesis/data/overnight_randlen_train", "Genesis/data/overnight_randlen_test",
         "Genesis/data/Sean (60 of 1230 *_data.pt files, strided -- the SAME 60 as EXP-0019)"]
  code_path: >
    experiments/EXP-0019-lejepa-encoder-pushlen-switched/code/ was executed
    UNCHANGED for training (`train_encoder.py`, only the `--sigreg-on-z 0.02`
    flag added), encoding (`encode_states.py`) and the headline fit
    (`fit_switched_hard.py`), so every number is apples-to-apples with
    EXP-0019. The only new code is
    experiments/EXP-0020-lejepa-sigreg-on-z/code/fit_switched_grouped_lam.py,
    which imports fit/predict/metric from `fit_switched_hard.py` and replaces
    ONLY the inner ridge-lambda validation split (see Threats).
  seed: 0
  split: >
    The pre-existing file-level split `Genesis/data/overnight_randlen_{train,
    test}` was REUSED, not re-derived (192 train files / 98304 transitions;
    21 test files / 10752 transitions). The encoder saw only train files plus
    Sean states. Ridge-lambda selection was done BOTH ways: EXP-0019's
    row-random 20% inner split (RUN-0003, for comparability) and a new
    FILE-DISJOINT grouped 5-fold inner CV over the 192 train files (RUN-0004,
    group id = row // 512, asserted against the file count).
  runtime: "~30 min GPU (encoder 1406 s, encode 84 s, three fits ~2 min), RTX 4070 Laptop 8 GB, peak 5811 MiB"
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004]
  env: "torch 2.11.0+cu130 (anaconda3/envs/pme), CUDA, RTX 4070 Laptop 8 GB"

budget:
  declared: "70 min wall-clock / 150k tokens"
  spent: "~60 min / ~100k tokens"
  outcome: within

design:
  varied: {sigreg_placement: ["p only (EXP-0019)", "p AND z, weight 0.02 (this record)"],
           gate: ["none (single global operator)", "HARD, 6 equal-width push-length bins"],
           ridge_lambda: [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5],
           lambda_selection: ["row-random inner split (EXP-0019 method)", "file-disjoint grouped 5-fold inner CV"],
           centring: ["z - mean(z_train)", "raw z"],
           encoder: ["EXP-0020 SIGReg-on-z", "EXP-0019 SIGReg-on-p-only", "EXP-0016 frozen random"]}
  held_fixed: {encoder_arch: "ResCNNEncoder(res=64, latent=256, n_res_blocks=2), 3593792 params",
               projector: "256-1024-256, BatchNorm+SiLU, 527616 params",
               lejepa: "alignment ||p1-p2||^2 + 0.02*SIGReg(p); this run ADDS 0.02*SIGReg(z)",
               views: "particle dropout keep~U(0.80,1.0) + footprint-radius jitter x U(0.85,1.15) + additive occupancy noise std 0.02 clamped to [0,1]; NO rotations, flips or crops",
               optimiser: "AdamW, OneCycle, max_lr 1e-3, wd 1e-4, bs 192, 12 epochs x 250 steps, seed 0",
               action_encoding: "[x_s, y_s, sin(theta), cos(theta), dx, dy] (EXP-0016 encode_action)",
               dynamics: "closed-form ridge (toward zero) of dz on [z, a, 1]; residual z' = z + (A_b z + B_b a + c_b)",
               bin_scheme: "Baselines/LinearForesight bin_index, 6 equal-width bins over [0, 80.00] mm, MIN_ROWS_PER_BIN 50",
               grid: 64,
               rasteriser: "EXP-0019 code/raster.py batch-vectorised disk splat, footprint radius 1.25 vox, bit-exact vs transforms.functional.particles_to_occupancy"}
  baselines: ["dz = 0 (do nothing / persistence-in-latent, 0 by construction)",
              "global single operator (one bin)",
              "EXP-0016 frozen-random-encoder floor, identical fit code",
              "EXP-0019 SIGReg-on-p-only encoder, identical fit code"]
  metric: >
    latent_r2 = 1 - mse(dz_pred - dz_true)/mse(dz_true), i.e. skill against a
    `Delta z = 0` persistence-in-latent baseline IN THE SAME LATENT SPACE
    (0 = no better than assuming the push did nothing). Same definition and
    same code as EXP-0016 and EXP-0019. LATENT R^2 IS STRICTLY COMPARABLE
    ONLY WITHIN ONE ENCODER -- a different encoder moves both the regression
    target dz and the denominator mse(dz_true). The three-way cross-encoder
    table below is therefore corroborated by the effective-rank and
    train-test-gap measurements, and is NOT asserted on latent R^2 alone.
    Secondary quantities, all named explicitly wherever used: effective rank
    of z (exp of the entropy of the centred singular-value power spectrum),
    per-dimension std of z, ||mean(z)||, centred rms of z, and the SIGReg
    Epps-Pulley statistic evaluated on z. `slateN` and `accuracy` were OUT OF
    SCOPE (no value readout / decoder was run).

noise_floor: >
  NOT CHARACTERISED. One encoder seed (0) and one deterministic closed-form
  ridge fit; there is no seed spread to quote for either the effective rank
  or latent R^2. The only spread available is the ridge-lambda sweep, over
  which latent R^2 is nearly flat (switched test +0.4055 to +0.4126 across
  lambda 0.01-100), so the headline latent-R^2 numbers are not a lambda
  artifact. The rank comparison 7.3 (this run) vs 7.5 (EXP-0019) is WELL
  INSIDE any plausible seed variation and is reported as UNRESOLVED/no
  difference, not as an improvement. The latent-R^2 gaps (+0.41 vs +0.06 vs
  +0.27) are far larger than the lambda spread but still rest on one seed.

depends_on: [randlen-train-test-file-disjoint, occ-rasteriser-consistency,
             hybrid-latent-stage2-anticollapse,
             readout-cv-folds-slate-aware]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  ENCODER (12 epochs, SIGReg on p AND z at weight 0.02 each): SIGReg statistic
  ON z fell 499 -> 244 (EXP-0019: 499 -> 479 flat), per-dimension std of z rose
  0.045 -> 0.892 (min 0.393, EXP-0019: 0.162/0.058), centred rms 0.046 ->
  0.923, ||mean(z)|| 5.12 -> 5.91. EFFECTIVE RANK of z: 44.6 at init -> 3.1 at
  epoch 0 -> 7.3 at epoch 11 (EXP-0019: 44.6 -> 10.2 -> 7.5). THE RANK COLLAPSE
  WAS NOT PREVENTED. Training alignment fell 0.089 -> 0.046 (EXP-0019 reached
  0.0053, i.e. alignment is less well minimised here). LATENT R^2 (test,
  file-disjoint `overnight_randlen_test`, vs dz=0), EXP-0019
  `fit_switched_hard.py` run unchanged: dz=0 +0.0000 by construction; global
  single operator +0.2923 (train +0.3338, lambda 1); HARD 6-bin push-length
  switched +0.4126 (train +0.4648, lambda 10). With FILE-DISJOINT grouped
  5-fold inner CV instead (RUN-0004), both lambdas select 100 and the held-out
  latent R^2 is +0.2925 global / +0.4124 switched -- indistinguishable.
  Per-bin latent R^2 on test (bin, mm, n_train/n_test): 0 [0,13.3] 7428/811
  +0.0777; 1 [13.3,26.7] 18330/2042 +0.2934; 2 [26.7,40.0] 24925/2769 +0.3062;
  3 [40.0,53.3] 22275/2413 +0.3722; 4 [53.3,66.7] 18680/1958 +0.4763;
  5 [66.7,80.0] 6666/759 +0.5288. No bin starved; none fell back to the global
  operator. Effective rank of z measured on the full 98304-row train corpus
  (a different population from the 8192-state training monitor) is 18.6 here
  vs 41.7 for EXP-0019 -- so on THAT population this encoder has LOWER rank
  while scoring 7x the latent R^2, which is reported as an anomaly, not
  explained.
verdict: refuted
downgrades: [imprecision, incomplete-design, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0019 diagnosed the rank collapse of `z` as caused by SIGReg being applied
to the projection `p` rather than to `z`, on the evidence that SIGReg(p) fell
13.4 -> 1.51 while the same statistic probed on `z` stayed flat. The
discriminating test is to put SIGReg on `z` too and see whether the rank
collapse stops. It did not. The diagnosis is therefore refuted as stated: the
placement is not what lets `z` lose rank.

The test was well-posed because the intervention demonstrably bites — the
SIGReg statistic on `z` actually falls (499 -> 244) and the per-dimension std
of `z` moves onto the unit scale SIGReg targets — so "the flag did nothing" is
ruled out by the very statistic the diagnosis was built on.

## The mechanism, verified in code (not assumed)

`experiments/EXP-0019-*/code/train_encoder.py` lines around the training step:

```python
sg = sig(torch.stack([p1, p2], dim=0))          # (T=2, B, D)
loss = align + args.lambda_sigreg * sg
if args.sigreg_on_z > 0:
    sgz = sig(torch.stack([z[:args.bs], z[args.bs:]], dim=0))
    loss = loss + args.sigreg_on_z * sgz
```

`--sigreg-on-z` does exactly what its name says: it adds a second SIGReg term
computed on the *pre-projector* latent, with its own weight, inside the
autograd graph (no `detach`). `le-wm/module.py::SIGReg` holds no learnable
parameters — only the `t`/`phi`/`weights` buffers — and draws fresh random
projection directions on every forward, so reusing one `SIGReg` instance for
both `p` and `z` is sound and the two terms are independent draws.

## What was actually run

- **RUN-0001** — `train_encoder.py --epochs 12 --steps-per-epoch 250 --bs 192
  --lambda-sigreg 0.02 --sigreg-on-z 0.02 --sean-files 60 --seed 0`, 214688
  training states from 252 files. The at-init monitor row is byte-identical to
  EXP-0019's (effective rank of z 44.59671, per-dim std 0.045309), confirming
  the two runs start from the same encoder weights and the same data.
- **RUN-0002** — `encode_states.py`, unchanged, 98304 train + 10752 test
  transitions, 84 s.
- **RUN-0003** — `fit_switched_hard.py`, unchanged (row-random inner split, as
  EXP-0019 used), for the apples-to-apples headline.
- **RUN-0004** — `code/fit_switched_grouped_lam.py` (new), file-disjoint
  grouped 5-fold inner CV for the ridge lambda, plus EXP-0019's
  `diag_lam_and_encoder.py` unchanged for the full lambda table against
  EXP-0016's frozen random latents.

## Per-epoch curves (same fields as EXP-0019, so they overlay)

`eff_rank` and `perdim_std` are measured on an 8192-state monitor sample;
`sigreg(z)` is the Epps-Pulley statistic evaluated on `z` (in EXP-0019 it was
a no-gradient probe, here it is also a loss term).

| epoch | align | sigreg(p) | perdim_std mean / min | eff_rank of z | \|\|mean z\|\| | centred_rms | sigreg(z) |
|---|---|---|---|---|---|---|---|
| -1 (init) | - | - | 0.0453 / 0.0327 | **44.60** | 5.125 | 0.0459 | 499 |
| 0 | 0.0890 | 18.46 | 0.5668 / 0.146 | **3.1** | 11.440 | 0.6084 | 541 |
| 1 | 0.0942 | 9.140 | 0.7289 / 0.237 | 4.7 | 7.545 | 0.7639 | 298 |
| 2 | 0.0962 | 6.267 | 0.8243 / 0.272 | 5.1 | 7.963 | 0.8632 | 294 |
| 3 | 0.0881 | 5.222 | 0.9182 / 0.245 | 5.2 | 7.507 | 0.9621 | 255 |
| 4 | 0.0877 | 4.692 | 0.9050 / 0.328 | 5.9 | 6.156 | 0.9432 | 199 |
| 5 | 0.0782 | 4.193 | 0.9393 / 0.372 | 6.1 | 6.382 | 0.9764 | 223 |
| 6 | 0.0710 | 3.859 | 0.9455 / 0.401 | 6.4 | 6.462 | 0.9808 | 216 |
| 7 | 0.0639 | 3.542 | 0.9342 / 0.397 | 6.7 | 6.662 | 0.9690 | 248 |
| 8 | 0.0575 | 3.308 | 0.8933 / 0.407 | 7.3 | 5.988 | 0.9239 | 240 |
| 9 | 0.0522 | 3.167 | 0.8900 / 0.396 | 7.2 | 6.068 | 0.9213 | 252 |
| 10 | 0.0481 | 3.053 | 0.8955 / 0.395 | 7.3 | 6.055 | 0.9267 | 242 |
| 11 | 0.0461 | 3.017 | 0.8924 / 0.393 | **7.3** | 5.912 | 0.9231 | 244 |

EXP-0019's effective rank of z for comparison: 44.6, 10.2, 13.9, 12.2, 10.1,
8.5, 8.1, 7.9, 7.8, 7.5, 7.4, 7.5, 7.5. Both runs end at ~7.3-7.5 of 256.
Adding SIGReg on `z` makes the epoch-0 rank drop **sharper** (3.1 vs 10.2),
then the rank climbs back monotonically; EXP-0019's climbed briefly then fell.

## Latent R^2: the three-way comparison

All three rows use the SAME closed-form hard-gate fit code
(`EXP-0019-*/code/fit_switched_hard.py` / `diag_lam_and_encoder.py`), the same
6-bin push-length scheme, the same file-disjoint
`overnight_randlen_{train,test}` split, at the best-on-test lambda. **Latent
R^2 is strictly comparable only WITHIN one encoder**; this table is reported
because the fit pipeline is held fixed, and it is corroborated by the
train-test gap (an encoder-internal quantity) rather than resting on the
cross-encoder latent-R^2 levels alone.

| encoder | eff. rank of z (monitor, final) | latent R^2 global test | latent R^2 switched test | train-test gap (switched) |
|---|---|---|---|---|
| EXP-0016 frozen RANDOM (never trained) | 44.6 (= init) | **+0.2087** | **+0.2693** | 0.045 |
| EXP-0019 LeJEPA, SIGReg on `p` only | 7.5 | +0.0497 | +0.0703 | 0.301 |
| **EXP-0020 LeJEPA, SIGReg on `p` AND `z`** | 7.3 | **+0.2925** | **+0.4126** | 0.052 |

At EXP-0019's own selected lambdas (row-random inner split) the EXP-0019 row
reads +0.0325 / +0.0597. The bar set for this record was "latent R^2 must beat
+0.209 global / +0.269 switched": **cleared**, +0.2925 / +0.4126. The other
half of the bar, "effective rank must stay well above 7.5": **not cleared**,
7.3.

## Ridge-lambda selection, and the leak EXP-0019 found

EXP-0019 found its row-random inner validation read +0.359 where the true
held-out latent R^2 was +0.060 — a 0.30 optimism gap — and selected lambda=1
where 10-100 was better. Both selections were run here:

| selection method | lambda global | lambda switched | inner score (switched) | held-out latent R^2 (switched) | optimism |
|---|---|---|---|---|---|
| row-random 20% inner split (EXP-0019 method) | 1 | 10 | +0.4590 | +0.4126 | 0.046 |
| **file-disjoint grouped 5-fold inner CV** | 100 | 100 | +0.4504 | +0.4124 | 0.038 |

The grouped CV is the method to trust (`fit_switched_grouped_lam.py`: fold of
row *i* = `(i // 512) % 5`, with `512 = 98304/192` asserted against the file
count, so no file spans two folds). Here it changes the held-out latent R^2 by
0.0002, because this encoder's representation generalises across files — the
leak was large in EXP-0019 precisely *because* that encoder did not. Grouped
CV does still select a 10-100x larger lambda, matching EXP-0019's finding
about which lambda is right.

## Threats and caveats

- **One encoder seed.** Nothing here has a seed uncertainty. 7.3 vs 7.5
  effective rank is UNRESOLVED, not an improvement.
- **The `sigreg_on_z` weight 0.02 is unswept**, chosen only to match
  `lambda_sigreg` as the brief required. Because SIGReg(z) starts at ~499
  against SIGReg(p)'s ~13, the z term dominates the loss for the first
  epochs (0.02 x 499 = 10 vs alignment 0.089), which is visible as the
  epoch-0 rank crash to 3.1 and as the alignment loss ending 9x higher than
  EXP-0019's (0.046 vs 0.0053). A smaller weight or a warm-up is untested.
- **Two effective-rank populations disagree.** The 8192-state training monitor
  (train files + Sean states) gives 7.3 here / 7.5 for EXP-0019; the 98304-row
  `overnight_randlen_train` z0 cache gives 18.6 here / 41.7 for EXP-0019. So
  by the corpus measure this encoder has *lower* rank while predicting far
  better. ANOMALY, CAUSE NOT ISOLATED. The two differ in sample size, in
  corpus (the monitor includes Sean's n50/n100 states) and in which states
  (both endpoints vs z0 only); which of those drives the reversal is untested.
  It is a reason not to lean on effective rank as the single summary of
  representation quality.
- **No `slateN`, no `accuracy`.** Neither a value readout nor a decoder was
  run, so this record says nothing about control utility. Per the
  `experiment-log` metric rule, a model comparison carrying no `slateN` is not
  finished; the cross-encoder ranking here must not be read as a control
  result.
- **Cross-encoder latent R^2 is not a clean ranking** (see `design.metric`).

## What would change the verdict

1. **A `sigreg_on_z` weight sweep** (0.002, 0.02, 0.2) with the alignment loss
   tracked — the cheapest test of whether the epoch-0 rank crash and the
   9x-worse alignment are a weight-balance artifact. Cost: ~23 min per point.
2. **Regularise `z` only, with no projector at all** — removes the
   two-objective balance entirely and is the cleanest test of whether the
   projector contributes anything. Cost: ~25 min.
3. **A second encoder seed for each of the two placements** — the only way to
   put an uncertainty on either the rank or the latent-R^2 numbers. Cost: ~50 min.
4. **Resolve the rank-population anomaly** — recompute effective rank of z on
   matched populations (same rows, same corpus, same sample size) for both
   encoders. Cost: ~5 min, and it should be done before anyone cites either
   rank number again.
5. **`slateN`** through a value readout, which is the metric that decides.

## Unrelated findings

- **EXP-0019's `fit_switched_hard.py` prints "the slate-aware-fold trap
  applies to DS-0001, not this corpus" in its own docstring, justifying the
  row-random inner split.** That justification is wrong for the reason
  EXP-0019 itself later measured: rows within one `overnight_randlen`
  `_data.pt` file share a spawn configuration and a simulation batch, so they
  are not independent *across files* even though they are independent
  transitions within one. The optimism is small when the encoder generalises
  (0.038 here) and large when it does not (0.30 in EXP-0019), which makes the
  row-random split a metric whose bias depends on the thing being measured.
  `fit_switched_grouped_lam.py` is the drop-in fix; `fit_switched_hard.py`
  itself was left UNCHANGED so EXP-0019's numbers stay reproducible.
- **`train_encoder.py` writes a full checkpoint every epoch** (encoder +
  projector state dicts + config + all curves) to the same path, so an
  interrupted run always leaves a loadable checkpoint at the last completed
  epoch. Worth knowing before anyone adds a "resume" feature that does not
  exist.
- **The `sigreg_of_z` field in the per-epoch monitor is computed on only the
  first 1024 of the 8192 monitor states** (`sigreg_probe(z[:1024][None])`),
  while `effective_rank` and the std fields use all 8192. The fields in one
  curve row therefore describe different sample sizes. Not a bug for the
  trend, but it is why `sigreg(z)` is the noisiest column in the table above
  (it bounces 199-255 over epochs 4-11 with no trend).

## Uncommitted state at run time (`dirty: true`)

The tree was dirty at commit `6ea03278`. None of the code this record
executed was modified: `experiments/EXP-0019-*/code/` and
`experiments/EXP-0016-*/code/` are clean and committed, as is
`le-wm/module.py` and `Baselines/LinearForesight/model.py`. The
uncommitted paths were:

- `.claude/skills/experiment-log/SKILL.md`
- `.claude/skills/project-overview/SKILL.md`
- `Genesis/sandbox_manipulation_clean.py`
- `docs/piled_collection.md`
- `experiments/COMMANDS.jsonl`
- `experiments/INVARIANTS.md`
- `experiments/METRICS.md`
- `experiments/REGISTER.md`
- `experiments/TEMP_LOG.md`
- `le-wm`
- `scripts/probes/pool_common.py`
- `scripts/probes/pool_inspect.py`
- `scripts/probes/pool_survey.py`
- `weights/MODEL-0001-stage2-visual-switched/tests.md`
- `weights/MODEL-0002-descriptor-only-D-all-local/tests.md`
- `weights/MODEL-0003-nfd-multistep-finetuned/tests.md`
- `.claude/orchestrator-notes.md`
- `.claude/skills/data-collection/`
- `.claude/skills/subagent-experimenter/`
- `Baselines/common/goal_configs.py`
- `Genesis/binned_slate_collection.py`
- `Genesis/binned_slate_dataset.py`
- `Genesis/data/Sean/`
- `Genesis/data/slates_binned/`
- `Genesis/data/slates_binned_collect.log`
- `datasets/`
- `docs/CODEMAP.md`
- `docs/experimental_design/jepa_based_encoder.md`
- `experiments/EXP-0011-descriptor-model-comparisons/`
- `experiments/EXP-0012-slates-binned-corpus-difficulty/`
- `experiments/EXP-0013-res32-vs-64-visual-operator/`
- `experiments/EXP-0014-binned-pool-model-ranking/`
- `experiments/EXP-0015-descriptor-value-readout/`
- `experiments/EXP-0016-lejepa-random-encoder-floor/`
- `experiments/EXP-0017-value-readout-instrument/`
- `experiments/EXP-0018-value-readout-diverse-states/`
- `experiments/EXP-0019-lejepa-encoder-pushlen-switched/`
- `experiments/EXP-0020-lejepa-sigreg-on-z/`
- `scripts/probes/binned_pool_cache.py`

plus this experiment's own new directory `experiments/EXP-0020-lejepa-sigreg-on-z/`.

## Artifacts

- `artifacts/RUN-0001/encoder_lejepa.pt` — trained encoder `state_dict` +
  resolved `config` + projector + `train_args` (the resolved config including
  `sigreg_on_z: 0.02`) + all 13 curve rows. **The reusable deliverable.**
- `artifacts/RUN-0001/train_curves.json` — per-epoch curves + args + peak MiB.
- `artifacts/RUN-0002/latents_{train,test}.pt` — latent cache (z0, z1, a,
  length_m, encoder config, ckpt path).
- `artifacts/RUN-0003/{operators.pt,switched_metrics.json}` — global + 6 hard
  bin operators, bin edges, mu, lambdas, counts (row-random lambda selection).
- `artifacts/RUN-0004/{operators_grouped.pt,switched_metrics_grouped.json}` —
  the same, with file-disjoint grouped lambda selection.
- `results/diag_lam_encoder.json` — full lambda x centring table for this
  encoder and for EXP-0016's frozen random encoder.
