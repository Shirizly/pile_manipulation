---
# ---- identity -------------------------------------------------------------
id: EXP-0019
title: >
  LeJEPA (view-alignment + SIGReg) encoder training runs end to end and does
  NOT variance-collapse, but the effective rank of z falls 44.6 -> 7.5 of 256
  and the resulting latent generalises far worse across held-out files than a
  frozen random encoder; hard push-length gating beats a single global operator
  in latent R^2 in both latent spaces
tier: T1
mode: exploratory
date: 2026-09-15
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  (1) Stage 1 of `docs/experimental_design/jepa_based_encoder.md` (LeJEPA =
  view-alignment on a projector output + SIGReg, lambda_sigreg 0.02) trains
  the EXP-0016 `ResCNNEncoder` (res 64, latent 256, 2 res-blocks) on
  `overnight_randlen_train` + 60 `Genesis/data/Sean` files WITHOUT the
  variance collapse of INVARIANTS `hybrid-latent-stage2-anticollapse`:
  per-dimension std RISES from 0.0453 at init to 0.162 (min 0.058), nowhere
  near the ~2e-6 collapse floor. (2) It nevertheless suffers a RANK collapse
  the design does not guard against: effective rank of z falls from 44.6 at
  random init to 7.5 of 256, because SIGReg is applied to the projection p,
  not to z, and the SIGReg statistic measured on z is flat (499 at init, 479
  at epoch 11). (3) Replacing the design's section-10 soft softmax gate with a
  HARD gate on push length (repo scheme: `Baselines/LinearForesight`,
  6 equal-width bins over [0, 80.00] mm, `MIN_ROWS_PER_BIN` 50) raises latent
  R^2 on the file-disjoint `overnight_randlen_test` split from +0.0325
  (single global operator) to +0.0597 (6 hard bins), with every bin holding
  6666-24925 train rows, i.e. no starvation. (4) With the fit method held
  exactly fixed, the SAME closed-form hard-gate fit on EXP-0016's FROZEN
  RANDOM encoder latents gives +0.2087 global / +0.2693 switched with a
  train-test gap of 0.03, against this LeJEPA encoder's gap of 0.28
  (train +0.379 / test +0.060) -- so the LeJEPA-trained representation
  transfers much worse to held-out files. Scoped to: this encoder
  architecture and seed 0, 12 epochs x 250 steps at batch 192, this view set
  (particle dropout + footprint-radius jitter + occupancy noise),
  closed-form ridge dynamics, `overnight_randlen_{train,test}`.

prediction: null

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0019-lejepa-encoder-pushlen-switched/code/
  data: ["Genesis/data/overnight_randlen_train", "Genesis/data/overnight_randlen_test",
         "Genesis/data/Sean (60 of 1230 *_data.pt files, strided)"]
  code_path: experiments/EXP-0019-lejepa-encoder-pushlen-switched/code/
  seed: 0
  split: >
    The pre-existing file-level split `Genesis/data/overnight_randlen_{train,
    test}` was REUSED, not re-derived (192 train files / 98304 transitions;
    21 test files / 10752 transitions). Encoder Stage 1 saw ONLY train files
    (plus Sean states, which are a separate corpus). Loaded with the fast path
    (glob `*_data.pt` + `torch.load`); `load_randlen_cell()` never called.
    The inner val split used for ridge-lambda selection was ROW-RANDOM over
    train rows and is shown below to be optimistic -- see Threats.
  runtime: "~30 min GPU (encoder 23 min, encode 1.4 min, fits ~2 min), RTX 4070 Laptop 8 GB"
  runs: [RUN-0001, RUN-0002, RUN-0003]
  env: "torch 2.11.0+cu130 (anaconda3/envs/pme), CUDA, RTX 4070 Laptop 8 GB"

budget:
  declared: "110 min wall-clock / 220k tokens"
  spent: "~95 min / ~140k tokens"
  outcome: within

design:
  varied: {gate: ["none (single global operator)", "HARD, 6 equal-width push-length bins"],
           ridge_lambda: [0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5],
           centring: ["z - mean(z_train)", "raw z"],
           encoder: ["EXP-0019 LeJEPA-trained", "EXP-0016 frozen random (diagnostic only)"]}
  held_fixed: {encoder_arch: "ResCNNEncoder(res=64, latent=256, n_res_blocks=2)",
               lejepa: "alignment ||p1-p2||^2 + 0.02 * SIGReg(p), projector 256-1024-256 with BatchNorm+SiLU",
               views: "particle dropout keep~U(0.80,1.0) + footprint-radius jitter x U(0.85,1.15) + additive occupancy noise std 0.02 clamped to [0,1]; NO rotations, flips or crops",
               optimiser: "AdamW, OneCycle, max_lr 1e-3, wd 1e-4, bs 192, 12 epochs x 250 steps",
               action_encoding: "[x_s, y_s, sin(theta), cos(theta), dx, dy] (EXP-0016 encode_action)",
               dynamics: "closed-form ridge (toward zero) of dz on [z, a, 1]; residual z' = z + (A_b z + B_b a + c_b)",
               bin_scheme: "Baselines/LinearForesight bin_index, 6 equal-width bins over [0, max train push length], MIN_ROWS_PER_BIN 50",
               grid: 64, rasteriser: "disk splat, footprint radius 1.25 vox (exact match to transforms.functional.particles_to_occupancy, asserted)"}
  baselines: ["dz = 0 (do nothing / persistence-in-latent, 0 by construction)",
              "global single operator (one bin)",
              "EXP-0016 frozen-random-encoder floor, same fit method"]
  metric: >
    latent_r2 = 1 - mse(dz_pred - dz_true)/mse(dz_true), i.e. skill against a
    `Delta z = 0` persistence-in-latent baseline IN THE SAME LATENT SPACE
    (0 = no better than assuming the push did nothing). Same definition as
    EXP-0016. COMPARABLE ONLY WITHIN ONE ENCODER -- a different encoder moves
    both the target and the denominator, so the cross-encoder rows below are
    a diagnostic of the fit pipeline, not a ranking of encoders. `slateN` was
    explicitly OUT OF SCOPE for this run (it needs the value readout another
    agent is building); no `accuracy` was measured either.

noise_floor: >
  NOT CHARACTERISED for this record -- a single encoder seed and a single
  dynamics fit were run, and the ridge fit is closed-form and deterministic
  (bitwise reproducible given the same latents), so there is no seed spread
  to quote. The switched-vs-global gap (+0.0597 vs +0.0325 at the selected
  lambda; +0.0703 vs +0.0497 at the best-on-test lambda) is therefore
  reported WITHOUT an uncertainty and should be treated as unresolved
  against encoder-seed variation, which is unmeasured. The lambda sweep is
  the only spread available: switched exceeds global at every lambda from
  0.1 to 1000, which is weak evidence the ordering is not a lambda artifact.

depends_on: [randlen-train-test-file-disjoint, occ-rasteriser-consistency,
             hybrid-latent-stage2-anticollapse,
             slates-multistep-single-pushlen-bin-starvation]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  ENCODER (Stage 1, 12 epochs): NO variance collapse -- per-dim std of z rose
  0.0453 -> 0.1620 (min 0.0579, max not tracked per-epoch), vs the ~2e-6
  collapse floor of `hybrid-latent-stage2-anticollapse`; but effective rank
  fell 44.6 -> 7.5 of 256 dims, and ||mean(z)|| 5.12 -> 3.95 against a centred
  rms of 0.168 (offset still ~24x the centred scale). Alignment loss fell
  0.0764 -> 0.0053; SIGReg(p) 13.4 -> 1.51; SIGReg(z), a no-gradient probe,
  stayed flat 499 -> 479. LATENT R^2 (test, file-disjoint, vs dz=0): dz=0
  0.0000 by construction; global single operator +0.0325 (train +0.3173);
  HARD 6-bin push-length switched +0.0597 (train +0.3789). Bin row counts
  (train / test): 7428/811, 18330/2042, 24925/2769, 22275/2413, 18680/1958,
  6666/759 -- NO bin starved, none fell back to the global operator, and no
  two per-bin operators are byte-identical (pairwise relative Frobenius
  0.70-3.61). DIAGNOSTIC, same fit method on EXP-0016's frozen random
  encoder: global +0.2087 (train +0.2358), switched +0.2693 (train +0.3146).
verdict: inconclusive
downgrades: [imprecision, incomplete-design, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If LeJEPA Stage 1 were working as designed, the encoder would leave its latent
approximately isotropic Gaussian (that is what SIGReg is for), and the latent
would transfer to unseen pile states. Effective rank of z and the SIGReg
statistic measured on z separate "the objective is regularising the space the
dynamics model uses" from "the objective is regularising only the projector
output." The file-disjoint train/test gap separates "the representation is
useful" from "the representation memorised the training files." Neither
quantity can be faked by the dynamics fit, which is closed-form.

## What was actually run

- **RUN-0001 (Stage 1 encoder training).** 214688 states (both `states` and
  `states_` of every transition in the 192 `overnight_randlen_train` files,
  plus 60 strided `Genesis/data/Sean/**/*_data.pt` files; `*_failed.pt` and
  `.zip` excluded). Particle rows padded to N=100 **by repeating existing
  particles** -- the splat is a boolean `any`, so duplicates are an exact
  no-op, and the real particle count is carried separately so dropout never
  sees the padding. Loss = `||p1-p2||^2 + 0.02 * SIGReg(stack(p1,p2))`, with
  `SIGReg` the vendored `le-wm/module.py` Epps-Pulley sketch (torch + einops
  only; `stable_pretraining` neither installed nor needed). Checkpoint written
  every epoch to `artifacts/RUN-0001/encoder_lejepa.pt` with its resolved
  config, the curves, and a one-line description of the view set.
- **Views** (design doc 6.2, deliberately NOT generic vision augmentation):
  particle dropout with keep fraction ~ U(0.80, 1.0) over the real grains
  (never all-dropped), footprint-radius jitter x U(0.85, 1.15) around the
  1.25-voxel cube footprint, and additive occupancy noise std 0.02 clamped to
  [0,1]. No rotation, flip, crop, or translation -- all of those would change
  the pile/tool frame relationship.
- **RUN-0002 (encode).** Deterministic rasterisation (no dropout, fixed
  radius) of both endpoints of all 98304 train / 10752 test transitions
  through the trained encoder; 75 s + 8 s on CUDA.
- **RUN-0003 (hard-gate fit).** Closed-form ridge, float64, on GPU. Bin edges
  from the TRAIN max push length (80.00 mm) only.
- **Deviation from the brief:** the design doc's soft learned gate was not run
  at all here; the hard gate replaced it, as instructed. `slateN` was not
  attempted. Only one encoder configuration was trained (no lambda_sigreg
  sweep, no `--sigreg-on-z` run) -- see "What would change the verdict".

### Dirty tree (`dirty: true`) — what was uncommitted at run time

`provenance.commit` 6ea03278 does NOT reconstruct this run. Modified and
tracked at run time: `.claude/skills/{experiment-log,project-overview}/SKILL.md`,
`Genesis/sandbox_manipulation_clean.py`, `docs/piled_collection.md`,
`experiments/{COMMANDS.jsonl,INVARIANTS.md,METRICS.md,REGISTER.md,TEMP_LOG.md}`,
`scripts/probes/pool_{common,inspect,survey}.py`,
`weights/MODEL-000{1,2,3}-*/tests.md`, and the `le-wm` nested repo.
Untracked at run time and load-bearing for THIS record: all of
`experiments/EXP-0019-lejepa-encoder-pushlen-switched/` (this record's own
code), `experiments/EXP-0016-lejepa-random-encoder-floor/` (whose `model.py`
this imports and whose latents the diagnostic table uses),
`docs/experimental_design/jepa_based_encoder.md`, `docs/CODEMAP.md`, and
`Genesis/data/Sean/` (a training corpus). Nothing in
`transforms/`, `Baselines/LinearForesight/`, or `le-wm/module.py` — the three
project modules this run imports from — was modified by this run.

## Numbers

### Stage 1 encoder training curves (RUN-0001, `artifacts/RUN-0001/train_curves.json`)

Statistics are of **z** (the encoder output the dynamics model uses), on 8192
train states, recomputed every epoch. `sigreg(z)` is a no-gradient probe: the
same Epps-Pulley statistic applied to z instead of p.

| epoch | align | SIGReg(p) | z per-dim std (mean / min) | eff. rank of 256 | \|\|mean z\|\| | centred rms | SIGReg(z) |
|---|---|---|---|---|---|---|---|
| -1 (random init) | - | - | 0.0453 / 0.0327 | 44.6 | 5.125 | 0.0459 | 499 |
| 0 | 0.0764 | 13.45 | 0.0745 / 0.0293 | 10.2 | 6.409 | 0.0767 | 535 |
| 1 | 0.0674 | 5.094 | 0.0573 / 0.0324 | 13.9 | 6.243 | 0.0585 | 521 |
| 2 | 0.0396 | 3.346 | 0.0620 / 0.0323 | 12.2 | 6.101 | 0.0637 | 524 |
| 3 | 0.0247 | 2.631 | 0.0771 / 0.0360 | 10.1 | 6.032 | 0.0797 | 532 |
| 4 | 0.0165 | 2.257 | 0.1029 / 0.0431 | 8.5 | 5.658 | 0.1066 | 523 |
| 5 | 0.0129 | 2.002 | 0.1048 / 0.0370 | 8.1 | 5.235 | 0.1090 | 502 |
| 6 | 0.0104 | 1.844 | 0.1281 / 0.0461 | 7.9 | 4.739 | 0.1328 | 484 |
| 7 | 0.0087 | 1.729 | 0.1354 / 0.0472 | 7.8 | 4.309 | 0.1405 | 481 |
| 8 | 0.0074 | 1.640 | 0.1624 / 0.0488 | 7.5 | 4.027 | 0.1685 | 478 |
| 9 | 0.0063 | 1.576 | 0.1572 / 0.0495 | 7.4 | 4.052 | 0.1631 | 475 |
| 10 | 0.0056 | 1.530 | 0.1639 / 0.0594 | 7.5 | 3.951 | 0.1702 | 481 |
| **11 (final)** | **0.0053** | **1.513** | **0.1620 / 0.0579** | **7.5** | **3.953** | **0.1681** | **479** |

Reference, EXP-0016's frozen random encoder measured on the full train set:
effective rank 52.6, per-dim std ~0.053, ||mean z|| 5.52 vs centred rms 0.054.
The init row above (44.6 on an 8192-row subsample) is the same encoder family
at a different seed and a smaller sample; effective rank is sample-size
sensitive, so 44.6 and 52.6 are consistent.

### latent R^2 (RUN-0003, selected lambda, z centred)

| cell | lambda | latent R^2 test | latent R^2 train |
|---|---|---|---|
| dz = 0 (do nothing, in-latent persistence) | - | **+0.0000** | +0.0000 |
| global single operator (one bin) | 0.1 | **+0.0325** | +0.3173 |
| HARD gate, 6 push-length bins | 1 | **+0.0597** | +0.3789 |

### per-bin rows and per-bin latent R^2 (RUN-0003)

| bin | range (mm) | n train | n test | starved? | latent R^2 on this bin's test rows, own operator | same rows, global operator | \|\|W_b\|\|_F | \|\|W_b - W_global\|\|_F |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0 - 13.3 | 7428 | 811 | no | -0.0082 | -0.2244 | 1.018 | 3.612 |
| 1 | 13.3 - 26.7 | 18330 | 2042 | no | -0.0030 | -0.0872 | 1.725 | 2.916 |
| 2 | 26.7 - 40.0 | 24925 | 2769 | no | +0.0084 | -0.0197 | 2.352 | 2.335 |
| 3 | 40.0 - 53.3 | 22275 | 2413 | no | +0.0413 | +0.0407 | 2.997 | 2.100 |
| 4 | 53.3 - 66.7 | 18680 | 1958 | no | +0.1014 | +0.0910 | 3.660 | 2.195 |
| 5 | 66.7 - 80.0 | 6666 | 759 | no | +0.1448 | +0.1148 | 3.427 | 2.949 |

No bin fell below `MIN_ROWS_PER_BIN` (50), none fell back to the global
operator, and no pair of per-bin operators is byte-identical (pairwise
relative Frobenius distance 0.70 - 3.61). The starvation trap
(`slates-multistep-single-pushlen-bin-starvation`) does NOT apply to
`overnight_randlen`, as expected.

### lambda x centring sweep, test split (`results/diag_lam_encoder.json`)

LeJEPA-trained encoder (this record), z centred:

| lambda | global train | global test | switched train | switched test |
|---|---|---|---|---|
| 0.01 | +0.3176 | +0.0302 | +0.3840 | +0.0299 |
| 0.1 | +0.3173 | +0.0325 | +0.3826 | +0.0428 |
| 1 | +0.3165 | +0.0377 | +0.3789 | +0.0597 |
| 10 | +0.3142 | +0.0453 | +0.3713 | +0.0703 |
| 100 | +0.3089 | +0.0497 | +0.3595 | +0.0703 |
| 1000 | +0.3000 | +0.0470 | +0.3272 | +0.0542 |
| 1e4 | +0.2633 | +0.0310 | +0.1848 | +0.0277 |
| 1e5 | +0.1162 | +0.0115 | +0.0381 | +0.0073 |

Uncentred z reproduces these to within 0.0002 at every lambda below 1e4
(e.g. lambda=1 switched test +0.0598 uncentred vs +0.0597 centred), because
the feature vector `[z, a, 1]` already carries an intercept column that
absorbs the constant offset exactly. Centring only matters at lambda >= 1e4,
where the ridge penalty starts fighting the intercept.

Frozen RANDOM encoder (EXP-0016 latents, IDENTICAL fit code and split) --
a pipeline diagnostic, NOT a cross-encoder ranking:

| lambda | global train | global test | switched train | switched test |
|---|---|---|---|---|
| 0.01 | +0.2364 | +0.2079 | +0.3184 | +0.2636 |
| 1 | +0.2358 | +0.2087 | +0.3146 | +0.2693 |
| 10 | +0.2335 | +0.2090 | +0.3018 | +0.2662 |
| 100 | +0.2207 | +0.1999 | +0.2545 | +0.2292 |
| 1000 | +0.1742 | +0.1583 | +0.1771 | +0.1608 |

Train-test gap at the best-test lambda: **0.03** (random encoder) vs **0.28**
(LeJEPA encoder). Hard push-length gating beats the single global operator in
BOTH latent spaces, so that part of the result is not specific to this
encoder. EXP-0016's own soft-gate SGD fit on the same random latents reported
+0.2483 (K=1) / +0.3726 (K=8); the closed-form K=1-equivalent here (+0.2087)
is slightly below their K=1, consistent with their action MLP and 60 epochs of
SGD buying a little over a closed-form linear-in-`a` fit.

## What would change the verdict

1. **`--sigreg-on-z` (implemented in `code/train_encoder.py`, NOT RUN).** Apply
   SIGReg directly to z as well as to p. If effective rank of z stays near 44
   and the train-test gap shrinks, the rank collapse and the transfer failure
   are both attributable to the design's SIGReg *placement*, which would be a
   concrete fix to the design doc. Cost: ~8 min for a 4-epoch run, ~25 min for
   a matched 12-epoch run, plus 4 min to re-encode and refit.
2. **A second encoder seed.** Everything here rests on encoder seed 0, so no
   part of the switched-vs-global gap has an uncertainty attached. Cost: one
   more 23-min training run.
3. **A lambda_sigreg sweep (0.02 is the design doc's starting value, untested
   here).** If the rank collapse is weight-sensitive it is a tuning problem,
   not a structural one.
4. **Longer / larger-batch training.** Alignment loss was still falling at
   epoch 11, so the encoder is not converged; the rank trend was flat from
   epoch 8 onward, so the rank result is probably not an undertraining
   artifact, but that is an inference from a flat curve, not a test.
5. **`slateN`** through the value readout being built elsewhere. Latent R^2
   says nothing about control utility, and EXP-0016 already found a case
   where a latent R^2 ordering did not reproduce under slateN.

## Threats

- **imprecision.** One encoder seed, one dynamics fit, no seed spread. The
  switched-vs-global gap (+0.027 at the selected lambda) has no uncertainty
  attached and is reported as unresolved against encoder-seed variation.
- **incomplete-design.** The decisive follow-up (SIGReg applied to z) was
  implemented but not run for budget; the lambda_sigreg sweep and the second
  seed were not run; `slateN` was out of scope by instruction. The verdict is
  therefore `inconclusive`, not `supported`: the record establishes what the
  encoder did, not why.
- **untested-dependency.** `randlen-train-test-file-disjoint` and
  `occ-rasteriser-consistency` are cited but not re-verified here. The
  rasteriser used is a new local implementation; it was asserted bit-exact
  against `transforms.functional.particles_to_occupancy` on 64 real rows
  (`code/check_raster.py`, exit 0, zero mismatched cells), which covers the
  rasteriser threat but not the split threat.
- **Ridge-lambda selection leaked, and it changed the headline.** The inner
  val split was ROW-RANDOM over train rows and scored +0.359 where the
  file-disjoint test scored +0.060 -- a 0.30 gap, which means rows within a
  train file are NOT independent for this purpose, contrary to the comment
  written in `fit_switched_hard.py`. It selected lambda=1 where the best
  test lambda was 10-100 (switched test +0.0703 vs +0.0597). The headline
  row uses the leak-selected lambda, so it understates the switched model by
  ~0.011; the full sweep is reported above so the reader can see both. This
  is the same failure mode as `readout-cv-folds-slate-aware` in a different
  corpus and would justify a new invariant tag, which this record does not
  claim (one encoder, one corpus).
- **Considered and dismissed: the constant latent offset dominating the fit.**
  ||mean z|| 3.83 is ~20x the centred rms 0.195, but the uncentred and
  centred sweeps agree to within 0.0002 below lambda=1e4 because `[z, a, 1]`
  carries an explicit intercept. The offset is not affecting these numbers.
- **Considered and dismissed: a broken fit pipeline explaining the low test
  R^2.** The identical code on EXP-0016's random latents gives +0.209/+0.269
  test with a 0.03 gap, so the pipeline generalises fine; the difference is
  the encoder.
- **Untested hypothesis, labelled as such: WHY the LeJEPA encoder transfers
  worse.** Two candidates were not separated -- (a) rank collapse to ~7
  dimensions throws away the state information the dynamics need, (b) the
  alignment objective drove the encoder to features specific to the training
  files. The `--sigreg-on-z` run above would discriminate them.

## Unrelated findings

- **`transforms/functional.py::particles_to_occupancy` is ~77x slower than it
  needs to be in its `footprint_radius` branch.** It loops `for b in range(B)`
  in Python and materialises an `(N, H, W, 2)` diff per sample. Measured on
  B=64, N=20 real rows on CUDA: 77 ms vs 1 ms for a batch-vectorised
  equivalent (`code/raster.py::rasterise`), which is **bit-exact** against it
  (zero mismatched cells, `code/check_raster.py`). The same Python-loop
  pattern is in the `sigma > 0` branch. Not fixed here; it is a project module
  and a change there touches every caller.
- **`Genesis/data/Sean` mixes particle counts 20/50/100 in one corpus.** Any
  code that stacks its `states` tensors must pad or group; naive
  `torch.cat` across files fails on dimension 1. Observed directly while
  loading (`code/data.py` pads by repeating particles, which is a no-op for a
  boolean splat).
- **`le-wm` is a git submodule or nested repo** -- `git status` reports it as
  `M le-wm` with no file detail. `le-wm/module.py::SIGReg` is genuinely
  dependency-free (torch + einops), so `stable_pretraining` is not needed; the
  `le-wm/jepa.py`/`train.py` around it are a ViT/CLS design and were not used.
- **`docs/CODEMAP.md` had no entry for `le-wm` or SIGReg.** Added in this run.
