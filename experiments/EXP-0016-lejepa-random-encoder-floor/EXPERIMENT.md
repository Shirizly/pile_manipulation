---
# ---- identity -------------------------------------------------------------
id: EXP-0016
title: >
  Kill-gate for the LeJEPA + switched-linear program: what a FROZEN,
  RANDOMLY-INITIALISED residual-CNN encoder already achieves, in latent R^2
  and in slateN on DS-0001, before any representation learning at all
tier: T1
mode: exploratory
date: 2026-09-15
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  With the encoder of `docs/experimental_design/jepa_based_encoder.md` §3.1
  FROZEN at random initialisation (seed 0, never trained), the design's
  residual switched-linear latent dynamics (§8-10) already show a large
  single-vs-switched advantage in normalised latent error: test R^2 against
  the `Delta z = 0` baseline rises from +0.2483 (sd 0.0004 over 3 dynamics
  seeds) at K=1 to +0.3726 (sd 0.0012) at K=8, a gap of +0.124 against a
  cross-seed sd of ~0.002. Therefore "switched beats single in latent
  R^2" is NOT evidence that LeJEPA produced a piecewise-linear latent
  space -- a random encoder produces the same signature. Separately, on
  DS-0001 (`corner`/`lyapunov`, step 0) the same frozen random encoder
  scored through this experiment's post-hoc decoder reaches slateN +0.419
  (K=1, sd 0.012 over 3 dynamics seeds; per-run between-slate sem 0.065-0.080
  over n=20 slates), above EXP-0014's `random` floor (-0.141) and
  `persistence` (-0.004) and approaching EXP-0014's MODEL-0003 (+0.516) --
  so a large part of the control performance a LeJEPA pipeline would report
  is available with no representation learning. On slateN the K=1 vs K=8
  difference (+0.419 vs +0.377) is INSIDE the cross-seed noise floor and is
  reported as UNRESOLVED, not as a ranking. Claim scoped to: this encoder
  architecture at res64/latent 256/2 res-blocks, seed 0, one encoder seed;
  `overnight_randlen_{train,test}` for the dynamics fit; DS-0001
  `corner`/`lyapunov` step 0 for slateN.

prediction: null

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0016-lejepa-random-encoder-floor/code/
  data: ["DS-0001", "Genesis/data/overnight_randlen_train",
         "Genesis/data/overnight_randlen_test"]
  code_path: experiments/EXP-0016-lejepa-random-encoder-floor/code/
  seed: 0
  split: >
    The pre-existing file-level split `Genesis/data/overnight_randlen_{train,
    test}` was REUSED, not re-derived (192 train files / 98304 transitions;
    21 test files / 10752 transitions; invariant
    `randlen-train-test-file-disjoint` HOLDS). Latents were read with the
    fast path (glob `*_data.pt` + `torch.load`), never `load_randlen_cell()`.
    DS-0001 is a fixed evaluation pool with no train/test split, exactly as
    EXP-0012/0013/0014 use it; nothing in this record was fitted on it.
  runtime: "~13 min GPU total (see the cost table in the body)"
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004]
  env: "python 3.x, torch 2.11.0+cu130 (anaconda3/envs/pme), RTX 4070 Laptop 8 GB"

budget:
  declared: "90 min wall-clock / 200k tokens"
  spent: "~75 min / ~120k tokens"
  outcome: within

design:
  varied: {K: [1, 4, 8], dynamics_seed: [0, 1, 2], readout: [latent-R2, decoder-slateN]}
  held_fixed: {encoder: "ResCNNEncoder(res=64, latent=256, n_res_blocks=2), FROZEN at random init, seed 0",
               action_encoding: "[x, y, sin(theta), cos(theta), dx, dy] (design doc 5 plus the corpus's displacement)",
               optimiser: "AdamW, OneCycle, lr 3e-3, wd 1e-4, bs 1024, 60 epochs",
               goal: corner, value_fn: lyapunov, step: 0, dataset: DS-0001,
               grid: 64, pool_size: 1000, n_slates: 20,
               encoder_seed: 0}
  baselines: ["dz=0 (persistence-in-latent, the do-nothing baseline for the latent fit)",
              "persistence (image prediction baseline)",
              "random (the slateN ranking floor)",
              "decode-z0 (the decoder readout with no dynamics at all)"]
  metric: "slateN"

noise_floor: >
  Latent R^2: sd over 3 dynamics seeds per K, reported per cell -- 0.0004
  (K=1), 0.0024 (K=4), 0.0012 (K=8). The K=1 vs K=8 gap of +0.1243 is ~50x
  that sd and is RESOLVED. K=4 vs K=8 (+0.0043 against a combined sd ~0.003)
  is marginal and is reported as unresolved.
  slateN: two separate scales, both reported. Between-slate sem over the 20
  slates within one run is 0.052-0.091; sd across the 3 dynamics seeds is
  0.012 (K=1), 0.060 (K=4), 0.062 (K=8). The K=1 vs K=8 difference (+0.042)
  is smaller than both, so it is UNRESOLVED. Every slateN number here rests
  on a SINGLE encoder seed, so the encoder-seed component of the noise floor
  is unmeasured.

depends_on: [randlen-train-test-file-disjoint, occ-rasteriser-consistency,
             goal-mask-axis-convention-row-y-col-x, slates-binned-uniform-difficulty]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  HEADLINE (slateN, DS-0001, corner/lyapunov, step 0, n=20 slates; mean over
  3 dynamics seeds, with the per-run between-slate sem range in brackets):
  frozen-random-encoder K=1 +0.4191 (sem 0.065-0.080), K=4 +0.4117 (sem
  0.052-0.071), K=8 +0.3765 (sem 0.065-0.091); decode-z0 (no dynamics)
  -0.0042 (sem 0.089); persistence -0.0042 (sem 0.089); random -0.1409 (sem
  0.101). The persistence and random rows reproduce EXP-0014's to 4 decimal
  places, which is the provenance check that this scoring path is the same
  one. EXP-0014 reference rows for context: MODEL-0001 +0.7399, MODEL-0003
  +0.5163, MODEL-0002 -0.0904.
  LATENT (test R^2 vs the `Delta z = 0` baseline in the SAME latent space;
  mean +/- sd over 3 dynamics seeds): dz=0 baseline 0.0000 by construction;
  K=1 +0.2483 +/- 0.0004 (train +0.279); K=4 +0.3683 +/- 0.0024 (train
  +0.459); K=8 +0.3726 +/- 0.0012 (train +0.486). Per-dimension-normalised
  R^2 tracks it (+0.2358 / +0.3508 / +0.3544). THE SWITCHED-VS-SINGLE
  ADVANTAGE IS ALREADY PRESENT ON A RANDOM ENCODER.
  LATENT STATISTICS: z per-dimension std mean 0.0531 (min 0.0389, max
  0.0951) -- not degenerate per-dimension, but the spectrum is: effective
  rank 52.6 of 256 dimensions. ||mean(z)|| = 5.52 against a centred rms of
  0.054, i.e. z is a large constant offset plus a small state-dependent part.
  rms(dz)/rms(z centred) = 0.738.
  DECODER (frozen random z, 24576 train pairs, 6144 held-out test pairs):
  autoencoding accuracy vs an empty frame -0.111 (WORSE than predicting an
  empty frame); train mse 0.0038 vs test mse 0.0331, a ~9x generalisation
  gap. Image `accuracy` for the one-step prediction, swept-region mask
  (8.1% of the frame), persistence baseline: K=1 +0.137, K=8 +0.191, with a
  decode-the-TRUE-next-latent ceiling of +0.174. Full-frame accuracy is
  -1.00 for every cell including the ceiling, i.e. the decoder is worse than
  persistence everywhere outside the swept band.
  GOAL DEGENERACY: frac(dv_true == 0) = 0.031 on DS-0001; dv_true sd 0.0257,
  51.0% of candidates improving.
verdict: supported
downgrades: [provenance, incomplete-design, imprecision, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Lead finding: the kill-gate fires

**A frozen, randomly-initialised encoder already shows the switched-vs-single
advantage the design's headline comparison would report.** Test R^2 against
`Delta z = 0` goes +0.2483 (K=1) -> +0.3726 (K=8), a gap of +0.124 against a
cross-seed sd of ~0.002. No representation learning of any kind was involved:
the encoder's weights are `torch.manual_seed(0)` initialisation and were never
touched by an optimiser (`requires_grad_(False)` on every parameter,
`encoder_seed0.pt` persisted so the exact object can be reloaded).

So "K>1 beats K=1 in latent R^2" is **uninformative about LeJEPA**. Any future
record that reports it as evidence for the LeJEPA hypothesis must report this
random-encoder floor beside it and show the LeJEPA encoder's gap is *larger*,
not merely positive.

Two things sharpen this rather than soften it:

- **The advantage is not a discrete switch.** At K=8 the mean gate entropy is
  1.71 nats (max ln 8 = 2.08) and **0.0%** of test rows have a gate
  max > 0.9. The model is not selecting a mode; it is smoothly blending all
  eight. Measured directly on the fitted gates, not inferred.
- **The K>1 arms carry extra non-linear capacity beyond the extra operators.**
  At K=1 the softmax over a single logit is identically 1, so the gate MLP is
  dead weight; at K>1 it is a live 2-layer MLP on `[z, a_e]`. The K=1 vs K=8
  comparison therefore confounds "more linear modes" with "a non-linear gate
  MLP", and this confound is a property of the DESIGN's own comparison, not of
  this implementation. An untested hypothesis, flagged as such: the gap may be
  mostly the gate's non-linearity. The cheap discriminating cell is a K=1 model
  with the gate MLP's output fed in as an extra feature; it was not run.

## Second finding: slateN is high without any representation learning

Led with `slateN`, as the repo requires. The frozen random encoder + switched
linear dynamics + this experiment's decoder reaches **+0.419 (K=1)** on
DS-0001 `corner`/`lyapunov` at step 0, against EXP-0014's `random` floor of
-0.141 and its MODEL-0003 (+0.516) and MODEL-0001 (+0.740). `persistence` and
`random` reproduce EXP-0014's numbers exactly, so this sits on the same scale.

**On slateN, K=1 vs K=8 is UNRESOLVED.** +0.4191 vs +0.3765, against a
cross-dynamics-seed sd of 0.012 and 0.062 respectively and per-run between-slate
sems of 0.05-0.09. It is not a ranking and must not be read as one. The latent
R^2 ordering does **not** reproduce under the control metric.

`decode-z0` (the decoder readout with no dynamics at all) scores exactly
persistence's -0.0042 with only 21 distinct predicted values over 20000 rows:
the readout on its own is inert, so the +0.42 comes from the dynamics term,
not from the decoder's picture of the initial state.

## Third finding: the decoder does not invert a random encoder

Reconstruction from frozen random `z` is worse than an empty frame
(accuracy -0.111 against a zero-image baseline), with train mse 0.0038 against
test mse 0.0331. That ~9x gap is a **generalisation** failure, not
undertraining: the decoder memorises its 24576 training frames and the frozen
random `z` does not carry transferable layout information. Consistent with the
effective rank of 52.6 out of 256.

The image `accuracy` numbers must be read with that: swept-region accuracy is
+0.137 (K=1) / +0.191 (K=8) against a decode-the-true-next-latent **ceiling of
+0.174** — K=8 exceeds its own ceiling, which means these are all noise around
a decoder that cannot resolve the swept band. Full-frame accuracy is -1.00
everywhere. `accuracy` is a diagnostic here and carries no verdict.

## Cost table (RTX 4070 Laptop, 8 GB; batch sizes as run)

| stage | what | wall | peak GPU | batch |
|---|---|---|---|---|
| RUN-0001 | rasterise + frozen encoder forward, 98304 train transitions (196608 encoder forwards) | 87.5 s | 1064 MiB | 512 |
| RUN-0001 | same, 10752 test transitions | 8.8 s | 1064 MiB | 512 |
| RUN-0002 | one dynamics fit, 60 epochs x 98304 rows, K=1 | 19.7 s | 248 MiB | 1024 |
| RUN-0002 | same, K=4 | 22.2 s | 259 MiB | 1024 |
| RUN-0002 | same, K=8 | 22.4 s | 278 MiB | 1024 |
| RUN-0003 | decoder training, 40 epochs x 24576 pairs | 256.3 s | 1616 MiB | 256 |
| RUN-0004 | full slateN pass over DS-0001 (20000 rows: rasterise x2, encode, 4 readouts, lyapunov) | ~95 s | ~1.1 GiB | 512 |

**Batch size that fits at res64** (measured, `results/batch_headroom.json`):
encoder forward no-grad **2048** (4160 MiB; 4096 OOMs); decoder training step
with backward **1024** (4013 MiB; 2048 OOMs). Encoder forward throughput
~2.7 k images/s at batch 512-2048 (186 ms / 512). Dynamics fitting is
essentially free once latents are cached — the 98304x256 latent cache is 200 MB
and the whole K x seed sweep is under 4 minutes.

Sizing implication for the next experiments: **the encoder forward is not the
bottleneck and the dynamics fit is not the bottleneck.** A LeJEPA pretraining
run at res64 will be dominated by encoder backward passes; at batch 1024 the
forward alone is ~0.37 s, so a 100-epoch pass over 98 k states is on the order
of an hour or two, not a day. The decoder is the expensive small component
(256 s for 40 epochs on 24 k pairs) and the one with the worst return.

## What was actually run

Four runs, all in `code/`, all with `python -u`, all in the foreground:

- **RUN-0001** `encode_corpus.py` — builds and persists the frozen random
  encoder, rasterises both splits at 64x64 with `transforms.functional.
  particles_to_occupancy` (BOUNDS +/-0.064 m, GRID 64, footprint radius from
  the 5 mm cube — the same constants `scripts/probes/binned_pool_cache.py`
  uses), encodes to z, caches latents and an occupancy subset.
- **RUN-0002** `fit_dynamics.py` — K in {1,4,8} x seeds {0,1,2}, 9 fits, plus
  the `Delta z = 0` baseline and the latent statistics. Every operator
  persisted as `dyn_K{K}_seed{s}.pt` with its config, `mu`/`sd`, and metrics.
- **RUN-0003** `train_decoder.py` — decoder on detached, disk-loaded z. The
  encoder object is never constructed in this script, so no gradient path to
  it exists by construction (asserted: `not z.requires_grad`).
- **RUN-0004** `score_slaten.py` — DS-0001 scoring, run once per dynamics seed.

**The slateN code path is EXP-0014's**, reused rather than reimplemented:
`Genesis/binned_slate_dataset.py::BinnedSlateCorpus`, the same rasteriser and
constants, `control_utility_test.lyapunov`/`lyapunov_weights`, the same
`dv = value(after) - value(before)` convention, the same `random` baseline
(`torch.Generator().manual_seed(0)`), and the canonical
`Baselines/common/goals.py::slate_n_capture`. The exact reproduction of
EXP-0014's `persistence` (-0.0042) and `random` (-0.1409) rows is the evidence
that it is the same path.

Device was asserted, not assumed (`docs/CODEMAP.md`'s
`eval-baseline-scorer-batch-on-requested-device` trap): `assert
occ0.device.type == DEV` in RUN-0001 and RUN-0004, and the latent/occupancy
tensors are moved to CUDA explicitly in RUN-0002/0003. Every forward in this
record ran on `cuda:0`.

Corpus characterised from ALL 192 train files, not `glob(...)[0]`.

**No LeJEPA training, no analytical-transformation pretraining (design doc
§6, §7), and no encoder fine-tuning were run.** That is the design of this
record, not an omission — see the droppable/missing list below.

## What was uncommitted (dirty tree)

The tree was dirty when every run executed, at commit `6ea03278`. Modified:
`.claude/skills/{experiment-log,project-overview}/SKILL.md`,
`Genesis/sandbox_manipulation_clean.py`, `docs/piled_collection.md`,
`experiments/{INVARIANTS,METRICS,REGISTER,TEMP_LOG}.md`,
`scripts/probes/{pool_common,pool_inspect,pool_survey}.py`,
`weights/MODEL-000{1,2,3}-*/tests.md`. Untracked:
`.claude/orchestrator-notes.md`, `.claude/skills/{data-collection,
subagent-experimenter}/`, `Baselines/common/goal_configs.py`,
`Genesis/binned_slate_{collection,dataset}.py`, `Genesis/data/{Sean,
slates_binned}/`, `datasets/`, `docs/CODEMAP.md`,
`docs/experimental_design/jepa_based_encoder.md`,
`experiments/EXP-001{1,2,3,4,5}-*/`, `scripts/probes/binned_pool_cache.py`,
and this record.

Of these, the ones this record's numbers actually depend on are
`Genesis/binned_slate_dataset.py` (reads DS-0001) and
`Genesis/data/slates_binned/` (is DS-0001). Nothing in
`scripts/probes/binned_pool_cache.py` was executed — its constants and
conventions were copied into `code/score_slaten.py`, which is why the
persistence/random reproduction check matters.

## Cells NOT run

- **No LeJEPA encoder.** This record is the floor, by design. The comparison
  the program actually needs — LeJEPA-K=8 vs LeJEPA-K=1 vs random-K=8 vs
  random-K=1, on the same metric — is one arm short and cannot be completed
  from here.
- **One encoder seed only (seed 0).** Every number rests on it. The
  encoder-seed component of the noise floor is unmeasured, and a single random
  draw could be unrepresentative. Cost to close: ~4 min per extra seed for
  RUN-0001+0002, ~6 min including the decoder.
- **One goal, one value function** (`corner`/`lyapunov`), one step, one pool.
  `slateN`'s known weakness is power; the repo standard is breadth.
  EXP-0013's 9-cell sweep is the template. Cost: ~15 min.
- **The gate-MLP-vs-extra-modes discriminating cell** (see above).
- **res32 and other latent dims / n_res_blocks.** `input_resolution`,
  `latent_dim` and `n_res_blocks` are configurable in `code/model.py` and
  were swept over exactly one value each.
- **The analytical-transformation pretraining of design doc §7.**

## What would change the verdict

- The claim that the switched-vs-single signature is uninformative would be
  weakened if the LeJEPA encoder's K=1->K=8 gap were much larger than +0.124
  under the same fitting protocol. That is the cell to run next, and it must
  use this record's exact `fit_dynamics.py` so the two gaps are comparable.
- The slateN +0.419 would be undermined if it did not survive more goals,
  more value functions, or a second encoder seed.
- The decoder result is the weakest part: the decoder is small, trained on a
  quarter of the corpus, and already overfitting. A bigger decoder trained on
  all 98 k frames might invert the random encoder better, which would raise
  the image-space floor further — it would not lower it.

## Unrelated findings

- **`experiments/METRICS.md` line 5 still says `slate4` where the rest of the
  file, and the repo standard since 2026-09-08, say `slateN`.** One-word doc
  inconsistency, noted not fixed.
- **`scripts/check_register.py` was run against this record and must exit 0;**
  see the runs directory for the invocation.
- **The `pme` conda environment is the working one for this repo.** The `cge`
  environment has a `cv2` / `libstdc++` `CXXABI_1.3.15` mismatch that makes
  `import Baselines.common.goals` fail outright. Added to `docs/CODEMAP.md` —
  two minutes were lost to it here and would be lost again.
