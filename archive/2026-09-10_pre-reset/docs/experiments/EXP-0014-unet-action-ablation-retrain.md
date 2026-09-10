---
# ---- identity -------------------------------------------------------------
id: EXP-0014
title: Retrained on the fixed raster, the UNet now depends heavily on its action channel
tier: T2
mode: confirmatory
date: 2026-09-05
hypothesis: H-A1

# ---- the claim ------------------------------------------------------------
claim: >
  C-020: the scattered-monolayer UNet's near-blindness to its action channel
  (EXP-0004: shuffling the action costs 3.1 of the model's 8.2-point whole-image
  rms advantage over persistence) was caused by the transposed occupancy/plate
  raster (EXP-0001), now fixed in aac084e3. If true, retraining the identical
  config on corrected data should make the action-shuffle gap widen
  substantially toward most of the model's (possibly larger) advantage over
  persistence.

prediction:
  supports: "On the retrained checkpoint, (persistence_rms - shuffled_rms) / (persistence_rms - true_action_rms) >= ~0.6, i.e. the shuffle gap captures most of the model's advantage over persistence (vs 3.1/8.2 = 38% in EXP-0004), AND true_action rms as %persistence is lower than EXP-0004's 91.8%."
  refutes:  "The shuffle gap stays within a couple of points of EXP-0004's 3.1 (i.e. shuffle_rms - true_rms stays in the 2-4 point range as %persistence), regardless of whether overall error falls -- meaning the transpose was not the main reason the model ignored its action."
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 006004d0 (training + ablation both run here; grid-convention fixed at aac084e3)
  script: scripts/probes/unet_action_ablation.py (run unmodified, only the `run` path variable substituted per checkpoint -- see "What was actually run")
  data: ["runs_cubes/unetfilm_corl_limited_100e_fixedraster/unet_best.pth (new)", "runs_cubes/unetfilm_corl_limited_100e/unet_best.pth (old, diagnostic only)", "corl_limited/cubes val split, 200 of 472 samples"]
  code_path: "PileSweepData raster, post-fix (the model's own training path for the new checkpoint; a distribution shift for the old one -- see Threats)"
  seed: "n/a (deterministic forward pass; batch permutation for the shuffle arm is unseeded, as in EXP-0004; training used no explicit seed, matching the original run -- see Threats)"
  split: "the run's own val split (deterministic md5-hash-of-filename bucketing, identical for both checkpoints since paths/val_pct/test_pct match)"
  runtime: "~24 min GPU (training, 100 epochs) + ~1 min CPU/GPU (both ablations)"

budget:
  declared: "100 min, 200k tokens"
  spent: "~55 min, ~90k tokens"
  outcome: within

design:
  varied: {checkpoint: ["old (trained pre-fix, transposed data)", "new (trained post-fix, corrected data)"], action_channel: [true, zeroed, shuffled], pile_channel: [true, transposed]}
  held_fixed: {architecture: unetfilm (in_channels=2, cond_dim=3, uses_physics=true, input_mode=standard), epochs: 100, batch_size: 32, lr: 1e-4 StepLR(step=50,gamma=0.75), loss: eulerian_combined(mse=1.0,mass=0.2), dataset: corl_limited/cubes, val_pct/test_pct: 10/10, resolution_scale: 1.0, mixed_precision: true, grad_clip_norm: 1.0, samples_evaluated: 200, sigmoid: applied, metric: whole-image rms}
  baselines: [persistence]
  metric: "pct_persistence_wholeimage — NOT comparable with swept-region numbers — see docs/experiments/METRICS.md. Originally: whole-image rms against the target occupancy, as a percentage of the persistence"

noise_floor: "not independently measured for this record either (inherited limitation from EXP-0004); the shuffle arm's own permutation is unseeded so its single value carries unquantified sampling noise -- see Threats. Given the effect sizes involved (single-digit points) this stays a T2 gate on the strength of the discriminating-prediction design, not on a measured floor, which is an honest gap flagged below rather than hidden."

depends_on: [grid-convention, rasteriser-identity]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  New checkpoint (post-fix): persistence 100.0%, true action 72.0%, action
  zeroed 99.7%, action shuffled 110.5% -- shuffling the action now makes the
  model WORSE than persistence, and zeroing it collapses the model to
  persistence. The 8.2-point advantage/3.1-point shuffle-gap of EXP-0004
  becomes a 28.0-point advantage whose shuffle-gap (38.5 points) exceeds the
  total advantage entirely. C-020 supported, decisively.
verdict: supported
downgrades: [imprecision, indirectness]
grade: low
supersedes: []
superseded_by: [EXP-0021]
invalidated_by: null
---

## Why this test discriminates

EXP-0004 found the trained UNet gets only 3.1 of its 8.2-point rms advantage
over persistence from actually knowing which push was applied (the rest
survives with the action zeroed or shuffled). Two explanations were live: (a)
the model can't use the action because the occupancy and action channels were
mutually transposed at training time (C-020), so any action-dependence it
learned would have to be a reflection; or (b) the model just doesn't need the
action much on this data/architecture/budget regardless of raster correctness
(too little data, 100 epochs, target dominated by unchanged pixels). Retraining
the identical config on the now-corrected raster and re-running the exact same
ablation separates these: (a) predicts the shuffle-gap grows substantially; (b)
predicts it does not.

## What was actually run

- New checkpoint: `python -m training.train configs/training/unetfilm_corl_limited_100e_fixedraster.yaml --no-resume`. This config is `runs_cubes/unetfilm_corl_limited_100e/run_config.yaml` copied verbatim (i.e. the trainer's own resolved-config snapshot of the original run, not the current `configs/dataset/genesis_corl_limited_cubes.yaml` / `configs/model/unetfilm_corl_limited.yaml` files, which have since drifted -- e.g. the current dataset config's `resolution_scale` is 0.5, but the original run used 1.0. Using `run_config.yaml` as the source of truth guarantees a true like-for-like retrain.) Only `output.log_dir` was changed, to `runs_cubes/unetfilm_corl_limited_100e_fixedraster`. No seed was set in either run (none was recorded in the original config), so this is a like-for-like omission, not a new source of variance; the train/val/test split is unaffected by seed because `_assign_split` buckets by an md5 hash of the file path, not by a random draw.
- Ablation: `scripts/probes/unet_action_ablation.py` was run unmodified except for the `run = "..."` path string, once per checkpoint. No other line was touched. Sigmoid is applied inside the script as before.
- Cost pilot (disclosed, no outcome kept): a 2-epoch timing run on a scratch log_dir was used only to measure per-epoch wall-clock (~14-20s/epoch) before committing to the full 100-epoch run; its weights and metrics were discarded and are not part of any reported number.
- The old checkpoint's ablation necessarily runs on today's code, which now rasterises occupancy in the corrected convention -- so its "true action" row is the old (pre-fix-trained) weights evaluated on a shifted input distribution, not a fair baseline. It is reported strictly as a diagnostic, per the task brief.
- Swept-region rms was not obtained: `scripts/probes/unet_action_ablation.py` calls `build_dataset(...)`, which returns the `EulerianDatasetWrapper` -- it does not expose the per-sample push start/stop pixel coordinates needed by `swept_region_mask` (`fit_linear_foresight.py`), only `{input, physics, target}`. Reconstructing them means reaching into `PileSweepData`'s internal `_run_lookup`/`_extract_sample_in_pxl` per validation index, which was judged not cheap enough to risk getting subtly wrong inside this budget. This was an optional stretch metric per the task brief ("if you can get it cheaply"), not a required design cell -- the required metric (whole-image rms, matching EXP-0004's own definition exactly) was obtained for all four arms on both checkpoints, so this is noted as a limitation rather than taken as an `incomplete-design` downgrade.

## Numbers

Whole-image rms, sigmoid applied, 200 validation samples, as % of persistence (persistence rms is identical for both checkpoints' data since it is computed from the same val split and a transpose does not change an L2 distance):

| checkpoint | variant | rms | vs persistence |
|---|---|---|---|
| — | persistence (copy input) | 0.12055 | 100.0% |
| **old** (pre-fix training, transposed data) | true action | 0.10570 | 87.7% |
| old | action shuffled across batch | 0.11392 | 94.5% |
| old | action zeroed | 0.11638 | 96.5% |
| old | pile channel transposed | 0.38557 | 319.8% |
| **new** (post-fix training, corrected data) | true action | 0.08678 | 72.0% |
| new | action shuffled across batch | 0.13316 | 110.5% |
| new | action zeroed | 0.12019 | 99.7% |
| new | pile channel transposed | 0.38651 | 320.6% |

Derived quantities (new checkpoint, the decisive row):

| quantity | EXP-0004 (pre-fix) | this record (post-fix) |
|---|---|---|
| total advantage over persistence | 8.2 points | 28.0 points |
| shuffle gap (shuffled% - true%) | 3.1 points | 38.5 points |
| zero gap (zeroed% - true%) | 4.7 points | 27.7 points |
| shuffle gap / total advantage | 38% | 138% (shuffled is now worse than persistence itself) |

The old-checkpoint diagnostic row (filled in ahead of the retrain, since it does
not depend on it): true 87.7%, shuffled 94.5%, zeroed 96.5% -- shuffle gap 6.8
points on 12.3 points of total advantage. It is *not* comparable to either the
new row or to EXP-0004's own numbers as a baseline (see caveats above); it is
included only because the task asked for it as a diagnostic.

## What would change the verdict

Little would flip this one: the shuffle-gap did not just widen, it inverted the
sign of the shuffled arm's skill (110.5% > 100% persistence) and the zeroed arm
landed within 0.3 points of persistence, both of which are qualitatively
different regimes from EXP-0004, not marginal movements. What *would* still be
worth doing:
- A seeded repeat (different training seed, different shuffle permutation) to
  put a number on `noise_floor`, since none was measured here or in EXP-0004.
  Cheap: ~35 min GPU for one more retrain, ~1 min for the ablation.
  Given the effect is 5-12x the size of EXP-0004's already-marginal one, it
  would need an unusually large seed-to-seed swing to overturn the verdict,
  but the record should not claim more precision than it has.
- The swept-region number (skipped here, see below) would show whether the
  in-band error is as dramatically affected, or whether some of the 28-point
  whole-image gain is from the ~95%-untouched background.

## Threats

- `untested-dependency`: none cited -- both `grid-convention` and
  `rasteriser-identity` are `fixed`/`fixed` per INVARIANTS.md as of 2026-09-05,
  which is required for this to be a valid T2 record.
- `imprecision`: 200 samples, one checkpoint per arm, unseeded shuffle
  permutation and unseeded training run -- no repeats of either. A single bad
  permutation draw could move the shuffle number by an amount this design
  cannot distinguish from a real effect. Noise floor not measured (see
  `noise_floor` field) -- this is a known gap, not a hidden one.
- `indirectness`: whole-image rms is ~95% pixels nothing could change, which
  compresses every difference (same caveat as EXP-0004). The optional
  swept-region number was not obtained this round (see "What was actually
  run") -- it would sharpen the picture, not overturn it, since the whole-image
  effect here (shuffled worse than persistence) is already a sign flip, not a
  marginal shift a denominator change could hide.
- `provenance`: the old-checkpoint row mixes a pre-fix-trained model with
  post-fix data -- a deliberate distribution-shift diagnostic, not a
  same-code-path comparison, and is never used as a baseline for the verdict.
- Considered and not dismissed: this is still `corl_limited`, a small dataset
  (3714 train samples) and only 100 epochs -- a model this size could underuse
  its action for reasons unrelated to the raster (explanation (b) above), which
  is exactly the alternative this design is built to keep open.

## Unrelated findings

- `configs/dataset/genesis_corl_limited_cubes.yaml` currently sets
  `resolution_scale: 0.5`, but the checked-in `run_config.yaml` for
  `unetfilm_corl_limited_100e` (the run this experiment retrains) recorded
  `resolution_scale: 1.0` at training time -- the config file drifted after
  that run (git blame: commit `80a09a77`, months before this run's timestamp
  metadata). Anyone reconstructing that run from the current `configs/`
  tree alone would silently get a different resolution. Not acted on; retrain
  here used the value in the saved `run_config.yaml`, not the current
  config file.
- `scripts/probes/unet_action_ablation.py`'s val-set size assumption
  (`n = min(200, len(ds))`) is now conservative relative to the dataset: the
  val split is 472 samples, not ~200, so a straightforward change would let a
  future run use more samples for less noise. Not acted on.


## Superseded, 2026-09-06

**This record is superseded by EXP-0021.** Its measurements stand as taken; do
not cite its conclusions. Reason: scored on whole-image rms (~95% untouched pixels), one dataset, no linear-operator anchor. EXP-0021 covers 7 cells on the swept region against persistence, mean-delta, identity and the linear operator, stratified on contact.

Kept rather than deleted because the register's audit trail depends on being
able to see what was believed and why it changed.
