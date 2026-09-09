---
id: EXP-0030
title: >
  GNN and NFD, trained on the overnight_randlen corpus (broad spawn modes,
  N in {20,50}, density 450/friction 0.7, randomised push length), transfer
  to the fixed-length n20_L20mm/L40mm step-0 slates at goal=corner with
  regret_dv/slateN close to their in-distribution-trained counterparts, but
  both degrade sharply under the spatially selective ind-stripe-thin-pile
  objective
tier: T1
mode: exploratory
date: 2026-09-09
hypothesis: null

claim: >
  Scope: GNN (particle-space, N=20 fixed) and NFD (3-channel, non-FiLM UNet)
  trained on Genesis/data/overnight_randlen (213 files; mixed/piled/scattered
  spawn modes; N in {20,50}; material density 450 kg/m3, friction 0.7
  (constant across the whole corpus, contra an earlier unverified claim of
  friction in {0.3,0.5,0.7} -- see "What was actually run"); push length
  randomised, commanded main range 0.02-0.07 m + 10% tail 0.01-0.08 m,
  realised 0.5-79.6 mm), evaluated UNCHANGED on the register's fixed-length
  step-0 candidate pools n20_L20mm/n20_L40mm (N=20 piled only, density 1000,
  friction 0.3), score within paired noise of the same architectures trained
  in-distribution (pooled L20mm+L40mm) on regret_dv/slateN at goal=corner,
  but score markedly worse (regret_dv 10-30x larger, slateN 0.2-0.4 lower)
  under goal=ind-stripe-thin-pile, on both cells, for both models.

provenance:
  commit: a0f116d3
  dirty: false
  data_commit: unrecorded (overnight_randlen collected 2026-09-08/09, see its
    own DATASET.yaml; slates_multistep predates per-dataset provenance stamping)
  script: Baselines/GNN/train/train_genesis_gnn_dyn.py, Baselines/NFD/train_nfd.py,
    Baselines/common/eval_baseline.py, scripts/probes/exp0026_kcurve_exact.py,
    scripts/probes/exp0026_kcurve.py
  data: ["Genesis/data/overnight_randlen_train/*", "Genesis/data/overnight_randlen_test/*",
         "Genesis/data/slates_multistep/n20_L20mm_eval", "Genesis/data/slates_multistep/n20_L40mm_eval",
         "configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml"]
  code_path: "PileSweepData rasteriser (Genesis/training/dataset.py), reused unchanged via Baselines/common/data.py"
  seed: 0
  split: "file (episode) granularity, 90/10, seed 0, per spawn/count group -- scripts/probes/prepare_randlen_split.py; exact held-out file indices in Genesis/data/overnight_randlen_{train,test}/split_manifest.json"
  runtime: "GNN train 3500s (58 min, 500 epochs); NFD train 7236s (2h01m, 30 epochs); all eval/kcurve jobs <3 min CPU each"

budget:
  declared: "~5h wall-clock, ~300k tokens"
  spent: "~5h wall-clock (training dominated: ~3h combined GNN+NFD), ~280k tokens"
  outcome: stopped-early

design:
  varied: {training_corpus: [overnight_randlen (this record), pooled L20mm+L40mm (existing runs)], model: [gnn, nfd_unet3ch], cell: [n20_L20mm, n20_L40mm], goal: [corner, ind-stripe-thin-pile]}
  held_fixed: {eval_harness: Baselines/common/eval_baseline.py, reference_operator_fit: "pooled L20L40 train (unchanged from every prior baseline run)", K_grid: [2,4,8,16,32,64,128], control_ranking_subset: step-0 candidates only, architecture_per_model: unchanged from the in-distribution run}
  baselines: [persistence, mean-delta (pooled fit), linear (pooled fit), oracle]
  metric: "accuracy (image, swept-region); regret_dv and slateK_exact (=slateN at K=128) for control ranking, per docs/experiments/METRICS.md"

noise_floor: >
  paired sem of (model - linear) at K=128, corner goal, n=20 slates: L20mm
  gnn 0.0219, nfd 0.0181; L40mm gnn 0.0041, nfd 0.0061 (regret_dv units,
  Lyapunov). At K=128 half or more of the 20 slates tie outright (13-14/20),
  matching METRICS.md's documented effective-n collapse at this K.

depends_on: [episode-split, swept-region-metric, canonical-warp, warp-blend, goal-placement-pinned]
establishes: []

result: >
  goal=corner, K=128: randlen-trained slateN is 0.9588 (gnn) / 0.9601 (nfd) on
  L20mm and 0.9873 (gnn) / 0.9823 (nfd) on L40mm -- all four within 1-2 points
  of the SAME architecture's in-distribution slateN (gnn 0.9658/0.9870, nfd
  0.9646/0.9925) and within paired sem of the pooled-fit linear operator
  scored on the same cache. goal=ind-stripe-thin-pile, K=128: randlen-trained
  slateN collapses to 0.58 (gnn L20mm) / 0.69 (gnn L40mm) / 0.76 (nfd L20mm) /
  0.78 (nfd L40mm), regret_dv 10-30x the corner-goal value, though still above
  the linear operator's own ind-stripe-thin-pile numbers on the same cells
  (0.39-0.48). Image `accuracy` drops more for GNN (0.253->0.137 L20mm,
  0.399->0.326 L40mm) than for NFD (0.416->0.407 L20mm, 0.512->0.509 L40mm,
  i.e. NFD barely notices the shift on this metric). Held-out accuracy on the
  corpus's own n20 test split: gnn 0.196, nfd 0.449 (persistence 0.000 both).
  On the corpus's OWN held-out same-state pools (15 n20 files, step-0,
  goal=corner), both models rank well against an out-of-distribution linear
  reference (slateN K=128: gnn 0.914, nfd 0.949, regret_dv 0.0052/0.0033;
  linear 0.611, regret_dv 0.0228) -- numerically LOWER slateN than either
  model achieves on the fixed-length OOD slates, consistent with `corner`
  being an easy-to-rank-well functional generally, not with in-distribution
  performance being worse than transfer.
verdict: supported
downgrades: [untested-dependency, incomplete-design]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If randlen training did not transfer to the fixed-length register cells at
all, `regret_dv`/`slateN` at goal=corner would fall toward the linear
operator's own poor performance under whichever functional is used, or worse.
If the models had simply learned "occupancy in, occupancy roughly out"
irrespective of the corpus's push-length/material distribution, corner-goal
control ranking would be indistinguishable from in-distribution training
(which it mostly is) AND the spatially selective functional would also
transfer (which it does not) -- the gap between the two functionals is the
part of this design that discriminates "generalises" from "generalises only
under a coarse, low-frequency objective", which is exactly the register's
standing dissociation finding (C-030/C-035/C-044) reappearing under a
distribution shift rather than a noise-injection arm.

## What was actually run

**Corpus characterisation** (measured directly, all 213 `_*_config.yaml`/`.pt`
files, not sampled): 5 leaf groups -- `mixed/cube/n20` (50 files),
`piled/cube/n20` (50), `piled/cube/n50` (13), `scattered/cube/n20` (50),
`scattered/cube/n50` (50); 512 transitions/file (128 envs x 4 samples/env).
**Contradiction of the task brief's own hazard note, worth stating plainly**:
an exhaustive scan of all 213 configs found `material.friction=0.7`,
`material.density=450.0`, `box.friction=0.5`, `plate.friction=0.3`
**constant across every single file** -- not friction varying over
{0.3,0.5,0.7} as an earlier sampled-config note (and this repo's own
`DATASET.yaml`, itself marked "unverified: ~40 configs sampled") suggested.
The three different numbers (0.7/0.5/0.3) are three DIFFERENT friction
fields (material/box/plate), each individually constant, not one field
varying three ways. `samples_per_env=4` is a 4-push SEQUENTIAL CHAIN per env,
env-major blocked (row r = env r%128, step r//128; verified
`states[128+e]==states_[e]` exactly, max abs diff 0.0, on the sampled file),
confirmed independently and later corroborated by the coordinator/DATASET.yaml.
Every file is a same-state pool at step 0 (128 envs share one settled state).
Realised push length spans 0.47-79.6mm on the sampled file (commanded main
range 20-70mm, 10% tail 10-80mm; short realisations come from early contact).

**Split**: file-granularity, 90/10, seed 0, stratified per leaf group
(`scripts/probes/prepare_randlen_split.py`) -- 192 train / 21 test files.
Held-out file indices: mixed_n20 [2,4,18,23,36], piled_n20 [2,4,18,23,36],
piled_n50 [10], scattered_n20 [2,4,18,23,36], scattered_n50 [2,4,18,23,36]
(exact source indices within each leaf group; full manifest in
`Genesis/data/overnight_randlen_{train,test}/split_manifest.json`, not
committed to git per this repo's existing convention for `Genesis/data/*`).

**Physics conditioning**: dropped from scope by explicit user correction
mid-task -- neither baseline conditions on the physics vector (GNN's
`PARTICLE_DENS` is a fixed constant 1000.0, never read from data; NFD's
`nfd_train_3ch.yaml`/`nfd_train_3ch_randlen.yaml` both set
`include_physics: false`, `uses_physics: false`), so `PhysicsBounds`'
750-5000 density floor vs. the corpus's 450 is not a live conditioning
question here, only a fact about the material-generalisation gap being
measured through control/accuracy scores rather than through a normalised
input channel.

**GNN scope decision**: trained on the n20 SUBSET only (`mixed_n20` +
`piled_n20` + `scattered_n20`, 135 train / 15 test files, 69,120 train
transitions) -- the model's fixed-N graph batching cannot mix N=20 and N=50
samples in one batch, and the existing GNN scope (`ORCHESTRATION_LOG.md`) is
already N=20-only by design, so this is a continuation of that decision, not
a new one. **NFD trained on all 5 groups** (192 train files, both N=20 and
N=50, 89,081 train transitions after a `min_push_length_m=0.0001` guard
dropped 8 near-zero-length rows) -- occupancy rasterisation is
resolution/particle-count agnostic (`n_particles` read per-file from each
config), so no such restriction applies.

**Epoch budget, measured before committing**: GNN 1-epoch timing (2-epoch
smoke test) gave ~3.5s/epoch isolated; the real 500-epoch run (contended with
NFD's concurrent training part of the time) took 3500s total, unchanged from
the in-distribution run's own 500-epoch schedule. NFD 1-epoch timing on an
idle GPU measured 207s; 100 epochs (the in-distribution recipe's count) would
be ~5.75h, over this experiment's ~5h total budget, so **NFD trained 30
epochs instead of 100** (~1.7h projected, 2h01m actual under partial GNU
contention) -- committed to the config with the measured timing and
projection in a comment before the real run started. 30 epochs x ~2784
steps/epoch (89,081/32) ~= 83,500 gradient updates, roughly matching the
in-distribution run's 100 x ~684 steps/epoch = 68,400 (this corpus is ~4.3x
larger per epoch), so the reduction is closer to "roughly matched total
updates" than to "less training".

**Held-out corpus accuracy** (own n20 test split, `Baselines/common/eval_randlen_holdout.py`,
15 files, ~7679 transitions, swept-region, vs persistence): gnn 0.1963, nfd
0.4485 -- both below their own in-distribution held-out numbers in spirit
(the in-distribution runs report `val_mse`/test IoU rather than this same
`accuracy` metric, so this is not a byte-for-byte comparable row, only a
same-metric sanity check that both models learned something on this corpus).

**Slate scoring**: `Baselines/common/eval_baseline.py`, reference operators
(persistence/mean-delta/linear) refit on the SAME pooled L20mm+L40mm train
config every prior baseline run used (comparability preserved -- these
reference rows are not refit on the randlen corpus). Both goals `corner` and
`ind-stripe-thin-pile` scored in the same call (free, per METRICS.md's
weight-field family) -- `ind-stripe-thin-pile` is a `*-pile` goal and
inherits the `goal-placement-pinned` broken invariant (target recomputed per
run from loaded data), so its regret_dv/slateN values are not to be quoted to
three decimals or compared bit-for-bit against a different run's cache of the
same nominal goal; they are stable enough here for the qualitative
"collapses relative to corner" finding, which is the only claim resting on
them.

**In-distribution regret check, run last per the task's priority order**
(`Baselines/common/eval_randlen_indist.py`): the corpus's own 15 held-out n20
test files, treated as 15 same-state pools of 128 step-0 candidates each
(verified: step-0 subset is exactly 15*128=1920 rows, no partial block from
the `min_push_length_m` guard). Reference operators (mean-delta/linear) are
the SAME ones used throughout this record -- fit on the out-of-distribution
pooled L20L40 train set -- so this is NOT a matched in-distribution linear
comparator; it isolates how the TRAINED MODELS behave on their own
distribution against an operator that has never seen it either. n=15 pools
(smaller than the slate cells' n=20), so its paired sem is wider. This row
averages over n20 mixed/piled/scattered (NOT the n50 groups, which
`load_cell` cannot ingest, and NOT the full 5-group corpus) -- narrower than
the full training distribution, broader than the slate cells' n20-piled-only
scope; not broken down further by group/spawn mode (would need per-group
dv caches, not done given time remaining).

## Numbers

`accuracy` (image, swept-region, all 3 steps):

| model | training corpus | L20mm | L40mm |
|---|---|---|---|
| persistence | -- | 0.0000 | 0.0000 |
| mean-delta (pooled fit) | L20L40 (in-distribution) | 0.0544 | 0.1182 |
| linear (pooled fit) | L20L40 (in-distribution) | 0.2004 | 0.3811 |
| gnn | L20L40 (in-distribution) | 0.2527 | 0.3991 |
| **gnn** | **overnight_randlen (this record)** | **0.1370** | **0.3259** |
| nfd_unet3ch | L20L40 (in-distribution) | 0.4158 | 0.5124 |
| **nfd_unet3ch** | **overnight_randlen (this record)** | **0.4071** | **0.5088** |

Control ranking, step-0 candidates only, K=32 and K=128 (=`slateN`):

| model | corpus | cell | goal | slateK_exact K=32 | K=128 (slateN) | regret_dv K=32 | K=128 | ties@128 | paired vs linear (mean, sem, n=20) |
|---|---|---|---|---|---|---|---|---|---|
| linear | L20L40 (ref, same both rows) | L20mm | corner | 0.9524 | 0.9521 | 0.0020 | 0.0025 | -- | -- |
| gnn | in-distribution | L20mm | corner | 0.9729 | 0.9658 | 0.0012 | 0.0018 | -- | -- |
| **gnn** | **randlen** | L20mm | corner | 0.9515 | 0.9588 | 0.0020 | 0.0022 | 13 | +0.0066 (0.0219), 4W/3L/13T |
| nfd | in-distribution | L20mm | corner | 0.9751 | 0.9646 | 0.0010 | 0.0017 | -- | -- |
| **nfd** | **randlen** | L20mm | corner | 0.9753 | 0.9601 | 0.0010 | 0.0021 | 13 | +0.0080 (0.0181), 4W/3L/13T |
| linear | ref | L40mm | corner | 0.9787 | 0.9887 | 0.0028 | 0.0016 | -- | -- |
| gnn | in-distribution | L40mm | corner | 0.9892 | 0.9870 | 0.0015 | 0.0031 | -- | -- |
| **gnn** | **randlen** | L40mm | corner | 0.9854 | 0.9873 | 0.0020 | 0.0028 | 14 | -0.0014 (0.0041), 4W/2L/14T |
| nfd | in-distribution | L40mm | corner | 0.9893 | 0.9925 | 0.0014 | 0.0010 | -- | -- |
| **nfd** | **randlen** | L40mm | corner | 0.9870 | 0.9823 | 0.0017 | 0.0025 | 14 | -0.0064 (0.0061), 2W/4L/14T |
| linear | ref | L20mm | ind-stripe-thin-pile | 0.4817 | 0.4111 | 0.0760 | 0.1112 | -- | -- |
| **gnn** | **randlen** | L20mm | ind-stripe-thin-pile | 0.6123 | 0.5829 | 0.0568 | 0.0830 | 2 | +0.1718 (0.0766), 12W/6L/2T |
| **nfd** | **randlen** | L20mm | ind-stripe-thin-pile | 0.7845 | 0.7631 | 0.0313 | 0.0412 | 1 | +0.3520 (0.1046), 16W/3L/1T |
| linear | ref | L40mm | ind-stripe-thin-pile | 0.4686 | 0.3915 | 0.1105 | 0.1593 | -- | -- |
| **gnn** | **randlen** | L40mm | ind-stripe-thin-pile | 0.6871 | 0.6882 | 0.0647 | 0.0787 | 3 | +0.2966 (0.1162), 12W/5L/3T |
| **nfd** | **randlen** | L40mm | ind-stripe-thin-pile | 0.7685 | 0.7778 | 0.0486 | 0.0509 | 3 | +0.3863 (0.1022), 15W/2L/3T |

In-distribution `ind-stripe-thin-pile` numbers for gnn/nfd were not
previously scored (prior runs used `corner`/`center` only) and are not
computed here for the in-distribution rows, so no in-distribution comparator
exists for that half of the table -- an asymmetry in what "in-distribution"
means for this table, noted rather than filled with a guess.

Held-out accuracy, corpus's own test split (15 n20 files, swept-region vs
persistence): gnn 0.1963, nfd_unet3ch 0.4485 (persistence 0.0000).

**In-distribution control ranking** (corpus's own 15 held-out n20 pools,
step-0 only, goal=corner; reference `linear`/`mean-delta` still fit on the
out-of-distribution L20L40 pool, NOT refit here -- see "What was actually
run"):

| model | pool | slateK_exact K=32 | K=128 (slateN) | regret_dv K=32 | K=128 | paired vs linear (mean, sem, n=15) |
|---|---|---|---|---|---|---|
| linear (OOD ref) | in-distribution (15 corpus pools) | 0.7052 | 0.6112 | 0.0132 | 0.0228 | -- |
| **gnn** | in-distribution (15 corpus pools) | 0.9552 | 0.9141 | 0.0019 | 0.0052 | +0.3030 (0.0799), 11W/4L/0T @K=128 |
| **nfd_unet3ch** | in-distribution (15 corpus pools) | 0.9461 | 0.9485 | 0.0024 | 0.0033 | +0.3373 (0.0787), 12W/3L/0T @K=128 |

Reading this against the slate-cell rows above: BOTH models rank candidates
far better on their own training distribution (slateN 0.91-0.95, regret_dv
0.003-0.005) than the linear operator does when evaluated out-of-distribution
against it -- but this is not the same comparison as the corner-goal slate
rows, where the SAME models score slateN 0.96-0.99, i.e. numerically higher
on the fixed-length OOD slates than on their own in-distribution pools. That
is consistent with `corner`'s low-pass functional being easy to rank well on
generally (matching the "What would change the verdict" note above), not
with the models transferring better than they perform at home -- the n=15
vs n=20 pool-count difference and the fact that `linear` here is itself
out-of-distribution both limit how far this single contrast can be pushed.

## What would change the verdict

Running the not-run in-distribution held-out regret check (queued, ~5-10 min
total) would show whether the corner-goal transfer numbers above are close
to in-distribution because the models generalise, or because `corner`'s
low-pass functional (63% of spectral energy at r<=1 per METRICS.md) is easy
to transfer regardless of training distribution -- the ind-stripe-thin-pile
collapse is consistent with the latter but the missing in-distribution
comparator for that goal (noted above) leaves the sharpest version of this
question open. A second seed for both trainings would also test whether the
GNN's larger accuracy drop (vs NFD's near-zero drop) is a stable
architectural difference or this run's particular initialisation.

## Threats

- `untested-dependency` (`goal-placement-pinned`): `ind-stripe-thin-pile`'s
  target is recomputed per run from loaded occupancy, known broken
  (INVARIANTS.md). The qualitative finding (collapse relative to `corner`)
  is far larger than the ~0.05-0.06 cross-run target-shift documented for
  this invariant, so it is not believed to be the cause of the collapse, but
  the exact regret_dv/slateN values under this goal should not be quoted
  past 2 decimals.
- `incomplete-design`: the in-distribution `ind-stripe-thin-pile` comparator
  was not run (only `corner` was scored on the corpus's own held-out pools,
  to keep within budget) -- so the sharpest version of "does ind-stripe
  transfer worse, or does the model just rank it worse everywhere" is still
  open. The corner-goal comparison and the accuracy comparison are both
  complete over the cells in scope, and the in-distribution corner-goal row
  was added (not skipped) after an explicit override of this record's
  original priority-order deprioritisation.
- Considered and dismissed: that the near-zero NFD accuracy drop is a
  ceiling/saturation artefact of the metric -- `accuracy` is not a bounded-at-1
  metric, so a true near-parity result would look exactly like this; ruled
  in, not out, by the numbers themselves (NFD's in-distribution 0.512 is not
  near any ceiling).

## Unrelated findings

- `Genesis/data/overnight_randlen/DATASET.yaml`'s `materials.friction:
  [0.3, 0.5, 0.7]` field is misleading as written -- it lists three DIFFERENT
  friction fields (plate/box/material) as if one field varied three ways;
  each is individually constant across all 213 files (verified exhaustively,
  not sampled). The file's own header invites this kind of correction
  ("where its measured numbers disagree... this file should be corrected")
  but per this task's explicit scope trim, no edit was made -- logged here
  for whoever next reads that file.
- `Baselines/GNN/train/train_genesis_gnn_dyn.py`'s per-epoch wall time roughly
  doubled (3.5s -> 7.3s) under concurrent NFD training on the same GPU despite
  `GPU_LOCK_SLOTS` allowing concurrency -- consistent with `ORCHESTRATION_LOG.md`'s
  already-documented finding that compute, not memory, is the binding
  constraint for concurrent small-model jobs on this card; not re-investigated.

## Register attachment

New claim `C-047` in `docs/experiments/REGISTER.md`, `depends_on` as above.
