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
  commit: 652fce1f
  dirty: false
  data_commit: unrecorded (overnight_randlen collected 2026-09-08/09, see its
    own DATASET.yaml; slates_multistep predates per-dataset provenance stamping)
  script: Baselines/GNN/train/train_genesis_gnn_dyn.py, Baselines/NFD/train_nfd.py,
    Baselines/common/eval_baseline.py, Baselines/common/eval_randlen_indist.py,
    Baselines/common/eval_randlen_holdout.py, scripts/probes/exp0026_kcurve_exact.py,
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
  varied: {training_corpus: [overnight_randlen (this record), pooled L20mm+L40mm (existing runs)], model: [gnn, nfd_unet3ch], cell: [n20_L20mm, n20_L40mm], goal: [corner, ind-stripe-thin-pile], "spawn_mode (follow-up, in-distribution held-out pools only, n20 only)": [mixed, piled, scattered]}
  held_fixed: {eval_harness: Baselines/common/eval_baseline.py, reference_operator_fit: "pooled L20L40 train (unchanged from every prior baseline run)", K_grid: [2,4,8,16,32,64,128], control_ranking_subset: step-0 candidates only, architecture_per_model: unchanged from the in-distribution run}
  baselines: [persistence, mean-delta (pooled fit), linear (pooled fit), oracle]
  metric: "accuracy (image, swept-region); regret_dv and slateK_exact (=slateN at K=128) for control ranking, per docs/experiments/METRICS.md"

noise_floor: >
  paired sem of (model - linear) at K=128, corner goal, n=20 slates: L20mm
  gnn 0.0219, nfd 0.0181; L40mm gnn 0.0041, nfd 0.0061 (regret_dv units,
  Lyapunov). At K=128 half or more of the 20 slates tie outright (13-14/20),
  matching METRICS.md's documented effective-n collapse at this K.

  Follow-up (spawn-mode-stratified in-distribution check, n=5 pools/stratum):
  paired sem of (model - linear) slateK_exact at K=128 ranges 0.069-0.163
  under corner and 0.125-0.305 under ind-stripe-thin-pile across the 3
  strata x 2 models -- roughly 1.7x the n=15 pooled corner row's own sem
  (sqrt(15/5)=1.73), as expected from n alone, not from a change in effect
  size. One cell (gnn, piled, ind-stripe-thin-pile) has mean +0.308 against
  sem 0.305 (t=1.01, 4/5 wins, 0 ties) -- inside its own noise floor, and
  reported below as such rather than as an established gnn-beats-linear
  result for that cell.

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

  Follow-up (2026-09-10), closing the `incomplete-design` gap: the
  in-distribution ind-stripe-thin-pile comparator IS now scored, stratified
  by spawn mode (mixed/piled/scattered, n20, 5 held-out pools each, own
  pile-relative target per stratum). None of the 6 strata-goal dv_true
  distributions is degenerate (sd 0.022-0.046, 32-42% helpful, all
  comparable to corner's own 35-39%). At K=128: corner-goal slateN is stable
  across strata for gnn (0.96/0.94/0.84 mixed/piled/scattered) and even more
  so for nfd (0.89/0.99/0.97). Under ind-stripe-thin-pile, gnn's own
  held-out-pool slateN is 0.87 (mixed) / 0.53 (piled) / 0.86 (scattered) --
  i.e. gnn's collapse relative to corner is almost entirely a PILED-stratum
  effect (ratio ind/corner 0.91 mixed, 0.56 piled, 1.03 scattered), while nfd
  stays flat across all three (0.92/0.92/0.98, ratio 1.04/0.93/1.01). This
  lines up with the register's fixed-length slate cells being piled-spawn
  geometry (N=20 piled only) -- consistent with, though not a proof of, the
  piled stratum specifically being what makes ind-stripe-thin-pile hard for
  gnn on those cells. Image accuracy shows the same split: gnn's own held-out
  accuracy is 0.25 (mixed) vs 0.16 (piled/scattered), nfd's is flat at
  0.44-0.47 across all three. The gnn-piled-ind-stripe cell itself is the
  weakest signal in the whole table (paired vs linear: mean +0.31, sem 0.31,
  t=1.01) -- inside noise, not read as gnn beating linear there. See the
  "Follow-up" section below for the full per-stratum table, the K-curve
  files, and the between-model small-n caveat this design was built to
  surface rather than paper over.
verdict: supported
downgrades: [untested-dependency]
grade: moderate
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
  **RESOLVED 2026-09-10** -- see "Follow-up" below: the comparator is now
  scored, stratified by spawn mode. `incomplete-design` is retired from this
  record's downgrades (n50 remains excluded from the in-distribution check,
  same as before, but that exclusion was never what this downgrade named --
  see the follow-up section for exactly what is and is not now closed).
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
- **(follow-up, 2026-09-10)** `Baselines/common/eval_randlen_indist.py` called
  `lyapunov_weights(..., goal, "cpu")` with no `pile_center` argument --
  correct for `corner` (which does not need one) but a hard `ValueError` for
  any `*-pile` goal, so this script could never have scored
  `ind-stripe-thin-pile` before today; fixed to compute `pile_center` from
  the loaded step-0 `occ0` the same way `Baselines/common/eval_baseline.py`
  already does (commit `652fce1f`). Not a bug in any existing number -- the
  old code path simply could not run under this goal at all -- so nothing
  upstream needs invalidating, but it is the reason this gap could not have
  been closed by re-running the old script with a different flag.
- **(follow-up, 2026-09-10)** A second, sharper hazard in the same script:
  `build_predictor()` for both `Baselines/GNN/predictor.py` and
  `Baselines/NFD/predictor.py` reads its checkpoint path from an env var
  (`GNN_CKPT`/`NFD_CKPT`) that **defaults to the IN-DISTRIBUTION (pooled
  L20L40) checkpoint**, not the randlen-trained one, and the module's own
  usage docstring did not set it. A first smoke-test run of this follow-up
  (mixed stratum, gnn) silently scored the wrong checkpoint under the
  `_randlen_` tag -- caught only because the printed checkpoint
  path/epoch (`.../ckpt_best.pth`, epoch=400) did not match every prior
  `_randlen_indist` run in this same record (`.../gnn_randlen/ckpt_best.pth`,
  epoch=224); re-run correctly before anything from it was used. The
  docstring now states this explicitly (see the script). No number in this
  record besides that one discarded smoke test was affected.

## Follow-up (2026-09-10): spawn-mode-stratified in-distribution `ind-stripe-thin-pile` check

**Closes the `incomplete-design` gap** named above: the in-distribution
`ind-stripe-thin-pile` comparator was not run at all in the original session
(only `corner` was scored on the corpus's own held-out pools). This section
runs it, and -- per the task that requested this follow-up -- stratifies by
spawn mode (mixed/piled/scattered) rather than pooling, for two reasons
stated up front rather than discovered after the fact:

1. `ind-stripe-thin-pile` is **pile-relative**: its target (a 4-px indicator
   stripe) is centred on a pile centroid computed from whatever data is
   loaded (`control_utility_test.py::pile_centroid_and_support`). Pooling
   mixed/piled/scattered spawns into one 15-pool set (as the existing
   `genesis_overnight_randlen_test_n20.yaml` does) places the stripe at the
   AVERAGE centroid of all three spawn modes' piles, which may sit well for
   none of them individually. Scoring each spawn mode separately gives each
   its own target, on its own piles.
2. The stratification (does the indicator goal behave differently by spawn
   mode) is the entire point of the request, not a side effect of design.

**What was run.** Three new test configs, one held-out spawn group each
(`configs/dataset/genesis_overnight_randlen_test_n20_{mixed,piled,scattered}.yaml`,
5 held-out files/pools each per
`Genesis/data/overnight_randlen_test/split_manifest.json` -- n50 still
excluded, unchanged from the original in-distribution check, for the same
`load_cell` N=20 hard-assumption reason, not a new hole). `Baselines/common/
eval_randlen_indist.py` fixed to support `*-pile` goals (see "Unrelated
findings") and re-run 6 times (2 models x 3 strata) with `--goals
corner,ind-stripe-thin-pile` so every stratum gets its own within-run
`corner` reference rather than being read against the original session's
pooled `corner` number. Reference operators (persistence/mean-delta/linear)
are refit each run on the SAME out-of-distribution pooled-train config used
throughout this record (`genesis_overnight_randlen_train_n20.yaml`, 135
files, 69,115 transitions) -- unchanged, so the `linear`/`mean-delta` rows
below are directly comparable across strata. `GNN_CKPT`/`NFD_CKPT` set
explicitly to the randlen-trained checkpoints (see "Unrelated findings").
Then `scripts/probes/exp0026_kcurve_exact.py` and `exp0026_kcurve.py` on
each of the 6 resulting caches, both goals (24 invocations), plus
`Baselines/common/eval_randlen_holdout.py` per stratum per model for image
`accuracy` (6 invocations, goal-independent, all push-steps, vs
persistence). All 36 runs went through `scripts/run_probe.py --exp
EXP-0030`; all exited 0; commit `652fce1f`, dirty=false throughout. Compute
cost: ~15 min wall-clock total (6 eval runs ~230s each, staggered 3-at-a-time;
24 kcurve invocations 1-18s each; 6 accuracy invocations ~10s each), all CPU,
against this follow-up's own declared budget of ~2h/~200k tokens.

### Degeneracy screen (checked before any metric was read, per C-040/METRICS.md)

| stratum | goal | dv_true mean | sd | helpful % | pile centroid (row,col) |
|---|---|---|---|---|---|
| mixed | corner | +0.01197 | 0.03778 | 35% | (30.28, 27.32) |
| mixed | ind-stripe-thin-pile | +0.01274 | 0.04622 | 32% | (30.28, 27.32) |
| piled | corner | +0.00333 | 0.02233 | 36% | (31.50, 32.51) |
| piled | ind-stripe-thin-pile | -0.00229 | 0.02543 | 42% | (31.50, 32.51) |
| scattered | corner | +0.00428 | 0.02321 | 39% | (29.66, 30.17) |
| scattered | ind-stripe-thin-pile | -0.00681 | 0.02832 | 40% | (29.66, 30.17) |

None of the 6 stratum-goal cells is degenerate in the C-040 sense (a
centred target with `dV` identically 0): every sd is comparable to
`corner`'s own, and helpful% sits at 32-42% throughout, nowhere near the
0%/100% saturation that would flag a target never overlapping the pile.
All three strata's pile centroids fall within about 2 px of each other and
of the 64x64 grid's own centre (32, 32) -- these are same-state pools,
settled near the tray centre by construction -- so the per-stratum stripe
placements differ only slightly (row 29.7-31.5), a real but small
difference against the `goal-placement-pinned` invariant's own documented
~0.05-0.06 slateK_exact shift for a comparably-sized target move (not the
same cells, cited only to size the effect): read as "each stratum gets its
own target," not as "the strata differ mainly because their targets moved."

### Numbers, step-0 candidates only, K=128 (`slateN`)

**corner:**

| stratum | model | slateN | regret_dv | vs linear: mean (sem, n=5) | wins/ties |
|---|---|---|---|---|---|
| mixed | persistence | 0.0546 | 0.0876 | -- | -- |
| mixed | mean-delta | 0.3600 | 0.0433 | -- | -- |
| mixed | linear | 0.6989 | 0.0192 | -- | -- |
| mixed | **gnn** | **0.9616** | **0.0027** | +0.2628 (0.0722) | 5/5, 0 |
| mixed | **nfd** | **0.8869** | **0.0078** | +0.1881 (0.0694) | 4/5, 0 |
| piled | persistence | 0.2793 | 0.0594 | -- | -- |
| piled | mean-delta | 0.3389 | 0.0411 | -- | -- |
| piled | linear | 0.4836 | 0.0316 | -- | -- |
| piled | **gnn** | **0.9429** | **0.0045** | +0.4593 (0.1623) | 4/5, 0 |
| piled | **nfd** | **0.9904** | **0.0005** | +0.5068 (0.1545) | 4/5, 0 |
| scattered | persistence | -0.1611 | 0.0441 | -- | -- |
| scattered | mean-delta | 0.5384 | 0.0233 | -- | -- |
| scattered | linear | 0.6510 | 0.0177 | -- | -- |
| scattered | **gnn** | **0.8379** | **0.0083** | +0.1868 (0.1601) | 2/5, 0 |
| scattered | **nfd** | **0.9680** | **0.0016** | +0.3170 (0.1533) | 4/5, 0 |

**ind-stripe-thin-pile:**

| stratum | model | slateN | regret_dv | vs linear: mean (sem, n=5) | wins/ties |
|---|---|---|---|---|---|
| mixed | persistence | 0.0504 | 0.1449 | -- | -- |
| mixed | mean-delta | 0.0900 | 0.1248 | -- | -- |
| mixed | linear | 0.5333 | 0.0623 | -- | -- |
| mixed | **gnn** | **0.8727** | **0.0168** | +0.3394 (0.1396) | 4/5, 0 |
| mixed | **nfd** | **0.9188** | **0.0110** | +0.3855 (0.1458) | 5/5, 0 |
| piled | persistence | -0.0060 | 0.1039 | -- | -- |
| piled | mean-delta | 0.0173 | 0.0988 | -- | -- |
| piled | linear | 0.2195 | 0.0764 | -- | -- |
| piled | **gnn** | **0.5274** | **0.0468** | +0.3079 (0.3048, t=1.01) | 4/5, 0 |
| piled | **nfd** | **0.9227** | **0.0090** | +0.7032 (0.2441) | 4/5, 0 |
| scattered | persistence | -0.1500 | 0.1335 | -- | -- |
| scattered | mean-delta | 0.0335 | 0.1194 | -- | -- |
| scattered | linear | 0.4944 | 0.0715 | -- | -- |
| scattered | **gnn** | **0.8604** | **0.0139** | +0.3660 (0.1853) | 3/5, 1 |
| scattered | **nfd** | **0.9773** | **0.0027** | +0.4829 (0.1384) | 5/5, 0 |

Image `accuracy` (swept-region vs persistence, all push-steps, goal-independent,
own held-out pools per stratum):

| stratum | gnn | nfd_unet3ch |
|---|---|---|
| mixed | 0.2545 | 0.4653 |
| piled | 0.1592 | 0.4370 |
| scattered | 0.1631 | 0.4398 |

Full K-curves (K=2..128) for both metrics, both goals, all 6 caches: `Baselines/{GNN,NFD}/runs/{gnn,nfd}_randlen_indist_{mixed,piled,scattered}_kcurve_{exact,sampled}_{corner,indstripethinpile}.json`.

### Reading it: within-model, between-stratum (the better-powered comparison)

This is what the design was built to answer, and it separates cleanly from
the noisier between-model reading below. Under `corner`, gnn's slateN is
fairly flat across spawn mode (0.96 / 0.94 / 0.84, mixed/piled/scattered)
and nfd's more so (0.89 / 0.99 / 0.97). Under `ind-stripe-thin-pile`, gnn's
slateN is **0.87 (mixed) / 0.53 (piled) / 0.86 (scattered)** -- i.e. gnn's
own held-out-pool performance collapses specifically on the PILED stratum,
not uniformly across spawn modes (ind/corner ratio: 0.91 mixed, 0.56 piled,
1.03 scattered -- scattered is not degraded at all by this measure). nfd's
ind-stripe slateN is 0.92 / 0.92 / 0.98 -- flat, ratio 1.04/0.93/1.01, no
stratum stands out. The same split shows in image `accuracy`: gnn's is 0.25
(mixed) vs 0.16 (piled AND scattered -- so accuracy does not single out
piled the way ind-stripe-thin-pile's control ranking does; the two metrics
disagree about WHICH stratum is hardest, consistent with this register's
standing accuracy/control dissociation, C-035/C-030, appearing again here
along a new axis), while nfd's is flat at 0.44-0.47 throughout.

The register's fixed-length slate cells that the original (pooled) part of
this record scored are **piled-spawn geometry** (`n20_L20mm`/`n20_L40mm`,
"N=20 piled only" per this record's own claim scope). gnn's collapse there
under `ind-stripe-thin-pile` (slateN 0.58/0.69, vs corner's 0.96/0.99) is
now seen to line up with the ONE stratum, of three tested in-distribution,
where gnn's own held-out performance under the same goal also collapses
hardest (0.53 piled vs 0.86-0.87 elsewhere). This is consistent with, and
offers a candidate mechanism for, the original collapse -- **it is not a
proof**: the slate cells differ from these held-out pools in push length
(fixed 20mm/40mm vs randomised), density/friction (1000/0.3 vs 450/0.7), and
data source entirely; "piled-like geometry is what ind-stripe-thin-pile
finds hard for gnn" is the reading that best fits both halves of the
record, not a re-derivation of one from the other.

### Reading it: between-model (the underpowered comparison -- read the sem, not the gap)

At n=5 pools/stratum, the between-model contrast is markedly less powered
than the n=15/n=20 comparisons elsewhere in this record, and the task this
follow-up answers to is explicit that a null here means little. Every
model-vs-linear row above ties 0-1 times out of 5 (the metric does not
collapse into a tie-heavy regime at this n the way K=128 does at n=20), but
sems are correspondingly wide (0.07-0.30). Read plainly: nfd's mean margin
over linear is numerically larger than gnn's in 5 of 6 stratum x goal cells
(the exception: scattered corner, gnn +0.19 vs nfd +0.32, both well inside
each other's sem) -- but only the piled/ind-stripe-thin-pile cell has a
paired difference (gnn +0.31 sem 0.30 vs nfd +0.70 sem 0.24) where the two
models' own sems do not overlap much; every other cell's gnn/nfd sems
overlap enough that "nfd beats gnn here" is not established by this data,
only suggested by it. The one cell most likely to invite over-reading --
gnn/piled/ind-stripe-thin-pile itself (mean +0.31, sem 0.31, t=1.01) -- is
flagged above as inside its own noise floor: gnn is not shown to beat
linear there at all, let alone to be beaten by nfd.

### What this closes, and what it does not

Closes: the in-distribution `ind-stripe-thin-pile` comparator now exists,
for all three n20 spawn modes separately -- the specific gap this record's
`incomplete-design` downgrade named. `incomplete-design` is retired from
this record's frontmatter; `untested-dependency` remains (`goal-placement-pinned`
is still broken, and every `*-pile` row here inherits it exactly as the
original ones did).

Does not close: n50 groups remain excluded from the in-distribution check
(`load_cell`'s N=20 hard-assumption, unchanged, not attempted here -- this
was never what the retired downgrade named, so its removal does not imply
n50 is now covered). The between-model comparison at n=5 is explicitly
underpowered (see above) and this follow-up does not claim to settle it. A
true test of the "piled geometry explains gnn's slate-cell collapse"
reading would need `ind-stripe-thin-pile` scored in-distribution on
piled-spawn data at the SLATE CELLS' OWN push-length/density/friction
settings, which do not exist in this corpus -- noted, not run.

## Register attachment

New claim `C-047` in `docs/experiments/REGISTER.md`, `depends_on` as above.
Updated 2026-09-10 with the spawn-mode-stratified in-distribution finding
above (same claim id, same record, no new EXP number).
