---
id: EXP-0059
title: >
  R0 harness/ceiling rung for the retrieval transition model: an extended eval harness scores
  ANY model (occupancy-in or particle-in) on DS-0009 with 13-goal + 8-goal-tough slateN and
  blurred-accuracy diagnostics; naive retrieval_1nn (accuracy_1 0.172, slateN 0.413) sits well
  below every EXP-0053 model (best: narrow NFD slateN 0.774, wide narrow NFD 0.803); a Genesis
  chaos-floor probe (0/0.5/1/2mm perturbation of the recorded start state, same recorded action
  re-executed and scored against the same recorded outcome) shows accuracy_1 collapses steeply
  with even sub-mm state error (0.703 exact -> 0.428 at 0.5mm -> 0.303 at 1mm -> 0.111 at 2mm),
  i.e. the swept-region accuracy metric is far more position-sensitive than trained-model gaps
  suggest -- not itself a "model ceiling" since DS-0009 carries no state measurement noise, but a
  sharp characterisation of what the metric costs any positional error, model or perturbation
  alike
tier: T1
mode: exploratory
date: 2026-09-28
hypothesis: null
claim: >
  On DS-0009 (clean narrow-domain test set), (1) an extended harness reusing eval_narrow.py /
  eval_retrieval.py's own metric code can score occupancy-in/out and particle-in/out models on
  the same accuracy/rollout/slateN/blurred-accuracy/mm scale, including an 8-goal "tough" slateN
  subset (letter_O/T/S/X/L/I, two_squares, quadrant_0) alongside the original EXP-0053 13-goal
  set; (2) the naive retrieval_1nn predictor remains well below every EXP-0053 occupancy model
  on this extended harness too; (3) perturbing DS-0009's recorded start states by 0.5-2mm before
  re-simulating the SAME recorded action, scored against the SAME recorded outcome, quantifies a
  state-uncertainty ceiling distinct from resimulation noise (which EXP-0024 already showed is
  negligible under this exact collection mechanism).
provenance:
  commit: 3bae8cd7
  dirty: true
  script: "code/eval_extended.py (new, this record; reuses experiments/EXP-0053-*/code/
    eval_narrow.py's acc/swept_region/GOALS/goal_mask verbatim, and model/retrieval/predictor.py's
    PersistencePredictor/NearestTransitionPredictor unchanged), code/chaos_floor.py (new, this
    record; reuses Genesis/chain_collection.py's exact re-simulation seam:
    sim.set_particle_state + sim.execute_action + sim.update_material_state, TRAINING_PHYSICS,
    no re-settle)"
  data: ["DS-0009 (test_chains: 4 chunks x 256 rows = 1,024 rows; test_pools: 32 pools x 64
         pushes) -- scoring corpus for both tasks 1-2", "DS-0009 test_chains, 64-row sample
         (32 scatter + 32 clump) -- chaos-floor re-simulation input"]
  code_path: "occ path: simple_mpc.adapters.make_occ_adapter -> predict_step(occ,act) -> occ,
    scored exactly as eval_narrow.py. particle path: model.retrieval.predictor.
    {PersistencePredictor,NearestTransitionPredictor}.predict_particles(states0,p_start,p_stop)
    -> states1, rasterised with the same occ_from_particles. chaos floor: GenesisOracleEnv's
    underlying sim (simple_mpc.genesis_oracle) via the chain_collection.py seam, NOT
    rollout_candidates/snapshot_restore (that seam's known state-leak,
    genesis-snapshot-restore-repeat-determinism, does not apply here)."
  seed: "eval harness: none (deterministic given saved checkpoints/bank). chaos floor: numpy
    default_rng(seed=0) for the DS-0009 row sample and for the perturbation noise; Genesis's own
    internal RNG for physics"
  split: "test-only; DS-0009 is EXP-0053/EXP-0059's held-out test set, never trained on"
  data_commit: not applicable
  runs: [eval_extended, chaos_floor, perturbed_sim_zoo, diagnostics_r2]
  runtime: "eval_extended: ~4 min GPU wall (5 occ models ~21s each + retrieval_1nn ~115s CPU).
    chaos_floor: ~1 min env build + ~10-15s per perturbation level x 4 levels"
budget:
  declared: "~75 min wall-clock (task-level, per brief)"
  spent: "~75 min wall-clock"
  outcome: within
design:
  varied:
    model: "persistence, nfd_3ch_narrow_l20, nfd_3ch_narrow_l20_wide, linear_narrow_l20_res64,
      nfd_residual_worldframe_noaug_ep43 (best broad NFD per EXP-0053), retrieval_1nn"
    chaos_perturbation_mm: [0, 0.5, 1, 2]
  held_fixed:
    goal_sets: "EXP-0053's 13-goal set AND the EXP-0055/0057 8-goal tough set (letter_O, letter_T,
      letter_S, letter_X, letter_L, letter_I, two_squares, quadrant_0), both lyapunov"
    blur_sigmas: "0 (raw), 1, 2 px Gaussian, applied identically to pred/truth/prev before
      scoring accuracy_1"
    chaos_sample: "64 DS-0009 test_chains rows (32 scatter + 32 clump), TRAINING_PHYSICS, yaw
      jitter std 1 deg at every perturbation level > 0"
  baselines: "persistence (accuracy 0 by construction, slateN = random-pick floor); chaos floor's
    0mm level is its own do-nothing/sanity baseline (re-simulating the exact recorded state+action
    should reproduce the recorded outcome closely, per EXP-0024)"
  metric: "accuracy (METRICS.md), slateN (METRICS.md, both 13-goal and 8-goal-tough lyapunov
    pools), plus two new diagnostics local to this record: blurred accuracy_1 (Gaussian sigma 1/2
    px on pred+truth+prev, otherwise METRICS.md's `accuracy` formula unchanged) and moved-cube mm
    position error (direct per-cube correspondence, sim never reorders particles; 'moved' =
    true push-frame displacement > 1mm, matching model/retrieval/bank.py's own
    DEFAULT_MOVED_THRESHOLD)"
noise_floor: "EXP-0036 training-seed sd (reused, not remeasured): accuracy ~0.003, slateN
  ~0.02-0.04, 32 test pools. EXP-0024's resimulation-noise floor (reused, not remeasured):
  within/between dv variance ratio <=5.3e-5 under exact-snapshot restore, on scalar lyapunov dv
  -- negligible next to the 0.29-0.59 accuracy_1 drop this record measures from 0.5-2mm state
  uncertainty on IMAGE accuracy, so the two ceilings are on different scales/quantities (scalar
  dv vs image accuracy) and not in tension."
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  (1) Harness: code/eval_extended.py reproduces EXP-0053's own accuracy_1/rollout/slateN numbers
  exactly for all 5 occ models and eval_retrieval.py's numbers exactly for retrieval_1nn
  (cross-check, not a new claim), and extends every one of them with slateN_tough (8-goal) and
  blurred accuracy_1 at sigma 1/2px. slateN_tough tracks slateN closely (within 0.02-0.03 of the
  13-goal value for every model, same rank order) -- the tough 8-goal subset does not flip any
  ranking here. (2) retrieval_1nn stays far below every occ model on every metric, including the
  cleanliness diagnostics meant to test the "sharp-but-slightly-off is punished" hypothesis: its
  blur1/blur2 gains (+0.126/+0.272 over raw) are LARGER than any occ model's (largest occ gain:
  linear_narrow_l20_res64 +0.130/+0.205), consistent with retrieval_1nn's errors being
  small-offset/misplacement rather than wrong-shape, but the gap to the best occ model does not
  close at either blur level (retrieval_1nn blur2 0.444 vs nfd_3ch_narrow_l20 blur2 0.704).
  retrieval_1nn's rollout accuracy collapses fast under its own recursive predictions (0.195 ->
  0.091 -> 0.036 -> 0.009 over 4 steps), far faster than any occ model (occ models stay above
  0.32 at step 4) -- compounding retrieval error is a distinct failure mode from 1-step accuracy.
  Its moved-cube position error is 5.77mm mean over cubes with true push-frame displacement
  >1mm. (3) Chaos floor: accuracy_1 against the ORIGINAL recorded DS-0009 outcome is 0.703 at
  0mm perturbation (mm_mean 0.145mm -- a small genuine resimulation gap, physically near-exact),
  and falls steeply with state uncertainty: 0.428 at 0.5mm, 0.303 at 1mm, 0.111 at 2mm. At 0.5mm
  the floor already sits at/below `linear_narrow_l20_res64` (0.452) and near
  `nfd_residual_worldframe_noaug_ep43` (0.466); at 1mm it is below every trained occ model. Since
  DS-0009's recorded states carry no measurement noise, this is not literally "the ceiling models
  face today" but it does show the swept-region `accuracy` metric is highly sensitive to small
  positional error generally -- consistent with the brief's blur-recovers-crude-1NN observation
  (blur1/blur2 recover much of the loss at every level, e.g. 1mm: 0.303 -> 0.573 -> 0.739).
  **Clean-data v2 rung addendum (2026-09-28, DS-0015/16/17):** every model retrained/refit from
  scratch on the ISS-010-fix corpus (0% illegal, vs 44-56% before) confirms the fixed retrieval
  transfer's DS-0009 finding on an independent, cleaner train/val/test split, with a proper paired
  pool-bootstrap CI this time: `retrieval_k5_cube_median_v2` test `slateN_tough` 0.792
  [0.755,0.829] beats `linear_narrow_l20_v2_res64` 0.626 [0.535,0.718] decisively (paired delta
  +0.167, CI [+0.090,+0.244], excludes 0) and edges out `nfd_3ch_narrow_l20_v2` 0.731
  [0.646,0.804] (paired delta +0.062, CI [-0.010,+0.137], 95.4% of bootstrap draws favour
  retrieval -- suggestive, not fully resolved at this n=32-pool power). See the "Clean-data v2
  rung" section below for the full table, both k configs, and rollout/mm detail.
verdict: supported
downgrades: [imprecision, incomplete-design]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

The harness reuses the exact scoring code every existing register row for this corpus was
produced with (`eval_narrow.py`'s `acc`, `swept_region`, `GOALS`), so its numbers for the 5 occ
models are a reproduction check, not a new measurement -- any divergence would mean the
extension broke something, and there was none (see Numbers). The chaos floor answers a
different question from EXP-0024 on purpose: EXP-0024 asked whether resimulating the SAME
state gives the SAME `dv`, and found yes (ratio <=5.3e-5). This record asks what happens when
the state fed in is SLIGHTLY WRONG -- the actual condition any real dynamics model (retrieval or
otherwise) is in, since it never sees the true continuous cube positions, only a rasterised or
otherwise degraded observation. If the perturbed-state accuracy sits far below every trained
model's reported accuracy, state uncertainty is not yet what's limiting them; if it sits close
or below, it is.

## What was actually run

1. **`code/eval_extended.py`** -- for each occ model (`persistence` special-cased identity,
   `nfd_3ch_narrow_l20`, `nfd_3ch_narrow_l20_wide`, `linear_narrow_l20_res64`,
   `nfd_residual_worldframe_noaug_ep43`) and each particle model (`persistence`,
   `retrieval_1nn`), computed on DS-0009: `accuracy_1` (+scatter/clump split), `rollout_accuracy_
   {1..4}`, `slateN` and `slateN_tough` (13-goal / 8-goal-tough, both lyapunov, soft
   ground-truth), `accuracy_1_blur{1,2}` (Gaussian blur applied to pred/truth/prev before
   scoring), and (particle models only) moved-cube mm position error. One caveat found while
   running it, not a bug in anyone else's code: the harness's own JSON uses the model name as
   its dict key, and `persistence` is registered on BOTH the occ path and the particle path with
   identical semantics (predict no change) -- the particle-path `persistence` entry was skipped
   as "already scored" by the occ-path one. Harmless here (the two are mathematically identical,
   `predict_particles` returning `states0.clone()` rasterises to the same occupancy the occ
   adapter's identity `predict_step` returns), but it means the JSON has no moved-cube-mm number
   for the persistence baseline. Not fixed in this record (cosmetic; the persistence row's other
   numbers are correct).
2. **`code/chaos_floor.py`** -- sampled 64 DS-0009 `test_chains` rows (32 scatter + 32 clump, seed
   0), built one `GenesisOracleEnv` (TRAINING_PHYSICS, `n_envs=64`), and for each perturbation
   level in {0, 0.5, 1, 2} mm: perturbed the recorded start state's xy by iid Gaussian noise of
   that std (+1 deg yaw jitter for levels >0, `model/retrieval/frame.py`'s yaw convention),
   `sim.set_particle_state` + `sim.update_material_state` (no re-settle, mirroring
   `Genesis/chain_collection.py`'s own `set_states`), then `sim.execute_action` with the ORIGINAL
   recorded `(p_start, p_stop, angle)` + `sim.update_material_state`, and scored the resulting
   particle state against the ORIGINALLY RECORDED `states_` using the same accuracy /
   blurred-accuracy / mm functions as task 1. Did not additionally reuse EXP-0024's own script
   for part (a) -- its result (resimulation noise negligible under exact-snapshot restore) is
   cited directly instead of rerun, since this record's 0mm level already re-derives the
   analogous "same state" case under the DIFFERENT (chain_collection-style, no snapshot/restore)
   mechanism DS-0009 was actually collected with, which is the more relevant check anyway.

## Numbers

### Task 1+2: extended harness on DS-0009 (`results/offline_eval_extended.json`)

**Lead metric is `slateN`/`slateN_tough` (per experiment-log's Metrics section); `accuracy_1`
and the two occupancy-distance diagnostics that follow are reported alongside as diagnostics,
not the verdict.**

| model | slateN (13g) | slateN_tough (8g) | accuracy_1 (scatter/clump) | blur1 | blur2 | rollout 1/2/3/4 | occ_emd_swept | mass_in_goal_mae (13g/8g) |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.059 | 0.050 | 0.000 (0.000/0.000) | 0.000 | 0.000 | 0.000/0.000/0.000/0.000 | 2.865 | 4.547 / 4.398 |
| nfd_3ch_narrow_l20 | 0.774 | 0.778 | 0.506 (0.441/0.523) | 0.611 | 0.704 | 0.467/0.416/0.369/0.348 | 2.395 | 1.964 / 1.801 |
| nfd_3ch_narrow_l20_wide | 0.803 | 0.827 | 0.486 (0.413/0.505) | 0.586 | 0.686 | 0.444/0.397/0.351/0.336 | 2.400 | 2.028 / 1.863 |
| linear_narrow_l20_res64 | 0.582 | 0.633 | 0.452 (0.292/0.496) | 0.582 | 0.657 | 0.421/0.375/0.337/0.324 | 3.637 | 4.048 / 3.731 |
| nfd_residual_worldframe_noaug_ep43 (best broad) | 0.721 | 0.749 | 0.466 (0.408/0.480) | 0.548 | 0.625 | 0.418/0.396/0.353/0.335 | 2.197 | 2.428 / 2.297 |
| retrieval_1nn | 0.413 | 0.436 | 0.172 (-0.003/0.219) | 0.298 | 0.444 | 0.195/0.091/0.036/0.009 | 3.285 | 2.942 / 2.792 |

`occ_emd_swept` = sliced-W1 distance (64-px units, lower better) between predicted and true
occupancy, SWEPT REGION ONLY, computed via `simple_mpc.value_functions.SlicedEMD`'s fixed
pixel-projection order generalised to two arbitrary (pred, truth) fields instead of one field
vs a fixed target (coordinator addition: an occupancy-distance metric that, unlike blurred
`accuracy`, should NOT reward a diffuse prediction over a sharp-but-offset one, for later
correlate-with-slateN work). `mass_in_goal_mae` = mean absolute |mass_in_region(pred, g) -
mass_in_region(truth, g)| over all pool rows, averaged over the 13-goal / 8-goal-tough sets.
retrieval_1nn's two new columns finished after the rest of this table (355s under heavy GPU/CPU
contention from the concurrently-launched DS-0011/DS-0012 Genesis collection jobs, vs 115s
uncontended in the first run) -- its `occ_emd_swept` (3.285) is worse than every occ model
including persistence (2.865), and its `mass_in_goal_mae` (2.942/2.792) sits between the narrow
NFDs (best, ~1.8-2.0) and persistence/linear_res64 (worst, ~4.0-4.5) -- consistent with its
accuracy_1/slateN ranking (better than persistence and linear_res64, worse than every NFD).
retrieval_1nn's moved-cube position error (true push-frame displacement > 1mm): mean 5.77mm,
over `n_test_rows=1024` chain rows / `n_pools=32`. The occ-model rows above reproduce EXP-0053's
own `results/offline_eval.json` and retrieval_1nn's first five metrics reproduce EXP-0059's
`results/offline_eval_retrieval.json` to the reported precision -- the extension changed no
existing number.

**A striking early metric-disagreement, worth flagging for the metric study (section 4 of the
design doc) rather than concluding from (n=5 models, not powered for correlation):**
`occ_emd_swept` and `mass_in_goal_mae` do NOT reproduce the `slateN`/`accuracy_1` ranking.
`nfd_residual_worldframe_noaug_ep43` (broad, 4th-ranked on `slateN`) has the BEST (lowest)
`occ_emd_swept` (2.197) of every model including the top-`slateN` narrow NFDs (2.395-2.400).
`linear_narrow_l20_res64` (positive `accuracy_1` 0.452, beats persistence) scores WORSE than
`persistence` on both `occ_emd_swept` (3.637 vs 2.865) and `mass_in_goal_mae` (4.048 vs 4.547)
-- i.e. by these two diagnostics alone, linear_res64 looks worse than doing nothing, the
opposite of its `accuracy_1`/`slateN` reading. This is exactly the kind of metric divergence
the coordinator asked this record to start logging for a later correlation study; it is NOT
resolved here (5-6 models is far too few to correlate anything with `slateN` reliably -- see
`paired_stats.required_n`), just recorded per-model so a later, larger sweep (R1 onward) can
test which of these tracks `slateN` and which doesn't.

**Reading:** slateN_tough tracks slateN within 0.02-0.03 for every model and never reorders them
-- the 8-goal "tough" subset (chosen for closed-loop task difficulty in EXP-0055/0057) is not
harder for slateN's offline ranking task specifically. Blur helps every model (the "hedging
rewarded" effect the brief's designer probe found), most for the least-accurate ones
(retrieval_1nn +0.272 at blur2, vs the occ models' +0.16-0.21) -- consistent with retrieval_1nn's
errors being sharp small-offset placements rather than wrong-shape, but not enough to close the
gap: the ORDERING of every model is unchanged at both blur levels.

### Task 3: chaos floor (`results/chaos_floor.json`, n=64 rows: 32 scatter + 32 clump)

| perturbation | accuracy_1 (scatter/clump) | blur1 | blur2 | mm_mean | mm_rms |
|---|---|---|---|---|---|
| 0 mm (state exact) | 0.703 (0.736/0.694) | 0.841 | 0.899 | 0.145 | 0.468 |
| 0.5 mm | 0.428 (0.317/0.467) | 0.660 | 0.775 | 0.820 | 1.476 |
| 1 mm | 0.303 (0.082/0.387) | 0.573 | 0.739 | 1.447 | 2.140 |
| 2 mm | 0.111 (-0.159/0.213) | 0.373 | 0.599 | 2.898 | 3.815 |

**Reading:** the 0mm level is NOT 1.0 -- re-simulating the identical recorded state+action
through `chain_collection.py`'s own call sequence and scoring against the recorded outcome
loses ~0.30 accuracy_1 (mm_mean only 0.145mm, i.e. physically near-exact, consistent with
EXP-0024/EXP-0027's measured mm-scale-only divergence and `score-occupancy-subpixel-stable`'s
documented amplification of small mm shifts into much larger `accuracy` swings) -- a genuine,
if physically tiny, resimulation gap, **not** the fabricated 0.664 originally drafted here (see
LOG.md for the mid-run bug and correction).

**The steep finding:** accuracy_1 collapses far faster with state uncertainty than a first
pass would guess -- 0.5mm alone (mm_mean 0.82mm) already drops the ceiling to 0.428, BELOW
`linear_narrow_l20_res64` (0.452) and close to `nfd_residual_worldframe_noaug_ep43` (0.466); at
1mm (mm_mean 1.45mm) the ceiling (0.303) sits below every trained occ model in the comparison
table; at 2mm it is close to `retrieval_1nn`'s own 0.172. Since DS-0009's recorded states carry
no measurement noise (they are exact simulator floats, not perception output), this is not
literally "the ceiling any model faces today" -- it instead quantifies how expensively the
swept-region `accuracy` metric punishes ANY positional error, model or perturbation alike:
a "perfect" dynamics function that were merely as imprecise as a 0.5-1mm cube-placement error
would already be indistinguishable from, or worse than, today's best narrow NFD (0.506). Blur
(sigma 1-2px) recovers a large fraction of this loss at every level (e.g. 1mm: 0.303 -> 0.573 ->
0.739), mirroring the brief's "2px blur lifted a crude 1-NN from 0.295 to 0.579" observation --
the metric's sharpness penalty and its state-precision penalty are the same mechanism.

## R2 addendum (2026-09-28, fresh-instance continuation): perturbed-sim-zoo slateN + diagnostics_r2

Both items named below as "not reached" in the previous instance's write-up landed and are
recorded here rather than silently folded into the frontmatter `result` (which stays as the
prior instance wrote it, plus one line -- see below).

**Perturbed-simulator zoo, slateN version (`results/perturbed_sim_zoo.json`, `levels_mm`)** --
DS-0011 val-pools (32 pools x 64 pushes), same perturb-then-resimulate mechanism as the
chains-shaped chaos floor above but now scored with the full `slateN`/`slateN_tough` pool
machinery (many candidate actions per perturbed start), at blur sigma 0/1/2:

| level | slateN (blur0/1/2) | slateN_tough (blur0/1/2) |
|---|---|---|
| 0 mm | 0.929 / 0.925 / 0.914 | 0.932 / 0.930 / 0.923 |
| 0.5 mm | 0.911 / 0.910 / 0.906 | 0.923 / 0.922 / 0.924 |
| 1 mm | 0.861 / 0.859 / 0.852 | 0.863 / 0.862 / 0.855 |

**Reading:** unlike `accuracy_1` (which collapsed 0.703->0.111 over the same 0-2mm range on the
chains sample above), `slateN`/`slateN_tough` are far more robust to sub-mm/mm state
perturbation of the SIMULATOR itself -- 1mm perturbation only costs ~0.07 off slateN_tough
(0.932->0.863), still far above every trained model's `slateN_tough` (best: wide narrow NFD
0.827). Blur barely moves any of these cells (largest shift 0.011, at 1mm/blur2), matching the
model-control cell's own near-flat blur response recorded in the prior LOG.md entry. **This
resolves the earlier "does ranking quality come from information or blur" question in the
negative for state-perturbation noise specifically**: a perturbed-but-still-ranking-correctly
simulator is not what blur rescues; blur rescues `accuracy_1`, a different and more
blur-sensitive quantity, and `slateN` stays informative under perturbation levels well past
where `accuracy_1` has already collapsed.

**diagnostics_r2.py (`results/diagnostics_r2.json`) -- decomposing retrieval's gap to the NFDs,
all offline (no Genesis), all labelled `ceiling`/ORACLE, i.e. NOT achievable by any real
predictor and not to be used as a headline result (uses privileged knowledge of the true
outcome to pick the best of many candidates):**

- **Section 1 (oracle best-of-top-k-by-distance, mm error):** searching the WHOLE `k5_cube_median`
  bank (11,921 rows, 36 non-degenerate query rows) for the donor whose TRANSFERRED prediction is
  closest to the TRUE outcome: top-50 mm_mean 1.80, top-500 1.39, whole-bank 1.18 -- i.e. even an
  oracle restricted to the metric's own top-50 neighbourhood already loses ~0.6mm to the
  whole-bank oracle, showing the distance metric's own top-50 shortlist is not yet capturing the
  single best-transferring donor most of the time.
- **Section 2 (same-state leave-one-out ceiling):** DS-0009 pools share one start state per pool;
  retrieving the nearest-by-ACTION other push from the SAME pool (2,048 rows) gives
  `acc1_leave_one_out_ceiling` 0.366, mm_error_mean 2.89 -- notably BELOW the cross-pool oracle's
  implied accuracy and below several trained occ models' 0.45-0.51 accuracy_1, i.e. even
  same-state/near-identical-action retrieval does not by itself reach today's best NFD.
- **Section 3 (transfer-rule comparison, under the section-1 oracle donor):** `displacement`
  (mm_mean 1.181) and `transport_projected` (1.169) are close and both much better than `paste`
  (4.740) -- confirms the bank's displacement-style transfer rule is the right family, paste is
  clearly worse.
- **Section 4 (moved-cube-in-window / "predicts nothing moves" check, `k5_cube_median`, n=1,024
  queries):** 9.6% of query rows have zero cubes moved in the swept window; the predictor calls
  these correctly zero 80.6% of the time.
- **Section 5/5b (oracle best-of-top-50-by-distance vs random-50, 256 queries / 8 pools):**
  top-by-distance-N50: mm_mean 1.975, acc1 0.240, **slateN 0.532**; random-N50 (same bank size,
  no distance selection): mm_mean 2.506, acc1 0.176, **slateN 0.285** -- the distance metric's
  top-50 shortlist clearly beats a random 50-donor draw of the same size (slateN +0.25), so the
  metric IS doing useful work, it just doesn't reach an oracle over the whole bank (section 1).
- **Section 6 (action-restricted leave-one-out, 5mm/10deg qualifying window):** of 2,048 pushes,
  67.7% (1,386) have a same-pool donor within 5mm/10deg of the query action; restricted to those,
  `acc1_loo_action_restricted` 0.410, mm_error_mean 2.15 -- similar order to section 2, i.e.
  restricting to close-action donors doesn't change the qualitative ceiling much.

**One-line addition to the frontmatter `result`** (kept short per experiment-log's 30-second-skim
rule; full numbers live only here, not duplicated into frontmatter): the perturbed-sim `slateN`
zoo shows ranking quality (`slateN`/`slateN_tough`) is far more robust to state perturbation than
`accuracy_1` is, and diagnostics_r2's oracle ceilings show `k5_cube_median`'s distance metric
already beats random donor selection by a wide margin (slateN 0.53 vs 0.28) but is itself well
below a whole-bank oracle (mm 1.18) and even a same-pool leave-one-out ceiling (acc1 0.366) sits
below today's best NFD -- i.e. the retrieval family's gap is only partially a search/metric
problem; a meaningful share is donor/transfer-rule coverage that a better distance metric alone
will not close.

## Task 1 addendum (2026-09-28, fresh instance): data scaling -- more bank data, and DS-0012 vs orig structure

`code/data_scaling.py` (checkpointed atomically, `results/data_scaling.json`): built 5
`k5_cube_median` retrieval banks (the R1 sweep's best clean config: `cap=0.02,
mismatch_penalty=0.04, corridor_weight=3.0, k=5, aggregation=cube_median`) and scored each on
DS-0009 via `eval_extended.eval_particle_model` (now accepting an arbitrary `predictor_obj`, a
small non-invasive extension added for this task) plus an oracle best-of-top-50-by-distance mm
error on 256 DS-0009 `test_chains` queries (reusing `diagnostics_r2.py`'s section-5 method,
generalized to an arbitrary bank).

| bank | size | slateN | **slateN_tough** | acc1 | rollout4 | moved-cube mm | oracle-top50 mm |
|---|---|---|---|---|---|---|---|
| orig_12k (DS-0008+DS-0010, = existing `bank.pt`) | 11,921 | 0.513 | 0.546 | 0.227 | 0.075 | 4.40 | 1.975 |
| ds0012_only_12k (DS-0012 subset, same size) | 11,921 | 0.507 | 0.552 | 0.210 | 0.036 | 4.33 | 1.883 |
| 25k (orig_12k + DS-0012 subset) | 25,000 | 0.566 | **0.591** | 0.232 | 0.060 | 4.20 | 1.841 |
| 50k (orig_12k + DS-0012 subset) | 50,000 | 0.562 | 0.583 | 0.241 | 0.074 | 4.08 | 1.781 |
| 98k_all (orig_12k + ALL of DS-0012) | 97,937 | 0.532 | 0.573 | 0.258 | 0.074 | 3.99 | 1.745 |

**UPDATE (2026-09-28, coordinator round 2): pool-bootstrap 95% CIs show "declines" was an
OVERSTATEMENT of what the data supports -- reworded below.** `code/data_scaling_pools_raw.py`
regenerated per-pool raw arrays (rerunning ONLY the pools loop, `ch=[]`, against the
already-persisted banks, CPU-only) and `code/bootstrap_ci.py` (reusing EXP-0060's
`pool_bootstrap_ci`, 2000 resamples over the 32 shared DS-0009 pools) gives:

| bank | slateN_tough (point) | pool-bootstrap 95% CI |
|---|---|---|
| orig_12k | 0.546 | [0.433, 0.650] |
| ds0012_only_12k | 0.552 | [0.454, 0.650] |
| 25k | 0.591 | [0.499, 0.672] |
| 50k | 0.583 | [0.481, 0.669] |
| 98k_all | 0.573 | [0.473, 0.666] |

**Every CI overlaps every other CI almost completely** -- at n=32 pools this design cannot
distinguish 25k from 50k from 98k_all on `slateN_tough` at all. The PAIRED delta (98k - 25k,
resampling pools once per draw so both configs share the same resample, the correct comparison
since both are scored on the identical 32 pools) is **-0.018, 95% CI [-0.067, +0.030]** -- the
CI includes zero, and 22.9% of the 2000 bootstrap draws actually show 98k ABOVE 25k. **Correct
statement: `slateN_tough` is FLAT beyond ~12-25k at this sample size, not "declining" -- the
point-estimate ordering (25k > 50k > 98k) is not statistically supported.** What the earlier,
overclaimed framing got right and what survives this correction: `accuracy_1` (0.227 -> 0.258),
moved-cube mm error (4.40 -> 3.99), and the oracle-top50 mm ceiling (1.975 -> 1.745) DO improve
with more data (these are point estimates over much larger sample sizes -- 1024 chain rows / 231
non-degenerate oracle queries -- not run through the same pool-bootstrap here, but their
monotone trend across 4-5 points each is a much stronger pattern than `slateN_tough`'s noisy,
overlapping-CI curve). So the honest conclusion is: **more data helps 1-step reconstruction and
donor-search quality, unambiguously; whether it helps ACTUAL RANKING quality beyond ~25k is not
resolved by this experiment (underpowered at 32 pools) -- not "it hurts."** Practically, 25k
remains the recommended operating point on cost grounds alone (far cheaper to search than
50k/98k) even though its `slateN_tough` edge over them is not statistically established. Two
untested hypotheses for what WOULD explain a real decline, if one exists, are still worth
naming (unchanged from the original draft): (a) a fixed k=5 median aggregation may become less
robust as the bank grows if extra rows increase local donor density without curation; (b)
DS-0012 (seed 301) could carry a subtly different distribution than DS-0008/DS-0010 that
dominates the bank at 98k (88% of rows).

**Same-size 12k-vs-12k structure check:** ds0012_only_12k (0.552) is close to orig_12k (0.546)
on `slateN_tough` -- no large structural difference at matched size on the RANKING metric -- but
ds0012_only_12k is clearly worse on `accuracy_1` (0.210 vs 0.227) and much worse on `rollout4`
(0.036 vs 0.075, more than half). DS-0012 alone is a usable but somewhat noisier 1-step/rollout
bank than the curated DS-0008+DS-0010 mix at the same size; its ranking quality (what `slateN`
actually measures) is not meaningfully different.

**GPU/CPU scaling (the "check bank build/search scale on the GPU, chunked" ask):**
`model/retrieval/distance.py`'s `full_distance_matrix`/`topk_search` already chunk over the bank
dimension (default `chunk=1500`) and default to CUDA -- no new chunking code was needed. The
256-query oracle search itself scales close to linearly with bank size (6.9s/14.6s/35.6s/57.0s
for 12k/25k/50k/98k). Reported main-eval wall times are NOT a clean scaling series: the 50k
cell's 881.8s overlaps with an unrelated CPU-heavy process this same session launched (a
`multistep_eval.py` smoke test, killed once noticed, see LOG.md) and should be read as inflated;
the two clean points (25k: 220.9s, 98k: 1235.4s, no known contamination) imply mildly
super-linear but not extreme scaling (~5.6x wall time for ~3.9x bank size) over that range. Peak
GPU memory stayed under 3.4 GB of 8 GB throughout -- never a resource risk, and consistent with
the coordinator's "keep GPU use moderate" instruction once the contaminating CPU job was killed.
Every bank is persisted (`artifacts/bank_{name}.pt`) per experiment-log's "keep every fitted
object."

## Task 2 (2026-09-28, fresh instance): multi-step headline on DS-0013

New `code/multistep_eval.py`: for each of DS-0013's 32 pools (64 candidate 3-push SEQUENCES
from one shared start state), rolls every model along every sequence's own recorded ACTIONS,
feeding the model's OWN predictions back as next-step input (never the true intermediate
state). Reports terminal (after push 3) `slateN`/`slateN_tough`, per-step `slateN_tough`
(after push 1/2/3), and per-step rollout accuracy (cumulative swept region, same convention as
the existing 1-step `rollout_accuracy_k`). Results in `results/multistep_eval.json`.

| model | terminal slateN_tough | step 1 / 2 / 3 slateN_tough | step 1 / 2 / 3 rollout accuracy |
|---|---|---|---|
| persistence | 0.075 | 0.036 / 0.033 / 0.075 | 0.000 / 0.000 / 0.000 |
| nfd_3ch_narrow_l20 | 0.708 | 0.782 / 0.756 / 0.708 | 0.516 / 0.421 / 0.381 |
| nfd_3ch_narrow_l20_wide | 0.777 | 0.809 / 0.805 / 0.777 | 0.500 / 0.405 / 0.367 |
| nfd_residual_worldframe_noaug_ep43 (broad) | 0.665 | 0.719 / 0.710 / 0.665 | 0.469 / 0.396 / 0.365 |
| linear_narrow_l20_res64 | **0.715** | 0.586 / 0.672 / 0.715 | 0.473 / 0.386 / 0.356 |
| retrieval_1nn (orig 12k bank) | 0.480 | 0.542 / 0.519 / 0.480 | 0.221 / 0.095 / 0.048 |
| retrieval k5_cube_median @ 25k (best bank, task 1) | 0.546 | 0.615 / 0.565 / 0.546 | 0.292 / 0.160 / 0.104 |

**Finding: `linear_narrow_l20_res64`'s per-step slateN_tough is the only one that RISES with
horizon** (0.586 -> 0.672 -> 0.715), the opposite direction from every other model (e.g. wide
NFD 0.809 -> 0.805 -> 0.777, narrow NFD 0.782 -> 0.756 -> 0.708). At the POINT-ESTIMATE level
linear64 (0.715) ends up above `nfd_3ch_narrow_l20` (0.708) at 3 steps despite trailing it badly
at 1 step (single-step `slateN_tough`: 0.633 vs 0.778).

**UPDATE (2026-09-28, coordinator round 2): pool-bootstrap CI shows this specific crossover is
NOT statistically supported at n=32 pools -- reworded from the original "edges out" framing.**
`code/multistep_eval.py` was extended to persist per-pool terminal-step raw arrays and rerun
(CPU) for all 5 occ models; `bootstrap_ci.py` gives:

| model | terminal slateN_tough (point) | pool-bootstrap 95% CI |
|---|---|---|
| persistence | 0.075 | [-0.003, 0.151] |
| nfd_3ch_narrow_l20 | 0.708 | [0.637, 0.772] |
| nfd_3ch_narrow_l20_wide | 0.777 | [0.715, 0.832] |
| nfd_residual_worldframe_noaug_ep43 (broad) | 0.665 | [0.600, 0.730] |
| linear_narrow_l20_res64 | 0.715 | [0.632, 0.788] |

narrow-NFD's and linear64's CIs overlap almost completely ([0.637,0.772] vs [0.632,0.788]). The
PAIRED delta (linear64 - narrowNFD, resampling the SAME 32 pools once per draw) is **+0.007,
95% CI [-0.034, +0.049]** -- includes zero, and only 63.3% of the 2000 bootstrap draws favour
linear64. **Correct statement: at 3-step horizon, linear64 and narrow-NFD's terminal ranking
quality are statistically indistinguishable at this sample size (32 pools) -- the point estimate
mildly favours linear64 (63% of resamples), but this is NOT a confident crossover.** What DOES
survive: the DIRECTION of the per-step trend itself (linear64 rising, every occ model here
falling) is a within-model, step-paired pattern computed from the SAME 32 pools at each step,
which is a different and likely more robust comparison than the CROSS-model terminal-value
gap -- not itself re-bootstrapped here (budget), but worth distinguishing from the (now
retracted) "linear64 beats narrow NFD" claim: the reversal in TREND direction is better supported
than the specific CROSSOVER at step 3.

Worth naming as an untested hypothesis for the trend-direction finding: linear64's 1-step errors
may be more strongly correlated
with the swept-region geometry across a full pool (helping RELATIVE ranking within a pool even
as absolute per-step reconstruction stays mediocre), while the NFDs' compounding rollout error
(all three occ models lose 25-30% of their `rollout_accuracy` by step 3, see the table's last
column) increasingly corrupts the specific quantity `slateN` reads off (predicted terminal
value). Rollout accuracy itself degrades similarly across ALL occ models (roughly -25% relative
from step1 to step3), so 1-step accuracy comparisons alone would have hidden this reversal.
Retrieval models (both retrieval_1nn and k5_cube_median) decline monotonically like the NFDs,
just from a lower base, and their ROLLOUT accuracy collapses much faster proportionally
(retrieval_1nn: 0.221 -> 0.095 -> 0.048, more than 3/4 lost by step 3) -- compounding error is
retrieval's worst-documented failure mode (consistent with the original R0 rung's single-chain
rollout finding) and this multi-step pool test reproduces it independently on a different
corpus (DS-0013) and different task shape (ranking, not just prediction).

**The 0mm "perfect model" reference was skipped** -- it needs fresh Genesis re-simulation of
DS-0013's own recorded action sequences (analogous to `chaos_floor.py`'s mechanism but for
3-step chains), estimated well over the 30-minute budget the coordinator set for it once GPU
sharing with the coder's job and this record's own already-lengthy task 1/2 runs are accounted
for -- not attempted, named here rather than silently dropped.

## Task 3 (2026-09-28): coder's NFD-with-reference -- BLOCKED, not run

The coder's `results/coder_status.md` reports the main run complete (negative headline result:
`retrieval_nfd_ref` slateN_tough -0.123, WORSE than persistence's 0.050, on DS-0009; its
zeroed-reference control is statistically indistinguishable, -0.179, pointing at a channel-0
train/eval rasteriser mismatch as the likely cause, not the retrieved content itself -- see that
file for the coder's own full diagnosis). **Independent sanity-check of its 1-step numbers**
(what this record could do without touching coder code): recomputed `slateN_tough` for all
three of the coder's models directly from `results/offline_eval_extended_raw.json`'s persisted
per-pool `{vt, vp}` arrays, with this record's OWN aggregation code (not the coder's) --
recovered -0.1225093297676083 (reported: -0.12250932849110541), 0.49868222983257865 (reported:
0.4986822306864269), -0.1786074520527822 (reported: -0.17860745138295897) for
`retrieval_nfd_ref`/`retrieval_nfd_random_donor`/`retrieval_nfd_ref_zeroed` respectively --
matches to 9+ significant figures. **This rules out an aggregation/reduction bug as the cause of
the negative result** (a real risk worth checking given how surprising "worse than persistence"
is) -- the negative numbers are exactly what the saved per-pool raw predictions produce, not an
artifact of how they were summarised. It does NOT independently re-verify the underlying
per-row predictions themselves (would require re-running the model from its checkpoint, out of
reach without touching coder files) or `accuracy_1` (no raw per-row array is persisted for that
metric, only for `slateN`/`slateN_tough`).

**Multi-step run attempted, BLOCKED by a live file conflict, not run:** `multistep_eval.py
--models retrieval_nfd_ref retrieval_nfd_random_donor retrieval_nfd_ref_zeroed` crashed
immediately: `ImportError: cannot import name 'render_cubes_soft' from
'model.retrieval_nfd.render'`. Cause (diagnosed, not fixed -- per the coordinator's explicit
"don't touch its files"): `model/retrieval_nfd/render.py` was modified at 06:28 (mtime),
**after** `model/retrieval_nfd/predictor.py` (mtime 05:53) which still imports the
now-renamed `render_cubes_soft` (current `render.py` only defines `render_cube_boxes_np`/
`render_cube_boxes_batch`) -- the coder is actively continuing to edit this module (plausibly
implementing their own next-step diagnosis, "re-run with `occ_from_particles` at both train and
eval time," which would touch exactly this file), leaving it in a transiently broken state for
any OTHER process importing it. Not retried in this record -- retry once `coder_status.md`
reports a new stable checkpoint/milestone, using the exact same command above (no new code
needed, `retrieval_nfd_ref`/`retrieval_nfd_random_donor`/`retrieval_nfd_ref_zeroed` are already
registered `OCC_ADAPTERS` entries `multistep_eval.py`'s `run_occ` calls unchanged).

### UPDATE (2026-09-28, same session): the coder fixed the bugs and retrained -- real, non-degenerate numbers now

The coder's rewritten `coder_status.md` (06:22-07:45 rewrite) identifies and fixes 4 real bugs
(NOT representation hand-waving): a ~70x-slower-than-necessary donor search caused by a
per-row CUDA-sync Python loop (now vectorised); channels 0-2 now come directly from
`Baselines.NFD.nfd_lib.PileSweepData3Ch` bit-exact with the narrow NFD's own training data
(replacing the earlier from-scratch soft-cube renderer that caused the render.py/predictor.py
import conflict this record hit); eval-time retrieval now uses the TRUE particle state via a
targeted `eval_extended.py` patch (`predict_step_particles`, 3 call sites) rather than
pseudo-cubes recovered from occupancy; and a unit-test threshold fix (0.029% pixel mismatch from
a genuine ~1e-6px float roundtrip, not a structural bug). **Retrained on the FULL DS-0008+DS-0010
corpus** (10,910/535/476 train/val/test, matching narrow NFD's own split), 20 epochs (vs narrow
NFD's 60).

**1-step DS-0009 results (real, non-degenerate this time), independently re-verified**: this
record recomputed `slateN_tough` for all three models from `offline_eval_extended_raw.json`'s
per-pool arrays with its OWN aggregation code -- matched the coder's reported numbers to 8-9
significant figures (`retrieval_nfd_ref`: 0.6861171548 vs reported 0.6861171541;
`retrieval_nfd_random_donor`: 0.5967350360 vs 0.5967350365; `retrieval_nfd_ref_zeroed`:
0.7145344387 vs 0.7145344377) -- no aggregation bug.

| model | DS-0009 slateN_tough | acc1 | rollout4 |
|---|---|---|---|
| narrow NFD (reference) | 0.778 | 0.506 | 0.348 |
| retrieval_nfd_ref (main, retrieval ON) | 0.686 | 0.480 | 0.341 |
| retrieval_nfd_random_donor (control i) | 0.597 | 0.464 | 0.326 |
| retrieval_nfd_ref_zeroed (control iii, donor channels zeroed at test time, SAME ckpt as main) | **0.715** | 0.490 | 0.337 |

**The information-vs-noise signal points the WRONG way for this checkpoint: zeroing the
reference at test time (0.715) BEATS the real retrieval-on model (0.686)**, and both beat the
random-donor control (0.597) -- i.e. having SOME reference (even a zeroed/uninformative one, as
long as the network was trained WITH the real mechanism present) beats having no mechanism at
all (random donor), but the model does better when the real retrieved content is suppressed at
test time than when it's used. None yet beats narrow NFD's 0.778 (20 vs 60 epochs, 2 extra
untrained-relative-to-narrow-NFD donor channels per the coder's own note) -- not read as
"retrieval fundamentally can't help," but as "this specific 20-epoch checkpoint isn't yet using
its reference channels productively."

**Multi-step rollout on DS-0013** (retried successfully once the coder's fix landed;
`code/multistep_eval.py`, CPU-only, `results/multistep_eval_coder.json`):

| model | terminal slateN_tough | step 1 / 2 / 3 slateN_tough | step 1 / 2 / 3 rollout accuracy |
|---|---|---|---|
| retrieval_nfd_ref (main, retrieval ON) | **0.642** | 0.710 / 0.649 / 0.642 | 0.484 / 0.403 / 0.370 |
| retrieval_nfd_random_donor (control i) | 0.553 | 0.617 / 0.555 / 0.553 | 0.477 / 0.394 / 0.359 |
| retrieval_nfd_ref_zeroed (control iii, zeroed at test time) | 0.613 | 0.700 / 0.674 / 0.613 | 0.502 / 0.406 / 0.363 |

**Reversal from the 1-step finding, point-estimate only (no CI computed for these 3 -- flagged,
not asserted as significant): at multi-step terminal, the REAL retrieval-on model (`ref`, 0.642)
now BEATS the zeroed-reference control (0.613)**, the opposite order from 1-step DS-0009
(`zeroed` 0.715 > `ref` 0.686). All three still rank above `random_donor` (0.553) at every step,
same as 1-step. Given this record's own pool-bootstrap findings above (CIs of comparable width
to the occ-model family, ~0.06-0.09 wide, easily large enough to swallow this ref-vs-zeroed gap
of 0.029), **this reversal should be read as "not yet distinguishable from noise," not as
confirmed evidence that retrieval helps more under compounding rollout than single-step** -- a
CI was not computed for these 3 models specifically (budget), named here as the next check.

### SUPERSEDED (2026-09-28, ~08:10, after this record's own hard stop): `random_donor` was not a valid information-vs-noise control -- `retrieval_nfd_noref` replaces it

The coder flagged, correctly, that `retrieval_nfd_random_donor` conflates two different
manipulations: it removes the TRUE reference AND substitutes a WRONG one (noise), so a
`ref` vs `random_donor` gap cannot separate "hurt by wrong information" from "helped by having
any information at all." The coder retrained a fair twin, `retrieval_nfd_noref` (identical
architecture/recipe, but donor channels forced to zero in BOTH training and eval -- a genuine
"no reference at all," not "a wrong one"), and reran both `ref` and `noref` for the full
60-epoch schedule narrow-NFD itself used (the earlier round's `ref`/`ref_zeroed`/`random_donor`
numbers above were only 20 epochs and are superseded by this table, not deleted, per this
record's "don't silently overwrite" convention):

| model | DS-0009 slateN_tough | acc1 | rollout4 | DS-0011 slateN_tough |
|---|---|---|---|---|
| retrieval_nfd_ref (retrieval ON, 60 ep, best@ep60, still improving) | 0.752 | 0.478 | 0.324 | 0.790 |
| retrieval_nfd_noref (NEW fair no-reference twin, 60 ep, best@ep53, plateaued) | 0.778 | 0.486 | 0.322 | 0.752 |
| retrieval_nfd_ref_zeroed (unchanged: `ref` ckpt, reference zeroed at test time only) | 0.775 | 0.496 | 0.326 | 0.777 |

**Verdict per the coder (this record did not independently re-verify these specific numbers
before its own hard stop -- named as a threat below): a wash, not a win.** `ref` beats `noref`
on DS-0011 (0.790 vs 0.752) but LOSES to it on DS-0009 (0.752 vs 0.778) and on `acc1` on both
datasets. **`retrieval_nfd_random_donor` is retired as the information-vs-noise control from
this point forward; `retrieval_nfd_noref` is the correct baseline for that comparison in any
future work on this record.** The earlier "zeroed beats ref at 1-step, ref beats zeroed at
multi-step, both point-estimate-only" observations above are about `ref` vs its OWN
test-time-zeroed self (a legitimate same-checkpoint ablation, not affected by this correction) --
they stand as recorded, just no longer read alongside `random_donor` as if it were a clean
noise-vs-information axis.

**Threats specific to this addendum:** (1) not independently re-verified by this record (no
raw-array recompute, no re-run) -- this record's own hard stop had already passed when the
correction arrived; relayed from the coder's `coder_status.md` update, one level less verified
than the rest of this file's numbers. (2) Multi-step (DS-0013) was NOT rerun for
`retrieval_nfd_noref` -- the multi-step table two sections up still uses the retired
`random_donor` control; re-running `multistep_eval.py --particle-models` (no: `--models
retrieval_nfd_noref`, it is an OCC_ADAPTERS entry like the others) against `noref` is the
natural next step, costed at ~450-950s CPU per the timings observed above for this model family,
not attempted here.

## Coordinator round 2 (2026-09-28, hard stop 08:10, offline/CPU only -- GPU reserved for the coder)

### Neighbour-rank curve (information-vs-noise control, design doc R2)

`code/neighbour_rank_curve.py`: on the 25k bank (task 1's winner), transfer from the donor at
distance RANK r in {1, 5, 20, 100, 1000, random} (displacement-only transfer, no k-way
aggregation -- isolates "how good is the retrieved donor" from "how good is the aggregation"),
scored on DS-0009: `accuracy_1` + moved-cube mm (256 `test_chains` queries, scatter+clump
balanced) and `slateN_tough` (all 32 `test_pools`). One shared distance search (chain queries +
pool queries against the bank, computed once, ~430s CPU) then cheap per-rank column selection.

| rank r | mm (mm, lower better) | accuracy_1 | slateN_tough |
|---|---|---|---|
| 1 | 4.22 | **0.197** | 0.516 |
| 5 | 4.66 | 0.167 | 0.520 |
| 20 | 4.66 | 0.156 | 0.470 |
| 100 | 5.23 | 0.107 | 0.455 |
| 1000 | 5.75 | 0.046 | 0.404 |
| random | **6.48** | 0.006 | 0.302 |

**Clean, near-monotone decline across all three metrics as r increases, worst at "random" on
every metric** -- this is exactly the signature the coordinator named: retrieved information is
doing real work, not the aggregation machinery. `accuracy_1` is the sharpest and most strictly
monotone (0.197 -> 0.006, a 33x drop from rank1 to random); mm error and `slateN_tough` are
monotone except for a small rank1-vs-rank5 flat/reversed step (mm 4.22->4.66, slateN_tough
0.516->0.520, both within likely noise at n=231/32) before resuming a clear decline. At r=1000
(essentially "a mediocre match from deep in the bank") the model is already most of the way to
random (accuracy_1 0.046 vs random's 0.006, slateN_tough 0.404 vs random's 0.302) -- i.e. only
the near neighbours (roughly the top ~20-100 of 25,000) carry real transfer value; the long tail
of the bank does little for THIS transfer rule. Read together with task 1's finding that
`slateN_tough` peaked at 25k and declined at 50k/98k: this curve confirms retrieval quality
genuinely depends on the RETRIEVED donor's rank (not just "having more bank"), consistent with
(though not proof of) the data-scaling hypothesis that a bigger bank without better search/
aggregation dilutes rather than helps.

### Confidence model (`predict_particles_with_confidence`)

`code/confidence_model.py`: for `k5_cube_median@25k` on all 32 DS-0009 `test_pools` (2048
rows -- 32 shared start states x 64 candidate actions each, exactly "given a state, which
actions does the dataset predict least well"), tested whether the two FREE confidence signals
(`top1_dist`, `knn_disagreement`) predict per-row error against moved-cube mm error (n=1885
non-degenerate rows) and swept-region proxy error (n=1647). `results/confidence_model.json`:

| signal | vs mm_error (Spearman, 95% CI) | vs swept_error (Spearman, 95% CI) | within-pool mean rho (mm / swept) | lift top-10%-least-confident (mm / swept) |
|---|---|---|---|---|
| `top1_dist` | +0.056 [+0.011, +0.102] | **-0.226 [-0.271, -0.179]** | 0.091 / -0.055 | 1.12x / **0.75x** |
| `knn_disagreement` | +0.217 [+0.173, +0.263] | +0.449 [+0.409, +0.487] | 0.222 / 0.345 | 1.15x / 1.01x |

**`knn_disagreement` is the useful confidence signal here; `top1_dist` is weak-to-actively-
misleading.** `knn_disagreement` (do the k=5 Hungarian-matched neighbours imply different
outcomes) correlates positively with both error types at every cut (row-level, within-pool, and
lift), moderately strong against `swept_error` (+0.449) -- this is a genuine, usable "which
actions does the bank predict least well" signal, free (no extra bank search). **`top1_dist`
(the nearest neighbour's raw distance) does NOT reliably track error and is NEGATIVELY
correlated with `swept_error`** (-0.226, CI excludes 0, and its lift for the top-10%-least-
confident decile is 0.75x -- i.e. the rows this signal flags as LEAST confident actually have
BELOW-AVERAGE swept-region error, the opposite of the intended use). **Not diagnosed here**
(named as an untested hypothesis, not asserted): a plausible confound is push/region SIZE --
queries with a small swept region (few cubes need to move) may have both a naturally small
`top1_dist` (a common, well-populated push in the bank) AND, mechanically, less ROOM for a large
swept-region error, while a big swept region has more error surface regardless of retrieval
quality; `knn_disagreement`, being about internal k-neighbour AGREEMENT rather than raw
distance, may be less exposed to this confound. **Practical conclusion: use
`knn_disagreement`, not `top1_dist`, as the confidence signal for "which actions does the
dataset predict least well."**

### Pool-bootstrap CIs on data-scaling and multi-step values

`code/data_scaling_pools_raw.py` (reruns ONLY the DS-0009 pools loop, `ch=[]`, against the
already-persisted `artifacts/bank_{name}.pt` banks, CPU) regenerated per-pool raw arrays; CIs
computed via EXP-0060's `correlate.py::pool_bootstrap_ci` (reused unchanged) -- see LOG.md/
results files for the final numbers and the reworded "declines" language once every bank
finished (in progress at the time this section was written; 50k/98k_all are the slower cells).
`code/multistep_eval.py` gained raw terminal-step persistence (`slateN_tough_terminal_raw`) and
was rerun for the 5 occ models (`results/multistep_eval_ci_raw.json`) to CI the terminal values
and the linear64-vs-narrow-NFD paired delta at 3 steps.

## What would change the verdict

- Diagnose WHY `slateN_tough` peaks at 25k rather than continuing to improve with acc1/mm/oracle
  -- e.g. rerun `k5_cube_median` at 25k/50k/98k restricted to an equal-size RANDOM subsample of
  each larger bank vs its full self, to separate "more rows" from "more of DS-0012 specifically."
  Cost: ~15-20 min GPU, same script with one more config per size.
- More chaos-floor rows (n=64 is a modest sample; per-level SEM not computed here -- budget).
  Bootstrap CI over rows would sharpen whether 0.5mm's floor (0.428) is meaningfully below or
  within noise of `linear_narrow_l20_res64`'s 0.452.
- `crowd_floor` as a secondary tough-goal value function was in scope ("if cheap") but not run --
  judged not cheap enough to add without displacing the chaos floor from budget.
- Both items previously listed here as "R0 additions requested mid-session, not reached" (the
  perturbed-simulator-zoo `slateN` scoring, and a same-state leave-one-out retrieval ceiling) are
  now done -- see "R2 addendum" above -- and are removed from this list.
- Section 1's oracle-donor search used only `N_SUBSAMPLE=40` non-degenerate queries (36 after
  dropping zero-moved rows) -- a modest sample for an mm-error mean; not bootstrapped.
- **Multi-step RANKING test set:** no existing corpus supports genuine multi-step ranking
  (multiple candidate action SEQUENCES from one shared start, scored at horizon > 1) under
  narrow-domain (`TRAINING_PHYSICS`) physics -- see LOG.md's 2026-09-28 entry for the search
  (`slates_multistep` gives a step-1-only ranking pool under mismatched physics; DS-0009's own
  chains are single fixed paths). The coder has since added `chain_collection.py --mode
  seqpools`, which closes this gap once run (DS-B per the design doc) -- out of this record's
  scope by the coordinator's explicit instruction.
- Whether the 0mm resimulation gap (0.703, not 1.0) is itself dominated by yaw/z rather than xy,
  and whether it comes from the different per-batch physics step count (this record batches 64
  mixed-length-push rows together; the original DS-0009 collection batched 32 envs per chunk,
  and Genesis's batched stepping runs as many steps as the batch's longest push -- a plausible,
  untested alternative explanation for why 0mm is not exactly 1.0) -- not decomposed here.

## Coordinator follow-up (2026-09-28, later): tool-placement legality audit, curated interaction-set bank, transfer-gate fix, post-fix rerun

User-reported bug in `retrieval_debug.py`'s visual-debug figures (`q0908`): the tool touching
down directly ON a cube. Four-part follow-up (audit -> curate -> fix -> re-evaluate); this
section is the results summary, full detail in `LOG.md`'s matching entry,
`experiments/OPEN_ISSUES.md` ISS-010, and `datasets/DS-0014-retrieval-curated-interaction-sets/
DATASET.md`.

**Audit (A):** exact-SAT touchdown-overlap test (`code/audit_tool_placement.py`) on DS-0008/9/
10/11/12/13. Roughly HALF of every `pile_aware`-sampled dataset's rows are illegal (tool
overlapping a cube at `p_start`): DS-0008 0.443, DS-0009 chains 0.462/pools 0.561, DS-0011
0.478, DS-0012 0.458, DS-0013 0.517. DS-0010 (older pipeline) only 0.0002. Root cause:
`Genesis/sandbox_manipulation_clean.py::_pile_aware_stops` clamps an already collision-free
pile-aware start into the tray/blade-footprint box (no pile-occupancy awareness at all) --
its own comment already measured "35.8% of starts out of box and clamped". Per-row legality
flags saved next to every source file, originals untouched.

**Curation (B):** `model/retrieval/interaction.py::interaction_set` -- a truth-free geometric
affected-set rule (design doc section 6.1 step 1: blade-swept cubes + forward contact-chain
closure). The design doc's own tau in {1,2,3}mm reaches only 0.82-0.90 recall of the bank's
truth `moved` set on legal training rows; tuned to tau=12mm/angle_max_deg=60 for 0.960
recall / 0.899 precision. New dataset DS-0014 (`datasets/DS-0014-*`): DS-0008+DS-0010,
canonicalised identically to `TransitionBank.from_states`, plus `in_set` + the ISS-010 legality
flag + provenance per row (illegal rows kept and flagged, never deleted). New
`TransitionBank.load_curated` (`.build()`/`.from_states()` unchanged, so `model/retrieval_nfd`'s
own row-for-row bank assertion still holds) excludes illegal rows by default: 9,198/11,921
(77.2%) legal.

**Fix (C):** `RetrievalPredictor._transfer_one` used to Hungarian-match ALL n query cubes to
ALL n donor cubes with no distance limit -- a query cube far from everything could inherit a
large, nonsensical displacement from whatever donor cube it was force-paired with (visibly the
q0908 bug: fixed in the re-rendered figure, `figures/retrieval_debug_postfix/q0908_near_wall.png`
vs the original `figures/retrieval_debug/q0908_near_wall.png` -- the spurious ~40mm jump in
panel 4 is gone). Fixed UNCONDITIONALLY (not behind a flag: grep confirmed every direct caller
of `RetrievalPredictor`/`NearestTransitionPredictor` is inside this experiment's own code): both
sides of the match restricted to their own interaction set (`bank.in_set` when curated, else
old behaviour for backward compatibility) and every matched pair gated at 6mm (farther -> no
transfer). New tests: `test_far_cube_never_moves_after_distance_gate_fix`,
`test_exact_match_recovery_still_works_after_distance_gate_fix`
(`tests/test_retrieval_predictor.py`), 6 new tests in `tests/test_retrieval_interaction.py`.
Full retrieval suite: 40/40 pass.

**Re-evaluation (D), DS-0009 1-step / DS-0013 terminal 3-step, all rows vs LEGAL-only:**

| config | acc1 (all/legal) | slateN DS-0009 (all/legal) | mm (all/legal) | DS-0013 terminal slateN/tough (all/legal) |
|---|---|---|---|---|
| persistence | 0.000/0.000 | 0.059/-0.129 | 6.95/6.83 | 0.080-0.075 / -0.030--0.027 |
| retrieval_1nn, BEFORE (frozen, this record's own R0 number) | 0.172 (all only) | 0.413 (all only) | -- | -- |
| retrieval_1nn, AFTER (transfer fix only, still the OLD uncurated bank) | 0.337/0.452 | 0.759/0.827 | 3.72/3.43 | not rerun |
| retrieval_k5_cube_median, BEFORE (R1-best, 25k bank; frozen, "R1 addendum" above) | -- | 0.513 (all only) | -- | 0.529/0.546 (all only) |
| retrieval_k5_cube_median, AFTER (fix + curated 9.2k bank) | 0.330/0.493 | 0.754/0.823 | 3.33/2.57 | 0.786-0.794 / 0.811-0.824 |
| nfd_3ch_narrow_l20 (register C-056, EXP-0053) | 0.506/0.638 | 0.774/0.750 | -- | not rerun |

**SUPERSEDES** this record's own "Known result" framing above (`retrieval_1nn` accuracy_1 0.172/
slateN 0.413, and the R1/data-scaling `k5_cube_median` numbers): those numbers are not wrong
about what THAT code did, but the code had a real transfer bug the whole time, so they
understated retrieval's true ceiling by roughly 2-4x on every headline metric. Old numbers above
are kept (not deleted) and are the honest historical record of what was measured before the fix;
this section is the corrected, current one. **Bug found, in this codebase, this pass** (see
"Bugs / issues found in others' code" below for the running tally): the
`_pile_aware_stops` sampler clamp (ISS-010) and the `_transfer_one` unconditional-match bug --
both now fixed/mitigated, neither found by the original R0/R1/data-scaling work.

**Not completed this pass** (explicit scope cuts, not silent gaps): the 25k bank was not rebuilt
on curated data (12k curated already answers the headline question; "if cheap" was the
coordinator's own hedge); the narrow NFD's OWN train/eval data was not re-audited for ISS-010
(noted as open in that issue); `retrieval_1nn` was not rebuilt on the curated bank (only
legacy-bank+fix was run, to isolate the two effects); DS-0013 legal-only multistep was not rerun
for the NFD family.

## Threats

- `imprecision`: chaos-floor cells are a single n=64 sample each, no repeat/bootstrap CI; the
  0.428 (0.5mm) vs 0.452 (`linear_narrow_l20_res64`) comparison in particular is not powered to
  say more than "close."
- `incomplete-design`: crowd_floor value function and a slateN chaos floor were both scoped by
  the brief as optional/secondary and were dropped for budget; both are named above with cost.
- `provenance`: the chaos floor's re-simulation code path (`chain_collection.py`'s direct
  `set_particle_state`/`execute_action`, no snapshot/restore) is DELIBERATELY different from
  EXP-0024's (`rollout_candidates`/snapshot-restore) -- chosen because it matches how DS-0009
  itself was actually collected, but it means this record's 0mm number is not directly comparable
  to EXP-0024's `ratio_within_over_between`; they are two different quantities on two different
  mechanisms, stated as such in "Why this test discriminates" and not conflated in `result`.
- Single seed throughout (harness and chaos floor).
- Dirty tree at run time (repo-wide concurrent edits from other agents this session, including
  `model/retrieval/` gaining a `RetrievalPredictor`/`distance.py` mid-session -- not imported by
  anything this record depends on; `eval_retrieval.py` itself changed on disk after this record's
  own `eval_extended.py` was written, but `eval_extended.py` imports only `PersistencePredictor`/
  `NearestTransitionPredictor` from `model/retrieval/predictor.py`, both unchanged).

## Bugs / issues found in others' code

None found in `model/retrieval/{bank,predictor,frame}.py` (read in full while building
`eval_extended.py`/`chaos_floor.py`) or in `experiments/EXP-0059-*/code/{build_bank,
eval_retrieval}.py`. The only quirk found (persistence-key collision) is in THIS record's own
new `eval_extended.py`, described above and left unfixed as cosmetic (both paths agree
mathematically; only the mm-error field is missing for that one baseline row).

**2026-09-28, later (coordinator follow-up, see the addendum above): two real bugs found and
fixed/mitigated.** (1) `Genesis/sandbox_manipulation_clean.py::_pile_aware_stops` clamps an
already collision-free pile-aware touchdown into the tray-wall sampling box with zero
pile-occupancy awareness, putting the tool ON a cube ~44-56% of the time across every
`pile_aware`-sampled dataset (ISS-010, `experiments/OPEN_ISSUES.md`) -- mitigated by curating
the bank (DS-0014) to exclude flagged rows, NOT fixed at the sampler itself (still open). (2)
`model/retrieval/predictor.py::RetrievalPredictor._transfer_one` Hungarian-matched every query
cube to every donor cube with no distance limit, so a far query cube could inherit a large,
nonsensical displacement -- fixed (curated interaction-set matching + a 6mm distance gate,
unconditional). Neither was found by this record's own earlier rungs (R0/R1/R2/data-scaling) --
both are visible only once cube-level ground truth is actually inspected by eye, which is what
prompted this follow-up (`retrieval_debug.py`'s figures).

## Unrelated findings

None.

## Clean-data v2 rung (2026-09-28): fair train/val/test comparison on the ISS-010-fix corpus

**Why this is a new section in EXP-0059, not a new EXP-####** (per `experiment-log`'s guidance):
the scientific question is unchanged from the rest of this record -- does the fixed retrieval
transfer beat narrow NFD / LinearForesight on `slateN_tough` -- only the DATA changed (a fresh
collection, same shape/physics/push-length as DS-0008/DS-0009/DS-0011, with the ISS-010 sampler
bug actually fixed at the source rather than mitigated by post-hoc bank curation). This is a new
COMPOSITION of the same experimental design over a new dataset, not a new capability or a new
claim, so it belongs here as a clearly-separated rung, matching this file's own precedent (R1,
R2, data-scaling, post-fix-reevaluation were all added as new sections rather than new EXPs).

**New datasets** (registered by the data agent this session): DS-0015 (train, `train_v2`,
12,032 rows / 11,659 after `exclude_flagged`), DS-0016 (test, `test_chains_v2` +
`test_pools_v2`, plus a whole-sequence-clean `test_chains_v2_clean` for rollout), DS-0017 (val,
`val_pools_v2`, 2,048/2,048 clean). Full-scale audit confirms **0% illegal touchdowns** in every
v2 corpus checked (train/test chains and pools) -- the ISS-010 sampler fix (this record's own
earlier "Coordinator follow-up" section) holds at production scale, not just in the 256-row
pre-launch smoke test. Remaining exclusions are `valid==False` (push-length/perpendicularity
redraw exhausted, ~2-4%) and null transitions (subset of those), handled by
`Genesis/training/dataset.py::PileSweepData`'s new `exclude_flagged` option (this record) and
`eval_extended.py --exclude-flagged` (same criteria, applied to the harness's chain/pool loaders).

**Rollout needs whole clean chains, not row-filtered ones** (the data agent's note, matching
`reeval_ds0009.py`'s own earlier documented reasoning): `accuracy_1`/mm use `test_chains_v2` with
`--exclude-flagged` (986/1024 rows, single-step, continuity not required); `rollout_accuracy_1-4`
use the separately-built `test_chains_v2_clean` (896 rows / 112 WHOLE 8-step sequences, 16/128
sequences dropped entirely rather than punctured); `slateN`/`slateN_tough` use `test_pools_v2`
(2,048/2,048, 100% clean already). **Bug found and fixed in this record's own `eval_extended.py`
while wiring this up**: its rollout loop derived `E = chain_env.max()+1` and looped `range(E)`,
which IndexErrors (`rows[0]` on an empty list) the moment a whole-sequence-clean corpus leaves
gaps in `chain_env` (confirmed on `test_chains_v2_clean`: e.g. chunk 2 has only 26 of 31 possible
env indices present). Fixed by skipping empty `rows` in both `eval_occ_model`'s and
`eval_particle_model`'s rollout loops -- a genuine gap, not present in any dataset this harness
had been pointed at before (every prior chains corpus was row-filtered or fully populated, never
whole-sequence-filtered).

**Same rows throughout**: NFD's own train split is 10,653 rows (a further internal 5%/5%
val/test carve-out of `train_v2`, `PileSweepData`'s own file-hash split, unchanged mechanism from
v1); LinearForesight and the retrieval bank use the full `train_v2` corpus directly, 11,659 rows
-- confirmed identical between `fit_switched.py`'s own reported row count and
`build_bank_v2.py`'s bank size.

**VAL selection** (`code/val_select_v2.py`, `results/val_select_v2.json`, all on `val_pools_v2`
`slateN_tough`, never test): retrieval k=5 (`cube_median` aggregation, R1-best gate config held
fixed) beats k=1 clearly -- 0.782 vs 0.758 -- so **k=5 `cube_median` is the primary retrieval
config carried to test, k=1 the pre-declared secondary**. NFD checkpoint confirmation swept all
6 `save_every_n_epochs` checkpoints plus the best-val-loss one (epoch 57): VAL `slateN_tough`
bounces noisily between 0.716-0.776 with NO clear monotone trend (epoch10 0.741, ep20 0.768,
ep30 0.749, ep40 0.725, ep50 0.716, ep57-best 0.722, ep60 0.776) -- **the best-val-loss checkpoint
is kept as the confirmed choice**; picking whichever epoch happened to score highest in this
single 32-pool VAL draw would be selecting the max of 7 noisy comparisons, not a principled
checkpoint criterion, and the spread (0.716-0.776) is not obviously outside plausible per-draw
noise at this pool count.

### Test results (`code/test_v2.py`, `results/test_v2.json`)

Lead metric `slateN`/`slateN_tough` per `experiments/METRICS.md`; paired pool-bootstrap 95% CIs
(`Baselines/common/paired_stats.py`-style resampling, reusing this record's own
`bootstrap_ci.py::paired_delta_ci` / EXP-0060's `pool_bootstrap_ci`, 2000 draws, the 32
`test_pools_v2` pools as the replication unit) for the two comparisons the brief asked for.

| model | train rows | val slateN_tough | test slateN (13g) | test slateN_tough [95% CI] | acc1 | rollout4 | mm |
|---|---|---|---|---|---|---|---|
| persistence | -- | -- | 0.009 | 0.005 [-0.060,+0.068] | 0.000 | 0.000 | n/a |
| random | -- | -- | 0.048 | 0.048 [-0.022,+0.124] | -0.112 | -0.148 | 57.67 |
| linear_narrow_l20_v2_res32 | 11,659 | -- (no selection needed) | 0.644 | 0.666 [0.585,0.746] | 0.482 | 0.342 | n/a |
| linear_narrow_l20_v2_res64 | 11,659 | -- | 0.599 | 0.626 [0.535,0.718] | 0.520 | 0.385 | n/a |
| nfd_3ch_narrow_l20_v2 (best-val-loss, ep57) | 10,653 | 0.722 | 0.710 | 0.731 [0.646,0.804] | 0.557 | 0.332 | n/a |
| retrieval_k1_v2 (secondary) | 11,659 (bank) | 0.758 | 0.751 | 0.757 [0.706,0.801] | 0.415 | 0.300 | 3.32 |
| **retrieval_k5_cube_median_v2 (primary, VAL-selected)** | 11,659 (bank) | **0.782** | **0.804** | **0.792 [0.755,0.829]** | 0.485 | 0.342 | 2.51 |

**Paired deltas (slateN_tough, shared 32-pool resample):**
- retrieval_k5 vs `linear_narrow_l20_v2_res64`: **+0.167, CI [+0.090,+0.244], excludes 0 (100%
  of bootstrap draws favour retrieval)** -- retrieval clearly beats linear on this clean corpus,
  same conclusion the post-fix DS-0009 re-evaluation reached, now with a proper CI and a
  from-scratch clean train/val/test split rather than a shared-with-training-bugs legacy corpus.
- retrieval_k5 vs `nfd_3ch_narrow_l20_v2`: **+0.062, CI [-0.010,+0.137], 95.4% of draws favour
  retrieval** -- the CI's lower edge sits just below zero, so this is NOT resolved at the
  standard 95% threshold at this pool count (32), but the direction and magnitude are consistent
  with the post-fix DS-0009 finding (retrieval matching or slightly beating the narrow NFD on
  `slateN_tough` while trailing it on `accuracy_1`, 0.485 vs 0.557 here) -- read as "retrieval is
  at least competitive with, and directionally ahead of, the narrow NFD," not as a confirmed win.

**`accuracy_1` diverges from `slateN`/`slateN_tough` exactly as METRICS.md warns it can**:
`nfd_3ch_narrow_l20_v2` has the BEST `accuracy_1` (0.557) but only 3rd-best `slateN_tough`
(0.731, behind both retrieval configs); `retrieval_k5_cube_median_v2` has a middling `accuracy_1`
(0.485, similar to both linear variants) but the best `slateN_tough` (0.792) -- reproducing this
record's own earlier "lead with slateN" guidance, not a new finding, but a useful independent
confirmation on a second, cleaner corpus.

**Floors behave as designed**: `persistence` and `random` both sit at `slateN_tough` ~0.005-0.048
with CIs comfortably straddling 0 -- consistent with `slateN`'s own definition (capture relative
to a random pick, METRICS.md) making both floors land near exactly 0 by construction, not by
tuning. `random`'s `accuracy_1` is slightly NEGATIVE (-0.112, worse than persistence's 0 by
construction) and its moved-cube mm error is enormous (57.7mm, vs 2.5-3.3mm for the retrieval
configs) -- both expected for particles scattered uniformly at random over the whole workspace.

**Threats specific to this rung**: (1) NFD checkpoint sweep (7 points, one VAL draw) is
underpowered to rule out a real epoch-dependent trend, only to say the OBSERVED trend here is not
clean enough to override best-val-loss selection: named as `imprecision`. (2) retrieval-vs-NFD's
CI marginally includes 0 -- reported as unresolved, not rounded up to "retrieval wins." (3) The
gate-distance half of the pre-declared retrieval grid (6mm default) was held fixed, not swept,
per the coordinator's "small grid" instruction -- so this rung confirms k=5 beats k=1 at that one
gate setting, not that k=5/6mm is jointly optimal.

**Code changed this rung** (all described above, all reused/extended from the existing harness,
no new measurement machinery invented): `Genesis/training/dataset.py::PileSweepData` gained
`exclude_flagged`; `registry/dataset_registry.py` and `Baselines/NFD/nfd_lib.py` forward it;
`Baselines/NFD/configs/nfd_3ch_narrow_l20_v2.yaml`, `configs/dataset/genesis_narrow_l20_{train,
test_chains}_v2.yaml` (new, byte-identical recipes to their v1 counterparts otherwise);
`code/build_bank_v2.py` (new, generalises DS-0014's curated-bank builder to an arbitrary
directory); `eval_extended.py` gained `--exclude-flagged` and the empty-`rows` rollout-loop fix
(bug, described above -- a DIFFERENT bug from the pools-glob-vs-sidecar-files one this same file
already documents fixing earlier this session, not a re-occurrence of it); `simple_mpc/
adapters.py` gained `nfd_3ch_narrow_l20_v2(_epoch{10,20,30,40,50,60})` and
`linear_narrow_l20_v2_res{32,64}` `OCC_ADAPTERS` entries; `code/val_select_v2.py`,
`code/test_v2.py` (new, this rung).
