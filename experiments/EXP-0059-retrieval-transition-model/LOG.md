# EXP-0059 -- running log

Chronological record of the R0 rung (harness, baselines, ceilings) for the retrieval
transition model. Numbers live in `results/`; this file says what was attempted, in what
order, and what state things are in.

## 2026-09-28 -- R0 rung

**Goal:** build an extended eval harness that scores ANY model (occupancy-in/out OR
particle-in/out) on the same accuracy/rollout/slateN scale, plus two new diagnostics
(blurred accuracy, moved-cube mm error), run it on the EXP-0053 register models +
`persistence` + the coder's `retrieval_1nn`, and separately measure a Genesis
chaos-floor ceiling (resimulation noise vs state-uncertainty).

1. Read `docs/CODEMAP.md`'s existing EXP-0059 entry, `experiments/EXP-0053-*/code/
   eval_narrow.py`, `experiments/EXP-0059-*/code/{build_bank,eval_retrieval}.py`, and
   `model/retrieval/{bank,predictor,frame}.py` in full before writing anything --
   found no bugs in any of these (see EXPERIMENT.md "Bugs / issues found in others'
   code").
2. Wrote `code/eval_extended.py`, reusing `eval_narrow.py`'s `acc`/`swept_region`/
   `GOALS`/`D` and `eval_retrieval.py`'s particle-scoring structure verbatim. Added:
   an 8-goal "tough" set (`letter_O/T/S/X/L/I`, `two_squares`, `quadrant_0`, per
   EXP-0055/0057) alongside the original 13-goal set; Gaussian-blur (sigma 1, 2 px)
   accuracy diagnostics applied identically to every model; moved-cube mm position
   error for particle models (direct per-cube correspondence, no Hungarian matching
   needed -- the sim never reorders particles).
3. First run failed: ran from the wrong CWD (`experiments/EXP-0059-*/code/`), so
   `Baselines/common/goals.py::letter_mask`'s relative asset path
   (`env/target_shapes/...`) didn't resolve. Reran from the repo root -- not a bug,
   the same relative-path convention every other script here uses.
4. `eval_extended.py` ran clean on `persistence, nfd_3ch_narrow_l20,
   nfd_3ch_narrow_l20_wide, linear_narrow_l20_res64, nfd_residual_worldframe_noaug_
   ep43` (occ path) + `persistence, retrieval_1nn` (particle path), ~4 min total.
   Cross-checked against `EXP-0053/results/offline_eval.json` and `EXP-0059/results/
   offline_eval_retrieval.json`: identical to reported precision for every shared
   metric -- the extension reproduces, doesn't just add columns.
   Found (own harness, not others' code): the particle-path `persistence` predictor
   shares its output JSON key with the occ-path `persistence` model, so the
   particle-path one was silently skipped as "already scored." Harmless (both are
   mathematically `predict no change`) but means no moved-cube-mm number exists for
   the persistence baseline. Left as-is (cosmetic), documented in EXPERIMENT.md.
5. Wrote `code/chaos_floor.py` for the Genesis re-simulation ceiling, mirroring
   `Genesis/chain_collection.py`'s exact seam (`TRAINING_PHYSICS`,
   `sim.set_particle_state` + `sim.execute_action` + `sim.update_material_state`, no
   snapshot/restore -- deliberately NOT EXP-0024's `rollout_candidates` mechanism,
   since chain_collection's is what actually produced DS-0009). Sampled 64 DS-0009
   `test_chains` rows (32 scatter + 32 clump, seed 0); for perturbation levels
   {0, 0.5, 1, 2} mm, perturbed the recorded start state's xy (+1 deg yaw jitter for
   levels > 0) before re-executing the SAME recorded action, and scored the result
   against the SAME recorded true outcome with the identical accuracy/blur/mm code
   `eval_extended.py` uses.
6. First run crashed: `Expected all tensors to be on the same device, but found at
   least two devices, cuda:0 and cpu` inside `eval_narrow.swept_region`. Cause:
   Genesis's `gs.init()` (triggered by building `GenesisOracleEnv`) sets torch's
   default device to `cuda` as a documented side effect (`data-collection` skill's
   "Genesis sets torch's default device to cuda" trap) -- `swept_region`'s internal
   `torch.linspace` calls then silently create cuda tensors while this script's own
   CPU-loaded `act`/state tensors stayed on cpu. Fixed by calling
   `torch.set_default_device("cpu")` right after building the env (every further
   Genesis call in the script already explicitly moves its own tensors `.to(gs.device)`,
   so this only affects this script's own bookkeeping code, not the sim). This is a
   bug in **my own new script**, not in `eval_narrow.py`, `Genesis/chain_collection.py`,
   or anyone else's code -- `eval_narrow.py` has never previously been imported into a
   process that also builds a live Genesis scene.
   IMPORTANT PROCESS NOTE: EXPERIMENT.md was drafted with the chaos-floor "Numbers"
   section filled in from anticipated/estimated values BEFORE this crash was noticed
   in the tool output -- caught immediately (before being reported to the orchestrator)
   and replaced with a PENDING placeholder; the real numbers below are from the actual
   corrected-run JSON, not the earlier fabricated draft. Recorded here so the process
   error is visible, not just quietly overwritten.
7. Reran after the fix -- see EXPERIMENT.md's Numbers section and `results/
   chaos_floor.json` for the actual output.

## 2026-09-28 -- mid-task coordinator additions

Two messages arrived while R0 was running, both addressed before closing out:

**(A) Reframing + metric-correlation diagnostics ("fold in if cheap").** Reworded
EXPERIMENT.md's Task 1+2 table to lead with `slateN`/`slateN_tough` (per
experiment-log's own metrics rule, which this record had already been treating
`accuracy` as diagnostic-only per the brief). Added two new per-model columns,
cheap to compute on data already gathered: `occ_emd_swept` (sliced-W1 distance
between predicted and true occupancy, swept region only, reusing
`simple_mpc.value_functions.SlicedEMD`'s fixed pixel-projection order
generalised to two arbitrary batched images instead of one image vs a fixed
target -- new code, `eval_extended.py::sliced_w1_pair`) and `mass_in_goal_mae`
(mean |mass_in_region(pred) - mass_in_region(truth)| over the same 13-goal /
8-goal pool loop already computing `slateN`, reusing `Baselines.common.goals.
mass_in_region` unchanged). Reran `eval_extended.py` on all 7 models/predictors
-- see EXPERIMENT.md for the resulting table and the early metric-disagreement
it surfaces (linear_narrow_l20_res64 scores WORSE than persistence on both new
diagnostics despite positive accuracy_1/slateN; the broad residual NFD has the
best occ_emd_swept despite ranking 4th on slateN). Not resolved here -- flagged
as input to the design doc's section-4 metric study, not a conclusion (5-6
models is not powered to correlate anything).

Multi-step ranking test-set search (requested item 2): checked
`Genesis/data/slates_multistep/{n20_L10mm,n20_L20mm,n20_L40mm}` (`same_state_
slate_collection.py`) and EXP-0010/EXP-0014's use of it. Finding: `slates_
multistep` gives a genuine 128-candidate SAME-STATE ranking pool at step 1
only (rows then diverge onto their own 3-step chains, useful for rollout
accuracy but not for ranking at horizon 2+), and its collector defaults to
friction 0.3 / density 1000 (NOT narrow-domain `TRAINING_PHYSICS`), so its
`slateN` would not be comparable to the EXP-0053 register row without
re-collecting under matched physics. DS-0009's own `test_chains` are single
fixed-path chains (no ranking pool at any step > 0). **Conclusion: no existing
corpus supports genuine multi-step RANKING (candidate pools at horizon > 1)
under narrow-domain physics.** While finishing this search, found (via a
concurrent `data-collection` skill file update, not this record's own work)
that the coder has since added `Genesis/chain_collection.py --mode seqpools`
-- exactly the DS-B multi-step test-pool shape the design doc's section 2.1
calls for -- so this gap is closing independently; not built or run here per
the coordinator's explicit "the coder will do it; don't."

**(B) Designer's plan + DS-0011/DS-0012 launch.** Read `docs/experimental_
design/retrieval_based_modeling.md` section 2 ("Tonight's plan") for the
exact recipe. Launched, in the true background (`setsid nohup ... &
disown`, so both survive this session ending):

- **DS-0011** (DS-A validation pools): `Genesis/chain_collection.py --mode
  pools --out Genesis/data/narrow_l20_n20/val_pools --n-chunks 1 --pool 64
  --starts mixed --clump-fn Genesis.clump_states:clump_starts --sampler
  '{"pile_aware": true, "min_swath_particles": 3, "push_length": 0.02}'
  --seed 201` -- PID 467354, log
  `<scratchpad>/ds0011_val_pools.log`. Same shape/physics/sampler as
  DS-0009's own `test_pools`, new seed (201, disjoint from 1/101/102).
- **DS-0012** (DS-C chain-shaped reservoir): `Genesis/chain_collection.py
  --mode chains --out Genesis/data/narrow_l20_n20/reservoir_dsC --n-chunks
  500 --steps 8 --n-envs 128 --starts mixed --clump-fn
  Genesis.clump_states:clump_starts --sampler '{"pile_aware": true,
  "min_swath_particles": 3, "push_length": 0.02}' --seed 301` -- PID
  467762, log `<scratchpad>/ds0012_reservoir.log`. **Simplification**
  (explicitly per the coordinator's folded-in instruction): chain-shaped
  only, same recipe as DS-0008 at `--n-envs 128` (vs DS-0008's 32) per the
  designer's "use 128 envs if stable" note; the design doc's additional
  "pool-shaped share" and "near-wall states" mix are NOT included in this
  launch. `--n-chunks 500` is a generous cap (128 envs x 8 steps = 1,024
  rows/chunk; 500 chunks would be 512k rows), expected to be stopped well
  before completion by the ~5h wall-clock budget, not by row count.
- Both wrote `DATASET.md` immediately (status `active`, before either job's
  own completion, per experiment-log's "IDs outlive their payload" --
  `datasets/DS-0011-narrow-l20-n20-val-pools/`, `datasets/DS-0012-narrow-l20-
  n20-reservoir/`).
- Bounded foreground check (~330s combined across two checks): both
  processes alive, no `Traceback`/`Error` in either log, both producing the
  expected per-step "pile-aware: N/{32,128} pushes ... shortened" progress
  warnings (a documented, non-fatal `chain_collection.py` behaviour, not a
  new bug), neither had written its first chunk (`_0_data.pt` /
  `pools_0.pt`) by the time this record closed. **ETA is genuinely unknown**
  -- DS-0008 (32 envs, 8 steps, 24 chunks) took long enough that no
  per-chunk timing was captured here either; the honest number is "check
  `manifest.json`'s `chunks` dict growth," not an estimate presented as
  measured. GPU was shared three ways (this record's own `eval_extended.py`
  rerun + both collection jobs) for part of this window, which plausibly
  slowed all three; `nvidia-smi` showed 6+ GB headroom throughout (never a
  memory risk, only a throughput one).

## 2026-09-28 -- DS-0011 finished; perturbed-sim zoo, raw arrays, DS-0011 scoring

Coordinator confirmed DS-0011 complete (`Genesis/data/narrow_l20_n20/val_pools`, 2048 rows,
`manifest.json` `complete: true`) and asked for three things:

1. **Launch `perturbed_sim_zoo.py`** (written earlier, deferred pending DS-0011) -- now safe
   since it only competes with DS-0012 + the coder's own sweep. Launched detached (`setsid
   nohup ... & disown`, PID 501949, log `<scratchpad>/perturbed_sim_zoo.log`). Its cheap,
   Genesis-free "model control" cell ran first and landed immediately: frozen
   `nfd_3ch_narrow_l20`'s own blurred-slateN on the SAME 32 DS-0009 pools --
   `blur0=0.778, blur1=0.784, blur2=0.773` (matches the already-recorded `slateN_tough=0.778`
   at blur0, a consistency check). Reading: for this already-good model, mild blur (sigma 1)
   barely moves ranking quality either way and sigma 2 is very slightly worse -- a sharp
   contrast with the same blur levels' large effect on raw `accuracy_1` (0.506 -> 0.611 ->
   0.704), i.e. blur helps the image-reconstruction metric far more than it helps (or hurts)
   the RANKING metric for this model. The Genesis-dependent perturbed-simulator cells (32
   pools x 64 pushes x 4 levels) were still running when this entry was written.
2. **Per-pool raw-array persistence** (the ~10-line change): `eval_extended.py`'s
   `eval_occ_model`/`eval_particle_model` now also return a second dict
   `{"slateN_raw": {goal: [{"pool": id, "vt": [...64 floats...], "vp": [...64 floats...]}, ...]},
   "slateN_tough_raw": {...}}` alongside the existing summary `r` -- one entry per (pool, goal),
   the exact true/predicted `dv` vectors `slate_capture` already computes internally but
   previously discarded after reducing to a single ratio. `main()` writes this to a SEPARATE
   file (`<out>_raw.json`, same atomic-write pattern) rather than bloating the summary JSON, per
   the artifact/result split (`experiment-log`'s "Artifact vs result"). This unblocks BOTH asks:
   a pool-bootstrap CI on `slateN`/`slateN_tough` (resample the per-pool capture values with
   replacement) and within-pool `Spearman(vp, vt)` (scipy on each `{"vt","vp"}` pair) -- neither
   is COMPUTED yet in this record (that's EXP-0060's job once the rerun lands), only the data
   they need is now saved. Relaunched all 6 already-scored models (persistence + 4 EXP-0053 occ
   models + retrieval_1nn) through the updated harness to backfill the raw file for them (moved
   the old summary JSON aside first as `.bak_no_raw` so nothing was silently overwritten); this
   rerun needs no Genesis and was explicitly the "cheap" one to run alongside DS-0012 +
   perturbed_sim_zoo.
3. **DS-0011 scoring support**: `eval_extended.py` gained `--pools-dir`/`--chains-dir`/`--out`
   CLI overrides and both eval functions now guard `ch == []` (DS-0011 has no chains
   equivalent) by setting `accuracy_1`/`accuracy_1_blur{1,2}`/`occ_emd_swept`/`n_test_rows` to
   `None`/`0` rather than crashing or fabricating a value -- `slateN`/`slateN_tough`/
   `mass_in_goal_mae` are computed exactly as on DS-0009, same keys, same code path. Smoke-tested
   on `persistence` against DS-0011's own pools (`--pools-dir .../val_pools --chains-dir
   .../val_pools`, the latter chosen only because it has no `_*_data.pt` chain files to match,
   giving `ch=[]` for free): `acc1 n/a (no chains)  slateN 0.019  slateN_tough 0.015` -- ran
   clean, no crash, correct "n/a" handling. Ready for the coder's retrieval variants to be
   frozen against DS-0011 the same way once they're built.

DS-0012 progress at this point: 5 of 500 chunks (5,120 rows) in ~33.5 min elapsed (~6.7
min/chunk steady-state, consistent with the earlier 2-chunk estimate) -- on pace for roughly
40-45 more chunks (~41-46k more rows) over the remaining ~4.4h, i.e. ~46-51k rows total by the
5h mark. Still short of the 150-250k stretch target; both collection jobs remain alive with no
errors.

## Status at end of budget

R0 rung complete: harness built and cross-checked (now also `occ_emd_swept`,
`mass_in_goal_mae`, per-pool raw arrays, and DS-0011 support), register-model
+ retrieval_1nn comparison table produced, chaos floor measured, perturbed-sim
zoo launched (model-control cell landed, Genesis cells in progress at record
time). DS-0011 complete; DS-0012 progressing (~46-51k rows projected by 5h).
Same-state leave-one-out retrieval ceiling remains NOT reached -- named in
EXPERIMENT.md's "What would change the verdict". See EXPERIMENT.md for the
full record and verdict, and EXP-0060 for the metric-correlation numbers.

## 2026-09-28 -- fresh instance (rate-limit handoff): recorded completed zoo + diagnostics

Picked up as a fresh EXPERIMENTER instance; both jobs the previous instance left running had
finished and their JSON was already on disk (`results/perturbed_sim_zoo.json`'s `levels_mm`
section complete for 0/0.5/1mm x blur0/1/2, `results/diagnostics_r2.json` complete for all 6
sections) -- no new computation needed, this entry only records numbers that existed but were
not yet written into EXPERIMENT.md.

1. Read both JSON files, cross-checked the headline numbers against the coordinator's briefed
   values (0mm 0.932, 0.5mm 0.923, 1mm 0.863 `slateN_tough`; oracle best-of-top-50 1.98mm/acc1
   0.24/slateN 0.53, random-50 2.5mm/slateN 0.28) -- all matched exactly.
2. Added a "R2 addendum" section to EXPERIMENT.md's Numbers with the full per-blur-sigma
   `slateN`/`slateN_tough` table for the perturbed-sim zoo and all six `diagnostics_r2.py`
   sections (oracle-donor mm ceiling at N=50/500/whole-bank, same-state leave-one-out ceiling,
   transfer-rule comparison, moved-cube-in-window check, oracle-vs-random-50 slateN, and the
   action-restricted leave-one-out). Removed the two now-resolved bullets from "What would
   change the verdict" (perturbed-sim `slateN` scoring, same-state leave-one-out ceiling) and
   added Section 1's small subsample (n=36) as a residual imprecision note.
3. Did not rerun either script -- both were complete, reruns would have cost GPU time for zero
   new information and this record's own instructions were to spend at most ~20 min on this
   item before moving to the data-scaling task.
4. Appended the zoo's flattened rows to EXP-0060's `correlate.py` sources (see EXP-0060's own
   LOG/EXPERIMENT for that side) -- `ingest_perturbed_sim_zoo.py` already existed for exactly
   this, was not yet run against the (at-that-time incomplete) zoo file.

## 2026-09-28 -- same instance: DS-0012/DS-0013 registration, data scaling, multi-step headline

5. Updated `datasets/DS-0012-*/DATASET.md`: reservoir stopped at 84 chunks/86,016 rows
   (deliberate, to free the GPU for the coder), `complete: false` reflects the stop not a
   failure. Registered `datasets/DS-0013-narrow-l20-n20-seqpools/DATASET.md` (DS-B multi-step
   pools, 32 pools x 64 sequences x 3 pushes = 6,144 rows, COMPLETE, produced by the coder).
6. **Task 1 (data scaling):** wrote `code/data_scaling.py`, built 5 banks (persisted to
   `artifacts/bank_{name}.pt`), scored `k5_cube_median` on each via a small non-invasive
   extension to `eval_extended.eval_particle_model` (`predictor_obj=` override, so an arbitrary
   `RetrievalPredictor` config can be scored without adding a name to the hardcoded dict). Full
   results + the "slateN_tough peaks at 25k, declines at 50k/98k despite acc1/mm/oracle all
   improving" finding are in EXPERIMENT.md's "Task 1 addendum." One contamination note: the 50k
   cell's wall time (881.8s) overlaps with a `multistep_eval.py` smoke test this record launched
   concurrently and then killed once noticed (system load average hit 26 on 20 cores) -- flagged
   in EXPERIMENT.md, not silently left in the reported number's context.
7. **Task 2 (multi-step headline):** wrote `code/multistep_eval.py` (DS-0013 pool rollout,
   predictions fed back, terminal + per-step slateN_tough + per-step rollout accuracy). Ran all
   7 required models (5 occ + retrieval_1nn + k5_cube_median@25k, the task-1 winner). Headline:
   `linear_narrow_l20_res64` is the only model whose ranking quality IMPROVES with horizon and
   ends up edging out `nfd_3ch_narrow_l20` at 3 steps despite trailing badly at 1 step -- see
   EXPERIMENT.md's "Task 2" section.
8. **Task 3 (coder's model):** coder's own `results/coder_status.md` reports a NEGATIVE
   headline (retrieval_nfd_ref worse than persistence, zeroed-ref control statistically
   indistinguishable from it, both on DS-0009). Independently recomputed its `slateN_tough` from
   `results/offline_eval_extended_raw.json`'s persisted per-pool arrays with this record's OWN
   aggregation code -- matched the coder's reported numbers to 9+ significant figures, ruling
   out an aggregation bug as the cause. Multi-step run BLOCKED (not silently skipped): `model/
   retrieval_nfd/render.py` was mid-edit (renamed `render_cubes_soft`, breaking `predictor.py`'s
   still-old import) when this record tried to import it via `make_occ_adapter` --
   `model/retrieval_nfd/` is the coder's live working file per the coordinator's explicit "don't
   touch its files," so this record did not attempt a fix, only diagnosed the exact cause and
   left a retry note in EXPERIMENT.md.
9. Kept `results/experimenter_status.md` current after each task (single table, every model x
   {1-step slateN_tough, multi-step terminal slateN_tough, acc1, rollout, mm}).
   `scripts/check_register.py` exits 0 throughout.

## 2026-09-28 -- coordinator round 2 (hard stop 08:10, offline/CPU only, GPU reserved for coder)

10. **Pool-bootstrap 95% CIs, both headline claims REWORDED as a result.** Wrote
    `code/data_scaling_pools_raw.py` (reruns ONLY the DS-0009 pools loop, `ch=[]`, against the
    5 already-persisted banks, CPU-forced via `CUDA_VISIBLE_DEVICES=`) and extended
    `code/multistep_eval.py` to persist per-pool terminal-step raw arrays (reran the 5 occ
    models, CPU). `code/bootstrap_ci.py` reuses EXP-0060's `pool_bootstrap_ci`/`_capture`
    unchanged, plus a new PAIRED-delta bootstrap (shared pool resample across the two compared
    models/configs, since both share the same 32 DS-0009 pools). Result: **every data-scaling
    bank's CI overlaps every other's almost completely** (25k [0.499,0.672] vs 98k
    [0.473,0.666]; paired delta 98k-25k = -0.018, CI [-0.067,+0.030], includes 0) -- the earlier
    "slateN_tough DECLINES past 25k" claim is downgraded to "FLAT beyond ~12-25k, not resolved
    by this underpowered (32-pool) design." Same correction for multi-step: narrow-NFD vs
    linear64's terminal CIs overlap almost completely (paired delta +0.007, CI [-0.034,+0.049],
    63.3% of bootstrap draws favour linear64) -- "linear64 edges out narrow NFD" downgraded to
    "statistically indistinguishable at this sample size; point estimate mildly favours
    linear64." Both EXPERIMENT.md sections rewritten in place with the corrected language,
    original (overclaimed) wording struck through/replaced rather than silently deleted.
11. **Neighbour-rank curve (information-vs-noise control).** New `code/
    neighbour_rank_curve.py`: ONE shared CPU distance search (256 `test_chains` queries + all 32
    `test_pools`, ~430s) against the 25k bank, then donor selection at distance rank r in {1, 5,
    20, 100, 1000, random} (displacement-only transfer, k=1 style, no k-way aggregation).
    Clean, near-monotone decline on all three metrics as r increases (accuracy_1 0.197 -> 0.006,
    mm 4.22 -> 6.48, slateN_tough 0.516 -> 0.302) -- retrieved information is doing real
    work, confirming the retrieval family isn't just benefiting from "having a big bank" for
    its own sake.
12. **Confidence model.** New `code/confidence_model.py`: on all 2048 DS-0009 `test_pools` rows
    (32 shared starts x 64 candidate actions each), `predict_particles_with_confidence`'s
    `knn_disagreement` signal correlates positively with both moved-cube mm error (rho +0.217)
    and swept-region proxy error (rho +0.449) at row level, and moderately within-pool (mean rho
    0.22/0.35) -- a genuinely usable "which actions does the bank predict least well" signal.
    `top1_dist` does NOT: weak/near-zero vs mm error (+0.056) and NEGATIVELY correlated with
    swept error (-0.226, CI excludes 0) -- its top-10%-least-confident decile actually has BELOW
    average swept error (lift 0.75x), the opposite of the intended use. Recommendation:
    `knn_disagreement`, not `top1_dist`, for this deliverable.
13. **Coder's model, checked repeatedly, still not ready.** `model/retrieval_nfd/predictor.py`
    was updated (06:45) to import the renamed `render_cube_boxes_batch` (the coder appears to be
    replacing the soft-union renderer with an exact box renderer -- likely their own next step
    after diagnosing the channel-0 train/eval mismatch, per their own status file's "what likely
    went wrong"), but `results/coder_status.md` itself was not updated past 06:21 by the time
    this record's budget ran out -- no new checkpoint/milestone to run multi-step on. Checked at
    06:44, 07:10, and again before wrap-up; still the same status each time.
14. All new/modified scripts run with `CUDA_VISIBLE_DEVICES=` (hides the GPU entirely) per the
    coordinator's explicit "offline only, GPU is the coder's for training" -- verified via an
    `assert not torch.cuda.is_available()` guard at the top of each new script's `main()`, not
    just a convention comment.
15. **Coder's model became ready mid-session (4 real bugs fixed, retrained on the FULL
    DS-0008+DS-0010 corpus).** Independently re-verified the new (real, non-degenerate) 1-step
    numbers the same way as before (raw-array recompute, matched to 8-9 sig figs). Ran
    `multistep_eval.py --models retrieval_nfd_ref retrieval_nfd_random_donor
    retrieval_nfd_ref_zeroed --out results/multistep_eval_coder.json` (CPU-forced, ~972s total
    for all 3). Result: terminal slateN_tough ref=0.642, zeroed=0.613, random_donor=0.553 --
    `ref` (retrieval ON) now beats `zeroed` at 3-step horizon, the OPPOSITE order from 1-step
    (zeroed 0.715 > ref 0.686 there) -- flagged as a point-estimate-only reversal, no CI computed
    for these 3 models (budget ran out), explicitly NOT claimed as significant given the
    ~0.06-0.09-wide CIs found for the occ-model family above. Replied to the coder's agent
    message (`acc539a6e5925141c`) confirming the cross-check and that the DS-0013 numbers were
    folded in.
16. **Past this record's own 08:10 hard stop, one more coder message arrived** (~08:14):
    `retrieval_nfd_random_donor` was not a valid information-vs-noise control (it conflates
    "wrong reference" with "no reference"); the coder retrained a fair `retrieval_nfd_noref`
    twin (donor channels zeroed in BOTH training and eval) at the full 60-epoch schedule and
    reran `ref`/`noref`/`ref_zeroed` -- verdict "a wash" (DS-0011 favours `ref`, DS-0009 and
    `acc1` favour `noref`). Added a clearly marked SUPERSEDED addendum to EXPERIMENT.md (did not
    delete or silently rewrite the earlier `random_donor`-based numbers) relaying the coder's new
    table, explicitly flagged as NOT independently re-verified by this record (hard stop already
    passed) and NOT rerun through multi-step. `retrieval_nfd_noref` is now the correct baseline
    for any future information-vs-noise work on this record; `random_donor` is retired from that
    role (its checkpoint still exists for a separate noise-vs-nothing question if wanted).

## 2026-09-28 ~08:40 -- 60-epoch NFD-with-reference vs no-reference twin (orchestrator note)

Both trained the narrow NFD's full 60-epoch schedule on DS-0008+DS-0010 (coder; details and
checkpoints in results/coder_status.md). slateN_tough DS-0011 / DS-0009:
`retrieval_nfd_ref` 0.790 / 0.752, `retrieval_nfd_noref` (donor channels zero at train and eval)
0.752 / 0.778, `retrieval_nfd_ref_zeroed` 0.777 / 0.775; acc1 0.478 / 0.486 / 0.496; rollout4
0.324 / 0.322 / 0.326. Sign flips between the two datasets; single seed, no paired CI -> no
evidence the retrieved reference helps (a wash). Supersedes reading the 20-epoch
ref > random-donor gap as "information": that gap showed random donors HURT, not that real ones
help. Multi-step on the 60-epoch pair not run.

## 2026-09-28 (later) -- coordinator follow-up: tool-placement legality audit (ISS-010), curated interaction-set bank (DS-0014), transfer-gate bug fix, post-fix re-evaluation

User noticed, in `retrieval_debug.py`'s visual-debug figures (`q0908`), the tool touching down
directly ON a cube -- illegal in the spirit of pile-aware sampling. Four-part follow-up, in
order: audit, curate, fix, re-evaluate. Full details/numbers in `experiments/OPEN_ISSUES.md`
ISS-010, `datasets/DS-0014-retrieval-curated-interaction-sets/DATASET.md`,
`docs/CODEMAP.md`/`docs/ARCHITECTURE.md` (model/retrieval section). This entry is the summary.

**A. Legality audit** (`code/audit_tool_placement.py`, exact SAT test of the blade footprint
(40x2mm) vs every cube's rotated-square footprint (5mm) at `p_start`, extending
`Baselines/common/cube_overlap.py` with a rectangle-vs-rectangle SAT (`overlaps_rect_pairs`) --
verified numerically identical to the old same-size-square path). Roughly HALF of every
`pile_aware`-sampled dataset's rows are illegal: DS-0008 0.443, DS-0009 chains 0.462 / pools
0.561, DS-0011 0.478, DS-0012 0.458, DS-0013 0.517; DS-0010 (older, likely `placement_aware`)
only 0.0002 exact-overlap but 0.334 within a 1mm margin. Root cause read in
`Genesis/sandbox_manipulation_clean.py::_pile_aware_stops`: it unconditionally clamps a
genuinely collision-free pile-aware start into the tray/blade-footprint sampling box (pure
wall-margin, ZERO pile-occupancy awareness) because "the pile spreads well past its spawn
extent" -- its OWN comment already measured 35.8% of starts clamped. Per-row flags saved next to
every source file (`_k_data_legality.pt`); originals untouched.

**B. Curated bank** (`model/retrieval/interaction.py::interaction_set` -- truth-free geometric
"affected set", design doc section 6.1 step 1 only: blade-swept cubes + forward contact-chain
closure, batched Bellman-Ford-style relaxation, no ground truth needed). Tuned on training data
(DS-0008+DS-0010, LEGAL rows only): the design doc's own tau in {1,2,3}mm only reaches
0.82-0.90 recall of the truth `moved` set; swept to tau=12mm/angle_max_deg=60 (recall 0.960,
precision 0.899 pooled over cubes) to clear the requested ~95%. New dataset DS-0014
(`datasets/DS-0014-retrieval-curated-interaction-sets/`, `build_curated_bank.py`): DS-0008+
DS-0010 canonicalised exactly as `TransitionBank.from_states` already does, PLUS the
interaction-set mask (`in_set`) and the ISS-010 legality flag per row (illegal rows KEPT and
flagged, not deleted) and provenance. `TransitionBank.load_curated` (new classmethod,
`.build()`/`.from_states()` unchanged for `model/retrieval_nfd`'s own row-for-row assertion)
excludes illegal rows by default -- 9,198/11,921 rows (77.2%) legal. Saved as
`artifacts/bank_curated.pt` (`build_bank.py --curated`).

**C. Transfer-gate fix** (`model/retrieval/predictor.py::RetrievalPredictor._transfer_one`,
UNCONDITIONAL, not a flag -- every existing direct caller is inside this experiment's own code,
verified by grep). Before: Hungarian-matched ALL n query cubes to ALL n donor cubes with no
distance limit, so a far query cube could inherit a large, nonsensical displacement from
whatever donor cube it was force-paired with (visibly fixed in `q0908`'s panel 4 -- the ~40mm
spurious jump is gone in the re-rendered figure). Now: both sides restricted to their own
interaction set (`bank.in_set` when curated, else old behaviour) AND every matched pair gated by
`distance_gate` (default 6mm) -- farther apart transfers nothing. New unit tests
(`tests/test_retrieval_predictor.py::test_far_cube_never_moves_after_distance_gate_fix`,
`test_exact_match_recovery_still_works_after_distance_gate_fix`) plus 6 new tests for
`interaction_set` itself (`tests/test_retrieval_interaction.py`). Full retrieval suite: 40/40
pass (32 pre-existing unchanged + 8 new).

**D. Post-fix re-evaluation** (`code/reeval_ds0009.py` new; `code/multistep_eval.py` extended
with `--legal-only`). Before/after (DS-0009 1-step; DS-0013 terminal 3-step):

| config | bank | accuracy_1 (all / legal) | slateN, DS-0009 (all / legal) | moved-mm (all / legal) | DS-0013 terminal slateN / tough (all / legal) |
|---|---|---|---|---|---|
| persistence (before=after, by construction) | -- | 0.000 / 0.000 | 0.059 / -0.129 | 6.95 / 6.83 | 0.080/0.075 / -0.030/-0.027 |
| retrieval_1nn, BEFORE (v0, frozen CODEMAP number, all rows only) | legacy 12k | 0.172 | 0.413 | -- | -- |
| retrieval_1nn, AFTER (fix only, legacy uncurated bank -- isolates the fix from curation) | legacy 12k | 0.337 / 0.452 | 0.759 / 0.827 | 3.72 / 3.43 | not rerun (1nn multistep not repeated post-fix, budget) |
| retrieval_k5_cube_median, BEFORE (R1-best, 25k bank, frozen) | 25k | -- | 0.513 (DS-0009, old sweep) | -- | 0.529 / 0.546 (all rows only) |
| retrieval_k5_cube_median, AFTER (fix + curated 9.2k bank -- the actual "after") | curated 9.2k | 0.330 / 0.493 | 0.754 / 0.823 | 3.33 / 2.57 | 0.786/0.794 / 0.811/0.824 |
| nfd_3ch_narrow_l20 (register C-056: acc1 0.506, slateN 0.774, all-rows) | -- | 0.506 / 0.638 | 0.774 / 0.750 | -- | not rerun (out of this pass's scope) |

**Headline: the transfer bug, not the retrieval/aggregation approach or bank size, was the
dominant cause of retrieval's underperformance vs the narrow NFD.** Even the naive 1-NN
predictor, fix-only (no bank curation at all, still the old uncurated 12k bank), already reaches
DS-0009 slateN 0.759-0.827 -- matching or beating the narrow NFD's registered 0.774, up from the
old v0 number of 0.413 (>4x). The curated k5_cube_median (fix + curated 9.2k bank) reaches
accuracy_1 0.493 / slateN 0.823 on the LEGAL-only subset, essentially matching NFD's accuracy_1
(0.638, still higher) while BEATING its slateN (0.750) on the identical rows -- consistent with
this repo's own established finding that slateN (ranking-relevant), not accuracy_1
(pixel-relevant), is what should decide a model comparison (`experiments/METRICS.md`, C-038).
The k5_cube_median terminal multi-step slateN_tough on DS-0013 rises from 0.546 (old, BIGGER
25k bank, still buggy transfer) to 0.794-0.824 (new, SMALLER curated 9.2k bank, fixed transfer)
-- a bigger, cleaner bank than before was NOT needed; the transfer bug was the thing to fix.
Curiosity for later: `retrieval_k5_cube_median` sits marginally BELOW `retrieval_1nn` on
DS-0009 1-step (0.330/0.493 vs 0.337/0.452 acc1; 0.754/0.823 vs 0.759/0.827 slateN) even though
it is clearly ahead at the DS-0013 3-step horizon (0.794-0.824 vs not rerun) -- plausibly the
smaller curated bank (9.2k vs 12k) costs `cube_median`'s k=5 neighbourhood coverage more than
`nn1`'s single best match, not decomposed further here.

**Scope not completed this pass** (explicitly, not silently): the 25k bank variant (not rebuilt
on curated data -- "if cheap" was the coordinator's own hedge, and the 12k curated result already
answers the headline question); the narrow NFD's own train/eval legality audit (ISS-010 notes
this as still open); DS-0013 legal-only multistep only ran for persistence/k5_cube_median_curated,
not the NFD family; `retrieval_1nn` was not rebuilt on the curated bank specifically (only the
legacy-bank+fix combination was run, to isolate the two effects) -- a straightforward follow-up,
not run for time.
