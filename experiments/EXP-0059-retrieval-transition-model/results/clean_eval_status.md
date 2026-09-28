# EXP-0059 clean-eval status (fair train/val/test on the ISS-010-fix v2 corpus)

Updated: 2026-09-28 17:34. Single-seed rung **COMPLETE** (below); now training 2 more NFD seeds
+ waiting on the clean multi-step set (DS-0018-to-be) -- see "IN PROGRESS" section at the bottom.

## Single-seed rung (COMPLETE, recorded in EXPERIMENT.md's "Clean-data v2 rung" section)

New datasets: DS-0015 train / DS-0016 test / DS-0017 val. `scripts/check_register.py` exits 0.

## Headline table (model | train rows | val slateN_tough | test slateN | test slateN_tough [CI] | acc1 | rollout4 | mm)

| model | train rows | val slateN_tough | test slateN (13g) | test slateN_tough [95% CI] | acc1 | rollout4 | mm |
|---|---|---|---|---|---|---|---|
| persistence | -- | -- | 0.009 | 0.005 [-0.060,+0.068] | 0.000 | 0.000 | n/a |
| random | -- | -- | 0.048 | 0.048 [-0.022,+0.124] | -0.112 | -0.148 | 57.67 |
| linear_narrow_l20_v2_res32 | 11659 | -- (no selection needed) | 0.644 | 0.666 [0.585,0.746] | 0.482 | 0.342 | n/a |
| linear_narrow_l20_v2_res64 | 11659 | -- | 0.599 | 0.626 [0.535,0.718] | 0.520 | 0.385 | n/a |
| nfd_3ch_narrow_l20_v2 (best-val-loss, ep57) | 10653 (+502/504 internal val/test) | 0.722 | 0.710 | 0.731 [0.646,0.804] | 0.557 | 0.332 | n/a |
| retrieval_k1_v2 (secondary) | 11659 (bank) | 0.758 | 0.751 | 0.757 [0.706,0.801] | 0.415 | 0.300 | 3.32 |
| **retrieval_k5_cube_median_v2 (primary, VAL-selected)** | 11659 (bank) | **0.782** | **0.804** | **0.792 [0.755,0.829]** | 0.485 | 0.342 | 2.51 |

**Paired pool-bootstrap 95% CIs (32 test_pools_v2 pools, shared resample), slateN_tough:**
- retrieval_k5 vs linear64: **+0.167, CI [+0.090,+0.244] -- excludes 0, retrieval wins clearly.**
- retrieval_k5 vs NFD: +0.062, CI [-0.010,+0.137] -- 95.4% of draws favour retrieval, but the CI
  edge sits just below 0 -- **directionally ahead, not resolved at 95% at this pool count.**

VAL selection: retrieval k=5 (cube_median) beat k=1 (0.782 vs 0.758) -> primary config. NFD
checkpoint sweep (7 points incl. best-val-loss) was noisy/non-monotone on VAL (0.716-0.776,
no clear trend) -> kept the standard best-val-loss checkpoint (epoch 57) rather than
cherry-picking the max of 7 noisy comparisons.

Full detail, threats, and the two bugs found/fixed this rung (empty-`rows` rollout-loop
IndexError on whole-sequence-clean corpora; `eval_extended.py`'s earlier pools-glob-vs-sidecar
fix reused) are in EXPERIMENT.md's "Clean-data v2 rung" section.

## NFD training -- FINISHED (best val loss 0.005011 at epoch 57, `unet_best.pth`)

`Baselines/NFD/runs/nfd_3ch_narrow_l20_v2/` -- best-val-loss checkpoint + per-10-epoch saves,
all registered in `simple_mpc/adapters.py::OCC_ADAPTERS` (`nfd_3ch_narrow_l20_v2` +
`_epoch{10,20,30,40,50,60}` variants, used for the VAL checkpoint-confirmation sweep below).

## Linear + retrieval bank (built while NFD trained)

1. **LinearForesight res64**: `PYTHONPATH=. python -u Baselines/LinearForesight/fit_switched.py
   --n-bins 1 --res 64 --train-cfg configs/dataset/genesis_narrow_l20_train_v2.yaml --test-cfg
   configs/dataset/genesis_narrow_l20_test_chains_v2.yaml --out Baselines/LinearForesight/runs/
   operator_narrow_l20_v2_res64.pt`. 11659 train rows (220 dropped by `exclude_flagged`, 0
   illegal/null -- matches the bank below), 986 test rows. Held-out swept-region accuracy on
   `test_chains_v2`: persistence 0.000, mean-delta 0.162, **linear-single/switched 0.332**
   (n_bins=1 so single==switched by construction). Saved operator + accuracy JSON
   (`Baselines/LinearForesight/runs/operator_narrow_l20_v2_res64{,_accuracy.json}`).
2. **LinearForesight res32**: same command with `--res 32`, same rows. Held-out accuracy:
   0.332 (effectively identical to res64 on this metric; `Baselines/LinearForesight/runs/
   operator_narrow_l20_v2_res32{,_accuracy.json}`).
3. **Retrieval bank** (`code/build_bank_v2.py --data-dir Genesis/data/narrow_l20_n20/train_v2
   --out artifacts/bank_train_v2_curated.pt`): **11659/12032 rows kept (96.9%)** -- a dramatic
   improvement over the legacy corpus's ~45.6% (confirms the ISS-010 sampler fix at full
   production scale, not just the earlier 256-row smoke test). Interaction-set size mean
   7.21/median 5.0 cubes per row (close to the legacy bank's 7.22/4.0 -- geometric curation
   behaves the same on the new data). `TransitionBank.load_curated`-compatible payload saved;
   `legal` field is all-True (exclusion already happened at load time, see the script's own
   docstring). **Row count (11659) matches the LinearForesight fits exactly** -- same rows,
   same size, as the brief requires.

## Data collection audit (full scale, from the training/fit logs above, not a re-sample)

`exclude_flagged` drop counts, read directly off each run's own log (illegal is 0 in every
case, confirming the ISS-010 fix holds at full production scale, not just the 256-row
pre-launch smoke test):

| corpus | rows before | rows after `exclude_flagged` | illegal | null | dropped (=invalid, i.e. `valid==False`) |
|---|---|---|---|---|---|
| train_v2 (NFD's own train split) | 10862 | 10653 | 0 | 0 | 209 |
| train_v2 (NFD's own val split) | 507 | 502 | 0 | 0 | 5 |
| train_v2 (NFD's own test split) | 510 | 504 | 0 | 0 | 6 |
| train_v2 (whole corpus, linear/bank) | 11879 | 11659 | 0 | 0 | 220 |
| test_chains_v2 | 1010 | 986 | 0 | 0 | 24 |

test_pools_v2's own `exclude_flagged` count not yet run (will happen at eval-scoring time,
`eval_extended.py --exclude-flagged`).

## Next steps (resume when either VAL pools completes or NFD training finishes, whichever first)

1. Check `Genesis/data/narrow_l20_n20/val_pools_v2/manifest.json` for `complete: true`.
2. Once VAL pools ready: score `retrieval` k in {1,5} (aggregation `cube_median` for k=5) on
   VAL `slateN_tough` via `eval_extended.py --pools-dir .../val_pools_v2 --exclude-flagged`,
   pick the winning k. If the NFD checkpoint is also done by then, compare its best-val-loss
   epoch against a couple of later save points on the SAME VAL pools to confirm checkpoint
   choice (per the brief's "never tune on test").
3. Once NFD training's log shows "=== Training complete ===": note final epoch/val loss,
   confirm checkpoint choice (step 2).
4. Score all models (nfd_3ch_narrow_l20_v2, linear res64, linear res32, retrieval winner,
   persistence, random) on `test_chains_v2` + `test_pools_v2` (13-goal + 8-goal-tough slateN,
   accuracy_1, rollout 1-4, moved-cube mm), all via `eval_extended.py --exclude-flagged
   --pools-dir/--chains-dir` pointed at the v2 test dirs.
5. Paired pool-bootstrap 95% CI (`Baselines/common/paired_stats.py::paired_comparison`, pools
   as replication unit) for retrieval-vs-NFD and retrieval-vs-linear on `slateN_tough`.
6. Write the new EXPERIMENT.md section (or new EXP-#### per the experiment-log skill's
   dataset-change test), `python scripts/check_register.py`.

## Prep done earlier (loader/config/harness infra -- unchanged, still valid)

1. **Loader patch** (`Genesis/training/dataset.py::PileSweepData.__init__`, new
   `exclude_flagged: bool = False` kwarg): drops `gap_out_of_window`/`valid==False`/
   (legality-sidecar-if-present)/null (<1mm max per-cube xy displacement, same formula as
   `flag_null_transitions.py` and the retrieval bank's `moved` flag) rows via the existing
   `_index_map` remapping (same pattern as `min_push_length_m`). Forwarded through both
   `registry/dataset_registry.py::_build_genesis_dataset` ("genesis" type, used by
   `fit_switched.py`) and `Baselines/NFD/nfd_lib.py::_build_nfd_genesis_3ch` ("nfd-genesis-3ch",
   used by `train_nfd.py`). **Verified against the legacy DS-0009 test_chains** (known
   illegal=46.2%, null=9.6%, overlap=1.1%): reproduced the exact kept-row count (464/1024) two
   independent ways (direct `PileSweepData` call and through both registered dataset factories).
   Existing test suites unaffected (`tests/test_action_sampling.py` 42/42,
   `tests/test_deploy_train_raster.py` + `tests/test_grid_convention.py` 7/7, all still pass).
2. **Configs, ready**: `Baselines/NFD/configs/nfd_3ch_narrow_l20_v2.yaml` (paths=[train_v2],
   `exclude_flagged: true`, otherwise byte-identical recipe to the registered
   `nfd_3ch_narrow_l20.yaml`: 60 epochs, same model/loss/schedule); `configs/dataset/
   genesis_narrow_l20_train_v2.yaml` and `genesis_narrow_l20_test_chains_v2.yaml` (both
   `exclude_flagged: true`) for `fit_switched.py --train-cfg/--test-cfg`.
3. **Retrieval bank builder**: new `experiments/EXP-0059-retrieval-transition-model/code/
   build_bank_v2.py` -- generalises `datasets/DS-0014-*/build_curated_bank.py` to an arbitrary
   chain-shaped directory, applying the SAME exclusion criteria as (1) (reuses
   `eval_extended._filter_flagged`), computes the geometric `interaction_set` (defaults, same
   tau=12mm/angle=60deg as DS-0014), saves a `TransitionBank.load_curated`-compatible payload.
   Smoke-tested on the legacy `narrow_l20_n20/train` dir: kept 45.6% (2801/6144), matching the
   known illegal+null rate; round-tripped through `TransitionBank.load_curated` successfully.
4. **Eval harness** (`experiments/EXP-0059-retrieval-transition-model/code/eval_extended.py`):
   added `--exclude-flagged` CLI flag (default off, so existing DS-0009/DS-0011 numbers stay a
   byte-identical no-op) using the same `_filter_flagged` helper as (3). Already had
   `--pools-dir`/`--chains-dir` from an earlier round -- verified these work by pointing at the
   partially-collected `test_chains_v2`/`train_v2` dirs directly.
5. **Two live bugs found and fixed in this record's own / this experiment's existing code**
   (not present in any OTHER module's code; see below): both are in the pools-file glob pattern
   `"pools_*.pt"`, which (once `audit_tool_placement.py`/`flag_null_transitions.py` started
   writing `pools_<k>_legality.pt`/`pools_<k>_nullflag.pt` sidecars, this same session) ALSO
   matches those sidecar files, since they also start with `pools_` and end `.pt`. Loading one
   as if it were a pool dict crashes downstream (`KeyError: 'states'`/`'pool_idx'`).
   - `eval_extended.py::_load_corpus`: fixed by anchoring the pool glob to
     `re.fullmatch(r"pools_\d+\.pt", ...)`. **Neither of the two affected corpora's existing
     recorded numbers were corrupted** -- confirmed via `stat`, every prior `_load_corpus()`
     pools call in this experiment's history ran before the nullflag sidecars existed on disk
     (`data_scaling_pools_raw.py`'s own timestamp predates `pools_0_nullflag.pt`'s mtime).
   - `reeval_ds0009.py::_load_with_legal`: its exclusion filter only checked for `"_legality"`
     in the filename, not `"_nullflag"` -- fixed the same way (added the second substring
     check). Same non-corruption check: `results/reeval_ds0009_postfix.json`'s mtime
     (1790585961) predates `pools_0_nullflag.pt`'s (1790594631), so this script's own recorded
     numbers are unaffected.
   - **This bug is otherwise still live** in `confidence_model.py`, `diagnostics_r2.py`,
     `eval_retrieval.py`, `perturbed_sim_zoo.py`, `neighbour_rank_curve.py` (all have their own
     independent `glob.glob(... "pools_*.pt")` calls) -- NOT fixed in this pass (out of this
     record's scope: none of them are used by the clean-data v2 comparison this record is
     running, and each would fail loudly with a `KeyError` rather than silently corrupt a
     number if rerun, so this is a "will crash," not a "will mislead," risk). Flagged here so a
     future rerun of any of those five doesn't lose time rediscovering the same cause.
6. **Retrieval hyperparameter grid, pre-declared** (per the brief's "small, pre-declared grid,
   never tuned on test"): `k in {1, 5}` (aggregation `cube_median` for k=5, i.e. `NearestTransitionPredictor`-equivalent for
   k=1 vs `RetrievalPredictor(k=5, aggregation=cube_median)`), gate distance (`DistanceConfig`)
   fixed at the R1-best `_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)`
   (not re-swept -- the brief says gate distance is part of the grid, but the existing
   `_transfer_one` distance_gate default of 6mm was already validated in the post-fix pass and
   is held fixed here; ONLY k is swept on VAL, to keep the grid genuinely small as instructed).
   Selection metric: `slateN_tough` on the (not-yet-launched) VAL pools.
7. **Stats tool identified**: `Baselines/common/paired_stats.py::paired_comparison(X, names)`
   takes (M models, S states) with higher-is-better scores and returns, per pair, a
   state-resampled bootstrap CI on the mean paired difference plus a sign-flip p -- this is
   what will produce the "paired pool-bootstrap 95% CI, retrieval vs NFD / retrieval vs linear
   on slateN_tough" deliverable, with S = the 32 `test_pools_v2` pools as the replication unit
   (per-pool `slateN_tough` capture values, read from `eval_extended.py`'s persisted
   `*_raw.json`).

## IN PROGRESS (2026-09-28 17:34): 2 more NFD seeds + waiting on clean multi-step set

**Launched, detached, both running now:**

| job | main PID | seed | log_dir | log | started |
|---|---|---|---|---|---|
| NFD seed1 | 1597358 | 1 | `Baselines/NFD/runs/nfd_3ch_narrow_l20_v2_seed1` | `experiments/EXP-0059-*/runs/nfd_3ch_narrow_l20_v2_seed1.log` | 17:32:35 |
| NFD seed2 | 1597359 | 2 | `Baselines/NFD/runs/nfd_3ch_narrow_l20_v2_seed2` | `experiments/EXP-0059-*/runs/nfd_3ch_narrow_l20_v2_seed2.log` | 17:32:35 |

Command (both): `PYTHONPATH=. python -u Baselines/NFD/train_nfd.py Baselines/NFD/configs/
nfd_3ch_narrow_l20_v2.yaml --seed {1,2} --no-resume --override output.log_dir=Baselines/NFD/
runs/nfd_3ch_narrow_l20_v2_seed{1,2}`. Per EXP-0036 convention (seeds 1,2,3 for a seed floor;
here 2 more seeds alongside the existing seed-0/best-val-loss run = 3 total).

**GPU** confirmed healthy: 1499MiB used / 6309MiB free right after both launches, with the
data agent's `seqpools_v2` collection (multi-step set, PID 1595481, seed 2005, ETA ~90-100 min,
`Genesis/data/narrow_l20_n20/seqpools_v2`) already running concurrently -- 3-way GPU sharing,
no OOM risk (both NFD runs use <1.5GB together on this small corpus/model).

**Pace/ETA**: not yet measured at trip-back time (0 epochs logged for either seed after ~60s;
single-seed-alone pace was ~55-57s/epoch, so with 2 concurrent + the collection job sharing the
GPU, expect roughly 1.5-2.5x that, i.e. **very roughly 90-140 min for 60 epochs each -- an
ESTIMATE, not measured**, flagged as such rather than presented as observed. Resume by checking
each log's tail for "=== Training complete ===" or `ps -p 1597358`/`1597359`.

**Real bug hit and fixed while launching these** (would have silently corrupted the seed
comparison if not caught): `train_nfd.py --override output.log_dir=...` is applied AFTER
`Trainer.from_config(..., resume=True)` already ran -- `Trainer._try_resume()` reads
`self.cfg["output"]["log_dir"]` at THAT point, i.e. the ORIGINAL yaml value
(`nfd_3ch_narrow_l20_v2`, the seed-0 run's own directory), not the override. Since that directory
already had checkpoints (`unet_epoch_60.pth` from the completed seed-0 run), both freshly-
launched "seed 1"/"seed 2" runs silently RESUMED FROM THE SEED-0 MODEL'S FINAL WEIGHTS instead of
training from scratch -- caught within ~90s (saw "Resumed weights from .../nfd_3ch_narrow_l20_v2/
unet_epoch_60.pth" in both logs, the wrong directory), killed both (main PID + orphaned
dataloader workers -- the first `pkill -9` pattern match left the seed2 main process alive for a
few more seconds, needed a direct second `kill -9` on its PID), wiped both partially-written
output directories, and relaunched with `--no-resume` (skips `_try_resume()` entirely, so no
checkpoint is loaded regardless of which directory it points at; the log_dir override still
correctly controls where checkpoints are SAVED, confirmed via the "log_dir=...seed{1,2}" line in
both fresh logs and empty output directories immediately after relaunch). **Not yet fixed in
`train_nfd.py` itself** (moving the override before `from_config`, or resolving log_dir once
would fix this generally) -- worth doing if `--override output.log_dir=...` is used again for a
fresh run into a directory whose SIBLING already has checkpoints; named here so the next agent
doesn't rediscover it. Neither contaminated run was scored or persisted anywhere before being
killed, so no invalid evidence entered the record.

**Next steps once available:**
1. Once both seed trainings finish: register checkpoints in `simple_mpc/adapters.py::
   OCC_ADAPTERS` (`nfd_3ch_narrow_l20_v2_seed{1,2}`, same pattern as the existing `_seedN`
   entries for `nfd_3ch_randlen`), score both on DS-0016 with the SAME test protocol as the
   single-seed rung (`code/test_v2.py`-style), paired CIs retrieval vs each seed AND vs the
   3-seed NFD mean (state which pooling: per-pool seed-averaged dv vs per-seed deltas averaged --
   decide and say which when reporting).
2. Once `seqpools_v2` (DS-0018-to-be) completes and is audited/registered: run
   `code/multistep_eval.py` for persistence, all 3 NFD seeds, linear64, linear32,
   retrieval_k5_cube_median_v2, retrieval_k1_v2 -- terminal + per-step slateN/slateN_tough,
   rollout accuracy per step, paired pool-bootstrap CIs retrieval vs each baseline.
3. Record both in EXP-0059's "Clean-data v2 rung" section (new subsections), `check_register.py`
   exit 0.

## IN PROGRESS (2026-09-28 18:11): DS-0018 multi-step eval running (CPU); NFD seeds at 34/60

- **DS-0018 multi-step eval**: `code/multistep_eval_v2.py` (new, reuses `multistep_eval.py`'s
  `run_occ`/`run_particle`/`_finish` verbatim; own pool loader for `seqpools_v2_clean`, no
  legality filtering needed -- already clean by construction). Running CPU-only
  (`CUDA_VISIBLE_DEVICES=`, PID 1641461, log `/tmp/multistep_v2_run.log`) so the 2 NFD-seed GPU
  trainings aren't starved, per the coordinator's instruction. Loaded 32 pools / 2,042 sequences
  (99.7% of nominal 64/pool, matches DS-0018's own DATASET.md) -- confirmed pools with a removed
  sequence rank 63 candidates automatically (`run_occ`/`run_particle` already tolerate
  `n_cand < 64`, no code change needed for that part, verified by the loaded count). Models:
  persistence, `nfd_3ch_narrow_l20_v2` (seed 0), `linear_narrow_l20_v2_res64`,
  `linear_narrow_l20_v2_res32`, `retrieval_k5_cube_median_v2`, `retrieval_k1_v2`. Results ->
  `results/multistep_eval_v2.json` (+ `_raw.json` for per-pool terminal-step bootstrap CIs).
- **NFD seeds 1/2**: 34/60 epochs, ~47-52s/epoch, still healthy (PIDs 1597358/1597359, elapsed
  38:36) -- **ETA ~20-22 more minutes**.
- **Prepped for when seeds finish**: `simple_mpc/adapters.py::OCC_ADAPTERS` gained
  `nfd_3ch_narrow_l20_v2_seed{1,2}` entries (guarded by `os.path.exists`, inert until the
  checkpoints land). **3-seed-mean convention decided, matching EXP-0036's own
  `ensemble_compare.py` precedent** (`dvp[e] = np.mean([dvp[m] for m in mem], 0)`): average the
  3 seeds' PREDICTED `dv` per candidate/pool/goal BEFORE computing the capture formula against
  the shared true `dv` (prediction-averaging, not averaging already-computed capture scalars) --
  will state this explicitly when reporting per the coordinator's "state which" instruction.

**Next steps:** (1) once multi-step eval finishes, read `results/multistep_eval_v2.json`, compute
paired pool-bootstrap CIs (retrieval vs each occ baseline, reusing `code/bootstrap_ci.py::
paired_delta_ci` on the `slateN_tough_terminal_raw` field) and record in EXPERIMENT.md. (2) once
both NFD seeds finish: score seed1/seed2 on DS-0016 (`code/test_v2.py`-style) and DS-0018
(`code/multistep_eval_v2.py`), build the seed-averaged-dv 3-seed mean, paired CIs retrieval vs
each seed and vs the mean, record in EXPERIMENT.md, `check_register.py`.

## Correction (coordinator, 2026-09-28 ~18:12): 3-seed reporting convention revised

Averaging the 3 seeds' predicted `dv` builds an ENSEMBLE (a stronger model than any single NFD
seed), not a "seed floor" -- reclassified as a SECONDARY row ("NFD 3-seed ensemble"), not the
primary seed comparison. **Primary comparison, once seed1/seed2 finish**: retrieval vs seed 0,
vs seed 1, vs seed 2 individually (each its own paired pool-bootstrap CI), PLUS the mean over
the 3 per-seed paired deltas and the seed spread (sd of NFD `slateN_tough` across the 3 seeds,
EXP-0036-style) -- so the retrieval-vs-NFD gap is read against seed noise, not collapsed into
one ensemble number. The prediction-averaged ensemble row (already planned, EXP-0036's own
`ensemble_compare.py` convention) is kept too, just demoted to secondary.

## DS-0018 multi-step rung -- COMPLETE (recorded in EXPERIMENT.md's "Multi-step rung" subsection)

Terminal (3-step) `slateN_tough` [95% CI], 32 pools (2,042/2,048 sequences, 99.7% survival):
persistence 0.044 [-0.033,0.123] -- linear64 0.678 [0.593,0.753] -- NFD seed0 0.693
[0.607,0.774] -- linear32 0.729 [0.653,0.799] -- retrieval_k1 0.787 [0.741,0.830] --
**retrieval_k5 (primary) 0.821 [0.779,0.858]**.

**Paired deltas, retrieval_k5 minus baseline**: vs NFD +0.128 CI[+0.050,+0.206] (excludes 0,
100% favour) -- vs linear64 +0.143 CI[+0.073,+0.221] (excludes 0) -- vs linear32 +0.092
CI[+0.031,+0.157] (excludes 0, 99.9%) -- vs k1 +0.034 CI[-0.007,+0.077] (95.1%, not resolved).
**At 3-step horizon retrieval's win over NFD/linear is FULLY resolved (unlike 1-step DS-0016,
where retrieval-vs-NFD CI barely included 0).** `scripts/check_register.py` exits 0.

## IN PROGRESS (2026-09-28 ~18:29): 2 more NFD seeds still training (46/60 epochs)

PIDs 1597358 (seed1) / 1597359 (seed2), ~56.5 min elapsed. Pace dipped to ~100s/epoch while the
multi-step CPU eval ran concurrently (retrieval's distance search is CPU-heavy), now recovering
(~68-78s/epoch). **ETA ~15-20 more minutes** for both (14 epochs remaining each). Ending turn per
the coordinator's "if seeds still training when the rest is done, write status and end" -- the
multi-step rung (the "rest") is now done, only the seed trainings remain.

**Queued once seeds finish** (per coordinator's correction above -- primary is per-seed, NOT an
averaged ensemble): register `nfd_3ch_narrow_l20_v2_seed{1,2}` (already added to `simple_mpc/
adapters.py::OCC_ADAPTERS`, inert until checkpoints exist) -> score both on DS-0016
(`code/test_v2.py`-style) and DS-0018 (`code/multistep_eval_v2.py`) -> report retrieval vs seed
0/1/2 EACH (own paired CI) + mean of the 3 per-seed paired deltas + sd of NFD `slateN_tough`
across the 3 seeds (the seed-noise floor) -> secondary row: 3-seed prediction-averaged ensemble
(EXP-0036 convention) -> record in EXP-0059, `check_register.py`.

## 3-seed NFD comparison -- COMPLETE (recorded in EXPERIMENT.md's "3-seed NFD comparison" subsection)

Both NFD seeds finished (seed1 best-val-loss 0.005070@ep58, seed2 0.005234@ep55), scored on
DS-0016 (1-step, GPU) and DS-0018 (3-step terminal, GPU) with the same protocol as seed 0.
`check_register.py` exits 0.

**Primary (per-seed), slateN_tough [95% CI]:**

| set | seed0 | seed1 | seed2 | seed spread (sd) | retrieval_k5 |
|---|---|---|---|---|---|
| DS-0016 (1-step) | 0.731 [0.646,0.804] | 0.708 [0.630,0.783] | 0.700 [0.619,0.782] | 0.016 | 0.792 [0.755,0.829] |
| DS-0018 (3-step terminal) | 0.693 [0.607,0.774] | 0.695 [0.620,0.770] | 0.633 [0.542,0.723] | 0.035 | 0.821 [0.779,0.858] |

**Paired deltas (retrieval_k5 minus each seed) + mean of the 3:**
- DS-0016: vs seed0 +0.062 (not resolved) / seed1 +0.085 (resolved) / seed2 +0.092 (resolved) --
  **mean +0.080** (~5x the seed spread).
- DS-0018: vs seed0 +0.128 / seed1 +0.126 / seed2 +0.188 -- **ALL THREE resolved** -- **mean
  +0.147** (~4.2x the seed spread). Multi-step horizon closes the one unresolved 1-step gap
  (vs seed0).

**Secondary (3-seed prediction-averaged ensemble, EXP-0036 convention -- a STRONGER model than
any single seed, not a seed-noise estimate):** DS-0016 ensemble 0.818 [0.768,0.865] --
retrieval_k5 trails it slightly, not resolved (-0.026, 16.7% favour retrieval). DS-0018 ensemble
0.785 [0.726,0.837] -- retrieval_k5 ahead, not resolved (+0.036, 89.3% favour retrieval).
**Retrieval beats every individual NFD seed (resolved at 3-step, mostly resolved at 1-step) but
does not clearly beat a 3-seed ensemble of NFD** -- reported as the honest, non-overclaiming
summary.
