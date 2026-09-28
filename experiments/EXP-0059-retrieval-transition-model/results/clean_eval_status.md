# EXP-0059 clean-eval status (fair train/val/test on the ISS-010-fix v2 corpus)

Updated: 2026-09-28 15:20. **COMPLETE** -- all cells filled, recorded in
`experiments/EXP-0059-retrieval-transition-model/EXPERIMENT.md`'s "Clean-data v2 rung" section
(new datasets: DS-0015 train / DS-0016 test / DS-0017 val). `scripts/check_register.py` exits 0.

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
