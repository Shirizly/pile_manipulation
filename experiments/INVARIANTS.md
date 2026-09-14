# Invariants — the `depends_on` tag registry

Every tag an experiment record cites must appear here. A tag names a property
the repo's results rest on, and — where one exists — the test that checks it.

**Why this file exists.** Assumptions written as sentences do not fail loudly.
See `.claude/skills/experiment-log/SKILL.md` for the incident history that
motivated this file's fields.

**Status vocabulary:** `holds` (a test asserts it and passes) · `unchecked`
(believed true, nothing enforces it) · `broken` (known false) · `fixed`
(was broken, now checked).

## Reset 2026-09-10

This registry was reset to empty along with `REGISTER.md` and every
`EXP-####` record — archived to
`archive/2026-09-10_pre-reset/docs/experiments/INVARIANTS.md`, not deleted.
Any invariant this project still relies on (grid convention, rasteriser
identity, episode-split, etc.) needs to be re-added below **with its current
status re-verified**, not copied forward assuming it still holds — the point
of a reset is not to inherit stale confidence.

| Tag | Property | Status | Test |
|---|---|---|---|
| `randlen-train-test-file-disjoint` | overnight_randlen's train/test split is disjoint at the FILE level, for every one of the 5 groups (mixed_n20, piled_n20, scattered_n20, piled_n50, scattered_n50) | holds | `scripts/probes/prepare_randlen_split.py`'s own in-script assertion at split-creation time (`set(test_idx).isdisjoint(train_idx)` and `set(test_idx)\|set(train_idx) == set(idx)`, per group) |
| `randlen-step0-pool-size-128` | every file in overnight_randlen is a same-state pool of exactly 128 step-0 candidates (env-major-blocked: row r = env r%128, step r//128) | holds | runtime assertion inside `Baselines/common/randlen_data.py::load_randlen_cell` (`n_s0 == n_files * 128`), which runs on every load with `need_step_idx=True` (the default) and would fail loudly if a `min_push_length_m` filter ever dropped a step-0 row. **Confirmed 2026-09-10 that this filter DOES drop rows on the pooled `_all.yaml` configs** (5 of 24576 expected step-0 rows missing on `genesis_overnight_randlen_train_all.yaml`) -- callers that only need `occ0`/`occ1`/`actions` (no step tagging) now pass `need_step_idx=False` to skip this assertion rather than being blocked by it (`Baselines/LinearForesight/fit_switched.py`); callers that DO use step/slate tagging (`eval_report.py`, `eval_randlen_indist.py`) still get it enforced by default |
| `occ-rasteriser-consistency` | occ0/occ1 built by `Baselines.common.randlen_data.load_randlen_cell` (bypasses `CellData` for N-agnosticism) are byte-identical to what `Baselines.common.data.load_cell` would produce for the same underlying transitions | unchecked | none written -- both paths call the same `raw[i][0][0][0]`/`raw[i][1]` expressions off the same registry-built `PileSweepData`, so this SHOULD hold by construction, but no test asserts it directly |
| `goal-mask-axis-convention-row-y-col-x` | `Baselines/common/goals.py`'s new goal masks (`quadrant_mask`, `letter_mask`) use the SAME row=y/col=x pixel convention `control_utility_test.lyapunov_weights` uses, not `transforms/functional.py`'s row=x/col=y convention | unchecked | none written -- matched by direct code comparison during authoring (EXP-0001), not by an automated test |
| `push-frame-warp-roundtrip` | `transforms/functional.py`'s SE(2) push-frame warp (`to_push_frame`/`from_push_frame`/`push_frame_validity_mask`/`blend_push_prediction`) round-trips a smooth image inside the validity mask, preserves mass, maps a push onto the canonical `+x` axis, and is differentiable -- the primitive every pixel-space linear-foresight operator (`fit_linear_foresight.py`, `Baselines/LinearForesight/`) is built on | holds | `tests/test_push_frame_warp.py` (15 tests: roundtrip, mass preservation, axis convention, validity-mask/blend behaviour, batch independence, differentiability) -- re-run 2026-09-10, all 15 pass |
| `dmdc-apply-operators-memory-safe` | `dmdc_baseline.apply_operators` does not OOM for realistic descriptor dimensionality D | broken | none written -- confirmed 2026-09-13 (EXP-0004) that `A[bins]` fancy-indexing materialises an `[N,D,D]` tensor and requests ~138GB at D=631, N~1e5. Workaround used in `experiments/temp/dmdc-lenbins/{fit_arbitration.py,fit_d.py}`: a custom `apply_operators_mem` that loops over bins instead of gathering `A[bins]` -- not yet upstreamed into `dmdc_baseline.py` itself |
| `run-probe-ledger-path-matches-doc` | `scripts/run_probe.py` writes its execution ledger to the documented `experiments/COMMANDS.jsonl` location | broken | none written -- `scripts/run_probe.py`'s own `LEDGER = ROOT / "runs" / "COMMANDS.jsonl"` constant writes to `runs/COMMANDS.jsonl` instead, confirmed by inspection 2026-09-13 (EXP-0004) |
| `hybrid-latent-stage2-anticollapse` | `docs/experimental_design/hybrid_linear_latent.md` section 9's Stage-2 loss (`L_latent = \|\|\hat z' - z'\|\|^2`, no regulariser), implemented naively, does not collapse the encoder | broken | none written -- no reference implementation of Stage 2 exists in the codebase yet, only `experiments/temp/latent-killprobe/probe.py` (a throwaway probe, not a pytest). It reproduced the collapse directly: z_std fell to ~2e-6 and both model/baseline losses converged to ~1e-9/1e-11 together with no regulariser; a per-channel variance hinge fixed it. Established by EXP-0007. Apply the xfail pattern once a real Stage-2 module is built |
| `hybrid-vis-desc-holdout-disjoint-from-train` | the 43-file `overnight_randlen` holdout consumed by `experiments/temp/hybrid-vis-desc/cache/{train_cache.pt,test_cache.pt}` (and therefore by every operator fit from it) is disjoint from the 170-file train set at the file level | holds | none written as a pytest -- verified directly inside `experiments/temp/slaten-broad/eval_slaten_broad.py`'s step [2/7]: independently re-ran `descriptors.py::list_files()+split_files(seed=0,holdout_frac=0.2)` and compared the reproduced file-id sets against `train_cache.pt`/`test_cache.pt`'s own recorded `file_id` fields (not re-derivation trusted blindly) -- exact match, 170+43=213, zero overlap. Established by EXP-0008 |

| `eval-baseline-scorer-batch-on-requested-device` | `Baselines/common/eval_baseline.py::_predictor_batch` builds `PredictorBatch` on the device the caller intended (e.g. matching a future `--device cuda`), not silently on CPU regardless of it | broken | none written -- confirmed 2026-09-13 (EXP-0009) by direct inspection: `_predictor_batch` passes `load_cell`'s CPU tensors straight through with no `.to(device)` call, so `NFDPredictor`/`SchenckPredictor` (which derive their compute device from `batch.occ0.device`) run on CPU as actually invoked by this scorer today, while `GNNPredictor` forces its own cuda-if-available device internally -- an inconsistency `Baselines/common/benchmark_time.py`'s own module docstring already flagged. `EXP-0009`'s own `bench.py` works around this locally (explicit `.to(DEVICE)` in its own batch construction) but does not patch `eval_baseline.py` itself |
| `slates-multistep-row-is-trajectory-identity` | in the raw `Genesis/data/slates_multistep/*/*_{batch}_data.pt` files (NOT the `*_eval` dataset configs, which apply `min_push_length_m` filtering that drops rows and breaks the alignment), row index (env id) is a STABLE per-env trajectory identity across the 3 rollout steps -- step k's row r and step k+1's row r describe the same simulated environment, never reset in between | holds | none written as a pytest -- verified numerically, independently, on both corpora in EXP-0010/RUN-0001: aligned (row r step0 vs row r step1) median L2 6.6e-08 (n20_L20mm) / 8.4e-08 (n20_L40mm) vs. a shuffled-row control 3.0e-02 / 1.4e-01 -- a >5 order-of-magnitude gap, confirming the property directly rather than assuming it from the data-generation code |
| `slates-multistep-single-pushlen-bin-starvation` | `slates_multistep`'s `n20_L20mm`/`n20_L40mm` cells are each SINGLE-push-length collections (fixed `push_length` per file, not `randlen`), so a 6-push-length-bin switched operator (MODEL-0001's scheme) sees rows concentrated in ~1 of its 6 bins across the entire corpus -- training or switching on these cells silently starves the other bins of any gradient/data, and any switched-vs-global comparison here reflects operator SPECIALISATION within one bin, not genuine bin-SWITCHING behaviour | holds | none written as a pytest -- verified directly in EXP-0010/RUN-0001 (bin-sequence distribution: n20_L20mm 6400/6400 rows in bin 1 at every step; n20_L40mm 6399/6400 in bin 3, 1 row dips to bin 2 at step 1) and independently reconfirmed in EXP-0010/RUN-0002 (4 of 6 switched-operator bins are byte-identical to their closed-form init after 150 training iterations -- zero gradient reached them) |

## Adding a tag

1. Name the property as something that can be **false**, not as a topic.
2. Give it a status honestly. `unchecked` is a normal answer.
3. If you are about to rely on it at T2, write the test first.

## The xfail pattern

For a tag that is `broken`, write the test now and mark it
`@pytest.mark.xfail(strict=True, ...)`. It records the defect in the suite
rather than in prose, and `strict=True` means that the moment someone fixes the
bug the test XPASSes and *fails the run*, telling them to drop the marker and
flip this table's status to `fixed`. A broken invariant then cannot be quietly
fixed without the register noticing.

## Broken tags block T2

A T2 record may not cite a tag whose status is `broken` or `unchecked`.
`scripts/check_register.py` enforces this. At T0/T1 it is allowed and costs the
`untested-dependency` downgrade.
