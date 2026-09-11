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
