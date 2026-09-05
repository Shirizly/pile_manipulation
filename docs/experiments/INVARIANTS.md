# Invariants — the `depends_on` tag registry

Every tag an experiment record cites must appear here. A tag names a property
the repo's results rest on, and — where one exists — the test that checks it.

**Why this file exists.** Assumptions written as sentences do not fail loudly.
The grid convention was documented in three places, wrongly, in two mutually
contradictory ways, for the whole linear-foresight programme
(`docs/prediction_difficulty_hypotheses.md` §1). A test would have failed on
day one.

**Status vocabulary:** `holds` (a test asserts it and passes) · `unchecked`
(believed true, nothing enforces it) · `broken` (known false) · `fixed`
(was broken, now checked).

| Tag | Property | Status | Test |
|---|---|---|---|
| `grid-convention` | The occupancy channel, the plate/action channel and `actions_to_pixels` all place world x on the same grid axis | **`fixed` (2026-09-05)** — was broken 2026-09-03..09-05; `PileSweepData._draw_particle_grid` now transposes out of OpenCV's (col,row) convention | `tests/test_grid_convention.py` (4 tests, all asserting) |
| `rasteriser-identity` | `PileSweepData._draw_particle_grid`, `transforms.particles_to_occupancy` and `transforms.particle_fields.points_to_mask` produce the same field (up to footprint radius) for one scene | **`fixed` (2026-09-05)** | `tests/test_grid_convention.py::test_dataset_occupancy_agrees_with_particles_to_occupancy` |
| `pixel-index-origin` | `particles_to_occupancy` (`* (res-1)`) and `actions_to_pixels` (`* res - 0.5`) agree on where cell centres are | **broken** (2026-09-03, ~1 px) | none |
| `canonical-warp` | The SE(2) push-frame warp is invertible, mass-preserving, and maps the push onto the canonical axis | `holds` | `tests/test_push_frame_warp.py` (11 tests) |
| `warp-blend` | Prediction outside the validity mask is exactly the input, so no model is credited for untouched pixels | `holds` | `tests/test_push_frame_warp.py::test_blend_keeps_the_original_outside_the_mask` |
| `footprint-splat` | `footprint_radius` fills a cube's real footprint rather than one cell, and `radius=0` matches the hard splat | `holds` | `tests/test_footprint_splat.py` |
| `particle-projection` | `points_to_density` / `points_to_heightmap` / `points_to_mask` are mass-preserving and drop rather than clamp out-of-bounds points | `holds` | `tests/test_particle_fields.py` (19 tests) |
| `episode-split` | Train/val splits are at episode (file) granularity — a transition-level split leaks, since sequential pushes share a pile | **`holds` (2026-09-05)** | `tests/test_metric_invariants.py` — asserts no episode appears on both sides, and measures that within-episode neighbours really are more similar than random pairs, so the rule is not vacuous |
| `swept-region-metric` | The scoring region covers the pixels a push can plausibly change; whole-image error is ~95% untouched pixels where persistence is exact | **`holds` (2026-09-05)** | `tests/test_metric_invariants.py` — >90% of the actual change falls inside the mask pooled, >75% for >95% of transitions individually, and the mask stays a small fraction of the grid |
| `mass-conservation` | World-frame `‖I_k‖₁ / ‖I_{k+1}‖₁ ≈ 1`. Holds to 0.4% on cubes — but **only in the world frame**; in the canonical crop material legitimately leaves | `unchecked` | none; measured per-run by `fit_linear_foresight.py` |
| `perpendicular-actions` | The action distribution in training data matches deployment | **broken** | none — training is oblique 92% of the time, every MPC executes perpendicular 99.6% of the time (`linear_foresight_report.md` §3). `--perpendicular-pushes` fixes collection |
| `settled-state` | Recorded `s'` is a pile at rest, not material still moving | `unchecked` **for the rigid cube path** (EXP-0006 cites it); **moot for MPM** — that path was abandoned 2026-09-05 (`docs/rejected_mpm_sand.md`) and its datasets deleted. Re-open the MPM scope only if a granular medium returns via DEM | `tests/test_config_no_duplicate_keys.py` guards one cause; the residual-speed check itself is unchecked |
| `config-keys-reach-sim` | Every key written in a config YAML actually reaches the simulator: no duplicate top-level key silently discards a block, no declared key is ignored | `fixed` (2026-09-04) | `tests/test_config_no_duplicate_keys.py` (15 tests) |
| `world-frame-alignment` | The grid convention is the physically correct one, not merely self-consistent: transport read out of the grid points the way the blade travels in world metres | **`holds` (2026-09-05)** — mean cos +0.979 under `dim0=world_x` against **−0.001** under the pre-fix alternative, n=1876 | `tests/test_world_frame_ground_truth.py` (2 tests) |
| `dataset-provenance` | Every collected dataset records the code state it was collected under, in its `_N_config.yaml` `provenance:` block | **`fixed` for new data (2026-09-05)**; **unrecorded for everything collected before it** — `Genesis/data/cube_spectrum`, `foresight/*`, `corl*`, `slates/*`. Their lineage is reconstructable only from file mtimes, which is guesswork | `Genesis/sandbox_manipulation_clean.py::_save_config` stamps `utils.git_provenance()`; no test asserts old data has it, because it cannot |
| `deploy-train-raster` | The occupancy and action channels a learned model sees at MPC time are built the same way as the ones it trained on | **`fixed` (2026-09-05)** — `model/eulerian_wrapper.py` was converting into the PRE-fix dataset convention, and its plate angle dropped the physical +π/2 to compensate for that transpose | `tests/test_deploy_train_raster.py` (3 tests, written before the fix and failing on it) |

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
