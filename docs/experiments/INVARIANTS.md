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
| `grid-convention` | The occupancy channel, the plate/action channel and `actions_to_pixels` all place world x on the same grid axis | **broken** (2026-09-03) | `tests/test_grid_convention.py` (3 pass, 1 `xfail(strict)`) |
| `rasteriser-identity` | `PileSweepData._draw_particle_grid`, `transforms.particles_to_occupancy` and `transforms.sand_occupancy.sand_to_mask` produce the same field (up to footprint radius) for one scene | **broken** (2026-09-03) | `tests/test_grid_convention.py::test_dataset_occupancy_agrees_with_particles_to_occupancy` (`xfail(strict)`) |
| `pixel-index-origin` | `particles_to_occupancy` (`* (res-1)`) and `actions_to_pixels` (`* res - 0.5`) agree on where cell centres are | **broken** (2026-09-03, ~1 px) | none |
| `canonical-warp` | The SE(2) push-frame warp is invertible, mass-preserving, and maps the push onto the canonical axis | `holds` | `tests/test_push_frame_warp.py` (11 tests) |
| `warp-blend` | Prediction outside the validity mask is exactly the input, so no model is credited for untouched pixels | `holds` | `tests/test_push_frame_warp.py::test_blend_keeps_the_original_outside_the_mask` |
| `footprint-splat` | `footprint_radius` fills a cube's real footprint rather than one cell, and `radius=0` matches the hard splat | `holds` | `tests/test_footprint_splat.py` |
| `sand-projection` | `sand_to_density` / `sand_to_heightmap` / `sand_to_mask` are mass-preserving and drop rather than clamp out-of-bounds grains | `holds` | `tests/test_sand_occupancy.py` (19 tests) |
| `episode-split` | Train/val splits are at episode (file) granularity — a transition-level split leaks, since five sequential pushes share a pile | `unchecked` | none; enforced by convention in `split_by_episode` and the sand loaders |
| `swept-region-metric` | The scoring region covers the pixels a push can plausibly change; whole-image error is ~95% untouched pixels where persistence is exact | `unchecked` | none |
| `mass-conservation` | World-frame `‖I_k‖₁ / ‖I_{k+1}‖₁ ≈ 1`. Holds to 0.4% on cubes and 1.0000 on sand — but **only in the world frame**; in the canonical crop material legitimately leaves | `unchecked` | none; measured per-run by `fit_linear_foresight.py` |
| `perpendicular-actions` | The action distribution in training data matches deployment | **broken** | none — training is oblique 92% of the time, every MPC executes perpendicular 99.6% of the time (`linear_foresight_report.md` §3). `--perpendicular-pushes` fixes collection |
| `settled-state` | Recorded `s'` is a pile at rest, not material still moving | `unchecked` | none; MPM path silently skipped the settle once (`sand_manipulation.md` §3) |
| `deploy-train-raster` | The occupancy a learned model sees at MPC time is produced the same way as the one it trained on | **broken** (suspected, 2026-09-03) | none — training uses the cv2 raster, `simple_mpc/*` uses `particles_to_occupancy` |

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
