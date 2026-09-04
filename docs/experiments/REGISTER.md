# Claim register

One row per claim this project has made or is testing. **The `depends on`
column is the point of the file:** when an invariant breaks, `grep` it here and
every affected claim is found in one step, instead of being reconstructed by
hand — which is what had to happen on 2026-09-03.

**Status:** `supported` · `refuted` · `open` (stated, under test) ·
`contested` (evidence points both ways) · `invalidated` (an input is now known
broken; the measurement stands, the conclusion does not) · `superseded`.

**Grade** is the best grade among supporting records, capped at `low` when
support is exploratory-only. It is recomputed by `scripts/check_register.py`.

Backfilled 2026-09-03 from `reports/linear_foresight_report.md`,
`docs/sand_manipulation.md` and `docs/prediction_difficulty_hypotheses.md`.
Rows without an `EXP` id predate this register; their evidence lives in the
cited section and has not been re-recorded.

## Open and contested

| ID | Claim | Status | Grade | Supported by | Contradicted by | Depends on |
|---|---|---|---|---|---|---|
| C-019 | The linear operator's margin over mean-delta is regime-independent (+0.15..+0.31 across monolayer, heap and continuum) | open | low | EXP-0002, EXP-0006 | — | `rasteriser-identity`, `swept-region-metric`, `episode-split`, `settled-state` |
| C-023 | Every sand transition collected before 2026-09-04 was recorded ~1.3 mm of mean grain displacement before the pile finished moving (settle capped at 100 steps, not 2500) | supported | moderate | EXP-0007 | — | `config-keys-reach-sim` |
| C-024 | MPM sand never reaches the rigid path's rest criterion: the median grain is at rest (0.05 mm/s) while the top 0.5% creeps indefinitely at ~0.4 um/step, so q=0.995 < 1 mm/s tests the tail rather than the pile | open | moderate | EXP-0007 | — | `settled-state` |
| C-025 | A 5-push sand episode yields only ~2-3 informative transitions: by push 4-5 the pile has spread to 63-82 mm and a push moves 0.25-1.4 mm | open | low | EXP-0007 | — | `settled-state` |
| C-022 | Pile depth is what makes the linear operator work on sand and not on cubes | refuted | low | — | EXP-0002, EXP-0006 | `rasteriser-identity` |
| C-020 | The scattered-monolayer UNet failure is caused by the transposed action channel | open | very-low | EXP-0004 | — | `grid-convention`, `deploy-train-raster` |
| C-014 | The binary mask view predicts better than the density view | contested | low | `sand_manipulation.md` §8, §9.2 | EXP-0003 (blur not held fixed in the original) | `sand-projection`, `swept-region-metric`, `settled-state` |
| C-016 | Blur hurts on a density map because it is already smooth | contested | low | `sand_manipulation.md` §6 | EXP-0003 (opposite sign, different dataset and estimator) | `sand-projection` |
| C-004 | Non-negativity beats ridge for the pixel operator (their Fig. 7) | open | — | `linear_foresight_report.md` §6 (cube fits — needs re-run) | — | `grid-convention` |

## Supported

| ID | Claim | Status | Grade | Supported by | Depends on |
|---|---|---|---|---|---|
| C-018 | The dataset's occupancy channel and its plate/action channel place world x on opposite grid axes | supported | moderate | EXP-0001 | — (establishes `grid-convention`) |
| C-017 | Height and density channels never beat the view matched to the prediction target | supported | low | EXP-0005 | `sand-projection`, `episode-split` |
| C-010 | On sand the linear operator explains ~45–59% of the change, far above persistence | supported | — | `sand_manipulation.md` §6, §9.2 | `sand-projection`, `canonical-warp`, `settled-state` |
| C-011 | mean-delta (zero parameters) is the baseline that matters, not persistence — the canonical frame normalises the action away | supported | — | `sand_manipulation.md` §7 | `canonical-warp` |
| C-012 | The sand operator is effectively low rank, and its rank tracks the input space's dimensionality (~4 single-pile, ~16 varied) | supported | — | `sand_manipulation.md` §7, §9.3 | `sand-projection`, `settled-state` |
| C-013 | Imposing mass conservation in the canonical window hurts, because material legitimately leaves the crop | supported | — | `sand_manipulation.md` §7, §9.2 | `mass-conservation`, `settled-state` |
| C-007 | In scalar targets, essentially all the nonlinearity is one variable: how much material the blade meets | supported | — | `linear_foresight_report.md` §2.6, §2.9 | — (particle-based, not grid-based) |
| C-009 | Per-push band displacement is 84% predictable from grid-visible features but only 58% linearly | supported | — | `linear_foresight_report.md` §2.3 | — (particle-based) |
| C-005 | Every MPC derives blade yaw from the push direction; training data is oblique 92% of the time and deployment perpendicular 99.6% | supported | — | `linear_foresight_report.md` §3 | — (action-based) |
| C-003 | World-frame occupancy mass is conserved to 0.4% on cubes and exactly on sand | supported | — | `linear_foresight_report.md` §4 | `mass-conservation` |
| C-002 | The SE(2) warp costs more than one push changes unless the field is smoothed to σ≈1 | supported | — | `linear_foresight_report.md` §1; identity-baseline rows of EXP-0001, EXP-0003 | `canonical-warp` |

## Invalidated

These were measured correctly and concluded wrongly, because an input was
broken. Listed so nothing cites them, and so the re-runs are queued.

| ID | Claim | Invalidated by | Was stated in | Depends on |
|---|---|---|---|---|
| C-001 | On scattered cube monolayers nothing beats persistence at one-step pixel prediction | EXP-0001 | `linear_foresight_report.md` §1, §2, §2.1–§2.3, Q2/Q3/Q8 | `grid-convention` |
| C-008 | Contact-switching does not transfer from scalar targets to the pixel operator | EXP-0001 | `linear_foresight_report.md` §2.7 | `grid-convention` |
| C-015 | It was pile DEPTH, not granularity: the operator does state-dependent work on heaps and not on monolayers | EXP-0001, EXP-0002 | `sand_manipulation.md` §8 | `grid-convention`, `rasteriser-identity` |
| C-006 | The paper's Fig. 5 deposit structure (depletion across the band, deposition just ahead) reproduces on our data | EXP-0001 | `linear_foresight_report.md` §5 | `grid-convention` |

## Queued re-runs

Ordered by what unblocks the most rows.

1. Fix `grid-convention` + `rasteriser-identity` + `pixel-index-origin`, with
   tests written first. Unblocks C-001, C-004, C-006, C-008, C-015.
2. Re-run `linear_foresight_report.md` §2 and §2.7 on corrected grids (C-001,
   C-008); §2.7 is the one likeliest to reverse.
3. EXP-0002 under leave-one-run-out, size-matched (C-019).
4. Resolve C-014/C-016 by running EXP-0003's cross on the single-pile `pile20`
   set with `linear-nonneg` — that isolates dataset from estimator.
5. Retrain one UNet config with the raster fixed (C-020). GPU, ~1 h.
