# Claim register

One row per claim this project has made or is testing. **The `depends on`
column is the point of the file:** when an invariant breaks, `grep` it here and
every affected claim is found in one step, instead of being reconstructed by
hand — which is what had to happen on 2026-09-03.

**Status:** `supported` · `refuted` · `open` (stated, under test) ·
`contested` (evidence points both ways) · `invalidated` (an input is now known
broken; the measurement stands, the conclusion does not) · `superseded` ·
`withdrawn` (the whole line of work was abandoned; see the note below).

**Grade** is the best grade among supporting records, capped at `low` when
support is exploratory-only. It is recomputed by `scripts/check_register.py`.

> **2026-09-05 — the MPM sand path was withdrawn.** The medium was found to be
> a soft deformable body rather than a granular one, for structural rather than
> parameter reasons (`docs/rejected_mpm_sand.md`), and the code and datasets
> were removed in `e9b83f99`. **Thirteen claims (C-010, C-012, C-013, C-014,
> C-016, C-021, C-023 … C-029) were withdrawn with it and are not restated
> here**, because carrying their numbers forward is exactly what this file
> exists to prevent. They are recoverable from git history if the conclusion
> rather than the measurement is ever wanted. Four surviving claims were
> **re-sourced** to cube-only evidence — C-002, C-003, C-011, C-017 — and are
> marked below; two, C-015/C-022, were already refuted and are now stated in
> cube-only terms.

Backfilled 2026-09-03 from `reports/linear_foresight_report.md` and
`docs/prediction_difficulty_hypotheses.md`. Rows without an `EXP` id predate
this register; their evidence lives in the cited section and has not been
re-recorded.

## Open and contested

| ID | Claim | Status | Grade | Supported by | Contradicted by | Depends on |
|---|---|---|---|---|---|---|
| C-019 | The linear operator's margin over mean-delta is regime-independent (+0.15..+0.31 across scattered monolayers and two-layer heaps) | open | low | EXP-0002, EXP-0006 | — | `rasteriser-identity`, `swept-region-metric`, `episode-split` |
| C-020 | The scattered-monolayer UNet failure is caused by the transposed action channel | open | very-low | EXP-0004 | — | `grid-convention`, `deploy-train-raster` |
| C-004 | Non-negativity beats ridge for the pixel operator (their Fig. 7) | open | — | `linear_foresight_report.md` §6 (cube fits — needs re-run) | — | `grid-convention` |
| C-030 | **Weakened, then re-opened 2026-09-05.** A signal-sensitive metric ranks models for MPC better than pixel rms does. The original mechanism ("the fine band is unpredictable") is refuted by EXP-0007; the *global* metric swap (FSS for rms) is refuted by EXP-0008; but EXP-0008's own table shows a strong **per-error-type** dissociation — see C-035, which is now where this programme lives | contested | very-low | — | EXP-0007, EXP-0008 (both refute the strong forms) | `canonical-warp`, `swept-region-metric` |

## Supported

| ID | Claim | Status | Grade | Supported by | Depends on |
|---|---|---|---|---|---|
| C-018 | The dataset's occupancy channel and its plate/action channel place world x on opposite grid axes | supported | moderate | EXP-0001 | — (establishes `grid-convention`) |
| C-031 | On cubes, blur moves the operator's error ~4x more than the choice of view does (33 points vs 8 across σ 0→1.5) | supported | very-low | EXP-0003 | `canonical-warp`, `warp-blend`, `swept-region-metric`, `episode-split`, `particle-projection` |
| C-017 | **Re-sourced, cube-only.** Adding a height or density channel to a mask input does not help a linear model predict the cube silhouette | supported | low | EXP-0005 | `particle-projection`, `episode-split` |
| C-011 | **Re-sourced, cube-only.** mean-delta (zero parameters) is the baseline that matters, not persistence — the canonical frame normalises the action away, and mean-delta alone reaches 0.12–0.35 explained | supported | low | EXP-0002, EXP-0005, EXP-0006 | `canonical-warp` |
| C-007 | In scalar targets, essentially all the nonlinearity is one variable: how much material the blade meets | supported | — | `linear_foresight_report.md` §2.6, §2.9 | — (particle-based, not grid-based) |
| C-009 | Per-push band displacement is 84% predictable from grid-visible features but only 58% linearly | supported | — | `linear_foresight_report.md` §2.3 | — (particle-based) |
| C-005 | Every MPC derives blade yaw from the push direction; training data is oblique 92% of the time and deployment perpendicular 99.6% | supported | — | `linear_foresight_report.md` §3 | — (action-based) |
| C-003 | **Re-sourced, cube-only.** World-frame occupancy mass is conserved to 0.4% on cubes | supported | — | `linear_foresight_report.md` §4 | `mass-conservation` |
| C-002 | **Re-sourced, cube-only.** The SE(2) warp costs more than one push changes unless the field is smoothed to σ≈1 | supported | — | `linear_foresight_report.md` §1; identity-baseline rows of EXP-0001, EXP-0003 | `canonical-warp` |

## Refuted and invalidated

Measured correctly, concluded wrongly. Listed so nothing cites them, and so the
re-runs are queued.

| ID | Claim | Status | By | Was stated in | Depends on |
|---|---|---|---|---|---|
| C-001 | On scattered cube monolayers nothing beats persistence at one-step pixel prediction | invalidated | EXP-0001 | `linear_foresight_report.md` §1, §2, §2.1–§2.3, Q2/Q3/Q8 | `grid-convention` |
| C-008 | Contact-switching does not transfer from scalar targets to the pixel operator | invalidated | EXP-0001 | `linear_foresight_report.md` §2.7 | `grid-convention` |
| C-006 | The paper's Fig. 5 deposit structure (depletion across the band, deposition just ahead) reproduces on our data | invalidated | EXP-0001 | `linear_foresight_report.md` §5 | `grid-convention` |
| C-015 | **Restated cube-only.** Pile depth is what makes the linear operator do state-dependent work — monolayers should show no margin over mean-delta | refuted | EXP-0002, EXP-0006 | withdrawn cube/continuum comparison, commit `242a9cc1` | `rasteriser-identity` |
| C-022 | *(duplicate of C-015 as originally filed; merged 2026-09-05)* | superseded | — | — | — |
| C-032 | P1 (`docs/ideas_log_signal_vs_detail.md` §5): FSS skill vs. neighbourhood radius rises and saturates by r≈3-5px on cube n20, near the σ≈1-1.5 that helped in EXP-0003 | refuted | EXP-0007 | `docs/ideas_log_signal_vs_detail.md` §5 | `swept-region-metric`, `episode-split`, `canonical-warp`, `warp-blend`, `footprint-splat` |
| C-033 | The linear operator's error as a fraction of the band signal falls ~3x from the finest scale (0.686) to ~8 px (0.221) — fine detail is harder, not unpredictable, and the operator has usable FSS skill (0.888) at the finest radius against a 0.549 threshold | open | very-low | EXP-0007 (M5 arm + reviewer amendment) | — | `swept-region-metric`, `episode-split` |
| C-035 | **The dissociation is by error TYPE, and rms gets two of three wrong.** At matched or lower rms, high-frequency noise destroys control utility (slate4 −0.04 at rms 0.241) while displacement preserves it (slate4 +0.55 at rms 0.281); amplitude and blur errors cost rms and cost control *nothing* (a=0.5: rms 0.218→0.271, dV Spearman 0.474→0.472) | open | very-low | EXP-0008 (reviewer re-analysis of its own table) | — | `swept-region-metric`, `episode-split` |
| C-034 | P2 (`docs/ideas_log_signal_vs_detail.md` §5): under synthetic degradation of the operator's cube n20 prediction, rms and Lyapunov-dV control utility cross order between displacement and hf-noise; and (partial P3) FSS(r=1) rank-correlates with control utility better than rms does across the spectrum | refuted | EXP-0008 | `docs/ideas_log_signal_vs_detail.md` §5 | `swept-region-metric`, `episode-split`, `canonical-warp`, `warp-blend`, `footprint-splat` |

## Queued re-runs

Ordered by what unblocks the most rows.

1. Fix `grid-convention` + `rasteriser-identity` + `pixel-index-origin`, with
   tests written first. Unblocks C-001, C-004, C-006, C-008, C-015.
2. Re-run `linear_foresight_report.md` §2 and §2.7 on corrected grids (C-001,
   C-008); §2.7 is the one likeliest to reverse.
3. EXP-0002 under leave-one-run-out (C-019).
4. A density-target cube run, to break the target-matching confound EXP-0005's
   amendment introduced (C-017).
5. Retrain one UNet config with the raster fixed (C-020). GPU, ~1 h.
6. The signal-vs-detail metric programme (C-030) —
   `docs/ideas_log_signal_vs_detail.md`.
