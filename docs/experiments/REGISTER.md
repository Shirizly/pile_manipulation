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
| C-004 | Non-negativity beats ridge for the pixel operator (their Fig. 7). **Re-run 2026-09-05 on the fixed grid, regime-split.** Wins 3 of 4 (dataset x crop) cells, replicating the original's "3 of 4" framing, but the margin only clears the noise floor on SCATTERED monolayers (nonneg 58.8-65.2% of persistence vs best-ridge 64.1-72.2%, margin 0.0058-0.0076 rms, res 32 crop 1.0/0.5) | contested | very-low | EXP-0010 (scattered cells, both crops) | EXP-0010 (piled n20 cells: statistical tie, margin 0.0005-0.0015 rms, inside the ~0.001-0.004 documented floor; ridge nominally wins one of the two) | `grid-convention`, `rasteriser-identity`, `swept-region-metric`, `episode-split` |
| C-036 | **Neither ridge shrinkage nor rank truncation trades predictive accuracy for control utility.** lambda*_rms = 10 and lambda*_control = 1..10; rank*_rms = 256 and rank*_control = 512 (truncation hurts both monotonically) | refuted | low | — | EXP-0013 | `canonical-warp`, `swept-region-metric`, `episode-split` |
| C-037 | **rms and control utility order nine real fitted models oppositely.** Every rank-truncated operator from rank 1 to rank 128 beats mean-delta on swept-region rms (53.9-60.5% vs 69.2%) and loses to it on slate-4 action selection (0.338-0.570 vs 0.601). Not explained by high-frequency energy, which is flat across the block | open | low | EXP-0013 (a by-product of a sweep built for C-036; wants its own confirmatory run) | — | `canonical-warp`, `swept-region-metric`, `episode-split` |
| C-038 | On scattered 50-cube monolayers at res 16, canonical crop 0.5 is much the best window: explained 0.415 vs 0.228 (crop 1.0) and 0.137 (crop 0.25) | open | moderate | EXP-0015 (incidental) | — | `canonical-warp`, `swept-region-metric` |
| C-039 | On SAME-state candidate slates (one settled pile, 32 differing actions per state), high-frequency prediction noise still damages within-slate action ranking more than displacement does, but by LESS than EXP-0008's cross-state measurement at the matched degradation level (corner goal: Spearman(dV) relative drop 51.6% same-state vs 84% cross-state at hf-noise m=2.0; displacement's drop is essentially unchanged, 7.1% vs 6% at k=4) -- direction consistent with EXP-0008's "independent noise per candidate" mechanism being partly a cross-state-data artefact, but n=6 states (of a planned 50) is too thin to confirm the specific threshold | open | very-low | EXP-0012 | — | `canonical-warp`, `swept-region-metric`, `episode-split`, `footprint-splat`, `settled-state` |
| C-030 | **Weakened, re-opened, now directly supported at model level by C-037 (2026-09-05).** A signal-sensitive metric ranks models for MPC better than pixel rms does. The original mechanism ("the fine band is unpredictable") is refuted by EXP-0007; the *global* metric swap (FSS for rms) is refuted by EXP-0008; but EXP-0008's own table shows a strong **per-error-type** dissociation — see C-035, which is now where this programme lives | contested | very-low | — | EXP-0007, EXP-0008 (both refute the strong forms) | `canonical-warp`, `swept-region-metric` |
| C-008 | Contact-switching does not transfer from scalar targets to the pixel operator. **Re-opened 2026-09-05**: the grid-convention bug this was measured under is now fixed, but the re-run (EXP-0009) queued the switched-operator cells (res=16/crop=0.25, bins=2/3, est. M/D 2.9-4.4 per bin) behind a much more expensive res=64 cell and none of them finished in budget — still genuinely untested on the fixed grid Measured on the fixed grid by EXP-0015 at three crops: switched beats single by +0.0001 / +0.0069 / +0.0050 explained -- consistently positive, 4-300x inside the ~0.03 noise floor, so still unanswered. | open | very-low | EXP-0009 (attempted; no cell completed), EXP-0015 | — | `grid-convention`, `rasteriser-identity`, `swept-region-metric`, `episode-split` |

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
| C-001 | **REVERSED 2026-09-05** (was: nothing beats persistence). With `grid-convention` actually fixed in the codebase (commit `aac084e3`), the linear/ridge/nonneg pixel operator on scattered cube monolayers beats persistence by 30-46 points of explained variance (54-58% of persistence rms at res=64/crop=0.5, 70.5% at res=8/crop=0.5) and beats mean-delta too (+0.17 to +0.38 explained) | supported | very-low | EXP-0001 (res=64 numbers, pre-fix simulated), EXP-0009 (res=8, actual fixed pipeline; res=64 same-session cell did not complete) | `grid-convention`, `rasteriser-identity` |
| C-006 | **RECONFIRMED 2026-09-05** (previously invalidated under the transposed grid — see below). Re-measured on the grid-convention-fixed pipeline (commit `aac084e3`): the paper's Fig. 5 deposit structure (depletion across the swept band, deposition peaking just ahead) reproduces on BOTH a scattered monolayer and a piled (n20) cube dataset — peak effects 10-100x the per-fold sd; piled cubes show ~2x the amplitude at about half the column-width of scattered | supported | low | EXP-0010 | `grid-convention`, `rasteriser-identity`, `canonical-warp`, `footprint-splat` |

## Refuted and invalidated

Measured correctly, concluded wrongly. Listed so nothing cites them, and so the
re-runs are queued.

| ID | Claim | Status | By | Was stated in | Depends on |
|---|---|---|---|---|---|
| C-015 | **Restated cube-only.** Pile depth is what makes the linear operator do state-dependent work — monolayers should show no margin over mean-delta | refuted | EXP-0002, EXP-0006 | withdrawn cube/continuum comparison, commit `242a9cc1` | `rasteriser-identity` |
| C-022 | *(duplicate of C-015 as originally filed; merged 2026-09-05)* | superseded | — | — | — |
| C-032 | P1 (`docs/ideas_log_signal_vs_detail.md` §5): FSS skill vs. neighbourhood radius rises and saturates by r≈3-5px on cube n20, near the σ≈1-1.5 that helped in EXP-0003 | refuted | EXP-0007 | `docs/ideas_log_signal_vs_detail.md` §5 | `swept-region-metric`, `episode-split`, `canonical-warp`, `warp-blend`, `footprint-splat` |
| C-033 | The linear operator's error as a fraction of the band signal falls ~3x from the finest scale (0.686) to ~8 px (0.221) — fine detail is harder, not unpredictable, and the operator has usable FSS skill (0.888) at the finest radius against a 0.549 threshold | open | very-low | EXP-0007 (M5 arm + reviewer amendment) | — | `swept-region-metric`, `episode-split` |
| C-035 | **The dissociation is by error TYPE, and rms gets two of three wrong.** At matched or lower rms, high-frequency noise destroys control utility (slate4 −0.04 at rms 0.241) while displacement preserves it (slate4 +0.55 at rms 0.281); amplitude and blur errors cost rms and cost control *nothing* (a=0.5: rms 0.218→0.271, dV Spearman 0.474→0.472) | open | very-low | EXP-0008 (reviewer re-analysis of its own table) | — | `swept-region-metric`, `episode-split` |
| C-034 | P2 (`docs/ideas_log_signal_vs_detail.md` §5): under synthetic degradation of the operator's cube n20 prediction, rms and Lyapunov-dV control utility cross order between displacement and hf-noise; and (partial P3) FSS(r=1) rank-correlates with control utility better than rms does across the spectrum | refuted | EXP-0008 | `docs/ideas_log_signal_vs_detail.md` §5 | `swept-region-metric`, `episode-split`, `canonical-warp`, `warp-blend`, `footprint-splat` |

## Queued re-runs

Ordered by what unblocks the most rows.

1. ~~Fix `grid-convention` + `rasteriser-identity` + `pixel-index-origin`~~ —
   done 2026-09-05, commit `aac084e3`. `pixel-index-origin` remains broken.
2. ~~Re-run `linear_foresight_report.md` §2 and §2.7 on corrected grids
   (C-001, C-008)~~ — **C-001 done, reversed (EXP-0009).** C-008 still queued:
   run the switched-operator cells (res=16/crop=0.25, bins=2 and 3) FIRST,
   before any res=64 cell — they are ~1000x cheaper (D=256 vs D=4096) and
   were the ones actually left undone when EXP-0009 ran out of budget.
   ~5 min CPU, no new data needed (`scripts/probes/exp0009_rerun.py --bins 2|3
   --res 16 --crop 0.25` already exists). A full same-session res=64 cell for
   C-001 (currently sourced from EXP-0001's pre-fix numbers, `provenance`
   downgrade) needs `fit_operator_nonneg` capped well below its default
   `max_iter=4000` (measured ~3.1 s/iter at D=4096 under CPU contention; even
   a 300-iter cap is ~15 min) or a GPU.
3. EXP-0002 under leave-one-run-out (C-019).
4. A density-target cube run, to break the target-matching confound EXP-0005's
   amendment introduced (C-017).
5. Retrain one UNet config with the raster fixed (C-020). GPU, ~1 h.
6. The signal-vs-detail metric programme (C-030) —
   `docs/ideas_log_signal_vs_detail.md`. Next: a dedicated confirmatory run for
   C-037 (rms vs control utility over real fitted models) with a measured noise
   floor and a second seed — EXP-0013 found it while testing something else.
7. C-008 is genuinely untested on the fixed grid (EXP-0009 ran out of budget
   before reaching it). Run the `--bins` config first: at res 16 / crop 0.25 it
   is ~1000x cheaper than the res-64 cell that starved it.
8. ~~Re-run `linear_foresight_report.md` §5 and §6 on corrected grids
   (C-006, C-004)~~ — **done (EXP-0010).** C-006 reconfirmed in both regimes.
   C-004 is regime-split (contested), on a 2-of-8 / 2-of-16 episode cap only
   — `transforms.functional.particles_to_occupancy`'s `footprint_radius` path
   costs ~60ms/transition independent of particle count (measured; an
   unrelated finding in EXP-0010), making the full episode sets ~15-18 min CPU
   to load alone. The full-episode load (tightens C-004's piled-cube noise
   floor) and a res=64 cell (out of scope per the task's own cost warning,
   projected >1h/fit at D=4096) are the two cells that would close it out.
