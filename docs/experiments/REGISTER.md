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
| C-043 | **New 2026-09-05, deliberately left open.** Both the UNet and the linear operator's swept-region advantage over persistence grows monotonically with object count (n=5→10→20) in all 4 independent series tested (UNet blind, UNet contact, linear blind, linear contact), while the UNet-over-linear margin itself stays roughly flat (~7-11 points) across n. NOT claimed as an established trend: one training run and one train/test split per cell, no seed-repeat noise floor measured — a 3-point curve per series is exactly the shape the experiment-log skill warns against over-reading | open | very-low | EXP-0021 (consistent direction across 4 series, no noise floor) | — | `swept-region-metric`, `episode-split` |
| C-004 | Non-negativity beats ridge for the pixel operator (their Fig. 7). **Re-run 2026-09-06 at FULL episode count (EXP-0023), superseding EXP-0010's 2-of-8/2-of-16-episode-capped read.** The capped run's clearest result does not survive full data: nonneg's raw-count win narrows from 3-of-4 to 2-of-4 cells (still both scattered), and its margin on those 2 cells shrinks from 0.0058-0.0076 rms (outside the documented floor) to 0.0011-0.0022 rms (inside it); the 2 piled cells stay tied, now with ridge nominally winning both (was 1-1). At full power, accuracy is 0.372/0.430 (nonneg) vs 0.351/0.420 (best ridge) on scattered (crop 1.0/0.5) and 0.639/0.644 (nonneg) vs 0.641/0.645 (best ridge) on piled — **no cell clears the (borrowed, not re-measured at this N) ~0.001-0.004 rms floor.** res=64 still untested; measured this session at a projected ~59 min/fit (D=4096), confirming EXP-0010's cost warning | contested | very-low | EXP-0010 (capped; superseded read) | EXP-0023 (full data, all 4 cells inside the documented floor) | `grid-convention`, `rasteriser-identity`, `swept-region-metric`, `episode-split` |
| C-036 | **Neither ridge shrinkage nor rank truncation trades predictive accuracy for control utility.** lambda*_rms = 10 and lambda*_control = 1..10; rank*_rms = 256 and rank*_control = 512 (truncation hurts both monotonically) | refuted | low | — | EXP-0013 | `canonical-warp`, `swept-region-metric`, `episode-split` |
| C-037 | **Narrowed and confirmed, 2026-09-05.** The rms/control-utility dissociation holds ONLY at rank=1, not across rank 1-128. Rank-1 beats mean-delta on swept-region rms (+8.8 to +18.7 points) but loses to it on slate-4 control utility (-0.087 to -0.096, partial-corr -0.088 to -0.096), resolved at 6.7-31 seed-sd across 5 seeds (res 32) and 3 seeds (res 64), on `corner` (`center` is 68-96% degenerate — C-040). Ranks 2-512 do NOT dissociate: they beat mean-delta on BOTH metrics, resolved at 6.6-31 seed-sd. The original "ranks 1-128 all dissociate" reading partly traced to a mistranscribed mean-delta baseline in EXP-0013's own Markdown table (0.601 = partial-corr value, not 0.395 = the true raw slate4 value, for the `center` column only — the script's own computed optima were unaffected) | supported | moderate | EXP-0013, EXP-0017 (5 seeds x res 32, 3 seeds x res 64, both goals, noise floor measured) | — | `canonical-warp`, `warp-blend`, `swept-region-metric`, `episode-split`, `particle-projection` |
| C-038 | On scattered 50-cube monolayers at res 16, canonical crop 0.5 is much the best window: explained 0.415 vs 0.228 (crop 1.0) and 0.137 (crop 0.25) | open | moderate | EXP-0015 (incidental) | — | `canonical-warp`, `swept-region-metric` |
| C-039 | **Completed at n=50, 2026-09-05.** Cross-state candidate slates inflate the measured ranking damage from high-frequency prediction noise about two-fold (84% -> 42% relative Spearman drop). The mechanism survives: same-state, noise still costs 5.5x more ranking quality than displacement (42.1% vs 7.6%), and the ordering holds at matched or lower rms | supported | moderate | EXP-0012 (1597 transitions, 50 verified same-state slates) | — | `settled-state`, `canonical-warp`, `swept-region-metric` |
| C-040 | A centred convex target is DEGENERATE for a centred pile: `dV = 0` identically, since no mass lies outside the mask before or after. Off-centre targets are the informative simple goals for MPC benchmarking | supported | moderate | EXP-0012 (2 of 50 slates showed any variation under `center`) | — | — |
| C-044 | **The UNet's advantage over the linear operator is entirely high-frequency.** On coarse structure surviving a sigma=1 blur the two model classes are equivalent (swing 9-12 points in 4/4 cells, ranking reverses in 2). Since EXP-0008/C-039 find control ranking is destroyed by high-frequency noise and untouched by blur, the UNet may be better at exactly the component MPC does not consume | open | low | EXP-0022 | — | `swept-region-metric`, `canonical-warp`, `episode-split` |
| C-030 | **Narrowed 2026-09-05.** A signal-sensitive metric ranks models for MPC better than pixel rms does. Both strong forms are refuted (EXP-0007: the fine band is not unpredictable; EXP-0008: a global FSS-for-rms swap loses). The model-level support it briefly had from C-037 is much weaker than recorded — C-037 now holds only at rank 1, and EXP-0013's table that suggested otherwise was mistranscribed | contested | low | — | EXP-0007, EXP-0008, EXP-0017 | `canonical-warp`, `swept-region-metric` |
| C-008 | Contact-switching does not transfer from scalar targets to the pixel operator. **Settled 2026-09-05 by leave-one-run-out (EXP-0016).** EXP-0015's +0.0001/+0.0069/+0.0050 (crop 0.25/0.5/1.0) was measured against a ~0.03 floor borrowed from a different design; EXP-0016 measured the floor directly on THIS design at crop 0.5 (8 folds, one run held out each time): mean paired diff (switched−single) = +0.0059, sd across folds = 0.0027 (the measured floor), sem = 0.0010, mean/sem = 6.1 — every one of 8 folds positive. The gap is real, ~20x the measured floor on its best fold, not the borrowed one. **The claim as stated (no transfer) is refuted** — switching does transfer, consistently, but the effect is small (~0.6 points of explained variance on an operator that already explains ~42) | refuted | moderate | — | EXP-0009 (attempted; no cell completed), EXP-0015, EXP-0016 (LORO, crop 0.5 only) | `grid-convention`, `rasteriser-identity`, `swept-region-metric`, `episode-split`, `canonical-warp`, `warp-blend` |
| C-007 | In scalar targets, essentially all the nonlinearity is one variable: how much material the blade meets. **Narrowed 2026-09-05, target-dependent (EXP-0019).** Reproduced §2.6's 4-bin contact stratification exactly (91/103/110/109% share) and confirmed it is NOT a booster-tuning artifact: a 4x-larger, depth-capped, early-stopped booster changes boosted R2 by <=0.02 in every stratum, and linear RMSE beats boosted RMSE in every one of the 4 strata (mean band displacement). **But on MAX per-particle displacement in the same 4 strata — same particles, same stratification, a differently-shaped scalar target — the claimed pattern is ABSENT**: linear share falls monotonically 95%→87%→80%→73% as contact grows, and boosted also wins on RMSE (e.g. largest-contact stratum: 5.03mm linear vs 3.98mm boosted) — the opposite of "share stays ~100% because it's all one scalar," resolved against a measured per-fold R2 sd of 0.012-0.043. So the claim holds for the mean/aggregate target §2.6 used and fails for a threshold-sensitive alternative computed from identical data | contested | low | EXP-0019 (mean-displacement arm) | EXP-0019 (max-displacement arm) | `episode-split`, `settled-state` |

## Supported

| ID | Claim | Status | Grade | Supported by | Depends on |
|---|---|---|---|---|---|
| C-018 | The dataset's occupancy channel and its plate/action channel placed world x on opposite grid axes — and the convention the fix standardised on (`dim0=world_x`) is the **physically correct** one, not merely the self-consistent one: grid transport aligns with the world-frame push at cos +0.979, against −0.001 for the pre-fix alternative | supported | high | EXP-0001, EXP-0020 | — (establishes `grid-convention`, `world-frame-alignment`) |
| C-020 | **Confirmed by retrain, 2026-09-05, then generalised across 7 cells same day.** The scattered-monolayer UNet's near-blindness to its action channel (EXP-0004: shuffling cost only 3.1 of 8.2 points) was caused by the transposed occupancy/plate raster. Retrained identically on the corrected raster: the model's advantage over persistence grows to 28.0 points, and shuffling the action now costs 38.5 points — enough that a wrong action scores *worse* than persistence (110.5%) — where zeroing it collapses the model to persistence (99.7%) (EXP-0014, whole-image rms, n=50 only). **EXP-0021 reruns this on the swept-region metric across 7 cells (n=5/10/20 blind and contact-filtered, plus n=50): the zero-action collapse replicates in all 7 (97.7-101.0% of persistence) and the shuffle-gap destroys 87-120% of the model's advantage in all 7, but the literal sign-flip (shuffled worse than persistence) recurs in only 2 of 7 cells — a real but milder-than-EXP-0014 effect, not a full replication of the single-cell 110.5% result.** | supported | low | EXP-0014 (EXP-0004 pre-fix, inconclusive), EXP-0021 (7 cells, swept-region metric, sign-flip only 2/7) | `grid-convention`, `rasteriser-identity` |
| C-041 | **Narrowed 2026-09-05.** The UNet beats the linear operator in 14/14 cells **on a sharp target** (by 4.4-10.9 points). Scored on a common blurred (sigma=1) target the advantage collapses to -4.8..+0.7 points and the ranking reverses in half the cells — the entire advantage is high-frequency detail | supported | low | EXP-0021 (sharp), EXP-0022 (the narrowing) | — | `grid-convention`, `swept-region-metric`, `episode-split` |
| C-042 | **New 2026-09-05.** The pre-registered concern that a UNet's swept-region advantage over persistence on scattered monolayers comes mainly from predicting no-ops is REFUTED for this design: restricting to contact>0 transitions leaves the UNet's (and the linear operator's) advantage over persistence unchanged or very slightly larger in all 7 cells, rather than collapsed toward zero | supported | very-low | EXP-0021 (7 cells x 2 strata) | `grid-convention`, `rasteriser-identity`, `swept-region-metric`, `episode-split` |
| C-031 | On cubes, blur moves the operator's error ~4x more than the choice of view does (33 points vs 8 across σ 0→1.5) | supported | very-low | EXP-0003 | `canonical-warp`, `warp-blend`, `swept-region-metric`, `episode-split`, `particle-projection` |
| C-017 | **Confound addressed 2026-09-06 (EXP-0023).** On PILED cubes (n20), the original narrow claim reconfirms exactly (mask 0.7393 vs height 0.6308, density 0.6557, matching EXP-0005's 0.739/0.631/0.656 to 3 dp) AND survives its own confound test: under a density-delta target on the SAME piled data, mask (0.4150) is not the worst input by a wide margin (density wins by only 2.3 pts, 0.4380 vs 0.4150; mask still beats height 0.4143) — consistent with "depth genuinely uninformative" for piled heaps specifically. But the analogous test on SCATTERED cubes (granularity/n20, blind, 40mm) shows the opposite: mask wins its own target (0.5223) yet loses the density target badly (0.6896 vs density 0.8317, height 0.8294 — a 13-14 pt gap), consistent with target-matching dominating there. So the claim holds on piled cubes but does not generalise to scattered ones on the same design — the confound is resolved differently by regime, not resolved once for all cube data | supported | low | EXP-0005 (piled, mask-target only), EXP-0023 (piled AND scattered, both targets — confound test) | `particle-projection`, `episode-split` |
| C-011 | **Re-sourced, cube-only.** mean-delta (zero parameters) is the baseline that matters, not persistence — the canonical frame normalises the action away, and mean-delta alone reaches 0.12–0.35 explained | supported | low | EXP-0002, EXP-0005, EXP-0006 | `canonical-warp` |
| C-009 | Per-push band displacement is 84% predictable from grid-visible features but only 58% linearly. **Re-tested 2026-09-05 (EXP-0019).** Reproduced exactly (linear 0.576, boosted 0.836, same L040/L040b glob, n=7680) and checked against the leading alternative explanation: only 4.6% of transitions have zero contact (band empty, displacement deterministically 0), and a single trivial feature (contact count alone) reaches only 17-25% R2 — nowhere near 84%. Restricting to contact>0 only widens the share (69%→76%), not narrows it. The 84%/58% headline is not an artifact of a near-deterministic zero-contact tail | supported | low | EXP-0019 | `episode-split`, `settled-state` |
| C-005 | Every MPC derives blade yaw from the push direction; training data is oblique 92% of the time and deployment perpendicular 99.6% | supported | — | `linear_foresight_report.md` §3 | — (action-based) |
| C-003 | **Re-sourced, cube-only.** World-frame occupancy mass is conserved to 0.4% on cubes | supported | — | `linear_foresight_report.md` §4 | `mass-conservation` |
| C-002 | **Re-sourced, cube-only.** The SE(2) warp costs more than one push changes unless the field is smoothed to σ≈1 | supported | — | `linear_foresight_report.md` §1; identity-baseline rows of EXP-0001, EXP-0003 | `canonical-warp` |
| C-001 | **REVERSED 2026-09-05, and the literal target cell now confirmed same-session** (was: nothing beats persistence). With `grid-convention` fixed (commit `aac084e3`), the linear/ridge pixel operator on scattered cube monolayers beats persistence by 30-46 points of explained variance across 11 of 12 tested res x crop cells (57.8% of persistence rms at res=64/crop=0.5 — run directly on the fixed pipeline this session, matching EXP-0001's earlier borrowed 57.9% to 0.1pt — and 59.2-70.5% at res=16/8/crop=0.5) and beats mean-delta too (+0.17 to +0.38 explained; margin clears an 8-fold LORO noise floor by ~45x at res=16/crop=0.5). One cell (res=8, crop=1.0) does not reverse (104.2%, worse than persistence) — a genuine, logged inconsistency outside the headline configuration. No LORO yet at the literal res=64 cell itself (single split only there) | supported | very-low | EXP-0001 (res=64 numbers, pre-fix simulated), EXP-0009 (res=8, actual fixed pipeline), EXP-0018 (res=64/crop=0.5 confirmed same-session; full res x crop grid; 8-fold LORO at res=16/crop=0.5; 5 adversarial attacks tested, none overturn it) | `grid-convention`, `rasteriser-identity` |
| C-006 | **RECONFIRMED 2026-09-05, then reconfirmed again at FULL episode count 2026-09-06 (EXP-0023), retiring the 2-episode-cap `incomplete-design`.** Re-measured on the grid-convention-fixed pipeline (commit `aac084e3`): the paper's Fig. 5 deposit structure (depletion across the swept band, deposition peaking just ahead) reproduces on BOTH a scattered monolayer (full 8/8 episodes, N=2560) and a piled (n20) cube dataset (full 16/16 episodes, N=4840) — peak effects 10-100x the per-fold sd, peak magnitudes within 5% and peak locations identical to the pixel vs the 2-episode-capped run; piled cubes show ~2x the amplitude at about half the column-width of scattered | supported | low | EXP-0010 (2-episode cap), EXP-0023 (full data, both regimes) | `grid-convention`, `rasteriser-identity`, `canonical-warp`, `footprint-splat` |

## Refuted and invalidated

Measured correctly, concluded wrongly. Listed so nothing cites them, and so the
re-runs are queued.

| ID | Claim | Status | By | Was stated in | Depends on |
|---|---|---|---|---|---|
| C-015 | **Restated cube-only.** Pile depth is what makes the linear operator do state-dependent work — monolayers should show no margin over mean-delta | refuted | EXP-0002, EXP-0006 | withdrawn cube/continuum comparison, commit `242a9cc1` | `rasteriser-identity` |
| C-022 | *(duplicate of C-015 as originally filed; merged 2026-09-05)* | superseded | — | — | — |
| C-032 | P1 (`docs/ideas_log_signal_vs_detail.md` §5): FSS skill vs. neighbourhood radius rises and saturates by r≈3-5px on cube n20, near the σ≈1-1.5 that helped in EXP-0003 | refuted | EXP-0007 | `docs/ideas_log_signal_vs_detail.md` §5 | `swept-region-metric`, `episode-split`, `canonical-warp`, `warp-blend`, `footprint-splat` |
| C-033 | The linear operator's error as a fraction of the band signal falls ~3x from the finest scale (0.686) to ~8 px (0.221) — fine detail is harder, not unpredictable, and the operator has usable FSS skill (0.888) at the finest radius against a 0.549 threshold | open | very-low | EXP-0007 (M5 arm + reviewer amendment) | — | `swept-region-metric`, `episode-split` |
| C-035 | **The dissociation is by error TYPE, and rms/accuracy gets two of three wrong. CONFIRMED CONFOUND-FREE (EXP-0025, same-state slates, n=50 states).** At matched accuracy, high-frequency noise still costs far more control utility than displacement or blur (accuracy 0.184 hf-noise m=1.0 vs 0.178 blur s=2.0, matched within 0.006: slate4 0.769 vs 0.931). Amplitude survives as free (<=0.8% relative slate4/spearman drop across all 4 levels, same-state). Blur survives QUALITATIVELY but not literally "free": a small, real, monotonic cost appears same-state (slate4 −2.6% at sigma=2) that EXP-0008's cross-state table did not show — still 10-30x smaller than hf-noise's cost at matched severity | supported | low | EXP-0008 (reviewer re-analysis, cross-state), EXP-0025 (same-state, full spectrum, matched-accuracy crossing, n=50) | — | `swept-region-metric`, `episode-split`, `canonical-warp`, `warp-blend`, `footprint-splat`, `settled-state` |
| C-034 | P2 (`docs/ideas_log_signal_vs_detail.md` §5): under synthetic degradation of the operator's cube n20 prediction, rms and Lyapunov-dV control utility cross order between displacement and hf-noise; and (partial P3) FSS(r=1) rank-correlates with control utility better than rms does across the spectrum. As LITERALLY STATED (matched-index comparison), still refuted — the reviewer amendment found the crossing exists but in the reverse direction from what was predicted (rms UNDER-penalises noise, not over-penalises it); see C-035 for that corrected-direction claim, now confirmed on same-state slates by EXP-0025. FSS-vs-rms (the P3 half) was not re-tested same-state and stands as EXP-0008 measured it | refuted | very-low | EXP-0008 | `docs/ideas_log_signal_vs_detail.md` §5 | `swept-region-metric`, `episode-split`, `canonical-warp`, `warp-blend`, `footprint-splat` |

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
