# State of play — generated, do not hand-edit

`python scripts/summarise_register.py` · 33 claims · 27 live records · 4 superseded

## Claims by status

### supported (17)

| id | grade | claim | evidence |
|---|---|---|---|
| C-001 | very-low | REVERSED 2026-09-05, and the literal target cell now confirmed same-session (was: nothing beats persistence). With `grid-convention` fixed (commit `aa | EXP-0001 (res=64 numbers, pre-fix simulated), EXP-0009 (res= |
| C-002 | — | Re-sourced, cube-only. The SE(2) warp costs more than one push changes unless the field is smoothed to σ≈1 | `linear_foresight_report.md` §1; identity-baseline rows of E |
| C-003 | — | Re-sourced, cube-only. World-frame occupancy mass is conserved to 0.4% on cubes | `linear_foresight_report.md` §4 `mass-conservation` |
| C-005 | — | Every MPC derives blade yaw from the push direction; training data is oblique 92% of the time and deployment perpendicular 99.6% | `linear_foresight_report.md` §3 — (action-based) |
| C-006 | low | RECONFIRMED 2026-09-05, then reconfirmed again at FULL episode count 2026-09-06 (EXP-0023), retiring the 2-episode-cap `incomplete-design`. Re-measure | EXP-0010 (2-episode cap), EXP-0023 (full data, both regimes) |
| C-009 | low | Per-push band displacement is 84% predictable from grid-visible features but only 58% linearly. Re-tested 2026-09-05 (EXP-0019). Reproduced exactly (l | EXP-0019 `episode-split`, `settled-state` |
| C-011 | low | Re-sourced, cube-only. mean-delta (zero parameters) is the baseline that matters, not persistence — the canonical frame normalises the action away, an | EXP-0002, EXP-0005, EXP-0006 `canonical-warp` |
| C-017 | low | Confound addressed 2026-09-06 (EXP-0023). On PILED cubes (n20), the original narrow claim reconfirms exactly (mask 0.7393 vs height 0.6308, density 0. | EXP-0005 (piled, mask-target only), EXP-0023 (piled AND scat |
| C-018 | high | The dataset's occupancy channel and its plate/action channel placed world x on opposite grid axes — and the convention the fix standardised on (`dim0= | EXP-0001, EXP-0020 — (establishes `grid-convention`, `world- |
| C-020 | low | Confirmed by retrain, 2026-09-05, then generalised across 7 cells same day. The scattered-monolayer UNet's near-blindness to its action channel (EXP-0 | EXP-0014 (EXP-0004 pre-fix, inconclusive), EXP-0021 (7 cells |
| C-031 | very-low | On cubes, blur moves the operator's error ~4x more than the choice of view does (33 points vs 8 across σ 0→1.5) | EXP-0003 `canonical-warp`, `warp-blend`, `swept-region-metri |
| C-035 | low | The dissociation is by error TYPE, and rms/accuracy gets two of three wrong. CONFIRMED CONFOUND-FREE (EXP-0025, same-state slates, n=50 states). At ma | EXP-0008 (reviewer re-analysis, cross-state), EXP-0025 (same |
| C-037 | moderate | Narrowed and confirmed, 2026-09-05. The rms/control-utility dissociation holds ONLY at rank=1, not across rank 1-128. Rank-1 beats mean-delta on swept | EXP-0013, EXP-0017 (5 seeds x res 32, 3 seeds x res 64, both |
| C-039 | moderate | Completed at n=50, 2026-09-05. Cross-state candidate slates inflate the measured ranking damage from high-frequency prediction noise about two-fold (8 | EXP-0012 (1597 transitions, 50 verified same-state slates) — |
| C-040 | moderate | A centred convex target is DEGENERATE for a centred pile: `dV = 0` identically, since no mass lies outside the mask before or after. Off-centre target | EXP-0012 (2 of 50 slates showed any variation under `center` |
| C-041 | low | Narrowed 2026-09-05. The UNet beats the linear operator in 14/14 cells on a sharp target (by 4.4-10.9 points). Scored on a common blurred (sigma=1) ta | EXP-0021 (sharp), EXP-0022 (the narrowing), EXP-0024 (contro |
| C-042 | very-low | New 2026-09-05. The pre-registered concern that a UNet's swept-region advantage over persistence on scattered monolayers comes mainly from predicting  | EXP-0021 (7 cells x 2 strata) `grid-convention`, `rasteriser |

### refuted (5)

| id | grade | claim | evidence |
|---|---|---|---|
| C-008 | moderate | Contact-switching does not transfer from scalar targets to the pixel operator. Settled 2026-09-05 by leave-one-run-out (EXP-0016). EXP-0015's +0.0001/ | — EXP-0009 (attempted; no cell completed), EXP-0015, EXP-001 |
| C-015 | EXP-0002, EX | Restated cube-only. Pile depth is what makes the linear operator do state-dependent work — monolayers should show no margin over mean-delta | withdrawn cube/continuum comparison, commit `242a9cc1` `rast |
| C-032 | EXP-0007 | P1 (`docs/ideas_log_signal_vs_detail.md` §5): FSS skill vs. neighbourhood radius rises and saturates by r≈3-5px on cube n20, near the σ≈1-1.5 that hel | `docs/ideas_log_signal_vs_detail.md` §5 `swept-region-metric |
| C-034 | very-low | P2 (`docs/ideas_log_signal_vs_detail.md` §5): under synthetic degradation of the operator's cube n20 prediction, rms and Lyapunov-dV control utility c | EXP-0008 `docs/ideas_log_signal_vs_detail.md` §5 `swept-regi |
| C-036 | low | Neither ridge shrinkage nor rank truncation trades predictive accuracy for control utility. lambda*_rms = 10 and lambda*_control = 1..10; rank*_rms =  | — EXP-0013 `canonical-warp`, `swept-region-metric`, `episode |

### contested (2)

| id | grade | claim | evidence |
|---|---|---|---|
| C-004 | very-low | Non-negativity beats ridge for the pixel operator (their Fig. 7). Re-run 2026-09-06 at FULL episode count (EXP-0023), superseding EXP-0010's 2-of-8/2- | EXP-0010 (capped; superseded read) EXP-0023 (full data, all  |
| C-007 | low | In scalar targets, essentially all the nonlinearity is one variable: how much material the blade meets. Narrowed 2026-09-05, target-dependent (EXP-001 | EXP-0019 (mean-displacement arm) EXP-0019 (max-displacement  |

### open (4)

| id | grade | claim | evidence |
|---|---|---|---|
| C-019 | low | The linear operator's margin over mean-delta is regime-independent (+0.15..+0.31 across scattered monolayers and two-layer heaps) | EXP-0002, EXP-0006 — `rasteriser-identity`, `swept-region-me |
| C-033 | very-low | The linear operator's error as a fraction of the band signal falls ~3x from the finest scale (0.686) to ~8 px (0.221) — fine detail is harder, not unp | EXP-0007 (M5 arm + reviewer amendment) — `swept-region-metri |
| C-038 | moderate | On scattered 50-cube monolayers at res 16, canonical crop 0.5 is much the best window: explained 0.415 vs 0.228 (crop 1.0) and 0.137 (crop 0.25) | EXP-0015 (incidental) — `canonical-warp`, `swept-region-metr |
| C-043 | very-low | New 2026-09-05, deliberately left open. Both the UNet and the linear operator's swept-region advantage over persistence grows monotonically with objec | EXP-0021 (consistent direction across 4 series, no noise flo |

### superseded (1)

| id | grade | claim | evidence |
|---|---|---|---|
| C-022 | — | *(duplicate of C-015 as originally filed; merged 2026-09-05)* | — — |

### diff\ (1)

| id | grade | claim | evidence |
|---|---|---|---|
| C-045 | =1.5e-8) and | Corrected 2026-09-06 (paired test). The UNet's image-accuracy edge over the linear operator DOES survive into control, but is small against a near-sat | R_K\ (Lyapunov units) L10mm pays is within ~2x of L20mm/L40m |

### narrowed (3)

| id | grade | claim | evidence |
|---|---|---|---|
| C-030 | low | Narrowed 2026-09-05. A signal-sensitive metric ranks models for MPC better than pixel rms does. Both strong forms are refuted (EXP-0007: the fine band | — EXP-0007, EXP-0008, EXP-0017, EXP-0024_v2 (a genuinely sha |
| C-044 | low | The UNet's advantage over the linear operator is entirely high-frequency. On coarse structure surviving a sigma=1 blur the two model classes are equiv | EXP-0022, EXP-0024 (control measured directly), EXP-0026 (K- |
| C-046 | low | New 2026-09-06. Selection pressure amplifies a prediction error's control cost only when the error is independent across candidates. Sweeping the cand | EXP-0026 (K-sweep over 18 arms, 50 slates, paired), EXP-0026 |

## Records

| id | verdict | grade | downgrades | title |
|---|---|---|---|---|
| EXP-0001 | supported | high | — | The dataset's occupancy and plate channels are mutually transposed |
| EXP-0002 | supported | moderate | imprecision | The operator's margin over mean-delta is roughly regime-independent |
| EXP-0003 | supported | low | imprecision,indirectness | Blur moves the operator's error by ~33 points on cubes; the view choic |
| EXP-0005 | supported | moderate | imprecision | Depth channels do not help predict the cube silhouette |
| EXP-0006 | supported | low | imprecision,untested-dependency | The margin over mean-delta is flat from 20 to 30 piled cubes, at 2-5x  |
| EXP-0007 | refuted | low | imprecision,indirectness | P1 refuted: deconfounded FSS skill is largest at r=1 and falls monoton |
| EXP-0008 | refuted | low | imprecision,incomplete-design | P2 refuted (no crossing: hf-noise hurts control utility more than disp |
| EXP-0010 | supported | low | imprecision,incomplete-design | Post-fix (aac084e3) re-run: the deposit profile reproduces in both reg |
| EXP-0012 | supported | low | imprecision,untested-dependency | Same-state candidate slates collected and verified (6 of a planned 50  |
| EXP-0013 | refuted | low | imprecision,incomplete-design | Neither shrinkage nor rank trades accuracy for control — but rms mis-r |
| EXP-0016 | supported | moderate | selection | Contact switching beats a single operator by +0.0059 explained, and an |
| EXP-0017 | refuted | moderate | indirectness | The rms/control dissociation survives a noise floor only at rank 1 --  |
| EXP-0018 | supported | very-low | imprecision,incomplete-design,inconsistency,un | Five attacks on C-001's reversal, all fail cleanly, and the literal re |
| EXP-0019 | refuted | low | imprecision,untested-dependency | C-009's 84%/58% survives a trivial-baseline check; C-007's "essentiall |
| EXP-0020 | supported | high | — | The fixed grid convention is right against physics, not merely self-co |
| EXP-0021 | supported | very-low | imprecision,inconsistency,indirectness,provena | Post-fix-raster UNet beats the linear operator by 4-11 points in every |
| EXP-0022 | supported | low | imprecision,indirectness | The UNet's win over the linear operator is entirely high-frequency; on |
| EXP-0023 | supported | very-low | imprecision,incomplete-design,inconsistency | Two full-power repeats: (1) the mask-vs-depth channel confound resolve |
| EXP-0024 | supported | low | imprecision,untested-dependency | Does the UNet's fine-detail image-accuracy advantage over the linear o |
| EXP-0024_v1 | supported | very-low | imprecision,provenance,untested-dependency | EXP-0024/EXP-0026 re-run on Genesis/data/slates_multistep/n20_L{10,20, |
| EXP-0024_v2 | inconclusive | very-low | imprecision,inconsistency,untested-dependency | CORRECTED 2026-09-07 (coordinator review). The first pass of this reco |
| EXP-0025 | supported | low | imprecision,untested-dependency | EXP-0008's full degradation spectrum re-run on same-state slates (n=50 |
| EXP-0026 | refuted | low | imprecision,untested-dependency | The K=4 objection fails: from top-1-of-4 to top-1-of-31 the ridge oper |
| EXP-0026_v1 | supported | very-low | imprecision,provenance,untested-dependency | C-046 (selection pressure amplifies only independent-per-candidate err |
| EXP-0026_v2 | refuted | very-low | imprecision,inconsistency,untested-dependency | C-046's clean "systematic degradation arms are K-invariant, only indep |
| EXP-0027 | supported | high | — | warp-only's positive image accuracy at L20mm/L40mm (+0.0254/+0.0442, E |
| EXP-0028 | supported | high | — | L10mm's accuracy=-0.44 / slateK_exact=0.91 dissociation (EXP-0024_v1)  |

### Superseded

- **EXP-0004** → ['EXP-0014', 'EXP-0021'] — The trained UNet barely uses its action channel
- **EXP-0009** → ['EXP-0018'] — Post-fix pixel operator beats persistence and mean-delta (C-001 reverses); the c
- **EXP-0014** → ['EXP-0021'] — Retrained on the fixed raster, the UNet now depends heavily on its action channe
- **EXP-0015** → ['EXP-0016'] — Contact switching helps the pixel operator by 0.000-0.007 explained — consistent

## Evidence health

| grade | n | records (downgrade-domain count) |
|---|---|---|
| high | 4 | EXP-0001(0), EXP-0020(0), EXP-0027(0), EXP-0028(0) |
| moderate | 4 | EXP-0002(1), EXP-0005(1), EXP-0016(1), EXP-0017(1) |
| low | 12 | EXP-0003(2), EXP-0006(2), EXP-0007(2), EXP-0008(2), EXP-0010(2), EXP-0012(2), EXP-0013(2), EXP-0019(2), EXP-0022(2), EXP-0024(2), EXP-0025(2), EXP-0026(2) |
| very-low | 7 | EXP-0018(4), EXP-0021(4), EXP-0023(3), EXP-0024_v1(3), EXP-0024_v2(3), EXP-0026_v1(3), EXP-0026_v2(3) |

| downgrade domain | records |
|---|---|
| imprecision | 21 |
| untested-dependency | 11 |
| indirectness | 5 |
| incomplete-design | 5 |
| inconsistency | 5 |
| provenance | 3 |
| selection | 1 |
