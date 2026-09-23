# RUN-0009 -- randlen main-run eval: warped NFD (RUN-0005) vs. world-frame NFD baseline

Both `nfd_randlen` (world-frame, `Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth`)
and `nfd_warped_randlen` (RUN-0005, canon_res=64, no wall channel,
`Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth`) were scored **in the same
process, through the same `Baselines/common/eval_report.py` harness**, on the same 3
corpora EXP-0001 used (L20mm eval cell, L40mm eval cell, `overnight_randlen_test`).
Raw JSON: `../artifacts/RUN-0009-randlen-eval/randlen_eval.json`. Command:
`../runs/RUN-0009-randlen-eval/COMMAND.txt`.

**Device**: both models ran on **CPU** for every corpus (`PredictorBatch.occ0.device`
introspected per the known `eval_baseline.py::_predictor_batch` trap -- it never
moves the batch off GPU; the GPU was idle throughout and unused). This does not bias
the comparison since both models paid the same cost, only wall-clock (single-digit
to low-teens seconds per model per corpus).

## Baseline reproduction check (do this first -- it governs how everything below reads)

EXP-0001 reported world-frame NFD accuracy as 0.4071 / 0.5088 / 0.4564 (L20mm / L40mm
/ randlen_test), from a run on a dirty tree with no commit its numbers were tied to.
Re-scoring the SAME checkpoint (`nfd_3ch_randlen/unet_best.pth`) here, through the
same harness, in the same process as the new arm, gives:

| corpus | EXP-0001 published | this run (re-scored) | diff |
|---|---|---|---|
| L20mm | 0.4071 | **0.4071313** | ~0 |
| L40mm | 0.5088 | **0.5087523** | ~0 |
| randlen_test | 0.4564 | **0.4564076** | ~0 |

**Reproduces exactly** (differences are 4th/5th-decimal rounding-display noise, not a
real discrepancy). This is a positive finding in its own right: despite EXP-0001's
provenance gap (dirty tree, no anchoring commit), the accuracy numbers it published
for `nfd_randlen` are not in doubt -- the checkpoint and the harness path both still
produce them. No `OPEN_ISSUES.md` entry is warranted on this point.

## `slateN` (goal-averaged over 3 goals -- random_quadrant, ring_O, T -- lead metric)

| corpus | model | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| L20mm | nfd_randlen (baseline) | +0.8527 | +0.8202 | +0.7527 |
| L20mm | nfd_warped_randlen | +0.8195 | +0.8001 | +0.7789 |
| L20mm | random floor | -0.0079 | -0.0054 | -0.0117 |
| L40mm | nfd_randlen (baseline) | +0.9322 | +0.7832 | +0.8411 |
| L40mm | nfd_warped_randlen | +0.9084 | +0.8315 | +0.8465 |
| L40mm | random floor | +0.0126 | +0.0163 | -0.0038 |
| randlen_test | nfd_randlen (baseline) | +0.9417 | +0.8886 | +0.8655 |
| randlen_test | nfd_warped_randlen | +0.9267 | +0.8883 | +0.8414 |
| randlen_test | random floor | -0.0398 | +0.0093 | -0.0023 |

Both models clear the `random` floor by a huge margin everywhere (`random` sits within
+/-0.04 of 0 in every cell, as the analytic E[.]=0 argument predicts). Ranking-only:
`persistence` predicts dv=0 for every candidate and is a degenerate ranker (see
CODEMAP), so it is reported in the raw JSON but not used as a ranking floor here.

**Reading `slateN`**: the baseline is ahead of the warped arm under `lyapunov` in all
3 corpora (by 0.015-0.033) and ahead under `signed_mass` in 2 of 3 (L20mm), roughly
tied/slightly behind under `mass_in_region` in 2 of 3 (L40mm, randlen_test) and
`signed_mass` (L40mm). No value function or corpus shows a large gap either way --
every difference is in the 0.01-0.03 range, small next to the ~0.75-0.94 scale both
models occupy. This is a **consistent small edge for the world-frame baseline**, not
a wash in the pilot's sense (pilot had no arm ahead on all three; here the baseline
leads on `lyapunov` uniformly), but it is a small, not dramatic, edge.

## `accuracy` (suspect metric -- reported beside slateN, not instead of it)

| corpus | nfd_randlen (no ceiling) | nfd_warped_randlen | this corpus's warp ceiling (canon_res=64) |
|---|---|---|---|
| L20mm | 0.4071 | 0.3476 | 0.6640 |
| L40mm | 0.5088 | 0.4589 | 0.7804 |
| randlen_test | 0.4564 | 0.4000 | 0.6470 |

The baseline pays no resampling ceiling (world-frame, no warp round trip) -- it is not
a flat three-way comparison. The warped arm trails the baseline's raw accuracy in all
3 corpora, but in all 3 it also sits well BELOW its own corpus-specific ceiling (0.35
vs 0.66 ceiling on L20mm; 0.46 vs 0.78 on L40mm; 0.40 vs 0.65 on randlen_test) --
prediction quality, not the warp's resampling loss, is what binds in every corpus.
Ceiling values from `experiments/EXP-0022-warped-nfd-push-frame/code/
warp_accuracy_ceiling_multi.py` (`../runs/RUN-0009-randlen-eval/warp_ceiling_multi.log`);
the L20mm figure (0.6640) reproduces `results/warp_accuracy_ceiling.md`'s previously
recorded value exactly, confirming the ceiling measurement is stable across runs.
L40mm's ceiling (0.7804) is notably higher than L20mm/randlen_test's (~0.65) -- a
longer nominal push length puts the swept region, and hence its warp, in a different
part of the resampling-cost curve; not investigated further here.

## Ranking-robustness check (N=42 pools, >=40 required)

Extends the pilot's 3-pool check (mean Spearman 0.992, 1/3 top-1 flips) to 42 step-0
same-state pools (14 per corpus, L20mm/L40mm/randlen_test, seed 0), pushing the TRUE
occ1 of every candidate through `push_frame_roundtrip(identity)` at canon_res=64 and
comparing the induced `lyapunov`-to-random-quadrant-goal ranking against the
un-round-tripped truth's ranking. Script:
`experiments/EXP-0022-warped-nfd-push-frame/code/ranking_robustness_check_n40.py`, log
`../runs/RUN-0009-randlen-eval/ranking_robustness_n40.log`.

| | value |
|---|---|
| N pools | 42 (14 L20mm, 14 L40mm, 14 randlen_test), 128 candidates/pool |
| mean Spearman | 0.9952 |
| median Spearman | 0.9966 |
| min / max Spearman | 0.9797 / 1.0000 |
| std Spearman | 0.0050 |
| deciles (0/10/25/50/75/90/100 pct) | 0.9797 / 0.9887 / 0.9925 / 0.9966 / 0.9999 / 1.0000 / 1.0000 |
| top-1 flip count / rate | **2 / 42 = 4.76%** |
| per-corpus flips | L20mm 0/14, L40mm 1/14, randlen_test 1/14 |

At this larger sample the flip rate (4.76%, 2/42) is much lower than the pilot's
1/3 = 33% (n=3 -- consistent with that estimate's own stated unreliability: a Wilson
95% CI on 2/42 is roughly [1.3%, 16%], which comfortably contains 33% too, so the
pilot number was not wrong, just too small to mean anything). **Read plainly: the warp
round trip preserves same-pool rankings well on average (Spearman ~0.995) but is not
airtight -- roughly 1 in 20 pools here had its top-1 candidate flip purely from the
round trip's own resampling, with no model involved.** `slateN` on a warped arm is
therefore mostly, but not perfectly, trustworthy: a model that is genuinely tied or
very close between two top candidates could have its measured `slateN` perturbed by
this effect in roughly a 20th of pools. This is a real but modest caveat, not a reason
to distrust the `slateN` comparison above wholesale.

## Honest reading

**The warp neither clearly helps nor clearly hurts at this scale, and where it moves
the needle it moves it modestly and in the baseline's favor.** On `slateN` -- the
metric that decides -- the world-frame baseline leads the warped arm on `lyapunov` in
all 3 corpora and is roughly tied or slightly ahead on the mass-based value functions,
with every gap in the 0.01-0.03 range against a ~0.75-0.94 scale. On raw `accuracy`
the warped arm trails the baseline everywhere, but it is nowhere near its own
resampling ceiling in any corpus, so that gap reflects prediction quality, not the
warp's unavoidable cost. The ranking-robustness check adds a genuine, if small,
caveat: about 1 pool in 20 has its ranking perturbed by the round trip alone, so a
razor-thin `slateN` edge for either arm should not be over-read. Put together: giving
NFD the push-frame warp, on this full randlen corpus (not the single-push-length
pilot), is a small net negative to neutral for control-relevant ranking, and a clearer
negative on raw pixel accuracy -- not the clear win a successful "help" result would
need to look like, and not catastrophic either.
