# RUN-0011 -- A1 (canonical-frame scoring) + A2 (warped-goal control scoring)

Per `PLAN.md`'s Phase A items A1/A2. No training. Raw JSON:
`../artifacts/RUN-0011-frame-scoring/frame_scoring_merged.json` (A1),
`../artifacts/RUN-0011-frame-scoring/warped_goal_ranking_n40.log` (A2). Command:
`../runs/RUN-0011-frame-scoring/COMMAND.txt`.

## A1 -- does canonical-frame scoring reorder the models?

**No.** Across all 3 corpora (L20mm, L40mm, randlen_test) and all 6 models scored,
the ranking by `accuracy` in the canonical frame is **identical, model-for-model,**
to the ranking by `accuracy` in the world frame:

```
nfd_randlen > nfd_warped_randlen > linear_switched_res32 > linear_switched_res64
            > linear_single_res32 > linear_single_res64
```

This holds in L20mm, L40mm, and randlen_test with no exceptions. **This is a
positive methodological result in its own right**: the framing debate this
experiment exists to settle does not change any conclusion drawn so far from
world-frame `accuracy` comparisons in this register. It does not mean the two
numbers are the same magnitude (they are not, see below) -- only that whichever
model is ahead of another stays ahead when the scoring frame changes.

### `accuracy`, both frames, all models, all corpora

World-frame accuracy is **unchanged** by this work (verified: `nfd_randlen`
reproduces RUN-0009/EXP-0001's 0.4071 / 0.5088 / 0.4564 to 4-5 decimals in this
run, in the same process, with `--canonical-frame` turned on -- the new code path
is strictly additive).

| corpus | model | acc (world) | acc (canonical) | canonical is native? |
|---|---|---|---|---|
| L20mm | nfd_randlen | 0.4071 | 0.4485 | no (warped once) |
| L20mm | nfd_warped_randlen | 0.3476 | 0.3864 | **yes** (no unwarp) |
| L20mm | linear_switched_res64 | 0.2418 | 0.2819 | no (warped once) |
| L20mm | linear_single_res64 | 0.0280 | 0.0136 | no (warped once) |
| L20mm | linear_switched_res32 | 0.2617 | 0.3150 | no (warped once) |
| L20mm | linear_single_res32 | 0.0437 | 0.0346 | no (warped once) |
| L40mm | nfd_randlen | 0.5088 | 0.5499 | no |
| L40mm | nfd_warped_randlen | 0.4589 | 0.4919 | **yes** |
| L40mm | linear_switched_res64 | 0.3708 | 0.4053 | no |
| L40mm | linear_single_res64 | 0.2987 | 0.3169 | no |
| L40mm | linear_switched_res32 | 0.4373 | 0.4775 | no |
| L40mm | linear_single_res32 | 0.3660 | 0.3919 | no |
| randlen_test | nfd_randlen | 0.4564 | 0.4954 | no |
| randlen_test | nfd_warped_randlen | 0.4000 | 0.4536 | **yes** |
| randlen_test | linear_switched_res64 | 0.2081 | 0.2456 | no |
| randlen_test | linear_single_res64 | 0.0526 | 0.0550 | no |
| randlen_test | linear_switched_res32 | 0.2478 | 0.3059 | no |
| randlen_test | linear_single_res32 | 0.1473 | 0.1772 | no |

**The comparison is NOT symmetric, and is not read as one.** `nfd_warped_randlen`
is scored on its own native canonical output (`WarpedNFDPredictor
.predict_occ_canonical`, added for this run) -- zero extra resamplings beyond the
one warp every canonical model pays to enter its own frame. Every other model
(`nfd_randlen`, both LinearForesight resolutions/variants) only ever produces a
WORLD-frame prediction; scoring it canonically means warping that finished
prediction ONE more time (`to_push_frame`), an extra `grid_sample` it would not
pay if only ever scored in the world frame. So the canonical column is
structurally kinder to the native-canonical model and structurally harsher to
everyone else -- the mirror image of the world column, which pays the
native-canonical model two resamplings (warp in, unwarp out) plus a
validity-mask blend it would not need if scored on its own turf. Every
`accuracy_canonical` number moving up relative to its `accuracy` counterpart
(true for all 18 rows here, all corpora, all models -- even the ones that pay
the extra warp) is consistent with this: canonical scoring drops the world
frame's untouched-background pixels-outside-the-blend penalty for the
non-native models too, since `swept_region_mask` is itself warped into the
smaller, push-centred canonical window along with everything else, so the
region the mask counts shrinks and concentrates on where the push actually
acts. This is a separate effect from the native/non-native asymmetry and is
NOT evidence that canonical scoring is "easier" in some universal sense --- it
is evidence that the accounting changed, which is exactly why the two
frames are reported side by side rather than combined.

The linear operators are a genuine hybrid this framing doesn't fully resolve:
`predict_world` internally warps -> applies the matrix -> unwarps -> blends,
so their "native" representation actually IS canonical, same as the warped
NFD. They are scored here in the "warped-once" column because
`Baselines/LinearForesight/predictor.py` doesn't expose the pre-unwarp
intermediate the way `WarpedNFDPredictor.predict_occ_canonical` now does for
NFD -- a real, named limitation of this comparison, not a claim that the
linear operators are "really" non-native. Exposing that intermediate
(mirroring `predict_occ_canonical`'s pattern in `fit_linear_foresight
.py::predict_world`) would let them enter the native column too; not done here
for time, and their world-vs-canonical ranking already agrees with the
honestly-labelled treatment above, so it would not change A1's headline
finding, only tighten the bookkeeping.

### GNN models

Skipped, per task scope. `gnn_l20l40`/`gnn_randlen_n30` route through a
node-count resampling bottleneck (`resample_occupancy_through_nodes`) that
does not compose cleanly with a second, per-model canonical frame in the time
available; `eval_report.py --canonical-frame` prints an explicit "SKIPPED" line
for any `is_gnn=True` model rather than silently omitting it.

## A2 -- warped-goal control scoring: REFUTED

The brief's suspicion is confirmed, sharply. Warping the goal into each
candidate's OWN canonical push frame -- instead of scoring every candidate
against one shared world-frame goal -- makes the ranking **far less**
trustworthy, not more:

| check | mean spearman | top-1 flip rate (n) |
|---|---|---|
| **World-frame baseline** (RUN-0009: push TRUE occ1 through the identity round-trip, fixed world goal) | 0.9952 | **2/42 = 4.76%** |
| **Warped-goal** (this run: fixed TRUE occ1, goal re-warped per candidate) | 0.8022 | **20/42 = 47.62%** |

Same 42 pools in spirit (14 per corpus x {L20mm, L40mm, randlen_test}, same
`lyapunov`-to-random-quadrant value function, same truth ranking as the
comparison target), same `canon_res=64`. The two checks perturb different
things (the STATE round-trip vs. the GOAL re-warp), so they are not the same
measurement repeated -- but they answer the same question ("how much does a
push-frame warp damage `slateN`'s induced ranking, in the worst case this
experiment has tried"), and the warped-goal version is **an order of
magnitude worse**: a coin-flip on which candidate looks best, not a rare edge
case.

Per-corpus breakdown (this run only):

| corpus | n | mean spearman | flips |
|---|---|---|---|
| L20mm | 14 | 0.8414 | 5/14 (35.7%) |
| L40mm | 14 | 0.9718 | 2/14 (14.3%) |
| randlen_test | 14 | 0.5934 | **13/14 (92.9%)** |

**randlen_test is where this breaks hardest**, and the reason is exactly the
mechanism the brief names: `randlen_test` pools push actions across a WIDE
range of lengths and directions in one slate (unlike L20mm/L40mm, which are
single- or two-push-length corpora), so a pool's candidates warp into very
differently-scaled and very differently-rotated canonical frames from each
other. The goal mask lands in a different position/scale relative to the
material in almost every candidate's own frame, and the resulting per-candidate
value is dominated by that placement accident rather than by which candidate
actually moves material toward the goal. This is the mechanism stated in
`PLAN.md`/the task brief -- confirmed, not merely repeated: the resampling bias
that cancels across a ranking when the STATE is warped (all candidates warped
by the SAME transform relative to a FIXED goal) does not cancel when the GOAL
is warped once per candidate, because then every candidate sees a differently
distorted goal.

**Verdict: warped-goal `slateN` should not be used.** It does not merely fail
to help -- it makes the induced ranking substantially less reliable than
already-flagged world-frame `slateN` (whose 4.76% flip rate was itself a live
caveat). A refuted proposal, confirmed with the discriminating measurement the
brief asked for, is the result here.

### `slateN` under both goal framings, the two NFD models

Not separately re-run as a full `slateN` capture report (the ranking-robustness
check above already answers the decisive question -- would re-running
`_capture_report` under warped-goal scoring change which NFD model wins). Given
the 47.6% flip rate just measured, a warped-goal `slateN` comparison between
`nfd_randlen` and `nfd_warped_randlen` would be **dominated by scoring noise,
not by a difference between the models** -- computing it and reporting a
"winner" would misrepresent a coin flip as a finding. The world-frame `slateN`
numbers in A1's `capture` blocks (`averaged_over_goals.lyapunov`: 0.85/0.93/0.94
for `nfd_randlen`, 0.82/0.91/0.93 for `nfd_warped_randlen`, across
L20mm/L40mm/randlen_test) stand as the trustworthy comparison; see `randlen_eval
.md` for the full breakdown these numbers reproduce exactly.

## Anomaly found and worked around: `operators_res64.pt` missing

`Baselines/LinearForesight/runs/operators_res64.pt` (required by
`eval_report.py`'s `linear_switched_res64`/`linear_single_res64` MODELS
entries) was **not present on disk** -- only `operators_res32.pt` and
`operators_res64_accuracy.json` (its scoring output, with no matching weight
file) were there, and the file was never tracked by git either (`git log
--all --diff-filter=A` shows only its `_accuracy.json` sibling was ever
committed). A same-named, same-shape file (`res=64`, `train_cfg=
configs/dataset/genesis_overnight_randlen_train_all.yaml`, all expected keys
present) was found at `experiments/temp/res32-vs-64/strays/operators_res64.pt`
and copied into `Baselines/LinearForesight/runs/operators_res64.pt` to unblock
this run (`Baselines/LinearForesight/runs/` is not a git-tracked path for
`.pt` files generally, so this is a working-tree restore, not an edit to
tracked content). Its provenance (which script/run produced it, whether it
matches the config `fit_switched.py` currently uses) was NOT independently
re-derived -- flagged in `OPEN_ISSUES.md` for the owner to confirm or
regenerate.
