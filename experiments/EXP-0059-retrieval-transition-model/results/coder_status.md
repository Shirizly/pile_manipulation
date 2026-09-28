# Coder status: NFD-with-reference (EXP-0059 section 8)

## UPDATE 2026-09-28 08:15 -- fair "ref vs no-reference" control, full 60-epoch schedule

Coordinator correction: the earlier random-donor control does not isolate "information" --
`ref > random_donor` just as plausibly means random_donor is HURT by noisy-but-present channels,
not that `ref`'s real reference helps. Fair test: `ref` vs an IDENTICAL twin with donor channels
**always zero in training AND eval** (`retrieval_nfd_noref`, dataset `dropout_p: 1.0`, eval
`zero_ref=True` -- same architecture/recipe/epochs as `ref`). Both trained the narrow NFD's own
60-epoch schedule this time (concurrently, ~29s/epoch, ~32 min each), checkpointed every 5 epochs.
`retrieval_nfd_ref` reached epoch 60 as its best (still improving at cutoff); `retrieval_nfd_noref`
plateaued at epoch 53 (7 epochs no-improve by epoch 60, patience=60 so it ran the full schedule
anyway). `retrieval_nfd_ref_zeroed` = the `ref` (A) checkpoint with the reference zeroed at test
time only (same checkpoint as `ref`, not a separately-trained model).

| model | DS-0011 slateN_tough | DS-0009 slateN_tough | acc1 | rollout4 |
|---|---|---|---|---|
| narrow NFD (reference, 60ep, different/smaller corpus regime) | n/a | 0.778 | 0.506 | 0.348 |
| retrieval_nfd_ref (A: real retrieval, 60ep, best=ep60) | 0.790 | 0.752 | 0.478 | 0.324 |
| retrieval_nfd_noref (B: true no-reference twin, 60ep, best=ep53) | 0.752 | 0.778 | 0.486 | 0.322 |
| retrieval_nfd_ref_zeroed (A's ckpt, ref zeroed at test only) | 0.777 | 0.775 | 0.496 | 0.326 |

**Verdict on the fair comparison: a wash, not a win.** A (real retrieval) beats B (true
no-reference) on DS-0011 (0.790 vs 0.752) but LOSES to it on DS-0009 (0.752 vs 0.778) and on acc1
both datasets. A's own zeroed-reference control is statistically indistinguishable from B on
DS-0009 (0.775 vs 0.778) and close on DS-0011 (0.777 vs 0.752, both below A-with-retrieval-on).
Reading all three together: A's real retrieval helps on ONE of the two held-out sets and is
roughly neutral-to-slightly-negative on the other -- there is no clean "information, not noise"
verdict here yet. The earlier `ref > random_donor` finding stands (retrieval beats a genuinely
noisy/wrong reference), but that is now understood as "less bad than noise", not "helps over
having none" -- the corrected, weaker, honest claim. Given A was still improving at epoch 60 while
B had plateaued at 53, more epochs might separate them further; not run (time budget).

---


REWRITE 2026-09-28 06:22-07:45 (coordinator feedback fixed 4 real bugs -- see below). Previous
version of this file / the underlying code had a badly negative result (acc1 -1.1); that run is
superseded and should not be cited. Current numbers are real, cross-validated by the experimenter
agent (independently recomputed slateN_tough from the raw per-pool JSON, matched to 8-9 sig figs).

## Bugs fixed this round (all confirmed root causes, not representation-shift hand-waving)

1. **Donor search was ~70x slower than necessary.** `topk_donors_excluding_chain` had a per-row
   Python loop with `idx[b].tolist()` -- a CUDA device-sync call executed ~12k times. Rewrote it
   fully vectorised (chain keys -> integer ids once, then a `(chunk, search_k)` boolean-mask +
   stable `argsort` per chunk, zero per-row Python/sync calls). Full 11921x11921 search now takes
   ~5.4 minutes (was extrapolating to ~70+ min), which is what let training run on the FULL corpus
   within budget.
2. **Channels 0-2/target now come directly from `Baselines.NFD.nfd_lib.PileSweepData3Ch`**
   (`model/retrieval_nfd/donors.py::build_query_rows_from_pilesweepdata`), bit-exact with the
   narrow NFD's own training data -- replaced the earlier from-scratch vectorised soft-cube
   renderer (a real deviation the coordinator caught). Donor channels 3/4 render with the SAME
   cv2-box rasteriser that produces channel 0 (`model/retrieval_nfd/render.py
   ::render_cube_boxes_batch`, reimplementing `_draw_particle_grid`'s box path), at both train
   and eval time.
3. **Eval-time retrieval now uses the TRUE particle state**, not pseudo-cubes, whenever the
   harness has one: `RetrievalRefPredictor.predict_step_particles(occ0, act, states0)`, dispatched
   from a small, targeted patch to `eval_extended.py::eval_occ_model` (`_predict_step_dispatch`,
   3 call sites: accuracy_1 chunk loop, slate-candidate loop, rollout loop). Pseudo-cube extraction
   from the occupancy image is now ONLY a fallback for rollout steps after the first (no true
   particles exist there -- exactly the design doc's own rollout spec).
4. Mandatory unit test's initial "exact reproduction" check was WRONG, not the rendering: `int()`
   pixel truncation (the same convention `_draw_particle_grid` uses -- verified `round()` gives
   ~100x MORE mismatches, i.e. truncation is the right convention to match) is sensitive to ~1e-6
   floating-point noise from the `world_to_push_frame -> push_frame_to_world` roundtrip (an
   analytically exact identity, not bit-identical in float) when a cube centre sits within ~1e-6px
   of an integer boundary. Measured 0.029% mismatched pixels (234/819200) over 200 real rows, ALL
   traced to ~1e-6px coordinate diffs -- not a structural axis/transpose bug (which would flip
   entire images). Test now asserts mismatched-pixel fraction < 0.1% (measured 0.029%), not exact
   zero, with the diagnosis in a code comment (`precompute.py::_unit_tests`).

Trained on the FULL DS-0008+DS-0010 corpus this time: 10910 train / 535 val / 476 test rows
(matches `nfd_3ch_narrow_l20.yaml`'s own split exactly, same `_filter_split` call). 20 epochs each
(narrow NFD used 60 on a smaller corpus at the time; 20 was chosen to fit the ~75 min budget this
round -- both models' val loss was still improving slowly at epoch 20, so more epochs would likely
help further). Both trained concurrently on the shared GPU (~59-62s/epoch each, ~21 min total).

## Results

| model | DS-0011 slateN_tough | DS-0009 slateN_tough | acc1 | rollout4 |
|---|---|---|---|---|
| narrow NFD (`nfd_3ch_narrow_l20`, reference) | n/a (not re-run) | 0.778 | 0.506 | 0.348 |
| retrieval_nfd_ref (main) | 0.725 | 0.686 | 0.480 | 0.341 |
| retrieval_nfd_random_donor (control i) | 0.653 | 0.597 | 0.464 | 0.326 |
| retrieval_nfd_ref_zeroed (control iii, test-time zeroed ref) | 0.751 | 0.715 | 0.490 | 0.337 |

(DS-0011 = `Genesis/data/narrow_l20_n20/val_pools`, pools-only, acc1/rollout n/a there by
construction -- same as before.) Full JSON: `offline_eval_extended.json` (DS-0009),
`offline_eval_extended_ds0011.json` (DS-0011); in-cache train/val/test loss curves in
`model/retrieval_nfd/runs/{retrieval_nfd_ref,retrieval_nfd_random_donor}/`.

All three models are now real, non-degenerate, comparable numbers (unlike the previous broken
run). None beats the narrow NFD yet (20 epochs vs its 60, and this model has 2 extra untrained-
from-scratch input channels to learn to use).

## The main finding: retrieval helps relative to noise, but hurts relative to no reference at all

- **retrieval_nfd_ref > retrieval_nfd_random_donor** on every metric, both datasets (e.g. DS-0009
  slateN_tough 0.686 vs 0.597; DS-0011 0.725 vs 0.653) -- confirms the retrieved reference DOES
  carry real, usable information the network learned to exploit (the "information" answer to the
  design doc's G6, not "noise" -- a random bank donor, same architecture/recipe, does measurably
  worse).
- **But retrieval_nfd_ref_zeroed (SAME checkpoint, reference forced to 0 at test time) beats
  retrieval_nfd_ref (real retrieval ON) on every metric, both datasets** (DS-0009 slateN_tough
  0.715 vs 0.686; DS-0011 0.751 vs 0.725). This is the opposite of what "the reference helps"
  would predict at face value, and is NOT explained by a representation mismatch this time (that
  bug is fixed -- channels 0-2 are bit-exact train/eval-consistent NFD featurisation, donor
  channels use the same rasteriser both places, and eval retrieval now uses the TRUE particle
  state).
- Plausible reading: the network learned to use SOME retrieved references well (better than
  random-donor noise), but the specific TOP-1-BY-FROZEN-KEY donor retrieved at eval time is, on
  net, still not good enough to beat just trusting the base occ0->prediction pathway -- consistent
  with the R1.5 diagnostic already on record for pure retrieval (best DS-0009 pure-retrieval
  `slateN ~0.51`, well below narrow NFD's `~0.77-0.80`; an oracle choosing among top-50 donors by
  TRUTH only reaches `~0.53`) -- i.e. the retrieval INDEX/KEY itself still leaves real headroom on
  the table (a metric problem, per the R1.5 table), and this 5-channel model has not yet closed
  that gap even though it demonstrably learned to extract positive signal from good donors when
  training saw them (top-3 resampled) more often than the single top-1 eval gets.
- Recommended next step (not done, time budget): retrain/eval with k=3 donors averaged/voted at
  test time too (matching training's own top-3 pool more closely, instead of a bare top-1), and/or
  more epochs (both models' val loss was still falling at epoch 20).

## Known, remaining deviations from the literal spec (time budget)

1. 20 epochs, not 40 (or the narrow NFD's 60) -- fits the ~75 min re-fix budget; val loss still
   improving at cutoff for both models.
2. 6th channel (retrieval clean prediction) not implemented -- 5-channel only.
3. Random-donor control's donor is a single FIXED bank row per training row (not resampled every
   epoch); the main model's IS resampled among its cached top-3 each `__getitem__` call, per spec.

## Status

| milestone | status |
|---|---|
| donor search vectorised + fixed (full 11921x11921, ~5.4 min) | done |
| channels 0-2/target bit-exact via real PileSweepData3Ch | done |
| donor channels via the matching cv2-box rasteriser (train AND eval) | done |
| mandatory unit tests (donor-path reproduces occ0 within FP-truncation tolerance; mirror round-trip) | done, PASSED |
| eval-time retrieval uses TRUE particles (particle-aware `eval_extended.py` patch) | done |
| main + random-donor control trained on FULL corpus (20 epochs each, concurrent) | done |
| zeroed-ref control (eval-only, same ckpt as main) | done |
| eval DS-0009 (1-step + rollout + slateN/slateN_tough) | done, see table (cross-validated by the experimenter agent) |
| eval DS-0011 (val_pools, slateN/slateN_tough only) | done, see table |
| multi-step rollout (DS-0013) | delegated to the experimenter agent (afe13fc23cf1d6301), running as of hand-off |
