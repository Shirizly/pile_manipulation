---
id: EXP-0009
title: >
  Post-fix pixel operator beats persistence and mean-delta (C-001 reverses);
  the contact-switch re-run (C-008) could not be completed in budget
tier: T1
mode: exploratory
date: 2026-09-05
hypothesis: H-A1
claim: >
  With PileSweepData._draw_particle_grid no longer transposed (commit
  aac084e3), the pixel operator's held-out swept-region rms on
  genesis_foresight_L040 falls from >=100% of persistence (the invalidated
  claim) to well below it, and mean-delta remains the harder baseline to beat;
  separately, a contact-switched operator (C-008) can be re-tested at a
  per-bin M/D comfortably above 1.
prediction:
  supports: "linear/ridge/nonneg swept-region rms drops from ~100-124% (pre-fix, EXP-0001/report) to a value >10 points below persistence's 100%, matching EXP-0001's own already-measured transposed-vs-stored swing (107.1% -> 57.9% at res 64/crop 0.5)"
  refutes: "the operator remains within ~1-2 points of 100% (persistence) on the now-fixed pipeline, i.e. the fix does not change pixel-level standing"
  discriminating: true
provenance:
  commit: aac084e3
  dirty: true                     # BACKFILLED 2026-09-05: `scripts/probes/exp0009_rerun.py`
                                  # did not exist at aac084e3; it was written in the same
                                  # session and first committed at 006004d0. So this sha
                                  # bounds the run from below only -- the analysis code
                                  # that actually ran is the 006004d0 version of that file.
  script_first_committed: 006004d0
  script: scripts/probes/exp0009_rerun.py
  data: ["configs/dataset/genesis_foresight_L040.yaml", "Genesis/data/foresight/L040/cube/n50/size0.005/**/*_data.pt"]
  code_path: "PileSweepData raster (now fixed), via fit_linear_foresight.py's own canonicalise/fit_operator/fit_operator_nonneg/predict_world/contact_score, imported directly rather than reimplemented"
  seed: 0
  split: "episode-level, 1/8 files held out (holdout_frac=0.2, 8 episodes -> floor gives 1 held out), seed 0"
  runtime: "res=8/crop=0.5 cell: ~75s CPU. res=64/crop=0.5 cell (the literal requested config) and the bins=2/3 cells at res=16/crop=0.25 did NOT complete -- see budget and Threats"
budget:
  declared: "60 min, 160k tokens"
  spent: "~100 min, ~140k tokens (background compute time not counted against wall-clock the same way, but wall-clock alone is well past declared)"
  outcome: exceeded
design:
  varied: {res_crop: ["8/0.5 (completed, this session)", "64/0.5 (from EXP-0001, prior session, same held_fixed config)"], bins: ["planned 2 and 3 at res=16/crop=0.25, NOT run"]}
  held_fixed: {view: mask, blur: 1.0, ridge_target: "toward identity", split_rule: identical, dataset: genesis_foresight_L040, holdout_frac: 0.2, split_seed: 0}
  baselines: [persistence, mean-delta, "identity (warp only)"]
  metric: "held-out rms over the swept region, as a percentage of persistence rms (100% = no better than predicting nothing moved); also expressed as explained variance vs persistence and vs mean-delta (1 - rms_model/rms_baseline)"
noise_floor: "~1 point, inherited from EXP-0001 (fold sd 0.004 rms against a persistence rms of ~0.11-0.16 in that record's own configuration); not independently re-measured here (single seed-0 split, no LORO)"
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend, swept-region-metric, episode-split, pixel-index-origin]
establishes: []
result: >
  C-001 REVERSES: linear/ridge/nonneg now beat persistence by a wide margin at
  every resolution checked -- 57.9% of persistence rms at res=64/crop=0.5
  (EXP-0001's own transposed-vs-stored row, now the ACTUAL on-disk behaviour
  post-fix) and 70.5% at res=8/crop=0.5 (this session, full pipeline,
  nonneg/ridge/ols agree to 3 decimal places). Both also beat mean-delta
  (explained_vs_meandelta +0.375 and +0.166 respectively). C-008 INCONCLUSIVE:
  the switched-operator re-run did not complete -- fit_operator_nonneg's FISTA
  is O(D^3)/iteration and the literal --res 64 --crop 0.5 command is
  infeasible on CPU within any T1 budget (measured 3.1 s/iter at D=4096 x
  4000 default iters = ~3.4 h); the cheap bins config (res=16/crop=0.25,
  M/D~2.9-4.4 per bin) was queued but never executed before the deadline.
verdict: inconclusive
downgrades: [incomplete-design, imprecision, provenance, untested-dependency]
grade: very-low
supersedes: []
superseded_by: [EXP-0018]
invalidated_by: null
---

## Why this test discriminates

C-001 claimed nothing beats persistence; EXP-0001 already showed the invalidating
mechanism (transpose) moves the operator by ~49 points at this exact
resolution/crop, so if the codebase fix (aac084e3) actually reproduces that
transposed-row behaviour, the operator must land near 54-58% of persistence,
not near 100%. Any config landing within noise of 100% would mean the fix did
not do what EXP-0001 predicted. For C-008, a per-bin M/D << 1 was the stated
reason the original switched fit could not be trusted; M/D >> 1 with the
switched operator still failing to beat a single operator would be a real
(not data-starved) refutation, but no cell reaching that M/D was completed
this session.

## What was actually run

1. **Cost pilot first (as the budget rules require).** Before committing to
   the literal `--res 64 --crop 0.5` sweep, a synthetic-size timing check
   (D=4096, dense G/C matrices matching the real op) measured FISTA at ~3.1
   s/iteration and one ridge `torch.linalg.solve` at ~6.7 s, under this
   session's CPU contention (two unrelated agent sessions were running
   concurrent CPU-bound fits throughout — see Unrelated findings). At
   `fit_operator_nonneg`'s default `max_iter=4000`, that is ~3.4 h for nonneg
   alone at res=64 — infeasible. This pilot measured cost only, not outcome,
   so it does not compromise the prediction above.
2. Wrote `scripts/probes/exp0009_rerun.py`, importing fit_linear_foresight.py's
   own `canonicalise`/`fit_operator`/`fit_operator_nonneg`/`predict_world`/
   `contact_score`/`actions_to_pixels` directly (same code path, same
   functions), adding only (a) a mean-delta baseline built from the identical
   warp/blend pipeline `predict_world` uses, and (b) a `--nonneg-max-iter` flag
   so the FISTA cap could be set below the infeasible default.
3. A first cell at `--res 8 --crop 0.5 --blur 1.0 --nonneg-max-iter 200` was
   run as a smoke test of the new script (D=64, trivially cheap: ~75 s total
   including a ~61 s data-load). **This cell's OUTCOME is being used as
   evidence, not discarded, so per the skill's pilot-disclosure rule this
   record is `mode: exploratory` rather than confirmatory** — the prediction
   above was written from EXP-0001's pre-existing numbers, not blind to this
   cell.
4. A background chain was launched for the actual target cells: `--res 64
   --crop 0.5` (the literal requested command, `--nonneg-max-iter 100`
   capped), then `--res 32 --crop 0.5` (full 4000-iter nonneg, D=1024 is 64x
   cheaper), then the C-008 bins=3 and bins=2 cells at `--res 16 --crop 0.25`
   (D=256, 4000-iter nonneg is trivial there). **None of the four finished
   before the deadline.** Python fully-buffers stdout when redirected to a
   file, so nothing was recoverable from the log even after ~13 minutes of
   the res=64 cell alone (which was still short of my own ~7 min estimate,
   consistent with the observed CPU contention from other sessions'
   concurrent jobs). The process was killed rather than left running past the
   write-up deadline.
5. In place of the missing res=64 cell, this record uses **EXP-0001's own
   published numbers**, which were measured at the identical held_fixed
   configuration (res=64, crop=0.5, blur=1.0, ridge=1.0 toward identity) one
   session before this bug was actually fixed in the codebase. EXP-0001's
   "registry, transposed" row is what the on-disk data now IS after aac084e3
   (a real transpose fix, not the manual/simulated one EXP-0001 applied by
   hand), so it is read as the res=64/crop=0.5 answer, with a `provenance`
   downgrade because it came from `scripts/probes/ab_occ.py`, a different
   script, in a different session.

## Numbers

**Swept-region rms, as % of persistence (100% = predicts nothing moved):**

| config | source | persistence | mean-delta | identity (warp only) | linear/ridge | nonneg |
|---|---|---|---|---|---|---|
| res=64, crop=0.5, blur=1.0 | EXP-0001 (registry, transposed = now-actual behaviour) | 100.0% | 87.1% | 98.3% | 57.9% (ridge=1) | not run in EXP-0001 |
| res=64, crop=0.5, blur=1.0 | EXP-0001 (re-rasterised, independent check) | 100.0% | 86.5% | 98.3% | 54.1% (ridge=1) | not run in EXP-0001 |
| res=8, crop=0.5, blur=1.0 | this session, full battery | 100.0% | 84.5% | 96.7% | 70.5% (ols=ridge1=nonneg, agree to 0.1pt) | 70.5% |
| res=64, crop=0.5, blur=1.0 (literal target) | this session | — | — | — | **DID NOT COMPLETE** | **DID NOT COMPLETE** |
| res=32, crop=0.5, blur=1.0 (2nd config) | this session | — | — | — | **DID NOT COMPLETE** | **DID NOT COMPLETE** |

**Same numbers as explained variance (1 - rms_model/rms_baseline; +ve = beats the baseline):**

| config | vs persistence: mean-delta | vs persistence: linear/nonneg | vs mean-delta: linear/nonneg |
|---|---|---|---|
| res=64/0.5 (EXP-0001, transposed) | +0.129 | **+0.421** | +0.335 |
| res=64/0.5 (EXP-0001, re-rasterised) | +0.135 | **+0.459** | +0.375 |
| res=8/0.5 (this session, nonneg) | +0.155 | **+0.295** | +0.166 |

Every completed cell, at both a coarse (res=8) and the paper-matched (res=64,
via EXP-0001) resolution, shows the same sign and an effect 30-46 points above
the ~1-point noise floor: **the operator now beats both persistence and
mean-delta.** This is the mirror image of the invalidated claim.

**C-008 (contact-switching): no completed cell.** Planned config and its
estimated M/D, computed from this session's own measured split (2240 train /
320 test, 8 runs total -- this split is res/crop-independent since
`split_by_episode` only depends on episode structure):

| bins | res | D=res² | train M per bin (2240/bins) | **M/D per bin (estimated, not measured)** |
|---|---|---|---|---|
| 3 | 16 | 256 | ~747 | ~2.92 |
| 2 | 16 | 256 | ~1120 | ~4.38 |

Both would have been "comfortably above 1" (the report's original crop=0.25
test had M/D=8.3 on a 3x larger, 7680-transition dataset not available here;
2.9-4.4 is smaller but still overdetermined per row). **This was never run.**

## What would change the verdict

For C-001: the literal res=64/crop=0.5 full battery (ols, ridge sweep, nonneg)
using `scripts/probes/exp0009_rerun.py` directly (already written, already
smoke-tested at res=8). Cost, measured: ols+5 ridge solves ~50s; nonneg is the
blocker -- at a capped 300 iterations (~15 min at the ~3.1s/iter measured under
contention, likely 2-5x faster without it) it would very likely still show the
same reversal, since FISTA starts from A=identity and the res=8 cell already
shows ols/ridge/nonneg agreeing to 0.1 point, i.e. the constraint is not
binding here. Running it to completion (even capped) would remove the
`provenance` downgrade for the headline resolution.

For C-008: run the queued bins=2 and bins=3 cells at res=16/crop=0.25
(D=256; 4000-iter nonneg there costs a few seconds per bin fit, trivial --
the entire blocker was that Config C/D in the background chain queued behind
the much more expensive res=64/res=32 cells and never got CPU time). Running
C-008's cells FIRST next time (they are ~1000x cheaper than the res=64 cell)
would answer that claim within minutes.

## Threats

- `incomplete-design`: the literal `--res 64 --crop 0.5` command from the task,
  the `--res 32 --crop 0.5` second config, and both C-008 bins configs were
  planned but not completed. C-001's verdict rests on EXP-0001's pre-existing
  res=64 numbers plus this session's own res=8 cell, not on a completed
  same-session res=64 run. C-008 has **no completed cell at all** — that claim
  is not "reversed" or "stays refuted", it is simply untested here.
- `imprecision`: single seed-0 split, no leave-one-run-out, for both the res=8
  cell and (inherited) EXP-0001's res=64 numbers.
- `provenance`: the res=64/crop=0.5 numbers come from `scripts/probes/ab_occ.py`
  (EXP-0001, a prior session), not from this record's own script. The two
  scripts fit the same estimator family (`ridge toward identity`) on the same
  held-out split logic, but are not the same file.
- `untested-dependency`: `pixel-index-origin` is broken (~1px, EXP-0001);
  `swept-region-metric` and `episode-split` are unchecked. None of the three
  differ between the compared rows, so none can produce the observed 30-46
  point effect, but all three are cited honestly.
- Considered and dismissed: **the res=8 cell is too coarse to trust alone.**
  It is not trusted alone — it is reported *because* it independently agrees
  in direction and rough magnitude with EXP-0001's res=64 numbers, which used
  a different script and a different (simulated) route to the same fixed
  grid.

## Unrelated findings

- Three other agent sessions were running CPU-bound probes
  (`scripts/probes/deposit_profile.py`, `scripts/probes/shrinkage_vs_control.py`)
  concurrently on this machine throughout this session, at times consuming
  600-700% CPU each on a 20-core box. This materially slowed every fit in this
  record; the 3.1 s/iter FISTA measurement and the res=64 cell's failure to
  finish in ~13 minutes (vs. a ~7 minute estimate) are both partly attributable
  to this contention, not only to D=4096's inherent cost.
- `fit_operator_nonneg`'s default `max_iter=4000` has no CLI-exposed override
  in `fit_linear_foresight.py`, and its per-iteration cost is O(D^3)
  (`Z @ G` where both are `D x D`). At `--res 64` (D=4096) this makes the
  script's own documented usage example infeasible on CPU (~3.4h) with no
  warning printed anywhere in the script or its `--help`. Existing records
  that report `nonneg` at res=64 with `blur` variations (`linear_foresight_report.md`
  §1-2) presumably ran on a machine without this session's contention and/or
  the FISTA implementation may have changed cost characteristics since
  (`fit_operator_nonneg`'s docstring notes it was rewritten from a slower
  per-row QP approach) — not verified here, logged for the user to check.
- Python's stdout is fully buffered (not line-buffered) when redirected to a
  file rather than a TTY; a background job piped through `| tail -N` or `>
  file` will show **no output at all** until the process exits, not
  incremental progress. This cost real debugging time in this session and
  would trip up any future agent monitoring a long CPU job the same way;
  `python -u` or `PYTHONUNBUFFERED=1` avoids it.


## Superseded, 2026-09-06

**This record is superseded by EXP-0018.** Its measurements stand as taken; do
not cite its conclusions. Reason: its only in-session completed cell was res=8, a degenerate resolution, and its headline was borrowed from EXP-0001. EXP-0018 closed that gap: the literal res=64/crop=0.5 cell measured in-session at 57.8%, matching the borrowed 57.9%, plus 8-fold LORO.

Kept rather than deleted because the register's audit trail depends on being
able to see what was believed and why it changed.
