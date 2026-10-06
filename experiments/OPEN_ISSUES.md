# Open issues — code defects and evidence debt awaiting an owner

**What this file is for.** An agent that finds a real problem mid-task usually
cannot fix it there and then: the fix is out of scope, out of budget, or
belongs to whoever owns the affected records. Historically that finding either
got buried in a run report nobody re-reads, or was silently re-discovered
weeks later. Record it here instead, in one entry, and move on with your task.

**What belongs here**

- a code defect too large to fix inside the current task
- a test or experiment that must be RE-RUN because an input changed
- record-keeping debt: a record asserting something since disproved, a
  register row needing revision, a citation that no longer resolves
- a measurement that is known wrong but whose blast radius is not yet mapped

**What does NOT belong here**

- a claim about the world — that is `REGISTER.md`
- a property results rest on — that is `INVARIANTS.md` (open an entry here too
  if the tag is `broken` and nobody is fixing it)
- a scratch probe — that is `TEMP_LOG.md`
- a bug you are about to fix in the same change: just fix it

**How to use it.** Append an entry with the next free `ISS-###`. Keep it short
enough to read in fifteen seconds and specific enough to act on without
re-deriving the finding: say what is wrong, how it is known (measured, or
read in the code), what it invalidates, and what closing it would take. When
you close one, set `status: closed` with the commit or record that closed it —
do not delete it, the history is the point. `blocked-by` names another entry
that must land first.

Status vocabulary: `open` · `in-progress` · `blocked` · `closed` · `wontfix`
(with a reason).

---

## ISS-001 — EXP-0019 and EXP-0020 assert a rank-collapse finding that is refuted

**status:** open · **found:** 2026-09-17 · **severity:** high (misleading records)

Both records state that LeJEPA training collapses the encoder latent from
effective rank 44.6 to ~7.5 of 256 dims, and EXP-0019 attributes it to SIGReg
being applied to the projection rather than to `z`.

**The rank number was measured on an unrepresentative slice.** The training
monitor (`EXP-0019-*/code/train_encoder.py::latent_stats`) encodes `pts[:8192]`
— the FIRST 8192 rows — and `data.py::load_states` concatenates files in order
with no shuffle, so that is roughly the first 8 files of 273. Recomputed on the
full 98,304-state corpus with the record's own definition:

| encoder | rank on monitor slice | rank on full corpus |
|---|---|---|
| random (untrained) | 44.6 | 40.5 |
| SIGReg on projection only | 7.5 | 41.7 |
| SIGReg on projection + latent | 7.3 | 18.6 |

So the first encoder did not collapse at all (41.7 vs random's 40.5); it
contracted LOCALLY, within a narrow band of similar states. EXP-0019's causal
claim is refuted, and EXP-0020's conclusion that the fix "failed" reads the
same misleading slice. What EXP-0020's fix DID do stands on other evidence:
latent R² +0.033/+0.060 → +0.293/+0.412 and the train-test gap 0.301 → 0.052.

**To close:** amend both records (verdict and body), and fix `latent_stats` to
sample rather than take a prefix. Neither is cited in `REGISTER.md` yet, so no
register row changes.

---

## ISS-002 — Goal libraries and EXP-0018 must be regenerated after the axis-convention fix

**status:** in-progress · **found:** 2026-09-17 · **blocked-by:** none (28271c09 landed)

`28271c09` fixed the goal-mask axis convention. Everything built against the
pre-fix convention is stale:

- `experiments/temp/goal-states/dataset.pt` (500 goals) and `dataset_v2.pt`
  (2000 goals) — mask/configuration pairs baked at generation time
- EXP-0018's cache and all 51 persisted readouts, which consumed `dataset_v2`

EXP-0018 is specifically exposed: goal FEATURES are descriptors of rasterised
configurations while TARGETS are computed from the raw mask arrays
(`stage2_build_cache.py`), so feature and label referred to mirrored goals.

**Mitigating measurement** (do not over-correct on this): transposing an image
acts on the 87-dim descriptor vector as a near-fixed linear reindexing —
held-out R² of that map is 1.0000 on mass/com/moments2 and ~0.80 on the two
Fourier blocks, 0.988 overall. A linear-or-better readout absorbs the geometric
part exactly, so descriptor-space results were largely insulated BY
CONSTRUCTION. **This does not extend to image-space models** — a CNN has no
mechanism to absorb a reindexing for free.

**Progress 2026-09-17.** Both libraries ARE regenerated and verified: goal
material now falls 95.7% (500-goal) / 95.4% (2000-goal) inside its own mask,
against 49.2% for the pre-fix copy, which is kept at
`experiments/temp/goal-states/pre-28271c09/` for before/after work. Families in
v2: 810 letter, 800 rectangle, 250 ellipse, 140 two-blob.

**Still open: EXP-0018's cache and its 51 persisted readouts are stale** — they
were built against the pre-fix library and must be rebuilt and re-run.

**Revision to the earlier worry about letters.** The concern was that letters
take the thin-glyph fallback (`_rejection_sample`), which had its own
independent transpose also fixed in `28271c09`, so letter cells would carry two
entangled causes. Measured on the regenerated v2: of 6000 configurations only
**4** used `rejection_sampling`. The rest are the grid path (4425 `grid`, 1571
`grid_dilated*` — dilation is still the grid path). So the second defect touches
0.07% of configurations and letters need NOT be excluded from the mechanism
test; carry the family field anyway and note the 4.

---

## ISS-009 — Promoted experiment scripts can silently write OUTSIDE the repo

**status:** closed (this instance) · **found:** 2026-09-17 · **severity:** medium

`EXP-0015-*/code/stage3a_generate_goals.py` computed
`REPO = Path(__file__).resolve().parents[4]`. That was correct when the script
lived in `experiments/temp/<slug>/code/`, but the file was later PROMOTED to
`experiments/EXP-0015-*/code/` — one level shallower — so `parents[4]` began
resolving to `/home/alon/Code`, the repo's PARENT. Re-running it created a
whole stray `/home/alon/Code/experiments/temp/goal-states/` tree and wrote the
regenerated library there, while the in-repo copy sat untouched and stale. It
failed silently: the run printed "saved ..." and exited 0.

Fixed to `parents[3]` with a comment, stray tree removed, library regenerated
in place. **The general hazard is open:** promotion from `experiments/temp/`
into an `EXP-####` directory changes path depth, and any script using a
hard-coded `parents[N]` breaks without erroring. Worth a grep across promoted
code (`grep -rn "parents\[" experiments/*/code/`) and, better, a shared
repo-root helper that does not depend on nesting depth.

---

## ISS-003 — Evidence cleanup after the axis-convention fix is unowned

**status:** open · **found:** 2026-09-17 · **severity:** high

`goal-mask-axis-convention-row-y-col-x` went `unchecked` → `broken` → `fixed`
in `28271c09`. Every result scored under an ASYMMETRIC goal before that commit
is affected. `corner`, `center`, `ind-corner`, `distclip-corner-r*`,
`ind-square8`, `dist-square8` are transpose-invariant and were verified
bit-identical across the fix; `stripe` changes; `quadrant_mask` 1 and 2 swap.

**The tag's own `depends_on` citations understate the blast radius.** Records
using `ring_O` / `T` / `random_quadrant` / `letter_T` cells without citing the
tag include EXP-0003, EXP-0011, EXP-0013 and EXP-0017. Find the affected set by
grepping for asymmetric-goal cells, NOT by grepping for the tag.

Claims plausibly touched: C-010, C-016, C-019, C-021. C-021 additionally
narrows for an unrelated reason — under grouped CV folds its two-to-three
orders of magnitude rests on `lyapunov` alone, carried by a single held-out
goal, with both mass functions unresolved at 6 goals.

**To close:** one session maps the set and revises `REGISTER.md`. Deliberately
left unowned — two sessions editing the ledger concurrently is the one thing
that would make this worse.

---

## ISS-004 — Random validation splits keep reappearing; there is no shared guard

**status:** open · **found:** 2026-09-15..17 · **severity:** high (recurring)

Four instances of one root cause, each found by hand:

1. `EXP-0015-*/code/{stage0_upper_bound,stage3b_regression}.py::fit_eval_cv` —
   inner folds `np.array_split(rng_cv.permutation(n), 3)` while the outer split
   is slate-aware. Fixed in those files.
2. `EXP-0017-*/code/value_readout.py` — selected λ with `RidgeCV`, whose
   default is leave-one-out GCV, i.e. row-random. `GroupRidgeCV` added, but
   **the default is still row-random**; callers must pass `cv_group_by`.
3. Same file, MLP path — `MLPRegressor(early_stopping=True)` holds out a random
   10% of (state, goal) pairs, so rows sharing a state straddle the split.
   NOT fixed.
4. `EXP-0019-*` dynamics fit — row-random inner validation scored +0.359 where
   the true held-out score was +0.060, and picked λ=1 where 10–100 was best.

All four bias PESSIMISTIC (under-regularisation → overfitting → worse held-out
scores), so nothing has been oversold — but every negative result from an
affected fit is suspect.

**To close:** a shared splitter in the owning module that refuses to split
without an explicit grouping key, and migrate the call sites. Prose rules have
not worked; this needs a mechanism.

---

## ISS-005 — `assert_no_penetration` rejects 91–99% of real simulated states

**status:** open · **found:** 2026-09-17 · **severity:** medium

`Baselines/common/goal_configs.py::assert_no_penetration` and `_penetrates`
test an axis-aligned keep-out box of half-width `cube_size*sqrt(2)` per axis.
That refuses legal diagonal contact by a factor of √2 and is ~2× stricter than
true aligned contact. Measured against DS-0002 real states: 274/300 (n=20),
298/300 (n=50), 297/300 (n=100) rejected. These states came out of the physics
engine and are valid by construction.

Consequence: the generator could not express "contact" at all, which is why no
compaction scheme could reach real packing density until an exact test existed.

An exact separating-axis test now lives at
`Baselines/common/cube_overlap.py::state_is_legal`. The old functions were
deliberately NOT changed — goal generation and C-021 rest on their current
behaviour.

**To close:** decide whether goal-configuration generation migrates to the
exact test, with a before/after on the generated configurations. Belongs with
whoever owns the ISS-002 regeneration.

---

## ISS-006 — `particles_to_occupancy` loops over the batch in Python

**status:** open · **found:** 2026-09-17 · **severity:** low (performance)

`transforms/functional.py::particles_to_occupancy` iterates `for b in
range(B)`. A bit-exact vectorised replacement exists at
`experiments/EXP-0019-*/code/raster.py::rasterise` — verified zero mismatched
cells, 80 ms → 1 ms per batch of 64. It is used project-wide, so the speedup is
broad, but upstreaming it touches every caller's numerics by construction
(identically) and deserves its own change.

**To close:** upstream, with the equivalence assertion as a pytest.

---

## ISS-007 — `check_register.py` emits an unclearable dirty-tree warning

**status:** open · **found:** 2026-09-17 · **severity:** low

`scripts/check_register.py:136` warns on a dirty tree unless `downgrades`
contains `"dirty"` — but `"dirty"` is not in the downgrade-domain vocabulary
that same script enforces at line 33. So the warning cannot be cleared by any
valid record, and 15+ records carry it permanently. Warnings that can never be
cleared train readers to ignore all warnings.

**To close:** either add the domain or drop the escape clause; the two lines
must agree.

---

## ISS-008 — Standing `broken` invariants with no owner

**status:** open · **severity:** medium

Three tags in `INVARIANTS.md` are `broken` with a known workaround that was
never upstreamed, so each new caller rediscovers them:

- `dmdc-apply-operators-memory-safe` — `dmdc_baseline.apply_operators`
  fancy-indexes `A[bins]` into an `[N,D,D]` tensor, ~138 GB at D=631. A
  loop-over-bins workaround lives in `experiments/temp/dmdc-lenbins/`.
- `run-probe-ledger-path-matches-doc` — `scripts/run_probe.py` writes to
  `runs/COMMANDS.jsonl`, not the documented `experiments/COMMANDS.jsonl`.
- `eval-baseline-scorer-batch-on-requested-device` —
  `Baselines/common/eval_baseline.py::_predictor_batch` builds `PredictorBatch`
  on CPU regardless of the requested device, so some models time on CPU while
  others force CUDA.

**To close:** upstream each workaround, write the xfail test first per
`register-validator`, and flip the tags to `fixed`.

## nfd_train_3ch_randlen.yaml's stated gradient-step count is 8x too low

`Baselines/NFD/configs/nfd_train_3ch_randlen.yaml`'s header justifies its
30-epoch setting with "~2784 steps/epoch" and "30 epochs gives ~83,500
gradient steps, close to the in-distribution run's 100 x ~684 = 68,400 --
roughly matched by total updates". Those are the **no-augmentation** figures
(89,081 / 32). The config sets `augmentation: true`, and
`training/trainer.py` then uses `loader_bs = batch_size // 8 = 4`, so the
real figures are **22,270 steps/epoch and 668,100 gradient steps** — 8x
larger. The same wrong arithmetic appears in `nfd_train_3ch.yaml`'s sibling
reasoning and was copied into EXP-0022's first four configs before being
caught.

The comparison EXP-0001 rests on is NOT affected: the baseline and every
arm compared against it run the same augmentation and so the same
accounting. What is wrong is the stated RATIONALE for the epoch count, and
any future run sized by trusting that comment. Found 2026-09-23 while
sizing EXP-0022's flip-only re-run, which is matched against the corrected
668,100 figure.

## `Baselines/LinearForesight/runs/operators_res64.pt` missing from disk, provenance not re-derived

`eval_report.py`'s `linear_switched_res64`/`linear_single_res64` MODELS
entries point at `Baselines/LinearForesight/runs/operators_res64.pt`, which
was **absent** when EXP-0022's RUN-0011 (frame-scoring) tried to load it --
only `operators_res32.pt` and `operators_res64_accuracy.json` (its scoring
output, with no matching weight file) were present. The file was never
git-tracked either (only its `_accuracy.json` sibling was ever committed, per
`git log --all --diff-filter=A`), so this isn't a checkout/branch artefact --
it is genuinely missing from the working tree it should have been fit into.

A same-named, same-shaped file (keys `operators, bin_edges, counts,
single_operator, mean_delta, res=64, crop, constraint, ridge, n_bins,
train_cfg=configs/dataset/genesis_overnight_randlen_train_all.yaml,
provenance`) was found at
`experiments/temp/res32-vs-64/strays/operators_res64.pt` and copied into
`Baselines/LinearForesight/runs/operators_res64.pt` to unblock RUN-0011 --
**its provenance was NOT independently re-verified** (which run/script
produced it, whether it was fit with the exact ridge/config
`fit_switched.py` currently defaults to). It looks right (matching res,
matching train_cfg, all expected keys, and RUN-0011's `linear_switched_res64`
numbers sit in a sane range next to `linear_switched_res32`), but "looks
right" is not the same as "confirmed identical to what a fresh
`fit_switched.py` run would produce".

**To close:** re-run `fit_switched.py` at res=64 and diff the resulting
`operators_res64.pt` against the restored one (or at minimum re-score it and
confirm the accuracy matches `operators_res64_accuracy.json`, which predates
the file's disappearance and should be the ground truth for what a correct
fit looks like). If they match, delete the stray copy; if they don't,
`operators_res64_accuracy.json` and everything scored against the restored
file needs re-examination. Found 2026-09-23 during RUN-0011.

## ISS-010 — Pile-aware action sampling routinely places the tool ON a cube at touchdown

**status:** closed (sampler fix) · **found:** 2026-09-28 · **closed:** 2026-09-28 ·
**severity:** high (affected the retrieval bank, DS-0008/9/11/12/13, and every model/metric fit
or scored on them)

Measured directly: `experiments/EXP-0059-retrieval-transition-model/code/audit_tool_placement.py`
tests every recorded transition's touchdown pose (`p_start`, blade yaw `angles`) against every
cube's actual rotated-square footprint with an exact SAT test
(`Baselines/common/cube_overlap.py::overlaps_rect_pairs`, a new rectangle-vs-rectangle
generalization of the existing same-size-square `overlaps_pairs`). Full numbers:
`experiments/EXP-0059-*/results/tool_placement_audit.json`; per-row flags saved next to every
source file as `<name>_legality.pt` (`illegal_0mm`, `illegal_1mm_margin`, per-row cube counts),
originals untouched.

**Illegal-at-touchdown rate (valid rows, exact SAT overlap, blade 40x2mm vs 5mm cubes):**

| dataset | frac illegal (0mm) | frac illegal (1mm margin) |
|---|---|---|
| DS-0008 train | 0.443 | 0.689 |
| DS-0009 test_chains | 0.462 | 0.692 |
| DS-0009 test_pools | 0.561 | 0.767 |
| DS-0010 extra_18_22 | 0.0002 | 0.334 |
| DS-0011 val_pools | 0.478 | 0.736 |
| DS-0012 reservoir | 0.458 | 0.702 |
| DS-0013 seqpools | 0.517 | 0.737 |

Roughly HALF of every `pile_aware`-sampled dataset's rows have the tool overlapping a cube at
touchdown. Consistent patterns: clump starts are worse than scatter (e.g. DS-0008 0.530 vs
0.356); illegal rate falls slightly across chain steps within an episode (DS-0008 step 0 0.490
-> step 7 0.408 — earlier pushes spread the pile out); DS-0009 pools (0.561) are worse than its
chains (0.462) — pools start from a freshly-spawned, denser state with no prior pushes to
de-clump it. DS-0010 is the outlier: essentially zero EXACT overlaps but a third of rows sit
within 1mm of one, consistent with a DIFFERENT, older sampler (see below).

**Root cause (read, not just measured):** `Genesis/sandbox_manipulation_clean.py::
_apply_pile_aware_starts` (+ `action_sampling.py::pile_contact_starts`) computes a genuinely
collision-free touchdown ("one particle-width behind the pile's near face, in the swath").
`_pile_aware_stops` (same file, ~line 1953-1966) THEN unconditionally clamps that start into the
yaw-dependent tray/blade-footprint sampling box (`action_sampling.sampling_box` — a pure
wall-margin bound with ZERO pile-occupancy awareness), because, in its own comment, "the pile
spreads well past its spawn extent" (particle radius reaches p95 34.6mm / max 54mm against a
blade box of only 23.5-42.5mm half-extent) — **its own measurement, quoted verbatim in that
comment: "35.8% of starts out of box and 3.3% of pushes travelling ~0mm"**. The clamp is
explicitly justified as "starting just inside the pile rather than just behind it, which still
sweeps material" — the author flagged the trade-off but not that "inside the pile" can mean
directly on top of an individual cube's footprint. This is the dominant mechanism for DS-0008/9/
11/12/13 (all `pile_aware=True`, confirmed per `chain_collection.py` commands in each
DATASET.md). DS-0010 comes from `overnight_randlen_train/{mixed,piled}_n20` + `Sean/n20`
(`datasets/DS-0010-*/extract.py`) — an older pipeline; its near-zero exact-overlap / high
1mm-margin rate instead matches `placement_sampling.py::free_placements`'s DEFAULT
`clearance=0.0` (a placement-aware, not pile-aware, sampler that draws touchdowns flush against
a cube, not overlapping one) — plausible for the "mixed"/"placement_aware" portion of that
older corpus, not independently re-verified per source file.

**What this invalidates:** any physics recorded at these ~45-55% of rows is not a clean push —
the first simulation step resolves a spawned interpenetration as a violent, physically
meaningless ejection, not a "the blade pushed this cube" transition. This affects the retrieval
bank (`model/retrieval/bank.py` loads DS-0008+DS-0010 unfiltered), `eval_retrieval.py`'s
DS-0009 numbers, and by extension every EXP-0059 accuracy_1/slateN/rollout number recorded so
far (not necessarily invalidated — the corrupted rows are a large but not overwhelming minority
of the population mean — but not clean either). Likely affects EXP-0053's narrow NFD training/
eval too (same DS-0008/9 source), not yet audited there.

**To close (original plan):** EXP-0059 built a curated bank (DS-0014) that excludes
`illegal_0mm` rows by default and restricts retrieval to a geometric interaction set, and
re-scores affected models on legal-only subsets for a fair comparison (see
`experiments/EXP-0059-*/LOG.md`, post-fix rerun).

**CLOSED 2026-09-28 -- the sampler itself is fixed, not just worked around.**
`Genesis/sandbox_manipulation_clean.py::_pile_aware_stops`'s clamp (item (1) above) is replaced
by `_pile_aware_action_legal` -> `Genesis/action_sampling.py::pile_aware_action_batch`: the
box-clamped touchdown is checked with an exact SAT test against every cube and REDRAWN (a fresh
heading, never a shortened/lengthened push) whenever it is illegal, up to `max_redraws=200`.
Also added the same task: a `start_gap_range` sampler config key that SAMPLES the touchdown gap
(blade front face to the first contacted cube's near face, along the push axis) uniformly in a
requested window instead of a fixed one-particle-width gap, retargeting (not accepting
out-of-window) until `max_redraws` is spent, then flagging `gap_out_of_window` as a last resort.
Two subtle bugs were found and fixed during this same task, both worth remembering for the next
sampler change in this area: (a) `pile_contact_starts`'s `clearance` is CENTRE-based but the
requested gap is FACE-based -- converting between them needs `+ blade_half_width + cube_half`,
easy to get wrong by ~3.5mm; (b) the box clamp moves the touchdown in WORLD (x,y), which can
silently invalidate the along-push-axis gap a draw was built with even when the clamped point is
still legal -- redrawing on illegality alone is not enough once gap PRECISION also matters, not
just legality.

Fresh corpora collected with the fixed sampler, verified at full scale (not just a smoke test):
**DS-0015** (train, replaces DS-0008+DS-0010, 0/12032 illegal, 0/12032 gap_out_of_window),
**DS-0016** (test chains+pools, replaces DS-0009, 0/1024 and 0/2048 illegal, 0 gap_out_of_window),
**DS-0017** (val pools, replaces DS-0011, collection status: see that record). The `Genesis/
training/dataset.py::PileSweepData(exclude_flagged=True)` training loader independently
confirmed the same drop counts. DS-0008/9/10/11/12/13's own payloads are UNCHANGED and their
DATASET.md files now say so; DS-0012/0013 were not recollected (out of this task's scope) and
still carry the original defect if used directly.

**Still open, not part of this closure:** (2) audit EXP-0053's narrow NFD train/eval numbers
against the pre-fix legality flags; (3) verify DS-0010's actual per-source-file sampler flags
rather than inferring them from the measured rate; (4) DS-0012 (reservoir) was left uncollected
under the fixed sampler -- recollect if it is needed clean. (DS-0013 is closed: **DS-0018**
recollects it with the fixed sampler, 0/6144 illegal, 2042/2048 sequences (99.7%) survive
whole-sequence archiving of the few remaining `valid==False` rows -- see
`datasets/DS-0018-narrow-l20-seqpools-clean/DATASET.md`.)

## ISS-011 — `train_nfd.py --override output.log_dir=...` silently resumes from the ORIGINAL log_dir's checkpoint

Found 2026-09-28 (EXP-0059, launching NFD v2 seeds 1/2). `Trainer.from_config(resume=True)`
runs `_try_resume()` against the YAML's own `output.log_dir` BEFORE the CLI override is
applied, so a "fresh" run pointed at a new log_dir resumes from whatever checkpoint sits in the
config's original directory (here: seed 0's `unet_epoch_60.pth`). Caught from the log line
"Resumed weights from .../nfd_3ch_narrow_l20_v2/unet_epoch_60.pth"; both runs were killed and
relaunched with `--no-resume` before anything was scored. **Workaround:** always pass
`--no-resume` when overriding `output.log_dir`, or give each run its own config. **Fix owed:**
apply overrides before resume resolution in `Baselines/NFD/train_nfd.py` / `training/trainer.py`.
**Possible exposure:** any earlier run that used `--override output.log_dir` without
`--no-resume` while the config's own log_dir held checkpoints — not audited.

**Audit 2026-10-04:** the code fix is in place (`Baselines/NFD/train_nfd.py` applies overrides before the
resume lookup). No `train_nfd` start in `COMMANDS.jsonl` used `--override output.log_dir`; every
'Resumed' line in saved run logs (EXP-0061 seeds 0-2) resumes the run's OWN `last_state.pt`. No exposure
found -- limited to runs whose logs were kept. **Status: closed (fixed, audited).**

## ISS-012 — slateN capture explodes on near-tie (pool, goal) cells; DS-0017 val has three

Found 2026-10-03 (EXP-0063). Capture = (mean(vt) - vt[pick]) / (mean(vt) - min(vt)) is
unbounded below, and EXP-0059 `eval_extended.py` skips a cell only when the denominator is
< 1e-9. DS-0017 `val_pools_v2` has three `quadrant_0` cells (pools 17, 22, 27) whose true dv
spread is 0-1% of the split median; one wrong pick there scores -1.3 to -40 and moves a model's
32-pool val slateN_tough by up to ~0.16 (EXP-0059's seed-2 val 0.567 -> 0.713 with them dropped;
EXP-0063 soft_s1 0.639 -> 0.798). DS-0016 test pools have none. **Fix owed:** a relative
degeneracy threshold (EXP-0063 `summarize.py::drop_degenerate` uses 5% of the split median),
or per-cell clipping, decided once and applied to every harness. **Exposure:** every
val_pools_v2 slateN/slateN_tough number (EXP-0059 val selection, EXP-0063 val column).

## ISS-013 — DS-0001 / DS-0006 (and pre-fix planner candidate banks) carry the ISS-010 illegal-touchdown defect

**status:** open · **found:** 2026-10-03 (EXP-0065, prompted by the 2026-10-03 experiment-summary audit) ·
**severity:** high (DS-0006 is the main offline benchmark corpus and the start-state source of every closed-loop record)

**Task definition (user, 2026-10-03):** in Genesis (rigid cubes) touchdown-illegal pushes are NOT part of the
task (they would mostly fail in reality), so everything below is a defect to fix and re-run. In FleX
(flexible carrots) they are allowed for now; FleX data/tests may be re-run later if that changes.

ISS-010 audited only DS-0008..0013. The same pre-fix pile-aware path also produced the candidate
pushes of DS-0001 and DS-0006 (`Genesis/binned_slate_collection.py`). EXP-0065 measured (exact SAT,
same code as ISS-010): **DS-0001 46.1 % (1 mm: 59.1 %), DS-0006 54.3 % (1 mm: 79.8 %)** of candidate
pushes put the blade on a cube at touchdown; per-slate 0.21-0.63 / 0.31-0.76.

**Measured on one table (EXP-0065 RUN-0002, EXP-0030's DS-0006 cache):** the true best push is an
illegal touchdown in 72 % (lyapunov) / 54 % (mass) of state x goal cells, and dropping illegal
candidates raises slateN beyond the pool-size effect UNEVENLY across models (linear_switched_hard
+0.113 lyapunov, NFDs +0.01..+0.07), reordering them (Kendall 0.83). So DS-0006 slateN partly
measures how well a model predicts illegal-touchdown outcomes.

**Measured for executed closed-loop pushes (EXP-0065 RUN-0003, lower bound from cube-disk bounds):**
planners EXECUTE illegal touchdowns in 16-53 % of pushes (EXP-0051 0.17, EXP-0052 0.36, EXP-0054 0.53,
EXP-0055 0.16-0.24), including the perfect-model simulator CEM (EXP-0057, 0.23-0.24) -- the simulator
lets the blade land on cubes and the optimisers exploit it. The rate is MODEL-dependent (EXP-0054:
worldframe NFD 0.69, the closed-loop winner, vs 0.46-0.51; EXP-0051: NFD ~2x linear), so closed-loop
model comparisons partly compare how much each model's objective rewards illegal pushes. Fix: `Genesis/action_sampling.py::legalize_pushes`
(2026-10-03; EXP-0043 driver `--legalize`) projects planned pushes to legal touchdowns before execution;
still owed: re-run the headline closed-loop cells with it (queued after the EXP-0064 overnight GPU queue).

**Exposure (the rest is unmeasured):**
1. Offline ranking on DS-0006 / DS-0001 pools: roughly half of every pool is illegal pushes whose
   simulated outcome (ejection / violent displacement) is both off-distribution for the models and
   often a large true dv, so slateN's denominator and the "best" action can be illegal pushes.
   Records: EXP-0011..0014, 0016, 0023, 0026 A3, 0029 (DS-0001); EXP-0030, 0036, 0037, 0038, 0040,
   0041, 0042, 0046 (redundancy), 0048, 0049 (DS-0006).
2. Closed loop (EXP-0032, 0039-0057): start states from DS-0006 are legal settled states, but planner
   candidates / initialisations were drawn with `generate_action_samples(pile_aware=True)` before
   2026-09-28. Executed (GD/CEM-refined) pushes have not been audited for touchdown legality; the
   flat rank-planner results (EXP-0039/0042/0045) and the sampler ranking of EXP-0049 (C-054) are the
   most exposed.
3. Training corpora: overnight_randlen-derived DS-0010 audited at 0.0002 (0 mm) / 0.33 (1 mm) in ISS-010,
   so the broad models' training data is mostly legal at 0 mm; DS-0007 (Sean) audited 2026-10-03 (EXP-0065): <= 0.2 % illegal in every n20/n50 shard except scattered_n50 1 % and scattered_n100 9 % -- EXP-0035 / EXP-0036's DS-0007 results are effectively clean.

**To close:** (a) re-score the key DS-0006 offline tables on legal-only candidates (filter with
`_row_illegal`) and report rank changes; (b) audit executed closed-loop pushes from the recorded
episode actions (EXP-0044/0051/0054/0055 artifacts) with the same SAT test; (c) recollect a legal
DS-0006 successor with `pile_aware_action_batch` if (a) moves rankings.

## ISS-014 — `legalize_pushes` ok=False pushes are executed unchanged by the closed-loop driver

**status:** in-progress (fix under EXP-0066, another agent, 2026-10-04) · **found:** 2026-10-04 (EXP-0039 extended analysis) ·
**severity:** medium (small rates; model orders unchanged without the affected episodes)

`Genesis/action_sampling.py::legalize_pushes` returns `ok=False` and the push UNCHANGED when no legal offset along the push
direction exists; `simple_mpc/learned_mpc.py::run_episodes_batched` records `legal_ok` but executes the returned push anyway,
so a `--legalize` run still executes illegal touchdowns at the `ok=False` rate. Measured (`results/run0002_extended.json`,
`results/run0003_noise_null.json` in EXP-0039): RUN-0002 CEM 350 / 16384 pushes = 2.1 % (per model 0.7-2.7 %, soft NFD sigma 2
5.9 %), RUN-0002 rank-128 0 / 8192, RUN-0003 scatter 195 / 8192 = 2.4 % (soft_s2 6.3 %), clump 11 / 8192 = 0.1 %. The CEM
planner (free refinement) produces them; the pile-aware rank candidates never do. Model orders are unchanged when each
model's affected episodes are dropped (RUN-0002 CEM tau 1.00; RUN-0003 scatter 0.79, clump 0.93), so C-073 / C-074 stand
with this caveat recorded. Exposure: EXP-0051 / EXP-0054 legal re-runs (2026-10-04) used the same path -- their `legal_ok`
rates are unmeasured. **To close:** the driver must replace an `ok=False` push (resample / skip / fall back to the best
legal candidate) rather than execute it; being done in EXP-0066. Then report the `ok=False` rate per record.

## ISS-015 — pasted-frame accuracy and slateN of window models are capped by raster-style mismatch (box window raster pasted onto the disc-style world raster)

**status:** open · **found:** 2026-10-06 (EXP-0072 ceiling study) · **severity:** medium for model comparisons among window/128 models, none for within-family comparisons

Rendering the TRUE next state in the window raster style and pasting its change onto the world raster scores accuracy_1 0.651 (zoom64 window) / 0.681 (zoom128) / 0.633 (world128), not 1.0, and pasted slateN 0.778 / 0.824 (vs 0.896 for the true hard w64 raster against the soft splat truth).
Zoom models (0.622/0.765, 0.640/0.809) sit within ~0.02 of these caps, so the pasted frame cannot resolve further improvements (pool sampling sd is 0.01-0.04). The cap differs per model type, so pasted numbers are not strictly comparable across frames.
**To close:** score window models in-window or through the soft/true-style raster; see experiments/EXP-0072-zoom-window-nfd/results/ceiling/results.md.
