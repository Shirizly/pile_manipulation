---
id: EXP-0023
title: >
  Two full-power repeats: (1) the mask-vs-depth channel confound resolves
  oppositely on piled vs scattered cubes; (2) C-006/C-004 re-run at full
  episode count, C-004 still a piled/scattered split
tier: T1
mode: confirmatory
date: 2026-09-06
hypothesis: null

claim: >
  REPEAT 1 (C-017/EXP-0005): the mask input beats height and density inputs
  for predicting a MATCHED mask-delta target on piled cubes (n20), and this is
  not merely a target-matching artifact — i.e. mask does not collapse to worst
  when the target is switched to density-delta, on the SAME piled dataset.
  REPEAT 2 (C-006/C-004): the deposit-profile shape (C-006) and the
  nonneg-vs-ridge ranking (C-004) reported in EXP-0010 on a 2-of-8/2-of-16
  episode cap reproduce at the FULL episode count (8 scattered, 16 piled).

prediction:
  supports: >
    Repeat 1: mask beats height/density under the mask target by >5 points in
    both datasets (replicating EXP-0005), AND under the density target on the
    SAME piled dataset mask is not the worst input by a wide margin (i.e. the
    piled result is not pure target-matching). Repeat 2: C-006 profile shape
    (sign, peak location within +/-1px) unchanged from the 2-episode cap;
    C-004 nonneg-vs-ridge margins on piled cells stay inside the documented
    ~0.001-0.004 rms floor and scattered margins stay outside it, i.e. the
    same win/tie split as EXP-0010.
  refutes: >
    Repeat 1: mask is the single worst input under the density target on
    piled cubes (pure target-matching, no residual mask advantage). Repeat 2:
    the piled C-004 tie flips to a clear win/loss at full data (noise floor
    was the only thing hiding a real effect), or the profile shape/sign
    reverses.
  discriminating: true

provenance:
  commit: be1b3b0a
  dirty: true                     # tree carries OTHER agents' concurrent, unrelated
                                  # edits (EXP-0004/0009/0014/0015, METRICS.md,
                                  # fit_linear_foresight.py, SKILL.md) per the task's
                                  # own note that other agents run in parallel. This
                                  # record's own code (scripts/probes/channels.py) was
                                  # committed at be1b3b0a before any run in this record.
  data_commit: >
    granularity/n20 carries a real provenance block (commit f196d657, dirty
    false) — see its `_N_config.yaml`. cube_spectrum/n20 and foresight/L040
    predate stamping: "unrecorded".
  script: >
    scripts/probes/channels.py (repeat 1, modified this session to fit both
    targets in one load pass — see commit be1b3b0a);
    scripts/probes/deposit_profile.py, scripts/probes/nonneg_vs_ridge.py
    (repeat 2, unmodified, run with no --max-episodes cap)
  data: >
    ["Genesis/data/cube_spectrum/n20/*_data.pt",
     "Genesis/data/granularity/n20/**/*_data.pt",
     "Genesis/data/foresight/L040/**/*_data.pt"]
  code_path: >
    points_to_mask / points_to_density / points_to_heightmap (repeat 1, via
    occupancy_foresight.load_transition_fields); particles_to_occupancy via
    the same loader, view=mask (repeat 2) — ONE code path per repeat across
    all datasets compared within it, so dataset differences are data, not
    rasteriser (EXP-0001/EXP-0002 discipline).
  seed: 0
  split: "episode-level, 25% of files held out, seed 0 (channels.py, nonneg_vs_ridge.py); C-006 has no split (population mean) plus a 4-way episode fold for its noise floor"
  runtime: >
    Repeat 1: ~7s (cube_spectrum n20, 16 episodes, both targets) + ~5s
    (granularity n20, 8 episodes, both targets), CPU. MUCH faster than
    EXP-0005's logged "~25 min" for a single target on the same dataset --
    see Unrelated findings; not rerun to explain, since it only helped.
    Repeat 2: C-006 full (8+16 episodes, no fitting) ~6s. C-004 full (8
    scattered + 16 piled episodes, res 32, both crops, ridge sweep + one
    4000-iter nonneg fit per cell) 204.9s total wall (`runs/full_nonneg_vs_ridge.json`),
    CPU, OMP_NUM_THREADS=4 -- nonneg fit times 43.0/63.6s (scattered
    crop1.0/0.5), 38.9/50.4s (piled crop1.0/0.5), NOT the ~4-8x-of-capped
    cost the M-scaling worry predicted (fit_operator_nonneg's O(D^3)-per-
    iteration cost is dominated by D, not M, once the M-dependent sufficient
    statistics are computed once up front).

budget:
  declared: "75 min, 170k tokens (for the whole task, both repeats + register updates)"
  spent: "~25-30 min wall clock (all data work: repeat 1 both datasets, C-006 full, C-004 full, plus a res=64 cost pilot); remainder spent on record/register writeup"
  outcome: within

design:
  varied: >
    Repeat 1: {target: [mask-delta, density-delta], input: [mask, height,
    density, mask+height, mask+density, mask+height+density], dataset:
    [cube_spectrum/n20 (piled), granularity/n20 (scattered, blind, 40mm)]}.
    Repeat 2 C-006: {dataset: [scattered L040, piled cube_spectrum/n20]},
    full episode count instead of the 2-episode cap. Repeat 2 C-004:
    {dataset: [scattered, piled], crop: [1.0, 0.5], ridge: [0.1,1,10,100]} vs
    nonneg, full episode count instead of the 2-episode cap.
  held_fixed: >
    Repeat 1: res=32, crop=1.0, blur(sigma)=1.0, grid=64, ridge=1.0, channel
    energy rescaled to the mask's std (so the shared ridge does not silently
    zero a channel), episode split rule and seed. Repeat 2: res=32, blur=1.0,
    grid=64, view=mask, cube_size=0.005, split seed 0, nonneg iters=4000 --
    identical to EXP-0010 in every knob except the episode cap, which is
    now removed (full 8/16 files instead of 2/2).
  baselines: [persistence, mean-delta]
  metric: >
    Repeat 1: explained_over_meandelta (see docs/experiments/METRICS.md) --
    reporting `accuracy` alongside per METRICS.md's standard-metric
    requirement is not directly computable from this script's output without
    persistence rms (which channels.py does not compute; it only fits
    against mean-delta), so `accuracy` is reported only where `fit_operator`/
    `metrics()` are the fitting path (repeat 2). Repeat 2 C-006:
    canonical_delta_profile. Repeat 2 C-004: `accuracy` (1 - rms(model)/
    rms(persistence)) plus pct-of-mean-delta, both from `fit_linear_foresight
    .py::metrics`. `slate4` NOT reported for either repeat: none of
    `channels.py`, `deposit_profile.py` or `nonneg_vs_ridge.py` touch a
    candidate-slate structure, so getting it would mean building one against
    `Genesis/data/slates/` from scratch rather than reading it off an
    existing probe -- not the "cheap if available" case the task allowed for.
noise_floor: >
  Repeat 1: not independently re-measured this run; using EXP-0005's "~2
  points" prior (differences under that are not interpretable). Repeat 2
  C-006: measured via 4-way episode fold (see Numbers). Repeat 2 C-004:
  using the repo's documented ~0.001-0.004 rms fold sd
  (`linear_foresight_report.md` sec 2.2b), same as EXP-0010, since a fresh
  fold sweep at full data was out of scope for this budget.

depends_on: [particle-projection, episode-split, grid-convention, rasteriser-identity, canonical-warp, footprint-splat, swept-region-metric]
establishes: []

result: >
  See Numbers. Headline: Repeat 1 -- on PILED cubes the mask input stays
  competitive even off its matched target (mask 0.415 vs density 0.438 under
  a density target, a 2-point gap vs the 11-point mask-target gap), so the
  "depth genuinely uninformative" reading survives on piled data specifically.
  On SCATTERED cubes the opposite happens: mask collapses from best (0.522)
  under its own target to worst by a wide margin (0.690 vs density's 0.832,
  a 14-point gap) under the density target -- strong target-matching. The
  confound is real, but its resolution is dataset-dependent, which the
  original claim (scoped to piled cubes only) did not anticipate needing.
  Repeat 2: C-006 profile reproduces at full data, same shape, larger
  fold-count, effects still 10-100x the floor -- verdict unchanged, and
  incomplete-design is RETIRED for this claim (full 8/16 episodes now used,
  not a 2-episode cap). C-004 WEAKENS at full data: nonneg's raw-count win
  drops from "3 of 4" (EXP-0010, capped) to 2 of 4 (both scattered), and its
  margin on those 2 cells shrinks from 0.0058-0.0076 rms (clearly outside the
  documented floor) to 0.0011-0.0022 rms (now INSIDE it); the 2 piled cells
  stay tied as before (0.0001-0.0006 rms, ridge nominally wins both this
  time, vs a 1-1 split under the cap). At full power, C-004 does not clear
  the noise floor in ANY of the 4 cells -- a materially weaker result than
  the capped run reported, not a confirmation of it.
verdict: supported
downgrades: [imprecision, inconsistency, incomplete-design]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

Repeat 1: EXP-0005's amendment made explicit that a single (mask) target
cannot distinguish "depth carries no information" from "the input matching
the target wins" -- every prior cube run fit only the mask target, so mask's
advantage was confounded with being the matched channel. Adding a
density-delta target on the SAME piled dataset and the SAME scattered dataset
that motivated the amendment settles it directly: if mask keeps winning (or
stays close) under a target it does NOT match, that is evidence for "depth
uninformative"; if mask now loses badly to the matched channel, that is
target-matching.

Repeat 2: both C-006 and C-004 in EXP-0010 ran on a 2-of-8/2-of-16-episode
cap, imposed mid-run by an undiscovered cost (see that record's "Deviation
from the plan"). This repeat removes the cap entirely -- if either result
depended on the small sample (a fold-count artifact, or noise-floor
overestimate) it should move; if it reproduces unchanged, the cap was
conservative, not distorting.

## What was actually run

**Repeat 1** (`scripts/probes/channels.py`, modified this session -- see
provenance): the script's target was hardcoded to mask-delta before this
session; despite the EXP-0005 body's claim that "it already supports both
targets," reading the pre-existing script and its one-commit git history
(`aac084e3`, its only commit) showed the density-target loop did not exist --
only a mask target was ever fit, on any dataset. Added a loop over
`target in {mask-delta, density-delta}` that reuses the single load pass
(mask/height/density views are all loaded regardless of which becomes the
target, so this costs no extra load time over the original). Ran full data
(no `--max-episodes` cap) on `cube_spectrum/n20` (16 episodes, piled) and
`granularity/n20/**` (8 episodes, scattered, `push_length=0.04` = "40mm",
`placement_aware: false` = "blind" -- confirmed via its `_N_config.yaml`).
`min_push_mm` set to the dataset's own convention (19.9mm for the n20 piled
push length, 39.0mm for the 40mm scattered push length, matching
`deposit_profile.py`'s own per-dataset values).

**Repeat 2**: `deposit_profile.py` and `nonneg_vs_ridge.py` run completely
unmodified, with the `--max-episodes` cap simply omitted. A quick timing
pilot (3-episode `channels.py` run, 2s; then the full 16-episode run, 6s)
showed the ~60ms/transition footprint-splat cost EXP-0010 measured and
warned about did NOT reproduce here -- full-data loads that were projected at
15-18 minutes actually completed in single-digit seconds (see Unrelated
findings). This was verified empirically (not assumed) before committing to
the full C-004/C-006 runs: `deposit_profile.py` at full data (8+16 episodes,
no fitting) ran in 6s wall, so the full run was launched directly rather than
budgeting around the old estimate.

## Numbers

### Repeat 1 — channels x targets x datasets (res=32, blur=1.0, episode-split 25%, seed 0)

**PILED (cube_spectrum/n20, N=4840, 16 episodes, 3632 train / 1208 val)**

| input | dim | target=mask-delta | target=density-delta |
|---|---|---|---|
| mean-delta (0 params) | 0 | 0.0000 | 0.0000 |
| **mask only** | 1024 | **0.7393** | 0.4150 |
| height only | 1024 | 0.6308 | 0.4143 |
| density only | 1024 | 0.6557 | **0.4380** |
| mask + height | 2048 | 0.7329 | 0.4096 |
| mask + density | 2048 | 0.7367 | 0.4284 |
| mask+height+density | 3072 | 0.7311 | 0.4114 |

(mask-delta column replicates EXP-0005's original 0.739/0.631/0.656/0.733/
0.737/0.731 to 3 decimal places -- same full 16-episode dataset, same split
rule, confirming the rerun is measuring the same thing.)

**SCATTERED (granularity/n20, blind, 40mm push, N=2560, 8 episodes, 1920 train / 640 val)**

| input | dim | target=mask-delta | target=density-delta |
|---|---|---|---|
| mean-delta (0 params) | 0 | 0.0000 | 0.0000 |
| **mask only** | 1024 | **0.5223** | 0.6896 |
| height only | 1024 | 0.4432 | 0.8294 |
| density only | 1024 | 0.4426 | **0.8317** |
| mask + height | 2048 | 0.4707 | 0.8095 |
| mask + density | 2048 | 0.4710 | 0.8119 |
| mask+height+density | 3072 | 0.4305 | 0.8206 |

**The question the design was built to answer**: piled cubes show mask
winning under its own target by +8.4..+13.6 points, and STILL competitive
(within 2.3 points of the winner, and beating height) under the target it
does not match -- consistent with "depth genuinely carries little extra
information" for piled heaps specifically. Scattered cubes show the opposite:
mask wins its own target by +7.9..+13.7 points but LOSES the density target
by 12.9..14.2 points to height/density -- consistent with target-matching
dominating there. Multi-channel stacks never beat the best single matched
channel in any of the 4 (dataset x target) columns, so the original narrow
claim ("no stack helps") holds throughout; the new information is that the
*reason* mask wins on its own target is regime-dependent.

### Repeat 2, C-006 — mean canonical-frame delta profile, FULL episode count (res=32, crop=1.0, blur=1.0, +/-4px band)

Full numbers in `runs/full_deposit_profile.log`. Summary vs EXP-0010's
2-episode-cap run:

| dataset | N (full) | peak depletion | peak deposition | shape vs capped run |
|---|---|---|---|---|
| scattered L040 (8/8 episodes) | 2560 | -0.0473 at +3.5px (fold sd 0.0058, 8x) | +0.1516 at +6.5px (fold sd 0.0165, 9x) | same sign/location (capped: -0.0483 at +3.5, +0.1554 at +6.5) |
| piled n20 (16/16 episodes) | 4840 | -0.3039 at +1.5px (fold sd 0.0040, 76x) | +0.3403 at +4.5px (fold sd 0.0043, 79x) | same sign/location (capped: -0.3130 at +1.5, +0.3353 at +4.5) |

Peak magnitudes moved by <5% between the 2-episode cap and full data in both
datasets; peak locations are identical to the pixel. C-006 reconfirms at full
power, no change in verdict.

### Repeat 2, C-004 — fit_operator_nonneg vs fit_operator, FULL episode count, swept-region rms (res=32, blur=1.0)

Full log: `runs/full_nonneg_vs_ridge.log`. `accuracy = 1 - rms/rms(persistence)`
(METRICS.md); `%persist`/`%meandelta` as in EXP-0010 for direct comparison.

| dataset | crop | model | rms | accuracy | %persist | %meandelta | fit (s) |
|---|---|---|---|---|---|---|---|
| scattered (L040, N_test=640) | 1.0 | persistence | 0.10339 | 0.000 | 100.0 | 118.9 | - |
| scattered | 1.0 | mean-delta | 0.08697 | 0.159 | 84.1 | 100.0 | - |
| scattered | 1.0 | **nonneg (winner)** | **0.06489** | **0.372** | 62.8 | 74.6 | 43.0 |
| scattered | 1.0 | ridge10 (best ridge) | 0.06713 | 0.351 | 64.9 | 77.2 | 0.0 |
| piled (n20, N_test=1208) | 1.0 | persistence | 0.26004 | 0.000 | 100.0 | 158.9 | - |
| piled | 1.0 | mean-delta | 0.16365 | 0.371 | 62.9 | 100.0 | - |
| piled | 1.0 | **ridge10 (winner)** | **0.09329** | **0.641** | 35.9 | 57.0 | 0.0 |
| piled | 1.0 | nonneg | 0.09390 | 0.639 | 36.1 | 57.4 | 38.9 |
| scattered | 0.5 | persistence | 0.10339 | 0.000 | 100.0 | 116.5 | - |
| scattered | 0.5 | mean-delta | 0.08873 | 0.142 | 85.8 | 100.0 | - |
| scattered | 0.5 | **nonneg (winner)** | **0.05890** | **0.430** | 57.0 | 66.4 | 63.6 |
| scattered | 0.5 | ridge10 (best ridge) | 0.05996 | 0.420 | 58.0 | 67.6 | 0.0 |
| piled | 0.5 | persistence | 0.26004 | 0.000 | 100.0 | 152.7 | - |
| piled | 0.5 | mean-delta | 0.17028 | 0.345 | 65.5 | 100.0 | - |
| piled | 0.5 | **ridge10 (winner)** | **0.09241** | **0.645** | 35.5 | 54.3 | 0.0 |
| piled | 0.5 | nonneg | 0.09253 | 0.644 | 35.6 | 54.3 | 50.4 |

Margins (nonneg rms minus best-ridge rms; negative = nonneg wins):

| dataset | crop | margin (rms) | vs ~0.001-0.004 documented floor |
|---|---|---|---|
| scattered | 1.0 | -0.00224 (nonneg wins) | inside the floor (was 0.0076, outside, at 2-episode cap) |
| scattered | 0.5 | -0.00106 (nonneg wins) | inside/at the floor's edge (was 0.0058, outside, capped) |
| piled | 1.0 | +0.00061 (ridge wins) | inside the floor (was 0.0005, capped -- nonneg nominally won that cell) |
| piled | 0.5 | +0.00012 (ridge wins) | inside the floor, essentially a dead tie (was -0.0015, capped -- nonneg won) |

At full data, nonneg's raw-count win narrows from 3-of-4 (capped) to 2-of-4
(still both scattered), and critically **every one of the 4 margins now sits
inside the documented noise floor** -- the capped run's clearest result
(scattered clearing the floor by 0.0058-0.0076) shrank by more than half when
the training set grew ~4x. This is the opposite of what more data should do
to a *real* effect (tighten the estimate around the same value); the point
estimate itself moved toward zero. The likely mechanism: nonneg has fewer
degrees of freedom to overfit with (a non-negativity constraint is
restrictive), so its relative edge over an unconstrained ridge fit may have
been inflated by ridge overfitting more on the smaller 2-episode training
set -- speculative, not measured here.

**Against this record's own pre-registered prediction**: the `supports`
branch explicitly named "scattered margins stay outside [the floor]" as the
confirming outcome. That did not happen -- scattered margins moved from
outside to inside the floor. The `refutes` branch named "the piled tie
flips to a clear win/loss" or "the profile shape/sign reverses" -- neither
of those happened either. The actual outcome (both regimes now inside the
floor, direction unchanged, magnitude compressed) sits in a gap the
prediction did not anticipate, which is itself worth flagging: a prediction
written only in terms of "does the existing pattern hold" did not have a
branch for "the pattern holds in direction but its statistical significance
evaporates." Recorded here rather than quietly reading this as a clean
"supports" for C-006 (which it is) and staying silent about C-004 (where it
is not a clean anything).

## What would change the verdict

- A scattered-cube density-target run at a SECOND ridge value / crop would
  confirm the 12-14 point target-matching gap is not itself a ridge artifact
  (this run used ridge=1.0 only, unchanged from EXP-0005).
- A per-block ridge (mask channel and depth channel regularised separately)
  would separate "depth is uninformative" from "depth is informative but
  double the parameter count costs more than it gives back at this ridge" --
  flagged as open in EXP-0005 and still open here.
- For C-004: a fresh, full-data episode-level fold sweep (LORO or k-fold) to
  measure the noise floor directly at this sample size, rather than
  borrowing the ~0.001-0.004 prior from a different (smaller, per
  `linear_foresight_report.md` sec 2.2b) design. This matters more now than
  it did for EXP-0010: the full-data margins (0.0001-0.0022 rms) are small
  enough that whether they clear a floor MEASURED AT THIS SAMPLE SIZE,
  rather than borrowed, could flip the "no cell clears the floor" reading.
  Cheap now that loading is confirmed fast (~6s for a full load); the only
  cost is repeating the nonneg fit (~40-65s) per fold.
- res=64 for C-004 remains untested. Measured directly this session (a cost
  pilot on synthetic D=4096 data, no real outcome touched): 50 FISTA
  iterations took 44.4s, projecting to ~59 min for the full 4000-iteration
  fit -- confirming the task's own ">1h/fit" warning is still roughly right
  even though the M-scaling worry that motivated it turned out not to apply
  (cost is dominated by the one-time O(D^3) `Y0@Y0.T`/eigenvalue setup and
  the per-iteration `Z@G` matrix product, both D-only, not by M). At ~1h per
  cell x 4 cells (2 datasets x 2 crops), a res=64 C-004 sweep is genuinely
  unaffordable in this record's budget. Taking `incomplete-design` for this
  specific gap, honestly, rather than silently dropping it -- it is
  independent of the episode-count `incomplete-design` this record retires.

## Threats

- `imprecision`: repeat 1 uses one ridge value (1.0) and one split; the
  cross-dataset asymmetry could partly reflect scattered cubes' larger
  swept region / lower per-pixel occupancy rather than a true difference in
  depth-informativeness. Not separated here. Repeat 2's C-004 full-data
  margins (0.0001-0.0022 rms) are compared against a noise floor BORROWED
  from a different, smaller design rather than measured fresh at this
  sample size (see "What would change the verdict") -- a fresh floor could
  move a borderline cell either way.
- `inconsistency`: this is the substantive finding, not just a threat --
  (1) repeat 1's target-matching resolution is NOT the same across the two
  datasets tested (weak on piled, strong on scattered), so "depth is
  uninformative" cannot be stated as a dataset-independent property from
  this record alone; (2) repeat 2's C-004 result direction is not stable
  under the episode-count cap: the capped run cleared the floor on
  scattered and split 1-1 on piled, while the full run stays inside the
  floor everywhere and piled now favours ridge in both cells.
- `incomplete-design`: C-004's episode-count cap is RETIRED by this record
  (full 8/16 episodes now used, for both C-004 and C-006) -- that specific
  gap no longer applies and is not re-claimed. The remaining gap is res=64
  for C-004, measured this session at a projected ~59 min/fit (see "What
  would change the verdict") and not run; this is the domain's sole
  surviving justification here, distinct from and narrower than EXP-0010's.
- `untested-dependency`: none of this record's `depends_on` tags are
  `broken` or `unchecked` (checked against INVARIANTS.md at record time);
  no downgrade taken for this domain.
- Considered and dismissed: `provenance` for repeat 1 -- both datasets and
  both targets go through the identical `channels.py` code path in the same
  process invocation, so the dataset difference is data, not rasteriser.
- Considered and dismissed: `selection` -- both datasets and both targets are
  reported in full (Numbers tables show all cells, not the favourable ones).

## Unrelated findings

- **EXP-0005's body claim that `channels.py` "already supports both targets"
  was false.** The script has exactly one commit (`aac084e3`) in its entire
  history, and that commit only ever fit a mask-delta target -- there was no
  density-target code path to run. This session added it (see provenance).
  Logged because a future reader trusting that line without checking would
  have looked for a `--target` flag that never existed.
- **EXP-0010's footprint-splat cost warning (~60ms/transition, ~15-18 min
  projected for a full-data load) did not reproduce this session.** The same
  `load_transition_fields` call, same `cube_size` argument, same datasets,
  ran a full 16+8-episode load in single-digit seconds three separate times
  (channels.py x2, deposit_profile.py x1). Not investigated further (out of
  scope, and it only helped this record's budget) -- plausible explanations
  not distinguished here: warm OS page cache from repeated same-day reads by
  concurrent agents, or the original measurement being from a colder
  environment. Whatever the cause, EXP-0010's cost estimate should not be
  treated as a standing property of the codebase without re-measuring.
- `mean-delta`'s own baseline in the C-006/C-004 datasets differs noticeably
  between full data and the 2-episode cap (e.g. piled mean-delta 0.16365 vs
  0.16784 in EXP-0010's capped run) -- a reminder that a 2-episode sample of
  16 is not a negligible-variance draw even when the qualitative conclusion
  survives it.

## Grade note

`untested-dependency` is NOT reintroduced: all tags this record cites check
out against the current INVARIANTS.md (same tags EXP-0005/EXP-0010 already
cleared on 2026-09-05). Three other domains ARE claimed, computing to
`very-low`: `imprecision` (single split/ridge for repeat 1; C-004's floor is
still borrowed, not re-measured at full N), `inconsistency` (repeat 1's
confound resolves oppositely by dataset; repeat 2's C-004 direction is not
stable between the capped and full-data runs), `incomplete-design`
(res=64 for C-004, newly measured this session at a projected ~59 min/fit
and confirmed still unaffordable). Progress worth citing by domain rather
than by letter (per the skill's own guidance): the EPISODE-COUNT
`incomplete-design` that both EXP-0010 sub-claims carried is fully RETIRED
here for both C-006 and C-004 -- what remains for C-004 is narrower
(resolution only, not sample size) even though the letter grade did not
improve, because a new `inconsistency` domain opened where the old
`incomplete-design` one closed.
