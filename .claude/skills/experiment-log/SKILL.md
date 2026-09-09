---
name: experiment-log
description: "Record an experiment, a hypothesis test, or a measured result in this repo so it can be compared with other evidence later and invalidated cleanly if an assumption breaks. Use when running any probe, fit, sweep, ablation or benchmark whose number might be cited; when stating or revising a claim; when a result contradicts an earlier one; and ALWAYS when a bug, convention error or code-path difference is found that could affect results already recorded. Also use before designing an experiment, to check the claim is falsifiable and the test discriminates."
argument-hint: "Optionally: 'new <slug>', 'register', 'invalidate <tag>', or the claim you are about to test"
user-invocable: true
---

# Experiment log — recording evidence so it composes

## Why this exists (read once; it justifies every field below)

This repo's docs are already precise and caveated. Precision was not the
problem. Six things went wrong anyway, and each one is now a required field:

| What happened | Field that prevents it |
|---|---|
| Occupancy and plate channels were mutually transposed for the whole linear-foresight programme; "nothing beats persistence" was an artifact (`docs/prediction_difficulty_hypotheses.md` §1) | `depends_on` + a backing invariant test |
| Two rows of one comparison came from two different rasterisers — the doc *warned about this in prose*, then did it anyway (retracted in `242a9cc1`) | `provenance.code_path` |
| One view was compared at σ=1 against another at σ=0, so the view effect and the blur effect were observationally identical (EXP-0003) | `design.held_fixed` |
| Effects of ~0.001 were interpreted against a fold sd of ~0.004 (`linear_foresight_report.md` §2.2b) | `noise_floor`, stated before the result |
| Per-pixel error was measured for a long time before asking whether it bore on the claim (§2.4) | `downgrades: [indirectness]` |
| When the transpose was found, nothing indexed which claims died | `REGISTER.md` `depends_on` column |

Prose warnings do not bind. Fields do, because the validator reads them.

**The reader is the next session, which has no memory.** Write frontmatter a
script can parse and a body someone can skim in 30 seconds. Do not write essays
here — the narrative docs in `docs/` are still where synthesis lives.

## The three artifacts

| File | Role |
|---|---|
| `docs/experiments/EXP-####-slug.md` | one record per experiment; frontmatter is the contract |
| `docs/experiments/REGISTER.md` | the claim ledger — one row per claim, with what supports it, what contradicts it, and what it depends on |
| `docs/experiments/INVARIANTS.md` | the `depends_on` tag registry: each tag names a checkable property and the test that checks it |

Validate with `python scripts/check_register.py` before considering any
experiment done. It is fast, needs only pyyaml, and enforces every rule below —
including recomputing the grade, so it cannot be inflated. `--fix-grades`
rewrites them in place.

**Expect most records to grade `low` right now.** Many invariants are
`unchecked` and two are `broken`, so almost everything takes
`untested-dependency`. That is an accurate description of the repo's current
evidential state, not a flaw in the scale, and it is the pressure that gets the
tests written.

## Pick a tier — this is the anti-ceremony guard

**T0 probe.** Frontmatter only: `id, title, tier, claim, provenance, result,
verdict`. No body. Use for quick looks — most of what gets run. Should take
under a minute; if it doesn't, the tier is wrong.

**T1 experiment.** Full frontmatter + body. Use when a number will be cited in
a report or doc, or when it will be compared against another experiment.

**T2 gate.** T1 plus three additions: the `prediction` block committed to git
*before* the run; a multiverse sweep (see below); and every `depends_on` tag
backed by a passing test. Use for a claim that will be leaned on — a headline
result, a go/no-go, or anything contradicting an existing register entry.

When in doubt use T1. Escalating later is cheap; a T0 number that ends up in a
paper is not.

## Before you run anything: the plan gate

At T1 and T2, write **≤200 words** and stop, before touching data:

1. the claim, as one sentence that could be false;
2. the prediction — `supports` / `refutes`, with the exact quantity named;
3. the design — what varies, what is held fixed, what the baselines are;
4. **the cost estimate**, and the cheapest check that could invalidate the
   whole plan (an import that fails, a dataset smaller than assumed, a fit that
   turns out to be O(D³));
5. the one result that would most embarrass the plan, and whether the design
   would catch it.

Run item 4 first. A two-minute feasibility check that saves an hour is always
in budget, and finding out that the expensive cell is unaffordable *before*
designing around it is worth more than any amount of careful writing after.

Keep the plan; it becomes the record's frontmatter almost verbatim.

## The record

Copy `references/experiment-template.md`. Field meanings:

- **`claim`** — one sentence that could be false, **and that names the
  conditions it is claimed under**. "Blur helps" is not a claim. "At matched
  blur, mask and density views differ by <5 points of explained variance on
  both materials" is. If you cannot name a measurement that would make it
  false, it is not ready to test.

  **Name the scope: dataset, objective/metric, and the design knobs the result
  could plausibly depend on.** This is not pedantry, it is the difference
  between a claim that gets narrowed later and one that gets "refuted" later
  for no reason but its own over-reach. A real case from this repo: every
  control claim (C-030, C-035, C-044, C-045, C-046) was stated as being about
  *control*, while every measurement behind it used one cost functional whose
  weight field has 63% of its spatial-variation energy in the lowest frequency
  bin. When a spatially selective objective was finally tried and the numbers
  moved, there was no way to tell from the claim text whether the earlier work
  was wrong or merely narrower — because the claim never said what it was
  conditional on. Had the scope been in the sentence, the later result would
  have *extended* the map instead of appearing to demolish it.
- **`prediction.supports` / `.refutes`** — observable outcomes *with
  thresholds*, written before the run. **Name the exact quantity**, including
  its normalisation and what it is measured against — not just a direction. "X
  rises with r" is not a prediction if "X" could mean the raw score, the score
  minus a baseline's score, or the score against a usable-skill threshold;
  those three can disagree in *shape*, and a reader picking a different one
  will reach the opposite verdict from the same numbers. This is not
  hypothetical: it happened in EXP-0007. A **cost pilot** — a short run that
  measures only runtime, memory or feasibility — does not compromise a later
  prediction, and needs no disclosure. A pilot that reveals any *outcome* does:
  either discard its cells and re-run after writing the prediction, or keep
  them and set `mode: exploratory`. Say which in "What was actually run". **`discriminating: false` means stop**:
  if both branches predict the same observation, the experiment cannot teach
  you anything and should be redesigned, not run.
- **`provenance.commit` / `.dirty` / `.data_commit`** — three different code
  states, and they are genuinely different questions. `commit` is the analysis
  code; `dirty` says whether a sha even identifies it (if the tree was
  modified, it does not, and you must say what was uncommitted); `data_commit`
  is the *simulator* the dataset was collected under, which is a separate
  lineage entirely — physics, spawn mode, settle criterion and action sampler
  all live in code and have all been changed mid-project. Get all three from
  `utils.git_provenance()`; the dataset's is stamped into its `_N_config.yaml`.
  **Commit before you run**, so `dirty` is false and the sha means something.
- **`provenance.code_path`** — which implementation produced the numbers
  (`points_to_mask` vs the `PileSweepData` rasteriser vs
  `particles_to_occupancy`). Two records may only be compared if this matches,
  or the comparison carries `downgrades: [provenance]`.
- **`design.varied` / `design.held_fixed`** — everything that moved and
  everything that did not. `held_fixed` must list every knob the compared
  configurations share. If a knob is neither varied nor held fixed, the design
  is incomplete.
- **`design.metric`** — report the two standard metrics, **`accuracy`** (image)
  and **`slate4`** (control), in every record. Both go up when better. Where the
  metric is itself what you are studying, give a standard one as a reference row
  anyway. Then name a metric **key from `docs/experiments/METRICS.md`**,
  not a prose description. "swept-region rms as % of persistence" does not say
  whether the average is a ratio of means or a mean of ratios, what the
  denominator is, or how it moves under preprocessing — and all three of those
  ambiguities have caused a real misreading here. If your metric is not in that
  file, add it there first, with its formula.
- **`design.baselines`** — non-empty, always, and must include a "do nothing"
  baseline. In this repo that is `persistence` for image prediction and
  `mean-delta` for canonical-frame prediction. `mean-delta` is the one that
  matters: a canonical frame normalises the action away, so beating persistence
  is nearly free, and mean-delta alone already reaches 0.12–0.35 explained
  (C-011).
- **`noise_floor`** — a number and its source, written before the result.
  Fold-to-fold sd, seed-to-seed sd, or a shuffled-label control. An effect
  inside it is `inconclusive`, never `refuted`.
- **`depends_on`** — tags from `INVARIANTS.md`. Every assumption whose failure
  would change the verdict. Be generous; the cost is one word and the payoff is
  automatic invalidation.
- **`establishes`** — tags whose *status this record sets*, as opposed to relies
  on. The record that discovers a broken invariant does not take the
  `untested-dependency` downgrade for it; it is the evidence, not a victim of
  it. Use sparingly — one record per tag.

## Verdicts and grades

Verdict is one of: **`supported`** · **`refuted`** · **`inconclusive`** (effect
inside the noise floor, or the design did not discriminate) · **`invalidated`**
(the result stands as measured, but an input is now known broken — distinct
from refuted, and the distinction matters when re-running).

### `narrowed` is a claim status, and usually the right one

Record **verdicts** stay the four above. Claim **statuses** in `REGISTER.md`
additionally include **`narrowed`**: the measurement stands, nothing about it
was wrong, and its generality is smaller than the claim asserted. Distinguish:

| status | when |
|---|---|
| `narrowed` | the result reproduces under the conditions it was measured in, and a new condition changes it. Nothing is wrong; the scope was unstated or overstated |
| `contested` | evidence genuinely conflicts **under matched conditions** — two comparable measurements disagree and no condition explains the difference |
| `invalidated` | an input is known broken (a bug, a confound, a wrong floor). The measurement itself does not stand |

### Words to use, and one to stop reaching for

**Reserve "artifact" for a measurement that was wrong** — a bug, a confound, a
borrowed noise floor, a mis-placed target. This repo has real examples: the
transposed occupancy/plate channels, and a goal whose target never overlapped
the pile. In those cases the number never meant what it appeared to mean.

**A result that holds under one condition and not another is not an artifact.**
It is a real effect with a governing condition, and the useful report names the
condition: "holds under a coarse objective, not under a selective one" tells
the next session where to look. "Was an artifact" tells them to throw away work
that is still valid inside its range, and quietly discards the information
about *which* variable governs the effect — usually the most valuable thing the
experiment produced.

Likewise **`refuted` is for a claim that fails under its own stated
conditions.** If a claim never stated its conditions and later fails outside
them, the honest entry is `narrowed`, plus a note that the original claim was
under-specified — which is a lesson about claim-writing, not about the
measurement. (This cuts both ways: an under-specified claim should not be able
to survive contact with evidence by being retroactively reinterpreted as
conditional. Fix the scope requirement at claim-writing time, above, and this
does not arise.)

**Grade is computed, never chosen.** Start at `high`; drop one level per
downgrade domain present. `high → moderate → low → very-low`. The validator
recomputes it and fails on a mismatch, so inflating it is not possible.

**The letter saturates at three domains, and the downgrade list is the finer
signal.** A record carrying four weaknesses and one carrying three both read
`very-low`, so retiring one does not move the letter — which means the letter
alone cannot show a claim getting *better*. That is a real limitation, found
when EXP-0018 retired `provenance` from C-001 and the grade did not budge.
When reporting progress on a claim, cite the **domains retired**, not the
letter: "provenance retired, three remain" says what "still very-low" hides.

Downgrade domains (adapted from GRADE; use only these words):

| Domain | Applies when |
|---|---|
| `provenance` | the comparison crosses code paths, rasterisers, datasets, or conventions |
| `imprecision` | effect not clearly above `noise_floor`, or a single split where folds were available |
| `indirectness` | the measured quantity is a proxy for the one the claim is about (one-step pixel rms standing in for control utility) |
| `inconsistency` | it does not hold across every seed, fold or configuration tested |
| `selection` | the reported configuration was chosen after seeing results |
| `incomplete-design` | a cell the design called for was not run — ran out of budget, compute, or data. Distinct from `imprecision`: the cells you have may be perfectly precise, but the design has a hole |
| `untested-dependency` | a `depends_on` tag is not `holds`/`fixed`. **Forbidden at T2**; does not apply to tags in `establishes` |

## Bounded metrics have a ceiling, and the ceiling makes trends

Before reading any curve, check whether the score you are plotting is bounded.
Scores like FSS, IoU, SSIM and R² converge to their ceiling as the comparison
is made easier (bigger neighbourhood, coarser scale, more smoothing). Two
consequences, and you must say in the record which you ruled out:

- **Raw curves rise trivially.** Everything looks skilful when smoothed enough,
  including "predict no change".
- **A difference of two bounded scores is forced to zero**, because both terms
  converge to the same ceiling. So `score(model) − score(baseline)` declining
  with scale is *not* evidence that skill declines with scale. Subtracting a
  baseline is the obvious fix for the first problem and it silently creates the
  second one.

The way out is a **reference level with meaning**, not a difference: for FSS,
Roberts & Lean's `FSS_useful = 0.5 + f0/2` for base rate `f0`; for R², a
shuffled-label floor; for IoU, the value a null model attains. Report the
scale at which the model crosses that level.

## Budgets

**Every experiment handed to an agent gets a declared budget, in the prompt,
before any work starts.** Budget in **wall-clock and tokens**, never in tool
calls — tool-call caps punish the cheap checks (a `grep`, a 3-line probe) that
make an experiment trustworthy, and reward one big unverified run.

Sizing is the task-giver's job, from their own estimate of scope:

| tier | typical budget | if it looks bigger than this |
|---|---|---|
| T0 probe | ~5 min, ~20k tokens | it is not a T0 |
| T1 experiment | ~20–40 min, ~80–120k tokens | split it, or say which cells are optional |
| T2 gate | ~1–2 h, ~250k tokens | it needs its own plan, not a prompt |

Record the budget and what was actually spent:

```yaml
budget:
  declared: "30 min, 100k tokens"
  spent: "~42 min, 178k tokens"
  outcome: exceeded        # within | exceeded | stopped-early
```

**What to do when the budget runs out — this is the part that matters.** Stop
and report what you have. Do **not** silently drop cells to fit, and do **not**
keep going quietly. Specifically:

- Cells you planned but could not run: name them, take the `incomplete-design`
  downgrade, and put the missing cell in "What would change the verdict" with
  its measured or estimated cost.
- If the missing cells are the ones that would discriminate, the verdict is
  `inconclusive`, not `supported`. A design with its decisive cell missing has
  not tested anything, however many other cells ran.
- **Measure the cost of the expensive thing before committing to a sweep.** A
  two-minute feasibility check that saves an hour is always in budget, and a
  cost pilot never compromises a prediction (see `prediction`, above).

**Use `scripts/run_probe.py` for anything long.**

    python scripts/run_probe.py --tag blur_sweep -- python scripts/probes/view_blur.py --res 32

It forces `python -u`, redirects straight to a file (no pipe to get wrong),
records the PID so a job is stopped by PID rather than by a `pkill` pattern
broad enough to match the killer, caps threads, and writes
`utils.git_provenance()` beside the log so the code state is captured at run
time instead of reconstructed later. The three hazards below are the reason it
exists; each one bit repeatedly on 2026-09-05, twice *after* the warning was
written, which is the evidence that prose was not enough.

## Every command gets logged *before* it runs

`provenance.script` names what ran; it does not name **how**. Several results in
this register turn on an argument — which slate config, which `--goals`, which
`--split` — and those were recoverable only for jobs that happened to go
through `run_probe.py`. Everything else had to be reconstructed later from
output filenames, which is guesswork wearing a timestamp
(`docs/experiment_commands.md` marks the reconstructed ones; there are eight).

**Two writes per run, and the first happens at launch, not at completion.**

1. **Append a start event to the ledger `runs/COMMANDS.jsonl`** — append-only,
   one JSON object per event, written *immediately before the process starts*:

   ```json
   {"run_id": "n20_L20mm-2026-09-08T14:03:11-4021", "event": "start",
    "tag": "n20_L20mm", "exp": "EXP-0024_v1", "commit": "9e78da53",
    "dirty": false, "out": "runs_expB/n20_L20mm",
    "cmd": ["python", "-u", "-m", "training.train", "..."], "status": "running"}
   ```

   `exp` may be `"unfiled"` — an unfiled command is still worth more than no
   command. `cmd` is argv, never a re-typed prose version of it.

2. **Drop a `COMMAND.txt` in the run's own output directory** (`--artifact-dir`),
   holding the same argv plus the commit sha. The ledger is for the human
   sweeping the project; this copy is so the command *travels with the data* —
   output dirs get moved, copied and shared, and a dataset whose invocation
   lives only in a central file elsewhere becomes unreconstructable the moment
   it is copied.

**On exit, append an `event: end` record** carrying `status` (`ok` / `failed` /
`killed`) and `seconds`, joined to its start on `run_id`. Completion is a second
record rather than an edit of the first, because two runs finishing in the same
moment would lose an update under read-modify-write. **A start with no end means
the job was interrupted** — which is information, not an error: a suspended
machine left two collections in exactly that state, and the ledger is how you
tell an interrupted run from a finished one without re-deriving it from file
counts.

**Write-before-run is the whole point.** `runs/<tag>.json` used to be written
only after the process exited, so every killed, OOMed or suspended run left a
log with no invocation beside it. `runs/n50_L20mm.log` is exactly that: a
stopped collection with a `.pid`, a log, and no record of the command that
produced it — and it is precisely the run whose command you need.

**`run_probe.py` does both writes for you, so route analysis runs through it
too, not just long ones.** A 25-second re-score whose `--goals` decided the
result is exactly the case this rule exists for. If you invoke something
directly, you own the ledger line by hand — and if that feels tedious, use the
wrapper.

**Never present a reconstructed command as a recorded one.** If you rebuilt an
invocation from output filenames or a usage docstring, label it as
reconstructed wherever you write it down.

**Never block on a watcher.** If you launch a long job in the background, do
not then wait on it, poll it, or set up a monitor and return. Read whatever its
output file already holds and write up what is in it, or run the job in the
foreground with a bounded timeout and a configuration small enough to finish.
Three separate agents stalled this way on 2026-09-05, each returning with no
deliverable while its own results sat in a file it had not read. An unwritten
result is worth nothing; a partial record is worth a lot.

Two practical corollaries, both learned the hard way the same day:
- **`python -u`, always.** Python fully buffers stdout when redirected to a
  file, so a long job looks frozen for its entire run and you cannot tell a
  slow job from a hung one.
- **Cap your threads** (`OMP_NUM_THREADS=4`) and check `uptime` before
  starting. This is a shared machine; four concurrent BLAS jobs took the load
  average to 37 on 20 cores and starved an agent's whole budget, so it reported
  a degenerate 8x8 configuration as its only completed cell.

A budget is a planning instrument, not a hard stop: exceeding it is allowed and
is recorded, not hidden. Repeatedly exceeding it means the tier or the sizing
is wrong, which is information about the process rather than about the science.

## Unrelated findings

Real work turns up things that have nothing to do with the claim: a function
that is quietly O(D³), a stale doc reference, a config key that does nothing, a
dataset that is half the size its name implies. These have no home in a claim
register and get lost in chat logs.

Every T1/T2 record ends with an **`## Unrelated findings`** section. Rules:

- One bullet each: what you found, where, and how you know.
- **Do not act on them** — no fixing, no filing, no refactoring. They are
  logged for the user to triage. Chasing them is how a 30-minute experiment
  becomes a three-hour one.
- If a finding would change this experiment's verdict, it is not unrelated —
  it belongs in Threats, or it is a new `depends_on` tag, or it stops the run.
- Write "none" rather than deleting the heading. An empty section is
  information; a missing one is ambiguous.

## Multiverse discipline (T2, and encouraged everywhere)

If you swept it, publish the whole sweep, not the winning cell. If you did not
sweep it, `held_fixed` says so and the body says why that value. A single cell
reported from an unreported sweep is `downgrades: [selection]`, and this repo
has already been burned by it — a crop=0.5 result "beat persistence" on one
seed and did not survive seeding (`linear_foresight_report.md` §2.2b).

## Exploration is not confirmation

T0 and exploratory T1 records do not carry a `prediction` block, and must set
`mode: exploratory`. That is fine and expected — most work is exploration. But
an exploratory number may not be cited as confirming a claim: promote it by
re-running as T2 with the prediction written first. The register enforces this
by refusing `grade: high` to claims supported only by exploratory records.

## When you find a bug

This is the case the whole design exists for, so it has its own procedure.
**Commit the fix as soon as it is plausible, even tentatively** — mark it as
tentative in the message and leave the invariant's status honest rather than
sitting on an uncommitted fix. Every experiment run against an uncommitted fix
records a sha that does not describe the code that ran, and every dataset
collected against one is unreconstructable. A tentative commit that is later
reverted costs nothing; a week of results with no recoverable code state costs
the results.

1. Add or update the tag in `INVARIANTS.md`, marking it **broken** with the
   date and the evidence.
2. `grep -l "<tag>" docs/experiments/*.md` and set every affected record's
   verdict to `invalidated`, with `invalidated_by:` pointing at the record that
   found the bug.
3. Update every `REGISTER.md` row whose `depends_on` contains the tag.
4. Write the invariant test **before** fixing the bug, so it fails first.
5. Only then fix it, and queue the re-runs.

Do not skip step 3. Not doing it is the failure that motivated this skill.

## Where this fits with the existing docs

This does not replace them, and should not grow into them:

- `docs/ideas_log.md` — the idea/proposal stage. An idea becomes an `EXP` when
  it is actually run. Its entry format (provenance · mechanism · for/against ·
  evidence status · verdict) is the ancestor of this one; keep using it for
  ideas that have not been tested.
- `docs/*_findings.md`, `reports/*.md` — synthesis and narrative. These cite
  `EXP-####` ids rather than restating numbers, so a later invalidation is
  traceable into them.
- `docs/analytic_descriptors_latent_space_plan_v2.md` §5 (gates G1–G6) — the
  acceptance criteria for a subsystem. A gate is a T2 experiment; reference the
  gate id in `hypothesis`.
- `docs/prediction_difficulty_hypotheses.md` — the current live hypothesis set
  (H-A1…H-C5). `hypothesis:` should name one of these where applicable.
