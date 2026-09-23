---
name: experiment-log
description: "Record an experiment, a hypothesis test, or a measured result in this repo so it can be compared with other evidence later and invalidated cleanly if an assumption breaks. Use when running any probe, fit, sweep, ablation, benchmark, data collection, model test, or MPC study whose outputs may matter; when creating or changing experiment records; when stating or revising a claim; when a result contradicts an earlier one; and ALWAYS when a bug, convention error, or code-path difference is found that could affect results already recorded. Also use before designing an experiment, to check the claim is falsifiable and the test discriminates."
argument-hint: "Optionally: 'new <slug>', 'register', 'test MODEL-xxxx', 'invalidate <tag>', or the claim being investigated"
user-invocable: true
---

# Experiment log — recording evidence so it composes

## Why frontmatter, not prose

This repo's docs used to be precise and caveated, and that was not enough:
real numbers were once produced by two mutually-transposed data channels, by
two different rasterisers feeding one comparison, by a view compared at one
blur level against another at a different one. Prose warnings did not stop
any of these; a validator reading actual fields would have. The full
incident list that justifies each field below lives in
`references/why-these-fields.md` — read it once, then work from the fields
themselves.

**The reader is the next session, which has no memory.** Write frontmatter a
script can parse and a body someone can skim in 30 seconds. Do not write
essays here — narrative synthesis belongs in a report or design doc, which
should cite `EXP-####` ids rather than restate numbers (see "Cross-experiment
synthesis," below).

## Philosophy

Organize research around **scientific objects and provenance**, not the
chronology of agent work.

- **Experiment:** the scientific question, scope, design, and interpretation.
- **Run:** one concrete execution of code/configuration.
- **Artifact:** a direct computational output of a run — distinct from an
  *epistemic* artifact (a measurement that turned out to be wrong, e.g. from
  a confound or a bug). When that distinction matters, call the latter a
  **confound**, not an artifact, so the two senses don't collide.
- **Result:** an analyzed/derived measurement or presentation product used to
  support a scientific conclusion or report.
- **Report:** interpretation and communication of results.
- **Dataset/model:** reusable scientific objects with identities independent
  of a particular experiment.

Experiment-local vs. reusable code is `project-overview`'s call to make, not
this skill's — in short, a new **composition** of existing capability is
usually local, a new **capability** belongs in its owning project module,
and neither is ever a reason to skip recording the experiment itself.

## Canonical storage

One root, `experiments/`, holds both the cross-experiment claim ledger and
every individual experiment's storage.

```text
experiments/
  REGISTER.md           # one row per claim — schema owned by register-validator
  INVARIANTS.md         # the depends_on tag registry
  METRICS.md            # design.metric key -> exact formula (accuracy, slateN, ...)
  COMMANDS.jsonl         # project-wide execution ledger, every run, filed or not
  TEMP_LOG.md           # one line per experiments/temp/ dir, written when it's CREATED
  temp/                  # scratch, BELOW T0 — see "Quick, uncited work" below
  EXP-0001-slug/
    EXPERIMENT.md        # frontmatter contract; see references/experiment-template.md
    config.yaml          # experiment-wide resolved config, if needed
    runs/
      RUN-0001-slug/
        RUN.md           # what exactly ran — see "Experiment and run records"
        config.yaml
        COMMAND.txt
        stdout.log / stderr.log
    artifacts/
      RUN-0001-slug/      # direct computational outputs, keyed by run so this
    results/              # tree can be pruned/ignored as a unit without
      RESULTS.md           # touching runs/
      metrics.*
      tables/
      figures/
    reports/
      REPORT.md
    code/                 # genuinely experiment-local code only

datasets/
  DS-0001-slug/
    DATASET.md            # narrative + status — see "IDs outlive their payload"
    config.yaml
    data/

weights/                  # reusable, resolved FITTED OBJECTS — network
  MODEL-0001-slug/          # checkpoints, a fitted linear operator, a heuristic
    MODEL.md                 # model's parameters. Not "neural-network weights"
    config.yaml               # specifically, and not to be confused with
    checkpoint.*              # model/, which is code, not instances
    tests.md                # short reverse index of what's been tested

configs/                  # reusable templates/schemas only
scripts/                  # thin entry points / validation tools
```

Directory names carry a `-slug`; a citation or cross-reference uses the bare
zero-padded ID (`EXP-0007`, `RUN-0003`, `DS-0012`, `MODEL-0042`) — never a
path, so a record survives its payload being renamed, moved, or deleted (see
"IDs outlive their payload").

**What's tracked, what's not.** `*.md`, `config.yaml`, and small curated
`tables/`/`figures/` are tracked. `data/`, `checkpoint.*`, `artifacts/`, and
`*.log` are gitignored — every ignored payload must be regenerable from its
tracked record (resolved config + exact command). `experiments/temp/` is
gitignored in full (`/experiments/temp/` in `.gitignore`, anchored so it
doesn't also catch an unrelated `temp/` elsewhere in the tree);
`experiments/TEMP_LOG.md` is a tracked sibling, not a child, so it survives
the directory being wiped.

`Baselines/<Name>/` mirrors this split internally: baseline-specific code
stays in `Baselines/<Name>/`, and its trained instances live in
`Baselines/<Name>/weights/MODEL-####-slug/` — the same instance structure as
top-level `weights/`, just nested under the baseline that produced it. Give
a baseline checkpoint the same `MODEL.md`/`tests.md` treatment the moment it
stops being purely internal to that baseline's own development (i.e. the
moment it's compared against anything else).

Do not create experiment-specific top-level trees such as `training/`,
`evaluation/`, `mpc/`, or `analysis/` merely because an experiment contains
those activities — they are **run types**; reusable code for them belongs in
the project architecture, execution records and outputs belong to the
experiment.

## IDs: allocation, citation, and outliving their payload

Next free number per prefix, zero-padded to 4, **never reused** — a deleted
`DS-0012` stays retired, it does not become a different dataset later.
`RUN-####` is scoped to its experiment, not global.

Datasets and trained instances are large and will eventually be deleted to
reclaim disk while records still cite them. When that happens, the
`DATASET.md`/`MODEL.md` file stays: set `status: deleted` and record the
regeneration command, so `DS-0007` always resolves to *something* even if
the payload is gone from disk. This is the reason records cite IDs, not
paths.

## Quick, uncited work: `experiments/temp/`

Below T0, for a check that doesn't yet deserve a tracked file: work in
`experiments/temp/<slug>/` — no ID, no fixed structure, whatever files the
check needs. **The moment you create the directory**, append one line to
`experiments/TEMP_LOG.md` (tracked, unlike `temp/` itself): slug, date, one
line on what you're checking. Update that line with the outcome once you
know it — dead end, or promoted.

**Promote before the number leaves the directory** — before it's pasted into
a summary, a report, a commit message, or a `REGISTER.md` row. Promoting
means: allocate a real `EXP-####`, move the content into
`experiments/EXP-####-slug/`, write it up to the tier the finding deserves
(below), and update the `TEMP_LOG.md` line to point at the new id.

**Keep every fitted object you produce, even in `temp/`.** A fitted
operator, a trained checkpoint, an encoder — save it to disk beside its
metrics, with the resolved config that produced it, the moment the fit
finishes. Metrics alone are not enough: a later test (a control evaluation,
an ablation, a re-score under a different metric) then has to refit the same
model from scratch, and a refit is only *probably* the same object. This
costs disk, which is cheap, and buys the ability to test a model again
without re-deriving it, which is not.

`experiments/temp/` is scratch in the sense that nothing there is *cited* —
not in the sense that it is disposable on sight. Clearing it, fitted objects
included, is the **user's decision**, taken when a line of inquiry has proved
unproductive and the whole directory goes at once. Do not delete another
agent's or an earlier session's temp work to reclaim space on your own
judgement. `TEMP_LOG.md`, written at creation, is what survives a dead end
being forgotten rather than closed out.

A fitted object that outlives its line of inquiry — one that something else
will be compared against — stops being scratch and belongs in
`weights/MODEL-####-slug/` with its own `MODEL.md`, per "Where things
belong" below.

## Tiers — pick one before running anything

**Load `register-validator` before writing an `EXPERIMENT.md`'s
frontmatter** — this section says which tier to pick; the exact required
fields and how they're checked are that skill's job.

- **T0 probe.** A real, tracked `experiments/EXP-####-slug/EXPERIMENT.md`,
  frontmatter only, no body. Should take under a minute to write; if it
  doesn't, the tier is wrong. This is the anti-ceremony guard — most
  day-to-day checks that are worth a citable id at all are T0, not T1.
- **T1 experiment.** Full record. Use when a number will be cited in a report
  or doc, or compared against another experiment.
- **T2 gate.** T1 plus: the `prediction` committed to git before the run;
  the full multiverse sweep reported, not the winning cell; and every
  `depends_on` tag actually `holds`/`fixed` (a tag the record itself
  `establishes` is exempt — the record that discovers a broken invariant is
  the evidence, not a victim of it). Use for a claim something else will
  lean on — a headline result, a go/no-go, or anything contradicting an
  existing register row.

When in doubt, use T1 — escalating later is cheap, a T0 number that ends up
in a report is not. **An `EXPERIMENT.md` is not done until
`python scripts/check_register.py` exits 0.**

Pure data collection or a training run with no claim being tested doesn't
need a tier at all — record it as a `DATASET.md`/`MODEL.md` plus a `RUN.md`.
A tier applies once the run is testing something.

## Before you run anything: the plan gate

At T1 and T2, write **≤200 words** and stop, before touching data:

1. the claim, as one sentence that could be false, **naming the conditions
   it's claimed under** (dataset, metric, the design knobs the result could
   plausibly depend on) — an unscoped claim that later needs narrowing reads
   as "refuted" for no reason but its own over-reach;
2. the prediction — `supports` / `refutes`, with the exact quantity and its
   normalisation named, not just a direction (a raw score, a score minus a
   baseline's, and a score against a usable-skill threshold can disagree in
   shape; naming only the direction lets two readers reach opposite verdicts
   from the same numbers). `discriminating: false` means stop and redesign,
   not run;
3. the design — what varies, what is held fixed, what the baselines are (a
   "do nothing" baseline is mandatory — see `register-validator`);
4. **the cost estimate**, and the cheapest check that could invalidate the
   whole plan;
5. the one result that would most embarrass the plan, and whether the design
   would catch it.

Run item 4 first — a two-minute feasibility check that saves an hour is
always in budget. A **cost pilot** (a short run measuring only runtime,
memory, or feasibility) doesn't compromise the prediction and needs no
disclosure; a pilot that reveals any *outcome* does — either discard those
cells and re-run after writing the prediction, or keep them and set
`mode: exploratory`, and say which in "What was actually run."

Keep the plan; it becomes the record's frontmatter almost verbatim.

## Budgets

**Every experiment handed to an agent gets a declared budget, in the prompt,
before any work starts** — in **wall-clock and tokens, never tool calls** (a
tool-call cap punishes the cheap `grep` or 3-line probe that makes a result
trustworthy, and rewards one big unverified run).

| tier | typical budget | if it looks bigger than this |
|---|---|---|
| T0 probe | ~5 min, ~20k tokens | it is not a T0 |
| T1 experiment | ~20–40 min, ~80–120k tokens | split it, or say which cells are optional |
| T2 gate | ~1–2 h, ~250k tokens | it needs its own plan, not a prompt |

Record `budget: {declared, spent, outcome: within|exceeded|stopped-early}`.
When it runs out: stop and report what you have. Name any cell you couldn't
run, take `incomplete-design`, and put it in "What would change the verdict"
with its cost. **If the missing cell is the one that would discriminate, the
verdict is `inconclusive`, not `supported`** — a design with its decisive
cell missing has tested nothing, however many other cells ran. Exceeding a
budget is allowed and gets recorded, not hidden; repeatedly exceeding it
means the tier or the sizing is wrong.

## Metrics

Report the two standard metrics, **`accuracy`** (image prediction) and
**`slateN`** (control), in every record — both go up when better.

**They are not equally trustworthy, and `slateN` is the one that decides.**
`accuracy` is usable for comparing variants *within* one model type — a
parameter sweep, a small design change — and is suspect everywhere else;
across model types it has been observed to rank models in an order `slateN`
does not reproduce, and to shift its own ranking under re-definitions of the
same quantity. Treat it as a diagnostic, not a verdict. Concretely:

- **A model comparison carrying only `accuracy` is not finished.** Repeat it
  under `slateN` before drawing a conclusion from it or reporting it as one.
- **Lead every summary with `slateN`**, and report `accuracy` beside it as
  additional, explicitly-flagged-as-suspect information.
- **A descriptor-space accuracy is comparable only within one descriptor set.** Two
  models predicting different descriptor vectors are scored on different quantities;
  ranking them that way rewards the set that is easier to predict. Within a shared
  set (same definition, normalisation and held-out rows) it is a valid axis, and the
  only prediction axis available when neither model can reconstruct an image.
- `slateN` itself is strongest with breadth — more validation pools, more
  than one goal, and every non-degenerate variation available — because its
  known weakness is power, not bias (see `experiments/METRICS.md` on ties
  and effective sample size). A `slateN` run on a single pool with a single
  goal is a weak `slateN` run; widen it before leaning on it. Where the
metric under study is something else, report these two as reference rows
anyway, so results stay comparable across the register. Any other
`design.metric` value must be a key defined in `experiments/METRICS.md` with
its exact formula, not a prose description ("swept-region rms" doesn't say
whether it's a ratio of means or a mean of ratios) — add the metric there,
with its formula, before citing its key.

Before trusting a rising curve, check whether it's bounded (FSS, IoU, SSIM,
R² converge to a ceiling as the comparison gets easier). A raw curve rises
trivially near that ceiling, and subtracting a baseline score from a bounded
score forces the difference toward zero for the same reason — neither is
evidence of a trend. Report a reference level with meaning instead (FSS's
useful-skill threshold, a shuffled-label floor for R², the value a null
model attains for IoU) and the scale at which the model crosses it.

## Configuration

Project-level configs are templates/schemas, not the authoritative record of
a completed instance (see `project-overview` for the full rule). Copy the
relevant template into the dataset/model/run/experiment instance, resolve it
there, and record which template version it came from
(`template: <path>@<sha>` in the instance's `config.yaml`) — this is
intentional duplication, so a later template edit never silently changes the
meaning of an old experiment.

## Artifact vs. result

The boundary is mechanical:

> **Artifact = what execution produced. Result = what analysis made meaningful.**

**Artifacts** are direct, pre-interpretation outputs of a run: raw collected
trajectories, per-sample predictions/losses, raw MPC trajectories, benchmark
timing/memory dumps, training checkpoints, execution logs. They may be large
and disposable, and live under `artifacts/RUN-####/` — kept as a tree
parallel to (not nested under) `runs/RUN-####/`, so the payload, which is
often large and always regenerable, can be pruned or gitignored as a unit
without touching the small, tracked run record.

**Results** are derived, curated products that answer the scientific
question or feed a report: aggregate metrics, confidence intervals and
statistical comparisons, benchmark summary tables, derived curves, selected
figures.

A useful test: **could the item be described as "the program emitted this
file"?** If yes, it's an artifact. If it took calculation, aggregation,
comparison, or selection to become scientifically useful, it's a result.
File type doesn't decide the category — a diagnostic plot emitted directly
by a run is an artifact; the same plot regenerated from analyzed metrics for
a report is a result. Don't copy raw artifacts into `results/` for
convenience; reports should point at results, not duplicate numbers from
them.

## Experiment and run records

`EXPERIMENT.md` is the compact scientific dossier: what question, under what
conditions, what conclusion does the evidence support — see
`references/experiment-template.md` for the exact frontmatter and
`register-validator` for what's enforced.

`RUN.md` answers what exactly was executed for one run within the
experiment: resolved configuration, code commit and dirty state, relevant
dataset/model IDs, exact command/argv, start/end/status, output/artifact
location. It is prose plus those fields, not a separately validated schema.
An experiment that spans several runs (e.g. data collection, training, and
evaluation together) names them in `EXPERIMENT.md`'s
`provenance.runs: [RUN-0001, ...]` and treats each `RUN.md` as authoritative
for its own details; for a single-run experiment these collapse to the same
information stated once.

`DATASET.md`/`MODEL.md` are the equivalent dossier for a reusable object:
what it is, how it was produced (the same provenance fields), and its
`status` (`active` / `deleted` — see "IDs outlive their payload").

One experiment may contain data collection, training, evaluation,
benchmarks, model tests, and MPC runs together, when they serve the same
question.

## Provenance and command logging

Every run is locally reconstructable from its `RUN.md` and config: run/
experiment IDs, resolved configuration, code commit and dirty state (commit
*before* you run, so `dirty` is false and the sha means something), relevant
dataset/model IDs, exact command/argv, start/end/status, output locations.

**Use `scripts/run_probe.py` for anything long, and route quick analysis
through it too — not just long jobs.** A 25-second re-score whose `--goals`
decided the result is exactly the case this exists for. It forces
`python -u` (Python fully buffers stdout when redirected to a file, so a
long job looks frozen for its entire run and you can't tell slow from
hung), redirects straight to a file instead of a pipe, records the PID so a
job is stopped by PID rather than a `pkill` pattern broad enough to match
the killer, caps threads, and writes `utils.git_provenance()` beside the log
at run time. Separately: cap your own threads (`OMP_NUM_THREADS=4`) and
check `uptime` before starting anything — this is a shared machine, and four
concurrent BLAS jobs once took the load average to 37 on 20 cores and
starved another agent's entire budget.

`experiments/COMMANDS.jsonl` is the project-wide ledger, mandatory for every
run whether filed or in `temp/` — write a start event **before** the process
begins and an end event on completion/failure/kill, joined on `run_id`; a
start with no end means the job was interrupted, which is information, not
an error. Drop `COMMAND.txt` (same argv + commit sha) in the run's own
output directory too, so the invocation travels with the artifact if the
directory is later copied or moved. `run_probe.py` writes both for you.

Never present a reconstructed command as recorded — label reconstruction as
such. Never wait on a background watcher: if you launch a long job in the
background, do not poll it or set up a monitor and return — either read
whatever its output file already holds and write up what's there, or run it
in the foreground with a bounded timeout and a small enough configuration to
finish. An unwritten result is worth nothing; a partial one is worth a lot.

## Model-instance test history

A specific trained instance maintains a concise reverse index,
`weights/MODEL-0042/tests.md` (cite ids, not paths, for the same reason
records cite `DS-####`/`MODEL-####` rather than a `data/` path):

```text
- held-out sequential prediction → EXP-0027 / RUN-0014 → results
- action-magnitude sensitivity → EXP-0031 / RUN-0003 → results
- heuristic comparison → EXP-0035 / RUN-0008 → results
```

This is **not** a second evidence store — the experiment/run remains
authoritative; it exists so future work can discover what's already been
tested on this exact instance and avoid repeating it. Intermediate training
checkpoints not promoted to their own instance are ordinary run artifacts.

## Evidence layer

`experiments/REGISTER.md` and `experiments/INVARIANTS.md` are the
cross-experiment index — not the primary home of run artifacts or results,
which stay under each `EXP-####-slug/`. Row/column schemas and structural
checks are `register-validator`'s territory.

Claim status vocabulary (`REGISTER.md`): `supported` · `refuted` (fails
**under its own stated conditions** — a claim that never stated its
conditions and later fails outside them is `narrowed`, not `refuted`, and
should not be rescued after the fact by reinterpreting it as conditional
either: fix the scope at claim-writing time, above, instead) · `open`
(stated, under test) · `narrowed` (the measurement stands and reproduces; a
newly tested condition changes the result, so it's narrower than the claim
asserted) · `contested` (evidence genuinely conflicts **under matched
conditions**, with no condition explaining the difference) · `invalidated`
(an input is now known broken; the measurement stands, the conclusion does
not) · `superseded` · `withdrawn`.

**Reserve "artifact" — in the epistemic sense (see Philosophy) — for a
measurement that was actually wrong.** A result that holds under one
condition and not another is not that: it's a real effect with a governing
condition, and naming the condition ("holds under a coarse objective, not a
selective one") is usually the most valuable thing the experiment produced.

Record-level `verdict` (`supported`/`refuted`/`inconclusive`/`invalidated`)
and `grade` are narrower than the claim-status vocabulary above, and `grade`
is computed, not chosen — see `register-validator` for the mechanics. The
grade letter saturates at three downgrade domains, so it cannot show a claim
getting *better*; when reporting progress, cite the **domains retired**, not
the letter ("provenance retired, three remain" says what "still very-low"
hides). Expect many records to grade low while `INVARIANTS.md` tags sit
`unchecked` — that is accurate information about the repo's evidential
state, not a flaw in the scale, and it's the pressure that gets the backing
tests written.

When a bug or convention error affects existing evidence, trace its
`depends_on` dependents (`grep -l <tag> experiments/*/EXPERIMENT.md`) and
mark them `invalidated` rather than silently leaving stale claims — see
`register-validator` for the exact procedure. **Commit the fix as soon as
it's plausible, even tentatively** (mark it tentative in the message): every
experiment run against an uncommitted fix records a sha that doesn't
describe the code that ran, and a tentative commit later reverted costs
nothing next to a week of unreconstructable results.

## Cross-experiment synthesis

A doc that synthesizes findings across multiple experiments (a design doc, a
findings writeup) cites `EXP-####` ids rather than restating their numbers,
so a later invalidation stays traceable into it. Such a doc is not itself
part of `experiments/` — it lives wherever `project-overview`'s doc map says
synthesis of that kind belongs.

## Where things belong

| Thing | Canonical home |
|---|---|
| Experiment question/design | `experiments/EXP-####/EXPERIMENT.md` |
| Exact execution/config | `experiments/EXP-####/runs/RUN-####/` |
| Direct run outputs | `experiments/EXP-####/artifacts/RUN-####/` |
| Analyzed evidence | `experiments/EXP-####/results/` |
| Interpretation | `experiments/EXP-####/reports/` |
| Below-T0 scratch | `experiments/temp/<slug>/` + a `TEMP_LOG.md` line |
| Reusable dataset | `datasets/DS-####/` |
| Reusable trained instance | `weights/MODEL-####/` (or `Baselines/<Name>/weights/MODEL-####/`) |
| Instance test history | `weights/MODEL-####/tests.md` |
| Config template | `configs/` |
| Experiment-local code | `experiments/EXP-####/code/` |
| Reusable code | owning project module — see `project-overview` |
| Cross-experiment claims/dependencies | `experiments/REGISTER.md`, `experiments/INVARIANTS.md` |
| Metric key definitions | `experiments/METRICS.md` |
| Execution ledger | `experiments/COMMANDS.jsonl` |
| Validator rules for the above | `register-validator` skill |

## Anti-sprawl rules specific to experiment storage

- Do not duplicate raw artifacts in `results/`, or duplicate numbers from
  results in a report.
- Do not let `experiments/temp/` become a permanent home for a cited number
  — promote it first. (Keeping a fitted *object* there is fine and expected;
  it is the cited *number* that must move.)
- Do not report only metrics for a fit whose fitted object you discarded —
  the next test then cannot reuse the model, only approximate it.
- Do not make a model/dataset test history into a second results database.
- Do not let a project-level config edit silently alter an old instance.

Code- and module-level anti-sprawl (composition vs. reusable capability,
avoiding `utils.py`-style dumping grounds) is `project-overview`'s territory,
not this skill's — see there.

## Existing project documentation

**Before searching for code, read `docs/CODEMAP.md`** — an index of which
function in which file does what. An audit of 26 subagent runs found 74% of all
tool calls were Bash and most were navigation (216 `grep`, 142 `sed -n`, 125
`cat`, 94 `ls`, 78 `find`, against 180 `python`), with ten agents each
independently rediscovering the same module. Add to the index when you find
something it lacks.

**Use `subagent-experimenter` when running or delegating a single experiment** —
it owns the execution rules (bounded foreground waits, `python -u`, persisting
fitted objects) and the measurement traps this repo has already hit.

Use `project-overview` before touching unfamiliar code or deciding where a
code/config change belongs. Use `register-validator` when writing an
`EXPERIMENT.md`'s frontmatter, when `check_register.py` fails, or when
adding/retiring an `INVARIANTS.md` tag.
