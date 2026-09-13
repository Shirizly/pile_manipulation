---
name: experiment-log
description: "Manage experimental research work in this repo: design experiments, record runs and outputs, preserve provenance, analyze results, track claims, and prevent experiment-driven code sprawl. Use before running a probe, fit, sweep, ablation, benchmark, data collection, model test, or MPC study whose outputs may matter; when creating or changing experiment records; and when a bug or convention error could affect existing evidence."
argument-hint: "Optionally: 'new <slug>', 'register', 'test MODEL-xxxx', 'invalidate <tag>', or the claim being investigated"
user-invocable: true
---

# Experiment management

## Philosophy

Organize research around **scientific objects and provenance**, not the
chronology of agent work.

- **Experiment:** the scientific question, scope, design, and interpretation.
- **Run:** one concrete execution of code/configuration.
- **Artifact:** a direct computational output of a run.
- **Result:** an analyzed/derived measurement or presentation product used to
  support a scientific conclusion or report.
- **Report:** interpretation and communication of results.
- **Dataset/model:** reusable scientific objects with identities independent of
  a particular experiment.

Keep scientific records local, but computational capabilities architectural.
Experiments may contain genuinely local code; reusable functionality belongs in
the normal project modules. A new **composition** of existing functionality is
usually experiment-local code. A new **utility/capability** that other work
should use belongs in its owning project module. When uncertain, keep it local
and small rather than prematurely generalizing it.

## Canonical storage

```text
experiments/
  EXP-####-slug/
    EXPERIMENT.md
    config.yaml                 # experiment-wide resolved config, if needed
    runs/
      RUN-####-slug/
        RUN.md
        config.yaml             # resolved run config, when applicable
        COMMAND.txt
        stdout.log / stderr.log
    artifacts/
      RUN-####-slug/            # direct computational outputs
        ...
    results/                    # analyzed evidence used by reports
      RESULTS.md
      metrics.*
      tables/
      figures/
    reports/                    # interpretation/presentation
      REPORT.md
    code/                       # genuinely experiment-local code only

datasets/
  DS-####-slug/
    dataset.yaml / DATASET.md
    config.yaml
    data/

models/
  MODEL-####-slug/
    model.yaml / MODEL.md
    config.yaml
    checkpoint.*
    tests.md                    # short reverse index of tests

configs/                        # reusable templates/schemas only
scripts/                        # thin entry points / validation tools
runs/COMMANDS.jsonl             # optional project-wide execution ledger
```

Do not create experiment-specific top-level trees such as `training/`,
`evaluation/`, `mpc/`, or `analysis/` merely because an experiment contains
those activities. They are **run types**. Their reusable code belongs in the
project architecture; their execution records and outputs belong to the
experiment.

## Configuration

Project-level configs are templates/schemas, not the authoritative record of a
completed instance. Copy the relevant template into the dataset, model,
training, run, or experiment instance and configure it there. The copied,
resolved config is authoritative for that object/run. Record the template
version or commit when useful for provenance.

This is intentional duplication: later edits to a project template must not
silently change the meaning of an old experiment. Keep each config near the
artifacts it governs.

## Artifact vs. result

The boundary should be mechanical:

> **Artifact = what execution produced. Result = what analysis made meaningful.**

### Artifacts

Direct outputs of a run, before scientific aggregation or interpretation:

- raw collected trajectories, recordings, or samples;
- per-sample predictions, losses, or errors;
- raw MPC trajectories;
- benchmark timing/memory dumps;
- training checkpoints;
- execution logs and diagnostic dumps.

Artifacts may be large, numerous, or disposable. Keep them under the
experiment's `artifacts/` area unless promoted to a reusable dataset/model.

### Results

Derived, curated products that answer the scientific question or feed a
report:

- aggregate metrics across samples/episodes/seeds;
- confidence intervals and statistical comparisons;
- benchmark summary tables;
- derived error/scaling curves;
- selected scientific figures;
- compact tables or values that a report actually relies upon.

A useful test is: **could the item be described simply as "the program emitted
this file"?** If yes, it is probably an artifact. If it required calculation,
aggregation, comparison, selection, or other analysis to become scientifically
useful, it is probably a result.

The file type does not decide the category. A diagnostic plot emitted directly
by a run is an artifact; a plot deliberately generated from analyzed metrics
for a report is a result.

Do not copy raw artifacts into `results/` for convenience. Results should be
small and curated; reports should point to them rather than duplicate them.

## Experiment and run records

`EXPERIMENT.md` is the compact scientific dossier. It answers:
**what question are we testing, under what conditions, and what conclusion does
the evidence support?**

A run record answers: **what exactly did we execute?** Capture the resolved
configuration, inputs/IDs, code provenance, exact command, status, and output
locations. A single experiment may contain data collection, training,
evaluation, benchmarks, model tests, and MPC runs when they serve the same
scientific question.

Before a nontrivial outcome-revealing run, record a compact plan:

1. falsifiable claim and scope;
2. prediction with exact observable and threshold/decision rule where useful;
3. varied factors, held-fixed factors, baselines, inputs, and metrics;
4. feasibility/cost and cheapest invalidating check;
5. the result most damaging to the plan and whether the design would catch it.

Exploratory work is legitimate. Mark it exploratory; do not later treat its
unplanned outcome as confirmation without rerunning the discriminating test
with the prediction fixed first.

## Code placement

Use this distinction before creating code:

**New composition:** existing loaders/transforms/models/metrics/MPC components
combined in a new way for one experiment. Put the orchestration in
`experiments/EXP-####/code/` or a thin entry point.

**New utility/capability:** a new reusable transformation, metric, model
component, dataset mechanism, analysis primitive, or subsystem behavior. Put
it in the existing project module that owns that responsibility; add tests/docs
as appropriate.

Search before creating helpers. Do not create `utils.py`, `analysis.py`,
`run.py`, or similar generic dumping grounds inside every experiment.

A local function that becomes useful to multiple experiments should be promoted
to the appropriate project module rather than copied.

Scripts are entry points, not default places for substantial logic. Prefer
scripts that load a config, call project functionality/local orchestration, and
write the run outputs.

## Model-instance test history

A specific trained model should maintain a concise reverse index, e.g.
`models/MODEL-0042/tests.md`:

```text
- held-out sequential prediction → EXP-0027 / RUN-014 → results/metrics.csv
- action-magnitude sensitivity → EXP-0031 / RUN-003 → results/table.csv
- heuristic comparison → EXP-0035 / RUN-008 → results/comparison.csv
```

This is **not** a second evidence store. The experiment/run remains
authoritative. The purpose is to let future work discover what has already been
tested on this exact model and avoid wasteful repetition.

Intermediate training checkpoints that are not promoted to model instances are
ordinary run artifacts.

## Provenance and command logging

Every nontrivial run must be locally reconstructable from its run record and
config. Capture at least:

- run/experiment IDs;
- resolved configuration;
- code commit and dirty state;
- relevant dataset/model IDs;
- exact command/argv;
- start/end/status;
- output/artifact locations.

When `runs/COMMANDS.jsonl` is used, write a start event **before** execution and
an end event on completion. Put `COMMAND.txt` in the run/output directory so
the invocation travels with the artifact.

Never present a reconstructed command as recorded. Label reconstruction as such.
For long jobs, use the repository's run wrapper when available; do not wait on
background watchers instead of reading their output.

## Evidence layer

Keep cross-experiment evidence separate from experiment storage. The project-
level `docs/experiments/` layer is an index of claims and checkable dependencies,
not the primary home of run artifacts/results.

Preserve these evidence distinctions:

- `supported` — evidence supports the claim under its stated conditions;
- `refuted` — the claim fails under its stated conditions;
- `inconclusive` — the design/uncertainty cannot discriminate;
- `invalidated` — a relevant input, implementation, convention, or assumption
  is known broken.

A result changing under a new condition is not automatically invalid. Prefer a
narrower, conditional claim when the original measurement remains valid.

When a bug or convention error affects existing evidence, trace its dependency
and invalidate the affected records rather than silently leaving stale claims.

## Where things belong

| Thing | Canonical home |
|---|---|
| Experiment question/design | `experiments/EXP-####/EXPERIMENT.md` |
| Exact execution/config | `experiments/EXP-####/runs/RUN-####/` |
| Direct run outputs | `experiments/EXP-####/artifacts/` |
| Analyzed evidence | `experiments/EXP-####/results/` |
| Interpretation | `experiments/EXP-####/reports/` |
| Reusable dataset | `datasets/DS-####/` |
| Reusable trained model | `models/MODEL-####/` |
| Model test history | `models/MODEL-####/tests.md` |
| Config template | `configs/` |
| Experiment-local code | `experiments/EXP-####/code/` |
| Reusable code | owning project module |
| Cross-experiment claims/dependencies | `docs/experiments/` |

## Anti-sprawl rules

- Do not make a new module merely because an experiment needs a new
  composition.
- Do not duplicate reusable code in experiment folders.
- Do not use generic experiment-local dumping grounds such as `utils.py` when
  a clearer local name or the existing architecture is appropriate.
- Do not duplicate raw artifacts in results or duplicate numbers in reports.
- Do not let project-level config edits silently alter old instances.
- Do not make model test histories into another results database.
- Prefer a small local solution to an uncertain abstraction; generalize when
  reuse is demonstrated.

## Existing project documentation

Use `project-overview` before touching unfamiliar code. It identifies the
module/doc responsible for major subsystems and defines the documentation
ownership boundaries. Preserve those boundaries when implementing experiment
work.
