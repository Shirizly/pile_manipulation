# experiments/EXP-####-slug/EXPERIMENT.md — frontmatter contract

This is the canonical field list. The `experiment-log` skill explains the
*workflow* (tiers, when to write a plan, artifact vs. result); the
`register-validator` skill explains what's *enforced* and how grade is
computed. This file is the schema itself — copy it for a new record rather
than an existing one, which may predate a rule change.

```yaml
---
# ---- identity -------------------------------------------------------------
id: EXP-0000                     # next free number; zero-padded to 4; never reused
title: <one line, states the finding not the activity>
tier: T1                         # T0 | T1 | T2
mode: exploratory                # exploratory | confirmatory (T2 must be confirmatory)
date: 2026-01-01
hypothesis: null                 # id from a live hypothesis-tracking doc, if this
                                  # project has one, or a gate id, or null — leave
                                  # null rather than inventing a doc that doesn't exist

# ---- the claim ------------------------------------------------------------
claim: >
  <One sentence that could be false, with the quantity and the threshold in it,
   AND the conditions it's claimed under (dataset, objective/metric, the
   design knobs the result could plausibly depend on). "X helps" is not a
   claim. "X reduces swept-region rms by >10% relative to persistence on
   dataset D" is. Without the conditions, a later result that merely narrows
   the claim's scope reads as refuting it instead.>

prediction:                      # REQUIRED at T2, and must be committed BEFORE the run
  supports: "<observable outcome with a threshold>"
  refutes:  "<observable outcome with a threshold>"
  discriminating: true           # false => the design cannot separate the branches. Redesign, do not run.

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: <git sha at run time — `python -c "import utils; print(utils.git_provenance())"`>
  dirty: false                    # was the working tree modified when this ran?
                                  # A sha alone does NOT reconstruct a run if it was.
                                  # If true, say in "What was actually run" what was uncommitted.
                                  # Commit BEFORE you run, so dirty is false and the sha means something.
  data_commit: <sha the DATASET was collected under, from its own config's
                `provenance:` block; "unrecorded" if predating that convention>
  script: <path, committed>
  data: ["<glob or dataset config, or a DS-#### id>"]
  code_path: <which implementation produced these numbers, if more than one exists>
  seed: 0
  split: "<episode-level 25% by file, seed 0 | LORO 8-fold | ...>"
  runtime: "<wall clock, device>"
  runs: []                        # RUN-#### ids this record's headline number came
                                  # from, for a multi-run experiment; each RUN.md is
                                  # authoritative for its own detail. Omit or leave
                                  # empty for a single-run experiment (this block IS
                                  # that run's provenance in that case).

budget:                           # declared by the task-giver BEFORE the run
  declared: "<e.g. 30 min, 100k tokens>"
  spent: "<e.g. ~42 min, 178k tokens>"
  outcome: within                 # within | exceeded | stopped-early

design:
  varied:     {<knob>: [<values>]}
  held_fixed: {<knob>: <value>}   # every knob the compared configs share
  baselines:  [persistence, mean-delta]   # non-empty; must include a do-nothing baseline
  metric: "<key from experiments/METRICS.md — not a prose description>"

noise_floor: "<number + how it was obtained>"   # written before the result

depends_on: [some-invariant-tag]  # tags from experiments/INVARIANTS.md whose failure
                                  # would change the verdict. Be generous — the cost
                                  # of citing one is a word, the payoff is automatic
                                  # invalidation later.
establishes: []                   # tags whose STATUS this record sets, as opposed to
                                  # relies on. Use sparingly — one record per tag.

# ---- outcome --------------------------------------------------------------
result: "<one line, the number that answers the claim>"
verdict: inconclusive             # supported | refuted | inconclusive | invalidated
downgrades: []                    # provenance | imprecision | indirectness | inconsistency | selection | untested-dependency | incomplete-design
grade: high                       # COMPUTED — see register-validator
supersedes: []                    # EXP ids this replaces
invalidated_by: null              # EXP id that broke this one's inputs
---
```

## Body sections (T1/T2)

### Why this test discriminates

Two or three sentences: what would the world look like if the claim were
false, and why does this measurement distinguish that from the claim being
true? If you cannot answer, the prediction block is not ready.

### What was actually run

The bits a reader could not reconstruct from `provenance`: preprocessing
order, what was excluded and why, anything that deviated from the plan.
Deviations are fine; unreported deviations are not.

### Numbers

A table. Include the baselines in it, not in prose. Include every cell of
any sweep that was run — not the best one.

### What would change the verdict

The specific follow-up that would flip it, and its cost.

### Threats

One line per downgrade domain claimed above, saying concretely why. Plus any
threat you considered and dismissed, with the reason.

### Unrelated findings

Things you tripped over that have nothing to do with this claim — a slow
function, a stale doc pointer, a config key that does nothing, a dataset
that isn't the size its name implies. One bullet each: what, where, how you
know.

**Do not act on them.** They're logged for the user to triage; fixing them
is how a 30-minute experiment becomes a three-hour one. If a finding would
change this experiment's verdict, it is not unrelated — it belongs in
Threats. Write "none" rather than deleting the heading — an empty section is
information, a missing one is ambiguous (and `register-validator` treats a
missing heading as an error at T1/T2).
