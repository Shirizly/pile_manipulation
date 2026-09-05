---
# ---- identity -------------------------------------------------------------
id: EXP-0000                     # next free number; zero-padded to 4
title: <one line, states the finding not the activity>
tier: T1                         # T0 | T1 | T2
mode: exploratory                # exploratory | confirmatory (T2 must be confirmatory)
date: 2026-01-01
hypothesis: H-A2                 # id from docs/prediction_difficulty_hypotheses.md, or a gate id (G1..G6), or null

# ---- the claim ------------------------------------------------------------
claim: >
  <One sentence that could be false, with the quantity and the threshold in it.
   "X helps" is not a claim. "X reduces swept-region rms by >10% relative to
   persistence on dataset D" is.>

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
  data_commit: <sha the DATASET was collected under, from its _N_config.yaml
                `provenance:` block; "unrecorded" for datasets predating 2026-09-05>
  script: <path, committed>
  data: ["<glob or dataset config>"]
  code_path: <points_to_mask | PileSweepData raster | particles_to_occupancy | ...>
  seed: 0
  split: "<episode-level 25% by file, seed 0 | LORO 8-fold | ...>"
  runtime: "<wall clock, device>"

budget:                           # declared by the task-giver BEFORE the run
  declared: "<e.g. 30 min, 100k tokens>"
  spent: "<e.g. ~42 min, 178k tokens>"
  outcome: within                 # within | exceeded | stopped-early

design:
  varied:     {<knob>: [<values>]}
  held_fixed: {<knob>: <value>}   # every knob the compared configs share
  baselines:  [persistence, mean-delta]   # non-empty; must include a do-nothing baseline
  metric: "<exact definition, including the region it is computed over>"

noise_floor: "<number + how it was obtained>"   # written before the result

depends_on: [grid-convention]     # tags from docs/experiments/INVARIANTS.md whose failure would change the verdict
establishes: []                   # tags whose STATUS this record sets; exempt from the untested-dependency downgrade

# ---- outcome --------------------------------------------------------------
result: "<one line, the number that answers the claim>"
verdict: inconclusive             # supported | refuted | inconclusive | invalidated
downgrades: []                    # provenance | imprecision | indirectness | inconsistency | selection | untested-dependency | incomplete-design
grade: high                       # COMPUTED: high, minus one level per downgrade
supersedes: []                    # EXP ids this replaces
invalidated_by: null              # EXP id that broke this one's inputs
---

## Why this test discriminates

<Two or three sentences. What would the world look like if the claim were
false, and why does this measurement distinguish that from the claim being
true? If you cannot answer, the prediction block is not ready.>

## What was actually run

<The bits a reader could not reconstruct from `provenance`: preprocessing
order, what was excluded and why, anything that deviated from the plan.
Deviations are fine; unreported deviations are not.>

## Numbers

<A table. Include the baselines in it, not in prose. Include every cell of any
sweep that was run — not the best one.>

## What would change the verdict

<The specific follow-up that would flip it, and its cost. This is what the next
session reads to decide whether re-running is worth it.>

## Threats

<One line per downgrade domain claimed above, saying concretely why. Plus any
threat you considered and dismissed, with the reason — those are the ones that
come back.>

## Unrelated findings

<Things you tripped over that have nothing to do with this claim: a slow
function, a stale doc pointer, a config key that does nothing, a dataset that
is not the size its name implies. One bullet each — what, where, how you know.

DO NOT ACT ON THEM. They are logged for the user to triage; fixing them is how
a 30-minute experiment becomes a three-hour one. If a finding would change this
experiment's verdict it is not unrelated — it belongs in Threats.

Write "none" rather than deleting the heading.>
