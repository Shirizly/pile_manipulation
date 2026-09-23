# Orchestrator notes — pile_manipulation, 2026-09-13/15

One session, ~30 subagents, 13 experiment records. What the seat actually taught me.

## Delegation is cheap; attribution is not

Spawning agents scaled fine. What did not scale was knowing whether to believe them.
Every substantive result I checked myself, I checked because a claim sounded load-bearing —
and roughly a third of those checks found something: records citing code that lived only in
a gitignored tree, a tracked EXP-0003 artifact silently overwritten, a causal explanation
contradicted by code one grep away. None were malice or even carelessness; agents report
confidently by default. **Budget for verification as a fixed cost of delegation, not an
exception.**

## My own errors outnumbered theirs

- Characterised a 1230-file banded corpus from `glob(...)[0]` and told an agent the wrong range.
- Recorded "every model scores near zero" as a `holds` invariant when every model tested was
  one model *class*; the first image-space model scored 0.39–0.87.
- Relayed an agent's mechanism claim without grepping the function that refuted it.
- Launched two agents into one working directory and caused the collision that overwrote the
  artifact above.

Three are now rules in `subagent-experimenter`. The pattern in all four: **I asserted across a
population or a mechanism I had sampled once.**

## What worked

- **Audit before promote.** Running the audit as a gate, not a follow-up, killed a false
  explanation before it entered the ledger.
- **Pre-declaring the expected result.** Telling an agent "near-zero is expected here, don't
  debug it" saved budget — and when the expectation was wrong, the agent reported the
  contradiction anyway. Framing did not suppress the finding.
- **Demanding both directions.** "This makes claim A stronger and claim B weaker; both must be
  visible" reliably produced honest writeups.
- **Controls chosen to discriminate**, not to decorate: the pooled-occupancy control that showed
  localisation-in-general bought nothing; the dim-matched control that exposed dilution.
- **Aggregating transcripts instead of reading them.** 26 agents summarised via a script, never
  into context.

## What did not work

Prose rules did not change behaviour. "Never park on a background watcher" was in every brief
and in a loaded skill; it broke six times. Scope instructions broke once, destructively.
Budgets overran on most runs. These need mechanism — a blocking wrapper, a pre-write path
check, a hard cap — not better wording. Documentation fixed *navigation* (the code map was
used and improved itself through agent feedback); it did not fix *conduct*.

## The metric lesson

The user's standing judgement — regret over accuracy — was vindicated repeatedly. Accuracy
ranked the same models three different ways under three defensible definitions, and a model
scoring exactly 0.000 ranked actions usefully. Lead with the metric that decides; treat the
convenient one as a diagnostic.
