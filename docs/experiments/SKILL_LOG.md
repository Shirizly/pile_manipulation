# Skill performance log — `experiment-log`

Each row is one subagent run: the skill variant it was given, the task, the
budget declared up front, what it actually spent, and how the record it
produced held up. The point is to judge **whether a change to the skill made
agents more efficient or more honest**, not to judge the science.

## Read this before drawing conclusions from the table

**Attribution here is weak by construction, and the table must not be read as
if it were a controlled experiment.** Each variant is run on a *different*
task, so variant and task difficulty are confounded, n = 1 per cell. This is
exactly the error the skill exists to prevent, committed knowingly because
running the same task twice would burn a research slot to buy one bit.

What the table can honestly support: **existence claims** ("v2 produced a
feasibility estimate before running; v1 did not") and **large, mechanical
differences** (budget overrun by 3x vs 1.1x). What it cannot support: small
differences in record quality, or any claim that one variant is "better".

To upgrade this to real evidence, run one task under both variants and compare.
Budget: one extra research slot per comparison.

## Variants

| Variant | Date | Change from the previous variant | Rationale |
|---|---|---|---|
| **v0** | 2026-09-03 | initial: tiers, computed grades, `depends_on`, invalidation procedure | — |
| **v1** | 2026-09-05 | (a) **declared budgets** in wall-clock and tokens, with an `incomplete-design` downgrade for cells the budget cut; (b) required **`## Unrelated findings`** section; (c) cost-pilot rule clarified for T1 sequencing | v0's one observed run overran ~40 min with no budget stated, and surfaced a real finding (`fit_operator_nonneg` is O(D³)) that had nowhere to live |
| **v2** | 2026-09-05 | (a) **prediction must name the exact quantity**, not just a direction, and must state the scale/normalisation it is measured on; (b) **bounded-metric warning**: any score with a ceiling compresses differences near it, so a difference of two such scores has a forced trend — say which artifact you ruled out; (c) **pre-flight plan gate**: ≤200 words of claim/prediction/design/cost before running anything | R2 refuted its prediction on an instrument with a built-in artifact, and the prediction was ambiguous enough that two readings of it disagreed. Both are cheap to prevent in the prompt-facing part of the skill |

## Runs

| # | Variant | Task | Budget declared | Spent | Verdict/grade produced | Notes |
|---|---|---|---|---|---|---|
| R1 | v0 | resolve C-021 (blur sign flip: dataset vs estimator) | **none** | ~42 min, 178k tok | `supported` / `very-low` | Designed before running; caught a broken invariant unprompted; hit a compute wall and reported the missing cell honestly. No budget to compare against, and the missing cell had no downgrade domain to land in — both fixed in v1. |
| R2 | v1 | FSS + scale-decomposed error on the constrained domain (P1 in `ideas_log_signal_vs_detail.md`) | 35 min, 120k tok | **~30 min, 98k tok — within** | `refuted` / `low`, corrected on review to `very-low` | **Budget rule worked**: stayed inside, ran the optional σ=1 arm, did not drop cells. **Unrelated-findings rule worked**: found `scripts/probes/regimes.py` has a dead import, logged it, did not fix it. **What it missed**: its deconfounding measure (FSS difference) has a forced decline because both terms converge to 1 — it identified this failure mode *in the abstract* in its own feedback and did not apply it to its own instrument. It also did not compute the standard usable-skill threshold, which reverses the conclusion's reason. Reviewer added an amendment and a downgrade. |
| R3 | v2 | synthetic-degradation study: does rms rank error *types* the way realised control utility does? (P2/P3) | 40 min, 130k tok | **~45 min, 117k tok — exceeded, disclosed** | `refuted` / `very-low` | **Plan gate worked, and visibly**: it wrote the "most embarrassing result" line before running and then hit exactly that outcome, which it reported without rationalising. **Bounded-metric section worked in an unexpected direction**: it used it to explain why FSS *lost*, rather than to defend FSS. **Budget rule worked**: exceeded and said so in the frontmatter rather than hiding it. **Two failures**: (1) it allocated a claim id already in use and overwrote another row — the validator now checks for duplicates; (2) it drew its headline conclusion from a pooled rank correlation over 14 heterogeneous models, which by construction cannot see the per-type dissociation its own table contains. The reviewer had to re-read the table to find the main result. |

## Candidate v3 changes, from R3

- **Claim-id allocation** needs a rule (grep the register, take max+1, and
  re-check immediately before writing). Now partly enforced by the validator's
  duplicate check, but the record still overwrote a row it did not own — a
  "never edit a row you did not create" rule would be stronger.
- **Pooled statistics over heterogeneous conditions** deserve their own warning
  next to the bounded-metric one: a rank correlation across error *types*
  averages away type-specific effects, which are often the thing under test.
  Compare like with like at matched error magnitude.
- **Writeup time, not compute time, is the binding constraint at T1** — R3's
  own observation, and it is right. The budget section should say so and split
  the declared budget into compute and writeup.

## What to look for

- **Budget realism.** Is the declared budget systematically too small? That is
  a fact about the task-giver's estimation, not about the agent.
- **Did the agent stop, or push through?** v1 says stop and report. An agent
  that quietly runs 3x over has not followed the skill, and that is the single
  most important thing this log measures.
- **Were missing cells declared?** `incomplete-design` should appear whenever
  the budget cut something. Its absence in an over-budget run is a red flag.
- **Unrelated findings.** Did the agent log them and *not* act on them? Acting
  on them is the main way budgets get blown.
- **Did the record need fixing afterwards?** Every correction the reviewer had
  to make is a gap in the skill, and belongs in the next variant.
