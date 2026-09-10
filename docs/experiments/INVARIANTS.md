# Invariants — the `depends_on` tag registry

Every tag an experiment record cites must appear here. A tag names a property
the repo's results rest on, and — where one exists — the test that checks it.

**Why this file exists.** Assumptions written as sentences do not fail loudly.
See `.claude/skills/experiment-log/SKILL.md` for the incident history that
motivated this file's fields.

**Status vocabulary:** `holds` (a test asserts it and passes) · `unchecked`
(believed true, nothing enforces it) · `broken` (known false) · `fixed`
(was broken, now checked).

## Reset 2026-09-10

This registry was reset to empty along with `REGISTER.md` and every
`EXP-####` record — archived to
`archive/2026-09-10_pre-reset/docs/experiments/INVARIANTS.md`, not deleted.
Any invariant this project still relies on (grid convention, rasteriser
identity, episode-split, etc.) needs to be re-added below **with its current
status re-verified**, not copied forward assuming it still holds — the point
of a reset is not to inherit stale confidence.

| Tag | Property | Status | Test |
|---|---|---|---|
| _(none yet)_ | | | |

## Adding a tag

1. Name the property as something that can be **false**, not as a topic.
2. Give it a status honestly. `unchecked` is a normal answer.
3. If you are about to rely on it at T2, write the test first.

## The xfail pattern

For a tag that is `broken`, write the test now and mark it
`@pytest.mark.xfail(strict=True, ...)`. It records the defect in the suite
rather than in prose, and `strict=True` means that the moment someone fixes the
bug the test XPASSes and *fails the run*, telling them to drop the marker and
flip this table's status to `fixed`. A broken invariant then cannot be quietly
fixed without the register noticing.

## Broken tags block T2

A T2 record may not cite a tag whose status is `broken` or `unchecked`.
`scripts/check_register.py` enforces this. At T0/T1 it is allowed and costs the
`untested-dependency` downgrade.
