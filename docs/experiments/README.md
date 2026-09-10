# docs/experiments — the evidence layer

Numbers live here so they can be **compared** and **invalidated**. Synthesis
still lives in `docs/*_findings.md` and `reports/*.md`, which cite `EXP-####`
ids rather than restating numbers.

| File | Role |
|---|---|
| `REGISTER.md` | one row per claim: status, what supports it, what contradicts it, and what it **depends on** |
| `INVARIANTS.md` | the `depends_on` tag registry — each tag is a property that can be false, with the test that checks it |
| `EXP-####-slug.md` | one record per experiment; the YAML frontmatter is the contract |

## Using it

```bash
python scripts/check_register.py              # validate (also runs in pytest)
python scripts/check_register.py --fix-grades # recompute grades in place
grep -l grid-convention docs/experiments/*.md # what dies if this breaks
```

The workflow, field meanings, tiers and grading rule are in the
`experiment-log` skill (`.claude/skills/experiment-log/SKILL.md`); the record
template is in its `references/`. Start there rather than copying an existing
record, which may predate a rule change.

## State as of 2026-09-10

Reset from scratch. The prior register, invariant registry, and all
`EXP-####` records were archived wholesale to
`archive/2026-09-10_pre-reset/docs/experiments/` — not because any of it was
found wrong, but to start the claim ledger clean rather than carry forward
numbers, ids, and dependency chains from a superseded work program. The old
material is fully intact there if a past claim or invariant needs to be
looked up or revived.

Nothing has been re-verified yet. Treat every invariant as `unchecked` until
it is re-registered in `INVARIANTS.md` with a real test, and every claim as
untested until an `EXP-####` record backs it.
