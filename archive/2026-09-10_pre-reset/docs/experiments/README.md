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

## State as of 2026-09-03

Backfilled from the linear-foresight work and, at the time, the MPM sand work.
The sand arm was withdrawn as non-physical on 2026-09-05
(`docs/rejected_mpm_sand.md`); its evidence is marked `invalidated` in
`REGISTER.md` rather than deleted, but the `EXP-####` records here (EXP-0001
through EXP-0006) are cube-only. Two invariants are **broken**
(`grid-convention`, `rasteriser-identity`) and most are `unchecked`, so most
records grade `low`. That is an accurate reading of the evidence, not a
miscalibrated scale — four claims in `REGISTER.md` are already marked
`invalidated` because of it, and the re-runs that would restore them are listed
at the bottom of that file.
