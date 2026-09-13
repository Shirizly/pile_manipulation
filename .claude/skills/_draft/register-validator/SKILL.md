---
name: register-validator
description: "Validation mechanics for the experiment claim ledger: what scripts/check_register.py enforces on experiments/EXP-####-slug/EXPERIMENT.md frontmatter, experiments/REGISTER.md, and experiments/INVARIANTS.md — required fields per tier, how grade is computed from downgrades, the downgrade-domain vocabulary, and the invariant status/xfail pattern. Use when writing or fixing an EXPERIMENT.md's frontmatter, when check_register.py fails, or when adding/retiring an INVARIANTS.md tag."
argument-hint: "Optionally: a validator error message, or the field/tag you're trying to satisfy"
user-invocable: true
---

# Register validator — mechanics

This skill owns the *enforcement* side of the claim ledger: what
`scripts/check_register.py` actually checks. For *why* the ledger is
organized this way, which tier to pick, budgets, and how experiment storage
is laid out, see the `experiment-log` skill.

```bash
python scripts/check_register.py              # validate (also runs in pytest)
python scripts/check_register.py --fix-grades # recompute grades in place
grep -l some-tag experiments/*/EXPERIMENT.md   # what depends on a tag
```

Exits non-zero on any error; warnings do not fail the run. **The script is
authoritative on any conflict with this file** — if they disagree, fix
whichever one is wrong in the same change, don't leave the mismatch.

## Required frontmatter fields, by tier

**T0** (probe): `id, title, tier, mode, date, claim, provenance, result,
verdict, downgrades, grade`. `provenance` needs at least `commit, script,
data, code_path, seed`. No body required.

**T1** (experiment): every T0 field, plus `prediction`, `design` (`varied,
held_fixed, baselines, metric`), `noise_floor`, `depends_on`, `budget`. Full
body, per `references/experiment-template.md` in `experiment-log`.
`design.baselines` must be non-empty and include a "do nothing" baseline
(`persistence` for image prediction, `mean-delta` for canonical-frame
prediction, in this repo); `prediction.discriminating` must be present and
`true`, or the record must justify why not; the body must include a
populated `## Unrelated findings` heading (even if its content is "none" —
a missing heading, not an empty one, is the error).

**T2** (gate): every T1 field, and additionally:
- `prediction` must be committed to git **before** the run — if the working
  tree was dirty when the run happened, the frontmatter's `dirty: true` must
  say so, and the body's "What was actually run" must say what was
  uncommitted; a T2 record with `dirty: true` and no such note is an error,
  not just a downgrade;
- every `depends_on` tag must currently be `holds` or `fixed` in
  `INVARIANTS.md` — a `broken` or `unchecked` tag is a **hard error** at T2,
  not merely a downgrade. This does not apply to a tag the same record lists
  in `establishes` (it's the evidence for that tag, not a claim resting on
  it already holding);
- the full multiverse sweep must be reported (see below), not the winning
  cell alone.

## Grade — computed, never chosen

Start at `high`. Subtract one level per **distinct downgrade domain**
present in `downgrades`, floor at `very-low`:

`high → moderate → low → very-low`

`check_register.py` recomputes this from the `downgrades` list and fails the
run if the stated `grade` doesn't match. `--fix-grades` rewrites it in
place; there is no way to set a grade higher than what the downgrades
imply.

## Downgrade domains (use only these words)

| Domain | Applies when |
|---|---|
| `provenance` | the comparison crosses code paths, rasterisers, datasets, or conventions |
| `imprecision` | effect not clearly above `noise_floor`, or a single split where folds were available |
| `indirectness` | the measured quantity is a proxy for the one the claim is about |
| `inconsistency` | it does not hold across every seed, fold, or configuration tested |
| `selection` | the reported configuration was chosen after seeing results |
| `incomplete-design` | a cell the design called for was not run |
| `untested-dependency` | a `depends_on` tag is not `holds`/`fixed`. **Forbidden at T2** (hard error, not a downgrade there); does not apply to tags in `establishes` |

## Verdict vocabulary (record-level, not claim-level)

`supported` · `refuted` · `inconclusive` (effect inside the noise floor, or
the design didn't discriminate — and this includes a T1/T2 record whose
missing cell, per budget overrun, would have been the discriminating one:
that's `inconclusive`, never `supported`) · `invalidated` (the result stands
as measured, but an input is now known broken).

This is distinct from — and narrower than — `REGISTER.md`'s claim-level
status vocabulary (`supported/refuted/open/narrowed/contested/invalidated/
superseded/withdrawn`), which `experiment-log` owns. A record's `verdict`
feeds into a claim's `status`, but the two vocabularies are not
interchangeable.

## Multiverse discipline (T2, encouraged everywhere)

If you swept it, publish the whole sweep, not the winning cell. If you did
not sweep it, `design.held_fixed` must say so, and the body must say why
that value was chosen. A single cell reported from an unreported sweep is
`downgrades: [selection]`.

## Exploration is not confirmation

T0 and exploratory T1 records carry no `prediction` block and must set
`mode: exploratory`. The validator refuses `grade: high` to a claim in
`REGISTER.md` supported only by exploratory records.

## `INVARIANTS.md` tags: status vocabulary and the xfail pattern

Status is one of `holds` (a test asserts it and passes) · `unchecked`
(believed true, nothing enforces it) · `broken` (known false) · `fixed` (was
broken, now checked).

For a tag that is `broken`, write the test now and mark it
`@pytest.mark.xfail(strict=True, ...)` — this records the defect in the
suite rather than in prose, and `strict=True` means the moment someone fixes
the underlying bug, the test XPASSes and **fails the run**, forcing the
marker to be dropped and this table's status flipped to `fixed`. A broken
invariant then cannot be quietly fixed without the register noticing.

Adding a tag: name the property as something that can be **false**, not as a
topic; give it a status honestly (`unchecked` is a normal answer, and
`depends_on` should be generous — the cost of citing a tag is one word, and
the payoff is automatic invalidation later); if you are about to rely on it
at T2, write the test first.

## When a bug is found (the case this whole design exists for)

1. Update the tag in `INVARIANTS.md` to `broken`, with the date and evidence.
2. `grep -l <tag> experiments/*/EXPERIMENT.md` and set every affected
   record's `verdict` to `invalidated`, with `invalidated_by:` pointing at
   the record that found the bug.
3. Update every `REGISTER.md` row whose `depends_on` contains the tag.
4. Write the invariant test **before** fixing the bug, so it fails first.
5. Only then fix it, and queue the re-runs.

Skipping step 3 is the specific failure this file exists to prevent — an
invariant can go `broken` and the claims that depended on it can keep
reading as if nothing changed. (`experiment-log` covers the companion rule:
commit the fix promptly, even tentatively, once it's plausible.)

## `REGISTER.md` row schema and structural checks

Each row: `ID, Claim, Status, Grade, Supported by, Contradicted by, Depends
on` (the "Open and contested" table also carries `Contradicted by`; the
"Supported" table omits it). `check_register.py` enforces:

- Every `EXP-####` cited anywhere in `REGISTER.md` has a matching record
  under `experiments/`.
- Every tag named in a `Depends on` column exists in `INVARIANTS.md`.
- No duplicate claim IDs (two rows silently allocating the same `C-####`).
- A record not cited anywhere in `REGISTER.md` is a warning: a number nobody
  can find is as good as unrecorded.
- **Not currently checked:** that a `design.metric` key actually resolves to
  an entry in `METRICS.md`. Verify that by hand until this is added.

## Bounded metrics — a modelling caveat, not a validator rule

The validator cannot check this, but it's the single most common way a
correct-looking curve misleads: FSS, IoU, SSIM, and R² are all bounded, and
converge to their ceiling as a comparison is made easier. A raw curve rises
trivially near that ceiling, and subtracting a baseline score from a bounded
score forces the difference toward zero for the same reason. See
`experiment-log`'s Metrics section for the fix (report a reference level
with meaning, not a raw difference).
