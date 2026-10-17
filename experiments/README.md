# experiments/ — evidence layer and experiment storage

Numbers live here so they can be **compared** and **invalidated**, and so
they have a home from data collection through to a citable result.

| File / dir | Role |
|---|---|
| `SUMMARY.md` | experiments grouped by question (not number), cross-cutting confounds, state of the hypotheses, holes and directions — synthesis only, cites ids |
| `REGISTER.md` | one row per claim: status, what supports it, what contradicts it, and what it **depends on** |
| `INVARIANTS.md` | the `depends_on` tag registry — each tag is a property that can be false, with the test that checks it |
| `METRICS.md` | `design.metric` key → exact formula |
| `COMMANDS.jsonl` | project-wide execution ledger (every run, filed or in `temp/`) |
| `TEMP_LOG.md` | one line per `experiments/temp/` scratch dir, written at creation |
| `temp/` | below-T0 scratch for quick, uncited checks — gitignored |
| `EXP-####-slug/` | one experiment: `EXPERIMENT.md`, `runs/`, `artifacts/`, `results/`, `reports/`, `code/` |

## Using it

```bash
python scripts/check_register.py               # validate (also runs in pytest)
python scripts/check_register.py --fix-grades  # recompute grades in place
grep -l grid-convention experiments/*/EXPERIMENT.md   # what dies if this breaks
```

The workflow (tiers, budgets, storage layout, `temp/`) is the
`experiment-log` skill; what's actually enforced (required fields, grade
computation, the invariant xfail pattern) is `register-validator`. Start
there rather than copying an existing record, which may predate a rule
change.

## Standing policies

**Narrow corpora and narrow-trained models are pilot instruments (user decision 2026-10-05).** The exact-20 mm n20 narrow-domain corpora (DS-0008..0013, DS-0015..0018; DS-0016 is the clean narrow benchmark) and every model trained on them (`nfd_3ch_narrow_l20_v2` and its seed / epoch10 / soft_s2 variants, `linear_narrow_l20_v2_res64` and its bootstraps, EXP-0053's narrow models, EXP-0059's retrieval banks, EXP-0063's soft NFDs) exist as a cheaper pilot task for rapid R&D, not as a separate avenue for training models. Any claim established only on them is **pilot-grade**: it must be replicated on broad training data (randlen, DS-0006, DS-0007) and a broad test set before it is treated as a project conclusion. Narrow-vs-broad training is NOT a research axis of this project.
Affected records carry a "Scope note (2026-10-05 policy)"; affected `REGISTER.md` rows carry a bracketed pilot-grade note; `SUMMARY.md` §0 and `TODO.md`'s standing rules restate it.

## 2026-09-13 — consolidated from `docs/experiments/` and restructured

Previously this evidence layer lived at `docs/experiments/` as flat
`EXP-####-slug.md` files, alongside a separate proposed `experiments/` tree
for run/artifact/result storage. The two were merged into this one root, and
the three records that existed at the time (`EXP-0001`–`EXP-0003`) were
moved from flat files into `EXP-####-slug/EXPERIMENT.md`, unchanged
otherwise. See `.claude/skills/experiment-log/SKILL.md` and
`.claude/skills/register-validator/SKILL.md` for the full design.

Not part of this move: `Baselines/*/runs/` (baseline checkpoints) were not
renamed to `weights/` or restructured into `MODEL-####` instances — every
current checkpoint path is hardcoded across several `Baselines/` and
`scripts/` files, so that rename is deferred to its own change rather than
bundled with this one. `Genesis/data/*` and `configs/dataset/*.yaml` were
likewise left as-is; migrating them into `datasets/DS-####/` instances is
deferred pending manual review of which existing config best represents
each template shape.

## 2026-09-10 — claim register reset

Before that, everything under the old `docs/experiments/` had itself been
reset from scratch — archived wholesale to
`archive/2026-09-10_pre-reset/docs/experiments/`, not because any of it was
found wrong, but to start the claim ledger clean rather than carry numbers,
ids, and dependency chains forward from a superseded work program. The old
material is fully intact there if a past claim or invariant needs to be
looked up.
