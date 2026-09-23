# EXP-0024 results

See `EXPERIMENT.md`'s Numbers table for the full per-cell breakdown. Summary:

| quantity | value |
|---|---|
| cells tested | 4 (3 reduced-fidelity states + 1 full-fidelity state), 3 actions x 5 repeats each |
| within-action / between-action dv variance ratio | 0 to 5.3e-5 (all 4 cells) |
| max repeat-to-repeat COM spread | 0 to 1.6e-5 mm (float32 roundoff scale) |
| between-action dv variance (the slateN signal) | 2.5e-4 to 2.3e-3 |

**Conclusion**: for n20 single-push candidates scored through
`GenesisOracleEnv`'s exact-snapshot-restore mechanism (the same broadcast
mechanism `same_state_slate_collection.py`/`binned_slate_collection.py` use
to build real slate corpora), resimulating the identical action from the
identical state returns `dv` unchanged to float32 roundoff. Resimulation
noise is not a plausible source of slateN's headroom ceiling at this scale;
the archived `docs/scaling_to_200_objects.md` section 8.8 finding that
"Genesis is not bit-deterministic" measures a different mechanism (a fresh
resettle from a stored config, not an exact live-tensor snapshot restore) and
does show real divergence there (up to ~38% relative at n=100) -- so physical
non-determinism is real in this simulator, just not reachable through the
mechanism slate scoring actually uses.
