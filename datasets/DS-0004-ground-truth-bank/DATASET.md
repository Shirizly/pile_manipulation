# DS-0004 — ground-truth bank: every simulated push outcome, reusable

**Status:** active (append-only; grows as benchmarks simulate new actions)
**Payload:** `datasets/DS-0004-ground-truth-bank/data/<sim_path>/<state_key>.pt` (gitignored)
**Code:** `simple_mpc/gt_bank.py::GroundTruthBank` (tests: `tests/test_gt_bank.py`)

## What this is

The final particle state (n, 7) of every simulated (start state, action) pair a
benchmark has paid for, keyed by execution path, start-state hash and exact
action bytes. It stores OUTCOMES, not `dv`, so any goal / value function can be
scored against already-simulated actions without Genesis. **Reuse is NOT yet sound for
`rollout_candidates` paths:** `genesis-snapshot-restore-repeat-determinism` is
BROKEN (EXP-0027, 2026-09-24) -- successive `rollout_candidates` calls from the
same snapshot give dv differing by up to 7e-3 (17 mm), depending on the
batch's other actions and on the env's history (up to 4.8e-2 between reset and
un-reset batches). Until that is fixed or its noise is characterised, do not
add `oracle_rollout_*` rows: a banked outcome would be one draw, served as if
it were THE outcome. (EXP-0027's 175 smoke-test rows were deleted for this
reason.) The seeded `binned_collection` rows are the corpus's own recorded
outcomes -- they are exactly what DS-0001 contains, but whether the COLLECTION
path is itself history-affected is untested.

## Rules the code enforces

- **Execution paths are never mixed.** `sim_path` must be registered in
  `gt_bank.SIM_PATHS`; a lookup under one path never returns another's rows.
  EXP-0023 re-executed DS-0001 actions via `rollout_candidates` (full fidelity)
  and got r = 0.959 against the corpus's own `dv`, not identity.
- **One fingerprint per (path, state) file**; adding rows under a different
  fingerprint (settle budget, config sha...) raises.
- **First write wins**; re-adding a stored action is a no-op.

## Contents

| sim_path | source | states | rows | built by |
|---|---|---|---|---|
| `binned_collection` | DS-0001 (20 slates x 1000) | 20 | 20,000 | `build_from_ds0001.py` (runs/ds0004_seed_from_ds0001.*) |

`data/binned_collection/extras.pt` maps DS-0001 slate id -> state key, and keeps
the corpus's per-row explicit yaw (`angles`); actions are keyed as
`[sx, sy, ex, ey]` metres, identical to `binned_pool_cache.py`'s `actions`.

**Validation:** `dv = lyapunov(corner)` recomputed from the stored final states
reproduces `experiments/temp/binned-pools/dv_cache_corner.pt`'s `dv_true` to
max abs error 1.2e-7 over all 20,000 rows.

## Regenerate

    PYTHONPATH=. python datasets/DS-0004-ground-truth-bank/build_from_ds0001.py

(any rows added later by experiments are regenerable only by re-running those
experiments; each file's `sources` list names them).
