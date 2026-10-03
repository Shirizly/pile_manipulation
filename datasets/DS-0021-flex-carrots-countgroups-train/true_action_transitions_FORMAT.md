# `true_action_transitions_*` data format

Sequential-rollout (state, action) -> next-state data, produced by
`collect_transitions.py`. **Different structure from `true_action_slates_*`**
(see `data/true_action_slates_FORMAT.md`): slates replay many independent
actions from the *same* fixed initial state (good for ranking actions against
one state); these `transitions` datasets instead sample many independent
initial states and apply a short SEQUENCE of actions to each, where every
action continues from the previous action's settled outcome. The point is
many more genuinely-visited states for the same action budget, for
train/val rather than action-ranking metrics.

- `true_action_transitions_carrots` — 2000 states x 10 sequential actions,
  carrots, same state/action sampling as `true_action_slates_objbiased`
  (`rand_blob`/`rand_spread` init mix, `obj_biased` action sampler).
- `true_action_transitions_coffee` — 2000 states x 10 sequential actions,
  coffee, same sampling as `true_action_slates_objbiased_coffee`.

Both target 2000*10 = 20000 transitions; actual counts are slightly lower
because a few trajectories end early (see "Incomplete trajectories" below).

## Directory layout

Same per-state layout as the slates datasets:

```
true_action_transitions_<material>/
  manifest.jsonl          # one JSON record per line (after merging, see below)
  manifest_shard*.jsonl   # raw per-process shards, kept alongside manifest.jsonl
  reset_crash_attempts_shard*.json  # bookkeeping, see "Incomplete trajectories"
  run_config.json
  <state_idx>/
    initial_particles.npy
    initial_state.npz
    initial_color.png
    <step_idx>_after_particles.npy   # step_idx = 0..9, IN ORDER
    <step_idx>_after_color.png
```

**The key structural difference from slates**: here, `<step_idx>_after_*`
are a chain, not independent branches. `0_after_particles.npy` is the result
of applying transition 0's action to `initial_particles.npy`; `1_after_*` is
the result of applying transition 1's action to `0_after_particles.npy`; and
so on through `9_after_*`. There is no "replay from the same start" here —
reconstructing step *k*'s input state means reading `(k-1)_after_particles.npy`
(or `initial_particles.npy` for k=0), not re-deriving it.

This was collected by running several OS processes concurrently, each over a
disjoint slice of states (`--shard i --n-shards N`, `state_idx % N == i`) --
not `FlexEnvMulti`'s intra-solver tiling, which slates uses. Each process
wrote `manifest_shard<i>.jsonl`; `collect_transitions.py --merge` combined
these into the single `manifest.jsonl` described below (deduping a
crash-then-redo's leftover partial records, keeping the last write for any
duplicate `(state_idx, step_idx)`).

## `manifest.jsonl`

Three record types:

### `state_init`

Same shape as slates' `state_init`, plus `init_pos` only for carrots (the
state's `rand_blob`/`rand_spread` draw); coffee has no `init_pos` field
(its variety comes from `yx_CoffeeGrid`'s per-bean jitter instead, keyed only
by the state's seed, not a named generator).

### `transition`

```json
{"type": "transition", "state_idx": 0, "step_idx": 3,
 "action": [x0, z0, x1, z1], "valid": true,
 "push_length": 4.21, "bin": 2, "retries": 0,
 "after_positions_path": "0/3_after_particles.npy",
 "after_color_path": "0/3_after_color.png", "timestamp": ...}
```

Same `action`/`push_length`/`after_*` semantics as slates' `action` records
(see `true_action_slates_FORMAT.md`), with two differences:

- `step_idx` replaces `action_idx`, and **is sequential, not independent**:
  step *k* was applied to the state step *k-1* produced (or the initial state
  for *k=0*).
- `bin` is **informational only** here — unlike slates, these actions are
  *not* stratified/batched by length bin (each step draws a single,
  unconstrained action from the sampler). It's recorded only so push lengths
  can be compared against the slates datasets' bins without recomputing edges.
- `retries`: how many resample attempts this step took before a push that
  didn't explode the solver (0 = succeeded first try). A step that explodes
  is **not** recorded as a separate `invalid` row the way slates does — the
  collector discards the exploded attempt, restores the pre-step state, and
  resamples a fresh action for the same `step_idx` (up to
  `transitions.max_retries`, default 20), so every recorded `transition` row
  is `valid: true` by construction. This is a real behavioral difference from
  slates: there, an exploded action is still logged (as `valid: false`); here,
  only the final successful action for that step is logged, and the
  exploded attempts leave no trace beyond the `retries` count.

### `trajectory_failed` / `state_failed`

Two distinct terminal-failure types — **do not confuse them**:

- `trajectory_failed`: a step exhausted `max_retries` resamples without a
  non-exploding push (same failure *mode* as slates' solver explosions, just
  bounded by retries instead of being logged once and dropped). That state's
  trajectory stops at `step_idx` — it has fewer than 10 `transition` records.
  Rate observed: 1/2000 for coffee (0.05%).
- `state_failed` (reason `repeated_reset_crash`, **carrots only**): the
  state's *initial* `reset()` itself crashed the whole process natively
  (confirmed cause: a degenerate convex mesh in carrots' random-pile
  generation — see `.github/skills/pyflex-simulation/SKILL.md`'s "Native
  reset() crashes" section) more than `transitions.max_reset_attempts`
  (default 2) times. This state has **zero** `transition` records and no
  per-state files at all beyond whatever a crashed attempt left behind. Rate
  observed: 2/2000 for carrots (0.1%).

## Incomplete trajectories

Filter on `len([t for t in transitions if t.state_idx == s]) == 10` (or
equivalently, the absence of a `trajectory_failed`/`state_failed` record for
that `state_idx`) if you need only fully-complete 10-step trajectories. For
carrots: 998/1000 and 1000/1000 across the two collection shards (998 fully
complete, 2 permanently skipped via `state_failed`) — merge the actual
numbers from your copy's `manifest.jsonl` rather than trusting this count
blindly, since it's a snapshot. For coffee: 1999/2000 complete, 1 stopped
early via `trajectory_failed` at step 0.

`reset_crash_attempts_shard*.json` (carrots only) is the persistent ledger
`collect_transitions.py` used to bound reset-crash retries across process
restarts; it has no analytic value, just collection-time bookkeeping.

## Quick loading sketch

```python
import json, numpy as np

records = [json.loads(l) for l in open(f'{root}/manifest.jsonl')]
by_state = {}
for r in records:
    if r['type'] == 'transition':
        by_state.setdefault(r['state_idx'], {})[r['step_idx']] = r

for state_idx, steps in by_state.items():
    if len(steps) < 10:
        continue  # incomplete trajectory, see above
    particles = np.load(f'{root}/{state_idx}/initial_particles.npy').reshape(-1, 4)
    for step_idx in range(10):
        r = steps[step_idx]
        action = np.array(r['action'])
        next_particles = np.load(f'{root}/{r["after_positions_path"]}').reshape(-1, 4)
        # (particles, action) -> next_particles
        particles = next_particles
```
