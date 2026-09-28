---
name: data-collection
description: "How to collect simulated push data in this repo: which driver script produces which corpus shape, the SandboxManipulation seams they all compose (state library, action sampling, execute_action, collect_data_samples), the config keys that matter, and the traps that have silently corrupted corpora before. Use when collecting a new dataset, extending or debugging a collection driver, reading an existing corpus off disk, or deciding which existing corpus answers a question."
argument-hint: "Optionally: the corpus shape you want, e.g. 'same-state slates', 'independent transitions', 'multi-step chains'"
user-invocable: true
---

# Collecting push data in Genesis

Everything here runs under `conda activate pme` and needs a GPU. Always
`python -u` (see `subagent-experimenter` rule 2).

**Checkpointing is mandatory** for anything longer than a few minutes: rewrite
results/manifests atomically after every unit of work so a cut-off run loses
at most one unit. The full rule is in `project-overview`, "Every job must
survive being cut off".

## Pick a driver by the corpus shape you need

| driver (`python -m Genesis.<mod>`) | corpus shape |
|---|---|
| `data_collection_clean.py` | independent transitions: one pile, each env its own action, `n_samples` pushes per env chained without reset. The general-purpose training corpus (`data/overnight_randlen`). |
| `cube_spectrum_collection.py` | the same, swept over particle size/count — the granularity spectrum. |
| `same_state_slate_collection.py` | **same-state slates**: one settled state broadcast to all envs, each env a different candidate action. `--n-steps>1` makes each env an independent chain from that shared start. **One fixed push length per corpus** (`data/slates`, `data/slates_multistep`). |
| `binned_slate_collection.py` | same-state slates as above, but push length drawn from 5 bins over 20–70 mm, mixed along each chain. Sampling and simulation are separate passes so each simulated batch is length-homogeneous (`data/slates_binned`). |
| `run_collection.py` | the batch/config-sweep entry point over the above. |
| `chain_collection.py` | chained pushes from GIVEN start states (not settled fresh each time): `--mode chains` (K different states, each its own --steps chain; DS-0008/DS-0010's recipe), `--mode pools` (same-state pools, one push each; DS-0009/DS-A), `--mode seqpools` (same-state pools of MULTI-STEP sequences -- one shared start broadcast to `--pool` envs, sub-batched by `--n-envs`, each running its own `--steps`-push chain; DS-B). All three write one atomic, resumable file per chunk; `seqpools` reuses `chains`' own `chain_env`/`chain_step` field names (sequence id / push index) plus a `pool_idx` column, so its output is a drop-in for `eval_retrieval.py`'s existing chain-rollout code. |

A slate corpus is what scores a model **as an action ranker** (the true result
of every candidate is known because all of them were simulated). An
independent-transition corpus is what you fit a dynamics model on.

## The seams every driver composes

All of these are on `Genesis/sandbox_manipulation_clean.py::SandboxManipulation`.

| need | call |
|---|---|
| settle a bank of reusable start states | `state_library.build_state_library(sim, n_settles=…)` → `lib.apply` (one state to all envs) / `lib.apply_per_env` (one per env) |
| write an arbitrary per-env state, no re-settle | `set_particle_state(pos (E,P,3), quat (E,P,4))` — also updates `sim._particle_state`, so action sampling sees it immediately |
| draw actions without simulating | `generate_action_samples(n_samples, …)` → `(starts (E,S,3), stops (E,S,3), angles (E,S))` |
| run one push on every env | `execute_action(p_start (E,3), p_stop (E,3), angle (E,))` → `(reached_goal, final_pos)`; follow with `update_material_state()` to settle and refresh `_particle_state` |
| the whole loop plus buffers/saving | `collect_data_samples(n_samples, path, …)` |

To make `collect_data_samples` execute actions *you* chose, monkeypatch
`sim.generate_action_samples` for the duration of the call — see
`same_state_slate_collection.py::_patched_action_sampler`. That is the only
supported seam; do not fork the buffer/save logic.

## Action sampling options, and how they interact

Set on `generate_action_samples` / `collect_data_samples`:

- `pile_aware=True` — blade starts one particle-width off the pile's near face
  (or a SAMPLED gap, see `start_gap_range` below), laterally aligned so its
  swath holds `min_swath_particles`. **Owns the geometry end to end and
  returns early**, so `placement_aware`, `perpendicular_pushes` and the
  fixed-length constraint below are all dead when it is on (it applies its
  own). Blind sampling put only ~14 % of the pile in a typical push's path,
  which is why this exists (`docs/piled_collection.md`). **Legal by
  construction since 2026-09-28 (ISS-010 fix):** the touchdown is checked with
  an exact SAT test against every cube and REDRAWN (a fresh heading, never a
  shortened/lengthened push) whenever it overlaps one, up to `max_redraws`
  (default 200) — see `Genesis/action_sampling.py::pile_aware_action_batch`
  and the trap below for what this replaced.
- `start_gap_range=(lo_margin, hi_margin)` — pile-aware only. SAMPLES the
  touchdown gap (blade front face to the first contacted cube's near face,
  along the push axis) uniformly in `[lo_margin, L - hi_margin]` (`L` = the
  scalar `push_length`) instead of the old fixed one-particle-width gap —
  e.g. `(0.005, 0.005)` with `L=0.02` gives a 5-15 mm gap, so the contacted
  cube travels 5-15 mm. `None` (default) reproduces the old fixed gap exactly.
  A slot with no legal AND in-window candidate after `max_redraws` is
  RETARGETED (fresh heading/contact cube), never accepted as drawn; only as a
  last resort is it accepted and flagged `gap_out_of_window=True` in the saved
  row (report this count — it should be ~0 at `max_redraws=200` for an n=20
  narrow-domain pile; check before trusting the gap distribution on a denser
  or larger pile). Requires a uniform particle size and a SCALAR `push_length`
  (raises `NotImplementedError` for a free-length draw or per-env tensor
  length — the target gap depends on knowing `L` before the touchdown is
  placed, which those cases cannot provide before the fact).
- `placement_aware=True` — touchdown pose drawn from the tool's free
  configuration space. Composes with `perpendicular_pushes` + `push_length`,
  which run *after* it in `_constrain_push_geometry`. Use this, not
  `pile_aware`, when you need an exact length.
- `perpendicular_pushes=True` — push along the blade face normal (planar-pushing
  convention; 5-DOF → 4-DOF).
- `push_length` — metres. `None` draws uniformly. A float fixes it for the whole
  batch. Under `pile_aware` it may also be a **tensor** of per-env targets
  (added for `binned_slate_collection.py`); `_pile_aware_stops` broadcasts it
  and reports shortfalls only under `--debug` in that case, because the per-env
  callers redraw instead.
- `shared_travel_distance=True` — one length per batch. A throughput win, because
  envs step in lockstep and the sweep is sized by the LONGEST travel in the
  batch. This is the whole reason a length-mixed corpus must batch by length bin.

## Spawn vocabulary — scatter vs pile

`--spawn-mode` takes the simulator's three names, but **docs and reports should
say scatter or pile**, which is the distinction that actually matters:

| mode | style | what it builds |
|---|---|---|
| `drop` | **scatter** | single layer — dropped cubes bounce outward on landing, 90-94 % settle in layer 0 |
| `heap` | **pile** | irregular two-layer heap, sites redrawn every call; use when episodes must start differently |
| `pyramid` | **pile** | stepped pyramid, same lattice every time, will not restructure |

`BinnedSlateCorpus.spawn_style` returns `"scatter"` / `"pile"` so a report does
not have to re-derive it from the mode. Never compare a scatter corpus with a
pile corpus without saying so — a push through one layer is not the same
operator as a push through two.

## Geometry limits — check before choosing a length

`configs/basic.yaml`: tray `box.vol = [0.128, 0.128, 0.04]`, granular
`material.vol = [0.127, 0.127, 0.05]`, `safety_margin = 0.02`, blade
`plate.size = [0.04, 0.002, 0.01]`.

`action_sampling.sampling_box` shrinks the blade-centre box with yaw: the usable
span is ~47 mm across the blade's long axis and ~85 mm along its normal. So a
perpendicular push tops out near **85 mm**, and anything past ~60 mm is
frequently wall-limited depending on where the pile put the contact point.
20–70 mm is the practical bin range.

## Traps that have actually corrupted corpora

- **(FIXED 2026-09-28, ISS-010) The old pile-aware clamp placed the tool ON a
  cube at touchdown in ~44-56% of rows.** `_pile_aware_stops` computed a
  genuinely collision-free touchdown (`pile_contact_starts`, one
  particle-width behind the pile's near face) and then unconditionally
  clamped it into the yaw-dependent sampling box with ZERO cube check,
  because the pile routinely spreads past the box (its own comment: "35.8% of
  starts out of box"). The clamp traded "start exactly at the pile edge" for
  "start just inside the pile", which can mean directly on top of an
  individual cube's footprint — measured on DS-0008/9/11/12/13, all
  `pile_aware=True` (DS-0010's older/mixed pipeline was ~clean by a different
  mechanism). **Fixed**: `_pile_aware_action_legal` ->
  `Genesis/action_sampling.py::pile_aware_action_batch` checks the FINAL
  touchdown with an exact SAT test and redraws any illegal slot instead of
  accepting it. Every corpus collected before the fix (DS-0008/9/10/11/12/13)
  keeps its original, still-illegal payload — check a corpus's own DATASET.md
  for whether a clean `_v2`/replacement set exists (DS-0015/16/17 replace
  DS-0008/10, DS-0009, DS-0011; DS-0012/13 do not yet have one) before reusing
  it. **If you are writing a NEW pile-aware collector or changing this
  geometry again**: the clamp is not the only place a "gap precision" bug can
  hide — a follow-up fix (`start_gap_range`, same date) found that the box
  clamp also silently invalidates the along-push-axis GAP a touchdown was
  built with (it moves the point in world x,y, not the push/lateral frame the
  gap was computed in), independently of whether the clamped point is still
  legal. If a future change sets the touchdown gap to anything other than the
  old fixed one-particle-width value, redraw on that condition too (see
  `pile_aware_action_batch`'s `_gap_bad`), not just on illegality — and if you
  add a NEW redraw condition to a loop that already has one, give the
  conditions a priority order (legality must always win) rather than an
  unconditional OR: sharing one redraw budget across two conditions
  undifferentiated let illegal touchdowns actually INCREASE in one measured
  case, because attempts were spent fixing the newer condition on slots that
  were already legal while genuinely illegal slots ran out of budget.

- **The binned collector's physics defaults do NOT match the training corpora.**
  `binned_slate_collection.py` defaults to friction 0.3 (particle and box) and
  density 1000; overnight_randlen and Sean use particle friction 0.7, box friction
  0.5, density 450, safety_margin 0.005. DS-0001 was collected on the defaults, so
  every model scored on it was tested off its training physics. For a test corpus,
  pass `--friction 0.7 --box-friction 0.5 --density 450 --safety-margin 0.005
  --settle-steps 3000` (or whatever the models' training corpus used -- read its
  `_config.yaml`), and state the physics in the DATASET.md.

- **A shortened push is not in its length bin.** `constrain_push` moves the
  START rather than shortening; `_pile_aware_stops` shortens and warns. Always
  recompute `‖p_stop − p_start‖` and check it against what you asked for before
  fitting a length-conditioned operator.
- **Genesis sets torch's default device to cuda.** A bare `torch.zeros(n)` in a
  driver lands on the GPU and then fails to accept a `.cpu()` assignment. Pass
  `device="cpu"` explicitly for host-side bookkeeping buffers.
- **`collect_data_samples` derives its batch index from the file count**
  (`len(listdir)/3`, since it writes `_k_data.pt` / `_k_failed.pt` /
  `_k_config.yaml`). Writing anything else into that directory mid-run
  desynchronises the numbering — put a `manifest.json` there only at the end,
  and record the batch→meaning mapping explicitly rather than relying on the
  convention.
- **`*_eval` dataset configs apply `min_push_length_m` filtering**, which drops
  rows and breaks row↔env alignment across steps. Load multi-step chains from
  the raw `_{batch}_data.pt` files.
- **Escaped particles poison every later transition in their env**, because each
  transition's `s` is the previous one's `s'`. `collect_data_samples` records
  `escaped_particles` and `contact_budget` in the saved config — read them.
- **Pile-aware actions must be redrawn per step**, not batched up front: one
  push moves the material the next would have aimed at (measured blade-to-pile
  gap 5.0 → 17.1 → 12.4 mm when drawn up front).
- **Single-push-length corpora leave most length bins empty**, which left 4 of 6
  length operators byte-identical to their initialisation in one fit. Use
  `binned_slate_collection.py` if the model is length-conditioned.

## The unified benchmark shape (target for every new TEST corpus, 2026-09-24)

A benchmark test corpus is a set of **same-state pools**: many candidate pushes
simulated from one identical start state, the true outcome of each stored as
full particle states. Store it as flat rows:
`states (R,n,7)`, `states_ (R,n,7)`, `p_starts (R,3)`, `p_stops (R,3)`,
`angles (R,)`, `pool_idx (R,)` (= start-state id), plus source bookkeeping,
and a DATASET.md stating: particle count, spawn style, **physics (particle
friction, box friction, density -- must match the models' training corpus)**,
push-length distribution, pool sizes. Truth is scored from `states_` with the
soft rasteriser (METRICS.md "Ground-truth scoring"), never from a stored image.
Two disjoint pools per state (or >= 2x the pool size you score with) lets
"state vs pool" be separated (EXP-0029). DS-0006 (binned corpus: `slate_idx` =
`pool_idx`) and DS-0007 already have this shape.

## Reading a corpus off disk

- `data_collection_clean` / `same_state_slate_collection` →
  `_{k}_data.pt` with `states, states_, p_starts, p_stops, angles` (+
  `_{k}_failed.pt`, `_{k}_config.yaml`). In a slate corpus, `manifest.json` maps
  batch index → (slate, step, state-library index).
- `binned_slate_collection` → **use `Genesis/binned_slate_dataset.py`**, do not
  read the files by hand:

  ```python
  from Genesis.binned_slate_dataset import BinnedSlateCorpus
  c = BinnedSlateCorpus.load(".../slates_binned/<tag>")
  c.verify()                     # chain continuity + same-state slates, asserts
  c.trajectories(slate=0)        # (X, N+1, P, 7) every sequence from state 0
  c.state_after(chain, 2)        # (P, 7) after that chain's first 2 pushes
  c.select(slate=0, bin=3)       # one slate's 50-60 mm sweeps
  c.by_bin()                     # {bin: Transitions} over the whole corpus
  c.select(step=0).group_by("slate_idx")
  ```

  Every selection returns a `Transitions` table carrying `chain_idx`,
  `step_idx`, `slate_idx`, `action_idx`, so a subset can always be re-grouped or
  joined back. `bin=` filters on the length *achieved*; `requested_bin=` on what
  the schedule asked for — they differ wherever the tray wall cut a push short,
  and the achieved one is what a length-conditioned operator must be fitted on.
  `python -m Genesis.binned_slate_dataset <dir>` prints the integrity report.

  Underneath: `step{k}.pt`, every tensor indexed by a flat chain id
  `c = slate_idx * n_actions + action_idx`, identical row order in every step
  file, so row `c` of `step1.pt` continues row `c` of `step0.pt`. Carries
  `len_target`, `len_realized`, `bin_requested`, `bin_realized` alongside the
  usual keys.

Existing corpora and their known problems are listed in `docs/CODEMAP.md`'s
Datasets table — check there before collecting something that already exists.
