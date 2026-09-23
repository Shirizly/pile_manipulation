---
name: data-collection
description: "How to collect simulated push data in this repo: which driver script produces which corpus shape, the SandboxManipulation seams they all compose (state library, action sampling, execute_action, collect_data_samples), the config keys that matter, and the traps that have silently corrupted corpora before. Use when collecting a new dataset, extending or debugging a collection driver, reading an existing corpus off disk, or deciding which existing corpus answers a question."
argument-hint: "Optionally: the corpus shape you want, e.g. 'same-state slates', 'independent transitions', 'multi-step chains'"
user-invocable: true
---

# Collecting push data in Genesis

Everything here runs under `conda activate pme` and needs a GPU. Always
`python -u` (see `subagent-experimenter` rule 2).

## Pick a driver by the corpus shape you need

| driver (`python -m Genesis.<mod>`) | corpus shape |
|---|---|
| `data_collection_clean.py` | independent transitions: one pile, each env its own action, `n_samples` pushes per env chained without reset. The general-purpose training corpus (`data/overnight_randlen`). |
| `cube_spectrum_collection.py` | the same, swept over particle size/count — the granularity spectrum. |
| `same_state_slate_collection.py` | **same-state slates**: one settled state broadcast to all envs, each env a different candidate action. `--n-steps>1` makes each env an independent chain from that shared start. **One fixed push length per corpus** (`data/slates`, `data/slates_multistep`). |
| `binned_slate_collection.py` | same-state slates as above, but push length drawn from 5 bins over 20–70 mm, mixed along each chain. Sampling and simulation are separate passes so each simulated batch is length-homogeneous (`data/slates_binned`). |
| `run_collection.py` | the batch/config-sweep entry point over the above. |

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

- `pile_aware=True` — blade starts one particle-width off the pile's near face,
  laterally aligned so its swath holds `min_swath_particles`. **Owns the
  geometry end to end and returns early**, so `placement_aware`,
  `perpendicular_pushes` and the fixed-length constraint below are all dead
  when it is on (it applies its own). Blind sampling put only ~14 % of the pile
  in a typical push's path, which is why this exists (`docs/piled_collection.md`).
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
