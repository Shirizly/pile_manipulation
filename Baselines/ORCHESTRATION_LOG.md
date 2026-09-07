# Baselines — orchestration log

**Owner:** orchestrator session (Claude Opus 5). Branch `baselines/overnight`,
off `VisualForesight` @ `9481c3e7`. Started 2026-09-08.

---

## SUMMARY FOR HANDOFF

**Goal.** Reproduce, as baselines, the dynamics models from the reference
papers in `docs/reference_papers/`, train them on the *limited* multistep
slate data, and score them with the register's standard metrics. This is
**exploratory scaffolding, not the final result**: the real models will be
trained on much larger datasets for much longer. What must survive the night
is *working, plugged-in, measured* models — not good numbers.

**Priority order (fixed by the user, do not reorder):**
1. Working models, tested in the MPC-proxy scoring and over the dataset.
2. As many papers' models recreated as possible.
3. Model performance.

**Scope decisions taken with the user (2026-09-08, before the overnight run):**

| decision | value |
|---|---|
| Data | `slates_multistep/n20_L20mm` + `n20_L40mm` **pooled** for training; eval per-cell on each cell's own `_eval` split. `L10mm` excluded (suspect, see EXP-0028/EXP-0029); `n50_L20mm` excluded (incomplete). |
| "Tested in MPC" | **Offline slate scoring only.** No closed-loop Genesis. `slateK_exact`, `regret_dv`, `worstK`, `rank_profile` + `accuracy`, per `docs/experiments/METRICS.md`. |
| Papers | 1. Dynamic-Resolution GNN · 2. NFD (UNet, **non-FiLM**) · 3. Gaussian Splatting VMPC (**from rasters of the stored states**, attempt-and-skip) · 4. Schenck "Learning Robotic Manipulation of Granular Media" CNN. Optimal-Transport paper: out of scope. |
| Git | Commit freely on `baselines/overnight`. No push, no merge to `main`. |
| Models | Sonnet for every subagent. |
| NFD base | **`model/UNetModels_modular.py` (`UNet`, registry `unet-modular`), NOT `model/NFDUNetFilm.py`** (user, 2026-09-08). Consequence: `unet-modular` is `forward(x)` only and consumes no physics vector, so the action must enter **entirely through input channels** — this makes NFD's action-encoding question load-bearing. `NFDUNetFiLM` stays in the tables only as the number to beat (0.419/0.504). |
| GNN scope | **No resolution regressor.** GNN only, at a CONSTANT node count (user, 2026-09-08). Our cells are exactly 20 cubes, so one node per cube, N=20 fixed. `Baselines/GNN/train/train_res_rgr.py`, `data_gen/res_rgr_data.py`, `dataset/dataset_res_rgr.py`, `model/res_regressor.py` are OUT OF SCOPE. |

**Layout.** Every baseline lives entirely under `Baselines/<NAME>/`, with its
own `LOG.md` (summary at the top, running notes below). Shared code is in
`Baselines/common/`.

**Reading order for a fresh session:** this file → the "Agent roster" table
below → the `LOG.md` of whichever baseline you are picking up.

---

## The key architectural fact (found before any agent was spawned)

**The evaluation harness already exists and must not be rewritten.**
`scripts/probes/expB_multistep_eval.py` already scores *these exact two cells*:
it fits/loads a model, computes swept-region `accuracy` on the full eval split,
and writes a per-candidate dV cache in precisely the format
`scripts/probes/exp0026_kcurve_exact.py` consumes for `slateK_exact` / `worstK`
/ `rank_profile`. Control utility is restricted to **step 0 only** (steps 1–2
are each env's own diverged rollout, not a same-state slate).

So a new baseline needs exactly two things:
1. a predictor that maps (state, action) → next state in *some* representation;
2. an adapter to 64×64 occupancy so the existing metric code applies unchanged.

Everything else — the Lyapunov weight fields, the swept-region mask, the
combinatorial K-weights — is already written, already validated against the
register, and is reused, not reimplemented.

## Reference numbers to beat (existing, per-cell UNet-FiLM, `runs_expB/*_accuracy.json`)

| model | `accuracy` L20mm | `accuracy` L40mm |
|---|---|---|
| persistence | 0.000 | 0.000 |
| mean-delta | 0.086 | 0.138 |
| linear operator | 0.301 | 0.447 |
| UNet-FiLM (per-cell) | **0.419** | **0.504** |

Note these were trained **per-cell**; ours are pooled, so a small difference in
either direction is expected and is not by itself evidence about architecture.

## Facts implementers should not have to rediscover

- **Training entry point** is `python -m training.train --config <cfg>` with the
  schema in `configs/training/expB_unetfilm_slates_multistep_n20_L20mm.yaml`
  (read it; it is the closest working precedent). Blocks: `model` / `dataset` /
  `training` / `inference` / `output`.
- **The existing UNet's inputs** are `in_channels: 2` (occupancy + an action
  delta channel built by `transforms/functional.py::build_action_delta`) and
  `cond_dim: 3` physics. Its recipe is 100 epochs, batch 32, Adam lr 1e-4,
  StepLR(50, 0.75), mixed precision, grad clip 1.0, loss
  `eulerian_combined` (mse 1.0 + mass 0.2).
- **Measured training cost**: the per-cell UNet ran 90 epochs in 34.5 min on
  11 520 transitions -> ~23 s/epoch. Pooled L20+L40 is 23 040 transitions, so
  budget **~45-80 min per 100-epoch UNet-scale run**. Three such models fit
  comfortably in one night on the single GPU; there is no need to cut epochs.
- **Grid**: box 0.128 m across, `resolution_scale: 0.5` -> 64x64.
- **Frame conventions are a known hazard** (claim C-018: occupancy and action
  channels once placed world x on opposite grid axes). `docs/INTERFACES.md`
  owns them. Do not invent a new rasteriser or a new axis convention; reuse.

## Hardware constraint

One RTX 4070 Laptop, **8 GB**. Two trainings at once will OOM and take both
down. Every CUDA job goes through `Baselines/common/gpu_lock.sh` (flock, 6 h
wait), so agents may run concurrently while GPU work serialises.

---

## Agent roster

Wave A (spawned 2026-09-08, all Sonnet, all running concurrently — none needs the GPU):

| agent | scope | writes | status |
|---|---|---|---|
| **A4-common** | DONE — **self-test PASSED**, reproduces `runs_expB/n20_L20mm_accuracy.json` exactly (mean-delta 0.08569, linear 0.30053, all three per-step rows) and `slateK_exact` at every K via `exp0026_kcurve_exact.py` run unmodified. The harness is trustworthy. |
| _(A4 original scope)_ | Shared infrastructure. Pooled L20+L40 train config; `data.py` (particle view + occupancy view, occupancy produced through the EXISTING registry path so grids are byte-identical to the register); `eval_baseline.py` = `expB_multistep_eval.py` generalised to a pluggable predictor. Gated on a **self-test that must reproduce `runs_expB/n20_L20mm_accuracy.json`** (mean-delta 0.08569, linear 0.30053) and the existing `slateK_exact`. | `Baselines/common/`, `configs/dataset/*L20L40*` | running |
| **A1-gnn-spec** | Design doc for the dynamic-resolution GNN: graph construction, exact layer sizes from `gnn_dyn.py` + the vendored checkpoint, how the pusher enters the graph, and a **format mapping from their PyFlex `*_particles.npy` / `actions.p` to our batched `.pt`**. No resolution regressor. | `Baselines/GNN/SPEC.md`, `LOG.md` | running |
| **A2-grid-specs** | Design docs for NFD (non-FiLM, on top of the existing UNet code) and the Schenck CNN. Must settle what NFD's action encoding actually is and whether FiLM came from the paper or from this repo. | `Baselines/NFD/SPEC.md`, `Baselines/SchenckCNN/SPEC.md`, both `LOG.md` | running |

**Wave A closed 2026-09-08. All three agents delivered; A4's self-test passed,
so the harness is trusted.**

Wave B (implementation + training; both spawned concurrently, GPU work
serialises through `gpu_lock.sh`):

| agent | scope | writes | status |
|---|---|---|---|
| **B1-gnn-impl** | Dynamic-resolution GNN, N=20 constant nodes, no regressor. Own dataset over the raw `.pt` files feeding the unmodified `model/gnn_dyn.py`; geometry sanity-check before training; predict particles then rasterise via `common.data.rasterize_particles`. | `Baselines/GNN/**` | running |
| **B2-nfd-impl** | NFD non-FiLM on `UNetModels_modular.UNet`. Faithful **3-channel** action encoding first, then the 2-channel-union ablation. | `Baselines/NFD/**` | running |

| **B3-schenck-impl** | Schenck CNN, **single-tower ablation only** (16x Conv 32@3x3 + ReLU, no pooling, Conv 1@1x1, residual output, L2). Two-tower scoop-and-dump + inter-tower mass channel deliberately out of scope. Told to reuse B2's action rasterisation rather than invent a third one. | `Baselines/SchenckCNN/**` | running |

### Bug fixed in shared infrastructure — `gpu_lock.sh` ate `PYTHONPATH`

**B2-nfd's first training died 24 s after acquiring the lock** with
`ModuleNotFoundError: No module named 'Baselines'`, leaving `Baselines/NFD/runs/`
empty. Cause: `gpu_lock.sh` starts a fresh `bash`, and a caller's
`PYTHONPATH=. cmd` prefix assignment does not survive being passed to it as
arguments. Every baseline imports `Baselines.*` and `training.*` by absolute
package path, so this would have hit every agent in turn.

**Fixed in `Baselines/common/gpu_lock.sh`**: it now resolves the repo root from
its own location and exports it into `PYTHONPATH` itself (idempotently, and
without clobbering a caller-set value). Verified: `bash
Baselines/common/gpu_lock.sh python -c "import Baselines.NFD.nfd_lib"` succeeds,
and a 1-epoch NFD run completes end to end through the wrapper.

Worth recording because the failure was silent in the worst way — the wrapper
reported "acquired", the process exited 0 from the shell's point of view, and
the only evidence was an empty `runs/` directory. **A training that produces no
checkpoint did not train, whatever the exit code says.** Check for artefacts,
not status.

(A second, unrelated crash — `val_pct: 0` making `Trainer.from_config`'s val
split empty, raising `ValueError: No configs found for dataset.` — B2 had
already found and fixed itself by setting `val_pct/test_pct: 5`. Held-out
scoring is entirely on the disjoint `_eval` cells, so an in-train val split
only affects monitoring, exactly as in the UNet-FiLM precedent.)

### Operational pattern — the ORCHESTRATOR owns the waiting

Instructing agents to block in-turn did NOT work. B1 and B2 both ended their
turns while their training was queued or running, and B2 did so again
immediately after being explicitly told not to — it was ~180k tokens deep, and
a context-heavy agent bails out of long waits regardless of instruction.

**The pattern that works:** the orchestrator runs one background waiter per
training process (`while kill -0 <pid>; do sleep 30; done`) and wakes the
owning agent only when there is *work to do*, never work to *wait for*. Agent
resumes then carry short, concrete instructions (score these two cells, run the
K-curve, commit) with no waiting in them.

Corollary for anyone extending this: do not spend agent budget on polling. An
agent woken to do 10 minutes of scoring is cheap; an agent sitting in a poll
loop burns context and then quits anyway.

### Operational lesson — agents must not end their turn mid-training

B1 launched a 500-epoch run in the background and then **ended its turn**,
which stops the agent and requires a manual resume. The training itself was
unaffected (it is a detached process holding the GPU lock), but the agent has
to be woken to score it. B2 and B3 were briefed explicitly not to do this:
either run training in the foreground with a generous timeout, or poll until it
exits. Anyone spawning further agents should carry that instruction forward.

### What wave A established

- **NFD's action encoding is the live architectural question.** The paper
  renders the pusher twice — start pose and end pose — as **two separate
  channels** (`in_channels=3`). This repo's pipeline unions them into ONE
  channel with a 0.5/1.0 intensity trick (`in_channels=2`). `draw_plate_soft`
  is already called twice per sample, so the faithful form is nearly free.
  B2 runs both arms.
- **FiLM is this repo's addition, not the paper's** — no conditioning vector
  appears anywhere in NFD, and `NFDUNetFilm.py`'s own docstring cites Perez et
  al. AAAI'18. Better still, the physics vector it conditions on is
  **dataset-wide constant** here, not per-sample, so dropping FiLM costs
  essentially nothing on this data. The user's preference for the modular UNet
  is a faithfulness *gain*, not a compromise.
- **NFD's loss is plain squared Frobenius (~MSE)**, no mass term. The repo's
  current best model adds `mass: 0.2` beyond the paper.
- **The GNN is tiny** — 38,403 params, 64-dim hidden, 3 message-passing rounds
  (`pstep=3`, hardcoded, not in any config). The pusher is NOT a node: the
  push becomes a per-node `s_delta` vector field masked to the swept lane, and
  it also shapes graph topology, since edges are built on the *anticipated
  post-action* positions `s_cur + s_delta`.
- **Schenck** is worth reproducing only as its single-tower ablation (~16-layer
  plain 3x3 conv stack, no pooling, residual output, L2). The headline
  two-tower scoop-and-dump architecture exists to separate lift/carry/pour
  phases our plate push does not have.

Wave C (exploratory, lowest priority, only if B is healthy): Gaussian Splatting
VMPC from rendered rasters of the stored states. The user's framing: render the
stored cube poses to images (`simple_mpc/adapters.py`'s `_w2c` already carries
the camera projection the Eulerian adapter uses), then run the paper's pipeline
on those. Attempt-and-skip: if it does not make sense on this data, say so and
stop rather than burning the night on it.

---

## Open questions for the user

_(answered ones move to the decisions table above)_

- None outstanding at spawn time.

## Verified data facts (orchestrator, measured — not inferred)

Checked directly because both bear on every baseline and this repo has a
history of frame-convention bugs (C-018):

1. **Step chaining.** Env `i` at step `k+1` starts from env `i`'s own outcome
   at step `k` (position agreement 1.4e-4 m max). So the action executed at
   step `k` by env `i` is row `i` of that step's batch, and **multi-step
   rollout training is reconstructable** — this RETRACTS hazard 3 of
   `Baselines/GNN/SPEC.md`, amended in place.
2. **Same-state structure.** Step 0's 128 envs share one start state to 1.5e-11;
   by step 1 they have diverged (spread 0.95). This is why control utility is
   scored at step 0 only.
3. **`angles` is the blade FACE orientation**, `angles - atan2(p_stop-p_start)
   == 90 deg (mod 180 deg)`, sd 1.9e-7. Mod 180, not 360 — so **the heading
   cannot be recovered from `angles`**. Always take the heading from
   `p_stop - p_start`. Confirms and sharpens hazard 1 of the GNN spec.

---

## Running notes

- **2026-09-08, setup.** Branch created, `gpu_lock.sh` written, this log
  started. Confirmed: `genesis` 1.3.3 imports, CUDA available, torch 2.11.
  `Baselines/GNN/model/gnn_dyn.py` is byte-identical to the repo's own
  `model/gnn_dyn.py` (vendored at the initial commit), and
  `simple_mpc/adapters.py` already carries a `GNNAdapter` — so the
  dynamic-resolution integration is a *training-data and training-loop*
  problem, not a porting problem.
- **2026-09-08, wave A running.** User redirected the NFD baseline onto
  `UNetModels_modular` mid-flight; A2-grid-specs was messaged with the
  corrected target and the `unet-modular` config-key list before it finished,
  so no rework was needed.
