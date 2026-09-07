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

## Hardware constraint

One RTX 4070 Laptop, **8 GB**. Two trainings at once will OOM and take both
down. Every CUDA job goes through `Baselines/common/gpu_lock.sh` (flock, 6 h
wait), so agents may run concurrently while GPU work serialises.

---

## Agent roster

| agent | scope | log | status |
|---|---|---|---|
| _(filled in as agents are spawned)_ | | | |

---

## Open questions for the user

_(answered ones move to the decisions table above)_

- None outstanding at spawn time.

---

## Running notes

- **2026-09-08, setup.** Branch created, `gpu_lock.sh` written, this log
  started. Confirmed: `genesis` 1.3.3 imports, CUDA available, torch 2.11.
  `Baselines/GNN/model/gnn_dyn.py` is byte-identical to the repo's own
  `model/gnn_dyn.py` (vendored at the initial commit), and
  `simple_mpc/adapters.py` already carries a `GNNAdapter` — so the
  dynamic-resolution integration is a *training-data and training-loop*
  problem, not a porting problem.
