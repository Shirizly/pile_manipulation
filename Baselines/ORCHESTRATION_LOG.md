# Baselines — orchestration log

**Owner:** orchestrator session (Claude Opus 5). Branch `baselines/overnight`,
off `VisualForesight` @ `9481c3e7`. Started 2026-09-08.

---

## SUMMARY FOR HANDOFF

### Bottom line (2026-09-08, overnight run complete)

**Three of the four in-scope papers were reproduced, trained on the pooled
L20+L40 data, and scored on both held-out eval cells with the register's
standard metrics. The fourth was assessed and correctly skipped.**

| model | paper | `accuracy` L20/L40 | `slateK_exact` K=128 L20/L40 | us/candidate @K=128 |
|---|---|---|---|---|
| **NFD (UNet-modular, non-FiLM)** | Neural Field Dynamics | 0.416 / 0.512 | **0.965 / 0.993** | **38.9** |
| **Schenck CNN (single-tower)** | Learning Robotic Manip. of Granular Media | **0.512 / 0.559** | 0.946 / 0.994 | 384 |
| **GNN (N=20, no res-regressor)** | Dynamic-Resolution Model Learning | 0.253 / 0.399 | 0.966 / — | 832 |
| _linear (switched, per-cell)_ | Suh & Tedrake (existing) | _0.301 / 0.447_ | _0.967 / 0.979_ | _180_ |
| _UNet-FiLM (per-cell, existing)_ | — | _0.419 / 0.504_ | _0.982 / 0.992_ | — |
| _Gaussian Splatting VMPC_ | — | **skipped — collapses to the GNN** | | |

**The recommendation, if one model has to be picked for MPC work: NFD.** It is
simultaneously the cheapest at MPC pool sizes (38.9 us/candidate, 4.6x cheaper
than the linear operator), top or near-top on control utility at every K on
both cells, level with the per-cell UNet-FiLM on accuracy despite being trained
pooled and without FiLM, and 30.5k parameters.

**Two claims made earlier in this log were overturned by measurement** and are
corrected in place below — the NFD 3-channel action encoding (a null result,
not the expected win) and the GNN's multi-step reconstructability (it IS
reconstructable). Both corrections are worth more than the results they
qualify.

**Everything is on branch `baselines/overnight`, committed, nothing pushed.**

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

## Reference numbers — READ THIS BEFORE QUOTING ANY COMPARISON

**There are TWO reference tables and they are not interchangeable.** The first
scored baseline (GNN, L20mm, 02:05) made the trap concrete.

### (a) Per-cell references — the pre-existing register numbers

From `runs_expB/*_accuracy.json`, each fit/trained on its OWN cell only:

| model | `accuracy` L20mm | `accuracy` L40mm |
|---|---|---|
| persistence | 0.000 | 0.000 |
| mean-delta | 0.086 | 0.138 |
| linear operator | 0.301 | 0.447 |
| UNet-FiLM (per-cell) | 0.419 | 0.504 |

### (b) Pooled references — the ones our baselines must be compared against

`Baselines/common/eval_baseline.py` refits mean-delta and the linear operator on
the **pooled L20+L40 train set**, which is what our baselines train on. Measured
on L20mm eval:

| model | per-cell fit | **pooled fit** |
|---|---|---|
| mean-delta | 0.086 | **0.054** |
| linear operator | 0.301 | **0.200** |

**Pooling costs the linear operator 10 accuracy points on L20mm** (0.301 ->
0.200). That is not noise and not a bug: one linear map cannot serve two push
lengths, so it fits a compromise. Every baseline's own
`<tag>_accuracy.json` carries the correct pooled comparators in the same file
as its own score — **use those rows, they are matched by construction.**

**Do not quote a pooled model against a per-cell reference.** Doing so would
have made the GNN look like it lost to the linear operator (0.253 vs 0.301)
when in the matched comparison it beats it (0.253 vs 0.200). This is
`docs/experiments/METRICS.md`'s standing warning — "compare within a
configuration; across configurations, say so" — biting immediately.

**RESOLVED by the user, 2026-09-08** — and the resolution is the more
interesting framing, so read this rather than treating pooling as a handicap to
be corrected:

> *"it is fine if only the switched-linear uses separate models for push
> lengths, it creates a certain weakness in the switched-linear, that it can't
> optimize on length itself."*

So **do not handicap the switched-linear to make the comparison symmetric.**
Per-cell fitting is the switched-linear method operating *as intended* — that is
what "switched" means. Report it at its best, and state the cost it pays for
that: **a per-length model cannot treat push length as a decision variable.**
An MPC using it can only choose among the lengths it has models for, and cannot
optimise length continuously or generalise to an unseen one. Every learned
baseline here takes the action (length included) as input and covers both
lengths with ONE model.

The headline comparison is therefore: **one pooled learned model vs the
switched-linear's best per-cell model**, with the switched-linear's structural
limitation named alongside. Keep the pooled-linear row too — it isolates how
much of the linear operator's strength comes from switching rather than from
linearity — but it is a diagnostic, not the headline.

### The switched-linear at its intended operating point (per-cell, from `runs_expB/*_kcurve_exact.json`, goal=corner, 20 slates)

| | `accuracy` | `slateK_exact` K=32 | K=128 | `worstK` K=4 |
|---|---|---|---|---|
| **L20mm** mean-delta | 0.086 | 0.3483 | 0.2871 | 1.9016 |
| **L20mm** linear (switched) | 0.301 | 0.9718 | 0.9670 | 1.0299 |
| **L20mm** UNet-FiLM | 0.419 | 0.9777 | 0.9821 | 0.7406 |
| **L40mm** mean-delta | 0.138 | 0.8007 | 0.7166 | 1.5865 |
| **L40mm** linear (switched) | 0.447 | 0.9797 | 0.9790 | 0.6150 |
| **L40mm** UNet-FiLM | 0.504 | 0.9919 | 0.9924 | 0.4847 |

**Against this, the GNN's L20mm `slateK_exact` of 0.966 with ONE model matches
the switched-linear's 0.967 achieved with TWO** (and beats the single pooled
linear map's 0.952). On image accuracy the GNN is still behind the switched
per-cell linear (0.253 vs 0.301) — reported plainly, not smoothed over.

### `goal=center` is degenerate here — use `corner`

The harness reports `goal=center: helpful 0%` on these slates. That is claim
**C-040** (`docs/experiments/REGISTER.md`): a centred convex target is
degenerate for a centred pile, `dV = 0` identically. **Report control utility
under `corner`**, the register's standard, and treat `center` as a null check.

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

### The exclusive GPU lock was wrong — replaced with a 3-slot semaphore

The lock was built on the assumption that two jobs on one 8 GB card would OOM.
**Measured, that was wrong by an order of magnitude.** With a 100-epoch NFD
training live, the card reported **0.39 GB in use of 8.19 GB** — every baseline
here is tiny (NFD UNet ~30k params, GNN ~38k, Schenck ~140k).

The lock had stopped being a safety device and become the critical path: at
02:02 a **90-minute NFD training was blocking a ~5-minute GNN EVALUATION**, and
Schenck's whole training behind that. Serialising three jobs that together want
~1.2 GB of an 8 GB card buys nothing and costs the night.

`gpu_lock.sh` is now a **counting semaphore, `GPU_LOCK_SLOTS` (default 3)** —
still bounded, so a genuinely large future model cannot be swamped by unbounded
concurrency, but no longer serialising work with no reason to serialise.
**Evaluation runs do not need the wrapper at all**; use it for training.

Requeued by hand at 02:03 so the change took effect immediately rather than
after NFD drained: Schenck's blocked training was killed and relaunched under
the semaphore (now running *alongside* NFD), and the GNN's blocked evaluation
was run directly without the wrapper.

**Correction, measured after the change (02:12).** The semaphore did NOT buy
throughput, and the original reasoning was only half right. Memory was never
the binding constraint (0.89 GB with three jobs live), but **compute is** —
NFD's epoch time went 54 s -> 90 s -> 128 s as the second and third job joined,
i.e. roughly 1/N each. Total GPU throughput is about the same either way.

What the semaphore actually bought is **latency for short jobs**: a 5-minute
evaluation no longer waits 90 minutes behind a training. That was the real
problem and it is genuinely fixed. But do not expect N trainings to finish in
the time of one.

Generalisable lesson, restated correctly: **measure the resource before
designing the contention policy, and be clear which resource you are
contending for.** Memory determines whether jobs CAN coexist; compute
determines whether coexisting HELPS. Here the answer was "yes they can, and it
helps only the short ones". For the real, larger models this repo is headed
toward, both constraints will bind and `GPU_LOCK_SLOTS=1` may be right again —
the knob is there.

### Do not edit a running script in place

Rewriting `gpu_lock.sh` while other processes were about to `bash` it caused a
transient syntax error in one agent's job (a partial read of the file), which
self-resolved on retry. Harmless here, but the safe form is write-new +
`mv` (atomic rename), not in-place rewrite.

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

Wave C — Gaussian Splatting VMPC: **ASSESSED AND SKIPPED, 2026-09-08.**
Full reasoning in `Baselines/GaussianSplatting/ASSESSMENT.md`; the short version
is below. Cost ~25 tool calls and no GPU, which is what a gated task is for.

---

## RESULTS

All figures are **pooled-trained** models scored on held-out eval cells, with
the **pooled** reference rows (see the reference-numbers section — do not
compare these against the per-cell register table).

### `accuracy` (swept region, all 3 steps, n=7680 per cell)

| model | L20mm | L40mm |
|---|---|---|
| persistence | 0.0000 | 0.0000 |
| mean-delta (pooled fit) | 0.0544 | 0.118 |
| linear operator (pooled fit) | 0.2004 | 0.381 |
| **GNN (dynamic-resolution, N=20)** | 0.2527 | 0.399 |
| **Schenck CNN (single-tower, 4ch)** | **0.5117** | **0.5593** |
| **NFD (UNet-modular, 3ch, no FiLM, no mass loss)** | 0.4158 | 0.5124 |
| _(per-cell UNet-FiLM, not pooled — context only)_ | _0.419_ | _0.504_ |

### Control utility, `goal=corner`, step-0 slates (20 slates x 128 candidates)

| model | cell | `slateK_exact` K=128 | `regret_dv` K=32 | `regret_dv` K=128 | spearman |
|---|---|---|---|---|---|
| linear | L20mm | 0.952 | 0.0020 | 0.0025 | 0.877 |
| **GNN** | L20mm | **0.966** | **0.0012** | **0.0018** | **0.948** |
| linear | L40mm | — | — | **0.0016** | — |
| GNN | L40mm | — | — | 0.0031 | — |

### NFD (non-FiLM, 3-channel) — matches the FiLM model it replaced

`accuracy` 0.4158 / 0.5124 against the per-cell UNet-FiLM's 0.419 / 0.504 —
**level on L20mm and slightly ahead on L40mm, while trained POOLED rather than
per-cell, without FiLM, and without the repo's extra `mass` loss term.** So the
paper-faithful, simpler model gives up nothing to the repo's elaborated one.
That is the cleanest confirmation of the wave-A finding that FiLM was a repo
addition conditioning on a dataset-wide constant.

Control utility, `goal=corner`:

| | `slateK_exact` K=32 | K=128 |
|---|---|---|
| NFD L20mm | **0.9751** | **0.9646** |
| linear (pooled) L20mm | 0.9524 | 0.9521 |
| NFD L40mm | **0.9893** | **0.9925** |
| linear (pooled) L40mm | 0.9787 | 0.9887 |

NFD beats the pooled linear operator at every K on both cells, and on L40mm
(0.9925 at K=128) it edges the per-cell UNet-FiLM's 0.9924 and the per-cell
switched-linear's 0.9790. On L20mm its 0.9646 sits just under the per-cell
switched-linear's 0.9670 — a 0.2-point gap on 20 slates, which is well inside
the noise this metric carries at K=128 (METRICS.md: ties collapse the effective
n to 5-11 pools). Do not read it as a real ordering either way.

**NFD is the best all-round baseline of the three**: near-top accuracy, top or
near-top control utility on both cells, no dissociation, and ~30k parameters.

### The 3-channel action encoding is a NULL RESULT — correcting an earlier claim

Wave A framed the 2-channel union as the repo losing information the paper
keeps, and this log called it "the live architectural question". **Measured, it
is not.** Same recipe, same architecture, only the action encoding changed:

| | `accuracy` L20 / L40 | `slateK_exact` K=128 L20 / L40 |
|---|---|---|
| 3-channel (paper-faithful) | 0.4158 / 0.5124 | 0.9646 / 0.9925 |
| 2-channel (repo's union) | **0.4211 / 0.5145** | 0.9580 / 0.9904 |

Val loss 0.006445 (3ch) vs 0.006459 (2ch). The union is very slightly ahead on
accuracy, the split very slightly ahead on control, and every gap is far inside
what 20 slates can resolve. **The split buys nothing measurable at this model
and dataset scale**; the repo's existing encoding was not costing anything.

Recorded because the opposite was asserted earlier in this log on the strength
of reading the paper alone. Running the ablation is why we know.

### Schenck CNN — best accuracy so far, but it dissociates at K=128

**Accuracy 0.512 / 0.559 beats the per-cell UNet-FiLM (0.419 / 0.504) on both
cells despite training pooled rather than per-cell** — i.e. it wins against a
model given the easier task. On raw image accuracy this is the strongest
baseline in the set.

Control utility does not follow, on L20mm:

| | `slateK_exact` K=32 | K=128 | `regret_dv` K=32 | K=128 |
|---|---|---|---|---|
| Schenck L20mm | 0.9826 | **0.9455** | 0.0007 | 0.0027 |
| Schenck L40mm | 0.9900 | 0.9938 | 0.0013 | 0.0009 |

At K=128 on L20mm it falls to 0.9455 — **below the per-cell switched-linear
(0.9670) and below the GNN (0.966), while having roughly double their image
accuracy.** That is a textbook instance of this register's central finding
(C-030/C-035/C-044): image accuracy and control utility dissociate, and the
dissociation bites hardest under maximum selection pressure. Note it does NOT
appear on L40mm, where K=128 is the model's best point (0.9938).

Config: 4 input channels — occupancy, two separate full-intensity plate renders
at `p_start`/`p_stop` (reused from NFD's faithful branch), plus a tiled constant
`angle` channel per the paper's angle-tiling recipe. 139,937 params, Adam 5e-4,
loss 0.0172 -> 0.0049. Axis-convention check passed on both an x-dominant and a
y-dominant push before training. No hyperparameter search was run.

### Reading the GNN result honestly

**On L20mm the GNN wins on both axes** — accuracy +5.2 pts over the pooled
linear operator, `slateK_exact` higher at every K, regret lower at every K,
spearman 0.948 vs 0.877. That is *not* the usual pattern in this register,
which is largely a record of image accuracy and control utility dissociating
(C-030/C-035/C-044).

**On L40mm it does not.** The accuracy margin shrinks to +1.8 pts (0.399 vs
0.381), and at K=128 the linear operator is *better on ranking* — `regret_dv`
0.0016 vs the GNN's 0.0031 — while still trailing on accuracy. So the
dissociation the register keeps finding reappears at the longer push length,
in the direction the register would predict. This is reported as measured; it
is one cell at one K and should not be over-read, but it must not be dropped
from the summary either.

Plain reading: **the GNN is a working, competitive dynamics model that clearly
beats the linear operator on the shorter push, and is roughly a wash with it on
the longer push once control utility rather than image error is the criterion.**

### Geometry constants (verified before training, not guessed)

`adj_thresh = 0.012 m` (2-3x cube edge; checked min in-degree 2-4, never
isolated, never saturated at the top-10 cap), `pusher_w = 0.02 m` (sourced from
the blade's `plate.size` in a cell config, not from the paper's normalised
units), `softness = 0.01 m`. The `s_delta` mask was visually confirmed to light
a few cubes for a 20 mm push and nearly all 20 for a 40 mm push — a longer
sweep window, not a bug. Tooling: `Baselines/GNN/scripts/check_geometry.py`.

## Wave C verdict — Gaussian Splatting: SKIP (and why that is a real finding)

In the paper, 3DGS is a **state representation**, fitted per frame from
calibrated multi-view RGBD. The **dynamics model on top of it is a
PropNet/DPI-Net-style GNN** predicting per-node translation and rotation under
a Chamfer loss, and its cost function is a KDE-style density field —
structurally the same idea as our occupancy-based goal reward.

Our data has no images at all (verified: `*_data.pt` carries only poses and
pushes), and this codebase uses a single camera throughout — there is no
multi-view rig to reconstruct from. Neither `gsplat` nor
`diff-gaussian-rasterization` is installed in `pme`.

**The disqualifying finding is not "too hard" — it is that it collapses.**
Because every cube's pose and 5 mm size are known exactly, the reconstruction
step is the *identity map*. Take that shortcut and the paper's dynamics model
is architecturally the same interaction-network GNN we have already built,
trained and scored in `Baselines/GNN/`. Building it again under a different
name would have produced a second copy of an existing row, at the cost of
several GPU-hours.

**The one genuine difference is worth keeping as a cheap follow-up**, filed
against the GNN rather than under this name: the paper predicts **per-node
rotation**, while our GNN reattaches the input quaternion unchanged. That is a
small separable ablation of a model that already exists.

## Compute cost — CLEAN RUN on an idle GPU (supersedes the provisional table)

`Baselines/common/benchmark_time.py`, full detail in `Baselines/common/TIMING.md`
and `timing_results.json`. One state, K candidate actions — the MPC inner loop.
CUDA-synced, 3 warm-ups discarded, median [IQR] over 10 repeats.

**Per-candidate microseconds (lower is better):**

| model | params | K=1 | K=32 | K=128 | K=1024 |
|---|---|---|---|---|---|
| **NFD (UNet 3ch)** | 30,541 | 2166 | 80.6 | **38.9** | **37.2** |
| mean-delta | 4,096 | 872 | 171.6 | 70.5 | 57.0 |
| linear operator | 16,777,216 | 5043 | 296.4 | 180.1 | 162.6 |
| Schenck CNN | 139,937 | 1419 | 203.5 | 384.2 | 382.5 |
| GNN (after rasteriser batching) | 38,403 | 2296 | 850.9 | 685.3 | 671.9 |

Read the K=1 column as fixed per-call overhead, not model cost; the K>=128
columns are the marginal cost that actually matters to an MPC pool.

**Three things this settles:**

1. **NFD is the cheapest model at MPC pool sizes AND the best all-round on
   quality** — 38.9 us/candidate at K=128, 4.6x cheaper than the linear
   operator and 21x cheaper than the GNN, while matching per-cell UNet-FiLM on
   accuracy and leading on `slateK_exact`. It amortises properly (80.6 -> 38.9
   -> 37.2 as K grows), which is exactly the batching behaviour an MPC wants.
2. **The GNN's cost is NOT mostly its rasteriser — prediction tested and
   largely wrong.** The unbatched per-candidate Python loop was the obvious
   suspect (flat cost in K), so it was batched and re-measured. Accuracy is
   unchanged on both cells (0.2527 / 0.3991, confirming the batched grids stay
   equivalent), but per-candidate cost only fell **832 -> 685 us at K=128, ~18%**,
   and it is still nearly flat in K (851 -> 685 -> 672). The GNN remains ~18x
   NFD. So the earlier claim in this log that batching "would likely move it
   into the NFD band" is **retracted**. The remaining cost is **20 separate
   per-particle `cv2.fillPoly` calls per candidate** — the batching removed the
   Python loop over candidates and vectorised the coordinate/quaternion maths,
   but each candidate's own per-particle CPU rasterisation is untouched.
   (An earlier orchestrator guess that graph construction was the cause is also
   wrong — the agent's measurement locates it in cv2, not the network or the
   graph.) Collapsing the 20 calls into one is the obvious next step and is
   **not safe naively** — see the fillPoly trap below.

   **A real trap, caught before it corrupted any score:** collapsing the 20
   per-particle `fillPoly` calls into a single multi-polygon call looks
   equivalent and is not. `cv2.fillPoly` fills a polygon list under an
   **even-odd rule**, which XORs overlapping regions — and these cube piles
   overlap constantly, so overlapping cubes would have been erased to zero.
   `Baselines/GNN/scripts/verify_rasterizer.py` caught it before it reached
   scoring. Any future attempt at this needs a union-aware fill, not a
   polygon list.

   Equivalence of the batched path was verified **bit-identical** (max abs
   diff 0.0) against the original on 400 real eval candidates, and both cells
   re-scored unchanged (L20mm 0.25271278619766235 to full float precision).
3. **The "cheap classical baseline" is the most expensive model in the set by
   parameter count** — the linear operator is a dense 4096x4096 map, 16.8M
   parameters, 550x the NFD, and 4.6x its per-candidate cost. Worth saying
   whenever the linear operator is described as the lightweight option.

Schenck is the one model whose per-candidate cost RISES from K=32 to K=128
(203 -> 384) and then flattens — a 16-layer full-resolution conv stack with no
pooling saturates the card sooner than the others.

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
