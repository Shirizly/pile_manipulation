# Plan — validating the model-selection results under real selection pressure

**Written:** 2026-09-06, opening the validation session that follows
[`handoff_model_selection_for_mpc.md`](handoff_model_selection_for_mpc.md).
**Purpose:** take the register's load-bearing claims and attack them where they
are weakest, with experiments large enough that the result either majorly
supports or refutes them. Nothing here restates numbers; they live in the
records.

---

## 1. The weakness the whole register shares

Every control number in the register — C-035, C-039, C-044, C-045, and the
handoff's headline "a ridge operator already captures 95% of the oracle's
advantage" — is `slate4` measured on
`Genesis/data/slates/n20_heap_5mm`: **pick the best of 4 randomly drawn
candidates from a 32-action slate.** Three separate things are wrong with that
as a stand-in for MPC utility, and they compound:

1. **K=4 is almost no selection pressure.** Real MPC compares hundreds to
   thousands of candidates per step (`simple_mpc/sampling_optimizers.py` CEM/MPPI
   populations; the gradient loop's `n_sample` batch). A model only has to beat a
   random pick out of 4 to score well, and the ceiling is close enough
   (0.950 for ridge) that no comparison run on it can produce a large effect —
   the handoff correctly reads this as "no headroom", but the missing
   possibility is that **the headroom is hidden by the metric, not absent from
   the domain**. Selection error concentrates at the top of the candidate
   distribution, and only large K probes there.
2. **The candidates are random, not optimized.** An optimizer does not sample
   the action space uniformly; it walks toward wherever the *model* says the
   value is high, which is precisely where model error is most likely to be
   exploited. `slate4` cannot see the optimizer's curse at all, and the
   exploitation gap is a per-model quantity that could reverse the ranking.
3. **This project's MPC is gradient descent** (`simple_mpc/mpc.py`: Adam on
   `act_seqs`, backprop through the adapter). Not one record measures whether a
   model's **action-gradient** points anywhere useful. Ranking a fixed slate and
   descending a model's action-gradient are different skills, and a model can
   have one without the other. `indirectness` is not currently scored for this
   on EXP-0024, and it should be.

So the programme below is organised around one axis — **selection pressure** —
from K=4 (what we have) through K=256 random, to continuous optimization by
sampling and by gradient descent (what MPC actually does). Each step is a
strictly harder test of the same claims, on the same domain, with the same
models, so a change in verdict cannot be blamed on a change of scenario.

**Both directions are informative.** If the ridge operator still captures ~95%
under CEM-1024 and its gradient is as useful as the UNet's, the handoff's
"switch model classes only where the oracle gap is large" recommendation is
confirmed on much stronger evidence and the cheap model wins outright. If it
collapses at K=64 while the UNet holds, C-045's "and it hardly matters" is
wrong, and every downstream compute-allocation conclusion changes.

---

## 2. What each critical claim rests on, and the expanded test

| claim | what it rests on now | weakest link | expanded test |
|---|---|---|---|
| **C-045** UNet's control edge is real but small (ridge = 95% of oracle) | EXP-0024, 49 slates, `slate4`, one UNet seed, one fit | K=4; random candidates; single seed | **EXP-A** (K-curve) → **EXP-B** (dense slates) → **EXP-C** (optimizer) → **EXP-F** (seeds) |
| **C-044/C-041** UNet's advantage is entirely high-frequency, and control does not consume that band | EXP-0022 (σ=1 common target), EXP-0024 (control at K=4) | "control does not consume it" was only ever measured at K=4 | **EXP-E** band-limited ablation × K |
| **C-035/C-039** hf-noise destroys ranking; amplitude/blur nearly free | EXP-0008, EXP-0025, all at `slate4`/spearman | ditto — "blur is free" may be a K=4 artifact | **EXP-A** and **EXP-E** re-run the degradation arms across K |
| **C-030** a signal-sensitive metric ranks models for MPC better than pixel rms | EXP-0007/0008/0017 | never tested against an actual optimizer's realized outcome | **EXP-C** gives the criterion metric the others are proxies for |
| handoff §4 "no inference cost measured" | nothing | — | **EXP-A2** latency/batch-scaling pilot |
| handoff §4 "everything is one-step" | nothing | — | **EXP-G**, sequenced last |

---

## 3. The experiments

Each is stated as the plan gate the `experiment-log` skill requires: claim,
prediction with a threshold, design, cost, and the result that would most
embarrass it. Predictions are to be committed to git **before** the
corresponding run.

### EXP-A — the selection-pressure curve (T2, free)

**Claim.** The ridge operator's near-oracle control utility is an artifact of
K=4: its captured fraction falls materially as K grows, and the UNet−linear gap
grows with K.

**Prediction (pre-registered).**
- *supports* — linear `slateK` at K=31 (the whole slate) is ≤ 0.90, i.e. at
  least 5 points below its `slate4` of 0.950; **and** the paired UNet−linear
  gap at K=31 is at least twice its K=4 value (+0.037 or more), against the
  paired sem over 49–50 slates.
- *refutes* — linear `slateK` at K=31 stays ≥ 0.94 and the paired gap does not
  exceed gap(4) + 1 sem. Then K=4 was not the problem, and C-045 stands on
  much stronger evidence.
- *discriminating*: yes — the two branches differ in the same measured
  quantity by more than the sem, on data already collected.

**Design.** Re-score the existing EXP-0024 predictions (persistence,
mean-delta, ridge→I, UNet, oracle) and EXP-0025's full degradation spectrum on
the same 49–50 slates, sweeping K = 2, 4, 8, 16, 24, 31. **Sampling without
replacement** — `control_utility_test.rank_metrics` currently draws slates with
`torch.randint`, i.e. *with* replacement, so its existing `slate16` is not a
16-distinct-candidate test and must not be quoted as one. At K = slate size the
metric becomes deterministic per slate (top-1 of everything), so the floor is a
bootstrap over slates, paired.
Report alongside the bounded `slateK`: **unnormalized regret in Lyapunov
units** and the **true-rank percentile of the chosen action**, because a
difference of two bounded metrics is forced toward zero (skill file, "Bounded
metrics have a ceiling") and `slateK`'s denominator itself grows with K.
Finally, fit a per-model Gaussian-copula noise model (ρ between predicted and
true dV, plus tail behaviour) and **extrapolate to K=10²–10³**, recording the
extrapolation as a prediction that EXP-B/EXP-C will test.

**Cost.** CPU only; re-running `exp0024_control_eval.py` is ~25 s. Half a day of
analysis at most. **Cheapest invalidating check:** confirm ≥31 usable
candidates per slate (already verified: 47 slates at 32, 3 at 31).

**What would embarrass it.** A flat curve for both models with the gap
unchanged — which is a real possibility and is exactly why this runs first, for
free, before any sim time is spent.

**OUTCOME, 2026-09-06 — [EXP-0026](experiments/EXP-0026-selection-pressure-curve.md),
prediction REFUTED.** Linear `slateK` 0.958 at K=4 and 0.958 at K=31 (never
below 0.94); the paired UNet−linear gap is flat at +0.0164 → +0.0177, below
`gap(4) + 1 sem`. Top-1-of-31 is no harder, relatively, than top-1-of-4, so
weakness §1.1 is closed on this domain — while `regret_dv` grows 2.7×, so the
*absolute* value left on the table does scale with K even though the captured
fraction does not. The degradation arms give the mechanism and are the finding
worth carrying forward (now C-046): **selection pressure amplifies only errors
that are independent per candidate.** Systematic arms are K-invariant (blur
s=1.0 0.951→0.960), hf-noise is not (m=1.0 0.878→0.802, excess regret ×4.7).
The Gaussian-copula extrapolation to K=10²–10³ was fitted and **failed
validation** on the measured range, in level and in direction, so the
"hundreds to thousands" question is genuinely open and EXP-B is not optional.

**Known discrepancy to resolve inside this run (5 min).** (Resolved: it was
neither the length filter nor a bad file — `PileSweepData._assign_group_splits`
cannot put every group in `test` for any `test_pct`, so one slate was always
withheld. New config `genesis_cube_spectrum_n20_slates_all.yaml` loads all 50.) A direct length filter
on the raw slate files gives 50 usable slates (47×32 + 3×31 = 1597
transitions); EXP-0024's registry path reported 49 usable and dropped one file
entirely. Two code paths disagreeing about which candidates exist is exactly a
`provenance` issue; settle it before quoting any K-curve.

### EXP-A2 — inference cost per candidate (T0 pilot, half an hour)

The handoff calls this the gate on the whole programme, and the K-curve makes it
urgent: if the UNet cannot score 1000 candidates within an MPC step budget on
this GPU, its advantage at large K is unusable regardless of size. Measure
per-candidate latency and batch scaling for the ridge operator (warp → matmul →
unwarp) and the UNet at batch 1/32/256/1024, plus memory ceiling, on the
RTX 4070 Laptop. Output: a `slateK`-per-unit-compute table — the cost-aware
metric the handoff asks for and nobody has.

### EXP-B — dense same-state slates (T1, one overnight run)

**Why.** EXP-A can only reach K=31, and its large-K answer is an extrapolation.
Real MPC candidates are also *dense* — neighbouring actions differing by
millimetres, whose true dV is highly correlated — which is a strictly harder
discrimination problem than 32 sampler draws. Both need new data.

**Design.** Reuse `Genesis/same_state_slate_collection.py` unchanged in
structure: settle one state, broadcast it to every env, execute one candidate
per env. The only change is **looping several batches per state** (the manifest
already records the state-library index per batch, so grouping stays explicit).
Two strata per state:
- **wide**: 256 candidates from the existing contact-aware sampler
  (8 × 32 envs, or fewer batches if a larger `n_envs` fits — pilot first);
- **local**: 128 candidates on a fine grid around the wide stratum's best
  region, spacing chosen so neighbours differ by ~2–3 mm in start position and
  ~5° in heading — the resolution an optimizer actually works at.

~16–20 states. Verify the same-state property across batches to EXP-0012's
standard (state identity to <1e-9 m) — this is load-bearing and is a *new* risk,
because the existing collection never re-applied a state across batches.

**Cost.** ~1 min per 32-candidate batch measured (EXP-0012: ~6 min for 6
states); 12 batches/state × 18 states ≈ 3.5–4 h. Pilot the `n_envs` ceiling
first: 32 → 64 → 128 on 8 GB, one state, timed. **Prerequisite (do it here):**
the `settled-state` residual-velocity check for the rigid-cube path. It is
`unchecked`, it is a `depends_on` of EXP-0012/0024/0025, and writing it while
new data is being collected retires an `untested-dependency` from the entire
programme for maybe 30 minutes of work.

#### EXP-B collection status, 2026-09-07 — 3 of 4 planned cells done

Collected under `Genesis/data/slates_multistep/` (data is untracked; the driver
is `Genesis/same_state_slate_collection.py`, extended for multi-step in
`4dd9a673`). Every cell is 50 states x 128 candidate ACTION SEQUENCES x 3 steps,
placement-aware starts, perpendicular, constant commanded push length, unique
actions, wall-clip rejection at the sampling stage:

| cell | slates | envs | transitions | wall clock | verified |
|---|---|---|---|---|---|
| `n20_L10mm` | 50 | 128 | 19,200 | ~48 min | yes |
| `n20_L20mm` | 50 | 128 | 19,200 | ~71 min | yes |
| `n20_L40mm` | 50 | 128 | 19,200 | ~100 min | yes |
| `n50_L20mm` | **6 of 20** | 64 | 1,152 | ~50 min (interrupted) | yes, salvaged |

Verification for all four (`scripts/probes/verify_slate_cell.py <cell> <L>`):
same-state spread across envs at step 0 between 0.0 and 4.66e-10 m — matching
EXP-0012's own independently measured 4.7e-10, i.e. float32 state-restore
precision; **0** duplicate (start, heading) pairs at 1 mm / 5 deg; realized push
length constant to ~1e-5 m with **0%** short pushes; 0 failed rows; 0 unresolved
resampling failures.

**What remains:** `n50_L20mm` slates 6-20 (the user approved 20 states, one
length, as a reduced-scope overnight cell), and the n=10 arm, which was never
piloted.

**How to resume — read this before relaunching.**

```
setsid nohup python scripts/run_probe.py --tag n50_L20mm_b --threads 4 -- \
  python -m Genesis.same_state_slate_collection \
  --n-cubes 50 --n-envs 64 --n-states 14 --n-steps 3 \
  --placement-aware --no-pile-aware --push-length 0.020 \
  --seed 1 --output-root data/slates_multistep --tag n50_L20mm_b \
  > /dev/null 2>&1 < /dev/null & disown
```

Four things that command encodes, each of which is a trap if ignored:

1. **A new `--tag`, not the existing one.** The driver's batch counter is
   `files_in_dir / 3`, so appending into `n50_L20mm/` would continue the
   numbering and break the `batch_idx == slate_idx * n_steps + step_idx`
   invariant that grouping depends on. Collect into a sibling cell and treat the
   two as one dataset at analysis time.
2. **A different `--seed`.** The state library is settled from `--seed`, so
   re-running with `--seed 0` regenerates *the same* initial states — the 6
   already collected would be duplicated rather than extended.
3. **`setsid nohup ... & disown`, not a bare `run_probe.py`.** `scripts/run_probe.py`
   does not detach its child from the launching shell's process group, so a long
   collection dies when that shell exits. This actually happened here and cost
   ~50 min of GPU on a restarted `n20_L20mm`. **Recommended fix: `setsid` (or
   equivalent) inside `run_probe.py`**, which is exactly what it exists to
   provide.
4. **`--placement-aware --no-pile-aware`.** They are mutually exclusive in code:
   `generate_action_samples` returns on the `pile_aware` branch before
   `placement_aware` ever runs, so passing both silently gives you pile-aware
   sampling.

**If a run is interrupted:** `manifest.json` is written only at the very end, and
the on-disk schema carries no step index, so a killed cell leaves batch files no
loader can group. `scripts/probes/rebuild_slate_manifest.py <cell> --apply`
reconstructs it from the verified deterministic mapping, drops any trailing
incomplete slate, and stamps `rebuilt: true` so a salvaged cell is
distinguishable from a clean one. That is how `n50_L20mm` above became usable.

**Cost model, measured on this GPU (RTX 4070 Laptop, heap spawn, placement-aware):**
cost is roughly **linear in push length** at fixed object count (48 / 71 / 100
min for 10 / 20 / 40 mm at n=20, 128 envs) but **~20x per transition from n=20 to
n=50** (>4.2 s/env-transition at n=50/64 envs against 0.107 s at n=20/128).
`Genesis/configs/measured/throughput_optimal.yaml` predicts 0.182 s/transition
at n=50 and is **not applicable** — it was measured on scattered piles with
pile-aware sampling. The wall is the solver, not the sampler
(`Genesis/placement_sampling.py` scales with env and yaw count, only weakly with
particle count), and it sits on a cliff this repo already documented elsewhere:
0.36 s/transition at n=20 vs 9.93 s at n=30 on heaps in `cube_spectrum_collection`.

**These cells are NOT drop-in comparable with `Genesis/data/slates/n20_heap_5mm`.**
Two deliberate differences: placement-aware starts are collision-free at
touchdown, and with `pile_aware=False` there is no contact-triggered early stop —
hence 0% short pushes here against ~5% in the older slates. Any comparison
against an EXP-0024/EXP-0026 number carries `provenance` until that is checked.

**Doc updates this collection earns** (not yet made): `docs/piled_collection.md`
§4 covers only pile-aware cost scaling and should carry the placement-aware heap
costs above; `docs/scaling_to_200_objects.md` §3/§3.1 should state that its
env-count table was measured on scattered piles with pile-aware sampling, since
both paths report through the same `n_envs` key and the gap at n=50 is the
difference between "fits in an evening" and "does not".

### EXP-C — optimizer-in-the-loop utility (T2, the decisive one)

**Claim.** When each model is asked to *find* a good action over the continuous
action space, rather than rank four random ones, the ridge operator no longer
captures most of the oracle's advantage, and the model classes separate.

**Prediction (pre-registered, thresholds to be fixed after EXP-A supplies the
extrapolation).** Provisionally: under CEM with N=256, the ridge operator's
capture fraction is ≤ 0.85 (vs 0.950 at `slate4`) and the UNet−ridge paired gap
exceeds +0.04, tested against the paired sem over states. *Refutes*: both stay
≥ 0.93 with a gap inside 1 sem.

**Design.** For each EXP-B state, each model (mean-delta, ridge→I, UNet)
proposes an action by real optimization, in two modes:
- **sampling**: CEM/MPPI from `simple_mpc/sampling_optimizers.py` at N=256 and
  N=1024, model-scored;
- **gradient**: multi-start Adam through the differentiable model, i.e. the
  repo's own `simple_mpc/mpc.py` loop. Both the ridge pipeline
  (`to_push_frame` → `A` → `from_push_frame`, all `grid_sample`-based) and the
  UNet are differentiable wrt the action; this needs no new model code.

**Constrain the optimization to the fixed-length manifold — see §5.1.** The
optimized variable is `(sx, sy, θ)` with the end point *derived*, never a free
4-D `[sx, sy, ex, ey]`. This is not a detail: every existing optimizer in the
repo violates it, so it is code that has to be written before EXP-C runs.

Each proposed action is then **executed in Genesis from the identical snapshot**
(`GenesisOracleEnv.rollout_candidates`, `use_rollout_fidelity=False` so the
ground truth matches the slate convention) and its true dV measured.
Ceiling: the best action in EXP-B's dense set for that state (free, already
collected), with a true Genesis-CEM oracle run on 5 states as a check that the
dense best is near the real optimum. Floor: a random draw from the sampler.

**Also report the `exploit_gap`** — the model's predicted dV for its chosen
action minus the realized dV. This is the winner's curse, per model, and it is
the quantity that decides whether more MPC compute helps or hurts. It has never
been measured here.

**Cost.** Optimization itself is seconds per state per model (EXP-A2 gives the
exact figure). Sim cost is only the *chosen* actions: ~18 states × 3 models × 2
modes × 2 population sizes ≈ 216 pushes ≈ 7 batches ≈ 15 min. The expensive
version — running Genesis itself as the CEM model at N=256 × 4 iterations for
every state — is ~30k pushes, roughly 16 h, and is deliberately replaced by the
dense-set ceiling plus a 5-state spot check.

**What would embarrass it.** Every model proposing near-identical actions
because the Lyapunov landscape on this domain has one obvious basin — in which
case the domain, not the metric, is the limit, and that is worth knowing before
any pipeline is built on it. The dense-set ceiling detects this directly (the
spread of true dV over the 256 wide candidates).

### EXP-D — are the gradients useful? (T1)

**Claim.** A model's action-gradient carries usable descent information, and the
model classes differ in how much.

**Design, two arms, both on EXP-B's dense states.**
- *Alignment*: estimate the true local ∂dV/∂(sx, sy, θ) — the same chart as §5.1 — by local-linear regression
  over the executed neighbours in the local stratum (with a noise estimate from
  duplicate actions), and compare each model's autograd gradient: cosine
  similarity and per-coordinate sign agreement. Baseline: mean-delta's gradient
  (non-trivial — it moves the canonical warp) and a random direction.
- *Realized improvement*: from a fixed starting action, take one step of size δ
  along each model's negative gradient, execute it, and compare the realized
  ΔdV against a step along the finite-difference true gradient and a
  random-direction step of equal size. Sweep δ over ~3 magnitudes to separate
  "locally right" from "usefully right".

**Prediction.** *supports*: at least one model's gradient step beats a
random-direction step of equal size by more than the paired sem across states,
at the smallest δ. *refutes*: no model's gradient beats random at any δ — which
would say gradient-descent MPC on this domain is unsupported by its own
dynamics models, a major negative result and directly actionable.

**Cost.** FD gradients are free from EXP-B. Execution: ~18 states × 4 directions
× 3 step sizes ≈ 216 pushes ≈ 15 min sim.

### EXP-E — which frequency band does selection consume, as a function of K? (T1)

**Claim (attacking C-044).** "Control does not consume the fine band" holds at
K=4 and fails under selection pressure, because discriminating between two
nearby candidate actions is exactly a fine-detail comparison.

**Design.** Laplacian-pyramid decomposition of each model's *prediction* (reuse
`scripts/probes/fss_scale_decomp.py::build_pyramid`), reconstruct with bands
progressively removed, and measure `slateK` and EXP-C's optimizer capture across
K = 4 … 256 on EXP-B's data. Hold the *target* fixed and ablate only the
prediction — blurring the target changes `accuracy`'s denominator and makes the
numbers incomparable across cells (METRICS.md, trap 2). Re-run EXP-0025's
degradation arms (amplitude, blur, hf-noise) on the same K grid in the same
pass: "blur is nearly free" is a K=4 statement and should be re-tested as one.

**Prediction.** *supports*: the control cost of removing the finest band grows
monotonically with K and exceeds 5 points of `slateK` by K=64. *refutes*: it
stays under 2 points at every K, in which case C-044's control reading is
confirmed at realistic selection pressure.

### EXP-F — model-seed and fit variance (T1, cheap, run early)

The register's dominant downgrade is `imprecision` (16/20 records), and every
UNet-vs-linear number rests on **one** training seed and **one** fit. The paired
t of ≈6 in EXP-0024's amendment is paired over *slates*, which says nothing
about how much the gap moves with the training seed. Train 3–5 UNet seeds
(~16 min GPU each) and fit the ridge operator at 3 (ridge, crop) settings;
re-score everything from EXP-A. Report the between-seed sd of the UNet−linear
gap as the honest floor for every model-class comparison in this programme, and
use *that* floor — not the across-slate sem — wherever a model-class claim is
made. This can run concurrently with EXP-A and B.

### EXP-G — horizon (T2, sequenced last)

Multi-step closed-loop episodes (5–10 steps) with each model against the
Genesis-oracle CEM ceiling, on ~10 states: the handoff's §4 open item, and the
one that checks whether the descriptor operator's rollout instability (10 of 55
eigenvalues outside the unit circle, `docs/ideas_log.md` P17) has a pixel-path
analogue. Deliberately after EXP-C: if the single-step optimizer story already
separates the models, the horizon experiment knows what to look for; if it does
not, the horizon experiment is the only place a difference could still live.

---

## 4. Order, dependencies, and what gets decided when

```
EXP-A  (free, CPU)  ──┬─→ EXP-C  (needs EXP-B) ──→ EXP-G
EXP-A2 (½ h, GPU)  ──┘        ↑
EXP-F  (1 h GPU)   ───────────┤
EXP-B  (overnight sim) ───────┴─→ EXP-D, EXP-E
```

1. **EXP-A + EXP-A2 + EXP-F first**, because they are cheap and EXP-A alone can
   already refute or confirm the central "K=4 was too easy" objection on data in
   hand. ~~If EXP-A's curve is flat, EXP-B/C shrink to a confirmation run.~~
   **EXP-A ran and its curve is flat (EXP-0026), and EXP-B/C do NOT shrink** —
   for a reason EXP-A also supplied. The K-axis is closed only for candidates
   that are (a) sparse sampler draws and (b) not chosen by the model itself.
   EXP-0026 shows the one error type selection pressure *does* punish is the
   independent-per-candidate kind, which is exactly what a dense candidate set
   (EXP-B) and an optimizer searching for the model's own optimum (EXP-C) put
   under load. The passive winner's curse is now measured; the active one is
   not.
2. **EXP-B overnight**, gated on its own `n_envs` pilot and on the
   `settled-state` check.
3. **EXP-C** is the decision point for the whole model-selection question.
4. **EXP-D/E** interpret C's result; **EXP-G** extends it.

---

## 5. Method rules carried into every record here

### 5.1 The action variable is `(sx, sy, θ)`, and the end point is derived

Push travel distance is held constant across every model, dataset and
comparison in this project (`push_length: 0.02` in the slate and
`cube_spectrum` configs; EXP-0024 filters to ≥19.9 mm precisely to keep it
so). Any optimization that treats the action as a free 4-D
`[sx, sy, ex, ey]` box therefore changes a variable that is constant
everywhere else — it walks off the training manifold, and it makes the
compared models incomparable to each other and to every `slateK` number in the
register. Every optimizer arm in EXP-C, EXP-D and EXP-G optimizes

    a = (sx, sy, θ)          e = s + L · (cos θ, sin θ),  L = 0.02 m fixed

and feeds the derived `[sx, sy, ex, ey]` to `action_to_pose`, whose 4-component
branch then derives plate yaw as `atan2(Δy, Δx) + π/2` — perpendicular, as
collected.

Three consequences that have to be implemented, not assumed:

- **The gradient must be chain-ruled into the chart, not clipped after the
  fact.** `∂L/∂θ` picks up the contribution through `e`, so the descent
  variable is the 3-vector: build `e` from `(s, θ)` inside the autograd graph
  and let `loss.backward()` flow through it. Optimizing 4-D and projecting the
  step back onto constant length afterwards is a different algorithm and gives
  a different answer.
- **Bounds become a projection in the chart.** `simple_mpc/mpc.py:331` clamps
  each of the 4 action dimensions independently, which does not preserve
  length. Feasibility (both endpoints inside the workspace, plate clear of the
  walls) must be enforced on `(sx, sy, θ)` — clip `s` to the workspace inset by
  `L` in the heading direction, wrap θ — or by rejecting infeasible candidates,
  never by per-dimension clamping of the derived 4-vector.
- **The sampling optimizers need the same chart.** `CEMOptimizer` /
  `MPPIOptimizer` sample and clip in the 4-D box
  (`sampling_optimizers.py:37–98`), so CEM at N=256 would otherwise be
  searching a superset of the manifold and would beat the gradient arm for a
  reason that has nothing to do with either model. Give both optimizers the
  3-D parameterisation, so the sampling and gradient arms search the identical
  space and only the search *method* differs.

Note the sim may still *realize* a shorter push than commanded when the blade
meets material early (~5% of collected pushes; EXP-0024's `min_push_length_m`
note). The commanded length is what is held fixed; realized length is an
outcome, and EXP-C should record it so a proposal that systematically stops
early is visible rather than silently scored.


Non-negotiable, because each one has already cost this project a verdict:

- **Compare a mean difference to its paired sem, never to an sd**, and report
  the win-rate alongside (handoff trap 1; C-008 and C-045 both flipped on this).
  For model-class claims, the floor is EXP-F's between-seed sd, not the
  across-slate sem.
- **Report an unbounded companion to every bounded metric.** `slateK` saturates
  and its denominator moves with K; regret in Lyapunov units does not.
- **New metric keys go into `METRICS.md` with their formulas before use**:
  `slateK` (explicitly without replacement, superseding the current
  with-replacement implementation), `regret_dv`, `pick_percentile`,
  `opt_capture`, `exploit_gap`, `grad_cos`, `step_gain`.
- **Never compare `accuracy` across preprocessing** — EXP-E ablates predictions
  only, with the target fixed.
- **One code path per comparison.** EXP-A must settle the 49-vs-50-slate
  discrepancy between the registry loader and the raw-file filter before
  quoting anything.
- `run_probe.py` for every long job; `OMP_NUM_THREADS` capped; the sim jobs are
  GPU-serial on one RTX 4070 Laptop, so they do not overlap.
- Commit before running so `provenance.dirty` is false and the sha means
  something.

## 6. Out of scope, on purpose

New scenarios (object count, granularity, goal geometry, goal tolerance). The
handoff proposes a goal-tolerance sweep as the sharpest test of the fine-band
hypothesis, and it is a good experiment — but it varies the *task* while the
weakness identified here is in the *measurement*. Running it before EXP-A/C
would test a new domain with the same too-easy metric. EXP-E answers the
fine-band question on the domain already characterised; the tolerance sweep
becomes the natural generalisation step after EXP-C, with a metric known to
have resolving power.
