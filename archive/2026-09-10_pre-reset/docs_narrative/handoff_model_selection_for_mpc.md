# Handoff — when is a costlier model worth it for MPC?

**Written:** 2026-09-06, at the end of the session that fixed the grid
convention. **Audience:** the next session, planning experiments to decide when
to switch model classes and how to spend MPC compute.

This is a pointer document. Numbers live in their records; nothing is restated
here except the few figures the plan turns on. Start with
[`docs/experiments/STATE_OF_PLAY.md`](experiments/STATE_OF_PLAY.md) (generated —
regenerate with `python scripts/summarise_register.py`) for the full claim and
record inventory.

---

## 1. The one result that should shape the plan

On the constrained domain — 20 piled cubes, fixed-length contact-aware pushes,
off-centre goal — measured on same-state candidate slates
([EXP-0024](experiments/EXP-0024-unet-control-utility.md), incl. its reviewer
amendment):

| model | `accuracy` | `slate4` |
|---|---|---|
| mean-delta (0 params) | 0.343 | 0.796 |
| **linear ridge→I** | 0.533 | **0.950** |
| **UNet** | 0.569 | **0.969** |
| oracle | 1.000 | 1.000 |

**A ridge operator already captures 95% of the oracle's advantage over a random
pick.** The UNet's control edge is real (paired t≈6, 42/49 slates) and takes
about a third of the five points that remain.

**The consequence for your plan is a design constraint, not a result.** On this
domain there is almost no headroom, so *no* model comparison here can produce a
large effect. Before comparing model classes in a new scenario, **measure the
oracle gap in that scenario first** and only run the comparison where the gap is
large. Otherwise you will spend compute discovering that everything ties.

I would make "oracle-gap screening" step one of the pipeline programme.

## 2. The mechanism that predicts *where* a costlier model should win

Three records, consistent:

- The UNet's whole accuracy advantage is **high-frequency**
  ([EXP-0022](experiments/EXP-0022-unet-advantage-is-high-frequency.md)): on a
  common blurred target its 7–11 point lead collapses and the ranking reverses
  in half the cells.
- Control ranking is destroyed by **variance**, not bias
  ([EXP-0008](experiments/EXP-0008-degradation-spectrum-p2-p3.md) + amendment,
  confirmed confound-free in
  [EXP-0025](experiments/EXP-0025-full-spectrum-same-state.md)): amplitude error
  costs ~0.1–0.8% of `slate4`, blur 1.4–2.6%, high-frequency noise 8.8–42.5%.
- So the UNet is better at exactly the band control does not consume, which is
  why its control gain is ~half its accuracy gain.

**The prediction this generates, and the natural spine of your programme:**
costlier models should become preferable precisely where the *task* needs the
fine band — tight tolerances, small targets, near-wall or near-obstacle
manoeuvres, late-stage refinement — and should stay unnecessary where the goal
is coarse. That is directly testable by varying goal tolerance while holding
everything else fixed, and it is the cheapest high-information experiment
available.

## 3. Scenario axes: what is already known

| axis | status | where |
|---|---|---|
| object count 5/10/20/50 | both models improve with n; margin roughly flat; **trend not resolved** (one run per cell) | [EXP-0021](experiments/EXP-0021-unet-vs-linear-granularity-contact.md), C-043 |
| action sampling (blind vs contact-aware) | large effect on how much dynamics the data contains; contact-aware halves zero-contact | EXP-0021, C-005 |
| pile depth (monolayer vs 2-layer heap) | **refuted** as the operative variable | C-015, EXP-0002/0006 |
| contact amount | nonlinearity is one scalar for **mean** targets, **not** for tail/max targets | C-007 **contested**, [EXP-0019](experiments/EXP-0019-contact-linearity-adversarial-check.md) |
| goal geometry | a centred convex target is **degenerate** for a centred pile (`dV≡0`) | C-040, [EXP-0012](experiments/EXP-0012-same-state-slates.md) |
| input representation | depth channels don't help on **piled**; on **scattered** it is pure target-matching | C-017, [EXP-0023](experiments/EXP-0023-channel-target-match-and-full-power-reruns.md) |

C-007's narrowing is the most interesting for you: linearity explains the
*average* response and not the tail. If a task's success depends on outlier
events, that is a regime where a linear model should fail and a costlier one
should pay — and it is untested.

## 4. What you will need that does not exist yet

- **Inference cost per candidate.** Everything measured so far is *fitting*
  cost (ridge: seconds; non-negative: O(D³)/iter, ~59 min/fit at D=4096;
  UNet: ~4.3 s/epoch). A switching pipeline needs *per-candidate inference
  latency and batch scaling*, which nobody has measured. Without it "optimal
  returns from compute" cannot be computed at all. **Measure this first —
  it is an afternoon and it gates the whole programme.**
- **Multi-step rollout behaviour.** Every result in this register is
  **one-step**. MPC spends compute between steps too, and rollout stability was
  flagged long ago for the descriptor operator (10 of 55 eigenvalues outside the
  unit circle — `docs/ideas_log.md`, probe P17) and never checked for the pixel
  operator or the UNet. A model that is fine one-step and diverges at horizon 5
  changes every conclusion here.
- **A cost-aware metric.** `slate4` per unit compute, or utility-at-fixed-budget.
  The current standard pair answers "which is better", not "which is worth it".

## 5. Infrastructure you inherit

- **Method:** the `experiment-log` skill (`.claude/skills/experiment-log/`) —
  tiers, budgets, plan gate, computed grades, invalidation procedure. Read it
  before designing; it encodes ~10 failures from this session.
- **Evidence:** [`REGISTER.md`](experiments/REGISTER.md) (claims, indexed by
  `depends_on`), [`INVARIANTS.md`](experiments/INVARIANTS.md) (what must be true,
  and its test), [`METRICS.md`](experiments/METRICS.md) (**read this before
  choosing a metric**).
- **Tools:** `scripts/run_probe.py` (use it for every long job),
  `scripts/check_register.py` (CI-enforced), `scripts/summarise_register.py`.
- **Data:** `Genesis/data/slates/n20_heap_5mm` — 50 states × 32 actions,
  verified, the **only confound-free ranking testbed**; the granularity series
  `Genesis/data/granularity/{n,c}{5,10,20}` (provenance-stamped);
  trained checkpoints under `runs_granularity/` and `runs_exp0024/`.

## 6. Traps that cost real time here

1. **A wrong noise floor produced a false negative twice**, and both
   corrections flipped a verdict. They were **two different errors** — worth
   separating, because they have different fixes:

   - **A borrowed floor** (C-008, [EXP-0015](experiments/EXP-0015-contact-switching-fixed-grid.md)
     → [EXP-0016](experiments/EXP-0016-contact-switching-loro-floor.md)). The
     effect (+0.0059) was compared against a floor of 0.030 taken from a
     *different design*, because none had been measured for this one. The
     measured floor was sd 0.0027 / sem 0.00096 over 8 LORO folds — the borrowed
     one was **11× too large against the sd, 31× against the sem**. t = 6.2,
     8/8 folds positive. *Fix: never import a floor you did not measure on this
     design.*
   - **An sd used where the sem was needed** (C-045, EXP-0024 + amendment). The
     effect (+0.0185) was compared against the **across-slate sd** (~0.020) —
     the spread of *individual slates* — instead of the **sem of the mean
     difference** (0.0031). That inflated the floor **6.5×**. Note pairing was
     *not* what rescued it: computed unpaired but with a correct sem the t is
     still 4.58. *Fix: compare a mean to a sem, never to an sd, and report
     `mean/sem` plus the win-rate (42/49, 8/8) so the error is visible.*

   Both share one shape: **a wrong yardstick makes a real effect look like
   nothing**, and the resulting "no effect" reads as the cautious call, so
   nobody re-examines it. Pairing is still worth doing whenever units are shared
   (it cost nothing and gained 23% precision in the second case) — it is just
   not the whole fix. `imprecision` is on 16 of 20 records; this is the
   register's dominant weakness.
2. **`accuracy` cannot be compared across preprocessing.** Its denominator is
   the size of the actual change, which shrinks under blur. See METRICS.md.
3. **Bounded metrics saturate**, and a *difference* of two of them trends toward
   zero regardless of skill (cost a whole experiment's conclusion —
   [EXP-0007](experiments/EXP-0007-fss-scale-decomp.md) amendment).
4. **Pooled statistics across heterogeneous conditions** hide type-specific
   effects by construction (EXP-0008 amendment).
5. **Goal choice interacts with pile geometry** — check `dv_true` carries signal
   before interpreting any control number (C-040).
6. **Machine:** cap threads, use `run_probe.py`, never `pkill -f` a pattern that
   matches your own shell.

## 7. Open and unresolved, ranked by what your plan needs

1. Inference cost per candidate (§4) — gates everything.
2. Oracle-gap screening across scenarios (§1) — decides where comparisons are
   even informative.
3. Goal-tolerance sweep (§2) — the sharpest test of the fine-band hypothesis.
4. Multi-step rollout stability (§4).
5. C-043 object-count trend — monotone but unresolved, needs seed repeats.
6. `pixel-index-origin` still **broken** (~1 px); `settled-state` **unchecked**
   for the cube path. Both in INVARIANTS.md.

## 8. Honest health of the evidence

2 records `high`, 4 `moderate`, 11 `low`, 3 `very-low`. `imprecision` dominates
(16/20) — almost everything rests on a single split or a single fit. Nothing
here is strong enough to build a pipeline on without repetition; the claims are
directionally trustworthy and quantitatively soft.

Regenerate the inventory rather than trusting this section:
`python scripts/summarise_register.py`.
