# Retrieval-based transition model: design and plan (EXP-0059)

Owner of the record: `experiments/EXP-0059-retrieval-transition-model/`. Code: `model/retrieval/`.
The plan (sections 0-12) is below the results summary. Full numbers live in the experiment
records, not here.

## Results & conclusions (2026-09-28, clean data)

**Sources.** `experiments/EXP-0059-retrieval-transition-model/EXPERIMENT.md`: "Clean-data v2
rung", "Multi-step rung" and "3-seed NFD comparison" sections.

**Test sets.**
- 1-step: DS-0016, 32 same-state pools × 64 pushes.
- 3-push: DS-0018, 32 pools × ~64 three-push sequences (2,042 total).
- Metric: `slateN_tough` (8-goal lyapunov).
- 95% CIs are paired pool-bootstrap intervals.

**Training data.** Every model uses the same DS-0015 rows (11,659 clean). Retrieval k was chosen
on DS-0017 validation pools.

### Two bugs changed the verdict (both found and fixed on 2026-09-28)

1. **Transfer bug.** Retrieval Hungarian-matched *all 20* cubes with no distance limit, so
   far-away cubes inherited the pushed cubes' displacements.
   - Fix: retrieval now uses only a geometric **interaction set**, computed without truth: the
     swept corridor plus a chain closure (τ = 12 mm, 60 deg). It recovers 96% of truly moved
     cubes at 90% precision.
   - The set is computed once and stored in the bank (DS-0014, `build_bank_v2.py`). Cube pairing
     is gated at 6 mm.
2. **Data bug (ISS-010).** In 44-56% of the old narrow transitions (DS-0008/9/11/12/13) the tool
   touched down *on* a cube, a side effect of the pile-aware stop clamp; a further 8-12% moved
   nothing.
   - Fix: the sampler checks legality and redraws, and samples the start gap from 5 mm to L − 5 mm
     along the push axis (`start_gap_range`).
   - The old sets are flagged and archived. Clean replacements: DS-0015 (train), DS-0016 (test),
     DS-0017 (validation), DS-0018 (3-push pools).

All earlier retrieval numbers, including the previous "no" verdict, are superseded. So are the
EXP-0053 baselines, which were trained and tested on illegal data.

### The main question: is retrieval better at evaluating action sets, especially multi-step?

**Yes against any single trained model, and more clearly multi-step. Not clearly against an
NFD ensemble.**

| model (trained on DS-0015) | 1-step `slateN_tough` (DS-0016) | 3-push terminal (DS-0018) |
|---|---|---|
| **retrieval, k5 per-cube median** | **0.792 [0.755, 0.829]** | **0.821 [0.779, 0.858]** |
| retrieval, k1 | 0.757 [0.706, 0.801] | 0.787 [0.741, 0.830] |
| NFD, seeds 0 / 1 / 2 | 0.731 / 0.708 / 0.700 (sd 0.016) | 0.693 / 0.695 / 0.633 (sd 0.035) |
| NFD 3-seed ensemble (average of predicted dv) | 0.818 [0.768, 0.865] | 0.785 [0.726, 0.837] |
| linear32 | 0.666 [0.585, 0.746] | 0.729 [0.653, 0.799] |
| linear64 | 0.626 [0.535, 0.718] | 0.678 [0.593, 0.753] |
| persistence | 0.005 | 0.044 |

**Paired differences, retrieval k5 minus each model:**

| compared with | 1-step | 3-push |
|---|---|---|
| NFD seed 0 | +0.062 [−0.010, +0.137] | +0.128 [+0.050, +0.206] |
| NFD seed 1 | +0.085 [+0.015, +0.160] | +0.126 [+0.050, +0.207] |
| NFD seed 2 | +0.092 [+0.019, +0.172] | +0.188 [+0.106, +0.272] |
| linear64 | +0.167 [+0.090, +0.244] | +0.143 [+0.073, +0.221] |
| linear32 | — | +0.092 [+0.031, +0.157] |
| NFD 3-seed ensemble | −0.026 [−0.074, +0.025] | +0.036 [−0.017, +0.092] |

- At 3 pushes, retrieval beats every NFD seed and both linear models, each CI excluding 0. At
  1 step it beats 2 of 3 NFD seeds.
- Against the 3-seed NFD ensemble the difference is not resolved in either direction.

**Horizon behaviour** (step 1 / 2 / 3):

| model | step 1 | step 2 | step 3 |
|---|---|---|---|
| retrieval k5 | 0.835 | 0.840 | 0.821 |
| NFD seed 0 | 0.761 | 0.676 | 0.693 |
| linear32 | 0.628 | 0.703 | 0.729 |

Retrieval's ranking holds across the horizon and the NFD's dips. Linear32 rises again, as it did
before the cleanup.

**Consistency.** Retrieval's CIs are about half as wide as the NFD's. It ranks more consistently
from one state to the next.

**Information or noise?** The evidence points to information:
- `accuracy` still favours the NFD (accuracy_1 0.557 vs 0.485), so retrieval's lead is not a
  "prettier image" artefact.
- Blurring predictions did not change `slateN` for the NFD or the simulators, measured before
  the cleanup (DS-0009).
- Before the cleanup, the neighbour-rank curve fell monotonically to a random donor. It has not
  been re-run on clean data.
- **Not yet tested:** whether the lead comes from cube-level feasibility (discrete,
  cube-conserving outputs) rather than better transition information. The planned control is
  the sharpened NFD, whose predicted cloud is turned into 20 cubes (section 3).

**Caveats.**
- **Unequal inputs.** Retrieval reads true cube poses; the NFD and linear models read occupancy.
  Cube centres can be recovered from occupancy, yaw cannot (yaw was irrelevant in R1).
- **One retrieval configuration.** The 6 mm gate was held fixed, not swept.
- **Small test sets.** Only 32 pools per test set.
- **One domain.** Narrow only: n20, single layer, exact 20 mm pushes.
- **No closed loop yet.**

### Secondary results

**Still valid (measure the metric or planning, not the models):**
- Chaos floor (DS-0009): exact re-simulation scores accuracy_1 0.70, falling to 0.43 at 0.5 mm of
  state error.
- Perturbed simulators score `slateN_tough` 0.92-0.93 at 0-0.5 mm, with the noise shared by the
  pool.

**Pre-fix, needs a re-check on clean data:**
- kNN disagreement predicted error; top-1 distance did not.
- Retrieval ranking plateaued with more data beyond 25k (bank data affected by both bugs).
- The NFD-with-reference model was a wash, but it was trained on pre-cleanup data.

**Also found:** `train_nfd.py` let `--override output.log_dir` apply after the resume check
(ISS-011), so a "new seed" could silently resume an old one. Use `--no-resume`.

### Ranked next steps

1. **Closed-loop MPC with retrieval.** Offline `slateN` has failed to predict control before
   (EXP-0054).
   - Planner: a sampling planner (CEM or pure sampling; retrieval is not differentiable) over
     20 mm perpendicular pushes.
   - Setup: 8-goal set; compare with the NFD (single seed and ensemble) under the same planner.
   - Metrics: in-goal mass and completion pushes.
   - Needed: a batched retrieval call fast enough for hundreds of candidates per decision.
2. **Controls for information vs representation** on clean data:
   - sharpened NFD (cube-feasible NFD outputs);
   - random-donor and neighbour-rank curve;
   - kinematic sweep;
   - retrieval with k-NN occupancy averaging.

   If a sharpened NFD closes the gap, the lesson is "predict feasible cubes", not "retrieve".
3. **Re-run on clean data:**
   - the confidence readout (kNN disagreement vs top-1 distance, at row and within-pool level);
   - data scaling / "which 50K" subsets (6k-50k, including branching vs chains). The collector
     now makes clean data cheaply.
4. **Retrieval vs the NFD ensemble.** Try a retrieval ensemble (several k / gate settings, or
   retrieval+NFD averaged dv), and more test pools (≥ 64) to resolve differences of about ±0.03.
5. **Occupancy-only retrieval** for real-robot use. Recover cube centres from occupancy peaks
   (or perception), drop yaw, and measure the cost against the pose-based key.
6. **Beyond the narrow domain, with a bank of at most 50K.**
   - Variable push length: put length in the key and scale displacements.
   - n50 / multi-layer piles: needs z in the key and a revisited interaction set.
   - The question is which 50K transitions cover the broad domain. Compositional per-group
     retrieval (section 6) is the natural next step for coverage.
7. **Metric study (EXP-0060) re-run on clean models** with the perturbed-simulator zoo scored on
   every candidate metric.

---

## 0. The main question

> **Does a retrieval model evaluate action sets for a task better than the existing models do,
> especially over several pushes? If it does, is that because it carries the information in
> transitions better, rather than because it adds the right kind of noise or sharpness?**

What answers it:
- **Headline.** 1-step `slateN` and multi-step terminal `slateN` on the 8-goal "tough" set.
- **Supporting.** Rollout prediction quality over k pushes.
- **Mechanism.** Information-vs-noise controls (section 3). A win without those controls doesn't
  answer the question.
- **Not the headline.** `accuracy` is reported, but it is not the headline. The R0 chaos floor
  shows it penalises sharp predictions (section 1), and EXP-0054 already found that accuracy
  fails to predict control.

Side deliverable, first-class: a **coverage/confidence model**. Given a new state, it should say
which actions the bank predicts worst, and it comes with a test that this signal predicts
error (section 5).

## 1. Setting and facts (measured; not assumptions)

### Domain

"Narrow" = n20 single layer, 5 mm cubes, exact 20 mm pushes with the blade perpendicular to
the push direction. The push direction is continuous. TRAINING_PHYSICS.

### State

20 cube poses `[x, y, z, qw, qx, qy, qz]`. The occupancy the baselines see is a 64 px image of
the cube centres over the 128 mm tray (2 mm/px, 5 mm footprint); it carries no yaw.

### Data (current: clean by construction, ISS-010 sampler fix)

| role | data |
|---|---|
| train / bank | DS-0015: 11,659 clean rows. The curated bank with stored interaction sets is built by `code/build_bank_v2.py` |
| validation | DS-0017: 32 pools × 64 pushes |
| 1-step test | DS-0016: 32 pools × 64 pushes, plus 112 whole clean chains for rollouts |
| 3-push test | DS-0018: 32 pools, 2,042 three-push sequences |

The old sets (DS-0008/9/11/12/13) had 44-56% illegal touchdowns. They are archived and flagged;
do not use them for new claims.

### Motion statistics per push

Measured on DS-0008 (legacy data). It should be re-measured on DS-0015.
- Cubes moved by more than 1 mm: median 3, mean 6. Clump pushes move many.
- Every mover starts within along [-2, 41] mm and |lat| ≤ 48 mm of the push start, in the push
  frame.
- Moved cubes travel a median of 11.6 mm.

### Current retrieval model (`model/retrieval/`, EXP-0059 primary config)

1. **Push frame.** Origin at the push start, x along the push. Each transition stores cube poses
   in this frame.
2. **Interaction set.** The query key is not all cubes in a window: it is the cubes in the swept
   corridor plus a chain closure (τ = 12 mm, 60 deg), computed without truth. This recovers 96%
   of truly moved cubes at 90% precision. Bank rows store theirs precomputed.
3. **Distance.** Capped Chamfer between the query's and the donors' interaction sets.
4. **Transfer.** Hungarian pairing gated at 6 mm. Each paired query cube takes its donor cube's
   push-frame displacement. Everything else stays put.
5. **Output.** k = 5 neighbours, per-cube median displacement. The output is clean cube poses,
   rendered to occupancy; rollouts carry poses forward.

### Measurement facts (still valid)

**Chaos floor.** Measured on DS-0009 rows, so on legacy data, but it measures the metric rather
than the models.

| state perturbation | accuracy_1 | blur2 |
|---|---|---|
| exact re-simulation | 0.70 | 0.90 |
| 0.5 mm | 0.43 | 0.78 |
| 1 mm | 0.30 | 0.74 |
| 2 mm | 0.11 | 0.60 |

- `accuracy` penalises sharp predictions. Lead with `slateN`.
- Perturbed simulators with the noise shared across the pool reach `slateN_tough` 0.92-0.93 at
  0-0.5 mm.

## 2. The overnight plan (executed 2026-09-28; kept for its rules and specs)

Dataset names in this plan map to registered IDs as follows:
- DS-A → DS-0017;
- DS-B → DS-0018;
- the test set → DS-0016;
- the train bank → DS-0015;
- DS-C (the reservoir, DS-0012) is legacy data with illegal touchdowns and needs re-collecting.

Where this plan says DS-0009, read DS-0016. The v0 below is superseded by the "Current
retrieval model" in section 1.

### 2.0 Rules that hold throughout

- **Tune on validation only.** Use validation chains, plus the validation pools (DS-A below)
  for `slateN`.
- **Evaluate DS-0009 on frozen variants only.** Allow at most ~5 variants per rung, and report
  all of them, including the losers.
- **Ceiling runs** that touch DS-0009 states are labelled `ceiling` and never feed a headline
  number.
- **Report on every row:** `slateN` (8-goal headline; 13-goal secondary) with wins/losses/ties
  and the paired sem, then `accuracy_1`, blur1/blur2 accuracy, moved-cube mm error, and
  `rollout_accuracy_k` for k = 1..8 on the DS-0009 chains.
- **Two outputs per retrieval variant:**
  - (i) a **clean** prediction: 20 feasible cube poses, rendered with `occ_from_particles`.
    This is what rollouts and `slateN` use by default.
  - (ii) an optional **hedged** occupancy: the mean of the k neighbours' clean renders. It is
    used only as an `accuracy` scoring view, and always shown next to (i).
- **Record-keeping.** Each rung is a RUN under EXP-0059, or its own EXP where it tests a
  separate claim, per the experiment-log skill. Every long script checkpoints atomically per
  unit.

### 2.1 Datasets to collect (Genesis; start early, run in the background)

All of these use the DS-0008 recipe: `Genesis/chain_collection.py`, pile-aware sampler,
exact-length and perpendicular validity checks, TRAINING_PHYSICS. Assign IDs in order when the
datasets are registered.

| name | shape | purpose | est. cost |
|---|---|---|---|
| **DS-A val pools** | 32 new start states (16 scatter / 16 clump) × 64 pushes, new seed | `slateN` for tuning, without touching DS-0009 | ~10 min |
| **DS-B multi-step test pools** | 32 start states × 64 **3-push sequences**, each run without reset (row = sequence id, same order in every step file, like `slates_multistep`) | headline multi-step ranking: terminal `slateN` | ~30 min |
| **DS-B' val twin of DS-B** | 16 start states × 64 sequences, different seed | multi-step tuning | ~15 min |
| **DS-C candidate pool** | as many narrow transitions as fit (target 150-250k; use 128 envs if stable). Start mix: scatter, clumps, near-wall states, and chains of 8 so later-step states occur. **Also a pool-shaped share**: same-state branches of 16-64 pushes | the reservoir from which "which 50K" subsets are drawn (R4) | ~4-5 h |
| **DS-D targeted acquisition** | built from validation failures (section 7) | R5 | ~1 h |

DS-B needs a small extension to the collector: a `--mode seqpools` that broadcasts one settled
state to all envs and then executes 3 pushes per env. Each push is drawn from that env's current
state by the pile-aware sampler.

`slates_multistep` is NOT usable here: it uses heap spawns, friction 0.3, multi-layer piles and
non-narrow physics.

### 2.2 v0: the simplest version (already mostly built in `model/retrieval/`)

1. **Push frame.** Origin at the push start, x along the push. Each transition stores cube
   (x, y, yaw) in this frame plus wall distances.
2. **Query key.** The cubes inside a local window (default: along [-15, 55] mm, |lat| < 40 mm).
3. **Distance.** Capped symmetric Chamfer between window cube sets. Exact brute force on the
   GPU (the bank is ≤ 50k × 20).
4. **Transfer.** Match query window cubes to donor window cubes (Hungarian). Each matched cube
   takes the donor's push-frame displacement and yaw change. Unmatched cubes stay put.
5. **Output.** Clean 20-cube poses. Rollouts feed the poses forward, not a re-rasterised image.

### 2.3 The ladder

Each rung below lists what varies, the metric, and the decision rule.

**R0 — harness, baselines, ceilings.** Done or in progress. Adds the 8-goal `slateN` and the
chaos floor.
Still to add:
- **Perturbed-simulator zoo members** for section 4: resimulate the DS-0009 pools at state
  noise of 0.5, 1 and 2 mm.
- **Same-state leave-one-out retrieval** inside the DS-0009 pools, labelled `ceiling`: the
  nearest *action* from the identical state.

**R1 — make v0 sound (validation only).**
- Varied:
  - window size and shape (fixed box vs a window grown from the donor's moved-cube set);
  - Chamfer cap and corridor weight;
  - yaw in or out of the key;
  - wall features;
  - Hungarian vs greedy matching;
  - k ∈ {1, 3, 8, 16};
  - aggregation (1-NN; per-cube median displacement; hedged mean for the `accuracy` view);
  - **mirror augmentation**: reflect every bank row across the push axis (v → −v, flip the
    displacement's v and the yaw sign). This doubles the bank for free. It assumes a symmetric
    blade and cubes, so check it first on the bank: a mirrored row should be retrieved as its
    original's twin.
- Also: **transfer rules** (G2). Additive occupancy delta; per-cube displacement; "paste",
  where the donor's post-push window replaces the query's window; and per-cube displacement
  followed by a 5 mm non-overlap projection, so the output stays feasible.
- Metric: validation `rollout_accuracy_4`, blurred accuracy, mm error, DS-A `slateN`.
- Decision: freeze the best variant on DS-A `slateN`, using rollout_4 to break ties. Go to R2.
  If no variant reaches the narrow NFD's DS-A `slateN` minus 0.05, first check that
  retrieval quality actually drives ranking (the R2 controls) before spending on data.

**R1.5 — diagnostics that pick the next move** (coder, running now). Report each one on
validation *and* DS-0009 as mm error on moved cubes, accuracy_1 (clean + blur2) and slateN:
- **Oracle-donor ceiling.** Choose the donor by *truth*: the one whose transferred prediction has
  the lowest mm error, among the top-50, the top-500, or the whole bank by key distance.
- **Same-state LOO ceiling.** For each DS-0009 pool row, the donor set is that pool's 63 sibling
  actions: same state, different action. Retrieve by action distance (lateral offset + angle
  in the push frame) and transfer.
- **Transfer rules under the oracle donor.** Displacement, paste, and displacement + non-overlap
  projection.
- **v0 reconciliation.** Why v0 scored 0.172 against the probe's 0.295.

"High" means moved-cube error ≤ 2.5 mm AND slateN ≥ 0.70. "Low" means error ≥ 4 mm or
slateN < 0.55. Cases in between go to the closer row, and the reasoning gets logged. The table
below is decided on the **top-500 oracle donor** (top-50 is noted where it matters) and the
**same-state LOO** result.

| oracle donor (top-500) | same-state LOO | diagnosis | do first | then |
|---|---|---|---|---|
| high, already at top-50 | any | Good donors are near in key space but ranked wrong: a **metric** problem | Metric fix: learn per-feature weights / corridor profile by ranking donors against oracle outcome distance on the train bank (leave-chain-out), and add mirror augmentation | (c) with top-3 donors as input: the network picks among good donors |
| high only at top-500 / whole bank | high | Good donors exist but are rare and far in key space; the whole-window key is too specific | **(a) compositional** + mirror augmentation (no Genesis cost) | (b) targeted acquisition for groups that still lack a close donor |
| low | high | The bank lacks local state matches; with the right state, action transfer works | **(a) compositional** (combinatorial coverage), with **(b)** started in parallel on the GPU slot (Genesis) | R4 bank selection with group-coverage stratification |
| low | low | Outcomes are too sensitive to state and action detail for delta transfer at any feasible density. **Pure retrieval's ceiling ≈ the LOO value** | **(c) NFD with reference** (learned correction on top of retrieved context) | Record the negative result for pure retrieval; keep the confidence model (section 5) |
| (any) | (any), and oracle is good only under paste or projection | The transfer rule is the bottleneck | Switch the transfer rule, re-run R1's best key | then re-enter the table |

Whatever the row, the section 3 controls run on whichever model wins.

**R2 — information vs noise (the main question; section 3).** Run the controls on the frozen
variant and on the baselines, on DS-0009 and DS-B.
- Decision:
  - if retrieval's `slateN` advantage survives the matched controls, the answer to the main
    question is "information";
  - if a sharpened NFD or random-donor retrieval matches it, the answer is "representation" or
    "noise". Record that, then decide whether retrieval is still worth pursuing for its
    confidence signal alone.

**R3 — multi-step.**
- Metrics:
  - DS-B terminal `slateN`: rank the 64 sequences by predicted Δlyapunov after 3 pushes;
  - `rollout_accuracy_1..8` and mm drift on the DS-0009 chains;
  - how the confidence signal evolves along rollouts.
- Compared: retrieval (clean) vs narrow NFD vs wide NFD vs sharpened NFD.
- Decision: this is the headline comparison. If retrieval loses here but wins on 1-step,
  diagnose where in the rollout it breaks (out-of-bank states after step 1, since the bank's
  post-push states are only as diverse as its chains).

**R4 — "which 50K" (data structure at a fixed budget; G3).** Draw subsets of DS-C (+ bank)
at budgets 6k / 12k / 25k / 50k.
- Strategies:
  - random;
  - stratified by number of cubes in the corridor × start kind;
  - coverage-greedy (k-center in the retrieval key space);
  - pool-shaped (branching) vs chain-shaped at equal budget;
  - confidence-driven (the rows the current bank predicts worst).
- Also: random subsets above 50k (100k, full pool) to measure the headroom that the budget
  gives up.
- Metric: DS-A `slateN`, validation rollout_4, mm error.
- Decision: pick the ≤ 50k bank for the final frozen DS-0009 and DS-B evaluation.

**R5 — targeted acquisition from failures (section 7).**
- Compared: +N targeted transitions vs +N random DS-C transitions, same N, on validation.
- Decision: keep targeted acquisition in the recipe only if it beats random by more than the
  seed-to-seed spread of random subsets. Run 3 random draws to measure that spread.

**R4b — compositional retrieval (section 6)** when the R1.5 table points to it. It is
validated like R1 (validation chains + DS-A) and frozen before DS-0009 / DS-B.

**R6 — confidence/coverage model (section 5).** Runs alongside R1-R5 on the same outputs.

**R7 — NFD with a retrieved reference (section 8).** Started when the R1.5 table says so, or
when retrieval has plateaued (two consecutive changes, each worth < 0.02 DS-A `slateN`)
while a ceiling (oracle donor, same-state LOO, perturbed simulator) beats it by ≥ 0.1.

**Optional R8 — closed loop.** Run this only if R3 is a clear win. Plan with CEM or pure
sampling, since the retrieval model is not differentiable. Candidates are 20 mm perpendicular
pushes. Use the EXP-0057 sampling planner structure with the model replacing the simulator:
8-goal set, DS-0006 starts, in-goal mass and completion pushes against the narrow NFD run under
the same planner.

### 2.4 Rough schedule (~8 h)

- **h0-1.** R0 finishes. Launch DS-A, then DS-B / DS-B', then start the DS-C background
  collection (checkpointed per chunk). The coder finishes v0 variants.
- **h1-3.** R1 on validation / DS-A. Build the R2 control models (sharpening operator, kinematic
  sweep, random donor).
- **h3-5.** R2 and R3 on DS-0009 / DS-B with the frozen variant. R6 analysis.
- **As soon as R1.5 lands:** pick (a) / (b) / (c) from the R1.5 table. (a) and (c) are GPU-light
  code work and can run next to the Genesis reservoir collection. (b) needs its own Genesis slot:
  pause the reservoir chunks while it runs.
- **h5-7.** R4 subsets as DS-C grows. R4b / R5 / R7 as the table dictates.
- **h7-8.** Final frozen evaluation (≤ 50k bank) on DS-0009 and DS-B, then records.

GPU contention: collection and evaluation share the GPU. Batch evaluations between collection
chunks, or run collection at 64 envs, if throughput drops.

## 3. Information vs noise: the controls

The worry: a retrieval prediction differs from a hedged NFD output in two confounded ways.
(a) It may carry *better information* about what moves where. (b) It is *sharp and discrete*.
Sharpness alone could change `slateN`, in either direction, through lyapunov's response to
where mass lands. The controls separate the two.

| control | what it isolates | expected if "information" |
|---|---|---|
| **Sharpened NFD**: pull 20 feasible cube positions out of the NFD's predicted occupancy (greedy peak picking with 5 mm exclusion, or mass-constrained assignment of the 20 query cubes to the predicted mass by OT), then re-render | discreteness/feasibility with the NFD's information | retrieval > sharpened NFD |
| **Hedged retrieval**: the k-NN mean occupancy | sharpness removed from retrieval | retrieval's ranking holds (lyapunov is linear, so the hedge should rank about the same as its members) |
| **Random donor**: same transfer rule, donor drawn at random from bank rows with the same number of corridor cubes and the same start kind | "right kind of noise": realistic displacements with no state match | retrieval ≫ random donor |
| **Neighbour-rank curve**: use the r-th neighbour, r ∈ {1, 2, 5, 20, 100, 1000} | whether matching quality carries the information | `slateN` and mm error decline monotonically in r |
| **Kinematic sweep** (no learning): cubes in the blade's swept rectangle are carried to the blade front, with a 5 mm overlap-resolution pass along the push | how much is trivial geometry | retrieval > sweep, especially on clumps |
| **Blur-matched**: score retrieval blurred at σ to match the NFD's sharpness (fit σ so the mean image entropy matches), and the NFD sharpened as above | the same comparison at equal sharpness, both directions | retrieval stays ahead at equal sharpness |
| **Perturbed simulator** (0.5 / 1 / 2 mm) | a sharp model with near-perfect information: calibrates what "information" buys at a given sharpness | sets the scale for the others |

Report all controls in the same table as the models, with paired per-pool wins/losses/ties
against the narrow NFD.

## 4. Metric study: an image metric that tracks `slateN` and does not punish sharpness

**Goal.** Find a one-step (and rollout) image metric under which a *feasible sample from the
predictive cloud* is scored at least as well as *the cloud itself*, whenever `slateN` agrees,
and which ranks models the way `slateN` does.

**Candidates.** Each is computed on the DS-0009 pool rows (the same rows `slateN` uses) and on
the chains:
1. `accuracy` (reference) and blurred accuracy at σ = 1, 2, 4 px;
2. multi-scale rms: average-pool the images by 2, 4 and 8, then average the per-scale accuracies;
3. optimal transport on soft-truth mass: sliced W1 and entropic W2 (`occ_for_scoring`
   densities), normalised by persistence's value;
4. cube-set Chamfer / Hungarian mm error. Models without cube outputs use peak extraction; the
   hedged cloud uses its own extraction;
5. goal-aware errors: |Δ in-goal mass (pred − true)| and |Δlyapunov (pred − true)| averaged over
   the 8 goals;
6. within-pool rank agreement: Spearman between predicted and true Δlyapunov across a pool's
   64 actions, averaged over goals. This is nearly `slateN`'s own ingredient. It is included to
   see how much of `slateN` a smoother statistic captures.

**Model zoo for validation** (≥ 15 members, spanning information and sharpness independently):
- the 7 EXP-0053 models and persistence;
- the retrieval variants (1-NN, k-NN median, hedged mean);
- the section 3 controls (random donor, rank-20 and rank-100 donors, kinematic sweep, sharpened
  NFD, blurred retrieval, blurred NFD);
- the perturbed simulators (0.5 / 1 / 2 mm).

**Validation procedure.**
1. **Across models.** Kendall τ and Spearman ρ between each metric and 8-goal `slateN`. Also
   against DS-B terminal `slateN` for the models that roll out. 95% CIs by bootstrapping pools,
   recomputing both metric and `slateN` per resample.
2. **Sharpness invariance.** For each model with a cloud and a sample form (NFD vs sharpened
   NFD; retrieval hedged vs 1-NN; simulator blurred vs raw), check that the sign of
   (metric_sample − metric_cloud) matches the sign of (slateN_sample − slateN_cloud).
   Count agreements over the pairs.
3. **Within model, across pools.** Does the per-pool metric predict per-pool capture?
   Spearman over 32 pools. This is secondary: `slateN` per pool is very noisy.

**Decision.** Recommend the metric with the highest across-model τ whose CI excludes the τ of
`accuracy`, and which passes sharpness invariance on ≥ 80% of pairs. If none separates from
`accuracy`, say so. The zoo is small, so this is expected to be underpowered. Report the τ with
its CI rather than a winner.

**Record.** This study is its own claim. Give it its own EXP number, and add the chosen metric
to `experiments/METRICS.md` only after it passes.

## 5. Confidence/coverage model (first-class deliverable)

**Question.** Given a state (sim or real) and a candidate action set, can the bank tell which
actions it will predict worst? And does collecting those actions fill the gaps faster than
random collection?

**Signals.**
- d1, the nearest-neighbour distance, and the mean distance over k;
- neighbour disagreement: the mean pairwise per-cube mm distance between the k neighbours'
  clean predictions, or the variance of their Δlyapunov;
- number of cubes in the corridor;
- wall proximity.

**Tests.**
1. **Row level** (validation chains, then DS-0009 chains). Spearman(signal, per-row mm error),
   and Spearman(signal, per-row swept rms). AUROC for flagging the worst 20% of rows.
   Risk-coverage curve: error of the most-confident x% of rows.
2. **Action level within a state** (DS-A pools, then DS-0009 pools). For each pool, per-pool
   Spearman between the confidence rank of the 64 actions and their true error rank, averaged
   over pools with a CI. This is the "which actions does the dataset predict least well" test.
3. **Along rollouts** (DS-B, DS-0009 chains). Does confidence fall as rollout error grows?
4. **Utility** (R4/R5). Adding the least-confident transitions improves validation error more
   per transition than adding random ones.

**Pass.** Spearman ≥ 0.3 at the row level and ≥ 0.2 at the within-pool action level, with CIs
excluding 0, and test 4 positive. Report the best single signal and a simple combination
(rank-average).

## 6. (a) Compositional retrieval: retrieve per interaction group

**Idea.** A 20 mm push usually touches 1-5 cubes (median 3 movers). Cubes that don't touch each
other during the push move roughly independently under a flat blade. So decompose the query into
**interaction groups**, retrieve a donor *per group*, and assemble. A bank of N transitions
holds ~2-4 N groups, each small, so the key is low-dimensional and effective coverage grows
combinatorially. This is the "same number of objects in the right small area" idea, applied at
query time instead of by collection.

Independence is a hypothesis. It is tested directly (step 6).

Code goes in `model/retrieval/groups.py`, with a `CompositionalPredictor` reusing
`TransitionBank`, `frame.py` and the Hungarian code.

1. **Affected set.** All geometry is in the push frame: u along the push from the push start,
   v lateral. The blade half-width is 20 mm, the cube half-size h = 2.5 mm, and the push length
   L = 20 mm.
   - **Direct:** cubes with |v| < 20 + h + m_v and u ∈ [−h, L + h], where m_v = 1 mm
     (swept).
   - **Chain closure:** repeatedly add any cube c whose centre is within d_c = 5 + τ mm of a
     cube a already in the set, and ahead of it: (c − a)·û > 0 and its angle to û is < 60 deg.
     Stop adding along a chain once the cumulative forward gap passes L. τ ∈ {1, 2, 3} mm;
     default 2.
   - Cubes outside the affected set are predicted to stay put.
   - Validate this rule on the bank: report the recall and precision of "affected" against
     "moved > 1 mm".
2. **Groups.** Connected components of the affected set, with edges between cubes closer than
   5 + τ mm. Two cubes pushed side by side by the blade with a gap > τ are separate groups.
3. **Group key.**
   - Cube coordinates (u, v, yaw mod 90 deg) *relative to the group's frame*: u absolute
     (distance to the blade matters), v absolute (the blade edge matters near |v| ≈ 20 mm).
   - A halo: non-group cubes within 8 mm of the group, down-weighted by 0.3. This is the
     "blocker" context.
   - A lateral-shift allowance: if every group cube has |v| < 20 − h − 3 mm, the group is
     "interior". Interior keys use v relative to the group centroid plus the centroid's
     clipped distance to the nearest blade edge (clip at 8 mm), which makes interior groups
     translation-invariant along the blade.
   - Mirror augmentation (v → −v) on by default.
4. **Group bank.** For every bank transition, run steps 1-3 using its *start* state and action.
   Store (transition id, cube ids, key, per-cube push-frame displacement and yaw change,
   interior flag, group size m). Index separately by m.
5. **Retrieval and transfer.**
   - **Hard gate:** the donor group has the same size m and the same interior flag.
   - **Distance:** Hungarian-matched cube cost (capped at 6 mm per cube) + 0.3 × the halo
     Chamfer.
   - **k and aggregation:** k = 5, per-cube median displacement (R1's best), plus a 1-NN
     variant.
   - **Fallbacks:**
     - groups with m > 6 (clump pushes) fall back to the whole-window retrieval (v0 best);
     - groups whose best distance exceeds d_max (95th percentile of validation distances) fall
       back to the kinematic sweep (section 3);
     - log the fraction that falls back.
   - **Assembly:** apply every group's displacements, then a non-overlap projection: 10
     iterations pushing overlapping cube pairs (centre distance < 5 mm) apart, mostly along û.
     Report how much this moves cubes.
6. **Independence check.**
   - (i) On the bank, the fraction of transitions where cubes from different groups end up in
     contact (post-push centre distance < 5.5 mm) and at least one of them moved > 1 mm. These
     are violations.
   - (ii) On validation, the error of composite predictions split into "no violation expected"
     vs "violation" rows.
   - If violations exceed 25% of multi-group rows, increase τ; that merges groups.
7. **Coverage report** (the point of the method). Nearest-donor distance per group, compared
   with the whole-window nearest distance, split by m. Also the mm error vs group size.
8. **Compare** against v0 best on validation chains, DS-A slateN and rollout_4, then freeze and
   run on DS-0009 / DS-B.
   - **Success:** mm error on moved cubes drops by ≥ 25% vs v0 best, AND DS-A slateN rises
     ≥ 0.1.
   - **Failure despite good coverage:** independence or transfer is broken. The step-6 check
     tells which.

## 7. (b) Targeted acquisition generator (layout injection)

The query source is **validation only**. DS-0009 is used only in a labelled `ceiling` run.

Split the validation failures into **val-A** (source) and **val-B** (judge) by chain, so gains
are measured on queries that were not themselves copied.

**Generator.** `model/retrieval/acquire.py` builds a layouts file and an actions file. A new
`Genesis/chain_collection.py --mode targeted` executes them. That mode reuses `set_states` (set
+ settle, as `make_clump_states` does) and `push`, with explicit per-env actions from a file
instead of `draw_valid`.

1. **Failure units.**
   - With compositional retrieval (section 6), the unit is a **group** whose best donor distance
     exceeds the 80th percentile, or whose validation mm error exceeds 5 mm.
   - Without it, the unit is a **query window**.
   - Take the worst ~128 units from val-A per round, de-duplicated so that no two come from
     adjacent steps of one chain.
2. **Local configuration.** The unit's cubes, plus a halo of 8 mm (group mode) or the whole
   window (window mode), as push-frame (u, v, yaw), plus the wall distances if a wall is within
   15 mm of the window.
3. **Variants per unit: 8**, one env each.
   - **2 × same place.** The query's world pose and action. Far cubes: in one variant, the
     query's own far cubes jittered by 2 mm; in the other, resampled at random.
   - **1 × mirror.** Reflected across the push axis, if the reflected window fits in the tray.
   - **5 × moved.** A random rigid transform of (local configuration + action) whose window
     stays ≥ 6 mm inside the tray. If the query had a wall in its window, keep the wall
     relation: only transforms that map that wall onto a tray wall at the same distance, i.e.
     rotations by k × 90 deg plus a slide along the wall.
   - **Local jitter:** position σ drawn from {0.5, 1.0, 1.5} mm (stratified across the variants,
     so error-vs-jitter can be read off), yaw σ = 5 deg.
   - **Action jitter** on 3 of the 8 variants: lateral ±1.5 mm, direction ±3 deg (the blade stays
     perpendicular, yaw = direction + π/2), length exactly 20 mm.
   - **Far-cube fill:** the other 20 − m cubes go uniformly in the tray, non-overlapping
     (centre distance ≥ 6 mm), outside the window dilated by 6 mm, and never inside the blade's
     start footprint or swept rectangle. Rejection-sample; keep n = 20.
4. **Batching.** A chunk is 32 envs = 4 units × 8 variants (64 envs if memory allows). Settle
   budget as DS-0008 (3000). Push once. Atomic per-chunk files; resumable; manifest records
   unit ids.
   - Measured throughput for comparison: DS-0008 did 8 pushes per chunk of 32 envs in ~55 s.
     One settle + one push should be ≈ 15-25 s per chunk.
   - 128 units × 8 = 1,024 rows ≈ 10-15 min per round.
5. **Acceptance.**
   - After settling, recompute the local configuration in the variant's push frame and
     Hungarian-match it to the target.
   - `accepted = (max per-cube error ≤ 1.5 mm) & single_layer (z < FLOOR_Z_MAX) & in-tray &
     valid action (check_actions)`.
   - Store the **settled** state, never the target. Rejected rows are still valid physics: keep
     them, flagged, but don't count them as targeted.
   - Log the acceptance rate. If it falls below 50%, reduce the jitter and add 1-2 mm of spacing
     slack in the layout.
6. **Output fields:**
   - the chains-mode keys (`states`, `states_`, `p_starts`, `p_stops`, `angles`, `valid`,
     `single_layer`);
   - plus `unit_id`, `source_row`, `variant_kind`, `jitter_mm`, `local_err_mm`, `accepted`.
7. **Evaluation (R5).** Add the accepted targeted rows (N) to the bank, and compare with:
   - +N random reservoir rows (3 draws);
   - +N **count-matched untargeted** rows: reservoir rows with the same m in the corridor, which
     tests whether the configuration matters or just more of that count.

   Metrics: val-B mm error and rollout_4, DS-A slateN, error on val-B units of the same m.
   - **Keep** targeted acquisition only if it beats both controls by more than the spread
     between random draws.
   - Iterate at most 3 rounds, re-selecting failures each round.

## 8. (c) NFD with a retrieved reference (≤ 1 h training on the shared 8 GB GPU)

- **Inputs.** Same grid and rasteriser as `nfd_3ch_narrow_l20`: its dataset uses
  `resolution_scale: 0.5`, so match whatever grid it produces. Channels:
  1. occ(s_q);
  2. and 3. the two NFD action channels, unchanged;
  4. occ(s_i), the k = 1 donor's start;
  5. occ(s'_i), the donor's next state;
  6. optional: the retrieval clean prediction ŝ'_q (best retrieval variant, feasible cubes).

  in_channels = 5, or 6 with ŝ'_q. Ablate 5 vs 6.
- **Donor frame.** Map the donor's cube poses from its push frame into the *query's* push frame
  (rotation + translation only), express them in query world coordinates, and render with the
  **same** rasteriser as channel 1. So the reference is aligned with the query action in world
  coordinates.
  - Unit test (mandatory, given the goal-mask transpose history): rendering the query's own
    state through the donor path must reproduce channel 1 exactly. Rendering a mirrored donor
    must match its mirrored cubes.
- **Target and head.** As `nfd_3ch_narrow_l20`: residual true (EXP-0022/0025 found the residual
  head helps), same loss.
- **Size.** features [4, 8, 16] (the narrow model's; tiny) with the input layer widened. Try
  [8, 16, 32] only if time allows.
  - The narrow NFD trained 60 epochs in ~30 min on 12k rows.
  - Budget: 40 epochs, early-stop on the 5% val slice, batch 64, AMP on. Expect ≤ 45 min at
    12-25k rows while Genesis shares the GPU.
  - Precompute donor renders into a cached uint8 tensor (N × 2 × H × W; 12k at 64 px ≈ 100 MB),
    rather than retrieving on the fly.
- **Donor selection in training.**
  - For row j, take the top-3 by the frozen retrieval key over the bank, **excluding the same
    chain** (same file + `chain_env`) and, for DS-0010, the same `source_file`. Pick one of the
    three at random each epoch.
  - Leave-chain-out donors are slightly worse than test donors, which is conservative.
  - **Reference dropout** p = 0.15: zero channels 4-5, so the model degrades to a plain NFD
    rather than collapsing when the donor is bad.
- **Test-time.**
  - 1-step: retrieve the top-1 from the full bank.
  - Rollouts: extract 20 cubes from the predicted occupancy with the section 3 sharpening
    operator, retrieve on those, render the donor. Also report a variant driven by the retrieval
    model's own clean pose rollout.
- **Controls.**
  - (i) The same architecture and recipe with a **random donor** (same corridor count),
    trained and tested that way. This isolates the reference's information from the extra
    input capacity.
  - (ii) The plain narrow NFD (existing).
  - (iii) Reference dropout at test time (all donors zeroed), to show how much the model relies
    on the reference.
- **Metrics.** The R2/R3 headline: DS-A slateN for model selection; DS-0009 slateN_tough and
  rollout_1..8; DS-B terminal slateN.
- **Success.** Beats the narrow NFD on DS-0009 slateN_tough by more than 0.03 (beyond the seed
  floor) or on rollout_4 by more than 0.03, AND beats its random-donor control on both.

## 9. Good-science questions and how each is measured

| # | question | measured by |
|---|---|---|
| G1 | Which distance works: cube-set Chamfer vs Hungarian vs occupancy-patch L2 vs blurred patch; corridor weight; yaw; walls? | R1 on validation. Also retrieval quality: rank correlation between key distance and **outcome** distance (mm between donor-transferred and true displacements) |
| G2 | How to apply deltas: additive occupancy vs per-cube transport vs paste vs projected transport? | R1; mm error, rollout_4, DS-A `slateN`; feasibility (overlap and ghost count) |
| G3 | How much data, of what structure (branching vs chains, stratified, coverage-greedy), within ≤ 50k? | R4 budget × strategy table; headroom above 50k |
| G4 | Is "which cubes moved" enough to define the relevant state, or do stationary cubes matter? | R1: window from the donor's movers only vs full window vs movers + blockers ahead of them |
| G4b | Do non-contacting interaction groups move independently, so that composition is valid? | section 6 step 6: violation rate on the bank; composite error on violation vs non-violation rows |
| G5 | How local and smooth is the transition map? | error vs d1 curve (R6); neighbour-rank curve (section 3) |
| G6 | Information or noise? | section 3 controls (R2) |
| G7 | Does confidence predict error, and does it pick good data? | section 5 |
| G8 | Which image metric tracks planning utility? | section 4 |

## 10. Claims inherited from the original design dialogue: hypotheses, not facts

These came from an LLM dialogue, stated with more confidence than evidence. The status column
says where each is tested. Anything not listed was cut as rhetoric.

| claim | status (clean data, 2026-09-28) |
|---|---|
| Retrieval can rank action sets as well as learned models | **supported vs single models** (beats 2/3 NFD seeds at 1 step, all 3 plus linear at 3 pushes); **not resolved vs a 3-seed NFD ensemble** |
| Retrieval's advantage grows with horizon (it compounds less) | **supported** (per-seed delta +0.08 mean at 1 step, +0.15 at 3 pushes; its step-wise slateN is flat, the NFD's dips) |
| Only a few cubes matter per push; a geometric interaction set captures them | **supported** (96% recall / 90% precision; fixing the transfer to this set was the decisive change) |
| k > 1 gives better predictions | **supported** (k5 median 0.792 vs k1 0.757 at 1 step; 0.821 vs 0.787 at 3 pushes; paired k5−k1 at 3 pushes +0.034 [−0.007, +0.077], not resolved) |
| Transport (displacement) deltas beat image deltas | **supported, pre-fix** (under the oracle donor: displacement 1.18 mm vs paste 4.74 mm) |
| `accuracy` is a poor target for sharp models / disagrees with slateN | **supported** (chaos floor; clean data: the NFD leads accuracy, 0.557 vs 0.485, retrieval leads slateN) |
| Retrieval's lead is information, not feasibility or representation | **open** (sharpened-NFD control not run on clean data) |
| NN distance predicts failure; kNN disagreement does | pre-fix: disagreement yes, top-1 distance no. **Re-check** |
| More data keeps improving retrieval ranking | pre-fix: plateau beyond 25k. **Re-check** |
| Branching datasets beat trajectories; dataset design beats the index | untested |
| Push-frame canonicalisation beats world-frame retrieval | untested |
| Contact-mode hard gating is needed | not needed in the narrow domain (the interaction set suffices); untested beyond it |
| A learned correction on top of a retrieved reference helps (NFD with reference) | inconclusive (pre-cleanup data) |

## 11. Later / out of scope tonight (condensed from the original doc)

- **Beyond the narrow domain.** Variable push length (length becomes part of the key and the
  transfer must rescale), multi-layer piles (needs z in the key, and contact-mode gating may
  matter), real data (needs cube tracking for the transport delta).
- **"MPC utility of a transition" as a data-selection score.** How much a transition changes
  planning decisions on held-out goals. A heavier version of section 5's utility test.
- **CEM vs gradient planning with a non-differentiable model**; learned metric; closed loop at
  scale.
- **Background.** This is memory-based / locally weighted modelling (Moore's nearest-experience
  control, locally weighted MPC, kernel MPC). The original dialogue, with its
  references, is archived verbatim in
  `docs/experimental_design/archive/retrieval_based_modeling_original_dialogue.md`.

## 12. Open questions for the user

1. **Closed-loop test (next step 1).** Is a sampling planner (CEM / pure sampling over 20 mm
   pushes) acceptable, since retrieval is not differentiable? Should the NFD comparison use that
   same planner, or its usual GD planner?
2. **Real-robot relevance.** Is pose-based retrieval acceptable as the headline, with
   occupancy-only retrieval (next step 5) as a follow-up? Or should occupancy-only come first?
3. **Beyond the narrow domain.** Which extension matters most: variable push length, n50, or
   multi-layer piles?
