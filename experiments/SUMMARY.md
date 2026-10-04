# Experiment summary — by question, not by number

Written 2026-10-03 from a full pass over EXP-0001..0065, `REGISTER.md`, `INVARIANTS.md` and
`OPEN_ISSUES.md`. Synthesis only: numbers live in the records (cite `EXP-####`, `C-###`); each
affected older record now ends with a "Later evidence (2026-10-03 audit)" section that points
forward. When this file and a record disagree, the record wins — fix this file.

**Project question.** Which dynamics models (and model variants, datasets, state/action sampling,
MPC settings) give the best *closed-loop MPC* performance, and which offline metrics predict that.
Founding hypotheses:
- **H-scene**: scene parameters (object count, clumped vs scattered) and task (simple vs complex
  goal shape) change difficulty in non-obvious ways AND change which model / optimiser wins.
- **H-metric**: some offline metrics track closed-loop performance better than others and should be
  used for testing and model selection.

**Status words used below.** `live` (current best evidence) · `narrowed` (stands, but a later
condition changes it) · `superseded` (a later record answers the same question better) ·
`stale` (an input is now known wrong; not re-run) · `exposed` (a confound applies, size unmeasured)
· `hiatus` (paused, expected to return once the approach matures) · `dead line` (the direction stopped; not formally withdrawn) · `pending`.

---

## 0. Cross-cutting confounds — read before citing anything

**Task definition (user decision 2026-10-03):** in Genesis (rigid cubes) a push whose tool lands on a
cube is NOT part of the task — it would mostly fail in reality — so X1/X12 are defects there and the
affected results need re-running with a legality constraint. In FleX (flexible carrots) such touchdowns
are allowed for now (less likely to cause an acute failure); FleX data and tests may be re-run later.

| # | confound | records exposed | status |
|---|---|---|---|
| X1 | **Illegal tool touchdowns from the pre-fix pile-aware sampler.** ~45-55 % of candidates put the blade on a cube (ISS-010: DS-0008..0013; **EXP-0065 / ISS-013: DS-0001 46 %, DS-0006 54 %**). DS-0006 is the main offline benchmark and the start-state source of every closed-loop record; planner candidate banks used the same sampler until 2026-09-28. | offline on DS-0001: 0011-0014, 0016, 0023, 0026 A3, 0029 · offline on DS-0006: 0030, 0036-0038, 0040-0042, 0046, 0049 · narrow data: 0053, 0054, 0059 (pre-v2), 0060 · closed-loop candidates: 0032, 0039-0057 | **open, high** — only the narrow-domain data was fixed (DS-0015..18). **Measured on one table (EXP-0065 RUN-0002, EXP-0030's cache):** the true best DS-0006 push is illegal in 72 % (lyapunov) / 54 % (mass) of cells; legal-only re-scoring reorders models (τ 0.83), the linear model gains most (+0.11 beyond the pool-size effect). DS-0007 (Sean) is clean (≤ 0.2 % at n20/n50), so 0035/0036's DS-0007 results are the trustworthy offline evidence in that line |
| X2 | Goal-mask axis transposed before `28271c09` (2026-09-17) for asymmetric goals. | 0001-0003, 0006, 0008, 0011, 0013, 0017 (+ ISS-002 goal library → 0015, 0018) | open (ISS-003), never rescored |
| X3 | Benchmark physics ≠ training physics (DS-0001, slates_multistep). | 0001, 0002, 0006, 0008, 0010-0014, 0016, 0022, 0023, 0026-0029 | partly mitigated by EXP-0048 (20 mm scatter only) |
| X4 | Mistuned planners (GD lr 1.5e-3 / CEM pop 64). | 0032, 0037, 0039, 0041 | superseded by 0042 → 0044 |
| X5 | Lyapunov saturates near its ceiling by ~8-16 pushes and is ~identical across letters; it misreports letter completion. | every closed-loop study scored on lyapunov (0039, 0043-0045) | known since 0045/0046/0051; in-goal mass is now reported |
| X6 | Power: slateN gaps < 0.03-0.04 are inside slate-sampling (0026) and training-seed (0036) noise; most model comparisons are one seed. On DS-0006 about half of that seed noise was illegal-push outcomes (legal-only sd 0.010 vs 0.019, EXP-0065 RUN-0006). | nearly every offline comparison | standing |
| X7 | Closed loop only ever on **n20 scatter starts** (DS-0006 states 40-47), never more than 4 models on one task (~9 distinct models across all records), rank/GD/CEM planners. | all of 0039-0057 | standing; the main H-scene gap |
| X8 | GNN node sampling seeded by batch row (broken tag). | GNN rows of 0001, 0002, 0026-0028, 0033 | open |
| X9 | FleX corpora store actions as (x, −z). Now `holds` (pytest); EXP-0064's own adapter violated it. | 0064 as run (MODEL-0011) | fixed in 0064 (RUN-0005 retrain) |
| X10 | Collector "valid" flags miss escaped particles (FleX). | DS-0019 (2.3 %), DS-0020, DS-0021 (3.5-14.6 % rising with pile size), DS-0022 | filtered in 0061/0062 loaders and 0064 re-score; DS-0021 training not filtered |
| X11 | ISS-011: `train_nfd.py --override output.log_dir` silently resumed the ORIGINAL log_dir's checkpoint; which NFD trainings before 2026-09-28 used the override is unaudited. | none found (2026-10-04 audit of saved logs + ledger) | closed — fix in `train_nfd.py` |
| X12 | **Closed-loop planners EXECUTE illegal touchdowns** (EXP-0065 RUN-0003, C-067): lower bounds 16-53 % of pushes, model-dependent (0054: the winning worldframe NFD 0.69 vs 0.46-0.51; 0051: NFD ~2x linear); even the perfect-model CEM (0057) does it. Closed-loop model comparisons partly compare how much each model's objective rewards illegal pushes. Decided 2026-10-03: not allowed in Genesis (see top of §0). | 0050-0057 measured; fix `legalize_pushes`; 0051 and 0054 re-run legally 2026-10-04 — conclusions hold (0054's inversion widens). Other closed-loop records still unrepeated 

---

## 1. Measurement instruments: ground truth, noise, power

| id | question | status | takeaway |
|---|---|---|---|
| 0024 | Is re-simulating one push from one snapshot repeatable? | narrowed by 0027 | Within one call yes; across calls ~1 mm drift (0027 RUN-0005), amplified ~13x by hard-image scoring |
| 0027 | 30-goal harness; GT repeatability | live | More goals resolve more pairs, but models reorder between goal sets (C-034 narrowed) |
| 0028 | Soft mass-conserving truth | live | Rankings unchanged, sub-pixel stable — now the standard truth |
| 0031 | Collector vs rollout path agree? | live (exposed X1) | r ≈ 0.99 but ~12 % of between-push sd; evaluate through ONE path |
| 0026 | Are benchmarks powered for the margins read? | live | No: < 0.03 slateN unresolvable at 20 slates x 3 goals; goal averaging helps (C-029, C-030) |
| 0036 | Training-seed noise | live | slateN seed sd 0.01-0.04, accuracy 0.003 — accuracy is far more seed-stable |
| 0009 | Equal-wall-clock candidate budgets | live, context | Throughput differs 100x between models; matters only at tight budgets (0045) |
| 0043 | Batched closed-loop runner = sequential? | live | Yes (7/8 cells); runner for all later closed loop |
| 0012 | DS-0001 near-zero slateN = corpus difficulty? | narrowed by 0013 (and X1, X3) | Only for point-mass readouts, not image predictors |

## 2. Metric validity (H-metric)

| id | question | status | takeaway |
|---|---|---|---|
| 0006 → 0008 | Does slateN validate the stage-2/3 family; does accuracy agree? | 0006 superseded by 0008; both X2/X3 | A 0-accuracy descriptor model ranks well in-corpus, fails out-of-corpus (C-008) |
| 0025 | Flow head vs direct NFD | live | GT-flow ceiling has LOWER accuracy than a learned model yet higher slateN — accuracy punishes plausible-but-shifted predictions |
| 0033, 0034 | Does accuracy rank like slateN (across / within family; with blur)? | live, offline-only | Across families yes (τ 0.79), within NFD family no; blur helps n.s. (0060 finds the opposite on narrow data) |
| 0038 | Are offline ranking metrics redundant? | live (X1) | Nearly (τ 0.89-1.0) for NFD-like models; not for retrieval (0060) |
| 0037 → 0041 | Does ranking skill predict gradient-objective quality? | narrowed (X4, X1) | No (C-041); accuracy predicts GD gain within NFDs (0041) — both at mistuned GD |
| 0039 → 0044 | Which offline metric orders models like closed loop does? | 0039 narrowed by 0042/0044 | Accuracy was **not computed for the linear model**, so it was scored on NFD-only pairs: on the common cells accuracy 4/4 vs slateN 2/4 (0039) and 3/4 vs 1/4 (0044). Seed-sized gaps of a saturated task (X5, X7); slateN was scored on 54 %-illegal DS-0006 pools over 160 states, closed loop on 8 of them (X1); the decisive model (worldframe ep43) was itself picked for its accuracy (selection) |
| 0054 | Narrow NFDs: more accurate and higher slateN offline — better closed loop? | live (X1: dirty training data) | **No — reversed.** Strongest evidence that both offline metrics can invert vs closed loop under a domain/action-space shift |
| 0059 (chaos floor) | Robustness of metrics to state perturbation | live, offline | slateN robust, accuracy collapses with 0.5-1 mm perturbation |
| 0060 | Which cheap metric tracks slateN_tough? | live (clean re-run 2026-10-04) | On clean DS-0016 across NFD / linear / retrieval (9-15 models), accuracy_1 does NOT track slateN_tough (τ +0.03..+0.22); in-goal-mass prediction error does (τ +0.70..+0.94) (C-072). Target is itself offline |
| 0064 | Particle accuracy vs slateN across pile sizes (FleX) | live, one seed | Corrected GNN: accuracy 0.48-0.58 vs an untrained push-field heuristic's 0.08-0.50, yet the heuristic ranks as well (point goals) or better (mask goals), increasingly on large piles. Another across-model accuracy/slateN dissociation (C-068) |
| 0064 (RUN-0013..0015) | NFD vs LinearForesight vs GNN on the same program, image-mask truth | live, one seed each | NFD > switched LF > untrained push field > particle GNN in every group (C-069). Accuracy and slateN agree across models within a group (Kendall 0.71-0.93) and moderately per state (Spearman 0.48-0.60), but not consistently across pile-size groups within a model (C-070) |

## 3. Closed-loop MPC: planners, objectives, sampling

| id | question | status | takeaway |
|---|---|---|---|
| 0023 | Model as gradient source (one step) | superseded by 0037 | Underpowered, off-physics |
| 0032 | Closed-loop pilot | superseded by 0039/0042/0044 | Planner choice dominated model choice |
| 0042 | Planner parameter sensitivity (one step) | live (X1) | Defaults far off; tuned GD ≈ tuned CEM; tuning moves GD more than models do |
| 0044 | Tuned planners, 12 goals x 8 starts | live (X5, X7, X1, X12) | All 4 models within ≤ 0.016 — an unresolved null at the design's own resolution (0.017 GD / 0.014 CEM; seed pair 0.0105); 3 saturating quadrants pooled with letters dilute gaps (C-049) |
| 0045 | Planning time vs number of pushes | live (X5, X7) | > 0.1 s (CEM) / 0.3 s (GD) gains nothing; model speed decides only below 0.1 s |
| 0052 | Success objective (lyap − w·mass) | refuted (as pre-registered) | Only w3-GD resolved (+0.09); the other 3 cells n.s. |
| 0055 | Capacity-aware values (EMD, OT, crowding) | live | Only crowding helps (+0.03), and it breaks quadrants |
| 0056 | Goal-aware candidate selection | live (X1) | No gain over pile-aware |
| 0040 | Objective smoothness explains GD quality? | dead line | Warp roughens the objective but roughness doesn't predict GD gain |
| 0049 | Which candidate sampler? | **stale until re-run** (X1) | "Pile-aware best" (C-054) measured with the illegal-touchdown sampler, and it is the rationale for every closed-loop candidate bank |

## 4. Task / goal difficulty and ceilings

| id | question | status | takeaway |
|---|---|---|---|
| 0046 | Achievable ceiling per goal | live | Every goal is nearly satisfiable by 20 flat cubes; 30 goals ≈ 12 distinct ranking clusters |
| 0051 | Does lyapunov MPC complete goals? | live | Quadrants yes (~5 pushes), two_squares mostly, **letters never** (0/48 at 0.9x) |
| 0050 | Simulator-as-model planning | live (X1) | Not a ceiling at feasible budgets; no better than learned models |
| 0057 | Perfect-model CEM budget split | **results on disk, record not written** (paused) | Perfect model at 256 sims/decision ≈ learned NFD; letters still mostly unsolved; more CEM iterations > bigger pool |
| 0058 | Human demonstrations | design only, no data, no EXPERIMENT.md | Would separate "letters are hard" from "planners are bad" |

## 5. Scene, state distribution and sampling (H-scene)

| id | question | status | takeaway |
|---|---|---|---|
| 0002 | Accuracy across spawn modes | live, weak (X2, X3, tiny n) | Accuracy stable across piled/scattered/mixed |
| 0029 → 0030 → 0035 | Is a model's per-state advantage a state property (switch per state)? | 0029 superseded; C-036 narrowed | **No at scale** — per-state switching is significantly worse than the best single model |
| 0047 | Is there clean narrow-domain data? | live | No → collected DS-0008..13 (later dirty, X1) → DS-0015..18 |
| 0053 | Narrow vs broad training (offline) | **exposed X1** (DS-0008/9 dirty) | Narrow better offline; clumps predicted better than scatter by every model |
| 0061 → 0062 | FleX: does model ranking depend on pile type? | 0061 narrowed by 0062 | The GNN-vs-LF flip between blob/spread piles disappears with matched training data (one seed each) — coverage dominates; a smaller scene effect is not excluded |
| 0064 | Object count / pile size (FleX) | live, one seed (as-run claim refuted) | The source study's "capture rises with object count" came from a z-mirrored action input (C-065 refuted). Corrected: accuracy falls with pile size, slateN is flat; the learned GNN adds ≤ ~0.02 slateN (point) over an untrained heuristic of its own action encoding at every size (C-071, 3 seeds + encoding + escape-filter variants); the encoding choice matters more than learning; the size trend of the GNN's edge flips with encoding (C-068 narrowed). Count is confounded with footprint and node budget |
| 0065 | Are DS-0001/0006 legal? | live | No (X1) |

## 6. Model families, compared offline

| id | question | status | takeaway |
|---|---|---|---|
| 0001 | NFD vs GNN (Genesis) | narrowed (X2, X3, X8, stale JSON) | NFD more accurate; replicated on FleX (0061, 0062) |
| 0003, 0013 | Switched vs global linear foresight; resolution | 0003 partly reversed by 0013 | Switching helps; 64x64 beats 32x32 on fair scoring |
| 0010 | Multi-step fine-tuning | live (X3) | Roughly doubles step-3 accuracy and helps terminal slateN (C-012); linear fine-tune diverges (C-013) |
| 0014, 0021 | Visual-switched vs NFD vs descriptor on DS-0001 | 0014 exposed (X1, X3); 0021 abandoned | One-goal ordering, never repeated |
| 0022 | Push-frame-warped NFD | refuted, mostly within noise | Warp doesn't help |
| 0025 | Flow-head NFD | refuted (L20mm only) | Direct prediction wins on accuracy |
| 0035, 0036, 0030 A5 | Ensembles | live (X1 for DS-0006 parts) | Ensemble wins at equal candidate count, loses at equal wall-clock (C-039 vs C-043) |
| 0053 | Narrow-trained NFD / linear | exposed X1 | — |
| 0063 | Soft-occupancy NFD | live, one seed | Blurred training raises slateN_tough; output blur doesn't — reaches retrieval's 3-push level |

## 7. Retrieval models

| id | question | status | takeaway |
|---|---|---|---|
| 0059 | Retrieval (per-cube nearest transition) vs NFD / linear (narrow clean domain) | live (offline; ISS-012 for k selection; narrowed by 0063) | At 3 pushes ranks better than every hard-raster NFD seed (all resolved); at 1 step only 2/3 seeds resolved; ≈ 3-seed ensemble; NFD leads accuracy. Reads true cube poses (information asymmetry); no retrieval noise floor (bank bootstrap); no closed loop |

## 8. Lines on hiatus, and stopped lines

**On hiatus** (user decision 2026-10-03): paused, not dead. Embedding-based prediction and retrieval
are expected to be useful once mature (retrieval, §7, is the active member of this family).

| line | records | where it paused |
|---|---|---|
| Analytic descriptors / DMDc / value readouts | 0004, 0005, 0011, 0015, 0017, 0018 | Descriptors added nothing over a visual channel (0005); descriptor models can't roll out multi-step without a reframing transform (C-014); readout instruments leaked (0017) and 0018's cache is stale (ISS-002) — fix those before resuming |
| Learned latents (Stage-2 hybrid, LeJEPA) | 0007, 0016, 0019, 0020, 0021 | The random-encoder floor was hard to beat (0016); **0019/0020's rank-collapse conclusions are refuted by ISS-001** (their latent-R² results stand); 0021 never ran |

**Stopped**

| line | records | why it stopped |
|---|---|---|
| Push-frame warping | 0022, 0040 | No gain, rougher objectives |
| Per-state model switching | 0029, 0030, 0035 | Fails at scale (scene-REGIME switching is untested — keep open) |

## 9. Physics and simulator fidelity

| id | question | status | takeaway |
|---|---|---|---|
| 0048 | Do our three physics sets change 20 mm push outcomes? | live, scatter only, 5/12 states | About as much as 0.5 mm action error; rankings unchanged (ρ 0.99) |
| 0024, 0031 | Repeatability, cross-path agreement | live | See §1 |
| 0064 (unrelated finding) | FleX explosion rate vs pile size | live | Valid share 96 % → 79 % with pile size; escaped particles slip through validity flags |

---

## 10. Where the hypotheses stand

**H-metric.** The only closed-loop anchors are EXP-0039/0044 (4 models, n20 scatter, saturated
lyapunov task) and EXP-0054 (one domain shift). The "accuracy beats slateN" lean rests on NFD-only
pairs (accuracy was never computed for the linear model), on seed-sized gaps, with slateN scored on
54 %-illegal pools over a different state population, and with the decisive model selected for its
accuracy. **Re-scored on legal-only candidates (EXP-0065 RUN-0005), slateN's agreement with those pairs goes
from 1/5 to 4/5 (by tiny margins), and optimism also agrees 4/5; the lean toward accuracy does not survive.**
Both metrics inverted in EXP-0054. Offline, accuracy and slateN disagree whenever model
structure differs (0006, 0025, 0053, 0059, 0063). slateN is noisier across seeds (0036) but robust to
state perturbation (0059). And closed-loop scores themselves are contaminated by executed illegal
pushes (X12). **EXP-0039 RUN-0002 (2026-10-04, pre-registered) ran that test:** legal actions, thin letters scored by in-goal
mass, 8 models across families, CEM and rank. Image accuracy is at chance (pair agreement 0.50 / 0.57);
goal-specific ranking metrics carry the signal (slateN_tough rho +0.60 / +0.57, top-1 regret +0.50 / +0.81;
C-073). The planner changes the winner, and closed-loop seed spread is as large as many family gaps.

**H-scene.** Goal complexity has a large, consistent effect: quadrants are easy, and letters are
never completed by any planner, the simulator planner or the perfect-model CEM (0046, 0050, 0051,
0057). Goal complexity changes which OBJECTIVE helps (0055: crowding helps letters, breaks
quadrants), but not yet which model wins, because closed-loop model gaps are unresolved everywhere
tested (X7). The one regime where models clearly switch winner is the **time budget** (0045: at
0.03 s the linear model beats the NFD by 0.20; C-043: the ensemble loses at matched time). Scene:
clumps are easier to predict offline (0053) and the sampler matters more on clumps (0049), but both
are exposed to X1. Per-state model choice fails (0035), but scene-REGIME switching (clump/scatter, n)
has never been tested. On FleX (0064) the learned GNN adds little over an untrained push heuristic built from its own action
encoding at any pile size (C-071); whether its small edge shrinks or grows with pile size depends on the
encoding (C-068 narrowed), so no robust model x pile-size interaction remains. The one apparent model × scene interaction (0061) was dominated by training coverage
(0062). **EXP-0039 RUN-0003 (2026-10-04, pre-registered) put clump starts into closed loop:** clump starts are much
easier (+0.22 in-goal mass / optimum), but the model order changes only modestly (Kendall 0.64, no resolved
reversal; C-074) -- regime switching is neither shown nor excluded. n50, pile and FleX starts remain untested.

**Which model is best for MPC.** On the only closed-loop task measured, models are not separable
after planner tuning (0044), and planner, objective and time budget matter more (0042, 0045, 0052,
0055). A perfect model doesn't fix letters (0050, 0057). Offline leaders: retrieval / soft NFD / NFD
ensembles (narrow Genesis), NFD (FleX). None has been tested in closed loop against the others, and
the leaders are slow or unmeasured for throughput, which is the regime that does separate models.

---

## 11. Holes and directions

Ranked by how much they could change the project's conclusions. Items marked *(review)* came from an
adversarial review of this file (`experiments/temp/2026-10-03-organize-and-summarize/review_fable.md`).

### Glaring (fix before building on the affected results)
1. **Legality of the action space (X1, X12; ISS-013).** Decided: illegal touchdowns are not part of the
   Genesis task. Add a legality projection to the planners' candidates/refinement and to
   `execute_action`, then re-run the headline closed-loop cells (0044 GD/CEM, 0051, 0054). Offline:
   EXP-0065 already re-scored EXP-0030's table (the best actions are mostly illegal; ordering moves);
   re-score 0036-0038 and 0042 on legal-only candidates. Re-run 0049 with `pile_aware_action_batch`
   (its C-054 is the sampler rationale for every candidate bank). Re-run the rank planner with the
   legal sampler: its unexplained flat result may be an X1 symptom *(review)*.
2. **Is slateN's closed-loop "loss" an artefact?** *(review)* Zero-simulation: re-score the 0039/0044
   models on legal-only DS-0006 pools restricted to the closed-loop states 40-47. Add accuracy for the
   linear model, plus optimism and large-K top-1 regret; optimism is the error a planner exploits and
   has never been correlated with closed loop. Then recount agreement with the resolved closed-loop
   pairs. < 1 day.
3. **DONE 2026-10-04 (EXP-0039 RUN-0002, C-073): accuracy at chance, slateN_tough / top-1 regret predictive; next: add retrieval (needs an OCC adapter) and clump starts.** Original plan: a closed-loop metric-validity run with headroom, across families. Score by in-goal mass /
   completion time at early k, letters separately from quadrants, with legal actions. Zoo: NFD seeds,
   ensemble, soft NFD (0063), retrieval k5 (0059), linear-switched, a weak model. Compute every
   offline metric for every member on ONE clean corpus (DS-0016) *(review)*. Free pre-step: re-analyse
   the existing 0044/0051/0054/0055 episodes letters-only per k, with on-policy predicted-vs-true dv
   per executed push, which gives on-policy accuracy and optimism *(review)*.
4. **Selection circularity on `nfd_residual_worldframe_noaug_ep43`** *(review)*. It is an interrupted,
   confounded run picked for its accuracy, and it carries most resolved closed-loop pairs. It is also
   the model that executes the most illegal pushes (0054). Train two converged seeds of that recipe
   and re-enter the 0044 cells.
5. **Partly done 2026-10-04 (EXP-0039 RUN-0003, C-074: clump much easier, order ~stable; n50 not yet).** Closed loop had only ever seen n20 scatter starts. Clump (0049 constructor) and n50 (DS-0007)
   starts, the same zoo, the legal sampler, scored by in-goal mass. If model order changes by start
   regime, regime switching (the founding H-scene idea) is back; if not, H-scene reduces to goal
   complexity and budget.
6. **Budget as a regime axis** *(review)*. Re-run 0045's 0.03 / 0.1 s cells with the ensemble,
   retrieval and soft NFD. It is the only regime with an observed large model gap, and the offline
   leaders are the slow ones.
7. **Records.** Write up EXP-0057 from `results/analysis.json` (paused by the user). Give EXP-0058 a
   stub or withdraw it. Audit ISS-011 exposure (X11).

### Tighten (worth more runs)
- **0059 retrieval vs 0063 soft NFD vs ensembles:** soft-NFD seeds, a retrieval noise floor (bank
  bootstrap) *(review)*, paired comparison on the same pools, ISS-012-clean re-selection of k,
  equalise retrieval's cube-pose input. Note that slateN_tough's goal subset was chosen after seeing
  the full table *(review)*. Then put the top two into closed loop (hole 3).
- **0054's inversion:** pushes were wall-clipped to ≥ 7 mm, out of distribution by construction for
  exact-20 mm narrow models *(review)*; count clipped pushes per model from the logs. Then re-run
  with DS-0015-trained narrow NFDs, with CEM as well as GD, and with legal actions.
- **0064 (FleX pile size):** C-068 (learned GNN loses to an untrained push heuristic as piles grow) is
  one seed. Add 2 seeds (~100 min each), an escape-filtered retrain and the baseline action encoding
  (issues.md I-2/I-3). Then use a matched design: count at fixed footprint, a node budget that scales
  with the pile, and clumped vs scattered at fixed count. That needs new FleX collection in the
  source repo.
- **Multi-step fine-tuning (0010, C-012)** never tried on clean narrow data or in closed loop
  *(review)*.
- **Seed floors** (0036-style) for every family used in a headline comparison; only NFD is measured.

### Drop or formally close
- Descriptor / value-readout and LeJEPA lines (§8) are on hiatus, not closed. Before resuming: regenerate
  0018's stale cache (ISS-002) and re-read 0019/0020 through ISS-001.
- C-036 per-state switching: withdraw the per-state claim. **Do not** close scene-REGIME switching
  until hole 5 runs *(review)*.
- Push-frame warping: closed. Flow heads: background, not refuted (C-028's ground-truth-flow
  ceiling ranks above every trained cell; 0063 shows blur helps ranking) *(review)*.
- C-020 / EXP-0021 (DS-0001 ordering): DS-0001 is off-physics and half illegal; don't repeat on it.

### Subtle holes
- **Normalised capture across scenes.** slateN's denominator changes with the scene (0064: it halves
  from small to large piles; 0028: up to 17.9x across push lengths). Cross-scene slateN needs regret
  in value units beside it (METRICS caveat 3).
- **Offline/closed-loop mismatches** *(review)*: offline metrics are scored on step-0 spawn states at
  K = 64-128, while planners act on half-organised piles and evaluate ~10^4 candidates. The offline
  state population (160 states) differs from the closed-loop one (8 starts).
- **Training/test action distribution vs the planner's search space.** Narrow models were trained on
  sampler starts the planner never uses (0051: planners start ~13 mm before the first cube). No study
  varies this.
- **The planner's objective is also the score** (lyapunov in, lyapunov out) in 0039-0045.
- **Retrieval's privileged input** (true cube poses vs occupancy).
- **One goal per state, few states** remain common (0064 as run; 4-8 closed-loop starts).
- **Collector validity flags are trusted by default.** FleX escaped rows (X10) and Genesis illegal
  touchdowns (X1, X12) were both found late, by hand. A shared legality/sanity check at dataset
  registration and at `execute_action` would have caught both.
