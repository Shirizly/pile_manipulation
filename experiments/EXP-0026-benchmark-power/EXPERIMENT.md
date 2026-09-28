---
# ---- identity -------------------------------------------------------------
id: EXP-0026
title: >
  The model benchmarks are underpowered for the margins they are read at:
  20-21 slates resolve slateN gaps of ~0.05-0.1, not the 0.01-0.03 the
  reference table is compared on; EXP-0023 would need ~75 states to rank arms
  on gradient_gain (dv_grad already separates them); and extra goals are
  nearly free replicates of extra states on homogeneous corpora
tier: T1
mode: confirmatory              # C2-C4 pre-registered in DESIGN.md before running; C1 is exploratory (see body)
date: 2026-09-23
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  Under a PAIRED, state-resampled analysis (Baselines/common/paired_stats.py):
  (C1) EXP-0023's 10 DS-0001 states (corner/lyapunov, 6 arms, 100-action pool,
  120 Adam steps) resolve no arm pair on gradient_gain after Holm correction,
  and a 0.01 gradient_gain difference needs >= 30 states at 80% power;
  (C2) on the eval_report harness (L20mm / L40mm / randlen_test, 20-21 slates,
  3 goals x 3 value fns, goal-averaged per slate) the 95% bootstrap CI of the
  lyapunov slateN difference linear_switched_res32 - nfd_randlen includes 0
  on at least 2 of 3 corpora; (C3) pairs with |delta slateN| < 0.03 are
  unresolved in >= 80% of cases on that harness; (C4) averaging per-slate
  slateN over 24 random goals instead of 3 shrinks the median per-pair sd of
  the paired difference by >= 2x.

prediction:
  supports: >
    C1: 0/15 Holm-significant pairs on gradient_gain AND required_n(0.01) >= 30.
    C2: CI includes 0 on >= 2 of 3 corpora. C3: >= 80% of |d|<0.03 pairs
    unresolved. C4: sd shrink G=3 -> G=24 >= 2x.
  refutes: >
    C1: any Holm-significant pair, or required_n(0.01) < 30. C2: CI excludes 0
    on >= 2 corpora. C3: < 80% unresolved. C4: shrink < 1.5x.
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 3bae8cd7
  dirty: true                   # see "What was actually run" -- this session's own
                                # uncommitted code (listed there) plus pre-existing
                                # unrelated user edits
  script: >
    code/a1_exp0023_paired.py, code/a2_slaten_paired.py, code/a3_pool_vs_states.py,
    code/a4_goal_scaling.py; Baselines/common/eval_report.py (RUN-0001);
    reusable: Baselines/common/paired_stats.py (new)
  data: >
    EXP-0023 results/metrics.json (DS-0001 states); eval_report CORPORA L20mm,
    L40mm (slates_multistep n20), randlen_test (overnight_randlen test_all);
    experiments/temp/binned-pools/dv_cache_corner.pt (DS-0001 20 x 1000)
  code_path: >
    eval_report._load_cell -> predictor.predict_occ -> _capture_report
    (goals.slate_n_capture, per slate) -> paired_stats; A3: slate_n_capture on
    subpools of cached dv; A4: same predictions, 24 random goals
  seed: "bootstrap/permutation rng seed 0; A3 subpool seed 0; A4 goal seed 12345, subset seed 0"
  split: "not applicable -- no fitting; eval corpora are the harness's held-out cells"
  data_commit: not applicable
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004, RUN-0005]
  runtime: "RUN-0001 ~25 min CPU; RUN-0005 ~40 min CPU; the rest seconds to 1 min"

budget:
  declared: "~3 h compute (user), Phase A scope"
  spent: "~2.5 h wall-clock, ~65 CPU-min, 0 GPU-min of model training; no Genesis"
  outcome: within

design:
  varied:
    models: "12 eval_report models (RUN-0001); 6 EXP-0023 arms (A1); 3 cached DS-0001 models (A3)"
    corpus: "L20mm, L40mm, randlen_test"
    value_fn: "lyapunov (primary), mass_in_region, signed_mass"
    pool_size_K: "25, 50, 100, 250, 500, 1000 (A3 only)"
    n_goals_G: "1, 3, 6, 12, 24 (A4 only)"
  held_fixed:
    checkpoints: "as registered in eval_report.MODELS / OCC_ADAPTERS at 3bae8cd7"
    slate_set: "the harness's own step-0 pools (20 / 20 / 21 slates)"
    goal_set_A2: "random_quadrant (per-slate seeded), ring_O, T -- the harness's"
    replication_unit: "the slate; goals averaged within slate before any test"
  baselines: >
    random (slateN expectation exactly 0, eval_report's MC floor reported in the
    artifact); persistence reported but degenerate as a ranker (METRICS.md).
    For A1, EXP-0023's own random-pool and pool-ceiling floors.
  metric: slateN (primary), gradient_gain, dv_grad, top1_regret (A3), accuracy (reported by RUN-0001, not analysed)

noise_floor: >
  This record's subject is the noise floor itself. Per-pair sd of the paired
  per-slate lyapunov slateN difference (median over 66 pairs): 0.136 L20mm,
  0.070 L40mm, 0.185 randlen_test (goal-averaged, G=3). EXP-0023: median
  per-pair sd of gradient_gain 0.030, dv_grad 0.033, dv_rank 0.015 (lyapunov
  units). No SEED-level floor is measured here (TODO H1) -- every model is one
  training run, so all margins below also omit training noise.

depends_on: [goal-mask-axis-convention-row-y-col-x, genesis-snapshot-restore-repeat-determinism,
             occ-gradient-adapter-matches-offline-predictor, randlen-step0-pool-size-128]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  C1 SUPPORTED (exploratory): 0/15 arm pairs Holm-significant on gradient_gain,
  Friedman p=0.33, 74 states needed for d=0.01 -- but dv_grad separates arms
  globally (Friedman p=0.004). C2 REFUTED as stated: the linear-vs-NFD CI
  excludes 0 on 2 of 3 corpora, in OPPOSITE directions (L20mm +0.049
  uncorrected, randlen_test -0.085). C3 SUPPORTED: 78/81 pairs with
  |d|<0.03 unresolved; a 0.02 margin needs 100-680 slates at 3 goals.
  C4 NARROWED: G=3->24 shrinks paired sd 2.30x / 2.85x (L20mm / L40mm,
  lyapunov) and 2.19-2.44x (mass_in_region, all corpora), but only 1.55x on
  randlen_test/lyapunov (goal ICC 0.13), between the pre-registered
  thresholds. The conjunctive claim fails on C2, hence `refuted`; the
  sub-claims carry their own register rows (C-029..C-032).
verdict: refuted
downgrades: [indirectness, inconsistency, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0023's power check compared the sd of arm means (0.0105) with the sd of
state means (0.0163). The second is a population property of the states; it
does not shrink with more states, so TODO M2's "add states until it passes"
could never be satisfied by the design it prescribes, and it is not the noise
a paired design is exposed to (every arm sees the same states, so difficulty
cancels in a difference). The noise that matters is the model x state
interaction. Every claim here is therefore a statement about per-pair paired
differences, and each has a plausible opposite outcome: if models agreed
slate-by-slate on which slates are hard, paired CIs would be tight and 20
slates would already have been enough.

## What was actually run

- RUN-0001: `eval_report.py` with a new additive `per_slate` output (the
  summary means are computed exactly as before), 12 models x 3 corpora.
- RUN-0002 (A1): re-analysis of EXP-0023's 60 cells; no simulation.
- RUN-0003 (A3): pool-size subsampling of DS-0001's 1000-candidate pools.
- RUN-0004 (A2): paired analysis of RUN-0001.
- RUN-0005 (A4): 24 random goals (disks / rectangles, 2-30% of the grid).

**C1 is exploratory**: EXP-0023's arm means were seen before this record's
prediction was written. C2-C4 were pre-registered in `DESIGN.md` (C4 in its
addendum, written after A2 showed goal ICC ~ 0 and before A4 ran). A 2-model
smoke run of A4 preceded the full run; it showed the random goals sit near
the capture ceiling -- noted in the session before the full run, NOT written
into DESIGN.md -- and its cells were not reused.

Dirty tree at run time: this session's uncommitted edits to
`Baselines/common/eval_report.py` (per-slate output, additive),
`Baselines/common/goals.py` (sense-safe metrics, additive),
`simple_mpc/adapters.py` (`assert_dv_convention` generalised, `value_fn` /
`higher_is_better` attributes), new `Baselines/common/paired_stats.py`, new
`simple_mpc/gt_bank.py`, plus the user's pre-existing edits (two MODELS
entries in eval_report.py, EXP-0022 logs, weights/*/tests.md).

**Known-number checks.** (1) A1: the two-way residual sd reproduces
EXP-0023's 0.01826 exactly after rescaling its df (it took np.std over all 60
cells, df 59; the two-way residual df is 45 -> 0.02091). (2) RUN-0001
reproduces every goal-averaged number in `experiments/HANDOFF-model-benchmarks.md`'s
reference table to 3 decimals. It does NOT reproduce the stored
`Baselines/common/runs/cross_corpus_report.json` (2026-09-10; diffs up to
0.063) -- that file predates the goal-mask transpose fix (28271c09) and is
stale; see Unrelated findings.

## Numbers

### C1 — EXP-0023, paired (results/a1_exp0023_paired.json)

| quantity (higher = better) | global test | pairs CI-resolved | Holm < .05 | median sd_diff | states for d=0.01 (Bonf) | d=0.02 (Bonf) |
|---|---|---|---|---|---|---|
| gradient_gain | Friedman p=0.33, ANOVA p=0.043 | 6/15 | 0/15 | 0.030 | 74 (135) | 20 (37) |
| -dv_grad | Friedman p=0.004, ANOVA p=3e-5 | 9/15 | 0/15 | 0.033 | 87 (158) | 24 (43) |
| pool_escape | identical to -dv_grad pairwise (pool ceiling cancels) | | | | | |
| -dv_rank | Friedman p=0.001, ANOVA p=1e-6 | 7/15 | 0/15 | 0.015 | 20 (37) | 7 (13) |

With S=10 the exact sign-flip test has minimum p = 2/1024; Holm over 15
pairs puts even a 9-1 split at 0.059, so NO pairwise result could pass at
this n regardless of the data. P(first) under state bootstrap: gradient_gain
nfd_warped_randlen 0.76; dv_grad nfd_warped_randlen 0.60 /
linear_switched_soft 0.27; dv_rank linear_switched_soft 0.74.

### C2 / C3 — eval_report harness, lyapunov, goal-averaged (results/a2_slaten_paired.json)

| corpus | slates | Friedman p (W) | pairs CI-resolved | Holm<.05 | |d|<0.03 unresolved | median sd_diff | slates for d=0.02 / 0.05 |
|---|---|---|---|---|---|---|---|
| L20mm | 20 | 4e-14 (0.40) | 38/66 | 19/66 | 19/19 | 0.136 | 368 / 61 |
| L40mm | 20 | 1e-7 (0.24) | 22/66 | 9/66 | 42/45 | 0.070 | 100 / 18 |
| randlen_test | 21 | 1e-27 (0.67) | 47/66 | 42/66 | 17/17 | 0.185 | 676 / 110 |

C3 over the three lyapunov cells: 78/81 = 96% of |d|<0.03 pairs unresolved.
mass_in_region / signed_mass are noisier still (e.g. randlen_test signed_mass
median sd_diff 0.201; see json).

C2, `linear_switched_res32 - nfd_randlen`, lyapunov:
L20mm +0.049 [+0.013, +0.090] (uncorrected p=0.023, Holm 0.88);
L40mm +0.011 [-0.013, +0.034];
randlen_test -0.085 [-0.140, -0.041] (Holm 0.01).
CI excludes 0 on 2 of 3 corpora, in OPPOSITE directions -> C2 refuted as
stated. Read as: linear beats NFD on L20mm (uncorrected only), NFD beats
linear on randlen_test, L40mm is a tie at this power. The handoff's "linear
beats NFD at both L20mm and L40mm" is supported at L20mm only, and only
without multiplicity correction.

Every NFD variant vs `nfd_randlen` on lyapunov (warped, warped+flip,
warped+flip+residual, epoch-30 checkpoint): no CI excludes 0 on any corpus
(largest |d| 0.033). Top-1 is unstable everywhere (bootstrap P(first) <= 0.56).

Model ORDER does not transfer across corpora: Kendall tau of the 12 model
means, lyapunov, L20mm~L40mm +0.42, L20mm~randlen +0.33, L40mm~randlen -0.12
(gnn_l20l40 is 1st on L20mm and 12th on randlen_test -- the L-corpora are
in-distribution for the l20l40-trained models).

Goal replication: median ICC of per-(slate, goal) paired differences
0.00-0.10 in every corpus x value-fn cell -- the three goals behave as
independent replicates of the slate.

### A3 — pool size (DS-0001, results/a3_pool_vs_states.json)

| K | nfd | visual-switched | descriptor | nfd - vs mean | sd(one pool/state) | per-state SNR |
|---|---|---|---|---|---|---|
| 25 | 0.652 | 0.775 | -0.031 | -0.123 | 0.419 | 0.29 |
| 100 | 0.589 | 0.746 | -0.028 | -0.156 | 0.356 | 0.44 |
| 1000 | 0.516 | 0.740 | +0.011 | -0.224 | 0.292 | 0.77 |

slateN is a function of K (NFD 0.65 -> 0.52), and model gaps WIDEN with K
(selection pressure). At K=100, within-slate pool-sampling variance of the
paired difference (0.115) is ~4x the between-slate variance of its mean
(0.030). Per-state SNR rises 2.6x from K=25 to K=1000 (~7x fewer states), but
per SIMULATED ACTION small pools are cheaper (S*K ~ 300 at K=25 vs ~1700 at
K=1000, relative units) -- and they measure a different quantity.

### C4 — goal scaling (results/a4_goal_scaling.json)

Median per-pair sd of the goal-averaged per-slate paired difference, and
slates needed for a 0.02 margin (80% power, alpha 0.05), by number of goals G
(24 random disk / rectangle goals, 200 random goal subsets per G):

| corpus / vf | G=1 | G=3 | G=6 | G=12 | G=24 | shrink G3->G24 | ICC(24) | pairs resolved G=3 -> 24 (one subset) |
|---|---|---|---|---|---|---|---|---|
| L20mm lyapunov | 0.131 (339) | 0.080 (128) | 0.059 (71) | 0.045 (42) | 0.035 (26) | **2.30x** | 0.016 | 30 -> 52 / 66 |
| L40mm lyapunov | 0.048 (48) | 0.033 (24) | 0.024 (14) | 0.017 (8) | 0.012 (5) | **2.85x** | 0.002 | 28 -> 54 / 66 |
| randlen_test lyapunov | 0.252 (1244) | 0.167 (547) | 0.136 (364) | 0.118 (275) | 0.108 (230) | **1.55x** | 0.128 | 44 -> 52 / 66 |
| L20mm mass_in_region | nan | 0.217 (929) | 0.162 (517) | 0.117 (269) | 0.089 (158) | 2.44x | nan | 42 -> 44 / 66 |
| L40mm mass_in_region | 0.153 (460) | 0.105 (219) | 0.084 (140) | 0.063 (80) | 0.048 (47) | 2.19x | 0.006 | 17 -> 48 / 66 |
| randlen_test mass_in_region | 0.315 (1947) | 0.190 (708) | 0.138 (379) | 0.104 (214) | 0.080 (128) | 2.37x | 0.015 | 45 -> 52 / 66 |

Independent replicates would give sqrt(8) = 2.83x. On L20mm/L40mm the goals
are close to that. On randlen_test (mixed n20/n50, piled/scattered/mixed
spawns) a slate-level component remains (ICC 0.13): models differ in WHICH
STATES they handle, which only more states can average out -- so goals are a
substitute for states on homogeneous corpora and only a partial one on a
heterogeneous corpus. Random goals are easier than the harness's (mean
lyapunov capture 0.92 / 0.97 / 0.83 vs 0.85-0.94 for the harness goals), so
absolute sds here are not comparable to A2's; the shrink factor is. 19% of
L20mm mass_in_region (slate, goal) cells are NaN (no candidate changes the
mass inside a small goal region, so capture is undefined).

## What would change the verdict

- **Seed noise (TODO H1).** Every margin above excludes training-seed
  variance; adding it can only widen the intervals, so C2/C3's "unresolved"
  readings are safe and the few "resolved" ones are optimistic.
- **Goal diversity in A4.** The 24 random goals are easier than the harness's
  (capture near the ceiling, a bounded metric), which shrinks absolute sds
  for reasons unrelated to replication. The shrink FACTOR (same goal family
  at G=3 vs G=24) is the reading that carries C4; a harder random-goal family
  could change it. Cost: minutes, CPU.
- **More states.** Required-n figures are extrapolations from 10-21 states
  of per-pair sd, themselves noisy at that n (sd of an sd at n=20 is ~16%).

## Threats

- `indirectness`: slateN on single-push pools is itself a proxy for
  closed-loop MPC performance; this record measures how precisely the proxy
  is estimated, not whether it predicts control (Phase D's question).
- The per-pair sd is summarised by its median; the noisiest pairs (max sd in
  the json) need more states than the table's figure.
- GNN predictions pass through eval_report's usual path; the GNN node-count
  bottleneck applies only to accuracy, not to slateN, as before.

## Unrelated findings

- `Baselines/common/runs/cross_corpus_report.json` is stale (predates the
  goal-mask transpose fix, 28271c09): its nfd_randlen slateN differs from a
  current run by up to 0.063. Anything that reads it as a reference is wrong;
  the handoff's table is current.
- `scripts/run_probe.py` still writes its ledger to `runs/COMMANDS.jsonl`
  (invariant `run-probe-ledger-path-matches-doc`, already `broken`); this
  record's entries were copied into `experiments/COMMANDS.jsonl` by hand.
- `eval_report` still predicts on CPU (`eval-baseline-scorer-batch-on-requested-device`,
  already `broken`): 12 models x 3 corpora took ~25 min, which matters once
  benchmarks scale to hundreds of slates.
- Building DS-0004 surfaced a real `torch.save` trap: saving a row VIEW of a
  large tensor writes the whole underlying storage (a 20-row bank was 225 MB
  instead of 12 MB). Fixed in `gt_bank.py` with `.clone()` and a test.
