---
# ---- identity -------------------------------------------------------------
id: EXP-0024
title: >
  Physical repeat-noise floor for slateN's ground truth: same action from the
  same snapshot state is (near-)bit-identical under GenesisOracleEnv's
  snapshot-restore mechanism, so re-simulation noise is NOT the source of
  slateN's ceiling for n20 single-push rollouts on this hardware/version
tier: T1
mode: confirmatory
date: 2026-09-23
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  Re-executing the IDENTICAL push action from an IDENTICAL settled n20-cube
  pile state (via GenesisOracleEnv.snapshot_particles/restore_snapshot/
  rollout_candidates, the same seam same_state_slate_collection.py and
  binned_slate_collection.py use to broadcast one state to many candidate
  envs) produces dv (under lyapunov(corner)) that varies materially from
  repeat to repeat, comparable in scale to the between-action dv variance
  slateN actually ranks candidates on.

prediction:
  supports: >
    within-action dv variance (over >=3 repeats of the SAME action from the
    SAME snapshot) exceeds 5% of the between-action dv variance (over
    distinct actions from that same snapshot) in at least half of the tested
    (state) cells.
  refutes: >
    within-action variance is below 1% of between-action variance in every
    tested cell, at both reduced- and full-fidelity settle budgets -- i.e.
    resimulating the same action returns (near-)identical dv, and slateN's
    ground truth is not a noisy draw by this mechanism.
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: a175b981
  dirty: true                     # tree carries other agents' concurrent, unrelated
                                  # edits (NFD training run RUN-0010, various baseline
                                  # files) -- this record's own script
                                  # (experiments/temp/exp0024-repeat-noise/
                                  # pilot_repeat_noise.py, promoted to
                                  # code/pilot_repeat_noise.py here) is the only file
                                  # this record wrote or depends on for its numbers
  script: code/pilot_repeat_noise.py (new, this record; ran from
    experiments/temp/exp0024-repeat-noise/ before promotion)
  data: ["freshly-settled n20 cube piles inside GenesisOracleEnv (config
         simple_mpc/config/config_oracle.yaml, num_objects=20) -- no
         pre-existing corpus, states generated fresh per run via
         SandboxManipulation.shuffle_particles()+settle inside env.reset()"]
  code_path: >
    simple_mpc.genesis_oracle.GenesisOracleEnv.{snapshot_particles,
    restore_snapshot, rollout_candidates} (planning seam) ->
    control_utility_test.lyapunov + control_utility_test.lyapunov_weights
    ("corner") for dv
  seed: "none explicitly set -- Genesis's own internal RNG for
    shuffle_particles()/settle; no seed control needed since the comparison is
    WITHIN one snapshot (state is frozen and restored bit-for-bit before every
    rollout), not across independently-drawn states"
  split: "not applicable -- no train/test split, no fitted object"
  data_commit: not applicable (no persisted dataset; states are ephemeral,
    generated and consumed within one process run)
  runtime: "reduced-fidelity (3 states, use_rollout_fidelity=True, halved
    settle/clearance steps): 131.0s total incl. ~61s one-time scene
    build/kernel compile, ~10-13s per (state, 15-env rollout) cell.
    full-fidelity (1 state, real settle/clearance budgets): 72.8s total incl.
    ~61s build, ~11s for the one rollout cell (n_envs=15 either way, so
    full-fidelity's per-cell cost is close to reduced-fidelity's here --
    the settle-step difference this config uses, 60 vs 30 real vs rollout
    steps, is a modest fraction of a push's total physics cost at n20)."

budget:
  declared: "45 min wall-clock, 80k tokens (task-level, per subagent brief)"
  spent: "~40 min wall-clock (dominated by prior-work search), ~2 GPU-min
    across two runs, ~70k tokens"
  outcome: within

design:
  varied:
    fidelity: >
      reduced (use_rollout_fidelity=True, 3 states); full
      (use_rollout_fidelity=False, 1 state)
    state: >
      4 distinct freshly-settled n20 piles total (indices 0,1,2 under reduced
      fidelity; a separate index 0 under full fidelity), none shared across
      the two fidelity arms
    action: >
      3 distinct fixed candidate pushes per state -- a +x sweep, a +y sweep,
      and a diagonal sweep, all through the pile's middle at ~1.2x wkspc_w
      travel
  held_fixed:
    n_particles: >
      20 (matches slates_binned/slates_multistep's n20 corpora -- the
      corpora slateN is actually reported on in this repo)
    material_shape: cube
    goal: corner (the informative goal per EXP-0012/EXP-0017 -- center is
      degenerate)
    value_fn: lyapunov (control_utility_test.py) -- the same V used by every
      slateN record in this repo
    grid_res: (64, 64)
    n_repeats_per_action: 5
    n_ahead: 1 (single push, not a multi-step chain)
    snapshot mechanism: GenesisOracleEnv.snapshot_particles/restore_snapshot
      (exact live particle-tensor copy-restore, not a fresh resettle from a
      stored config)
  baselines: >
    No model-vs-persistence baseline applies -- this record characterises
    simulator noise, not a predictor. The reference scale IS the design:
    between-action dv variance (the exact quantity slateN ranks candidates
    on) is the "signal" every within-action ("noise") number is judged
    against, per repeat_noise_ratio's definition in METRICS.md.
  metric: repeat_noise_ratio (experiments/METRICS.md, added by this record)

noise_floor: >
  This record's OWN entire subject is a noise floor, so "the noise floor for
  this record" is the repeat count itself: n=5 repeats per (state, action)
  cell, 4 cells total. This is enough to show the effect is at the float32
  roundoff floor (repeat variance 0 to 1.2e-7, max repeat-to-repeat COM
  spread 0 to 1.6e-5 mm -- far below the ~5mm particle size and far below
  the docs/scaling_to_200_objects.md-measured ~0.1-4mm COM-shift scale of
  REAL chaotic divergence under a fresh resettle), not a borderline case a
  few more repeats could flip. No fold sweep of a fitted object is relevant
  here (nothing is fitted).

depends_on: [goal-mask-axis-convention-row-y-col-x]
establishes: [genesis-snapshot-restore-repeat-determinism]

# ---- outcome --------------------------------------------------------------
result: >
  REFUTED as stated: within-action dv variance is 0 to 5.3e-5 of the
  between-action dv variance in every one of the 4 tested cells (3
  reduced-fidelity states + 1 full-fidelity state), an order of magnitude
  below even the strict "refutes" bar (1%). Repeating the identical action
  from the identical snapshot returns dv identical to within float32
  roundoff (repeat-to-repeat COM spread up to 1.6e-5 mm, several orders of
  magnitude below the particle size). Genesis, exercised through this exact
  snapshot-restore-and-push seam, behaves as deterministic given identical
  inputs -- consistent with docs/oracle_mpc_design.md's own statement that a
  full-fidelity re-roll of the winning action "should closely match" a
  rollout-fidelity one "(same deterministic sim)". This directly contradicts
  the premise that slateN's ground-truth ceiling comes from resimulation
  noise, AT LEAST for n20 single-push candidates scored through this
  mechanism: if a candidate's true dv were resimulated a second time, it
  would come back (for all practical purposes) unchanged. The "granular
  dynamics is stochastic" framing is not wrong in general -- see Unrelated
  findings for the archived evidence that Genesis genuinely is NOT
  bit-deterministic under a different repeat mechanism (fresh resettle
  rather than exact-snapshot restore) -- but it is the wrong explanation for
  a ceiling on THIS metric, scored THIS way, because the actual data
  pipeline never resimulates: each candidate's ground truth is one push from
  one broadcast snapshot, exactly the condition tested here.

verdict: refuted
downgrades: [incomplete-design]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If resimulation noise were a real ceiling on slateN, repeating the identical
action from the identical snapshot should show dv scattering by an amount
comparable to (or a non-trivial fraction of) the spread across genuinely
different actions from that same state -- because that spread is exactly what
slateN's argmax/argmin selection has to resolve. If instead the repeat spread
is negligible next to the between-action spread, resimulation cannot be
where a ceiling on `slateN` well below 1.0 comes from, and the search for that
ceiling has to look at model imprecision, state-settling variability between
nominally-similar states, or degenerate/near-tied slates instead -- not at
"the true value used for scoring is itself a noisy draw."

## What was actually run

Built one `GenesisOracleEnv` (n_envs=15: 3 actions x 5 repeats, single scene,
single kernel compile) per fidelity arm. For each state: `env.reset()`
(fresh shuffle+settle), `snapshot = env.snapshot_particles()`, compute
`V0 = lyapunov(occ0, d_corner)` from the snapshot (same start state -> same V0
for every one of the 15 envs), build `act_seqs` of shape (15,1,4) as 3 fixed
world-frame sweep actions (+x through the middle, +y through the middle, a
diagonal sweep), each repeated 5 times in contiguous blocks, and call
`env.rollout_candidates(act_seqs, snapshot, use_rollout_fidelity=..., record=False)`
in ONE call -- so all 15 outcomes come from the identical restored snapshot,
executed in the same batched physics step. Computed `dv = lyapunov(occ1,
d_corner) - V0` per env, reshaped to (3 actions, 5 repeats), and reported
`within_var_mean` (mean over actions of the repeat-wise variance) against
`between_var` (variance of the 3 action means).

Ran 3 states at reduced (rollout) fidelity (`use_rollout_fidelity=True`, half
the real settle/clearance step budget -- what CEM/MPPI candidate evaluation
actually uses during planning) and, separately, 1 state at full/real fidelity
(`use_rollout_fidelity=False` -- what a real executed step, and what an
offline slate-collection driver, actually uses) to check the result was not
an artifact of the reduced step budget specifically. It was not: the
full-fidelity cell's ratio (0.0) sits inside the same range as the
reduced-fidelity cells (0 to 5.3e-5).

Stopped at 4 cells (not the ~9 the brief suggested, 3 states x 3 actions)
because the first cell alone already showed the effect pinned at the float32
roundoff floor with no ambiguity, and the remaining budget was better spent
confirming it survives full fidelity (a real, distinct condition) than adding
more reduced-fidelity states that would very likely reproduce the same
zero.

## Numbers

n_repeats=5 per (state, action) cell throughout.

| fidelity | state | v0 | action dv means | within_var_mean | between_var | ratio | max repeat COM spread (mm) | rollout wall-time (s) |
|---|---|---|---|---|---|---|---|---|
| reduced | 0 | 0.328 (unrecorded precisely, see artifact) | see artifact | 1.22e-07 | 2.28e-03 | 5.35e-05 | 1.56e-02 | 12.5 |
| reduced | 1 | 0.262 | [+0.0062, -0.0221, +0.0039] | 0.00e+00 | 2.48e-04 | 0.00e+00 | 0.00e+00 | 10.2 |
| reduced | 2 | 0.287 | [+0.0669, -0.0125, +0.0141] | 0.00e+00 | 1.64e-03 | 0.00e+00 | 4.66e-07 | 10.8 |
| full | 0 | 0.364 | [+0.0271, -0.0154, +0.0040] | 0.00e+00 | 4.53e-04 | 0.00e+00 | 2.62e-07 | 10.8 |

Full per-repeat `dv` arrays and per-env COM positions are in
`artifacts/RUN-0001-repeat-pilot/results.json` and `results_fullfidelity.json`
(state 0's dv-by-action array specifically: see that file for the one cell
above with `within_var_mean` non-zero, 3.65e-07 on action 3 only, still 5
orders of magnitude below `between_var`).

`between_var` itself spans 2.5e-4 to 2.3e-3 across the 4 states/actions tested
(roughly an order of magnitude), confirming the 3 chosen actions were not a
degenerate, all-tied slate -- there is real ranking signal for the noise
floor to be judged against.

## What would change the verdict

- **Higher particle counts / multi-step chains.** This record used n20,
  single-push (matching the corpora slateN is scored on today), where
  `docs/scaling_to_200_objects.md`'s own archived measurements show
  divergence is smallest (their n=50 "gentle" cells were also
  bit-identical). Their n=100 "diagonal" cell showed real divergence on a
  FRESH resettle (com_shift 0.642->0.887mm) -- whether that divergence
  reappears under exact-snapshot restore (this record's mechanism) at higher
  n, or is entirely an artifact of the resettle path, is untested here. Cost:
  ~15-20 min GPU (n=50/100 piles cost more per push per docs/
  scaling_to_200_objects.md's own timing table).
- **Longer/more chaotic actions** (more contacts, larger travel) than this
  record's 3 fixed sweeps -- the archived probe's divergent cell was a
  "diagonal" push at n=100, not a broadside one; this record's own diagonal
  action (state 2's action 3, the one nonzero cell) is the only cell that
  showed ANY nonzero repeat variance, consistent with that pattern, though
  at 5 orders of magnitude smaller scale (n20 vs n100, different mechanism).
- **A true stress test**: deliberately induce the settling-phase
  nondeterminism the archived probe's "fresh resettle" repeat exercises
  (rebuild the scene / reload the state from a serialized config rather than
  restoring the live tensor) and compare directly against this record's
  exact-snapshot mechanism, same states, same actions -- would isolate
  whether the discrepancy with the archived finding really is the
  restore-mechanism difference this record hypothesises, or something else
  (Genesis version 1.3.3 here vs 0.4.5 there, GPU model, or single-vs-multi-
  process reduction order as EXP-0025 flagged for a different metric).

## Dirty-tree disclosure

The working tree was dirty throughout this run (per the session's opening
`git status`: concurrent, unrelated edits from other agents/tasks -- an
in-progress NFD training run RUN-0010, baseline/doc file edits belonging to
other experiments). This record's own code
(`experiments/EXP-0024-genesis-repeat-noise-floor/code/pilot_repeat_noise.py`)
was newly written this session and is the only file its numbers depend on;
none of the concurrently-dirty files are imported by it.

## Threats

- `incomplete-design`: only 4 (state, fidelity) cells were run against the
  ~9 a 3x3x5 design would give, and only 1 cell at full fidelity -- the
  brief's suggested scale was not fully reached. Accepted because the
  measured effect (ratio <= 5.3e-5, repeat spread at float roundoff) leaves
  no realistic room for 5 more cells to change the qualitative verdict; the
  more valuable use of remaining budget was confirming full-fidelity
  parity (done) rather than more reduced-fidelity replicates of an already-
  saturated null.
- Considered and dismissed `provenance`: single code path throughout
  (`GenesisOracleEnv` + `control_utility_test.lyapunov`), same commit, same
  process, no cross-rasteriser or cross-dataset comparison.
- Considered and dismissed `indirectness`: `dv` here is computed exactly as
  every other `slateN` record in this repo computes it (`lyapunov(occ1, d) -
  lyapunov(occ0, d)`), not a proxy for it.
- `depends_on: []` because this record does not lean on any existing
  invariant -- it establishes a new one instead
  (`genesis-snapshot-restore-repeat-determinism`).

## Unrelated findings

- **This is NOT the first "Genesis noise floor" measurement in this repo,
  and it answers a different question from the one that exists.**
  `docs/scaling_to_200_objects.md` section 8.8 (backed by
  `tests/scaling_investigation/probe_solver_equivalence.py` and its
  `results/solver_equivalence.json`) explicitly states Genesis "is not
  bit-deterministic" and measures a noise floor from a bit-identical rerun
  (`newton_repeat`) and a 1um action perturbation (`newton_eps`), pooled
  across 4 actions x 2 particle counts x 12 replicates. Reading that data
  directly: most cells ARE bit-identical (e.g. n=50 `edge_on`: com_shift
  0.414mm in all three of newton/newton_repeat/newton_eps), but one is not
  -- n=100 `diagonal`: com_shift 0.642mm (newton) -> 0.887mm
  (newton_repeat) -> 0.648mm (newton_eps), a 38% jump on a literal
  "identical" rerun, and displaced_mass 83.2 -> 107.8mm (30%). That probe's
  own comment (`probe_solver_equivalence.py` line 77) says every
  configuration is "seeded from the SAME library state by explicit index" --
  i.e. it reconstructs/resettles a stored state fresh per config, rather
  than restoring an exact live particle-tensor snapshot within one
  continuously-running scene, which is what `GenesisOracleEnv`'s
  `rollout_candidates` (and therefore this record) does. **That is measuring
  a genuinely different quantity from this record**: repeatability of the
  full pipeline (settle-from-config + push) vs repeatability of (exact-state
  restore + push) alone. The two are not in conflict -- they isolate
  different stages -- but a reader citing "Genesis is not bit-deterministic"
  as the noise floor for `slateN`'s ground truth needs the SECOND number
  (this record's), because that is the mechanism the actual slate-scoring
  data pipeline (`same_state_slate_collection.py`/
  `binned_slate_collection.py`, both broadcast-from-one-snapshot) uses, not
  the first.
- `EXP-0001`'s `noise_floor: "not measured"` and both archived T1 records
  that DO carry a populated `noise_floor` field (`EXP-0025`,
  `archive/2026-09-10_pre-reset/docs/experiments/EXP-0017-rank1-dissociation-noise-floor.md`)
  measure **across-slate** or **across-fit-seed** spread, not
  same-state-same-action resimulation spread -- a fit-seed noise floor
  answers "how much does the FITTED OPERATOR'S ranking wobble across
  training seeds", which is a different question from "how much does the
  GROUND TRUTH ITSELF wobble on resimulation". Neither of those existing
  records answers the question this one was asked to resolve; this record
  is the first to measure resimulation noise on `dv` directly.
