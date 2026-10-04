# Design: closed-loop metric validity with headroom, across families, legal actions (SUMMARY §11 hole 3)

Advisor note, 2026-10-04 (read-only, nothing run). Fills §10's gap: no offline metric has been validated against closed loop on a task with headroom, across families, with legal actions.

## 1. Question and pre-registered prediction
H-metric: which offline metric, computed on ONE clean legal corpus, orders a cross-family zoo the way
closed-loop task performance does? Candidates: accuracy_1, slateN, slateN_tough, optimism-at-pick,
top-1 regret, within-pool Spearman. (grad_capture dropped: it needs EXP-0037's re-simulation harness and
is undefined for retrieval; optional follow-up for the differentiable members only.)
- **P1** slateN_tough (legal DS-0016) has the highest Spearman with closed-loop score, rho >= 0.7, and agrees
  with >= 80 % of Holm-resolved closed-loop pairs. **P2** accuracy_1 agrees with < 70 % (EXP-0054 inversion,
  EXP-0065 RUN-0005). **P3** optimism is negatively correlated (rho <= -0.5): the planner exploits optimistic
  models. **Refutes** P1 if no metric reaches rho >= 0.5 / 70 % pairs: offline selection unsupported here.
- Uninformative outcome guard: the run must Holm-resolve >= 8 of 28 pairs (the 3 seeds give the floor;
  the weak model and the cross-family spread are there to guarantee resolved pairs exist).

## 2. Model zoo (8 in the main arm, all existing `OCC_ADAPTERS` entries; checkpoints verified on disk)
| role | OCC_ADAPTERS name | ckpt |
|---|---|---|
| narrow NFD seeds 0/1/2 (seed floor) | `nfd_3ch_narrow_l20_v2`, `_seed1`, `_seed2` | Baselines/NFD/runs/.../unet_best.pth |
| soft NFD sigma 2 (EXP-0063 leader) | `nfd_3ch_narrow_l20_v2_soft_s2` | present (`encode_state` blur handled by ModelObjective) |
| linear switched, narrow clean | `linear_narrow_l20_v2_res64` | Baselines/LinearForesight/runs/operator_narrow_l20_v2_res64.pt |
| linear switched, broad (EXP-0044 family) | `linear_switched_soft` | weights/MODEL-0001 |
| broad NFD (EXP-0044 family) | `nfd_3ch_randlen` | present |
| deliberately weak | `nfd_3ch_narrow_l20_v2_epoch10` | unet_epoch_10.pth present |
Rank-arm only (section 4): **retrieval k5 v2** (`RetrievalPredictor(bank_train_v2_curated, _CAPPED, k=5,
cube_median)`, EXP-0059 `test_v2.py`). It has **no OCC adapter** (particle-in/out by design, CODEMAP) and
is non-differentiable; 26 ms/candidate (benchmark_timing.json) so it cannot run CEM-1024 at 1 s.
Minimal adapter (new, ~40 lines + 1 hook): `RetrievalOccAdapter.predict_step(occ0, acts)` ignores `occ0`,
uses particles set by `set_particles(states)` -> `predict_particles(states.expand(K), p_start, p_stop)` ->
`occ_from_particles`; hook in `ModelObjective.__init__`: `if hasattr(m,"set_particles"): m.set_particles(particles)`.
Register as `retrieval_k5_v2` in `OCC_ADAPTERS`. The 3-seed ensemble is also NOT runnable via `--models`
(driver maps one name -> one adapter); a registry entry returning a list would need `make_occ_adapter` to
accept lists (skip; ensemble is an EXP-0059 "tighten" item).

## 3. Task
- Starts: DS-0006 states 40-55 (16 n20 scatter starts; 40-47 = all prior closed loop). Clump starts: not
  plumbed (`--corpus` must be a BinnedSlateCorpus; EXP-0049's clump constructor is not) -> defer to hole 5.
- Goals: 8 letters O T S L X Z C H (in-goal mass 0.55 at k8 vs optimum 0.92-0.97, EXP-0051: headroom).
  No quadrants (saturate by k5). Optional 1 quadrant sanity cell if budget remains.
- Actions: `push_len 0.02` (narrow models and retrieval are 20 mm-only; broad models cover 20-70). Legal:
  `--legalize` (translates, keeps length). 16 pushes. Objective: plain lyapunov (the SCORE is in-goal mass,
  so the objective is not the score, unlike 0039-0045).
- Score per episode: **AUC of in-goal mass fraction / per-goal optimum (EXP-0046) over pushes 4..16**
  (primary); in-goal mass at k=8, k=16 and completion at 0.8x optimum (secondary). Needs `--record-states`;
  EXP-0051 `code/analyse.py` already computes mass_in_region from recorded states.

## 4. Planner arms
- **A (main): CEM** tuned as EXP-0044 (`cem_pop 1024, cem_elite_frac 0.25`), **budget 0.5 s** (EXP-0045:
  > 0.1 s buys nothing for CEM; halves wall time). 8 models x 8 goals x 16 starts x 16 pushes = 128 episodes/model.
- **B (matched evaluations, includes retrieval): rank-128**, `--n-cand 128 --budget 0.0` (time_left() is
  false at once, so every model scores exactly the same 128 pile-aware legal candidates, no code change).
  9 models x 8 goals x 8 starts (40-47) x 16 pushes. Retrieval: 128 x 26 ms = 3.4 s/step.
- GD omitted (tuned GD = tuned CEM in 0044; retrieval has no gradient).

## 5. Power (EXP-0044 components rescaled to in-goal mass)
EXP-0044: residual pair-difference sd 0.031/episode on lyapunov improvement -> 0.017 at 12x8 (96 eps, 2.8 SE).
In-goal mass is noisier: EXP-0054 letters, 20 mm, paired model-difference sd at k12 = **0.059-0.069** (28 eps).
Arm A, 128 paired episodes: SE 0.0057 -> **resolvable 0.016** at a single k; the 4..16 AUC averages 13
correlated readouts (expect ~0.012). Observed gaps to detect: EXP-0054 k12 0.01-0.04, k24 0.02-0.12; the
epoch-10 model and cross-family gaps should exceed 0.03. Arm B (64 eps): resolvable 0.023; rank-only, so
expect larger spreads (0054's lyapunov-per-push ratio of 2x between models).

## 6. Offline metrics, one corpus: DS-0016, legal by construction
`Genesis/data/narrow_l20_n20/test_pools_v2` (32 same-state pools x 64 pushes, fixed sampler, 0 % illegal) and
`test_chains_v2_clean` (986 rows), via EXP-0059 `code/test_v2.py --models <OCC names> --splits test`
(`exclude_flagged` drops any remaining flagged row). Already scored there: v2 seeds 0/1/2, linear v2 res32/64,
retrieval k5, persistence. To add: `soft_s2`, `nfd_3ch_randlen`, `linear_switched_soft`, `epoch10`
(~40 s GPU each). Broad models scored on the narrow pools is correct: it IS the closed-loop action space.
From `test_v2.json[model]["raw"]["slateN_tough_raw"][goal]` (per pool: `vt`, `vp`) compute, per (pool, goal)
then mean: slateN (13 goals) and slateN_tough (8), **within-pool Spearman(vp, vt)**, **optimism** =
sign*(vp[pick]-vt[pick]) at pick=argmax predicted, **top-1 regret** = max vt - vt[pick] (same formulas as
EXP-0065 `closed_loop_models_legal_rescore.py` lines 55-61). accuracy_1 from `metrics`. New ~40-line script.
Caveat to record: offline is step-0 spawn states; closed loop acts on half-organised piles (SUMMARY subtle holes).

## 7. Analysis
1. Per arm: 8 (9) model means of the AUC score with paired bootstrap CIs; all 28 (36) pairwise paired
   differences, Holm at 0.05 -> set of resolved pairs with sign.
2. Per metric: Spearman rho (and Kendall tau) between metric and closed-loop mean over models, CI by
   bootstrap over episodes (re-draw episodes, recompute closed-loop means, rho); n = 8 models, so report
   also **pair agreement** = fraction of resolved pairs the metric orders correctly (EXP-0044 convention).
3. Report the seed-triplet spread as the closed-loop floor; a metric that "resolves" within-seed pairs is
   over-reading. Repeat with arm B to see whether the planner changes the winner.
4. Also report per-model legal shift rate (`legal_shift_m` > 0) and executed push-length distribution:
   EXP-0054's wall clipping (pushes shortened to >= 7 mm by `project_push`'s box) must be counted per model.

## 8. Cost and checkpointing (RTX 4070 8 GB, nothing else on the GPU)
EXP-0044 rate 1.4 s/env-step at 1 s budget -> ~0.9 s at 0.5 s. Arm A: 8 x 128 x 16 = 16,384 env-steps ~ 4.1 h.
Arm B: NFD/linear 8 x 64 x 16 ~ 0.4 s -> 55 min; retrieval 64 x 16 x 3.8 s ~ 65 min. Offline: ~5 min.
**Total ~6.2 h GPU.** Units = one driver call per (arm, model); the driver checkpoints atomically after
every push (`.tmp` + `os.replace`) and resumes by tag, so a crash loses < 1 chunk-step. Run as a sequential
queue (`legal_rerun_queue.sh` pattern), `python -u`, `.pid` files, GPU exclusive.

## 9. Commands (EXP-0039 RUN-0010; `G="letter_O letter_T letter_S letter_L letter_X letter_Z letter_C letter_H"`)
```
# Arm A, one unit per model M
python -u experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py --tag hm_cem_$M \
  --results-dir experiments/EXP-0039-closed-loop-metric-validity/results --models $M --planners cem --goals $G \
  --starts $(seq 40 55) --steps 16 --budget 0.5 --n-cand 1024 \
  --cells '{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25, "push_len": 0.02}}' --legalize --record-states
# Arm B, one unit per model M (add retrieval_k5_v2 once registered)
python -u experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py --tag hm_rank_$M \
  --results-dir experiments/EXP-0039-closed-loop-metric-validity/results --models $M --planners rank --goals $G \
  --starts $(seq 40 47) --steps 16 --budget 0.0 --n-cand 128 \
  --cells '{"l20r": {"push_len": 0.02}}' --legalize --record-states
# Offline (DS-0016), missing zoo members
python -u experiments/EXP-0059-retrieval-transition-model/code/test_v2.py --splits test \
  --models nfd_3ch_narrow_l20_v2_soft_s2 nfd_3ch_randlen linear_switched_soft nfd_3ch_narrow_l20_v2_epoch10 \
  --out experiments/EXP-0039-closed-loop-metric-validity/results/offline_ds0016.json
```
Pilot first (1 chunk: 1 model, 2 goals, 16 starts, 4 pushes, ~5 min) to confirm `legal_ok`, push lengths
= 20 mm, and `n_evals` = 128 for every rank-arm model.

## 10. Code changes needed before it can run
- **Required only for retrieval in arm B:** `RetrievalOccAdapter` + `set_particles` hook in `ModelObjective`
  (section 2). Arms A and B for the 8 OCC models run with the driver as is.
- **Analysis (new, not driver):** ~40-line offline-metric script over `test_v2.json` raw; per-k in-goal mass
  and AUC from recorded states (adapt EXP-0051 `analyse.py`, which stops at k24 and lacks AUC).
- Noted: `--cells` values pass straight into `plan()` kwargs (`cem_pop`/`cem_elite_frac`/`push_len` work); `--budget 0.0` matched rank relies on `time_left()` (confirm in pilot). Ensemble, clump starts: out of scope.
