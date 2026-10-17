# EXP-0071 design (pre-registration, written before any scored closed-loop cell ran)

## Offline context (EXPLORATORY, already seen; not part of the pre-registered claim)
DS-0016 clean test, member predictions cached by experiments/temp/ensemble-proto: accuracy_1 best single 0.570 (NFD seed 2),
equal blend NFD x3 0.582, gate (5 occupancy members, MODEL-0015) 0.607 [adapter parity run, code/parity.py]; slateN_tough on
DS-0016 test_pools_v2: best single NFD 0.731, mean-blend NFD x3 0.822, retrieval-free gate 0.83-0.84 (3 gate seeds), mass_in_goal_mae_tough
1.36 vs 1.64 best NFD. (The gate seed used in closed loop is seed 0, fixed before the closed-loop cells; the gate was fit on DS-0017 only.)

## Claim (H1, closed-loop, rank arm)
In legal closed loop (letters O T S L X Z C H, DS-0006 starts 40-47, 16 pushes of 20 mm, rank planner on the same 128 pile-aware
legal candidates per decision, `--legalize`, score = AUC of in-goal mass / per-goal optimum over pushes 4..16, EXP-0039 RUN-0002 formula),
the gated ensemble `gated_ens_narrow_v2` scores above the MEAN of its three NFD members' cell scores (RUN-0002 `hm_rank_nfd_3ch_narrow_l20_v2{,_seed1,_seed2}`,
same 64 (goal, start) episodes), paired-over-episodes bootstrap 95 % CI excluding 0.
- supports: gate - NFD-3-mean >= +0.02 with CI excluding 0. refutes: CI includes 0. Embarrassing: the gate falls below its mean member.
- H2 (does the gate add over seed averaging): gate - `nfd3_meanblend_narrow_v2` CI excludes 0 (same arm). Prediction: not resolved at n = 64 (offline mean-blend already
  captures most of the slateN gain: 0.82 vs 0.84). Reported either way.
- H3 (CEM arm, EXPLORATORY, equal wall-clock 0.5 s, pop 1024/elite 0.25, starts 40-55, 128 episodes): gate - NFD-3-mean; predicted <= 0.01 because the
  5-member ensemble evaluates ~5x fewer candidates in the same 0.5 s (EXP-0035 C-043). n_evals/step is reported.
- Baselines: do-nothing = score of push 0 level (start in-goal mass; reported), the RUN-0002 NFD seeds 0/1/2, linear res64, the equal-weight blend control.
- Power: paired episode-difference sd ~0.11 (EXP-0066) -> SE ~0.014 at n = 64; MDG ~0.04 (seeds fixed); a +0.02 prediction is below MDG, so H1 can only be read if the effect is larger -- stated up front.
- Pilot (cost only, no outcome read): code pilot_rank 2 goals x 8 starts x 2 pushes, 52 s incl. startup.

## Cells (one GPU job at a time, driver EXP-0043 batched_closed_loop.py, flags exactly as RUN-0002 code/run0002_queue.sh)
1. rank: gated_ens_narrow_v2, nfd3_meanblend_narrow_v2   (64 episodes each)
2. cem : gated_ens_narrow_v2, nfd3_meanblend_narrow_v2   (128 episodes each)
Fixed: planner seed unset as in RUN-0002 (candidate sampling not replicated; RUN-0002 baselines are single cells), one gate seed, no tuning.
Caveat registered now: baselines come from a different session (RUN-0002, 2026-10-04); code differs only by `legalize_fallback` default off and unseeded planner, both unchanged here.
