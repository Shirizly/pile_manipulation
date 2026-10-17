# RUN-0002 — legal, headroom, cross-family closed-loop metric validity (pre-registered; design in DESIGN.md)

- Pre-registration committed before any cell ran: 6adbad48 / 8836dc25 (2026-10-04 06:31). Run commit cc654308 (dirty: the
  EXP-0043 driver's `--legalize` / `--record-states` path was in the working tree; committed with the results in 81f90250).
- Queue: `code/run0002_queue.sh full` (one `scripts/run_probe.py` unit per (arm, model); provenance per unit in
  `results/run0002/rp_hm_{cem,rank}_<model>.{json,log}`). Pilot stage (1 model, 2 goals, 4 pushes) ran first, 2026-10-04 ~08:20-08:29
  (`results/run0002/pilot_{cem,rank}.json`); it confirmed `legal_ok`, 20 mm pushes and n_evals = 128 for the rank arm, and
  revealed no outcome used in the analysis.
- Full stage 2026-10-04 08:30-12:36 CEST (`results/run0002/full.done`); 16/16 units, every episode complete.
  Arm A: CEM pop 1024 / elite 0.25, 0.5 s, starts DS-0006 40-55, 8 letters, 16 pushes -> 128 episodes per model.
  Arm B: rank-128 at `--budget 0.0`, starts 40-47 -> 64 episodes per model. Both `--legalize --record-states`.
- Offline DS-0016 scores for the 4 zoo members EXP-0059 had not scored: `code/offline_ds0016.py` ->
  `results/run0002/offline_ds0016_scores.json` (the other 4 come from EXP-0059 `results/test_v2.json`).
- Analysis (pre-registered quantities): `CUDA_VISIBLE_DEVICES= python -u code/analyse_run0002.py` ->
  `results/run0002_analysis.{json,md}`. Results and the verdict against P1-P3 are in EXPERIMENT.md.

## Addendum (2026-10-04, extended analysis; zero simulation; EXPLORATORY unless marked)

`CUDA_VISIBLE_DEVICES= python -u code/analyse_run0002_extended.py` -> `results/run0002_extended.{json,md}` (full tables
there; recomputed closed-loop means reproduce `run0002_analysis.json` exactly). What it adds, and what it found:

1. **Metrics the caches already held but the pre-registration did not list** (accuracy_1_blur1/2, _scatter, _clump,
   rollout_accuracy_1-4, occ_emd_swept, mass_in_goal_mae[_tough]). Spearman rho with the closed-loop score, exact
   one-sided permutation p (8! relabelings), CEM / rank:
   mass_in_goal_mae +0.79 (p 0.014) / +0.48 (0.12); accuracy_1_blur2 +0.76 (0.018) / +0.29 (0.25); mass_in_goal_mae_tough
   +0.74 (0.023) / +0.45; occ_emd_swept +0.60 (0.066) / +0.45; accuracy_1_blur1 +0.52 / +0.10; rollout_accuracy_3 -0.19 / -0.17;
   accuracy_1_scatter -0.05 / +0.36; accuracy_1_clump +0.40 / -0.14.
   Pre-registered, same statistics: slateN_tough +0.60 (0.066) / +0.57 (0.076); top-1 regret +0.50 (0.108) / +0.81 (0.011);
   within-pool Spearman +0.14 / +0.52 (0.098); slateN +0.31 / +0.24; optimism +0.14 / +0.26; accuracy_1 +0.10 (0.42) / +0.17 (0.35).
   **Multiplicity:** Holm over the 6 pre-registered metrics within an arm leaves nothing below 0.05 (closest: top-1 regret
   under rank, Holm p 0.065; slateN_tough under CEM 0.40). Holm over all 25 rows per arm: nothing (best 0.09, on-policy corr
   under rank). The reported episode-bootstrap CIs resample episodes only and say nothing about the n = 8 model sample.
2. **Within-seed pairs.** 2 of the 22 CEM and 3 of the 14 rank Holm-resolved pairs are between narrow v2 seeds 0/1/2
   (closed-loop seed spread 0.053 in both arms). Excluding them: CEM (20 pairs) mass_in_goal_mae 0.90, accuracy_1_blur2 0.90,
   on-policy MAE 0.90, slateN_tough 0.70, top-1 regret 0.65, accuracy_1 0.55; rank (11 pairs) on-policy corr 1.00, top-1 regret
   0.91, slateN_tough 0.91, mass_in_goal_mae 0.73, accuracy_1 0.64. The exploratory image metrics' CEM advantage does NOT
   disappear when within-seed pairs are excluded.
3. **slateN_tough's goal set vs the closed-loop letters.** GOALS_TOUGH (EXP-0059) = O T S X L I two_squares quadrant_0;
   overlap with the 8 closed-loop letters O T S L X Z C H is 5/8. Rescored on exactly those 5 letters from the cached
   per-pool (vt, vp) (`*_matched5`): slateN +0.50 / +0.50, top-1 regret +0.50 / +0.81, within-pool Spearman +0.17 / +0.48,
   optimism +0.14 / +0.26 -- the same picture as the 8-goal tough set. Z, C and H were never scored offline, so an exact
   8-letter rescore needs a GPU pass (not done here).
4. **On-policy prediction quality from the recorded pred_dv / true_dv of every executed push** (never analysed before).
   Per-model corr(pred, true) over 2048 (CEM) / 1024 (rank) pushes: under CEM +0.61 nfd_3ch_randlen, +0.65 linear_switched_soft,
   and -0.13..+0.00 for EVERY narrow NFD and the narrow linear; under rank +0.23..+0.52 for all. Rank correlation with the
   closed-loop order: on-policy MAE +0.79 (p 0.014) under CEM; on-policy corr +0.88 (p 0.004) under rank, agreeing with
   14/14 resolved pairs. Reading (untested): CEM-1024 drives the narrow models to pushes where their predictions are
   uncorrelated with the outcome (planner exploitation); the matched-evaluation rank planner does not.
5. **Executed un-legalisable pushes** (`legal_ok == False`: `legalize_pushes` found no legal offset and the driver executed
   the push unchanged, `simple_mpc/learned_mpc.py::run_episodes_batched`): CEM 350 / 16384 = 2.1 % (per model 0.7-2.7 %,
   soft_s2 5.9 %); rank 0 / 8192. Order robustness: dropping each model's own affected episodes, Kendall tau vs the full
   order +1.00 (CEM); dropping an episode for all models if any model was affected keeps only 16 / 128 episodes, tau +0.79.
   Logged as OPEN_ISSUES ISS-014; fix under EXP-0066 (another agent).
6. **Per goal.** Per-letter model orders vs the pooled order: tau +0.57..+0.93 except letter_T (+0.36 CEM, +0.79 rank);
   letter_T is also the only letter where cached per-goal slateN anti-correlates with per-goal closed loop (-0.55 CEM).
   The pooled result is not carried by one letter.
