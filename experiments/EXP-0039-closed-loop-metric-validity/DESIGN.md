# EXP-0039 — which offline metrics predict closed-loop MPC performance? (TODO G1b, user goal 1)

Written 2026-09-24 BEFORE running (plan gate). Runs after EXP-0037 stage 2 and the
EXP-0036 seed trainings (GPU queue).

## Design
- Model population (12): nfd_3ch_randlen, nfd_warped_randlen, nfd_warped_randlen_flipaug,
  nfd_warped_randlen_flipaug_epoch30, nfd_residual_warped_flipaug_randlen,
  nfd_residual_worldframe_noaug_ep43, nfd_3ch_finetuned, linear_switched_soft,
  ensemble_nfd, and the 3 new seeds of nfd_3ch_randlen (EXP-0036).
- Planners: rank (budget-matched: resamples candidates until the budget is spent),
  gd (8 restarts), cem. Budget 1.0 s per decision (one level; a second level only if
  time allows). Execution: GenesisOracleEnv.step with TRAINING_PHYSICS.
- Episodes: 8 pushes; 16 (goal, start) combos per (model, planner) = 4 goals
  (quadrant_0, letter_O, letter_T, letter_S) x 4 DS-0006 starts (states 40-43),
  identical across models (paired). 12 x 3 x 16 = 576 episodes (~9 h).
- Closed-loop score: improvement V0 - V8 (soft lyapunov), per episode.
- Offline metrics per model (EXP-0038 table + EXP-0037 where available): slateN,
  top1_regret, pool spearman, optimism, top8_capture (DS-0006); accuracy
  (randlen_test harness); grad_capture / gain_capture (EXP-0037, 7 arms);
  model speed (evaluations per second, measured in the episodes themselves).

## Hypotheses (predictions)
H1 slateN predicts closed-loop improvement under the RANK planner: Spearman across
   the 12 models >= 0.6.
H2 under CEM, whole-pool spearman predicts closed-loop improvement better than
   slateN (higher Spearman across models), and adding model speed improves it further.
H3 under GD, grad_capture (EXP-0037) predicts closed-loop improvement better than
   slateN (on the 7 arms it covers).
H4 accuracy predicts closed-loop improvement WITHIN the NFD family no better than
   chance (|Spearman| < 0.3), for every planner -- a null expected to hold.
H5 the ensemble has the best closed-loop improvement under rank and GD (it wins
   offline); under CEM its 5x evaluation cost may cancel that.
Correlations are reported with bootstrap-over-models CIs; with 12 models every
correlation is imprecise, so the record reports effect sizes, not just p-values.

## Addendum (before running): which ensemble
`ensemble_nfd` here = the mean predicted dv of EXP-0037's FIVE NFD arms
(nfd_3ch_randlen, nfd_warped_randlen, nfd_warped_randlen_flipaug,
nfd_residual_warped_flipaug_randlen, nfd_residual_worldframe_noaug_ep43), so its
gradient metrics (EXP-0037) and its closed-loop score describe the same object.
Its offline slateN-family metrics are added to EXP-0038's table as `ensemble_nfd5`.
(EXP-0030/0035/0038's `ensemble_nfd` averaged the 7 'nfd*' models incl. epoch30 and
the weak nfd_3ch_finetuned.)

## Addendum 2 (2026-09-24, before EXP-0039 started): a new hypothesis from EXP-0041
EXP-0041 (exploratory, n=5 NFD models) found image `accuracy` (randlen_test) orders
the arms exactly as EXP-0037's gain_capture does (Spearman +1.00) while slateN does
not (+0.43). Pre-registered here, before any closed-loop data exists:
H6 under the GD planner, accuracy predicts closed-loop improvement across the NFD
   models at least as well as slateN does (Spearman(accuracy) >= Spearman(slateN));
   under the rank planner the reverse (slateN >= accuracy).
H4 (accuracy uninformative within the NFD family for EVERY planner) is kept as
   written; H4 and H6 cannot both hold for GD -- the data will decide.

## Addendum 3 (2026-09-24, before EXP-0039 started): population cut to 4 models, ~3 h
User decision: the ensemble and the warped NFD variants are not worth closed-loop
testing at this stage; cap the run at ~3-4 h. The population is now 4 models:
- nfd_3ch_randlen (the original NFD);
- nfd_3ch_randlen_seed1 (same recipe, different seed, EXP-0036) -- the pair gives a
  closed-loop SEED noise floor that every other model gap is read against;
- linear_switched_soft (the fast model; tests speed under the budget);
- nfd_residual_worldframe_noaug_ep43 -- chosen as the model most likely to behave
  differently: it ranks 6th offline but is the 2nd-best gradient source (EXP-0037),
  so it is the one case where slateN and grad_capture disagree most.
Everything else (3 planners, 1.0 s budget, 4 goals x 4 starts, 8 pushes) is unchanged:
4 x 3 x 16 = 192 episodes, ~2.7 h at the pilot's 51 s/episode.

Consequences for the hypotheses, stated before any data:
- H5 (ensemble) is dropped here. The ensemble's time-limited question moves offline,
  to a budget-matched slateN ranking test recorded in EXP-0030 (addendum there).
- With 4 models an across-model Spearman cannot reach p < 0.05 (n=4: the smallest
  two-sided p is 1/12). H1-H4 and H6 are therefore reported DESCRIPTIVELY (the
  model orderings side by side), not as tests. H3 has 3 models with grad_capture
  (seed1 was not in EXP-0037), H4/H6 have 3 NFD models.
- The primary analysis becomes PAIRED: for each planner, all 6 model pairs'
  closed-loop differences over the 16 (goal, start) episodes (sign-flip p, Holm),
  and for each pair whether the offline metrics (slateN, grad_capture, accuracy)
  order it the same way. A model gap is called real only if it is significant AND
  larger than the seed pair's gap under the same planner.
