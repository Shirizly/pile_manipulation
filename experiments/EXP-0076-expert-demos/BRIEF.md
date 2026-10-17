# EXP-0076 brief — re-read this before every new round (user prompt, 2026-10-16, refined)

## Two deliverables
1. **Expert demonstrations for imitation learning.** A few hundred Genesis trajectories that SOLVE the task
   (>= 95-100 % satisfied) in MINIMUM ACTION (execution) TIME. Planning time is NOT a constraint here.
   Aim: < 10 pushes per solved episode; fewer on easy tasks (few objects, big goal) — exploit those harder.
   Hard tasks: OK if they cannot be done in fewer.
2. **The best practical pipeline** for solving the tasks: may switch models, optimisers, horizons and planning
   budgets within an episode and between tasks, by empirical rules. Found by rigorous small ablations around
   the pipelines that look promising.

## Two pipelines, kept separate
- **Expert pipeline (anything goes):** oracle = simulator-as-model, privileged state, huge budgets, long-horizon
  search, "weird / cheaty" methods — only the produced pushes must be legal, executable Genesis actions.
- **Feasible pipeline:** only what a real robot could run (learned models on observations, bounded planning time).

## Task space
- Goals: shape letters, squares, circular regions. Object counts and initialisation types vary.
- Feasibility rule: total cube top-view area (linear in object count) <= 60 % of goal-region area.
  Check every task against it before use.

## Every round of experiments must end with
- at least one **"best trajectory for this one specific scenario and task"** saved (the per-scenario best
  demonstration, kept and replaced only by a better one);
- updates to the two living files: `LOG.md` (everything tried + direct results) and `SUMMARY.md` (state:
  what works, how many expert demos collected, split by object count x initialisation type x goal spec).

## Working rules
- Use the repo's skills (experiment-log, subagent-experimenter, data-collection, visualization).
  Subagents may test ideas. ONE GPU job at a time (shared 8 GB laptop GPU).
- Checkpoint every long run atomically; python -u; COMMANDS.jsonl ledger.
- Genesis legality: no blade touchdown on cubes (legalise every executed push).
- ~14 h unsupervised after the Q&A.

## User answers (2026-10-16 18:40) — binding
- Solved = in-goal MASS fraction >= 0.95 (soft-splat occ_for_scoring, the repo metric; raw fraction, not /V*).
  Objective = PHYSICAL execution time (EXP-0070 action_time: lift + transit + descent + sweep); push count reported too.
- Scenes: n20 and n50 only x {scattered, inbetween, piled} (Sean modes; start states from Sean TEST-split files).
- Goals: discs + squares at random position/size (cube area / goal area in [rho_lo, 0.6]) + letters O/T/S/X at w20/w15
  where the 60 % rule passes; no w10 letters.
- Feasible pipeline: planning time <= 3 s per decision (this laptop GPU).
