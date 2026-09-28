# EXP-0058 — human demonstrations on the closed-loop benchmark tasks (design)

**Purpose.** A human-performance reference on exactly the tasks the perfect-model oracle
(EXP-0057) and the learned-model MPC runs (EXP-0055) are scored on: how many pushes does a
person need, and does a person solve tasks the oracle does not? Collection only; no claim yet.

**Tasks.** 8 goals x 2 starts = 16 episodes: goals `letter_O letter_T letter_S letter_X
letter_L letter_I two_squares quadrant_0`, start states DS-0006 corpus slates 40 and 41 (the
EXP-0057 task set). Goal masks via `batched_closed_loop.goal_mask` (same as oracle.py).

**Simulator.** One Genesis env set up exactly like `experiments/EXP-0050-oracle-ceiling/code/oracle.py`
(`oracle_config_with_physics` + `apply_physics`, real settle/clearance steps; start = corpus
state set + `update_material_state`; each push = `set_particle_state` from the current state,
`action_to_pose`, `execute_action`, `update_material_state`). Checked: the settled start states
match EXP-0057's recorded ones exactly; replaying EXP-0057's first pushes gives differences
(max ~4 mm on one cube, mean <1 mm) of the same size as this sim's own run-to-run repeat spread,
i.e. Genesis contact nondeterminism, not a setup difference.

**Rules (pure human).**
- Push = blade-centre start -> end, blade 40 mm perpendicular to the push; the drawn push is
  projected by `simple_mpc.learned_mpc.project_push` (length 20-70 mm, endpoints inside the tray)
  and executed exactly as projected. No simulator refinement, no free yaw, no undo (undo would
  let the person use the simulator as a model). The person sees only the true current state.
- Stop at solved (in-goal mass fraction via `occ_for_scoring` x mask >= 0.9 x EXP-0046
  `mass_frac_best_placement`) or after 20 pushes; "Give up" marks the task unsolved.
- Display: world x right, world y down (the demo-GIF orientation in which letters read correctly).

**Recorded** (atomic JSON per task, rewritten after every push):
`results/<operator>/<goal>_s<start>.json` with `actions` (executed, metres, [sx,sy,ex,ey]),
`states` (particle xyz before the first and after every push), `mass_frac`, `lyap`,
`think_s` (state shown -> Execute), `sim_s`, `opt`, `theta`, `solved_at`, `finished`, `gave_up`.

**Comparison plan.** `code/analyse.py` prints per task and overall: solved?, pushes to solve
(first k >= 1 with mass rel. optimum >= 0.9 -- the EXP-0055 completion rule), final in-goal
mass rel. optimum, mean think time, next to EXP-0057 `oracle_A_32x8` and `oracle_A_default`
on the same tasks. Learned-model cells from EXP-0055 can be added the same way (their episodes
carry `goal`, `start`, `states`). Wall-clock comparison: human think time per push vs planner
plan time per push. Caveat: one operator, learning effects across tasks (record task order).

Tool: `human_benchmark_gui.py` (repo root); see docs/human_demo_design.md.
