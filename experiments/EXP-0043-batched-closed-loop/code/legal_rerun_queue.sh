#!/usr/bin/env bash
# ISS-013 legal re-runs (2026-10-04): EXP-0051 and EXP-0054 headline cells, exact recorded commands + --legalize,
# sequential. Results land beside the originals with a _legal tag.
cd "$(dirname "$0")/../../.."
D=experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py
PYTHONPATH=. python scripts/run_probe.py --tag exp0051_success_legal --out-dir experiments/EXP-0051-task-success-completion-time/results --exp EXP-0051 -- \
  python $D --tag success_states_legal --results-dir experiments/EXP-0051-task-success-completion-time/results --record-states --legalize \
  --models nfd_residual_worldframe_noaug_ep43 linear_switched_soft --goals quadrant_0 quadrant_3 letter_O letter_T letter_S letter_L letter_X letter_Z two_squares \
  --starts 40 41 --steps 24 --cells '{"tuned": {"gd": {"lr": 0.005, "n_restarts": 32}, "cem": {"n_cand": 1024, "cem_pop": 1024, "cem_elite_frac": 0.25}}}'
PYTHONPATH=. python scripts/run_probe.py --tag exp0054_narrow_legal --out-dir experiments/EXP-0054-narrow-closed-loop/results --exp EXP-0054 -- \
  python $D --tag narrow_gd_w3_legal --results-dir experiments/EXP-0054-narrow-closed-loop/results --record-states --legalize \
  --models nfd_3ch_narrow_l20 nfd_3ch_narrow_l20_wide nfd_3ch_randlen nfd_residual_worldframe_noaug_ep43 --planners gd \
  --goals letter_O letter_T letter_S letter_L letter_X letter_Z two_squares --starts 40 41 42 43 --steps 24 \
  --cells '{"gd_w3_l20": {"gd": {"lr": 0.005, "n_restarts": 32, "mass_weight": 3.0, "push_len": 0.02}}}'
echo "$(date) LEGAL QUEUE DONE" > experiments/EXP-0043-batched-closed-loop/code/legal_rerun_queue.done
