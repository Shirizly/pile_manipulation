#!/usr/bin/env bash
# EXP-0039 RUN-0002 queue. Stage "pilot" (default) or "full". Waits for the ISS-013 legal re-runs to finish (GPU exclusive).
cd "$(dirname "$0")/../../.."
E=experiments/EXP-0039-closed-loop-metric-validity; R=$E/results/run0002; mkdir -p $R
D=experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py
G="letter_O letter_T letter_S letter_L letter_X letter_Z letter_C letter_H"
MODELS="nfd_3ch_narrow_l20_v2 nfd_3ch_narrow_l20_v2_seed1 nfd_3ch_narrow_l20_v2_seed2 nfd_3ch_narrow_l20_v2_soft_s2 linear_narrow_l20_v2_res64 linear_switched_soft nfd_3ch_randlen nfd_3ch_narrow_l20_v2_epoch10"
until [ -f experiments/EXP-0043-batched-closed-loop/code/legal_rerun_queue.done ]; do sleep 60; done
run() { PYTHONPATH=. python scripts/run_probe.py --tag "rp_$1" --out-dir $R --exp EXP-0039 -- python $D --tag "$1" --results-dir $R "${@:2}"; }
if [ "${1:-pilot}" = pilot ]; then
  run pilot_cem --models nfd_3ch_narrow_l20_v2 --planners cem --goals letter_O letter_T --starts $(seq 40 55) --steps 4 --budget 0.5 --n-cand 1024 \
      --cells '{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25, "push_len": 0.02}}' --legalize --record-states
  run pilot_rank --models linear_switched_soft --planners rank --goals letter_O letter_T --starts $(seq 40 47) --steps 2 --budget 0.0 --n-cand 128 \
      --cells '{"l20r": {"push_len": 0.02}}' --legalize --record-states
  echo "$(date) PILOT DONE" > $R/pilot.done
else
  PYTHONPATH=. python scripts/run_probe.py --tag rp_offline_ds0016 --out-dir $R --exp EXP-0039 -- python $E/code/offline_ds0016.py \
      --models nfd_3ch_narrow_l20_v2_soft_s2 nfd_3ch_randlen linear_switched_soft nfd_3ch_narrow_l20_v2_epoch10 --out $R/offline_ds0016_scores.json
  for M in $MODELS; do
    run hm_cem_$M --models $M --planners cem --goals $G --starts $(seq 40 55) --steps 16 --budget 0.5 --n-cand 1024 \
        --cells '{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25, "push_len": 0.02}}' --legalize --record-states
  done
  for M in $MODELS; do
    run hm_rank_$M --models $M --planners rank --goals $G --starts $(seq 40 47) --steps 16 --budget 0.0 --n-cand 128 \
        --cells '{"l20r": {"push_len": 0.02}}' --legalize --record-states
  done
  echo "$(date) FULL DONE" > $R/full.done
fi
