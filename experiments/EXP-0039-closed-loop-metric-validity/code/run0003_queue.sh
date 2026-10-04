#!/usr/bin/env bash
# EXP-0039 RUN-0003: start-regime test, CEM arm, one unit per model (scatter pools 0-7 + clump pools 16-23 of DS-0016).
cd "$(dirname "$0")/../../.."
E=experiments/EXP-0039-closed-loop-metric-validity; R=$E/results/run0003; mkdir -p $R
D=experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py
G="letter_O letter_T letter_S letter_L letter_X letter_Z letter_C letter_H"
for M in nfd_3ch_narrow_l20_v2 nfd_3ch_narrow_l20_v2_seed1 nfd_3ch_narrow_l20_v2_seed2 nfd_3ch_narrow_l20_v2_soft_s2 linear_narrow_l20_v2_res64 linear_switched_soft nfd_3ch_randlen nfd_3ch_narrow_l20_v2_epoch10; do
  PYTHONPATH=. python scripts/run_probe.py --tag rp_hm_cem_$M --out-dir $R --exp EXP-0039 -- python $D --tag hm_cem_$M --results-dir $R \
    --starts-file Genesis/data/narrow_l20_n20/test_pools_v2/pools_0.pt --starts 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 \
    --models $M --planners cem --goals $G --steps 16 --budget 0.5 --n-cand 1024 \
    --cells '{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25, "push_len": 0.02}}' --legalize --record-states
done
echo "$(date) RUN0003 DONE" > $R/full.done
