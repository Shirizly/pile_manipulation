#!/usr/bin/env bash
# EXP-0071 queue (flags as EXP-0039 RUN-0002 run0002_queue.sh). Each unit is atomic (driver writes results/<tag>.json after every chunk).
cd "$(dirname "$0")/../../.."; export OMP_NUM_THREADS=4
E=experiments/EXP-0071-gated-ensemble; R=$E/results/run0001; mkdir -p $R
D=experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py
G="letter_O letter_T letter_S letter_L letter_X letter_Z letter_C letter_H"
run() { PYTHONPATH=. python scripts/run_probe.py --tag "rp_$1" --out-dir $R --exp EXP-0071 -- python $D --tag "$1" --results-dir $R "${@:2}"; }
for M in gated_ens_narrow_v2 nfd3_meanblend_narrow_v2; do
  [ -f $R/hm_rank_$M.json ] || run hm_rank_$M --models $M --planners rank --goals $G --starts $(seq 40 47) --steps 16 --budget 0.0 --n-cand 128 \
      --cells '{"l20r": {"push_len": 0.02}}' --legalize --record-states
done
for M in gated_ens_narrow_v2 nfd3_meanblend_narrow_v2; do
  [ -f $R/hm_cem_$M.json ] || run hm_cem_$M --models $M --planners cem --goals $G --starts $(seq 40 55) --steps 16 --budget 0.5 --n-cand 1024 \
      --cells '{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25, "push_len": 0.02}}' --legalize --record-states
done
echo "$(date) DONE" > $R/full.done
