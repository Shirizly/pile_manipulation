#!/bin/bash
# EXP-0075 T0a CLEAN timing: every group in 3 separate processes, reps interleaved (rep outer); skips files already present; aggregate after each process.
cd /home/alon/Code/pile_manipulation; source ~/anaconda3/etc/profile.d/conda.sh; conda activate pme; export OMP_NUM_THREADS=4 PYTHONPATH=.
D=experiments/EXP-0075-closed-loop-benchmark
for rep in 0 1 2; do for g in legal lf vanilla64 zoom64 vanilla128 zoom128 ens128 overhead gd_lf gd_vanilla64 gd_zoom64 gd_vanilla128 gd_zoom128 gd_ens128; do
  [ -f $D/results/timing_clean_raw/${g}_p${rep}.json ] && continue
  echo "== $(date +%T) $g rep $rep $(cut -d' ' -f1-3 /proc/loadavg)"; python -u $D/code/time_clean.py --group $g --rep $rep 2>&1 | grep -v -i warning
  python -u $D/code/time_clean.py --aggregate
done; done; echo "== DONE $(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"
