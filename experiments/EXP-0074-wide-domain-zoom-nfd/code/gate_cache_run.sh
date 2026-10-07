#!/bin/bash
# usage: gate_cache_run.sh SPLIT  -- caches z, w, lf for one split sequentially
cd /home/alon/Code/pile_manipulation
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0074-wide-domain-zoom-nfd
export OMP_NUM_THREADS=3
for m in z w lf; do
  $P -u $E/code/gate_cache.py $1 $m
done
