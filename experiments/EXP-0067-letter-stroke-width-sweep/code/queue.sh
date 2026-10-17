#!/usr/bin/env bash
# EXP-0067 queue: learned-model width sweep (w10 -> w15 -> w20), then the oracle anchors A1/A2 if the DESIGN.md budget rule
# allows. Sequential, GPU exclusive, python -u, pid files, driver checkpoints per push, ledger lines in COMMANDS.jsonl.
cd "$(dirname "$0")/../../.."
PY=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0067-letter-stroke-width-sweep; R=$E/results; L=$E/logs; mkdir -p $R $L
D=experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py
O=experiments/EXP-0050-oracle-ceiling/code/oracle.py
STARTS=$(seq 40 71)
CELLS='{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25, "push_len": 0.02}}'
SHA=$(git rev-parse --short HEAD)
T0=$(date +%s)
ledger() { printf '{"ts":"%s","run_id":"EXP-0067/RUN-0001/%s","event":"%s"%s}\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" "$3" >> experiments/COMMANDS.jsonl; }
gpu_h() { echo "scale=2; ($(date +%s) - $T0) / 3600" | bc; }
run_learned() {  # tag width
  local tag=$1 w=$2; local G="letter_O_w$w letter_T_w$w letter_S_w$w letter_X_w$w"
  if [ -f $R/$tag.done ]; then echo "$tag already done"; return; fi
  if [ $(date +%s) -gt $(date -d "13:00" +%s) ]; then echo "$(date) SKIP $tag (budget)" | tee -a $R/skipped.txt; return; fi
  echo "$(date) START $tag"
  ledger $tag start ",\"cmd\":\"$D --tag $tag --models nfd_3ch_randlen --goals $G --starts 40..71 --planner-seed 0 (EXP-0066 settings)\",\"commit\":\"$SHA\",\"dirty\":true"
  OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u $D --tag $tag --results-dir $R --models nfd_3ch_randlen --planners cem --goals $G --starts $STARTS \
    --steps 16 --budget 0.5 --n-cand 1024 --cells "$CELLS" --legalize --legalize-fallback --planner-seed 0 --record-states > $L/$tag.log 2>&1 &
  echo $! > $L/$tag.pid; echo "$! EXP-0067" > experiments/temp/gpu.lock; wait $!; rc=$?
  ledger $tag end ",\"status\":\"$([ $rc = 0 ] && echo ok || echo fail)\",\"returncode\":$rc"
  [ $rc = 0 ] && date > $R/$tag.done
  echo "$(date) END $tag rc=$rc"
}
run_oracle() {  # tag width deadline(HH:MM) max_gpu_h_before_start
  local tag=$1 w=$2 dl=$3 maxh=$4; local G="letter_O_w$w letter_T_w$w"
  if [ -f $R/$tag.done ]; then echo "$tag already done"; return; fi
  if [ $(date +%s) -gt $(date -d "$dl" +%s) ] || [ $(echo "$(gpu_h) > $maxh" | bc) = 1 ]; then echo "$(date) SKIP $tag (budget, gpu_h $(gpu_h))" | tee -a $R/skipped.txt; return; fi
  echo "$(date) START $tag"
  ledger $tag start ",\"cmd\":\"$O --mode cem --tag $tag --n-envs 128 --cem-pop 64 --cem-iters 4 --cem-elite 16 --steps 16 --goals $G --starts 40 41 42 43 --seed-base 0 --stop-solved --vstar-files $R/vstar_width.json\",\"commit\":\"$SHA\",\"dirty\":true"
  OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u $O --mode cem --tag $tag --n-envs 128 --cem-pop 64 --cem-iters 4 --cem-elite 16 --steps 16 \
    --goals $G --starts 40 41 42 43 --out-dir $R --seed-base 0 --stop-solved --vstar-files $R/vstar_width.json > $L/$tag.log 2>&1 &
  echo $! > $L/$tag.pid; echo "$! EXP-0067" > experiments/temp/gpu.lock; wait $!; rc=$?
  ledger $tag end ",\"status\":\"$([ $rc = 0 ] && echo ok || echo fail)\",\"returncode\":$rc"
  [ $rc = 0 ] && date > $R/$tag.done
  echo "$(date) END $tag rc=$rc"
}
run_learned w10 10
run_learned w15 15
run_learned w20 20
run_oracle oracle_w15 15 "12:30" 3.5
run_oracle oracle_w10 10 "13:30" 3.5
echo "$(date) QUEUE DONE gpu_h $(gpu_h)" > $R/queue.done
