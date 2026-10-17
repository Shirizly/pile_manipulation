#!/usr/bin/env bash
# EXP-0068 CPU tail: analysis with ledger lines (run after queue.sh wrote results/queue.done).
cd "$(dirname "$0")/../../.."
PY=/home/alon/anaconda3/envs/pme/bin/python; E=experiments/EXP-0068-visited-state-metrics; SHA=$(git rev-parse --short HEAD)
ledger() { printf '{"ts":"%s","run_id":"EXP-0068/RUN-0003/analyse","event":"%s"%s}\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> experiments/COMMANDS.jsonl; }
ledger start ",\"cmd\":\"CUDA_VISIBLE_DEVICES= python -u $E/code/analyse.py\",\"commit\":\"$SHA\",\"dirty\":true"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=. $PY -u $E/code/analyse.py > $E/logs/analyse.log 2>&1; rc=$?
ledger end ",\"status\":\"$([ $rc = 0 ] && echo ok || echo fail)\",\"returncode\":$rc"
echo "analyse rc=$rc"; tail -60 $E/logs/analyse.log
