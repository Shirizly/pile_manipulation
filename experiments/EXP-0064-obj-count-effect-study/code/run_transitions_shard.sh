#!/bin/bash
# Supervisor for one collect_transitions.py shard: restarts it on a crash
# (segfault during env.reset(), see collect_transitions.py's module docstring
# -- a rare but real native crash that can't be caught from Python) until it
# exits 0 (all assigned states complete or permanently skipped) or the
# restart cap is hit.
#
# Usage: run_transitions_shard.sh <config.yaml> <shard> <n_shards> <logfile>

set -u
CONFIG="$1"
SHARD="$2"
N_SHARDS="$3"
LOGFILE="$4"
MAX_RESTARTS=200

for ((i = 1; i <= MAX_RESTARTS; i++)); do
    echo "=== attempt $i/$MAX_RESTARTS ===" >> "$LOGFILE"
    PYFLEX_HEADLESS_OVERRIDE=0 xvfb-run -a -s "-screen 0 1280x1024x24" \
        python -u collect_transitions.py "$CONFIG" --shard "$SHARD" --n-shards "$N_SHARDS" >> "$LOGFILE" 2>&1
    code=$?
    if [ $code -eq 0 ]; then
        echo "=== shard $SHARD finished cleanly after $i attempt(s) ===" >> "$LOGFILE"
        exit 0
    fi
    echo "=== attempt $i exited $code, restarting ===" >> "$LOGFILE"
    sleep 2
done

echo "=== gave up after $MAX_RESTARTS attempts ===" >> "$LOGFILE"
exit 1
