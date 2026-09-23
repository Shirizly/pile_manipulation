#!/usr/bin/env bash
# EXP-0022 -- train the four warped NFD arms on overnight_randlen, SEQUENTIALLY.
#
# Sequential, never parallel: one 8 GB GPU on a shared machine. Each arm takes
# the SAME 30 epochs as the world-frame baseline it is compared against
# (Baselines/NFD/runs/nfd_3ch_randlen, scored in EXP-0001), so the arms are
# matched by gradient-step count rather than by wall-clock.
#
# Writes per-run logs + COMMAND.txt under the run dir, and start/end events to
# experiments/COMMANDS.jsonl joined on run_id.
set -u
cd /home/alon/Code/pile_manipulation
export OMP_NUM_THREADS=4
export PYTHONPATH=.
PY=/home/alon/anaconda3/envs/pme/bin/python
LEDGER=experiments/COMMANDS.jsonl
SHA=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain | wc -l)

run_arm () {
  local RUN_ID=$1 CFG=$2
  local DIR=experiments/EXP-0022-warped-nfd-push-frame/runs/$RUN_ID
  mkdir -p "$DIR"
  cp "$CFG" "$DIR/config.yaml"
  local ARGV="$PY -u Baselines/NFD/train_nfd.py $CFG --no-resume"
  printf '%s\ncommit: %s (dirty_files=%s)\n' "$ARGV" "$SHA" "$DIRTY" > "$DIR/COMMAND.txt"
  printf '{"event":"start","experiment":"EXP-0022","run_id":"%s","ts":"%s","commit":"%s","dirty":%s,"argv":"%s"}\n' \
    "$RUN_ID" "$(date -Is)" "$SHA" "$([ "$DIRTY" -gt 0 ] && echo true || echo false)" "$ARGV" >> $LEDGER
  local T0=$(date +%s)
  $ARGV > "$DIR/stdout.log" 2> "$DIR/stderr.log"
  local RC=$?
  printf '{"event":"end","experiment":"EXP-0022","run_id":"%s","ts":"%s","exit_code":%s,"seconds":%s}\n' \
    "$RUN_ID" "$(date -Is)" "$RC" "$(( $(date +%s) - T0 ))" >> $LEDGER
  echo "=== $RUN_ID finished rc=$RC in $(( $(date +%s) - T0 ))s ==="
}

run_arm RUN-0005-warped-randlen-r64       Baselines/NFD/configs/nfd_train_warped_randlen.yaml
run_arm RUN-0006-warped-walls-randlen-r64 Baselines/NFD/configs/nfd_train_warped_walls_randlen.yaml
run_arm RUN-0007-warped-randlen-r96       Baselines/NFD/configs/nfd_train_warped_r96_randlen.yaml
run_arm RUN-0008-warped-walls-randlen-r96 Baselines/NFD/configs/nfd_train_warped_walls_r96_randlen.yaml
echo "ALL FOUR ARMS DONE"
