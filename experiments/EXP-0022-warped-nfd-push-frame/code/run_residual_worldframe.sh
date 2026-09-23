#!/usr/bin/env bash
# EXP-0022 RUN-0022/0023 -- WORLD-FRAME NFD + explicit residual head on
# overnight_randlen, no-augmentation then flip-only. Sequential: one 8 GB GPU.
set -u
cd /home/alon/Code/pile_manipulation
export OMP_NUM_THREADS=4 PYTHONPATH=.
PY=/home/alon/anaconda3/envs/pme/bin/python
SHA=$(git rev-parse HEAD)

run_arm () {
  local RUN_ID=$1 CFG=$2
  local D=experiments/EXP-0022-warped-nfd-push-frame/runs/$RUN_ID
  mkdir -p "$D"; cp "$CFG" "$D/config.yaml"
  local ARGV="$PY -u Baselines/NFD/train_nfd.py $CFG --no-resume"
  printf '%s\ncommit: %s\n' "$ARGV" "$SHA" > "$D/COMMAND.txt"
  printf '{"event":"start","experiment":"EXP-0022","run_id":"%s","ts":"%s","commit":"%s","dirty":true,"argv":"%s"}\n' \
    "$RUN_ID" "$(date -Is)" "$SHA" "$ARGV" >> experiments/COMMANDS.jsonl
  local T0=$(date +%s)
  $ARGV > "$D/stdout.log" 2> "$D/stderr.log"; local RC=$?
  printf '{"event":"end","experiment":"EXP-0022","run_id":"%s","ts":"%s","exit_code":%s,"seconds":%s}\n' \
    "$RUN_ID" "$(date -Is)" "$RC" "$(( $(date +%s) - T0 ))" >> experiments/COMMANDS.jsonl
  echo "=== $RUN_ID rc=$RC ==="; tail -3 "$D/stdout.log"
}

run_arm RUN-0022-residual-worldframe-noaug   Baselines/NFD/configs/nfd_train_residual_unwarped_noaug_randlen.yaml
run_arm RUN-0023-residual-worldframe-flipaug Baselines/NFD/configs/nfd_train_residual_unwarped_flipaug_randlen.yaml
echo "BOTH RESIDUAL WORLD-FRAME ARMS DONE"
