#!/usr/bin/env bash
# EXP-0064 overnight queue (2026-10-03): RUN-0008..0011 one at a time, each scored right after training
# (RUN-0012 eval dirs), then one summary over every scored model. Re-runnable: a finished run (net_best.pth +
# 'valid [99/100]' in its log) or a scored model (manifest with 200 states) is skipped.
cd "$(dirname "$0")/../../.."
E=experiments/EXP-0064-obj-count-effect-study
EV=$E/artifacts/RUN-0012-eval-overnight
mkdir -p $EV
declare -A ENC=( [RUN-0008-train-drop-escaped]=tube [RUN-0009-train-orig-encoding]=orig [RUN-0010-train-seed43]=tube [RUN-0011-train-seed44]=tube )
declare -A NAME=( [RUN-0008-train-drop-escaped]=gnn_dropesc [RUN-0009-train-orig-encoding]=gnn_origenc [RUN-0010-train-seed43]=gnn_fixed_s43 [RUN-0011-train-seed44]=gnn_fixed_s44 )
for R in RUN-0008-train-drop-escaped RUN-0009-train-orig-encoding RUN-0010-train-seed43 RUN-0011-train-seed44; do
  A=$E/artifacts/$R; mkdir -p $A
  echo "$(date) START $R" >> $EV/queue_status.log
  if ! grep -qs "valid \[99/100\] Loss" $A/*.log; then
    PYTHONPATH=. python scripts/run_probe.py --tag exp0064_${R,,} --out-dir $A --exp EXP-0064 --artifact-dir $A -- \
      python $E/code/train_gnn_dyn_grouped.py --config $E/runs/$R/config.yaml --out-root $A >> $EV/queue_status.log 2>&1
  fi
  CK=$(ls -d $A/20*/ 2>/dev/null | tail -1)net_best.pth
  echo "$(date) TRAINED $R ckpt=$CK" >> $EV/queue_status.log
  N=${NAME[$R]}
  if [ -f "$CK" ] && ! python -c "import json,sys; sys.exit(0 if len(json.load(open('$EV/$N/manifest.json'))['done'])>=200 else 1)" 2>/dev/null; then
    PYTHONPATH=. python scripts/run_probe.py --tag exp0064_run0012_$N --out-dir $EV --exp EXP-0064 --artifact-dir $EV -- \
      python $E/code/eval_extended.py --out $EV/$N --models $N:-1:$CK:${ENC[$R]} --fps-reps 3 --n-extra-goals 16 >> $EV/queue_status.log 2>&1
  fi
  echo "$(date) SCORED $R" >> $EV/queue_status.log
  OMP_NUM_THREADS=4 python $E/code/summarize_extended.py --dirs $E/artifacts/RUN-0006-eval-extended/asrun $E/artifacts/RUN-0006-eval-extended/fixed $EV/*/ --out $E/results > /dev/null 2>&1
done
echo "$(date) QUEUE DONE" >> $EV/queue_status.log
