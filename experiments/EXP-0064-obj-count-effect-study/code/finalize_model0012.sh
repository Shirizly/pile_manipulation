#!/usr/bin/env bash
# Promote RUN-0005's net_best.pth to MODEL-0012 and re-score it (RUN-0006 'fixed' arm), then rebuild results/.
# Run from the repo root after RUN-0005 finishes (or on its current best checkpoint, labelled as such in the record).
set -euo pipefail
E=experiments/EXP-0064-obj-count-effect-study
RUN_DIR=$(ls -d $E/artifacts/RUN-0005-train-fixed-frame/20*/ | tail -1)
W=weights/MODEL-0012-gnn-flex-particles-countgroups-fixed-frame
mkdir -p $W/epochs
cp "$RUN_DIR/net_best.pth" $W/checkpoint.pth
cp "$RUN_DIR"/net_*.pth $W/epochs/ 2>/dev/null || true
cp "$RUN_DIR/log.txt" $W/epochs/train_log.txt
cp $E/runs/RUN-0005-train-fixed-frame/config.yaml $W/config.yaml
sha256sum $W/checkpoint.pth
rm -rf $E/artifacts/RUN-0006-eval-extended/fixed
PYTHONPATH=. python scripts/run_probe.py --tag exp0064_run0006_fixed --out-dir $E/artifacts/RUN-0006-eval-extended --exp EXP-0064 \
  --artifact-dir $E/artifacts/RUN-0006-eval-extended -- python $E/code/eval_extended.py --out $E/artifacts/RUN-0006-eval-extended/fixed \
  --models gnn_fixed:-1:$W/checkpoint.pth --fps-reps 3 --n-extra-goals 16
OMP_NUM_THREADS=4 python $E/code/summarize_extended.py --dirs $E/artifacts/RUN-0006-eval-extended/asrun $E/artifacts/RUN-0006-eval-extended/fixed --out $E/results
