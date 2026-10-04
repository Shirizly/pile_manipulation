#!/bin/bash
# EXP-0064 RUN-0014 + RUN-0015 tail: waits for the overnight GNN queue to release the GPU
# (line "QUEUE DONE" in artifacts/RUN-0012-eval-overnight/queue_status.log), then
#   1. trains NFD (EXP-0062 nfd_3ch_flex_mask_v2 recipe on DS-0021, seed 0, plateau stop; resumable),
#   2. promotes unet_best.pth -> weights/MODEL-0014-nfd-flex-mask-countgroups-seed0/checkpoint.pth,
#   3. scores it on DS-0022 (score_image_metrics.py score --models nfd14 --device cuda) and re-runs analyze.
# Every step is idempotent (resume / skip-if-present). Status lines -> artifacts/RUN-0014-nfd-train/chain_status.log
set -u
cd /home/alon/Code/pile_manipulation
source ~/anaconda3/etc/profile.d/conda.sh; conda activate pme
export PYTHONPATH=.
E=experiments/EXP-0064-obj-count-effect-study
A=$E/artifacts/RUN-0014-nfd-train
S=$A/chain_status.log
Q=$E/artifacts/RUN-0012-eval-overnight/queue_status.log
RUN=Baselines/NFD/runs/nfd_3ch_flex_mask_ds0021_seed0
M=weights/MODEL-0014-nfd-flex-mask-countgroups-seed0
echo "$(date) WAIT for QUEUE DONE" >> $S
until grep -q 'QUEUE DONE' $Q; do sleep 300; done
echo "$(date) QUEUE DONE seen; START NFD training" >> $S
if [ ! -f $RUN/DONE_CHAIN ]; then
  RES=""; [ -f $RUN/last_state.pt ] && RES="--resume"
  python scripts/run_probe.py --tag exp0064_run0014_nfd_train --threads 8 --exp EXP-0064 --out-dir $A --artifact-dir $A -- \
    python Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_3ch_flex_mask_ds0021.yaml --seed 0 $RES || { echo "$(date) TRAIN FAILED" >> $S; exit 1; }
  touch $RUN/DONE_CHAIN
fi
echo "$(date) TRAINED" >> $S
mkdir -p $M
cp $RUN/unet_best.pth $M/checkpoint.pth
[ -f $RUN/run_config.yaml ] && cp $RUN/run_config.yaml $M/config.yaml
sha256sum $M/checkpoint.pth > $M/checkpoint.sha256
echo "$(date) PROMOTED $(cat $M/checkpoint.sha256)" >> $S
python scripts/run_probe.py --tag exp0064_run0015_score_nfd14 --threads 8 --exp EXP-0064 --out-dir $E/artifacts/RUN-0015-image-metric-scoring -- \
  python $E/code/score_image_metrics.py score --models nfd14 --device cuda || { echo "$(date) SCORE FAILED" >> $S; exit 1; }
python scripts/run_probe.py --tag exp0064_run0015_analyze --threads 8 --exp EXP-0064 --out-dir $E/artifacts/RUN-0015-image-metric-scoring -- \
  python $E/code/score_image_metrics.py analyze
echo "$(date) CHAIN DONE" >> $S
