#!/bin/bash
# chain.sh KIND RES FEATS NAME CUT_EPOCH MS_EPOCHS : wait for epoch CUT in runs/NAME.log, stop the single-step run, eval it (NAME), unrolled T=4 fine-tune (NAME_ms4), eval it.
cd /home/alon/Code/pile_manipulation; E=experiments/EXP-0074-wide-domain-zoom-nfd; P=/home/alon/anaconda3/envs/pme/bin/python
KIND=$1; RES=$2; FEATS=$3; NAME=$4; CUT=$5; MSE=$6
until grep -q "^$CUT " $E/runs/$NAME.log; do sleep 20; done
ids=$(ps aux | grep -F -- "--out $E/runs/$NAME" | grep -v grep | grep train_wide | awk '{print $2}'); [ -n "$ids" ] && kill $ids; sleep 3
cp $E/runs/$NAME/unet_best.pth $E/runs/$NAME/unet_ss_best.pth
export EVAL_FEAT=$FEATS
nohup $P -u $E/code/eval_wide.py --$KIND $RES $E/runs/$NAME/unet_ss_best.pth --name $NAME > $E/results/eval_$NAME.log 2>&1 &
$P -u $E/code/ms_wide.py --kind $KIND --res $RES --features $FEATS --init $E/runs/$NAME/unet_ss_best.pth --T 4 --epochs $MSE --out $E/runs/${NAME}_ms4 > $E/runs/${NAME}_ms4.log 2>&1
$P -u $E/code/eval_wide.py --$KIND $RES $E/runs/${NAME}_ms4/unet_best.pth --name ${NAME}_ms4 > $E/results/eval_${NAME}_ms4.log 2>&1
$P $E/code/make_table.py > /dev/null
