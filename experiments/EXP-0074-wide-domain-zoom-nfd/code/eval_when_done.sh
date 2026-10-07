#!/bin/bash
# usage: eval_when_done.sh LOGFILE FINAL_EPOCH NAME KIND(zoom|world) RES CKPT [ENV=VAL...]  -- waits for "^FINAL_EPOCH " in LOGFILE then runs eval_wide.py and regenerates the table
cd /home/alon/Code/pile_manipulation
LOG=$1; EP=$2; NAME=$3; KIND=$4; RES=$5; CK=$6; shift 6
until grep -q "^$EP " $LOG; do sleep 20; done
env "$@" /home/alon/anaconda3/envs/pme/bin/python -u experiments/EXP-0074-wide-domain-zoom-nfd/code/eval_wide.py --$KIND $RES $CK --name $NAME > experiments/EXP-0074-wide-domain-zoom-nfd/results/eval_$NAME.log 2>&1
/home/alon/anaconda3/envs/pme/bin/python experiments/EXP-0074-wide-domain-zoom-nfd/code/make_table.py > /dev/null
