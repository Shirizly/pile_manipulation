#!/bin/bash
# zoom-factor ablation on short-push scattered scenes (L<=35 mm, scattered_n20+n50 train rows): same zoom64 [8,16,32] net, window side natural (max(64,L+44)) vs constant 79 / 114 / 150 mm; plus vanilla64 [8,16,32] on the same rows
cd /home/alon/Code/pile_manipulation; E=experiments/EXP-0074-wide-domain-zoom-nfd; P=/home/alon/anaconda3/envs/pme/bin/python
run() { name=$1; kind=$2; cache=$3; side=$4
  if [ -n "$side" ]; then export ZSIDE_FIXED=$side; else unset ZSIDE_FIXED; fi
  $P -u $E/code/train_wide.py --kind $kind --res 64 --features 8,16,32 --epochs 40 --cache $E/artifacts/$cache --out $E/runs/dom_$name > $E/runs/dom_$name.log 2>&1
  EVAL_FEAT=8,16,32 $P -u $E/code/eval_wide.py --$kind 64 $E/runs/dom_$name/unet_best.pth --name dom_$name --shards scattered_n20,scattered_n50 > $E/results/eval_dom_$name.log 2>&1; }
run zoom_nat zoom zoom_r64_sc35.pt "" &
run zoom_s79 zoom zoom_r64_sc35_s79.pt 79 &
run zoom_s114 zoom zoom_r64_sc35_s114.pt 114 &
run zoom_s150 zoom zoom_r64_sc35_s150.pt 150 &
run vanilla world world_r64_sc35.pt "" &
wait
