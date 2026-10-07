#!/bin/bash
# long pushes (>=55 mm), scattered shards: constant window side 114 mm at res 128 (0.9 mm/px) vs res 228 (0.5 mm/px, the narrow-pilot zoom) vs vanilla world at res 128 (1 mm/px) and 228 (0.56 mm/px); [8,16,32], 30 epochs, one-step acc1_L>=55 on the scattered test shards
cd /home/alon/Code/pile_manipulation; E=experiments/EXP-0074-wide-domain-zoom-nfd; P=/home/alon/anaconda3/envs/pme/bin/python
run() { name=$1; kind=$2; res=$3; cache=$4; side=$5
  if [ -n "$side" ]; then export ZSIDE_FIXED=$side; else unset ZSIDE_FIXED; fi
  $P -u $E/code/train_wide.py --kind $kind --res $res --features 8,16,32 --epochs 30 --cache $E/artifacts/$cache --out $E/runs/dom_$name > $E/runs/dom_$name.log 2>&1
  EVAL_FEAT=8,16,32 $P -u $E/code/eval_wide.py --$kind $res $E/runs/dom_$name/unet_best.pth --name dom_$name --shards scattered_n20,scattered_n50,scattered_n100 > $E/results/eval_dom_$name.log 2>&1; }
run L55_zoom128 zoom 128 zoom_r128_sc55_s114.pt 114 &
run L55_zoom228 zoom 228 zoom_r228_sc55_s114.pt 114 &
run L55_vanilla128 world 128 world_r128_sc55.pt "" &
run L55_vanilla228 world 228 world_r228_sc55.pt "" &
wait
