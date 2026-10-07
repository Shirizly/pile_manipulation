#!/bin/bash
cd /home/alon/Code/pile_manipulation; E=experiments/EXP-0074-wide-domain-zoom-nfd; P=/home/alon/anaconda3/envs/pme/bin/python
unset ZSIDE_FIXED
$P -u $E/code/train_wide.py --kind world --res 228 --features 8,16,32 --epochs 30 --bs 32 --cache $E/artifacts/world_r228_sc55.pt --out $E/runs/dom_L55_vanilla228 > $E/runs/dom_L55_vanilla228.log 2>&1
EVAL_FEAT=8,16,32 $P -u $E/code/eval_wide.py --world 228 $E/runs/dom_L55_vanilla228/unet_best.pth --name dom_L55_vanilla228 --shards scattered_n20,scattered_n50,scattered_n100 > $E/results/eval_dom_L55_vanilla228.log 2>&1
