#!/bin/bash
cd /home/alon/Code/pile_manipulation; E=experiments/EXP-0074-wide-domain-zoom-nfd; P=/home/alon/anaconda3/envs/pme/bin/python
ft() { name=$1; shift; $P -u $E/code/train_wide.py --kind world --res 128 --features 8,16,32 --init $E/runs/w128_f8/unet_ss_best.pth --lr 5e-5 "$@" --out $E/runs/sp_world_$name > $E/runs/sp_world_$name.log 2>&1; }
ft n20 --epochs 8 --shards scattered_n20,inbetween_n20,piled_n20
ft n50 --epochs 8 --shards scattered_n50,inbetween_n50,piled_n50
ft n100 --epochs 8 --shards scattered_n100,inbetween_n100
ft control --epochs 3
