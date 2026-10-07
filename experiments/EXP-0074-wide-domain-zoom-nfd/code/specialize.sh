#!/bin/bash
# post-training specialisation of the single-step generalists (z128_f8 / w128_f8, seed 0) on object-count or push-length slices of the SAME training cache (no new data); lr 5e-5, 8 epochs; control = generalist continued on all rows (3 epochs ~ same number of steps)
cd /home/alon/Code/pile_manipulation; E=experiments/EXP-0074-wide-domain-zoom-nfd; P=/home/alon/anaconda3/envs/pme/bin/python
ft() { kind=$1; name=$2; shift 2; base=$([ $kind = zoom ] && echo z128_f8 || echo w128_f8)
  $P -u $E/code/train_wide.py --kind $kind --res 128 --features 8,16,32 --init $E/runs/$base/unet_ss_best.pth --lr 5e-5 "$@" --out $E/runs/sp_${kind}_$name > $E/runs/sp_${kind}_$name.log 2>&1; }
lane() { kind=$1
  ft $kind n20 --epochs 8 --shards scattered_n20,inbetween_n20,piled_n20
  ft $kind n50 --epochs 8 --shards scattered_n50,inbetween_n50,piled_n50
  ft $kind n100 --epochs 8 --shards scattered_n100,inbetween_n100
  ft $kind control --epochs 3; }
lanel() { kind=$1
  ft $kind Lshort --epochs 8 --lmax 34.999
  ft $kind Lmid --epochs 8 --lmin 35 --lmax 54.999
  ft $kind Llong --epochs 8 --lmin 55; }
lane zoom & lane world & lanel zoom & lanel world & wait
