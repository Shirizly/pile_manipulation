#!/bin/bash
# single-step zoom64, 2000-row seed subset, scratch, 300 epochs, several UNet widths/depths
cd /home/alon/Code/pile_manipulation; R=experiments/EXP-0072-zoom-window-nfd; P=/home/alon/anaconda3/envs/pme/bin/python
for f in 4,8,16 8,16,32 16,32,64 4,8,16,32 8,16,32,64 16,32,64,128; do
  n=arch_sub2000_$(echo $f | tr , _)
  $P -u $R/code/train_zoom.py --scratch --subset 2000 --epochs 300 --features $f --out $R/runs/$n > $R/runs/$n.log 2>&1
done
echo done > $R/runs/arch_sweep.done
