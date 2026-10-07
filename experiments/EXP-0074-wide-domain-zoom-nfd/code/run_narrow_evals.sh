#!/bin/bash
cd /home/alon/Code/pile_manipulation; P=/home/alon/anaconda3/envs/pme/bin/python; E=experiments/EXP-0074-wide-domain-zoom-nfd; N=experiments/EXP-0072-zoom-window-nfd/runs; R=experiments/EXP-0074-wide-domain-zoom-nfd/runs
ev() { name=$1; kind=$2; res=$3; ck=$4; feat=$5; EVAL_FEAT=$feat $P -u $E/code/eval_wide.py --$kind $res $ck --sets narrow --name narrowtest_$name > $E/results/eval_narrowtest_$name.log 2>&1; }
ev narrowtrained_zoom64_ss zoom 64 $N/ft300/unet_best.pth 4,8,16
ev narrowtrained_zoom64_ms zoom 64 $N/ms64_T8/unet_last.pth 4,8,16
ev narrowtrained_zoom128_ss zoom 128 $N/zoom128/unet_best.pth 4,8,16
ev narrowtrained_zoom128_ms zoom 128 $N/ms128_T8/unet_last.pth 4,8,16
ev narrowtrained_world128_ss world 128 $N/world128_300/unet_best.pth 4,8,16
ev narrowtrained_world128_ms world 128 $N/msw128_T4/unet_last.pth 4,8,16
ev widetrained_zoom64_ms zoom 64 $R/z64_f8_ms4/unet_best.pth 8,16,32
ev widetrained_world64_ms world 64 $R/w64_s0_ms4/unet_best.pth 4,8,16
ev widetrained_zoom128_ms zoom 128 $R/z128_f8_ms4/unet_best.pth 8,16,32
ev widetrained_world128_ms world 128 $R/w128_f8_ms4/unet_best.pth 8,16,32
