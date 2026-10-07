#!/bin/bash
cd /home/alon/Code/pile_manipulation; E=experiments/EXP-0074-wide-domain-zoom-nfd; P=/home/alon/anaconda3/envs/pme/bin/python; R=experiments/EXP-0074-wide-domain-zoom-nfd/runs
waitfor() { until grep -q "^$2 " $R/sp_$1.log 2>/dev/null; do sleep 20; done; }
one() { kind=$1; name=$2; shards=$3; sp="{\"type\":\"ens\",\"members\":[[\"$kind\",128,\"$R/sp_${kind}_$name/unet_last.pth\",\"8,16,32\"]]}"; $P -u $E/code/eval_variants.py --name sp_${kind}_$name --spec "$sp" --shards "$shards" --rolls $4 > $E/results/eval_sp_${kind}_$name.log 2>&1; }
for kind in zoom world; do
  waitfor ${kind}_n20 8; one $kind n20 scattered_n20,inbetween_n20,piled_n20 1
  waitfor ${kind}_n50 8; one $kind n50 scattered_n50,inbetween_n50,piled_n50 1
  waitfor ${kind}_n100 8; one $kind n100 scattered_n100,inbetween_n100 1
  waitfor ${kind}_control 3; one $kind control "" 1
  waitfor ${kind}_Llong 8
  sp="{\"type\":\"routed\",\"edges\":[35,55],\"members\":[[\"$kind\",128,\"$R/sp_${kind}_Lshort/unet_last.pth\",\"8,16,32\"],[\"$kind\",128,\"$R/sp_${kind}_Lmid/unet_last.pth\",\"8,16,32\"],[\"$kind\",128,\"$R/sp_${kind}_Llong/unet_last.pth\",\"8,16,32\"]]}"
  $P -u $E/code/eval_variants.py --name sp_${kind}_Lrouted --spec "$sp" --rolls 0 > $E/results/eval_sp_${kind}_Lrouted.log 2>&1
done
