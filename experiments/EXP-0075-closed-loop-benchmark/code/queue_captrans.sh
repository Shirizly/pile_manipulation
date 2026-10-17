#!/bin/bash
# EXP-0075 capacity-transport objective / proposals test on thin letters (LOG.md 2026-10-11, results/captrans/). One driver call per (width, H) group; the driver
# checkpoints results/captrans/<group>.json after every step and resumes incomplete cells; ledger lines in experiments/COMMANDS.jsonl. Priority order a..d (brief);
# e (w20 H=2/H=4 at 320 evals, fetch vs OT pool) only if time remains.
# Cell tag ct_<width>_<H>_<obj>_<prop>: obj cur = lyapunov - in-goal fraction, captrans = + crowding-transport term; prop fetch = fetch extras, captrans = capacity-transport extras
# (H=2: distinct-cube suffixes from the same rows, pool_mode flot). All arms: ens2bal, 10 pushes, full push range, fetch/captrans extras n_extra 64, legal scoring, GD 20 steps, 1,280 evals.
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
P=/home/alon/anaconda3/envs/pme/bin/python
E=experiments/EXP-0075-closed-loop-benchmark
D=$E/code/closed_loop.py
R=$E/results/captrans
STARTS="40 41 42 43 44 45 46 47"
SHA=$(git rev-parse --short HEAD)
C='"model":"ens2bal","planner":"cem","legal_score":true,"gd_steps":20'
run() { tag=$1; goals=$2; cells=$3; [ -f $R/$tag.done ] && return; rid="EXP-0075/captrans/$tag-$(date +%s)"; t0=$(date +%s)
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $goals --starts $STARTS --steps 10 --cells <see queue_captrans.sh>\",\"note\":\"EXP-0075 captrans cells, HEAD $SHA + uncommitted (dirty)\"}" >> experiments/COMMANDS.jsonl
  PYTHONPATH=. $P -u $D --tag $tag --results-dir $R --goals $goals --starts $STARTS --cells "$cells" --steps 10 > $E/logs/captrans_$tag.log 2>&1; rc=$?
  echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
  [ $rc = 0 ] && touch $R/$tag.done; }
for W in 15 10; do
  G="letter_O_w$W letter_T_w$W letter_S_w$W letter_X_w$W"
  run w${W}_H1 "$G" "{\"ct_w${W}_H1_cur_fetch\":{$C,\"H\":1,\"fetch\":true}, \"ct_w${W}_H1_cur_captrans\":{$C,\"H\":1,\"captrans\":true}, \"ct_w${W}_H1_captrans_captrans\":{$C,\"H\":1,\"captrans\":true,\"objective\":\"captrans\"}}"
done
for W in 15 10; do
  G="letter_O_w$W letter_T_w$W letter_S_w$W letter_X_w$W"
  run w${W}_H2 "$G" "{\"ct_w${W}_H2_cur_fetch\":{$C,\"H\":2,\"obj_mode\":\"cumspill\",\"fetch\":true}, \"ct_w${W}_H2_captrans_captrans\":{$C,\"H\":2,\"obj_mode\":\"cumspill\",\"captrans\":true,\"objective\":\"captrans\"}}"
done
# e. (time permitting) w20 H=2 and H=4 at 320 evaluations + GD, current objective, fetch (fl) vs OT pool (the matched-time question left open by the OT-pool test)
G20="letter_O_w20 letter_T_w20 letter_S_w20 letter_X_w20"
run e_H2_E320 "$G20" "{\"ct_e_fl_H2_E320\":{$C,\"H\":2,\"obj_mode\":\"cumspill\",\"fetch\":true,\"pool_mode\":\"fl\",\"n_pool\":64,\"pop\":64,\"n_extra\":16}, \"ct_e_ot_H2_E320\":{$C,\"H\":2,\"obj_mode\":\"cumspill\",\"fetch\":true,\"ot\":true,\"pool_mode\":\"ot\",\"n_pool\":64,\"pop\":64}}"
run e_H4_E320 "$G20" "{\"ct_e_fl_H4_E320\":{$C,\"H\":4,\"obj_mode\":\"cumspill\",\"fetch\":true,\"pool_mode\":\"fl\",\"n_pool\":64,\"pop\":64,\"n_extra\":16}, \"ct_e_ot_H4_E320\":{$C,\"H\":4,\"obj_mode\":\"cumspill\",\"fetch\":true,\"ot\":true,\"pool_mode\":\"ot\",\"n_pool\":64,\"pop\":64}}"
echo "$(date) DONE" > $R/queue.done
