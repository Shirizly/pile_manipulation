#!/bin/bash
# EXP-0075 G4 oracle pilot (pre-registration LOG.md 2026-10-15 05:45; command = LOG 03:30 entry verbatim).
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=4
CMD="CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u experiments/EXP-0050-oracle-ceiling/code/oracle.py --mode cem --legal --tag g4_pilot_O_w20_s40 --goals letter_O_w20 --starts 40 --steps 10 --n-envs 32 --cem-pop 128 --cem-iters 5 --cem-elite 16 --mass-weight 1.0 --seed-base 0 --out-dir experiments/EXP-0075-closed-loop-benchmark/results/oracle_g4"
rid="EXP-0075/oracle_g4/g4_pilot-$(date +%s)"; t0=$(date +%s)
echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"start\",\"cmd\":\"$CMD\",\"note\":\"EXP-0075 G4 oracle pilot, HEAD $(git rev-parse --short HEAD) + uncommitted (dirty)\"}" >> experiments/COMMANDS.jsonl
echo "$CMD" > experiments/EXP-0075-closed-loop-benchmark/results/oracle_g4/COMMAND.txt
eval $CMD > experiments/EXP-0075-closed-loop-benchmark/logs/oracle_g4_pilot.log 2>&1; rc=$?
echo "{\"ts\":\"$(date +%FT%T)\",\"run_id\":\"$rid\",\"event\":\"end\",\"status\":\"$([ $rc = 0 ] && echo ok || echo failed)\",\"returncode\":$rc,\"seconds\":$(( $(date +%s) - t0 ))}" >> experiments/COMMANDS.jsonl
echo "rc=$rc $(date)" > experiments/EXP-0075-closed-loop-benchmark/results/oracle_g4/pilot.done
