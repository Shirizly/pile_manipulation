"""replay plan files in the simulator (open loop, legalised), 32 per batch; saves states/actions (states_<tag>.npz) and replay_<tag>.json.
usage: bs_replay.py <glob under results/budget_study/plans> <tag prefix> [shard nshards]"""
import sys, glob, os
from bs_lib import *
pat, pre = sys.argv[1], sys.argv[2]; files = sorted(glob.glob(str(BS / "plans" / pat))); items = [json.load(open(f)) for f in files]
shard, nsh = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
sim = Sim(32)
order = list(enumerate(range(0, len(items), 32)))
if os.environ.get('REV'): order = order[::-1]
for b, b0 in order:
    if b % nsh != shard or (BS / f"replay_{pre}{b:02d}.json").exists(): continue
    t0 = time.time(); out = replay(sim, items[b0:b0 + 32], f"{pre}{b:02d}"); print("batch", b, len(out), f"{time.time() - t0:.0f}s", flush=True)
sim.close()
