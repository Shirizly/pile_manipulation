import json, sys, time
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import occ_from_particles
import wide_planner as wp
from intensive import propose_from_occ, legal_first
rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0); S0 = rows.states[(rows.slate_idx == 40).nonzero()[0, 0]].float(); occ0 = occ_from_particles(S0[None]).to(wp.DEV)[0]; r = {}
for n in (500, 2000, 10000, 60000):
    propose_from_occ(occ0, n); torch.cuda.synchronize(); t0 = time.time(); a = propose_from_occ(occ0, n); torch.cuda.synchronize(); r[f"propose_gpu_{n}_s"] = time.time() - t0
    t0 = time.time(); legal_first(a[:min(len(a), 2000)], S0); r[f"legalize_{min(len(a),2000)}_s"] = time.time() - t0
print(r); json.dump(r, open(REPO / "experiments/EXP-0075-closed-loop-benchmark/results/timing_units3.json", "w"))
