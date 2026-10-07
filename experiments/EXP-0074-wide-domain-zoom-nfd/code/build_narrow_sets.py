"""DS-0016 (narrow exact-20 mm clean test) in the eval_wide test-set format, so narrow and wide models are scored by the SAME code:
one 'shard' called `narrow`: rows = all non-null test_chains_v2_clean rows, chains = non-overlapping 4-push segments of the fully valid 8-push chains, pools = test_pools_v2 same-state pools (64 pushes)."""
import sys, glob, numpy as np, torch
sys.path.insert(0, ".")
from model.zoom_nfd.rollout import load_chains
ch = load_chains("Genesis/data/narrow_l20_n20/test_chains_v2_clean/_*_data.pt"); E, T = ch["S"].shape[:2]
S, S_, P0, P1 = (ch[k].reshape(E * T, *ch[k].shape[2:]) for k in ("S", "S_", "P0", "P1"))
chains = [[e * T + s + k for k in range(4)] for e in range(E) for s in (0, 4)]
pools = []; PS, PS_, PP0, PP1 = [], [], [], []; base = len(S)
for f in sorted(glob.glob("Genesis/data/narrow_l20_n20/test_pools_v2/pools_*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    for pi in torch.unique(d["pool_idx"]):
        ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
        PS.append(d["states"][ix].float().cpu()); PS_.append(d["states_"][ix].float().cpu()); PP0.append(d["p_starts"][ix, :2].float().cpu()); PP1.append(d["p_stops"][ix, :2].float().cpu())
        pools.append(np.arange(base, base + len(ix))); base += len(ix)
t = dict(S=torch.cat([S] + PS), S_=torch.cat([S_] + PS_), P0=torch.cat([P0] + PP0), P1=torch.cat([P1] + PP1))
rows = np.arange(len(S))
torch.save({"narrow": dict(t=t, rows=rows, chains=chains, pools=pools[:60])}, "experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/narrow_sets.pt")
print("rows", len(rows), "chains", len(chains), "pools", len(pools), "L mm", float((P1 - P0).norm(dim=1).mean() * 1000))
