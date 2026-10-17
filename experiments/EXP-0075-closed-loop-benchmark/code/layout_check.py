"""Unit check: LayoutTransport (raw * d_max = metres) vs exact Hungarian on TRUE rasterised visited states (results/captrans w15_H1 / w10_H1 episodes), + timing."""
import sys, json, time, glob, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "experiments/EXP-0043-batched-closed-loop/code"))
from layout_term import LayoutTransport, hungarian_cost
from simple_mpc.adapters import occ_for_scoring
from simple_mpc.value_functions import uniform_layout
from batched_closed_loop import goal_mask
from scipy.stats import spearmanr
import itertools
res = {}; Q = []
CFG = [(30,1),(10,1),(10,2),(8,2),(6,4)]
for W in (15, 10):
    xs, hs, ts = [], [], []
    eps = [e for e in json.load(open(HERE / f"results/captrans/w{W}_H1.json"))["episodes"] if e.get("complete") and e["cell"].endswith("cur_fetch")]
    for e in eps[:16]:
        m = goal_mask(e["goal"]); sl = uniform_layout(m, 20); lt = LayoutTransport(sl, m)
        S = torch.tensor(e["states"]).float()
        for k in range(0, len(S), 2):
            occ = occ_for_scoring(S[k:k + 1, :, :3]).cuda()
            xs.append(float(lt.raw(occ)[0]) * lt.d_max * 1000); hs.append(1000 * hungarian_cost(S[k, :, :2].numpy(), sl))
    xs, hs = np.array(xs), np.array(hs); res[W] = dict(n=len(xs), spearman=float(spearmanr(xs, hs)[0]), pearson=float(np.corrcoef(xs, hs)[0, 1]), ratio_mean=float(xs.mean() / hs.mean()), mean_sink_mm=float(xs.mean()), mean_hung_mm=float(hs.mean()))
    print(W, res[W], flush=True)
lt = LayoutTransport(sl, m); occ = torch.rand(128, 64, 64, device="cuda", requires_grad=True)
for _ in range(2):
    torch.cuda.synchronize(); t = time.time(); y = lt(occ); y.sum().backward(); torch.cuda.synchronize()
print("128-batch fwd+bwd s", time.time() - t)
res["t128"] = time.time() - t; (HERE / "results/layout").mkdir(exist_ok=True); json.dump(res, open(HERE / "results/layout/unit_check.json", "w"), indent=1)
