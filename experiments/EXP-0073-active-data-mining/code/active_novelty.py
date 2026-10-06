"""PILOT: novelty proxy = distance of a test row's window input (state raster + 2 plate channels, avg-pooled to 16x16) to its k-th
nearest DS-0015 train window (runs/window_cache.pt). Does error track coverage (-> data helps) or not (-> noise/capacity)?"""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code")
import numpy as np, torch, json
from scipy.stats import spearmanr, rankdata
from active_proxies import load_rows, build_x, spec
R = "experiments/EXP-0072-zoom-window-nfd/runs/"
AP = "experiments/EXP-0073-active-data-mining/runs/active_pilot/"
Dd = load_rows(); _, x = build_x(Dd["S"], Dd["P0"], Dd["P1"])
tr = torch.load(R + "window_cache.pt")["train"]["x"].float()
pool = lambda t: torch.nn.functional.avg_pool2d(t, 4).flatten(1)
A, T = pool(x).cuda(), pool(tr).cuda()
d = torch.cdist(A, T); kn = d.topk(5, largest=False).values.cpu().numpy()
Z = np.load(AP + "proxy_rows.npz", allow_pickle=True); e = Z["ft100_e_sw"]; sel = np.load(AP + "resim_rows.npz", allow_pickle=True)["sel"]
pm = Z["proxy_pred_mass"]
def res(y, xx): ry, rx = rankdata(y), rankdata(xx); b = np.polyfit(rx, ry, 1); return ry - np.polyval(b, rx)
out = {}
for k in (0, 4):
    nv = kn[:, k]; out[f"nn{k+1}"] = dict(rho_e=float(spearmanr(nv, e)[0]), partial_given_pred_mass=float(spearmanr(res(nv, pm), res(e, pm))[0]))
    print(f"novelty (dist to {k+1}-th NN train window): rho with e_sw {out[f'nn{k+1}']['rho_e']:+.3f}, partial|pred_mass {out[f'nn{k+1}']['partial_given_pred_mass']:+.3f}")
# within pool states (same state, different action) novelty is a pure action effect
pool_i = Z["pool"] >= 0; nv = kn[:, 0]; e2 = e.copy(); n2 = nv.copy()
for p in np.unique(Z["pool"][pool_i]):
    m = Z["pool"] == p; e2[m] -= e[m].mean(); n2[m] -= nv[m].mean()
print("within-state rho(novelty, e_sw) %.3f" % spearmanr(n2[pool_i], e2[pool_i])[0])
q = np.quantile(nv, [.5, .9]); print("mean e_sw by novelty bin  <median %.2f | median-p90 %.2f | >p90 %.2f" % (e[nv < q[0]].mean(), e[(nv >= q[0]) & (nv < q[1])].mean(), e[nv >= q[1]].mean()))
np.save(AP + "novelty_nn.npy", kn); json.dump(out, open("experiments/EXP-0073-active-data-mining/results/active_novelty.json", "w"))
