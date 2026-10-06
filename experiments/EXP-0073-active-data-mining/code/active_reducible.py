"""PILOT (plan v2, item 1): de-noised target and REDUCIBLE error from the ceiling agent's re-sim artifacts (artifacts/ceiling/resim_<lvl>mm_r<k>.pt,
NOT re-simulated here) + the recorded truth. Replicates used = recorded outcome + every re-sim at level <= LMAX mm (default 0.25: below the
1 mm/px raster). Per row (window frame, swept region):
  n_i   = sum_R unbiased pixel variance across the K replicates            (noise floor)
  M_i   = mean replicate window                                            (de-noised target)
  e_raw = sum_R (P - recorded)^2 ; e_mean = sum_R (P - M)^2
  e_red = e_mean - n_i/K  (unbiased estimate of ||P - E[outcome]||^2)      (REDUCIBLE error)
Then rank-correlates every proxy with e_raw / n_i / e_red. Saves runs/active_pilot/reducible_rows.npz.
"""
import sys, glob, json, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code")
from active_common import *
from active_proxies import load_rows
from scipy.stats import spearmanr, rankdata
LMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
A = "experiments/EXP-0072-zoom-window-nfd/artifacts/ceiling/"
Dd = load_rows(); S, S_, P0, P1 = Dd["S"], Dd["S_"], Dd["P0"], Dd["P1"]; B = len(S)
perm = np.load(AP + 'chain_perm.npy')     # ceiling chain row j <-> my row perm[j] (matched on outcomes; the ceiling sorts rows differently)
assert len(perm) == 896
reps = {}
for f in sorted(glob.glob(A + "resim_*.pt")):
    r = torch.load(f, map_location="cpu", weights_only=False)
    if r["level_mm"] > LMAX: continue
    st = torch.zeros(B, 20, 7); st[perm] = r["onestep"].float(); st[896:] = r["pools"].float(); reps[f.split("resim_")[1][:-3]] = st
for k, v in reps.items():
    dd = (v[..., :2] - S_[..., :2]).norm(dim=-1) * 1000
    print(f"replicate {k:10s} mean cube diff to recorded {dd.mean():.3f} mm; moved-cube mean {dd[(S_[..., :2]-S[..., :2]).norm(dim=-1)*1000 > .5].mean():.3f}; rows>1mm {(dd.max(1).values>1).float().mean():.2f}")
Rw = torch.stack([swept_region_window(P0[b], P1[b], spec) for b in range(B)]).float()
Yw = window_batch(S_, SIZES, P0, P1, spec); Xw = window_batch(S, SIZES, P0, P1, spec)
W = torch.stack([Yw] + [window_batch(v, SIZES, P0, P1, spec) for v in reps.values()]); K = len(W); print("K replicates (incl. recorded) =", K, list(reps))
n_i = (W.var(0, unbiased=True) * Rw).sum((1, 2)).numpy(); M = W.mean(0)
Z = np.load(AP + "proxy_rows.npz", allow_pickle=True)
x = torch.cat([Xw[:, None], plates_batch(P0, P1, spec)], 1)
res = {}
def model_errs(P):
    e_raw = (((P - Yw) ** 2) * Rw).sum((1, 2)).numpy(); e_mean = (((P - M) ** 2) * Rw).sum((1, 2)).numpy(); return e_raw, e_mean, e_mean - n_i / K
nets = {n: load_net(rp(p)) for n, p in dict(ft100="ft100/unet_best.pth", scratch100="scratch100/unet_best.pth", s1="active_pilot/lc_scratch_full_seed1/unet_best.pth", s2="active_pilot/lc_scratch_full_seed3/unet_best.pth", ft300="ft300/unet_best.pth").items() if os.path.exists(rp(p))}
P = {n: predict(net, x) for n, net in nets.items()}
e_raw, e_mean, e_red = model_errs(P["ft100"])
print(f"ft100: mean e_raw {e_raw.mean():.3f}  e_mean {e_mean.mean():.3f}  noise floor n_i {n_i.mean():.3f}  n_i/K {n_i.mean()/K:.3f}  e_red {e_red.mean():.3f}")
q = np.quantile(e_raw, .8); top = e_raw >= q
print(f"top-20% raw-error rows: e_raw {e_raw[top].mean():.2f} n_i {n_i[top].mean():.2f} e_red {e_red[top].mean():.2f}  | rest: e_raw {e_raw[~top].mean():.3f} n_i {n_i[~top].mean():.3f} e_red {e_red[~top].mean():.3f}")
print("share of TOTAL e_red held by top-20pct rows by e_red: %.2f ; by e_raw %.2f" % (np.sort(e_red.clip(0))[::-1][:B // 5].sum() / e_red.clip(0).sum(), e_red.clip(0)[top].sum() / e_red.clip(0).sum()))
print("total: sum n_i/sum e_raw = %.2f  (noise share of raw error)  ; sum e_red/sum e_raw = %.2f" % (n_i.sum() / e_raw.sum(), e_red.sum() / e_raw.sum()))
print("rho(e_raw, n_i) %.3f   rho(e_red, n_i) %.3f   rho(e_raw, e_red) %.3f" % (spearmanr(e_raw, n_i)[0], spearmanr(e_red, n_i)[0], spearmanr(e_raw, e_red)[0]))
# proxies
prox = {k[6:]: Z[k] for k in Z.files if k.startswith("proxy_")}
if len(P) >= 3:                                   # independent-seed ensemble of full-data scratch models (init-independent)
    Pi = torch.stack([P[n] for n in ("scratch100", "s1", "s2") if n in P])
    prox["indep3_var_sw"] = (Pi.var(0) * Rw).sum((1, 2)).numpy(); prox["indep3_var_win"] = Pi.var(0).sum((1, 2)).numpy()
    g = (Pi * (1 - Pi) * Rw).sum((2, 3)).mean(0).numpy(); prox["indep3_gray_sw"] = g
    prox["indep3_epi_ratio"] = prox["indep3_var_sw"] / (g + 0.5)
    err_ens = (((Pi.mean(0) - M) ** 2) * Rw).sum((1, 2)).numpy() - n_i / K; print("indep-3 ensemble-mean: mean e_red %.3f (single scratch100: %.3f)" % (err_ens.mean(), model_errs(P['scratch100'])[2].mean()))
out = {}; pm = prox["pred_mass"]
def resid(y, xx): ry, rx = rankdata(y), rankdata(xx); b = np.polyfit(rx, ry, 1); return ry - np.polyval(b, rx)
print(f"{'proxy':26s} rho(e_raw) rho(noise) rho(e_red) | partial(e_red|noise) lift_red")
for k, p in sorted(prox.items()):
    if np.std(p) == 0: continue
    r1, r2, r3 = spearmanr(p, e_raw)[0], spearmanr(p, n_i)[0], spearmanr(p, e_red)[0]; r4 = spearmanr(resid(p, n_i), resid(e_red, n_i))[0]
    kk = int(B * .1); lift = float(e_red.clip(0)[np.argsort(-p)[:kk]].sum() / e_red.clip(0).sum() / .1)
    out[k] = dict(rho_raw=r1, rho_noise=r2, rho_red=r3, partial_red_given_noise=r4, lift_red_top10=lift); print(f"{k:26s} {r1:+.3f}    {r2:+.3f}    {r3:+.3f}   | {r4:+.3f}   {lift:.2f}")
json.dump(dict(K=K, replicates=list(reps), summary=dict(e_raw=float(e_raw.mean()), e_red=float(e_red.mean()), noise=float(n_i.mean())), proxies=out), open("experiments/EXP-0073-active-data-mining/results/active_reducible.json", "w"), indent=1)
np.savez_compressed(AP + "reducible_rows.npz", e_raw=e_raw, e_mean=e_mean, e_red=e_red, n_i=n_i, K=K, **{f"P_{n}": 0 for n in ()})
