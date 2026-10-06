"""PILOT: rank-correlate each proxy with realised per-row error (reads runs/active_pilot/proxy_rows.npz)."""
import numpy as np, json
from scipy.stats import spearmanr, rankdata
Z = np.load("experiments/EXP-0073-active-data-mining/runs/active_pilot/proxy_rows.npz", allow_pickle=True)
prox = {k[6:]: Z[k] for k in Z.files if k.startswith("proxy_")}
pool = Z["pool"]; src = Z["src"]
def resid(y, x):          # rank-residual of y on x
    ry, rx = rankdata(y), rankdata(x); b = np.polyfit(rx, ry, 1); return ry - np.polyval(b, rx)
def within(y):            # demean inside each pool state
    y = y.astype(float).copy()
    for p in np.unique(pool[pool >= 0]):
        m = pool == p; y[m] -= y[m].mean()
    return y
def lift(score, err, frac=0.1):
    n = len(score); k = int(n * frac); top = np.argsort(-score)[:k]
    return float(err[top].sum() / err.sum() / frac)       # share of total error in the selected top-10% / 10%
out = {}
for tgt in ("e_sw", "e_chg", "e_win"):
    err = Z[f"ft100_{tgt}"]; res = {}
    pm = prox["pred_mass"]; ispool = pool >= 0
    for k, s in prox.items():
        if np.std(s) == 0: continue
        r_all = spearmanr(s, err)[0]
        r_pool_within = spearmanr(within(s)[ispool], within(err)[ispool])[0]
        r_part = spearmanr(resid(s, pm), resid(err, pm))[0]
        res[k] = dict(rho_all=round(r_all, 3), rho_within_state=round(r_pool_within, 3), rho_partial_given_pred_mass=round(r_part, 3), lift_top10=round(lift(s, err), 2))
    out[tgt] = res
    print(f"== target {tgt}  (n={len(err)}, pool rows {int(ispool.sum())}; lift = share of all error in top-10% by score / 0.10)")
    for k, v in sorted(res.items(), key=lambda kv: -abs(kv[1]["rho_all"])):
        print(f"  {k:26s} rho {v['rho_all']:+.3f}  within-state {v['rho_within_state']:+.3f}  | pred_mass {v['rho_partial_given_pred_mass']:+.3f}  lift {v['lift_top10']:.2f}")
# oracle-ish reference: error vs persistence error (how much truly moved)
print("ref rho(e_sw, e_pers_sw) =", round(spearmanr(Z["ft100_e_sw"], Z["ft100_e_pers_sw"])[0], 3))
# per-model error levels and ensemble mean
for m in ("ft100", "ft300", "scratch100", "world128_300", "ens3mean"):
    key = f"err_{m}_e_sw" if m != "ft100" else "ft100_e_sw"
    key = key if key in Z.files else f"{m}_e_sw"
    print(m, "mean e_sw", round(float(Z[key].mean()), 2))
print("rho(e_sw ft100, ft300) =", round(spearmanr(Z["ft100_e_sw"], Z["err_ft300_e_sw"])[0], 3), " ft100 vs scratch", round(spearmanr(Z["ft100_e_sw"], Z["err_scratch100_e_sw"])[0], 3))
print("top-decile share of error:", round(np.sort(Z["ft100_e_sw"])[::-1][:len(Z["ft100_e_sw"]) // 10].sum() / Z["ft100_e_sw"].sum(), 3))
json.dump(out, open("experiments/EXP-0073-active-data-mining/results/active_proxy_corr.json", "w"), indent=1)
