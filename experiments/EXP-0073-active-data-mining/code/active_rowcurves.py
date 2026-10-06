"""PILOT (plan v2, item 3): per-row learning curves. Same architecture/recipe trained on 25 / 50 / 100 % of DS-0015 (scratch, equal steps; 100 % has 2-3 seeds).
Which DS-0016 rows keep improving with data and which plateau (noise or capacity)? Error = swept-region SE in the zoom window (e_raw) and e_red (vs de-noised mean)."""
import sys, json, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code")
from active_common import *
from active_proxies import load_rows
from scipy.stats import spearmanr
Dd = load_rows(); S, S_, P0, P1 = Dd["S"], Dd["S_"], Dd["P0"], Dd["P1"]; B = len(S)
Xw = window_batch(S, SIZES, P0, P1, spec); Yw = window_batch(S_, SIZES, P0, P1, spec); x = torch.cat([Xw[:, None], plates_batch(P0, P1, spec)], 1)
Rw = torch.stack([swept_region_window(P0[b], P1[b], spec) for b in range(B)]).float()
names = dict(f25="active_pilot/lc_scratch_sub25_400ep", f50="active_pilot/lc_scratch_sub50_200ep", s0="scratch100", s1="active_pilot/lc_scratch_full_seed1", s3="active_pilot/lc_scratch_full_seed3")
E = {}
for k, p in names.items():
    if os.path.exists(rp(p) + "/unet_best.pth"): E[k] = (((predict(load_net(rp(p) + "/unet_best.pth"), x) - Yw) ** 2) * Rw).sum((1, 2)).numpy()
full = [k for k in ("s0", "s1", "s3") if k in E]; e100 = np.mean([E[k] for k in full], 0); seed_sd = np.std([E[k] for k in full], 0, ddof=1)
print("models:", list(E), "mean e_sw:", {k: round(float(v.mean()), 3) for k, v in E.items()})
q = np.quantile(e100, .8); top = e100 >= q
d_tot = E["f25"] - e100; d_half = E["f50"] - e100
thr = 2 * seed_sd + 0.05           # improvement must beat 2 seed-sd of that row (+0.05 absolute floor)
imp = d_tot > thr; plateau = np.abs(d_tot) <= thr; worse = d_tot < -thr
for nm, m in (("top-20% rows", top), ("other rows", ~top), ("all", np.ones(B, bool))):
    print(f"{nm:14s} n={m.sum():4d}  e25 {E['f25'][m].mean():.3f} e50 {E['f50'][m].mean():.3f} e100 {e100[m].mean():.3f}  improve(25->100) {imp[m].mean():.2f}  plateau {plateau[m].mean():.2f}  worse {worse[m].mean():.2f}  share of total 25->100 reduction {d_tot[m].sum()/d_tot.sum():.2f}")
red = np.load(AP + "reducible_rows.npz")
for nm, key in (("n_i (noise floor)", "n_i"), ("e_red", "e_red")):
    print(f"rho(improvement 25->100, {nm}) = {spearmanr(d_tot, red[key])[0]:+.3f}; within top-20%: {spearmanr(d_tot[top], red[key][top])[0]:+.3f}")
from scipy.stats import rankdata; rk = rankdata(red["n_i"]) / B; b = np.digitize(rk, [1 / 3, 2 / 3])
for i, nm in enumerate(("low", "mid", "high")):
    m = (b == i) & top; print(f"top rows, noise-floor {nm}: n={m.sum()} improve {imp[m].mean():.2f} plateau {plateau[m].mean():.2f} e25 {E['f25'][m].mean():.2f} e100 {e100[m].mean():.2f}")
print("seed-to-seed: median row |e_s0 - e_s1| = %.3f ; mean row-sd %.3f" % (np.median(np.abs(E['s0'] - E['s1'])), seed_sd.mean()))
print("rho(e per row, 25 vs 100) = %.3f" % spearmanr(E['f25'], e100)[0])
json.dump(dict(mean={k: float(v.mean()) for k, v in E.items()}, top_improve=float(imp[top].mean()), top_plateau=float(plateau[top].mean()), top_worse=float(worse[top].mean()), all_improve=float(imp.mean()),
               share_top_reduction=float(d_tot[top].sum() / d_tot.sum())), open("experiments/EXP-0073-active-data-mining/results/active_rowcurves.json", "w"), indent=1)
