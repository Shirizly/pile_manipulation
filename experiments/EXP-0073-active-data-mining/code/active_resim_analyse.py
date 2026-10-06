"""PILOT: separate epistemic from aleatoric on the 96 re-simulated rows (reads resim_rows.npz + proxy_rows.npz)."""
import sys, json, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code")
from scipy.stats import spearmanr
from active_proxies import load_rows, spec, SIZES
from model.zoom_nfd.window import window_batch, swept_region_window
R = "experiments/EXP-0072-zoom-window-nfd/runs/"
AP = "experiments/EXP-0073-active-data-mining/runs/active_pilot/"
Z = np.load(AP + "proxy_rows.npz", allow_pickle=True); Q = np.load(AP + "resim_rows.npz", allow_pickle=True)
Dd = load_rows(); sel = Q["sel"]; S, S_, P0, P1 = Dd["S"][sel], Dd["S_"][sel], Dd["P0"][sel], Dd["P1"][sel]
Rw = torch.stack([swept_region_window(P0[b], P1[b], spec) for b in range(96)]).float()
Yw = window_batch(S_, SIZES, P0, P1, spec)
W = {k: window_batch(torch.from_numpy(Q[k]), SIZES, P0, P1, spec) for k in ("repeat", "p025_a", "p025_b", "p025_c", "p1")}
mm = lambda A, B: (torch.from_numpy(A[..., :2] - B[..., :2]).norm(dim=-1) * 1000)
rep = mm(Q["repeat"], S_.numpy()); print("exact repeat vs recorded truth: max per-cube xy diff mm  median %.4f  p90 %.4f  max %.3f ; rows with any cube >0.1mm: %d/96" % (rep.max(1).values.median(), rep.max(1).values.quantile(.9), rep.max(), (rep.max(1).values > .1).sum()))
for k in ("p025_a", "p1"):
    d = mm(Q[k], S_.numpy()); print(f"{k} vs recorded truth: median-of-row-max cube diff {d.max(1).values.median():.3f} mm, mean cube diff {d.mean():.3f}, rows with any cube >1mm: {(d.max(1).values > 1).sum()}/96")
V = torch.stack([W[k] for k in ("p025_a", "p025_b", "p025_c")])
alea = (V.var(0, unbiased=True) * Rw).sum((1, 2)).numpy()                        # chaos variance at sub-pixel state noise
efloor = np.mean([(((V[i] - Yw) ** 2) * Rw).sum((1, 2)).numpy() for i in range(3)], 0)   # error of a single noisy re-sim vs truth
e_model = Z["ft100_e_sw"][sel]; e_alt = Z["err_ft300_e_sw"][sel]
cube_sd = torch.stack([torch.from_numpy(Q[k][..., :2]) for k in ("p025_a", "p025_b", "p025_c")]).std(0).norm(dim=-1).mul(1000)   # per-cube spread mm
print("per-cube xy spread across 3 re-sims at 0.25 mm: median-of-row-max %.3f mm; rows with spread>1 mm: %d/96" % (cube_sd.max(1).values.median(), (cube_sd.max(1).values > 1).sum()))
g = np.r_[np.zeros(48, int), np.ones(48, int)]
for name, m in (("random-ish (48)", g == 0), ("top-20%-error (48)", g == 1)):
    print(f"{name}: mean e_model {e_model[m].mean():.2f}  mean aleatoric (chaos var, swept) {alea[m].mean():.2f}  mean single-resim err {efloor[m].mean():.2f}  frac rows with e_model > 2*aleat+0.5: {np.mean(e_model[m] > 2 * alea[m] + 0.5):.2f}")
print("rho(e_model, aleatoric) all %.3f | top group %.3f | random group %.3f" % (spearmanr(e_model, alea)[0], spearmanr(e_model[g == 1], alea[g == 1])[0], spearmanr(e_model[g == 0], alea[g == 0])[0]))
excess = e_model - alea
out = dict(rho_e_aleat=float(spearmanr(e_model, alea)[0]), mean_e_model=float(e_model.mean()), mean_alea=float(alea.mean()))
print("proxy vs [e_model | aleatoric | excess(e_model-aleat)] spearman on the 96 rows:")
for k in [k for k in Z.files if k.startswith("proxy_")]:
    p = Z[k][sel]
    if np.std(p) == 0: continue
    r = (spearmanr(p, e_model)[0], spearmanr(p, alea)[0], spearmanr(p, excess)[0]); out[k] = [round(float(x), 3) for x in r]
    print(f"  {k[6:]:26s} {r[0]:+.2f} {r[1]:+.2f} {r[2]:+.2f}")
print("rho(excess, ft300 error) =", round(spearmanr(excess, e_alt)[0], 3))
json.dump(out, open("experiments/EXP-0073-active-data-mining/results/active_resim.json", "w"), indent=1)
