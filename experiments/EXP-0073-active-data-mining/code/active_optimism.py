"""PILOT (plan v2, item 4): decision relevance. On DS-0016 pools: for every (pool, goal) take the model's top-1 push (lowest predicted window dv).
optimism = predicted dv - simulated dv of that push (negative = predicted better than reality). Compare picked rows with the average row."""
import sys, json, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code")
from active_common import *
from active_proxies import load_rows
from scipy.stats import spearmanr
Dd = load_rows(); es = EvalSet(Dd["S"], Dd["S_"], Dd["P0"], Dd["P1"], np.where(Dd["pool"] >= 0, Dd["pool"], -1), "ds16")
red = np.load(AP + "reducible_rows.npz"); Z = np.load(AP + "proxy_rows.npz", allow_pickle=True)
nets = {n: load_net(rp(p)) for n, p in dict(ft100="ft100/unet_best.pth", scratch100="scratch100/unet_best.pth", s1="active_pilot/lc_scratch_full_seed1/unet_best.pth", s3="active_pilot/lc_scratch_full_seed3/unet_best.pth").items()}
P = {n: predict(m, es.x) for n, m in nets.items()}
out = {}
for n in ("ft100", "scratch100"):
    op = es.optimism(P[n]); a = np.array([(o[0], o[3], o[4]) for o in op]); rows, opt, rk = a[:, 0].astype(int), a[:, 1], a[:, 2]
    e = es.evaluate(P[n])["e_sw"]; allrows = np.concatenate([ix for ix in es.pools.values()])
    print(f"[{n}] picks {len(a)}; mean optimism (pred-true dv) {opt.mean():+.4f} (sd {opt.std():.3f}); frac optimistic (pred<true) {np.mean(opt < 0):.2f}; mean TRUE rank percentile of the picked push {rk.mean():.3f} (0 = truly best; random 0.5)")
    print(f"      error of picked rows: mean e_sw {e[rows].mean():.2f} vs all pool rows {e[allrows].mean():.2f}; top-20%-error share among picked {np.mean(e[rows] >= np.quantile(e, .8)):.2f} (random 0.20)")
    print(f"      rho(optimism, e_sw of picked row) {spearmanr(opt, e[rows])[0]:+.3f}; rho(|optimism|, n_i) {spearmanr(np.abs(opt), red['n_i'][rows])[0]:+.3f}; rho(|optimism|, e_red) {spearmanr(np.abs(opt), red['e_red'][rows])[0]:+.3f}")
    for k in ("ens3_var_sw", "pred_grayness_sw", "grad_all", "pert_inp_shift_sw"):
        print(f"      rho(|optimism|, proxy {k}) {spearmanr(np.abs(opt), Z['proxy_' + k][rows])[0]:+.3f}", end="")
    print(); out[n] = dict(mean_opt=float(opt.mean()), frac_optimistic=float(np.mean(opt < 0)), true_rank=float(rk.mean()), picked_e=float(e[rows].mean()), all_e=float(e[allrows].mean()), picked_top20=float(np.mean(e[rows] >= np.quantile(e, .8))))
# ensemble-picked
Pm = torch.stack(list(P.values())).mean(0); op = es.optimism(Pm); a = np.array([(o[0], o[3], o[4]) for o in op]); print(f"[ens4 mean] mean optimism {a[:,1].mean():+.4f}, frac optimistic {np.mean(a[:,1]<0):.2f}, true rank {a[:,2].mean():.3f}")
json.dump(out, open("experiments/EXP-0073-active-data-mining/results/active_optimism.json", "w"), indent=1)
