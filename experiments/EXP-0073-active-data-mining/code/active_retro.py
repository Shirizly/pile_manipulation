"""PILOT (plan v2, item 2/4/5): RETROSPECTIVE active learning on DS-0015 at zero sim cost.
seed set = 2000 random DS-0015 train rows (fixed, seed 0); reservoir = the other 8653 train rows (outcomes known, treated as unsimulated).
A scoring ensemble (4 members trained on the seed set only: 2 pretrained-init ft + 2 scratch, different seeds) scores the reservoir; each ARM adds
500 rows chosen by its rule; every arm model = pretrained-init fine-tune, 150 epochs, same recipe, retrain seeds 0/1/2.
Arms: base (no add), rand (500, fresh draw per seed), rand1000, ens (ensemble variance, swept), gray (members' own p(1-p): aleatoric tracker),
epi (variance/(gray+0.5)), dec (decision relevance: largest predicted dv improvement under any of 13 goals = 'attractive to a planner'),
div (farthest-point diversity vs seed set + chosen), oracle (largest TRUE error of the ensemble mean: labels used, ceiling for error mining).
Eval: DS-0017 (val_pools_v2; selection decisions) and DS-0016 (chains+pools; reported for noise-floor stratification only, no decision uses it).
Resumable: every trained model / eval is cached under runs/active_pilot/retro/.   python -u active_retro.py [stage]
"""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code")
from active_common import *
from active_proxies import load_rows
RR = AP + "retro/"; os.makedirs(RR, exist_ok=True)
INIT = "Baselines/NFD/runs/nfd_warped_randlen_flipaug/unet_best.pth"
NSEED, NADD, EP = 2000, 500, 150
T = train_rows(); C = torch.load(R + "window_cache.pt"); Xtr, Ytr, Xva, Yva = C["train"]["x"], C["train"]["y"], C["val"]["x"], C["val"]["y"]
N = len(Xtr); g = np.random.default_rng(0); perm = g.permutation(N); seed_ix = np.sort(perm[:NSEED]); res_ix = np.sort(perm[NSEED:])
Rtr = torch.stack([swept_region_window(T["train"]["P0"][b], T["train"]["P1"][b], spec) for b in range(N)]).float()


def members():
    out = []
    for n, (init, seed, ep) in dict(m_ft0=(INIT, 10, 200), m_ft1=(INIT, 11, 200), m_sc0=(None, 12, 250), m_sc1=(None, 13, 250)).items():
        p = train_unet(Xtr[seed_ix], Ytr[seed_ix], Xva, Yva, ep, seed, RR + n, init=init); out.append(load_net(p))
    return out


def scores():
    f = RR + "scores.npz"
    if os.path.exists(f): return dict(np.load(f))
    ms = members(); x = Xtr[res_ix]; P = torch.stack([predict(m, x) for m in ms]); R = Rtr[res_ix]
    S = {}; S["ens"] = (P.var(0) * R).sum((1, 2)).numpy(); S["gray"] = (P * (1 - P) * R).sum((2, 3)).mean(0).numpy(); S["epi"] = S["ens"] / (S["gray"] + 0.5)
    Pm = P.mean(0); Y = Ytr[res_ix].float(); S["oracle"] = (((Pm - Y) ** 2) * R).sum((1, 2)).numpy(); S["err_each"] = np.stack([(((P[i] - Y) ** 2) * R).sum((1, 2)).numpy() for i in range(len(ms))])
    # decision relevance: best predicted dv improvement over the 13 goals (window frame, mass-normalised lyapunov)
    Dist = {gl: torch.from_numpy(dist_field_from_mask(goal_mask(gl))).float() for gl in GOALS}; dec = np.zeros(len(res_ix)); Xw = x[:, 0].float()
    for i0 in range(0, len(res_ix), 256):
        ii = res_ix[i0:i0 + 256]
        Dw = torch.stack([torch.stack([sample_world_field_into_window(Dist[gl], T["train"]["P0"][b], T["train"]["P1"][b], spec) for gl in GOALS]) for b in ii])
        sl = slice(i0, i0 + len(ii)); vp = EvalSet.lw(Pm[sl], Dw) - EvalSet.lw(Xw[sl], Dw); dec[sl] = (-vp).max(1).values.numpy()
    S["dec"] = dec
    # diversity: greedy farthest point (pooled window feature) w.r.t. seed set
    pool = lambda t: torch.nn.functional.avg_pool2d(t.float(), 4).flatten(1).to(DEV); A, B_ = pool(Xtr[res_ix]), pool(Xtr[seed_ix])
    md = torch.cdist(A, B_).min(1).values; chosen = []
    for _ in range(NADD):
        j = int(md.argmax()); chosen.append(j); md = torch.minimum(md, torch.cdist(A, A[j:j + 1]).squeeze(1))
    S["div_sel"] = np.array(chosen); np.savez(f, **S); return S


def select(arm, S, seed):
    r = np.random.default_rng(1000 + seed)
    if arm == "base": return np.array([], int)
    if arm == "rand": return r.choice(len(res_ix), NADD, replace=False)
    if arm == "rand1000": return r.choice(len(res_ix), 2 * NADD, replace=False)
    if arm == "div": return S["div_sel"]
    return np.argsort(-S[arm])[:NADD]


def main():
    S = scores(); print("scores ready; corr of arm scores with oracle error:", {k: round(float(__import__("scipy.stats").stats.spearmanr(S[k], S["oracle"])[0]), 3) for k in ("ens", "gray", "epi", "dec")}, flush=True)
    # eval sets
    sets = {}
    sets["ds17"] = eval_set_pools(D + "val_pools_v2/pools_0.pt", "ds17")
    Dd = load_rows(); sets["ds16"] = EvalSet(Dd["S"], Dd["S_"], Dd["P0"], Dd["P1"], np.where(Dd["pool"] >= 0, Dd["pool"], -1), "ds16")
    rd = np.load(AP + "reducible_rows.npz") if os.path.exists(AP + "reducible_rows.npz") else None
    arms = ["base", "rand", "rand1000", "ens", "gray", "epi", "dec", "div", "oracle"]
    for seed in (0, 1, 2):
        for arm in arms:
            tag = f"{arm}_s{seed}"; fj = RR + f"eval_{tag}.npz"
            if os.path.exists(fj): continue
            t0 = time.time(); add = res_ix[select(arm, S, seed)]; ix = np.concatenate([seed_ix, add])
            p = train_unet(Xtr[ix], Ytr[ix], Xva, Yva, EP, 100 + seed, RR + tag, init=INIT); net = load_net(p); out = {}
            for nm, es in sets.items():
                Pm = predict(net, es.x); r = es.evaluate(Pm); out[nm + "_slateN"] = r["slateN"]; out[nm + "_acc1"] = r["acc1"]; out[nm + "_e_sw"] = r["e_sw"]
            out["n_rows"] = len(ix); np.savez(fj, **out)
            print(tag, f"rows {len(ix)} ds17 slateN {out['ds17_slateN']:.3f} acc1 {out['ds17_acc1']:.3f} e_sw {out['ds17_e_sw'].mean():.3f} | ds16 slateN {out['ds16_slateN']:.3f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
