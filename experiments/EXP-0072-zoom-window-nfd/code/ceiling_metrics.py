"""Ceiling study part 2: alternative accuracy-style metrics vs slateN, across the 7 cached models (pasted world-64 frame),
and the same metrics for a perfect-physics predictor under simulator noise (resim ceiling).
Every metric is a skill 1 - sqrt(sum num / sum den) or 1 - sum num/sum den (persistence = 0, perfect = 1) built from per-row (num, den)
so it can be pooled over any subset of rows (all rows, one pool, a bootstrap resample).
Writes results/ceiling/metrics_vs_slate.json."""
import json, os, sys, glob, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code")
from ceiling_lib import *
from scipy.stats import kendalltau, spearmanr
ART = "experiments/EXP-0072-zoom-window-nfd/artifacts/ceiling/"
CORE = ["nfd64", "scratch100", "ft100", "ft300", "zoom128", "world128", "world128_300"]
EXTRA = ["nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2", "nfd_3ch_narrow_l20_v2_sharp_w03", "nfd_3ch_narrow_l20_v2_sharp_w07",
         "nfd_3ch_narrow_l20_v2_soft_s1", "nfd_3ch_narrow_l20_v2_soft_s2", "linear_narrow_l20_v2_res64", "linear_narrow_l20_v2_res32"]
MODELS = CORE + EXTRA
c = load_chains(); S0c, S1c = c["states"].float(), c["states_"].float(); actc = torch.cat([c["p_starts"][:, :2], c["p_stops"][:, :2]], 1).float()
S0p, S1p, PP0, PP1 = load_pools(); actp = torch.cat([torch.from_numpy(PP0), torch.from_numpy(PP1)], 1).float()
S0pr = S0p.repeat_interleave(64, 0)
T = {"chains": (occ_from_particles(S1c).cpu(), occ_from_particles(S0c).cpu(), swept_region(actc, "cpu"), occ_for_scoring(S1c[:, :, :3]), occ_for_scoring(S0c[:, :, :3])),
     "pools": (occ_from_particles(S1p).cpu(), occ_from_particles(S0pr).cpu(), swept_region(actp, "cpu"), occ_for_scoring(S1p[:, :, :3]), occ_for_scoring(S0pr[:, :, :3]))}
XY = torch.stack(torch.meshgrid(torch.linspace(-.064, .064, 64), torch.linspace(-.064, .064, 64), indexing="ij"), -1)   # metres, (64,64,2)
DIRS = torch.tensor([[np.cos(a), np.sin(a)] for a in np.linspace(0, np.pi, 8, endpoint=False)]).float()
GM = {g: goal_mask(g) for g in GOALS}
GMT = {g: torch.from_numpy(np.asarray(GM[g])).float() for g in GOALS}


def blur(x, s):
    k = torch.arange(-int(3 * s) - 1, int(3 * s) + 2).float(); k = torch.exp(-k ** 2 / (2 * s * s)); k /= k.sum(); r = len(k) // 2
    x = torch.nn.functional.pad(x[:, None], (r, r, r, r), mode="reflect")
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1)); x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1)); return x[:, 0]


def sw(A, B, R):
    """sliced 1-D Wasserstein/CDF-L1 between mass images A,B restricted to region R (unnormalised: includes mass mismatch), per row, metres*mass."""
    out = torch.zeros(len(A))
    for d in DIRS:
        proj = (XY @ d).flatten(); order = proj.argsort(); ps = proj[order]; w = torch.diff(ps, append=ps[-1:])
        ca = (A * R).flatten(1)[:, order].cumsum(1); cb = (B * R).flatten(1)[:, order].cumsum(1)
        out += ((ca - cb).abs() * w).sum(1)
    return out / len(DIRS)


def centroid(A, R):
    m = (A * R).flatten(1); s = m.sum(1, keepdim=True).clamp_min(1e-6); return (m @ XY.reshape(-1, 2)) / s


def row_terms(P, which):
    """per-row dict metric -> (kind, num, den). P: (n,64,64) predicted next raster (hard-style, same as occ_from_particles / pasted)."""
    Tr, O, R, Ts, T0s = T[which]; Rf = R.float(); n = len(P); out = {}
    sq = lambda a, b: (((a - b) ** 2) * Rf).flatten(1).sum(1)
    out["acc_hard"] = ("rms", sq(P, Tr), sq(O, Tr))
    for s in (1, 2, 4):
        Pb, Tb, Ob = blur(P, s), blur(Tr, s), blur(O, s); out[f"acc_blur{s}"] = ("rms", sq(Pb, Tb), sq(Ob, Tb))
    out["acc_hard_fullframe"] = ("rms", ((P - Tr) ** 2).flatten(1).sum(1), ((O - Tr) ** 2).flatten(1).sum(1))
    # changed-cell IoU (skill vs persistence = 0 changed cells predicted): iou in [0,1], reported as is
    cp, ct = ((P - O).abs() > .5) & R, ((Tr - O).abs() > .5) & R
    out["changed_cell_IoU"] = ("iou", (cp & ct).flatten(1).sum(1).float(), (cp | ct).flatten(1).sum(1).float())
    mr = lambda A: (A * Rf).flatten(1).sum(1)
    out["mass_in_region"] = ("lin_abs", (mr(P) - mr(Tr)).abs(), (mr(O) - mr(Tr)).abs())
    out["region_centroid_mm"] = ("lin", 1000 * (centroid(P, Rf) - centroid(Tr, Rf)).norm(dim=1), 1000 * (centroid(O, Rf) - centroid(Tr, Rf)).norm(dim=1))
    out["sliced_EMD_region"] = ("lin", sw(P, Tr, Rf), sw(O, Tr, Rf))
    # goal metrics on soft truth: dv error and in-goal mass error
    dvt = {}; num_dv = torch.zeros(n); den_dv = torch.zeros(n); num_gm = torch.zeros(n); den_gm = torch.zeros(n); sgn = torch.zeros(n)
    for g in GOALS:
        vt = lyap(Ts, DIST[g]) - lyap(T0s, DIST[g]); vp = lyap(P, DIST[g]) - lyap(O, DIST[g]); vo = torch.zeros(n)
        num_dv += (vp - vt) ** 2; den_dv += (vo - vt) ** 2; sgn += (vp - vt)
        gm = lambda A: (A * GMT[g]).flatten(1).sum(1) / A.flatten(1).sum(1).clamp_min(1e-6)
        mt = gm(Ts) - gm(T0s); mp = gm(P) - gm(O); num_gm += (mp - mt) ** 2; den_gm += mt ** 2
    out["dv_err_allgoals"] = ("rms", num_dv, den_dv); out["in_goal_mass_err"] = ("rms", num_gm, den_gm)
    out["abs_dv_bias(optimism)"] = ("lin_abs", sgn.abs(), den_dv.sqrt())
    return out


def pool(kind, num, den, idx=None):
    if idx is not None: num, den = num[idx], den[idx]
    if kind == "rms": return float(1 - torch.sqrt(num.sum() / den.sum().clamp_min(1e-12)))
    if kind == "iou": return float(num.sum() / den.sum().clamp_min(1e-12))
    if kind == "lin": return float(1 - num.mean() / den.mean().clamp_min(1e-12))
    if kind == "lin_abs": return float(1 - num.mean() / den.mean().clamp_min(1e-12))


def main():
    res = {"models": {}, "ceiling": {}}; terms = {}
    slate = {}; caps = {}
    for m in MODELS:
        d = torch.load(ART + f"preds_{m}.pt"); terms[m] = {w: row_terms(d[w].float(), w) for w in ("chains", "pools")}
        cp = pool_caps(d["pools"].float(), S0p, S1p); caps[m] = cp; slate[m] = float(np.nanmean(np.nanmean(cp, 0)))
        res["models"][m] = {"slateN": slate[m], "metrics_chains": {k: pool(*v) for k, v in terms[m]["chains"].items()}, "metrics_pools": {k: pool(*v) for k, v in terms[m]["pools"].items()}}
        print(m, "slateN", round(slate[m], 3), "acc_hard chains", round(res["models"][m]["metrics_chains"]["acc_hard"], 3), flush=True)
        json.dump(res, open(OUT + "metrics_vs_slate.json", "w"), indent=1)
    res["correlations_model_level"] = {}; res["correlations_per_pool"] = {}
    names = list(terms[MODELS[0]]["pools"].keys()); cm = {m: np.nanmean(caps[m], 1) for m in MODELS}
    for sname, ms in (("core7", CORE), ("all15", MODELS)):
        cor = {}; sl = np.array([slate[m] for m in ms])
        for which in ("chains", "pools"):
            for k in names:
                v = np.array([pool(*terms[m][which][k]) for m in ms]); cor[f"{which}:{k}"] = {"kendall": float(kendalltau(v, sl)[0]), "spearman": float(spearmanr(v, sl)[0])}
        rng = np.random.default_rng(0); boot = {k: [] for k in names}
        for b_ in range(200):
            pi = rng.integers(0, 32, 32); rows = torch.from_numpy((pi[:, None] * 64 + np.arange(64)).reshape(-1)); slb = np.array([np.nanmean(cm[m][pi]) for m in ms])
            for k in names:
                v = np.array([pool(*terms[m]["pools"][k], idx=rows) for m in ms]); boot[k].append(kendalltau(v, slb)[0])
        for k in names: a_ = np.array(boot[k]); cor[f"pools:{k}"].update({"boot_tau_mean": float(np.nanmean(a_)), "boot_tau_sd": float(np.nanstd(a_)), "boot_tau_p05": float(np.nanpercentile(a_, 5))})
        pp = {}
        for k in names:
            V = np.array([[pool(*terms[m]["pools"][k], idx=torch.arange(p * 64, (p + 1) * 64)) for p in range(32)] for m in ms]); C = np.array([cm[m] for m in ms])
            rs = [spearmanr(V[i][~np.isnan(C[i])], C[i][~np.isnan(C[i])])[0] for i in range(len(ms))]
            ok = ~np.isnan(C).any(0); Vd = V - V[:, ok].mean(0); Cd = C - C[:, ok].mean(0)
            pp[k] = {"mean_within_model_spearman_over_pools": float(np.nanmean(rs)), "within_pool_across_models_spearman": float(spearmanr(Vd[:, ok].ravel(), Cd[:, ok].ravel())[0])}
        res["correlations_model_level"][sname] = cor; res["correlations_per_pool"][sname] = pp
    json.dump(res, open(OUT + "metrics_vs_slate.json", "w"), indent=1)
    # ---- ceilings: perfect-physics predictor under noise, as w64_native raster and as paste_zoom64 raster
    for f in sorted(glob.glob(ART + "resim_*.pt")):
        tag = os.path.basename(f)[6:-3]; d = torch.load(f, weights_only=False); res["ceiling"][tag] = {}
        for frame in ("w64_native", "paste_zoom64"):
            ent = {}
            for which, Sq, S0q, p0, p1 in (("chains", d["onestep"], S0c, c["p_starts"][:, :2].numpy(), c["p_stops"][:, :2].numpy()), ("pools", d["pools"], S0pr, PP0, PP1)):
                if frame == "w64_native": P = occ_from_particles(Sq).cpu()
                else:
                    sp = WindowSpec(res=64); Pw = window_batch(Sq, SIZES, p0, p1, sp); Xw = window_batch(S0q, SIZES, p0, p1, sp); O = T[which][1]
                    P = torch.stack([paste_delta(Pw[b], Xw[b], O[b], p0[b], p1[b], sp) for b in range(len(Sq))])
                ent[which] = {k: pool(*v) for k, v in row_terms(P, which).items()}
                if which == "pools": cp = pool_caps(P, S0p, S1p); ent["slateN"] = float(np.nanmean(np.nanmean(cp, 0)))
            res["ceiling"][tag][frame] = ent
        print("ceiling", tag, {fr: round(res["ceiling"][tag][fr]["chains"]["acc_hard"], 3) for fr in res["ceiling"][tag]}, flush=True)
        json.dump(res, open(OUT + "metrics_vs_slate.json", "w"), indent=1)

if __name__ == "__main__": main()
