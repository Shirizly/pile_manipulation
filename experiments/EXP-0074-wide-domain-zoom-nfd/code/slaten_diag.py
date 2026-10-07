"""slateN diagnosis: zoom128 (z128_f8_ms4) vs vanilla128 (w128_f8_ms4) on the wide Sean test pools + post-hoc fixes.
Phase 1 (GPU, light, cached per shard in artifacts/slaten_diag/<model>_<shard>.pt): pasted predictions for acc1 rows and same-state pools, plus the window footprint W on the 64 grid.
Phase 2 (CPU): diagnostics + fix variants.  Usage: python -u slaten_diag.py [--shards a,b] ; PYTHONPATH=.
Zoom/World classes reimplemented here only to also return the window mask W (rollout math identical to eval_wide)."""
import sys, os, json, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as E
from eval_wide import *  # noqa
from scipy.stats import spearmanr
R0 = "experiments/EXP-0074-wide-domain-zoom-nfd/"; AR = R0 + "artifacts/slaten_diag/"
MODELS = {"zoom": ("z128_f8_ms4", "zoom"), "vanilla": ("w128_f8_ms4", "world")}

def load_model(kind):
    nm, k = MODELS[kind]; net = make_net(R0 + f"runs/{nm}/unet_best.pth", (8, 16, 32)); return Zoom(net, 128) if k == "zoom" else World(net, 128)

@torch.no_grad()
def predict(kind, m, S0, P0, P1, canvas):
    """one push per row -> pasted frame (B,64,64) and window footprint W (B,64,64) in [0,1] (ones for vanilla)."""
    if kind == "vanilla": return m.rollout(S0, P0[:, None], P1[:, None])[0], torch.ones(len(S0), 64, 64)
    pr = m.rollout(S0, P0[:, None], P1[:, None], canvas)[0]; P0d, P1d = P0.to(dev), P1.to(dev); side = side_for((P1d - P0d).norm(dim=-1))
    W = delta_to_world64_v(torch.ones(len(S0), 128, 128, device=dev), P0d, P1d, side).cpu(); return pr, W

def phase1(kind, sh, s, bs=32):
    f = AR + f"{kind}_{sh}.pt"
    if os.path.exists(f): return torch.load(f, weights_only=False)
    m = load_model(kind); t = s["t"]; out = {}
    rows = s["rows"]; P, W, T, O, Rg, Ls = [], [], [], [], [], []
    for i in range(0, len(rows), bs):
        ix = torch.from_numpy(rows[i:i + bs]); S0, S1, P0, P1 = t["S"][ix], t["S_"][ix], t["P0"][ix], t["P1"][ix]
        pr, w = predict(kind, m, S0, P0, P1, canv(sh, t, ix)); P.append(pr); W.append(w); T.append(occ_from_particles(S1)); O.append(occ_from_particles(S0))
        Rg.append(swept_region(torch.cat([P0, P1], 1), "cpu").cpu()); Ls.append((P1 - P0).norm(dim=1) * 1000)
    out["rows"] = dict(P=torch.cat(P), W=torch.cat(W), T=torch.cat(T), O=torch.cat(O), R=torch.cat(Rg), L=torch.cat(Ls))
    pools = []
    for pidx in s["pools"]:
        ix = torch.from_numpy(pidx); S0 = t["S"][ix]; P0, P1 = t["P0"][ix], t["P1"][ix]
        pr, w = predict(kind, m, S0, P0, P1, canv(sh, t, ix[:1]).expand(len(ix), -1, -1))
        pools.append(dict(pr=pr, W=w, o0=occ_from_particles(S0[:1])[0], tr=occ_for_scoring(t["S_"][ix][:, :, :3]), t0=occ_for_scoring(S0[:1][:, :, :3])[0],
                          th=occ_from_particles(t["S_"][ix]), R=swept_region(torch.cat([P0, P1], 1), "cpu").cpu(), L=(P1 - P0).norm(dim=1) * 1000))
    out["pools"] = pools; torch.save(out, f); return out

# ---------------- image fixes: (pr, o0, R, W) -> pr'
def dil(R, k): return (F.max_pool2d(R.float()[:, None], 2 * k + 1, 1, k)[:, 0] > 0).float()
def thr(t): return lambda pr, o0, R, W: o0 + (lambda d: d * (d.abs() >= t))(pr - o0)
def clipR(k): return lambda pr, o0, R, W: o0 + (pr - o0) * (dil(R, k) if k else R.float())
def resc(pr, o0, R, W): return pr * (o0.sum((-1, -2), keepdim=True) / pr.sum((-1, -2), keepdim=True).clamp_min(1e-6))
def bal(pr, o0, R, W):
    d = pr - o0; pos = d.clamp_min(0); neg = (-d).clamp_min(0); ps, ns = pos.sum((-1, -2), keepdim=True), neg.sum((-1, -2), keepdim=True); tg = (ps + ns) / 2
    return o0 + pos * tg / ps.clamp_min(1e-9) - neg * tg / ns.clamp_min(1e-9)
def chain(*fs):
    def g(pr, o0, R, W):
        for f in fs: pr = f(pr, o0, R, W)
        return pr
    return g
IMG = {"base": lambda pr, o0, R, W: pr, "thr.05": thr(.05), "thr.1": thr(.1), "thr.2": thr(.2), "rescale": resc, "balance": bal, "clipR": clipR(0), "clipR+4px": clipR(4),
       "clipR4+thr.1": chain(clipR(4), thr(.1)), "clipR4+thr.1+bal": chain(clipR(4), thr(.1), bal)}

# ---------------- lyap-evaluation variants: (pr_fixed, o0, R, W, Dist, union) -> dv (n,)
def ly(f, D): return (f.reshape(len(f), -1) * D.reshape(1, -1)).sum(1) / f.reshape(len(f), -1).sum(1).clamp_min(1e-6)
def ev_std(pr, o0, R, W, D, U): return ly(pr, D) - ly(o0[None], D)
def ev_union(pr, o0, R, W, D, U): return ly(pr * U, D) - ly((o0 * U)[None], D)
def ev_win(pr, o0, R, W, D, U): return ly(pr * W, D) - ly(o0[None] * W, D)
def ev_fixM(pr, o0, R, W, D, U): return ((pr - o0[None]) * D).sum((1, 2)) / o0.sum()                     # numerator change over FIXED start mass (no renormalisation)
def ev_cent(pr, o0, R, W, D, U): L0 = ly(o0[None], D)[0]; return ((pr - o0[None]) * (D - L0)).sum((1, 2)) / o0.sum()   # same, mass-change invariant (first order lyap change)
EVS = {"std": ev_std, "union_swept": ev_union, "window_only": ev_win, "fixedMass": ev_fixM, "centered": ev_cent}

def pool_scores(pools, Dist, img="base", ev="std", hard=False):
    """returns slateN (mean over goals of mean over pools of capture) + per-(pool,goal) records."""
    caps = {g: [] for g in GOALS}; recs = []
    for p in pools:
        pr = IMG[img](p["pr"], p["o0"][None], p["R"], p["W"]); U = (p["R"].any(0)).float(); tr = p["th"] if hard else p["tr"]; t0 = p["o0"] if hard else p["t0"]
        for g in GOALS:
            D = Dist[g]; vt = (ly(tr, D) - ly(t0[None], D)).numpy(); vp = EVS[ev](pr, p["o0"], p["R"], p["W"], D, U).numpy(); den = vt.mean() - vt.min()
            if den > 1e-9:
                c = float((vt.mean() - vt[vp.argmin()]) / den); caps[g].append(c); recs.append((vt, vp, c))
    return float(np.mean([np.mean(v) for v in caps.values() if v])), recs

def acc_rows(r, img):
    P = IMG[img](r["P"], r["O"], r["R"], r["W"]).clamp(0, 1); return acc(P, r["T"], r["O"], r["R"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shards", default=",".join(SHARDS)); a = ap.parse_args(); shs = a.shards.split(",")
    sets = test_sets(); Dist = {g: torch.from_numpy(dist_field_from_mask(goal_mask(g))).float() for g in GOALS}; OUT = {}
    for sh in shs:
        D = {k: phase1(k, sh, sets[sh]) for k in MODELS}; o = OUT[sh] = {}; print(sh, flush=True)
        for k in MODELS:
            d = D[k]; pools = d["pools"]; r = {}
            # ---- fix variants (img x ev) slateN + acc1
            r["variants"] = {}
            for img in IMG:
                r["variants"][f"img:{img}"] = dict(acc1=acc_rows(d["rows"], img), slateN=pool_scores(pools, Dist, img)[0])
            for ev in EVS:
                if ev == "window_only" and k == "vanilla": continue
                r["variants"][f"ev:{ev}"] = dict(acc1=acc_rows(d["rows"], "base"), slateN=pool_scores(pools, Dist, "base", ev)[0])
            r["variants"]["truth:hard"] = dict(slateN=pool_scores(pools, Dist, "base", "std", hard=True)[0])
            # ---- diagnostics (base)
            _, rec = pool_scores(pools, Dist); sp = [spearmanr(v, t)[0] for t, v, c in rec if np.std(t) > 0 and np.std(v) > 0]
            rk = [float((t < t[v.argmin()]).sum() / max(len(t) - 1, 1)) for t, v, c in rec]   # fraction of pushes truly better than the pick (0 = pick is true best)
            r["diag"] = dict(spearman=float(np.nanmean(sp)), pick_true_rank_frac=float(np.mean(rk)), pick_is_true_best=float(np.mean([t.argmin() == v.argmin() for t, v, c in rec])),
                             capture=float(np.mean([c for *_, c in rec])), n_rec=len(rec))
            # mass
            dM, tM, sM, hM, o0M, mot, Ls = [], [], [], [], [], [], []
            for p in pools:
                dM.append(p["pr"].sum((1, 2)) - p["o0"].sum()); hM.append(p["th"].sum((1, 2)) - p["o0"].sum()); sM.append(p["tr"].sum((1, 2)) - p["t0"].sum()); o0M.append(p["o0"].sum().expand(len(p["pr"])))
                mot.append((p["tr"] - p["t0"]).abs().sum((1, 2)) / p["t0"].sum()); Ls.append(p["L"])
            dM, hM, sM, o0M, mot, Ls = map(torch.cat, (dM, hM, sM, o0M, mot, Ls))
            r["mass"] = dict(pred_dM_over_M0_mean=float((dM / o0M).mean()), pred_dM_abs=float((dM / o0M).abs().mean()), hardtruth_dM_abs=float((hM / o0M).abs().mean()), softtruth_dM_abs=float((sM / o0M).abs().mean()),
                             pred_dM_std_within_pool=float(np.mean([ (p["pr"].sum((1, 2)) / p["o0"].sum()).std() for p in pools])))
            # dv error decomposition per (pool,goal) within-pool centred: err = vp - vt ; in = vp_clipR4 - vt ; out = vp - vp_clipR4 ; and truth spread
            e_all, e_in, e_out, vts, byL = [], [], [], [], {}
            for p in pools:
                prc = IMG["clipR+4px"](p["pr"], p["o0"][None], p["R"], p["W"]); b = np.digitize(p["L"].numpy(), [20, 35, 50])
                for g in GOALS:
                    Dg = Dist[g]; vt = (ly(p["tr"], Dg) - ly(p["t0"][None], Dg)).numpy(); vp = ly(p["pr"], Dg).numpy() - ly(p["o0"][None], Dg).numpy(); vi = ly(prc, Dg).numpy() - ly(p["o0"][None], Dg).numpy()
                    c = lambda x: x - x.mean(); e_all.append(c(vp - vt)); e_in.append(c(vi - vt)); e_out.append(c(vp - vi)); vts.append(c(vt))
                    for j in range(4): byL.setdefault(j, []).append(((vp - vt) - (vp - vt).mean())[b == j])
            rms = lambda L: float(np.sqrt(np.mean(np.concatenate(L) ** 2)))
            r["dv_err"] = dict(rms_true_dv_spread=rms(vts), rms_err=rms(e_all), rms_err_inside_clipR4=rms(e_in), rms_err_outside_swept=rms(e_out),
                               err_centered_mean_by_Lbin_lt20_20_35_35_50_ge50=[float(np.concatenate(byL[j]).mean()) if len(np.concatenate(byL[j])) else None for j in range(4)],
                               n_by_Lbin=[int(sum(len(x) for x in byL[j])) for j in range(4)])
            # null-ish pushes: soft truth motion < 1.5 % of mass
            null = (mot < 0.015); r["null"] = dict(frac_null=float(null.float().mean()), pred_dM_abs_null=float((dM / o0M).abs()[null].mean()) if null.any() else None,
                                                    pred_dM_abs_nonnull=float((dM / o0M).abs()[~null].mean()),
                                                    pred_abs_delta_over_M0_null=float(np.mean([ ((p["pr"] - p["o0"]).abs().sum((1, 2)) / p["o0"].sum())[(((p["tr"] - p["t0"]).abs().sum((1, 2)) / p["t0"].sum()) < .015)].sum().item() for p in pools]) / max(null.float().sum().item() / len(pools), 1e-9)))
            pk_null, bst_null = [], []
            for p in pools:
                mo = ((p["tr"] - p["t0"]).abs().sum((1, 2)) / p["t0"].sum()) < .015
                for g in list(GOALS)[:]:
                    Dg = Dist[g]; vt = (ly(p["tr"], Dg) - ly(p["t0"][None], Dg)).numpy(); vp = (ly(p["pr"], Dg) - ly(p["o0"][None], Dg)).numpy()
                    if vt.mean() - vt.min() > 1e-9: pk_null.append(bool(mo[vp.argmin()])); bst_null.append(bool(mo[vt.argmin()]))
            r["null"]["pick_is_null_frac"] = float(np.mean(pk_null)); r["null"]["true_best_is_null_frac"] = float(np.mean(bst_null))
            o[k] = r
            print(sh, k, "slateN", round(r["variants"]["img:base"]["slateN"], 3), "acc1", round(r["variants"]["img:base"]["acc1"], 3), "spearman", round(r["diag"]["spearman"], 3), flush=True)
        json.dump(OUT, open(R0 + "results/slateN_diagnosis.json", "w"), indent=1)   # atomic enough: rewritten per shard
    # macro means of variants
    M = {k: {v: {m: float(np.mean([OUT[s][k]["variants"][v][m] for s in shs if m in OUT[s][k]["variants"][v]])) for m in ("acc1", "slateN") if any(m in OUT[s][k]["variants"][v] for s in shs)} for v in OUT[shs[0]][k]["variants"]} for k in MODELS}
    OUT["_macro_variants"] = M; json.dump(OUT, open(R0 + "results/slateN_diagnosis.json", "w"), indent=1)

if __name__ == "__main__": main()
