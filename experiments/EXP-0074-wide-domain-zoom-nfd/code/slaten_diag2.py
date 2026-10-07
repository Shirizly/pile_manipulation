"""Follow-up to slaten_diag.py (uses its cached predictions): (1) mass drift vs push length / window size / true motion, how much of dv error is explained by the mass drift,
(2) more mass-conservation variants. Writes results/slateN_diagnosis2.json. Usage: PYTHONPATH=. python -u slaten_diag2.py"""
import sys, json, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import slaten_diag as S
from slaten_diag import *
def balt(mode):
    def g(pr, o0, R, W):
        d = pr - o0; pos = d.clamp_min(0); neg = (-d).clamp_min(0); ps, ns = pos.sum((-1, -2), keepdim=True), neg.sum((-1, -2), keepdim=True)
        tg = {"neg": ns, "pos": ps, "avg": (ps + ns) / 2}[mode]
        return o0 + pos * tg / ps.clamp_min(1e-9) - neg * tg / ns.clamp_min(1e-9)
    return g
S.IMG.update({"bal_toNeg": balt("neg"), "bal_toPos": balt("pos"), "thr.1+bal": chain(thr(.1), bal), "thr.2+bal": chain(thr(.2), bal), "thr.1+bal+rescale": chain(thr(.1), bal, resc)})
sets = test_sets(); Dist = {g: torch.from_numpy(dist_field_from_mask(goal_mask(g))).float() for g in GOALS}; OUT = {}
shs = SHARDS
for sh in shs:
    OUT[sh] = {}
    for k in MODELS:
        d = S.phase1(k, sh, sets[sh]); pools = d["pools"]; r = {}
        for v in ["bal_toNeg", "bal_toPos", "thr.1+bal", "thr.2+bal", "thr.1+bal+rescale"]:
            r[v] = dict(acc1=acc_rows(d["rows"], v), slateN=pool_scores(pools, Dist, v)[0])
        # mass drift by L bin and by window area, rows
        rw = d["rows"]; dM = ((rw["P"].sum((1, 2)) - rw["O"].sum((1, 2))) / rw["O"].sum((1, 2))); L = rw["L"]; hM = ((rw["T"].sum((1, 2)) - rw["O"].sum((1, 2))) / rw["O"].sum((1, 2)))
        b = np.digitize(L.numpy(), [20, 35, 50]); r["dM_over_M0_by_Lbin"] = [float(dM[b == j].mean()) if (b == j).any() else None for j in range(4)]; r["hardtruth_dM_by_Lbin"] = [float(hM[b == j].mean()) if (b == j).any() else None for j in range(4)]
        r["dM_L_corr"] = float(np.corrcoef(dM.numpy(), L.numpy())[0, 1]); r["dM_over_M0_rows_mean"] = float(dM.mean())
        mot = (rw["T"] - rw["O"]).abs().sum((1, 2)) / rw["O"].sum((1, 2)); q = torch.quantile(mot, torch.tensor([.1, .5])); lo = mot <= q[0]
        r["lowest10pct_motion"] = dict(true_motion_over_M0=float(mot[lo].mean()), pred_dM_over_M0=float(dM[lo].mean()), pred_abs_delta_over_M0=float(((rw["P"] - rw["O"]).abs().sum((1, 2)) / rw["O"].sum((1, 2)))[lo].mean()))
        # pools: within-pool share of dv-error variance explained by mass drift (regress err on dM within pool/goal), corr(vp-pool-centred, dM), and slateN if dM known exactly (oracle-removal: vp corrected by -slope*dM)
        ex, tot, caps_corr = [], [], []
        for p in pools:
            dm = (p["pr"].sum((1, 2)) - p["o0"].sum()) / p["o0"].sum(); dm = (dm - dm.mean()).numpy()
            for g in GOALS:
                D = Dist[g]; vt = (ly(p["tr"], D) - ly(p["t0"][None], D)).numpy(); vp = (ly(p["pr"], D) - ly(p["o0"][None], D)).numpy(); e = (vp - vt) - (vp - vt).mean()
                if dm.std() > 0: sl = (e * dm).sum() / (dm ** 2).sum(); ex.append(((sl * dm) ** 2).sum()); tot.append((e ** 2).sum())
        r["frac_dv_err_var_explained_by_dM"] = float(np.sum(ex) / np.sum(tot))
        OUT[sh][k] = r; print(sh, k, {a: (round(b["slateN"], 3), round(b["acc1"], 3)) for a, b in r.items() if isinstance(b, dict) and "slateN" in b}, "dM/L", [None if x is None else round(x, 3) for x in r["dM_over_M0_by_Lbin"]], "R2dM", round(r["frac_dv_err_var_explained_by_dM"], 3), flush=True)
        json.dump(OUT, open("experiments/EXP-0074-wide-domain-zoom-nfd/results/slateN_diagnosis2.json", "w"), indent=1)
