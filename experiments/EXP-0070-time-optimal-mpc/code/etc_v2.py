"""EXP-0070 v2 of the time-to-go map, following the user's loop (2026-10-06):
  etc_g(s) = f_g(r(s, g)),  r = a value readout:  'lyap' (true lyapunov V)  or  'gap' (mass gap to completion,
  max(0, theta - mass_rel); 0 exactly at completion, so f(0) = 0 makes the map completion-aware).
  f_g(r) = c_g * r ** p_g  (monotone, f(0) = 0).
Fits:
  mc  -- least squares on Monte-Carlo remaining time of COMPLETED episodes (the session-1 bootstrap);
  td  -- fitted value iteration: target_k = dt_k + f(r_{k+1}) (0 at completion), all episodes incl. censored ones
         (a censored episode's last state is bootstrapped, never terminal); `--weight-new TAG=W` up-weights the
         newest trajectories (the loop's on-policy data).
Model offset: for each readout, b_r(r) = mean(model-predicted change - true change) of executed pushes, in 5
quantile bins of r (the model reads change from its own input image, see RUN-0008), stored with the map so the
planner reads r' = r(s) + (dr_model - b_r(r(s))).
Held-out evaluation (GroupKFold by start state): MC R2 on completed episodes, mean Bellman residual
f(r_k) - dt_k - f(r_{k+1}) on all non-terminal transitions.

  etc_v2.py calib            -> results/etc_v2_calib.json        (model vs true change, both readouts; GPU)
  etc_v2.py fit OUT [--weight-new TAG=W ...] [--data TAG ...]    -> results/OUT.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
GOALS = ["quadrant_0", "quadrant_3", "two_squares", "letter_O", "letter_T", "letter_S", "letter_X"]
EP_FILES = {"retro": "retro_episodes.json", "pilot_v0": "pilot_v0_episodes.json", "pilot_v0b": "pilot_v0b_episodes.json",
            "pilot_v1": "pilot_v1_episodes.json", "pilot_b01": "pilot_b01_episodes.json"}


def readout(r, name):
    if name == "lyap":
        return np.array(r["V"], float)
    return np.maximum(0.0, r["theta"] - np.array(r["mass_rel"], float))


def transitions(rows, name):
    """per episode: r_k, dt_k, terminal flags; yields arrays over all k < last."""
    out = []
    for r in rows:
        x = readout(r, name)
        dt = np.array(r["plan"]) + np.array(r["t_act"]["total"])
        k = r["kstar"]; last = k if k else len(dt)
        for j in range(last):
            out.append((x[j], dt[j], x[j + 1], k is not None and j + 1 == k, r["start"], r.get("_w", 1.0),
                        float(dt[j:last].sum()) if k else np.nan))
    if not out:
        return None
    a = np.array(out, dtype=object)
    return dict(r=a[:, 0].astype(float), dt=a[:, 1].astype(float), r1=a[:, 2].astype(float), term=a[:, 3].astype(bool),
                grp=a[:, 4].astype(int), w=a[:, 5].astype(float), mc=a[:, 6].astype(float))


def fit_power(x, y, w):
    best = None
    for p in np.linspace(0.2, 2.5, 47):
        z = np.clip(x, 1e-6, None) ** p
        c = float((w * z * y).sum() / max((w * z * z).sum(), 1e-12))
        if c <= 0:
            continue
        s = float((w * (y - c * z) ** 2).sum())
        if best is None or s < best[0]:
            best = (s, c, float(p))
    return dict(c=best[1], p=best[2])


f_eval = lambda f, x: f["c"] * np.clip(x, 0, None) ** f["p"]


def fit(T, method):
    if method == "mc":
        m = ~np.isnan(T["mc"])
        return fit_power(T["r"][m], T["mc"][m], T["w"][m]) if m.sum() > 5 else None
    m = ~np.isnan(T["mc"])
    f = fit_power(T["r"][m], T["mc"][m], T["w"][m]) if m.sum() > 5 else dict(c=30.0, p=1.0)   # MC init
    for _ in range(30):
        y = T["dt"] + np.where(T["term"], 0.0, f_eval(f, T["r1"]))
        f = fit_power(T["r"], y, T["w"])
    return f


def evaluate(T, f):
    m = ~np.isnan(T["mc"])
    pr = f_eval(f, T["r"])
    r2 = 1 - ((pr[m] - T["mc"][m]) ** 2).mean() / max(((T["mc"][m] - T["mc"][m].mean()) ** 2).mean(), 1e-9) if m.sum() > 2 else np.nan
    bres = pr - T["dt"] - np.where(T["term"], 0.0, f_eval(f, T["r1"]))
    return dict(mc_r2=float(r2), bellman_mean=float(bres.mean()), bellman_mae=float(np.abs(bres).mean()), n=int(len(pr)),
                n_mc=int(m.sum()))


def load_rows(tags, weights):
    rows = []
    for t in tags:
        R = json.loads((HERE / "results" / EP_FILES.get(t, f"{t}_episodes.json")).read_text())
        for r in R:
            r["_w"] = weights.get(t, 1.0)
        rows += R
    return rows


def cmd_fit(out, tags, weights, calib):
    rows = load_rows(tags, weights)
    res = dict(data=tags, weights=weights, goals={})
    from sklearn.model_selection import GroupKFold
    for g in GOALS:
        R = [r for r in rows if r["goal"] == g]
        res["goals"][g] = dict(theta=R[0]["theta"], mopt=VS.get(g, {}).get("mass_frac_best_placement", 1.0), n_episodes=len(R),
                               n_completed=sum(r["kstar"] is not None for r in R))
        for name in ("lyap", "gap"):
            T = transitions(R, name)
            for method in ("mc", "td"):
                f = fit(T, method)
                if f is None:
                    continue
                # held-out (by start state)
                ev = []
                for tr, te in GroupKFold(n_splits=min(4, len(set(T["grp"])))).split(T["r"], groups=T["grp"]):
                    Ttr = {k: v[tr] for k, v in T.items()}; Tte = {k: v[te] for k, v in T.items()}
                    ff = fit(Ttr, method)
                    if ff is not None:
                        ev.append(evaluate(Tte, ff))
                cv = {k: float(np.nanmean([e[k] for e in ev])) for k in ("mc_r2", "bellman_mean", "bellman_mae")} if ev else {}
                res["goals"][g][f"{name}|{method}"] = dict(**f, cv=cv, train=evaluate(T, f))
            if calib and name in calib and g in calib[name]:
                res["goals"][g][f"{name}|offset"] = calib[name][g]
    p = HERE / f"results/{out}.json"; tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, p)
    print(f"{'goal':12s} {'map':10s} {'c':>7s} {'p':>5s} {'cv MC R2':>9s} {'cv bellman':>10s} {'mae':>6s}")
    for g, v in res["goals"].items():
        for k, f in v.items():
            if "|" in k and "offset" not in k:
                print(f"{g:12s} {k:10s} {f['c']:7.1f} {f['p']:5.2f} {f['cv'].get('mc_r2', np.nan):9.2f} {f['cv'].get('bellman_mean', np.nan):+10.2f} {f['cv'].get('bellman_mae', np.nan):6.2f}")


def cmd_calib():
    """model-predicted vs true change of BOTH readouts for every executed push of the EXP-0070 runs."""
    import torch
    from batched_closed_loop import goal_dist, goal_mask
    from simple_mpc.adapters import make_occ_adapter, occ_from_particles, occ_for_scoring
    from simple_mpc.learned_mpc import lyap
    from Baselines.common.goals import mass_in_region
    dev = "cuda"; ad = make_occ_adapter("nfd_3ch_randlen", dev, "corner")
    rows = []
    for tag in ("pilot_v0", "pilot_v0b", "pilot_v1", "pilot_b01"):
        d = json.loads((HERE / f"results/{tag}.json").read_text())
        for e in d["episodes"]:
            if not e.get("complete"):
                continue
            g = e["goal"]; dist = goal_dist(g).to(dev); m = torch.from_numpy(goal_mask(g)).to(dev).float()
            mopt = VS.get(g, {}).get("mass_frac_best_placement", 1.0)
            S = torch.tensor(e["states"]).float()
            occ_s = occ_for_scoring(S.to(dev)); tot = occ_s.reshape(len(S), -1).sum(1)
            mr = (mass_in_region(occ_s, m) / tot / mopt).cpu().numpy()
            occ_in = occ_from_particles(S[:-1], dev)
            A = torch.tensor(e["actions"]).float().to(dev)
            with torch.no_grad():
                pr = ad.predict_step(occ_in, A)
            fin = lambda o: ((o * m).reshape(len(o), -1).sum(1) / o.reshape(len(o), -1).sum(1).clamp_min(1e-6))
            dV_pred = (lyap(pr, dist) - lyap(occ_in, dist)).cpu().numpy()
            dM_pred = ((fin(pr) - fin(occ_in)) / mopt).cpu().numpy()
            V = np.array(e["values"])
            for k in range(len(A)):
                rows.append(dict(goal=g, V=V[k], dV_true=V[k + 1] - V[k], dV_pred=float(dV_pred[k]), M=float(mr[k]),
                                 dM_true=float(mr[k + 1] - mr[k]), dM_pred=float(dM_pred[k])))
    theta = lambda g: 0.9 if (g.startswith("quad") or g == "two_squares") else 0.7
    calib = {"lyap": {}, "gap": {}}
    for g in GOALS:
        R = [r for r in rows if r["goal"] == g]
        for name, x, dp, dt in (("lyap", "V", "dV_pred", "dV_true"), ("gap", "M", "dM_pred", "dM_true")):
            xv = np.array([r[x] for r in R]); err = np.array([r[dp] - r[dt] for r in R])
            if name == "gap":
                xv = np.maximum(0.0, theta(g) - xv); err = -err             # gap change = -mass change
            edges = np.unique(np.quantile(xv, np.linspace(0, 1, 6)))
            cen, bias = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = (xv >= lo) & (xv <= hi)
                if s.sum() >= 5:
                    cen.append(float(xv[s].mean())); bias.append(float(err[s].mean()))
            slope = float(np.polyfit([r[dt] for r in R], [r[dp] for r in R], 1)[0])
            calib[name][g] = dict(r=cen, bias=bias, pred_vs_true_slope=slope, n=len(R))
    p = HERE / "results/etc_v2_calib.json"; p.write_text(json.dumps(calib, indent=1))
    for name in calib:
        for g, v in calib[name].items():
            print(name, g, "slope", round(v["pred_vs_true_slope"], 2), "bias by r:", [(round(a, 3), round(b, 4)) for a, b in zip(v["r"], v["bias"])])


if __name__ == "__main__":
    if sys.argv[1] == "calib":
        cmd_calib()
    else:
        a = sys.argv[2:]
        out = a[0]; w = {}; tags = ["retro", "pilot_v0b", "pilot_v1", "pilot_b01"]
        if "--weight-new" in a:
            for kv in a[a.index("--weight-new") + 1:]:
                if kv.startswith("--"):
                    break
                k, v = kv.split("="); w[k] = float(v)
        if "--data" in a:
            tags = []
            for t in a[a.index("--data") + 1:]:
                if t.startswith("--"):
                    break
                tags.append(t)
        cal = json.loads((HERE / "results/etc_v2_calib.json").read_text()) if (HERE / "results/etc_v2_calib.json").exists() else None
        cmd_fit(out, tags, w, cal)
