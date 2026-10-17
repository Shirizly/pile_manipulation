"""EXP-0070 RUN-0005 (test D): is remaining task time a function of V alone, or does a learned
etc(s, plate, g) on cheap state features predict it better -- and does a TD (Bellman) fit beat the
Monte-Carlo fit, using censored episodes without imputation?

Data: the recorded closed-loop episodes of retro_time.py's SOURCES (states recorded), new clock
(plan_time + physical action time), completion as in retro_time (theta 0.9 / letters 0.7).
Features of state s_k (cube centres xy, goal mask, plate = end of push k, or HOME at k = 0):
  V (true lyapunov), mass_rel, n_out (cubes whose centre pixel is outside the mask), mean / max / sum of
  the distance field at outside cubes, spread (rms distance to COM), plate_to_out (plate -> nearest
  outside cube, m), plate_to_out_mean, goal one-hot.
Models, evaluated on held-out (start-state) groups, target = MC remaining time of COMPLETED episodes:
  fV      per-goal power law c V^p (the pilot's planning map), fitted on the training folds
  ridgeF  per-family ridge on standardised features
  gbF     HistGradientBoosting on all features + goal one-hot
  gbF_td  same model class, fitted by fitted value iteration (8 sweeps) on ALL states incl. censored
          episodes: target_k = dt_k + etc(s_{k+1}), etc = 0 at completion; scored on the same MC targets
  gbF_noplate  gbF without the two plate features (does the plate pose matter for etc?)
-> results/etc_learned.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from batched_closed_loop import goal_mask
from Baselines.common.goals import dist_field_from_mask
from retro_time import SOURCES, family
from action_time import HOME

HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
ROWS = {(r["src"], r["model"], r["planner"], r["cell"], r["goal"], r["start"]): r
        for r in json.loads((HERE / "results/retro_episodes.json").read_text())}
LO, HI, RES = -0.064, 0.064, 64
GOALS = sorted({k[4] for k in ROWS})
MASK = {g: goal_mask(g) for g in GOALS}
DIST = {g: dist_field_from_mask(MASK[g]) for g in GOALS}
FEATS = ["V", "mass_rel", "n_out", "d_mean", "d_max", "d_sum", "spread", "plate_to_out", "plate_to_out_mean"]


def feats(xy, plate, g):
    pix = np.clip(np.rint((xy - LO) / (HI - LO) * (RES - 1)).astype(int), 0, RES - 1)
    inside = MASK[g][pix[:, 0], pix[:, 1]]
    d = DIST[g][pix[:, 0], pix[:, 1]]
    out = ~inside
    com = xy.mean(0)
    po = np.linalg.norm(xy[out] - plate[None], axis=1) if out.any() else np.zeros(1)
    return [int(out.sum()), float(d[out].mean()) if out.any() else 0.0, float(d.max()), float(d.sum()),
            float(np.sqrt(((xy - com) ** 2).sum(1).mean())), float(po.min()), float(po.mean())]


def build():
    X, y_mc, done_mc, grp, gname, nxt, dt, term, ep_id, cl = [], [], [], [], [], [], [], [], [], []
    eid = 0
    for src, p in SOURCES.items():
        for e in json.loads((REPO / p).read_text())["episodes"]:
            if not (e.get("complete") and "states" in e):
                continue
            r = ROWS[(src, e["model"], e["planner"], e.get("cell", ""), e["goal"], e["start"])]
            S = np.array(e["states"])[:, :, :2]; A = np.array(e["actions"])
            step = np.array(r["plan"]) + np.array(r["t_act"]["total"])
            k = r["kstar"]; last = k if k else len(step)
            for j in range(last + 1):
                plate = np.array(HOME) if j == 0 else A[j - 1, 2:4]
                X.append([r["V"][j], r["mass_rel"][j]] + feats(S[j], plate, e["goal"]))
                gname.append(e["goal"]); grp.append(e["start"]); ep_id.append(eid)
                is_term = (k is not None and j == k)
                term.append(is_term)
                y_mc.append(float(step[j:k].sum()) if (k is not None and j < k) else np.nan)
                dt.append(float(step[j]) if j < last else 0.0)
                nxt.append(len(X) if j < last else -1)          # index of s_{j+1}
                cl.append(k is None and j == last)               # censored episode's last state: NOT terminal
            eid += 1
    return (np.array(X, float), np.array(y_mc), np.array(grp), np.array(gname), np.array(nxt), np.array(dt),
            np.array(term), np.array(ep_id), np.array(cl))


def onehot(gn):
    return np.array([[float(g == h) for h in GOALS] for g in gn])


def power_fit(v, y):
    best = None
    for pw in np.linspace(0.2, 2.0, 37):
        z = np.clip(v, 1e-6, None) ** pw; c = (z * y).sum() / (z * z).sum(); s = ((y - c * z) ** 2).sum()
        if best is None or s < best[0]:
            best = (s, c, pw)
    return best[1], best[2]


def main():
    X, y, grp, gn, nxt, dt, term, ep, cl = build()
    has = ~np.isnan(y)
    Xg = np.hstack([X, onehot(gn)])
    noplate = [i for i, f in enumerate(FEATS) if not f.startswith("plate")]
    Xnp = np.hstack([X[:, noplate], onehot(gn)])
    fam = np.array([family(g) for g in gn])
    pred = {m: np.full(len(y), np.nan) for m in ("fV", "ridgeF", "gbF", "gbF_noplate", "gbF_td")}
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, grp):
        trm, tem = tr[has[tr]], te[has[te]]
        for g in GOALS:
            a, b = trm[gn[trm] == g], tem[gn[tem] == g]
            if len(a) > 5 and len(b):
                c, p = power_fit(X[a, 0], y[a]); pred["fV"][b] = c * np.clip(X[b, 0], 1e-6, None) ** p
        for f in set(fam):
            a, b = trm[fam[trm] == f], tem[fam[tem] == f]
            if len(a) > 20 and len(b):
                mu, sd = X[a].mean(0), X[a].std(0) + 1e-9
                m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((X[a] - mu) / sd, y[a])
                pred["ridgeF"][b] = m.predict((X[b] - mu) / sd)
        gb = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
        pred["gbF"][tem] = HistGradientBoostingRegressor(**gb).fit(Xg[trm], y[trm]).predict(Xg[tem])
        pred["gbF_noplate"][tem] = HistGradientBoostingRegressor(**gb).fit(Xnp[trm], y[trm]).predict(Xnp[tem])
        # fitted value iteration on ALL training-fold states (censored episodes included)
        etc = np.where(has, y, np.nanmean(y))            # init from MC where known
        etc[term] = 0.0
        m = None
        trtd = tr[~cl[tr]]                                # censored last states: value comes from the model
        for _ in range(8):
            tgt = dt[trtd] + np.where(nxt[trtd] >= 0, etc[np.maximum(nxt[trtd], 0)], 0.0)
            tgt[term[trtd]] = 0.0
            m = HistGradientBoostingRegressor(**gb).fit(Xg[trtd], tgt)
            etc = m.predict(Xg); etc[term] = 0.0
        pred["gbF_td"][tem] = m.predict(Xg[tem])
    res = dict(n_states=int(len(y)), n_states_mc=int(has.sum()), n_episodes=int(ep.max() + 1), features=FEATS, models={})
    for mname, pr in pred.items():
        for f in ["ALL", "quadrant", "two_squares", "letter"]:
            s = has & ~np.isnan(pr) & ((fam == f) if f != "ALL" else True)
            if s.sum() < 10:
                continue
            err = pr[s] - y[s]
            r2 = 1 - (err ** 2).mean() / ((y[s] - y[s].mean()) ** 2).mean()
            # within-goal R2 (the planner compares states of ONE goal)
            r2w = []
            for g in set(gn[s]):
                q = s & (gn == g)
                if q.sum() > 20:
                    r2w.append(1 - ((pr[q] - y[q]) ** 2).mean() / ((y[q] - y[q].mean()) ** 2).mean())
            res["models"][f"{mname}|{f}"] = dict(n=int(s.sum()), r2=float(r2), mae_s=float(np.abs(err).mean()),
                                                 r2_within_goal_mean=float(np.mean(r2w)) if r2w else None)
    # Bellman residual of each model on held-out consecutive pairs: etc(s_k) - dt_k - etc(s_{k+1})
    for mname in ("fV", "gbF", "gbF_td"):
        pr = pred[mname]; i = np.nonzero((nxt >= 0) & ~np.isnan(pr))[0]; j = nxt[i]
        ok = ~np.isnan(pr[j]) | term[j]
        nv = np.where(term[j], 0.0, pr[j])
        r = (pr[i] - dt[i] - nv)[ok]
        res[f"bellman_residual_{mname}"] = dict(mean=float(np.nanmean(r)), mae=float(np.nanmean(np.abs(r))))
    out = HERE / "results/etc_learned.json"; tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)
    for k, v in res["models"].items():
        print(f"{k:24s} n {v['n']:6d} R2 {v['r2']:+.3f} within-goal R2 {v['r2_within_goal_mean'] if v['r2_within_goal_mean'] is None else round(v['r2_within_goal_mean'],3)} MAE {v['mae_s']:.1f}s")
    for k in res:
        if k.startswith("bellman"):
            print(k, res[k])


if __name__ == "__main__":
    main()
