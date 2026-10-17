"""EXP-0070 RUN-0002: re-time existing closed-loop episodes with the physical action-time model
and bootstrap the value -> expected-time-to-completion mapping etc(v).

Inputs: every legal (or, flagged, non-legal) closed-loop episode set that recorded particle states.
Per episode / push k: t_act(k) from action_time.episode_action_times (plate starts at HOME at
safe height; each push charged lift-from-previous-end + transit + descent + sweep), planning time
plan_time_s, in-goal mass relative to the goal's optimum (EXP-0046), lyapunov achieved fraction.
Completion = first k with mass_rel >= theta(goal family) (METRICS.md `completion_time`).

Outputs results/retro_time.json:
  * time_components: per source, mean/quantiles of total / transit / sweep / up+down per push;
  * completion: per source x (model, planner, cell) completion fraction and median completion time
    under OLD clock (plan + 2 s flat) and NEW clock (plan + t_act), and the cell order under each;
  * etc_fit: for each goal family and each value readout (mass_rel, lyap achieved), the binned
    monotone map v -> mean remaining time to completion over completed episodes, with per-bin
    completion probability (survivor bias made visible), and Spearman(v, remaining time).
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batched_closed_loop import goal_mask
from Baselines.common.goals import mass_in_region
from simple_mpc.adapters import occ_for_scoring
from action_time import episode_action_times

HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
SOURCES = {
    "EXP-0051_legal": "experiments/EXP-0051-task-success-completion-time/results/success_states_legal.json",
    "EXP-0054_legal": "experiments/EXP-0054-narrow-closed-loop/results/narrow_gd_w3_legal.json",
    "EXP-0052": "experiments/EXP-0052-success-objective/results/objective_states.json",
    "EXP-0055_all": "experiments/EXP-0055-capacity-aware-value/results/all.json",
    **{f"EXP-0066_{t}": f"experiments/EXP-0066-closed-loop-power-study/results/{t}.json"
       for t in ["broad_s0_ps0", "broad_s1_ps0", "lin_orig_ps0", "nfd_s0_ps0", "nfd_s1_ps0", "nfd_s2_ps0"]},
}
LEGAL = {"EXP-0051_legal", "EXP-0054_legal"} | {k for k in SOURCES if k.startswith("EXP-0066")}
THETA = {"quadrant": 0.9, "two_squares": 0.9, "letter": float(os.environ.get("THETA_LETTER", 0.7))}
T_OLD = 2.0


def family(g):
    return "quadrant" if g.startswith("quadrant") else ("two_squares" if g == "two_squares" else "letter")


def episode_rows(src, e, dev):
    g = e["goal"]; m = torch.from_numpy(goal_mask(g)).to(dev)
    S = torch.tensor(e["states"]).float().to(dev)
    occ = occ_for_scoring(S)
    mf = (mass_in_region(occ, m) / occ.reshape(len(S), -1).sum(1)).cpu().numpy()
    vs = VS.get(g, {}); mopt = vs.get("mass_frac_best_placement", 1.0); vstar = vs.get("vstar_map_value", 0.0)
    v = np.array(e["values"]); ach = (v[0] - v) / max(v[0] - vstar, 1e-9)
    ta = episode_action_times(e["actions"])
    pt = np.array(e["plan_time_s"], dtype=float)
    rel = mf / mopt
    th = THETA[family(g)]
    hit = np.nonzero(rel[1:] >= th)[0]
    kstar = int(hit[0]) + 1 if len(hit) else None
    return dict(src=src, model=e["model"], planner=e["planner"], cell=e.get("cell", ""), goal=g, fam=family(g),
                start=e["start"], mass_rel=rel.tolist(), ach=ach.tolist(), V=v.tolist(), t_act=ta, plan=pt.tolist(), kstar=kstar,
                theta=th, budget=e.get("budget_s"))


def summarise_components(rows):
    out = {}
    for src in sorted({r["src"] for r in rows}):
        R = [r for r in rows if r["src"] == src]
        comp = {k: np.concatenate([np.array(r["t_act"][k]) for r in R]) for k in ("total", "transit", "sweep", "up", "down")}
        plan = np.concatenate([np.array(r["plan"]) for r in R])
        out[src] = {k: dict(mean=float(v.mean()), p10=float(np.quantile(v, .1)), p50=float(np.median(v)),
                            p90=float(np.quantile(v, .9))) for k, v in comp.items()}
        out[src]["plan"] = dict(mean=float(plan.mean()))
        out[src]["transit_share_of_exec"] = float(comp["transit"].sum() / comp["total"].sum())
        out[src]["n_pushes"] = int(len(plan))
    return out


def completion_table(rows):
    out = {}
    for src in sorted({r["src"] for r in rows}):
        R = [r for r in rows if r["src"] == src]
        cells = sorted({(r["model"], r["planner"], r["cell"]) for r in R})
        tab = {}
        for c in cells:
            Rc = [r for r in R if (r["model"], r["planner"], r["cell"]) == c]
            for fam in sorted({r["fam"] for r in Rc}):
                Rf = [r for r in Rc if r["fam"] == fam]
                K = len(Rf[0]["plan"])
                old, new, ks = [], [], []
                for r in Rf:
                    k = r["kstar"]
                    p = np.array(r["plan"]); t = np.array(r["t_act"]["total"])
                    kk = k if k else K                       # censored at K pushes (METRICS convention)
                    old.append(p[:kk].sum() + T_OLD * kk); new.append(p[:kk].sum() + t[:kk].sum()); ks.append(kk)
                tab[f"{c[0]}|{c[1]}|{c[2]}|{fam}"] = dict(
                    n=len(Rf), frac_complete=float(np.mean([r["kstar"] is not None for r in Rf])),
                    mean_censored_time_old=float(np.mean(old)), mean_censored_time_new=float(np.mean(new)),
                    mean_censored_pushes=float(np.mean(ks)),
                    exec_per_push_new=float(np.mean([np.mean(r["t_act"]["total"]) for r in Rf])))
        out[src] = tab
    return out


def etc_fit(rows, readout, fam, n_bins=10):
    """Remaining time to completion (NEW clock: plan + t_act) from each visited state of completed
    episodes, binned by the readout value of that state. Returns the binned map + diagnostics."""
    xs, ys, all_x, all_done = [], [], [], []
    for r in rows:
        if r["fam"] != fam:
            continue
        v = np.array(r[readout]); p = np.array(r["plan"]); t = np.array(r["t_act"]["total"])
        step = p + t                                   # time of push j+1 (index j)
        k = r["kstar"]
        last = k if k else len(step)
        for j in range(last):                          # states s_0..s_{k*-1} (before completion)
            all_x.append(v[j]); all_done.append(k is not None)
            if k:
                xs.append(v[j]); ys.append(step[j:k].sum())
    xs, ys = np.array(xs), np.array(ys); all_x, all_done = np.array(all_x), np.array(all_done)
    if len(xs) < 20:
        return dict(n=int(len(xs)))
    edges = np.unique(np.quantile(all_x, np.linspace(0, 1, n_bins + 1)))
    centers, means, pdone, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (xs >= lo) & (xs <= hi); sel_all = (all_x >= lo) & (all_x <= hi)
        if sel.sum() == 0:
            continue
        centers.append(float(xs[sel].mean())); means.append(float(ys[sel].mean()))
        pdone.append(float(all_done[sel_all].mean())); counts.append(int(sel.sum()))
    # monotone (non-increasing in v) by pooling adjacent violators
    m = list(means); w = list(counts); c = list(centers)
    i = 0
    while i < len(m) - 1:
        if m[i] < m[i + 1]:
            tot = w[i] + w[i + 1]
            m[i] = (m[i] * w[i] + m[i + 1] * w[i + 1]) / tot; c[i] = (c[i] * w[i] + c[i + 1] * w[i + 1]) / tot
            w[i] = tot; del m[i + 1], w[i + 1], c[i + 1]; i = max(i - 1, 0)
        else:
            i += 1
    rho = spearmanr(xs, ys).correlation
    # linear fit slope: seconds per unit of readout
    A = np.vstack([xs, np.ones_like(xs)]).T; slope, icpt = np.linalg.lstsq(A, ys, rcond=None)[0]
    resid = ys - (slope * xs + icpt); r2 = 1 - resid.var() / ys.var()
    return dict(n=int(len(xs)), n_all_states=int(len(all_x)), spearman=float(rho), linear_slope_s=float(slope),
                linear_icpt_s=float(icpt), linear_r2=float(r2), bins=dict(center=centers, mean_remaining_s=means,
                p_complete=pdone, count=counts), monotone=dict(v=c, etc_s=m))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for src, p in SOURCES.items():
        d = json.loads((REPO / p).read_text())
        for e in d["episodes"]:
            if e.get("complete") and "states" in e:
                rows.append(episode_rows(src, e, dev))
        print(src, len(rows), flush=True)
    res = dict(theta=THETA, t_old=T_OLD, legal_sources=sorted(LEGAL),
               time_components=summarise_components(rows), completion=completion_table(rows), etc_fit={})
    legal_rows = [r for r in rows if r["src"] in LEGAL]
    for fam in ("quadrant", "two_squares", "letter"):
        for ro in ("mass_rel", "ach"):
            res["etc_fit"][f"{fam}|{ro}|legal"] = etc_fit(legal_rows, ro, fam)
            res["etc_fit"][f"{fam}|{ro}|all"] = etc_fit(rows, ro, fam)
    # reachability of letter thresholds (to document the theta choice)
    L = [r for r in rows if r["fam"] == "letter"]
    res["letter_reach"] = {str(th): float(np.mean([max(r["mass_rel"][1:]) >= th for r in L])) for th in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9)}
    out = HERE / "results/retro_time.json"; tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)
    # compact per-episode table for later refits (mapping inputs)
    ep = HERE / "results/retro_episodes.json"; tmp = Path(str(ep) + ".tmp")
    tmp.write_text(json.dumps(rows)); os.replace(tmp, ep)
    print(json.dumps({k: v for k, v in res.items() if k in ("time_components", "letter_reach")}, indent=1))
    for k, v in res["etc_fit"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("bins", "monotone")})


if __name__ == "__main__":
    main()
