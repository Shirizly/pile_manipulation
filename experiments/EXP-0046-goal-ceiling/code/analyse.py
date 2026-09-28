"""EXP-0046 steps 4-5: achieved_fraction of closed-loop runs against V*(g),
and goal redundancy (shape + push-ranking + model-ranking + closed-loop).

Inputs: results/vstar.json (vstar.py), EXP-0044/0045 episode JSONs,
EXP-0030 RUN-0001 truth/pred dv (20480 rows x 30 goals x [lyap, mass]).
Writes results/achieved_fraction.json and results/redundancy.json atomically.
"""
import json
import os
import sys
from itertools import combinations

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.getcwd())
from Baselines.common.eval_report import _fixed_goal  # noqa: E402
from simple_mpc.adapters import occ_for_scoring  # noqa: E402
from simple_mpc.learned_mpc import lyap  # noqa: E402

R = "experiments/EXP-0046-goal-ceiling/results"
GOALS = [f"letter_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + [f"quadrant_{q}" for q in range(4)]
CORPUS = "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys/step0.pt"
A30 = "experiments/EXP-0030-state-superiority-ds0005/artifacts/RUN-0001"
MODELS = ["linear_switched_hard", "nfd_3ch_finetuned", "nfd_3ch_randlen",
          "nfd_residual_warped_flipaug_randlen", "nfd_residual_worldframe_noaug_ep43",
          "nfd_warped_randlen_flipaug_epoch30", "nfd_warped_randlen_flipaug", "nfd_warped_randlen"]


def save(obj, name):
    p = os.path.join(R, name); tmp = p + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, p)


def achieved(vstar):
    out = {}
    for tag, f, ks in [("EXP-0044", "experiments/EXP-0044-closed-loop-goal-breadth/results/goal_breadth_tuned.json", [8]),
                       ("EXP-0045", "experiments/EXP-0045-decision-time-vs-actions/results/time_vs_actions.json", [8, 24])]:
        E = [e for e in json.load(open(f))["episodes"] if e.get("complete", True)]
        rows = []
        for e in E:
            v = np.asarray(e["values"]); vs = vstar[e["goal"]]["vstar"]["lyapunov"]
            r = dict(goal=e["goal"], model=e["model"], planner=e["planner"], cell=e["cell"], start=e["start"],
                     V0=float(v[0]), Vstar=vs)
            for k in ks:
                if len(v) > k:
                    r[f"impr_{k}"] = float(v[0] - v[k])
                    r[f"af_{k}"] = float((v[0] - v[k]) / (v[0] - vs))
            rows.append(r)

        def summ(sel):
            d = dict(n=len(sel), V0=float(np.mean([r["V0"] for r in sel])),
                     Vstar=float(np.mean([r["Vstar"] for r in sel])))
            for k in ks:
                a = [r[f"af_{k}"] for r in sel if f"af_{k}" in r]
                d[f"impr_{k}"] = float(np.mean([r[f"impr_{k}"] for r in sel if f"impr_{k}" in r]))
                d[f"af_{k}_mean"] = float(np.mean(a)); d[f"af_{k}_median"] = float(np.median(a))
                d[f"af_{k}_ratio_of_means"] = d[f"impr_{k}"] / (d["V0"] - d["Vstar"])
            return d
        res = dict(overall=summ(rows), per_goal={}, per_model_planner={}, per_cell={})
        for g in sorted({r["goal"] for r in rows}):
            res["per_goal"][g] = summ([r for r in rows if r["goal"] == g])
        for mp in sorted({(r["model"], r["planner"]) for r in rows}):
            res["per_model_planner"]["/".join(mp)] = summ([r for r in rows if (r["model"], r["planner"]) == mp])
        for c in sorted({r["cell"] for r in rows}):
            res["per_cell"][c] = summ([r for r in rows if r["cell"] == c])
        out[tag] = res
    return out


def v0_check():
    """V0 of EXP-0044 start 40 / quadrant_0 recomputed from the corpus."""
    d = torch.load(CORPUS, map_location="cpu", weights_only=False)
    row = int(torch.nonzero(d["slate_idx"] == 40)[0])
    _, dist = _fixed_goal("quadrant_0", 64, 64)
    return float(lyap(occ_for_scoring(d["states"][row:row + 1]), dist)[0])


def capture(vp, vt):
    """cost sense (lyapunov dv): pick argmin pred, (mean - chosen)/(mean - best)."""
    pick = vp.argmin(0); chosen = vt[pick, np.arange(vt.shape[1])]
    mean, best = vt.mean(0), vt.min(0); den = mean - best
    return np.where(np.abs(den) > 1e-9, (mean - chosen) / np.where(den == 0, 1, den), np.nan)


def redundancy(ach44):
    G = len(GOALS)
    masks = [_fixed_goal(g, 64, 64) for g in GOALS]
    M = np.stack([m.numpy().ravel() > 0 for m, _ in masks]); D = np.stack([d.numpy().ravel() for _, d in masks])
    iou = np.zeros((G, G)); dcorr = np.corrcoef(D)
    for i in range(G):
        for j in range(G):
            iou[i, j] = (M[i] & M[j]).sum() / max((M[i] | M[j]).sum(), 1)
    slate = torch.load(CORPUS, map_location="cpu", weights_only=False)["slate_idx"].long().numpy()
    vt = torch.load(f"{A30}/truth.pt", weights_only=False)["dv"].numpy()[:, :, 0]     # lyapunov dv
    states = np.unique(slate)
    # push-ranking agreement: Spearman across a state's 128 pushes, averaged over states
    rk = np.zeros((G, G)); n = 0
    for s in states:
        pool = vt[slate == s]
        if np.all(pool.std(0) > 0):
            rk += spearmanr(pool).correlation; n += 1
    rk /= n
    # per-goal slateN per model, and model-ranking Kendall between goals
    sn = {}
    for m in MODELS:
        vp = torch.load(f"{A30}/pred_{m}.pt", weights_only=False)["dv"].numpy()[:, :, 0]
        sn[m] = np.nanmean([capture(vp[slate == s], vt[slate == s]) for s in states], 0)
    SN = np.stack([sn[m] for m in MODELS], 1)                    # (G, models)
    kt = np.ones((G, G))
    for i, j in combinations(range(G), 2):
        kt[i, j] = kt[j, i] = kendalltau(SN[i], SN[j]).statistic
    # clustering on push-ranking agreement (what matters for MPC choice)
    dist = np.clip(1 - rk, 0, None); np.fill_diagonal(dist, 0); dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), "average")
    clusters = {}
    for k in (8, 10, 12):
        lab = fcluster(Z, k, "maxclust")
        groups = []
        for c in sorted(set(lab)):
            idx = np.nonzero(lab == c)[0]
            med = idx[np.argmax(rk[np.ix_(idx, idx)].mean(1))]
            groups.append(dict(members=[GOALS[i] for i in idx], medoid=GOALS[med],
                               mean_within_rank_corr=float(rk[np.ix_(idx, idx)].mean())))
        clusters[k] = groups
    pairs = []
    for i, j in combinations(range(G), 2):
        pairs.append(dict(a=GOALS[i], b=GOALS[j], iou=float(iou[i, j]), dist_corr=float(dcorr[i, j]),
                          push_rank_corr=float(rk[i, j]), model_rank_kendall=float(kt[i, j])))
    pairs.sort(key=lambda p: -p["push_rank_corr"])
    cl44 = {g: v["impr_8"] for g, v in ach44["per_goal"].items()}
    return dict(goals=GOALS, models=MODELS, iou=iou.tolist(), dist_corr=dcorr.tolist(), push_rank_corr=rk.tolist(),
                model_rank_kendall=kt.tolist(), slateN_lyap=SN.tolist(),
                slateN_goal_mean=dict(zip(GOALS, SN.mean(1).tolist())),
                closed_loop_impr8_EXP0044=cl44, clusters=clusters, top_pairs=pairs[:40],
                n_states_used=n)


def main():
    vstar = json.load(open(f"{R}/vstar.json"))
    v0 = v0_check()
    ach = achieved(vstar); ach["v0_check_start40_quadrant0"] = v0
    save(ach, "achieved_fraction.json")
    for t in ("EXP-0044", "EXP-0045"):
        print(t, json.dumps(ach[t]["overall"]))
        for g, v in ach[t]["per_goal"].items():
            print(f"  {g:11s} V0={v['V0']:.3f} V*={v['Vstar']:.4f} impr8={v['impr_8']:.3f} af8={v['af_8_mean']:.3f}"
                  + (f" af24={v['af_24_mean']:.3f}" if "af_24_mean" in v else ""))
        for c, v in ach[t]["per_cell"].items():
            print("  cell", c, {k: round(x, 3) for k, x in v.items() if k.startswith("af")})
        for c, v in ach[t]["per_model_planner"].items():
            print("  mp", c, {k: round(x, 3) for k, x in v.items() if k.endswith("_mean")})
    print("V0 check (start40, quadrant_0):", v0)
    red = redundancy(ach["EXP-0044"])
    save(red, "redundancy.json")
    for p in red["top_pairs"][:25]:
        print(f"  {p['a']:11s} {p['b']:11s} rank={p['push_rank_corr']:.3f} iou={p['iou']:.2f} "
              f"dcorr={p['dist_corr']:.3f} kendall={p['model_rank_kendall']:.2f}")
    for k, gs in red["clusters"].items():
        print(k, [(g["medoid"], len(g["members"])) for g in gs])
    for g in red["clusters"][12]:
        print("   ", g["medoid"], g["members"], round(g["mean_within_rank_corr"], 3))
    print({g: round(v, 3) for g, v in red["slateN_goal_mean"].items()})


if __name__ == "__main__":
    main()
