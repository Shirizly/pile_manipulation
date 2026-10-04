"""EXP-0051 analysis: success-type value functions and task-completion time (DESIGN.md).

Per episode and push k: lyapunov achieved_fraction (EXP-0046 V*), in-goal mass fraction
(mass_in_region / total mass, soft scoring) and signed-mass fraction, each also relative
to the goal's optimum (EXP-0046 mass_frac_best_placement / signed_frac_at_vstar).
Completion = first k with in-goal mass fraction >= theta x optimum; completion time =
sum of (actual planning time + t_act) over pushes 1..k. Censored at the episode length.
-> results/analysis.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask
from Baselines.common.goals import mass_in_region, signed_mass_in_region
from simple_mpc.adapters import occ_for_scoring

HERE = REPO / "experiments/EXP-0051-task-success-completion-time"
tag = sys.argv[1] if len(sys.argv) > 1 else "success_states"
E = [e for e in json.loads((HERE / f"results/{tag}.json").read_text())["episodes"] if e.get("complete")]
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
THETAS, TACTS = (0.8, 0.9, 0.95), (1.0, 2.0, 5.0, 10.0)
rows = []
for e in E:
    g = e["goal"]; m = torch.from_numpy(goal_mask(g))
    S = torch.tensor(e["states"]).float()                       # (k+1, n, 3)
    occ = occ_for_scoring(S)
    tot = occ.reshape(len(S), -1).sum(1)
    mf = (mass_in_region(occ, m) / tot).numpy(); sf = (signed_mass_in_region(occ, m) / tot).numpy()
    vs = VS.get(g, {})
    mopt = vs.get("mass_frac_best_placement", 1.0); sopt = vs.get("signed_frac_at_vstar", 1.0)
    vstar = vs.get("vstar_map_value", 0.0)
    v = np.array(e["values"]); ach = (v[0] - v) / max(v[0] - vstar, 1e-9)
    rel = mf / mopt
    pt = np.array(e["plan_time_s"])
    comp = {}
    for th in THETAS:
        hit = np.nonzero(rel[1:] >= th)[0]
        k = int(hit[0]) + 1 if len(hit) else None
        comp[str(th)] = dict(pushes=k, time={str(ta): (float(pt[:k].sum() + ta * k) if k else None) for ta in TACTS})
    rows.append(dict(model=e["model"], planner=e["planner"], goal=g, start=e["start"], mass_frac=mf.tolist(),
                     signed_frac=sf.tolist(), mass_rel=rel.tolist(), signed_rel=(sf / sopt).tolist(),
                     achieved=ach.tolist(), mass_opt=mopt, completion=comp, plan_s=float(pt.mean())))
out = {"episodes": rows, "summary": {}}
cfgs = sorted({(r["model"], r["planner"]) for r in rows})
for (m, pl) in cfgs + [("ALL", "ALL")]:
    R = [r for r in rows if (m == "ALL" or (r["model"] == m and r["planner"] == pl))]
    key = f"{m}/{pl}"
    sm = dict(n=len(R))
    for k in (4, 8, 16, 24):
        sm[f"k{k}"] = dict(achieved=float(np.mean([r["achieved"][min(k, len(r["achieved"]) - 1)] for r in R])),
                           mass_frac=float(np.mean([r["mass_frac"][min(k, len(r["mass_frac"]) - 1)] for r in R])),
                           mass_rel=float(np.mean([r["mass_rel"][min(k, len(r["mass_rel"]) - 1)] for r in R])),
                           signed_rel=float(np.mean([r["signed_rel"][min(k, len(r["signed_rel"]) - 1)] for r in R])))
    sm["k0_mass_frac"] = float(np.mean([r["mass_frac"][0] for r in R]))
    for th in THETAS:
        ks = [r["completion"][str(th)]["pushes"] for r in R]
        done = [k for k in ks if k is not None]
        sm[f"complete_{th}"] = dict(frac=len(done) / len(ks), median_pushes=float(np.median(done)) if done else None,
                                    median_time={str(ta): float(np.median([r["completion"][str(th)]["time"][str(ta)]
                                                                            for r in R if r["completion"][str(th)]["pushes"]]))
                                                 if done else None for ta in TACTS})
    out["summary"][key] = sm
per_goal = {}
for g in sorted({r["goal"] for r in rows}):
    R = [r for r in rows if r["goal"] == g]
    per_goal[g] = dict(mass_opt=R[0]["mass_opt"], mass_frac_k8=float(np.mean([r["mass_frac"][8] for r in R])),
                       mass_frac_k24=float(np.mean([r["mass_frac"][-1] for r in R])),
                       achieved_k24=float(np.mean([r["achieved"][-1] for r in R])),
                       complete_09=float(np.mean([r["completion"]["0.9"]["pushes"] is not None for r in R])))
out["per_goal"] = per_goal
_dst = HERE / ("results/analysis.json" if tag in ("success_states", "narrow_gd_w3") else f"results/analysis_{tag}.json")  # 2026-10-04: per-tag output so re-runs do not overwrite the original
_dst.write_text(json.dumps(out, indent=1))
for key, sm in out["summary"].items():
    print(f"\n== {key} (n={sm['n']}): start mass-in-goal {sm['k0_mass_frac']:.2f}")
    for k in (4, 8, 16, 24):
        r = sm[f"k{k}"]
        print(f"  k={k:2d}: lyap achieved {r['achieved']:.2f} | mass in goal {r['mass_frac']:.2f} ({r['mass_rel']:.2f} of optimum) | signed {r['signed_rel']:.2f} of optimum")
    for th in THETAS:
        c = sm[f"complete_{th}"]
        print(f"  complete at {th:.2f} x optimum: {c['frac']:.2f} of episodes, median {c['median_pushes']} pushes, "
              f"median time {c['median_time']}")
print("\nper goal:")
for g, r in per_goal.items():
    print(f"  {g:12s} opt {r['mass_opt']:.2f}  mass k8 {r['mass_frac_k8']:.2f} k24 {r['mass_frac_k24']:.2f}  lyap achieved k24 {r['achieved_k24']:.2f}  complete@0.9 {r['complete_09']:.2f}")
