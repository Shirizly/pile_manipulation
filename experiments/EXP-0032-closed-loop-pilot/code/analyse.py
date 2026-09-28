"""EXP-0032 pilot analysis: per model x planner, closed-loop improvement
(V0 - V_T, soft truth; lyapunov is a cost so positive = better), its spread
across episodes (sizes the real study), predicted-vs-true per-step dv, and the
models' OFFLINE slateN on DS-0006 (EXP-0030) beside it. Pilot: descriptive."""
import json, itertools
from pathlib import Path
import numpy as np
from scipy import stats
REPO = Path(__file__).resolve().parents[3]
eps = [e for e in json.load(open(REPO / "experiments/EXP-0032-closed-loop-pilot/results/episodes.json"))["episodes"] if e.get("complete")]
off = json.load(open(REPO / "experiments/EXP-0030-state-superiority-ds0005/results/analysis.json"))["per_vf"]["lyapunov"]["A3"]["model_means"]
models = sorted({e["model"] for e in eps}); planners = ["rank", "gd", "cem"]
print(f"{len(eps)} complete episodes")
res = {}
for m in models:
    for pl in planners:
        E = [e for e in eps if e["model"] == m and e["planner"] == pl]
        if not E: continue
        imp = np.array([e["values"][0] - e["values"][-1] for e in E])
        pred = np.concatenate([e["pred_dv"] for e in E]); true = np.concatenate([e["true_dv"] for e in E])
        r = float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 2 else float("nan")
        res[f"{m}|{pl}"] = dict(n=len(E), improve_mean=float(imp.mean()), improve_sd=float(imp.std(ddof=1)),
                                pred_true_r=r, optimism=float((pred - true).mean()),
                                evals_per_step=float(np.mean([np.mean(e["n_evals"]) for e in E])),
                                wall_s=float(np.mean([e.get("wall_s", np.nan) for e in E])))
        v = res[f"{m}|{pl}"]
        print(f"{m:28s} {pl:5s} n={v['n']:2d} improve {v['improve_mean']:+.3f} (sd {v['improve_sd']:.3f})  "
              f"pred-vs-true r {v['pred_true_r']:+.2f}  optimism {v['optimism']:+.4f}  evals/step {v['evals_per_step']:.0f}  "
              f"offline slateN {off.get(m, float('nan')):.3f}")
# paired planner comparison within model (episodes matched on goal x start)
print("\nplanner differences (paired over goal x start):")
for m in models:
    for a, b in itertools.combinations(planners, 2):
        key = lambda e: (e["goal"], e["start"])
        A = {key(e): e["values"][0] - e["values"][-1] for e in eps if e["model"] == m and e["planner"] == a}
        B = {key(e): e["values"][0] - e["values"][-1] for e in eps if e["model"] == m and e["planner"] == b}
        ks = sorted(set(A) & set(B))
        if len(ks) > 2:
            d = np.array([A[k] - B[k] for k in ks])
            print(f"   {m:28s} {a}-{b}: {d.mean():+.3f} (sd {d.std(ddof=1):.3f}, n {len(ks)}, wins {int((d > 0).sum())}-{int((d < 0).sum())})")
json.dump(res, open(REPO / "experiments/EXP-0032-closed-loop-pilot/results/summary.json", "w"), indent=1)
