"""EXP-0045 analysis: value after k pushes per (model, planner, budget), the actual planning
time per push, and the fixed-total-time trade-off. For total time T and execution time
t_act per push, k(b) = floor(T / (plan_time(b) + t_act)) pushes (capped at 24);
improvement V0 - V_k(b). -> results/analysis.json"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
E = [e for e in json.loads((HERE / "results/time_vs_actions.json").read_text())["episodes"] if e["complete"]]
M = ["nfd_residual_worldframe_noaug_ep43", "linear_switched_soft"]; P = ["gd", "cem"]
B = ["b0.03", "b0.1", "b0.3", "b1.0"]
keys = sorted({(e["goal"], e["start"]) for e in E})
V = np.full((len(M), len(P), len(B), len(keys), 25), np.nan); PT = np.full((len(M), len(P), len(B), len(keys)), np.nan)
EV = np.full_like(PT, np.nan)
for e in E:
    i = (M.index(e["model"]), P.index(e["planner"]), B.index(e["cell"]), keys.index((e["goal"], e["start"])))
    V[i] = e["values"][0] - np.array(e["values"]); PT[i] = np.mean(e["plan_time_s"]); EV[i] = np.mean(e["n_evals"])
assert not np.isnan(V).any()
rng = np.random.default_rng(0)
def ci(x):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
out = dict(models=M, planners=P, budgets=B, curves={}, plan_time={}, evals={}, tradeoff={}, checks={})
for i, m in enumerate(M):
    for p, pl in enumerate(P):
        for b, bn in enumerate(B):
            k = f"{m}/{pl}/{bn}"
            out["curves"][k] = V[i, p, b].mean(0).tolist()
            out["plan_time"][k] = float(PT[i, p, b].mean()); out["evals"][k] = float(EV[i, p, b].mean())
# Q1 concavity: mean second difference of the mean curve <= 0 over k = 1..23
out["checks"]["Q1_frac_concave_steps"] = {k: float(np.mean(np.diff(np.array(c), 2) <= 0.002)) for k, c in out["curves"].items()}
# Q2 budget effects at k = 1 and k = 8 (paired over episodes)
q2 = {}
for i, m in enumerate(M):
    for p, pl in enumerate(P):
        for kk in (1, 8, 24):
            for lo, hi in (("b0.1", "b1.0"), ("b0.03", "b0.3"), ("b0.03", "b1.0")):
                d = V[i, p, B.index(lo), :, kk] - V[i, p, B.index(hi), :, kk]
                q2[f"{m}/{pl}/k{kk}/{lo}-{hi}"] = dict(mean=float(d.mean()), ci=ci(d))
out["checks"]["Q2_budget_effects"] = q2
# Q4 model gap vs budget, at k = 8 and 24
q4 = {}
for p, pl in enumerate(P):
    for b, bn in enumerate(B):
        for kk in (8, 24):
            d = V[0, p, b, :, kk] - V[1, p, b, :, kk]
            q4[f"{pl}/{bn}/k{kk}"] = dict(mean=float(d.mean()), ci=ci(d))
out["checks"]["Q4_worldframe_minus_linear"] = q4
# trade-off: for each (T, t_act) the value per budget, and the best budget
for t_act in (0.0, 0.5, 2.0, 5.0, 10.0):
    for T_label, T in (("8 pushes at 1 s", 8 * (1.0 + t_act)), ("4 pushes at 1 s", 4 * (1.0 + t_act)),
                       ("16 pushes at 1 s", 16 * (1.0 + t_act))):
        for i, m in enumerate(M):
            for p, pl in enumerate(P):
                row = {}
                for b, bn in enumerate(B):
                    pt = PT[i, p, b].mean()
                    k = int(min(24, np.floor(T / (pt + t_act + 1e-12))))
                    row[bn] = dict(k=k, value=float(V[i, p, b, :, k].mean()), capped=bool(T / (pt + t_act + 1e-12) > 24))
                best = max(row, key=lambda x: row[x]["value"])
                out["tradeoff"][f"t_act={t_act}/T={T_label}/{m}/{pl}"] = dict(T_s=T, per_budget=row, best=best)
(HERE / "results/analysis.json").write_text(json.dumps(out, indent=1))
print("actual plan time per push / evals:")
for k in out["plan_time"]:
    c = out["curves"][k]
    print(f"  {k:52s} {out['plan_time'][k]:.3f}s  evals {out['evals'][k]:6.0f}  V(1) {c[1]:.3f} V(4) {c[4]:.3f} V(8) {c[8]:.3f} V(16) {c[16]:.3f} V(24) {c[24]:.3f}")
print("\nQ2:"); [print(f"  {k:52s} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]") for k, v in q2.items()]
print("\nQ4 worldframe - linear:"); [print(f"  {k:20s} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]") for k, v in q4.items()]
print("\ntrade-off (T = 8 pushes at 1 s + t_act each):")
for k, v in out["tradeoff"].items():
    if "T=8 pushes" in k:
        print(f"  {k:80s} best {v['best']:6s} " + " ".join(f"{b}:k{r['k']}{'*' if r['capped'] else ''}={r['value']:.3f}" for b, r in v["per_budget"].items()))
