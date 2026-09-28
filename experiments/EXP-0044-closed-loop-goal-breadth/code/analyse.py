"""EXP-0044 analysis (DESIGN.md P1-P4 + variance components / power table).
X[model, planner, goal, start] = improvement V0 - V8. -> results/analysis.json"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.paired_stats import paired_comparison
HERE = REPO / "experiments/EXP-0044-closed-loop-goal-breadth"
E = [e for e in json.loads((HERE / "results/goal_breadth_tuned.json").read_text())["episodes"] if e["complete"]]
M = ["nfd_3ch_randlen", "nfd_3ch_randlen_seed1", "linear_switched_soft", "nfd_residual_worldframe_noaug_ep43"]
P = ["gd", "cem"]
G = sorted({e["goal"] for e in E}); S = sorted({e["start"] for e in E})
X = np.full((len(M), len(P), len(G), len(S)), np.nan); NE = np.full_like(X, np.nan)
for e in E:
    i = (M.index(e["model"]), P.index(e["planner"]), G.index(e["goal"]), S.index(e["start"]))
    X[i] = e["values"][0] - e["values"][-1]; NE[i] = np.mean(e["n_evals"])
assert not np.isnan(X).any()
out = dict(models=M, planners=P, goals=G, starts=S, means={}, pairs={}, per_goal_best={}, variance={}, power={})
rng = np.random.default_rng(0)
for p, pl in enumerate(P):
    Y = X[:, p].reshape(len(M), -1)
    out["means"][pl] = {m: dict(mean=float(Y[i].mean()), sd=float(Y[i].std(ddof=1)), evals=float(NE[i, p].mean()))
                        for i, m in enumerate(M)}
    out["pairs"][pl] = paired_comparison(Y, M)
    # per-goal: best model, and whether worldframe is within the best's paired CI (tied for best)
    w = M.index("nfd_residual_worldframe_noaug_ep43"); pg = {}
    for g, gn in enumerate(G):
        mu = X[:, p, g].mean(1); b = int(mu.argmax())
        d = X[b, p, g] - X[w, p, g]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        pg[gn] = dict(best=M[b], worldframe_minus_best=float(-d.mean()), tied=bool(b == w or np.quantile(bs, .025) <= 0))
    out["per_goal_best"][pl] = pg
    # variance components of the model effect (episode effects removed)
    D = X[:, p] - X[:, p].mean(0, keepdims=True)
    a = D.mean((1, 2), keepdims=True); mg = D.mean(2, keepdims=True) - a; ms = D.mean(1, keepdims=True) - a
    r = D - a - mg - ms
    k = len(M) / (len(M) - 1)
    vg, vs, vr = 2 * k * mg.var(), 2 * k * ms.var(), 2 * k * r.var()
    out["variance"][pl] = dict(pairdiff_goal=float(vg), pairdiff_start=float(vs), pairdiff_resid=float(vr),
                               model_main_sd=float(a.std()))
    out["power"][pl] = {f"{ng}x{ns}": float(2.8 * np.sqrt(vg / ng + vs / ns + vr / (ng * ns)))
                        for ng, ns in [(4, 4), (12, 8), (24, 8), (24, 16), (48, 16)]}
# P3: tuned GD - tuned CEM, per model and averaged
out["gd_minus_cem"] = {m: float((X[i, 0] - X[i, 1]).mean()) for i, m in enumerate(M)}
d = (X[:, 0] - X[:, 1]).mean(0).reshape(-1)
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
out["gd_minus_cem_avg"] = dict(mean=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
# P4: tuned vs default on EXP-0039's 16 cells (default = mean of EXP-0039 and EXP-0043 replicate)
old = {}
for f in ("experiments/EXP-0039-closed-loop-metric-validity/results/episodes.json",
          "experiments/EXP-0043-batched-closed-loop/results/replicate_exp0039.json"):
    for e in json.loads((REPO / f).read_text())["episodes"]:
        if e.get("complete") and e["planner"] in P:
            old.setdefault((e["model"], e["planner"], e["goal"], e["start"]), []).append(e["values"][0] - e["values"][-1])
out["tuned_minus_default"] = {}
for p, pl in enumerate(P):
    for i, m in enumerate(M):
        ks = [k for k in old if k[0] == m and k[1] == pl]
        d = np.array([X[i, p, G.index(k[2]), S.index(k[3])] - np.mean(old[k]) for k in ks])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        out["tuned_minus_default"][f"{m}/{pl}"] = dict(n=len(d), mean=float(d.mean()),
                                                       ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
(HERE / "results/analysis.json").write_text(json.dumps(out, indent=1))
for pl in P:
    print(f"\n== {pl}: means (sd) [evals]")
    for m, r in sorted(out["means"][pl].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {m:36s} {r['mean']:.3f} ({r['sd']:.3f}) [{r['evals']:.0f}]")
    for r in out["pairs"][pl]:
        print(f"  {r['a']:36s} - {r['b']:36s} {r['mean_diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] Holm {r['p_holm']:.3g}")
    pg = out["per_goal_best"][pl]
    print("  worldframe tied-or-best on", sum(v["tied"] for v in pg.values()), "/", len(pg), "goals;",
          {g: v["best"][:12] for g, v in pg.items()})
    print("  variance parts", {k: f"{v:.2e}" if "pair" in k else round(v, 4) for k, v in out["variance"][pl].items()})
    print("  resolvable gap by design", {k: round(v, 3) for k, v in out["power"][pl].items()})
print("\nGD - CEM per model", {m: round(v, 3) for m, v in out["gd_minus_cem"].items()}, "avg", out["gd_minus_cem_avg"])
print("tuned - default (EXP-0039 16 cells):")
for k, v in out["tuned_minus_default"].items():
    print(f"  {k:44s} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] n={v['n']}")
