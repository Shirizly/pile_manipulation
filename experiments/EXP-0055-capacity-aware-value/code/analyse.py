"""EXP-0055 / EXP-0056 analysis of the shared closed-loop run (results/main.json).

Per episode and push k, from the recorded particle states (soft scoring images):
  lyapunov achieved_fraction (EXP-0046 V*), in-goal mass fraction (relative to the EXP-0046
  per-goal optimum), signed-mass fraction, coverage_emd and covered_frac (METRICS.md;
  relative to the per-goal uniform-layout ceiling: best of 8 Lloyd layouts of 20 cubes).
Completion (the headline, METRICS.md `completion_time`): first k with in-goal mass >= 0.9 x
optimum ("mass"), or covered_frac >= 0.8 x ceiling ("cover"); time = sum(plan time) + t_act k.
Paired differences vs the `lyap` cell per (goal, start), bootstrap 95% CI.
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
from simple_mpc.value_functions import coverage_metrics, uniform_layout

HERE = REPO / "experiments/EXP-0055-capacity-aware-value"
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
E = [e for e in json.loads((HERE / f"results/{tag}.json").read_text())["episodes"] if e.get("complete")]
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
TACTS = (1.0, 2.0, 5.0, 10.0)
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


ceil = {}
for g in sorted({e["goal"] for e in E}):
    m = goal_mask(g); best = None
    for sd in range(8):
        xy = uniform_layout(m, seed=sd)
        occ = occ_for_scoring(torch.tensor(np.c_[xy, np.full(len(xy), 0.0025)])[None].float())
        em, cv = coverage_metrics(occ, m)
        if best is None or float(em) < best[0]:
            best = (float(em), float(cv))
    ceil[g] = dict(emd=best[0], cover=best[1])

rows = []
for e in E:
    g = e["goal"]; m = goal_mask(g); mt = torch.from_numpy(m)
    occ = occ_for_scoring(torch.tensor(e["states"]).float())
    tot = occ.reshape(len(occ), -1).sum(1)
    mf = (mass_in_region(occ, mt) / tot).numpy(); sf = (signed_mass_in_region(occ, mt) / tot).numpy()
    em, cv = (x.numpy() for x in coverage_metrics(occ, m))
    vs = VS.get(g, {}); mopt = vs.get("mass_frac_best_placement", 1.0); vstar = vs.get("vstar_map_value", 0.0)
    v = np.array(e["values"]); ach = (v[0] - v) / max(v[0] - vstar, 1e-9)
    cov_rel = cv / ceil[g]["cover"]; emd_ach = (em[0] - em) / max(em[0] - ceil[g]["emd"], 1e-9)
    pt = np.array(e["plan_time_s"])
    comp = {}
    for name, series, th in (("mass", mf / mopt, 0.9), ("cover", cov_rel, 0.8)):
        hit = np.nonzero(series[1:] >= th)[0]; k = int(hit[0]) + 1 if len(hit) else None
        comp[name] = dict(pushes=k, time={str(ta): (float(pt[:k].sum() + ta * k) if k else None) for ta in TACTS})
    rows.append(dict(cell=e["cell"], goal=g, start=e["start"], lyap_ach=ach.tolist(), mass_frac=mf.tolist(),
                     mass_rel=(mf / mopt).tolist(), signed_frac=sf.tolist(), emd=em.tolist(), emd_ach=emd_ach.tolist(),
                     cover=cv.tolist(), cover_rel=cov_rel.tolist(), completion=comp, plan_s=float(pt.mean())))

CEN = 20 * 1.0 + 20 * 2.0            # censored completion time (t_act = 2 s) for never-completed episodes
out = dict(ceilings=ceil, summary={}, paired_vs_lyap={}, per_goal={}, episodes=rows)
cells = sorted({r["cell"] for r in rows}, key=lambda c: (c != "lyap", c))
key = lambda r: (r["goal"], r["start"])
base = {key(r): r for r in rows if r["cell"] == "lyap"}
for c in cells:
    R = [r for r in rows if r["cell"] == c]
    sm = dict(n=len(R))
    for k in (5, 10, 20):
        sm[f"k{k}"] = {q: float(np.mean([r[q][min(k, len(r[q]) - 1)] for r in R]))
                       for q in ("lyap_ach", "mass_frac", "mass_rel", "signed_frac", "emd_ach", "cover_rel")}
    for name in ("mass", "cover"):
        ks = [r["completion"][name]["pushes"] for r in R]; done = [k for k in ks if k]
        tt = [r["completion"][name]["time"]["2.0"] or CEN for r in R]
        sm[f"complete_{name}"] = dict(frac=len(done) / len(ks), median_pushes=float(np.median(done)) if done else None,
                                      mean_time_tact2_censored=float(np.mean(tt)))
    out["summary"][c] = sm
    if c != "lyap":
        P = [(r, base[key(r)]) for r in R if key(r) in base]
        d = {}
        for q in ("lyap_ach", "mass_rel", "emd_ach", "cover_rel"):
            x = [a[q][-1] - b[q][-1] for a, b in P]; d[f"{q}_k20"] = dict(mean=float(np.mean(x)), ci=ci(x))
        for name in ("mass", "cover"):
            x = [(a["completion"][name]["time"]["2.0"] or CEN) - (b["completion"][name]["time"]["2.0"] or CEN) for a, b in P]
            d[f"time_{name}_tact2"] = dict(mean=float(np.mean(x)), ci=ci(x))
        out["paired_vs_lyap"][c] = d
        # the pre-registered O1/O2 test: thin goals only (quadrant_0 is the O3 regression check)
        Pt = [(a, b) for a, b in P if not a["goal"].startswith("quadrant")]
        dt = {}
        for q in ("lyap_ach", "mass_rel", "signed_frac", "emd_ach", "cover_rel"):
            x = [a[q][-1] - b[q][-1] for a, b in Pt]; dt[f"{q}_k20"] = dict(mean=float(np.mean(x)), ci=ci(x))
        for name in ("mass", "cover"):
            x = [(a["completion"][name]["time"]["2.0"] or CEN) - (b["completion"][name]["time"]["2.0"] or CEN) for a, b in Pt]
            dt[f"time_{name}_tact2"] = dict(mean=float(np.mean(x)), ci=ci(x))
        dt["n"] = len(Pt)
        out.setdefault("paired_vs_lyap_thin", {})[c] = dt
for g in sorted({r["goal"] for r in rows}):
    out["per_goal"][g] = {c: dict(mass_rel=float(np.mean([r["mass_rel"][-1] for r in rows if r["goal"] == g and r["cell"] == c])),
                                  cover_rel=float(np.mean([r["cover_rel"][-1] for r in rows if r["goal"] == g and r["cell"] == c])),
                                  lyap_ach=float(np.mean([r["lyap_ach"][-1] for r in rows if r["goal"] == g and r["cell"] == c])))
                          for c in cells if any(r["goal"] == g and r["cell"] == c for r in rows)}
p = HERE / f"results/analysis_{tag}.json"; tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); tmp.replace(p)

print("ceilings:", {g: (round(v["emd"], 2), round(v["cover"], 2)) for g, v in ceil.items()})
print(f"\n{'cell':12s} {'n':>3s} | k20: lyap_ach mass_rel signed emd_ach cover_rel | done(mass) med_k  t2 | done(cover) med_k  t2")
for c, sm in out["summary"].items():
    k = sm["k20"]; cm, cc = sm["complete_mass"], sm["complete_cover"]
    print(f"{c:12s} {sm['n']:3d} | {k['lyap_ach']:.2f} {k['mass_rel']:.2f} {k['signed_frac']:+.2f} {k['emd_ach']:.2f} {k['cover_rel']:.2f} | "
          f"{cm['frac']:.2f} {cm['median_pushes']} {cm['mean_time_tact2_censored']:.0f} | {cc['frac']:.2f} {cc['median_pushes']} {cc['mean_time_tact2_censored']:.0f}")
print("\npaired vs lyap (k=20 final, and completion time at t_act=2 s, censored at 60 s):")
for c, d in out["paired_vs_lyap"].items():
    print(f"  {c:12s} " + "  ".join(f"{q} {v['mean']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for q, v in d.items()))
print("\npaired vs lyap, THIN goals only (pre-registered O1/O2):")
for c, d in out.get("paired_vs_lyap_thin", {}).items():
    print(f"  {c:12s} n={d['n']} " + "  ".join(f"{q} {v['mean']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for q, v in d.items() if q != "n"))
print("\nper goal (mass_rel / cover_rel at k=20):")
for g, d in out["per_goal"].items():
    print(f"  {g:12s} " + "  ".join(f"{c}:{v['mass_rel']:.2f}/{v['cover_rel']:.2f}" for c, v in d.items()))
