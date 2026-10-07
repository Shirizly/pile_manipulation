"""EXP-0075 analysis. rel_k = in-goal mass fraction of the TRUE state after push k (soft truth, occ_for_scoring) / the goal's placement optimum (EXP-0067 vstar_width.json; 1.0 at w15/w20).
Per cell (planner x horizon x push mode): rel at k = 2, 4, 6, 8, 10 (mean), mean rel over pushes 1-10, completion at 0.9 / 0.8 x optimum within 10 pushes (Wilson CI) and within 6, median pushes to complete 0.8,
fallback / shifted push counts, plan time per decision. Paired differences between cells (same goal x start) with a bootstrap over episodes.
usage: python -u analyse.py results/main_L20.json results/main_Lfull.json [results/base64.json] -> results/analysis.md / analysis.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.goals import letter_goal_mask, mass_in_region
from simple_mpc.adapters import OCC_GRID, occ_for_scoring
HERE = REPO / "experiments/EXP-0075-closed-loop-benchmark"
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text()); VS.update(json.loads((REPO / "experiments/EXP-0067-letter-stroke-width-sweep/results/vstar_width.json").read_text()))
MASK = {}
DIST = {}


def rel_curve(states, goal):
    if goal not in MASK:
        MASK[goal] = torch.from_numpy(letter_goal_mask(goal, OCC_GRID, OCC_GRID))
    S = torch.tensor(states).float(); occ = occ_for_scoring(S)
    return (mass_in_region(occ, MASK[goal]) / occ.reshape(len(S), -1).sum(1)).numpy() / VS[goal]["mass_frac_best_placement"]


def value_curve(states, goal):
    """planner's objective value (lyapunov - 1.0 * in-goal fraction) of the TRUE (soft-truth) state after each push."""
    from Baselines.common.goals import dist_field_from_mask
    from simple_mpc.learned_mpc import lyap
    if goal not in DIST:
        DIST[goal] = torch.from_numpy(dist_field_from_mask(letter_goal_mask(goal, OCC_GRID, OCC_GRID))).float()
        MASK.setdefault(goal, torch.from_numpy(letter_goal_mask(goal, OCC_GRID, OCC_GRID)))
    occ = occ_for_scoring(torch.tensor(states).float()); f = occ.reshape(len(occ), -1)
    return (lyap(occ, DIST[goal]) - (f * MASK[goal].reshape(1, -1).float()).sum(1) / f.sum(1)).numpy()


def wilson(x, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = x / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(c - h), 3), round(float(c + h), 3)]


def load(paths):
    cells = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for e in d["episodes"]:
            if not e.get("complete") or not e.get("states"):
                continue
            rel = rel_curve(e["states"], e["goal"]); cells.setdefault(e["cell"], {})[(e["goal"], e["start"])] = dict(
                rel=rel, fb=int(sum(e.get("legal_fallback", []))), shift=sum(1 for s in e.get("legal_shift_m", []) if s > 0), ill=sum(1 for o in e.get("legal_ok", []) if not o),
                t=float(np.mean(e["plan_time_s"])), wall=e.get("wall_s", 0.0), H=e.get("H", 1),
                opt=(np.array(e["pred_cost"]) - np.diff(value_curve(e["states"], e["goal"]))[: len(e["pred_cost"])]) if e.get("H", 1) == 1 else None)
    return cells


def main():
    cells = load(sys.argv[1:]); out = {}; lines = ["| cell | n | rel0 | rel@2 | rel@4 | rel@6 | rel@8 | rel@10 | mean 1-10 | done@0.9 (<=10) | done@0.8 (<=10) | done@0.8 (<=6) | median push to 0.8 | fallbacks | plan s/decision |", "|" + "---|" * 15]
    for c, eps in sorted(cells.items()):
        R = np.stack([e["rel"] for e in eps.values()]); n = len(R); K = R.shape[1] - 1
        d9 = int((R[:, :K + 1].max(1) >= 0.9).sum()); d8 = int((R[:, :K + 1].max(1) >= 0.8).sum()); d86 = int((R[:, :min(7, K + 1)].max(1) >= 0.8).sum())
        first = [int(np.argmax(r >= 0.8)) for r in R if (r >= 0.8).any()]; med = float(np.median(first)) if len(first) > n / 2 else float("nan")
        g = lambda k: float(R[:, min(k, K)].mean())
        out[c] = dict(n=n, rel0=g(0), rel={k: g(k) for k in (1, 2, 4, 6, 8, 10)}, mean_1_10=float(R[:, 1:].mean()), done09=d9, done08=d8, done08_6=d86, wilson09=wilson(d9, n), wilson08=wilson(d8, n),
                      fallbacks=int(sum(e["fb"] for e in eps.values())), shifted=int(sum(e["shift"] for e in eps.values())), illegal=int(sum(e["ill"] for e in eps.values())), plan_s=float(np.mean([e["t"] for e in eps.values()])))
        lines.append(f"| {c} | {n} | {g(0):.2f} | {g(2):.2f} | {g(4):.2f} | {g(6):.2f} | {g(8):.2f} | {g(10):.2f} | {R[:,1:].mean():.3f} | {d9}/{n} {wilson(d9,n)} | {d8}/{n} {wilson(d8,n)} | {d86}/{n} | {med} | {out[c]['fallbacks']} | {out[c]['plan_s']:.2f} |")
    for c, eps in cells.items():
        o = [e["opt"] for e in eps.values() if e["opt"] is not None]
        if o:
            O = np.stack(o); out[c]["optimism_pred_minus_true_dv"] = float(O.mean()); out[c]["optimism_by_push"] = [round(float(x), 4) for x in O.mean(0)]
    # paired differences (mean rel over pushes 1-10 and rel@10) between cells that share goal x start
    names = sorted(cells); rng = np.random.default_rng(0); pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(cells[a]) & set(cells[b]))
            if len(common) < 8:
                continue
            for nm, f in (("mean rel 1-10", lambda r: r[1:].mean()), ("rel@10", lambda r: r[-1])):
                dlt = np.array([f(cells[b][k]["rel"]) - f(cells[a][k]["rel"]) for k in common]); bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(2000)]
                pairs.append((a, b, nm, float(dlt.mean()), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], len(common)))
    out["paired"] = [dict(a=a, b=b, metric=m, diff_b_minus_a=d, ci95=ci, n=n) for a, b, m, d, ci, n in pairs]
    md = ["# EXP-0075 closed-loop results (rel = in-goal mass / placement optimum; letters w20; DS-0006 starts 40-47)\n"] + lines + ["\n## Paired differences (b - a), bootstrap 95% CI over goal x start episodes\n", "| a | b | metric | diff | CI | n |", "|---|---|---|---|---|---|"]
    md += [f"| {a} | {b} | {m} | {d:+.3f} | {ci} | {n} |" for a, b, m, d, ci, n in pairs if m == "mean rel 1-10"]
    md += ["\n## Optimism of the chosen push (H=1 cells): mean predicted - true change of the objective value (negative = the model over-promised; lower value is better)\n", "| cell | mean | by push 1..10 |", "|---|---|---|"] + [f"| {c} | {v['optimism_pred_minus_true_dv']:+.4f} | {v['optimism_by_push']} |" for c, v in sorted(out.items()) if isinstance(v, dict) and 'optimism_pred_minus_true_dv' in v]
    (HERE / "results/analysis.md").write_text("\n".join(md) + "\n"); (HERE / "results/analysis.json").write_text(json.dumps(out, indent=1)); print("\n".join(md))


if __name__ == "__main__":
    main()
