"""EXP-0075: summary of gd_budget (model-predicted terminal value, 10 tasks) and its simulator replay (results/gd_budget_sim.json)."""
import json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1] / "results"; PL = ["ref", "gd_ref", "gd5", "gd10", "cem", "gd_full"]
T = [json.load(open(f)) for f in sorted((R / "gd_budget").glob("*.json"))]; P = np.array([[t[p]["cost"] for p in PL] for t in T])
ev = {"ref": "1,280", "gd_ref": "1,280 + GD", "gd5": "10,000 + GD", "gd10": "20,000 + GD", "cem": "210,000", "gd_full": "210,000 + GD"}
out = [f"# gd_budget: {len(T)} tasks, H=4, predicted terminal value change (lower = better)\n", "| task | " + " | ".join(PL) + " |", "|---|" + "---|" * len(PL)]
for t, r in zip(T, P): out.append(f"| {t['goal'][7]}/{t['start']} | " + " | ".join(f"{x:+.3f}" for x in r) + " |")
out.append("| **mean** | " + " | ".join(f"**{x:+.3f}**" for x in P.mean(0)) + " |"); out.append("| sd/sqrt(n) | " + " | ".join(f"{x:.3f}" for x in P.std(0, ddof=1) / np.sqrt(len(P))) + " |")
out.append("\nevaluations before GD: " + ", ".join(f"{p}: {ev[p]}" for p in PL) + f"; GD = 150 Adam steps x 24 sequences (3,600 differentiable evaluations, ~26 s) in every gd_* column.")
out.append("\nPaired differences (mean [min, max] over tasks; negative = first is better):")
for a, b in (("gd_ref", "ref"), ("gd5", "gd_ref"), ("gd10", "gd5"), ("gd_full", "gd10"), ("gd_full", "gd_ref"), ("gd10", "cem"), ("gd_ref", "cem")):
    d = P[:, PL.index(a)] - P[:, PL.index(b)]; out.append(f"* {a} - {b}: {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}]  (better in {(d < 0).sum()}/{len(d)})")
fr = 1 - P[:, PL.index("gd_full")] / P[:, PL.index("gd_full")]; best = P.min(1)
out.append("\nFraction of the best plan found per task (value / min over the six plans): " + ", ".join(f"{p}: {np.mean(P[:, i] / best):.2f}" for i, p in enumerate(PL)))
if (R / "gd_budget_sim.json").exists():
    S = json.load(open(R / "gd_budget_sim.json")); out.append("\n## Simulator replay (open loop, pushes legalised; one run per plan)\n"); out.append("| plan | predicted | simulator | optimism (pred - sim) | sd of optimism | shifted pushes /plan |"); out.append("|---|---|---|---|---|---|")
    for p in PL:
        r = [s for s in S if s["plan"] == p]; pr = np.array([s["pred_terminal"] for s in r]); tr = np.array([s["true_terminal"] for s in r]); out.append(f"| {p} | {pr.mean():+.3f} | {tr.mean():+.3f} | {(pr - tr).mean():+.3f} | {(pr - tr).std(ddof=1):.3f} | {np.mean([s['shifted_pushes'] for s in r]):.2f} |")
    out.append("\nSimulator paired differences (true terminal value change; negative = first is better):")
    for a, b in (("gd_ref", "ref"), ("gd5", "gd_ref"), ("gd10", "gd5"), ("gd_full", "gd10"), ("gd10", "cem")):
        d = np.array([[s["true_terminal"] for s in S if s["plan"] == a and s["goal"] == t["goal"] and s["start"] == t["start"]][0] - [s["true_terminal"] for s in S if s["plan"] == b and s["goal"] == t["goal"] and s["start"] == t["start"]][0] for t in T])
        out.append(f"* {a} - {b}: {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}]  (better in {(d < 0).sum()}/{len(d)})")
(R / "gd_budget_summary.md").write_text("\n".join(out)); print("\n".join(out))
