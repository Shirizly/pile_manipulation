"""Aggregate RUN-0001 cells into the results tables (mean +/- spread over splits)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[1]
cells = json.load(open(EXP / "artifacts/RUN-0001/cells.json"))
rows = cells["rows"] if isinstance(cells, dict) else cells
agg = defaultdict(list)
for r in rows:
    agg[(r["split_kind"], r["capacity"], r["mode"], r["value_fn"])].append(r["r2"])

def s(k):
    v = agg.get(k)
    return None if not v else (float(np.mean(v)), float(np.std(v)), len(v))

VF = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
out = []
for kind in ["goal", "state"]:
    out.append(f"\n### held-out {kind.upper()}\n")
    out.append("| value_fn | capacity | mode | R2 mean | spread (sd) | n splits | goal_only (same capacity) | increment |")
    out.append("|---|---|---|---|---|---|---|---|")
    for v in VF:
        for cap in ["linear", "poly2", "mlp"]:
            go = s((kind, cap, "goal_only", v))
            for mode in ["diff", "concat", "state_only", "mean_only"]:
                m = s((kind, cap, mode, v))
                if m is None: continue
                inc = f"{m[0]-go[0]:+.3f}" if go else "n/a"
                gos = f"{go[0]:+.3f}+-{go[1]:.3f} (n={go[2]})" if go else "not run"
                out.append(f"| {v} | {cap} | {mode} | {m[0]:+.3f} | {m[1]:.3f} | {m[2]} | {gos} | {inc} |")
            if go:
                out.append(f"| {v} | {cap} | goal_only | {go[0]:+.3f} | {go[1]:.3f} | {go[2]} | — | 0 |")
txt = "\n".join(out)
print(txt)
(EXP / "results").mkdir(exist_ok=True)
open(EXP / "results/tables_capacity.md", "w").write(txt + "\n")

# cv leak table
cv = json.load(open(EXP / "artifacts/RUN-0001/cv_leak.json"))
cvr = cv["rows"] if isinstance(cv, dict) else cv
t = ["\n### Fourier order x inner-CV fold construction (held-out-STATE, 3 reps)\n",
     "| goal | value_fn | n_fourier | D | row-random R2 | lambda(row) | slate-aware R2 | lambda(slate) |",
     "|---|---|---|---|---|---|---|---|"]
by = defaultdict(dict)
for r in cvr:
    by[(r["goal"], r["value_fn"], r["n_fourier"], r["dim"])][r["inner_cv"]] = r
for k in sorted(by):
    a, b = by[k].get("row_random"), by[k].get("slate_aware")
    t.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {a['r2_mean']:+.3f}+-{a['r2_std']:.3f} | {a['lams']} "
             f"| {b['r2_mean']:+.3f}+-{b['r2_std']:.3f} | {b['lams']} |")
txt2 = "\n".join(t)
print(txt2)
open(EXP / "results/tables_cv_leak.md", "w").write(txt2 + "\n")
