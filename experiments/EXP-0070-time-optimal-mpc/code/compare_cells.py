"""EXP-0070: paired equal-wall-time in-goal mass and task time of every cell vs a chosen BASE cell of one results file.
Usage: compare_cells.py TAG BASE_CELL [BASE_TAG]  (BASE_TAG: take the base cell's episodes from another results file)"""
import json, sys, subprocess
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
tag, base_cell = sys.argv[1:3]; base_tag = sys.argv[3] if len(sys.argv) > 3 else tag
def rows(t):
    p = HERE / f"results/{t}_episodes.json"
    if not p.exists() or p.stat().st_mtime < (HERE / f"results/{t}.json").stat().st_mtime:
        subprocess.run([sys.executable, str(HERE / "code/analyse_pilot.py"), t], check=True, capture_output=True)
    return json.loads(p.read_text())
R = rows(tag); RB = rows(base_tag) if base_tag != tag else R; rng = np.random.default_rng(0)
def m_at(r, T):
    t = np.cumsum(np.array(r["plan"]) + np.array(r["t_act"]["total"])); return r["mass_rel"][int(np.searchsorted(t, T, side="right"))]
def ttime(r):
    k = r["kstar"] or len(r["plan"]); return float(np.sum(r["plan"][:k]) + np.sum(r["t_act"]["total"][:k]))
B = {(r["goal"], r["start"]): r for r in RB if r["cell"] == base_cell}
out = {}
for c in sorted({r["cell"] for r in R} - {base_cell}):
    for fam in ("ALL", "quadrant", "two_squares", "letter"):
        P = [(r, B[(r["goal"], r["start"])]) for r in R if r["cell"] == c and (fam == "ALL" or r["fam"] == fam) and (r["goal"], r["start"]) in B]
        if not P:
            continue
        row = {}
        for T in (4, 8, 12, 16, 20, 30):
            d = np.array([m_at(a, T) - m_at(b, T) for a, b in P]); bs = rng.choice(d, (4000, len(d))).mean(1)
            row[f"T{T}"] = (float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), float(np.mean([m_at(b, T) for _, b in P])))
        d = np.array([ttime(a) - ttime(b) for a, b in P]); bs = rng.choice(d, (4000, len(d))).mean(1)
        row["task_time"] = (float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)))
        row["done"] = (float(np.mean([a["kstar"] is not None for a, _ in P])), float(np.mean([b["kstar"] is not None for _, b in P])))
        row["plan_per_push"] = (float(np.mean([np.mean(a["plan"]) for a, _ in P])), float(np.mean([np.mean(b["plan"]) for _, b in P])))
        out[f"{c}|{fam}"] = row
        print(f"{c:18s} {fam:11s} n{len(P):3d} " + "  ".join(f"{k}: {v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]" for k, v in row.items() if k.startswith("T"))
              + f"  | task time {row['task_time'][0]:+.2f}[{row['task_time'][1]:+.2f},{row['task_time'][2]:+.2f}]s  done {row['done'][0]:.2f} vs {row['done'][1]:.2f}  plan/push {row['plan_per_push'][0]:.3f} vs {row['plan_per_push'][1]:.3f}")
(HERE / f"results/compare_{tag}_vs_{base_tag}_{base_cell.replace('@', '_at_')}.json").write_text(json.dumps(out, indent=1))
