"""EXP-0075 mass paired comparison analysis (queue_mass.sh, results/mass/*.json): per (width, H) arms ot / fl / van; metrics rel@10, mean rel 1-10, done@0.9, leaving, coverage@10,
plan s/decision; paired differences (bootstrap 95 % CI over episodes, 4,000 resamples) ot-fl, ot-van, fl-van with wins/ties/losses (mean rel, tie |d| < 0.005); by-letter table.
Main population = starts >= 48 (fresh); starts 40-47 reported separately as cross-check. usage: PYTHONPATH=. python -u mass_analyse.py"""
import json, sys, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE.parents[1]))
from analyse_fixes import episode, wilson
R = HERE / "results/mass"; eps = {}   # (w,H,arm) -> {(goal,start): episode}
for p in sorted(R.glob("*.json")):
    if p.name.startswith("summary"):
        continue
    for e in json.loads(p.read_text())["episodes"]:
        m = re.match(r"mass_w(\d+)_H(\d)_(\w+)", e["cell"])
        if e.get("complete") and e.get("states") and m:
            eps.setdefault((int(m[1]), int(m[2]), m[3]), {})[(e["goal"], e["start"])] = episode(e)
rng = np.random.default_rng(0)


def bt(d, B=4000):
    b = rng.choice(d, (B, len(d))).mean(1); return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def sel(w, H, arm, cross):
    return {k: v for k, v in eps.get((w, H, arm), {}).items() if (k[1] < 48) == cross}


md = ["# EXP-0075 mass paired comparison (ens2bal, planner seed 0, letters O/T/S/X, 10 pushes; ot / fl = 320 evals + GD 20 + fetch + legal, cumspill H>1; van = plain bank CEM 1,280 evals, terminal obj.)\n",
      "Plan s/decision measured while the GPU was shared with another job (contended). Episodes = (letter, start) pairs.\n"]; out = {}
for cross in (False, True):
    md.append("\n## " + ("CROSS-CHECK starts 40-47" if cross else "MAIN: fresh starts >= 48") + "\n")
    md += ["| w | H | arm | n | rel@10 | mean rel 1-10 | done@0.9 | cubes leaving | coverage@10 | plan s/dec |", "|" + "---|" * 10]
    for w in (20, 15, 10):
        for H in (1, 2, 4):
            for arm in ("ot", "fl", "van"):
                E = list(sel(w, H, arm, cross).values())
                if not E:
                    continue
                Rm = np.stack([x["rel"] for x in E]); n = len(E); d9 = int((Rm.max(1) >= 0.9).sum())
                md.append(f"| {w} | {H} | {arm} | {n} | {Rm[:, -1].mean():.3f} | {Rm[:, 1:].mean():.3f} | {d9}/{n} | {np.mean([x['leaving'] for x in E]):.2f} | {np.mean([x['cov'][-1] for x in E]):.3f} | {np.mean([x['t'] for x in E]):.2f} |")
    md += ["\nPaired b - a (bootstrap 95 % CI over episodes); W/T/L on mean rel 1-10 (tie |d| < 0.005)\n", "| w | H | b - a | n | d rel@10 | d mean rel 1-10 | d coverage@10 | d leaving | W / T / L |", "|" + "---|" * 9]
    for w in (20, 15, 10):
        for H in (1, 2, 4):
            for b, a in (("ot", "fl"), ("ot", "van"), ("fl", "van")):
                A, B = sel(w, H, a, cross), sel(w, H, b, cross); common = sorted(set(A) & set(B))
                if len(common) < 2:
                    continue
                row = []; wtl = None
                for nm, f in (("rel10", lambda x: x["rel"][-1]), ("mean", lambda x: x["rel"][1:].mean()), ("cov", lambda x: x["cov"][-1]), ("lea", lambda x: x["leaving"])):
                    d = np.array([f(B[k]) - f(A[k]) for k in common]); m_, lo, hi = bt(d); row.append(f"{m_:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                    out[f"{'x' if cross else 'm'}_w{w}_H{H}_{b}-{a}_{nm}"] = [m_, lo, hi, len(common)]
                    if nm == "mean":
                        wtl = (int((d > .005).sum()), int((abs(d) <= .005).sum()), int((d < -.005).sum()))
                md.append(f"| {w} | {H} | {b} - {a} | {len(common)} | " + " | ".join(row) + f" | {wtl[0]} / {wtl[1]} / {wtl[2]} |")
    if not cross:
        md += ["\nBy letter: mean rel 1-10 (n episodes per letter) and paired differences\n", "| w | H | letter | n | ot | fl | van | ot-fl | ot-van | fl-van |", "|" + "---|" * 10]
        for w in (20, 15, 10):
            for H in (1, 2, 4):
                for L in "OTSX":
                    g = f"letter_{L}_w{w}"; D = {arm: {k: v for k, v in sel(w, H, arm, False).items() if k[0] == g} for arm in ("ot", "fl", "van")}
                    if not all(D.values()):
                        continue
                    common = sorted(set(D["ot"]) & set(D["fl"]) & set(D["van"])); f = lambda arm: np.array([D[arm][k]["rel"][1:].mean() for k in common])
                    md.append(f"| {w} | {H} | {L} | {len(common)} | {f('ot').mean():.3f} | {f('fl').mean():.3f} | {f('van').mean():.3f} | {(f('ot') - f('fl')).mean():+.3f} | {(f('ot') - f('van')).mean():+.3f} | {(f('fl') - f('van')).mean():+.3f} |")
(R / "summary.md").write_text("\n".join(md) + "\n"); (R / "summary.json").write_text(json.dumps(out, indent=1)); print("\n".join(md))
