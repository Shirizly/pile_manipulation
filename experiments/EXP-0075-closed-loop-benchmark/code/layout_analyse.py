"""EXP-0075 layout-transport analysis (2026-10-12; queue_layout.sh; results/layout/). Cells lo_w<W>_H<H>_{cur,lay}. Metrics via analyse_fixes.episode (rel, done@0.9, coverage@10, leaving ...) plus transport to the
uniform_layout slots (Hungarian, mm, TRUE state), plan time; paired bootstrap 95 % CIs (lay - cur) over the common (goal, start) episodes. usage: PYTHONPATH=. python -u layout_analyse.py"""
import json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "experiments/EXP-0043-batched-closed-loop/code"))
from analyse_fixes import episode, wilson
from layout_term import hungarian_cost
from simple_mpc.value_functions import uniform_layout
from batched_closed_loop import goal_mask
R = HERE / "results/layout"; PAT = re.compile(r"lo_w(\d+)_H(\d)_(cur|lay)$"); rng = np.random.default_rng(0)
boot = lambda dl, B=2000: (float(dl.mean()), [round(float(np.percentile([rng.choice(dl, len(dl)).mean() for _ in range(B)], q)), 3) for q in (2.5, 97.5)])
cells, raw, SL = {}, {}, {}
for p in sorted(R.glob("w*_H*.json")):
    for e in json.loads(p.read_text())["episodes"]:
        if e.get("complete") and e.get("states") and PAT.match(e["cell"]):
            g = e["goal"]; SL.setdefault(g, uniform_layout(goal_mask(g), 20)); x = episode(e); S = np.array(e["states"])[:, :, :2]
            x["tl"] = np.array([1000 * hungarian_cost(S[k], SL[g]) for k in range(len(S))]); cells.setdefault(e["cell"], {})[(g, e["start"])] = x; raw.setdefault(e["cell"], {})[(g, e["start"])] = e
out = {"cells": {}, "paired": []}; md = ["# EXP-0075 layout-transport objective (ens2bal, 10 pushes, legal scoring, GD 20, 320 evals, fetch + OT pool in both arms; starts 40-47 x O/T/S/X = 32 paired episodes; H=2 cumspill). cur = current objective, lay = + 2.0 x Sinkhorn W1 to uniform_layout slots\n",
      "| cell | n | rel@10 | mean rel 1-10 | done@0.9 | coverage@10 | cov>=0.8 | transport-to-layout mm @10 | mean 1-10 | cubes leaving / ep | plan s/dec |", "|" + "---|" * 11]
for c in sorted(cells, key=lambda c: (-int(PAT.match(c).group(1)), PAT.match(c).group(2), PAT.match(c).group(3))):
    El = list(cells[c].values()); n = len(El); Rm = np.stack([x["rel"] for x in El]); C = np.stack([x["cov"] for x in El]); T = np.stack([x["tl"] for x in El]); d9 = int((Rm.max(1) >= 0.9).sum()); c8 = int((C[:, -1] >= 0.8).sum())
    o = dict(n=n, rel10=float(Rm[:, -1].mean()), mean_1_10=float(Rm[:, 1:].mean()), done09=d9, cov10=float(C[:, -1].mean()), cov08=c8, tl10=float(T[:, -1].mean()), tl_mean=float(T[:, 1:].mean()), leaving=float(np.mean([x["leaving"] for x in El])), plan_s=float(np.mean([x["t"] for x in El])))
    out["cells"][c] = o; md.append(f"| {c} | {n} | {o['rel10']:.3f} | {o['mean_1_10']:.3f} | {d9}/{n} | {o['cov10']:.3f} | {c8}/{n} | {o['tl10']:.1f} | {o['tl_mean']:.1f} | {o['leaving']:.2f} | {o['plan_s']:.2f} |")
md += ["\n## Paired lay - cur, bootstrap 95 % CI\n", "| group | n | d coverage@10 | d transport-to-layout mm @10 | d transport mean 1-10 | d rel@10 | d mean rel 1-10 | d leaving | d done@0.9 | rule: coverage | rule: transport+rel |", "|" + "---|" * 11]
for W in (15, 10):
    for H in (1, 2):
        a, b = f"lo_w{W}_H{H}_cur", f"lo_w{W}_H{H}_lay"; common = sorted(set(cells.get(a, {})) & set(cells.get(b, {})))
        if not common:
            continue
        row = dict(group=f"w{W}_H{H}", n=len(common)); txt = []
        for nm, f in (("cov10", lambda x: x["cov"][-1]), ("tl10", lambda x: x["tl"][-1]), ("tl_mean", lambda x: x["tl"][1:].mean()), ("rel10", lambda x: x["rel"][-1]), ("mean_1_10", lambda x: x["rel"][1:].mean()), ("leaving", lambda x: x["leaving"]), ("done09", lambda x: float(x["rel"].max() >= 0.9))):
            dl = np.array([f(cells[b][k]) - f(cells[a][k]) for k in common]); d, ci = boot(dl); row[nm] = dict(diff=d, ci95=ci); txt.append(f"{d:+.3f} {ci}")
        r1 = row["cov10"]["diff"] >= 0.05 and row["cov10"]["ci95"][0] > 0; r2 = row["tl10"]["ci95"][1] < 0 and row["rel10"]["diff"] > -0.02; row["rule_cov"], row["rule_tl_rel"] = bool(r1), bool(r2)
        out["paired"].append(row); md.append(f"| w{W} H{H} | {len(common)} | " + " | ".join(txt) + f" | {'MET' if r1 else 'no'} | {'MET' if r2 else 'no'} |")
(R / "summary.md").write_text("\n".join(md) + "\n"); (R / "summary.json").write_text(json.dumps(out, indent=1)); print("\n".join(md))
