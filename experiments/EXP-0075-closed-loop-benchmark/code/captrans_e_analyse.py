"""EXP-0075 captrans group e (2026-10-11; queue_captrans.sh; results/captrans/e_*.json): w20 H=2 / H=4 at 320 evaluations + GD, current objective, fetch pool (fl) vs OT pool (ot).
Standard closed-loop metrics (analyse_fixes.episode), paired ot - fl per H, and each 320-eval cell against the recorded 1,280-eval cell of the same pool (results/ot_pool/H{2,4}_E1280.json)
= the matched-time / budget reading left open by the OT-pool test. usage: PYTHONPATH=. python -u captrans_e_analyse.py"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE.parents[1]))
from analyse_fixes import episode, wilson
from ot_pool_analyse import boot

R = HERE / "results/captrans"; cells, raw = {}, {}
for p in [R / "e_H2_E320.json", R / "e_H4_E320.json", HERE / "results/ot_pool/H2_E1280.json", HERE / "results/ot_pool/H4_E1280.json"]:
    if p.exists():
        for e in json.loads(p.read_text())["episodes"]:
            if e.get("complete") and e.get("states"):
                cells.setdefault(e["cell"], {})[(e["goal"], e["start"])] = episode(e); raw.setdefault(e["cell"], {})[(e["goal"], e["start"])] = e
md = ["# EXP-0075 captrans group e: w20 H=2 / H=4 at 320 evaluations + GD 20 steps, current objective, fetch pool (fl) vs OT pool (ot); reference rows = the OT-pool test's 1,280-evaluation cells\n",
      "| cell | n | rel@10 | mean rel 1-10 | done@0.9 | cubes in @10 | entering / leaving | wasted 7-10 | coverage@10 | plan s/dec | GD s/dec |", "|" + "---|" * 11]; out = {"cells": {}, "paired": []}
for c in ["ct_e_fl_H2_E320", "ct_e_ot_H2_E320", "ot_fl_H2_E1280", "ot_ot_H2_E1280", "ct_e_fl_H4_E320", "ct_e_ot_H4_E320", "ot_fl_H4_E1280", "ot_ot_H4_E1280"]:
    if c not in cells:
        continue
    El = list(cells[c].values()); n = len(El); Rm = np.stack([x["rel"] for x in El]); C = np.stack([x["cov"] for x in El]); d9 = int((Rm.max(1) >= 0.9).sum()); m = lambda k: float(np.nanmean([x[k] for x in El]))
    gd = float(np.mean([v for e in raw[c].values() for v in e.get("gd_time_s", [np.nan])]))
    o = dict(n=n, rel10=float(Rm[:, -1].mean()), mean_1_10=float(Rm[:, 1:].mean()), done09=d9, cubes10=m("cubes10"), entering=m("entering"), leaving=m("leaving"), wasted_late=m("wasted_late"), cov10=float(C[:, -1].mean()), plan_s=m("t"), gd_s=gd); out["cells"][c] = o
    md.append(f"| {c} | {n} | {o['rel10']:.3f} | {o['mean_1_10']:.3f} | {d9}/{n} {wilson(d9, n)} | {o['cubes10']:.1f} | {o['entering']:.1f} / {o['leaving']:.1f} | {100 * o['wasted_late']:.0f} % | {o['cov10']:.3f} | {o['plan_s']:.2f} | {gd:.2f} |")
rng = np.random.default_rng(0); md += ["\n## Paired differences (b - a), bootstrap 95 % CI\n", "| a | b | n | d rel@10 [CI] | d mean rel 1-10 [CI] | d coverage@10 [CI] | d leaving [CI] | better / worse (mean rel) |", "|" + "---|" * 8]
for ca, cb in [("ct_e_fl_H2_E320", "ct_e_ot_H2_E320"), ("ct_e_fl_H4_E320", "ct_e_ot_H4_E320"), ("ct_e_fl_H2_E320", "ot_fl_H2_E1280"), ("ct_e_ot_H2_E320", "ot_ot_H2_E1280"), ("ct_e_fl_H4_E320", "ot_fl_H4_E1280"), ("ct_e_ot_H4_E320", "ot_ot_H4_E1280"), ("ct_e_fl_H2_E320", "ct_e_fl_H4_E320"), ("ct_e_ot_H2_E320", "ct_e_ot_H4_E320")]:
    common = sorted(set(cells.get(ca, {})) & set(cells.get(cb, {})))
    if not common:
        continue
    row = dict(a=ca, b=cb, n=len(common)); txt = []
    for nm, f in (("rel10", lambda x: x["rel"][-1]), ("mean_1_10", lambda x: x["rel"][1:].mean()), ("cov10", lambda x: x["cov"][-1]), ("leaving", lambda x: x["leaving"])):
        dl = np.array([f(cells[cb][k]) - f(cells[ca][k]) for k in common]); d, ci = boot(dl, rng); row[nm] = dict(diff=d, ci95=ci, better=int((dl > 0).sum()), worse=int((dl < 0).sum())); txt.append(f"{d:+.3f} {ci}")
    out["paired"].append(row); md.append(f"| {ca} | {cb} | {len(common)} | " + " | ".join(txt) + f" | {row['mean_1_10']['better']} / {row['mean_1_10']['worse']} |")
(R / "summary_e.md").write_text("\n".join(md) + "\n"); (R / "summary_e.json").write_text(json.dumps(out, indent=1)); print("\n".join(md))
