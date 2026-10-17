"""EXP-0075 layout-weight study analysis (2026-10-12; queue_layout_lam.sh; results/layout_lam/). Cells lw_w<W>_lam<L>, 16 pushes, H=1. Metrics: coverage@10 / @16 (skeleton_coverage_r5), cov>=0.8 episodes, rel@10/@16,
mean rel 1-16, done@0.9, transport to uniform_layout slots (Hungarian, mm, true state), per-cube placement error of the FINAL state (Hungarian-assigned distance to its slot: mean, median, share <=5 / <=10 mm),
cubes leaving. Paired bootstrap 95 % CI (2000 resamples, seed 0) over common (goal, start) episodes for lam vs control and lam 20 vs lam 2. With --diag (loads the model; run when no other job of ours is on the GPU):
per executed push, model-PREDICTED vs TRUE change of the Sinkhorn layout term (lam=1 units, true = Sinkhorn of the rasterised next state) and exact Hungarian true change (mm); optimism = true - predicted.
usage: PYTHONPATH=. python -u layout_lam_analyse.py [--diag]"""
import json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "experiments/EXP-0043-batched-closed-loop/code"))
from analyse_fixes import episode
from layout_term import hungarian_cost
from scipy.optimize import linear_sum_assignment
from simple_mpc.value_functions import uniform_layout
from batched_closed_loop import goal_mask
R = HERE / "results/layout_lam"; PAT = re.compile(r"lw_w(\d+)_lam([\d.]+)$"); rng = np.random.default_rng(0)
boot = lambda dl, B=2000: (float(dl.mean()), [round(float(np.percentile([rng.choice(dl, len(dl)).mean() for _ in range(B)], q)), 3) for q in (2.5, 97.5)])
cells, raw, SL = {}, {}, {}
for p in sorted(R.glob("w*_lam*.json")):
    for e in json.loads(p.read_text())["episodes"]:
        if e.get("complete") and e.get("states") and PAT.match(e["cell"]):
            g = e["goal"]; SL.setdefault(g, uniform_layout(goal_mask(g), 20)); x = episode(e); S = np.array(e["states"])[:, :, :2]
            x["tl"] = np.array([1000 * hungarian_cost(S[k], SL[g]) for k in range(len(S))])
            P = np.asarray(S[-1]); D = np.linalg.norm(P[:, None] - SL[g][None], axis=-1); r, c = linear_sum_assignment(D); x["perr"] = 1000 * D[r, c]
            cells.setdefault(e["cell"], {})[(g, e["start"])] = x; raw.setdefault(e["cell"], {})[(g, e["start"])] = e
key = lambda c: (-int(PAT.match(c).group(1)), float(PAT.match(c).group(2)))
out = {"cells": {}, "paired": []}; md = ["# EXP-0075 layout weight study (ens2bal, H=1, 16 pushes, legal scoring, GD 20, 320 evals, fetch + OT pool; starts 40-47 x O/T/S/X = 32 episodes/cell). lam 0 = control (objective cur)\n",
      "| cell | n | cov@10 | cov@16 | cov>=0.8 @10 / @16 | rel@10 | rel@16 | mean rel 1-16 | done@0.9 | transport mm @10 / @16 | placement err mm mean / median (final) | share <=5 / <=10 mm | leaving/ep | plan s |", "|" + "---|" * 14]
for c in sorted(cells, key=key):
    El = list(cells[c].values()); n = len(El); Rm = np.stack([x["rel"] for x in El]); C = np.stack([x["cov"] for x in El]); T = np.stack([x["tl"] for x in El]); PE = np.concatenate([x["perr"] for x in El])
    o = dict(n=n, cov10=float(C[:, 10].mean()), cov16=float(C[:, -1].mean()), c8_10=int((C[:, 10] >= .8).sum()), c8_16=int((C[:, -1] >= .8).sum()), rel10=float(Rm[:, 10].mean()), rel16=float(Rm[:, -1].mean()), mean_rel=float(Rm[:, 1:].mean()),
             done09=int((Rm.max(1) >= .9).sum()), tl10=float(T[:, 10].mean()), tl16=float(T[:, -1].mean()), perr_mean=float(PE.mean()), perr_med=float(np.median(PE)), le5=float((PE <= 5).mean()), le10=float((PE <= 10).mean()),
             leaving=float(np.mean([x["leaving"] for x in El])), plan_s=float(np.mean([x["t"] for x in El])))
    out["cells"][c] = o; md.append(f"| {c} | {n} | {o['cov10']:.3f} | {o['cov16']:.3f} | {o['c8_10']} / {o['c8_16']} | {o['rel10']:.3f} | {o['rel16']:.3f} | {o['mean_rel']:.3f} | {o['done09']}/{n} | {o['tl10']:.1f} / {o['tl16']:.1f} | {o['perr_mean']:.1f} / {o['perr_med']:.1f} | {o['le5']:.2f} / {o['le10']:.2f} | {o['leaving']:.2f} | {o['plan_s']:.2f} |")
md += ["\n## Paired b - a, bootstrap 95 % CI\n", "| width | a -> b | n | d cov@10 | d cov@16 | d transport@16 mm | d placement err mean (per episode) mm | d rel@16 | d mean rel 1-16 | d leaving | d done@0.9 |", "|" + "---|" * 11]
lams = lambda W: sorted({float(PAT.match(c).group(2)) for c in cells if PAT.match(c).group(1) == str(W)})
for W in (15, 10):
    L = lams(W); pairs = [(0.0, l) for l in L if l > 0] + [(a, b) for a, b in ((0.5, 2.0), (2.0, 6.0), (6.0, 20.0), (2.0, 20.0)) if a in L and b in L]
    for a_, b_ in pairs:
        a, b = (f"lw_w{W}_lam{int(x) if x == int(x) else x}" for x in (a_, b_))
        if a not in cells or b not in cells:
            continue
        common = sorted(set(cells[a]) & set(cells[b])); row = dict(width=W, a=a_, b=b_, n=len(common)); txt = []
        for nm, f in (("cov10", lambda x: x["cov"][10]), ("cov16", lambda x: x["cov"][-1]), ("tl16", lambda x: x["tl"][-1]), ("perr", lambda x: x["perr"].mean()), ("rel16", lambda x: x["rel"][-1]), ("mean_rel", lambda x: x["rel"][1:].mean()),
                      ("leaving", lambda x: x["leaving"]), ("done09", lambda x: float(x["rel"].max() >= .9))):
            dl = np.array([f(cells[b][k]) - f(cells[a][k]) for k in common]); d, ci = boot(dl); row[nm] = dict(diff=d, ci95=ci); txt.append(f"{d:+.3f} {ci}")
        out["paired"].append(row); md.append(f"| w{W} | {a_:g} -> {b_:g} | {len(common)} | " + " | ".join(txt) + " |")
if "--diag" in sys.argv:
    import torch, wide_planner as wp
    from closed_loop import MODELS
    from layout_term import LayoutTransport
    model = MODELS["ens2bal"](); md += ["\n## Layout-term optimism of the executed pushes (lam=1 Sinkhorn units; gain = -(change), positive = the layout improved; optimism = predicted gain - true gain)\n", "| cell | pushes | pred gain | true gain (Sinkhorn) | optimism | true Hungarian gain mm/push | share of pushes with true gain > 0 | share with pred gain > 0 |", "|" + "---|" * 8]; out["diag"] = {}
    for c in sorted(cells, key=key):
        pg, tg, hg = [], [], []
        for (g, st), e in raw[c].items():
            mask = torch.from_numpy(goal_mask(g)); lt = LayoutTransport(SL[g], mask.numpy(), lam=1.0)
            for t in range(len(e["actions"])):
                P0 = torch.cat([torch.tensor(e["states"][t]).float(), torch.tensor(e["quats"][t]).float()], 1); P1 = torch.cat([torch.tensor(e["states"][t + 1]).float(), torch.tensor(e["quats"][t + 1]).float()], 1)
                a = torch.tensor(e["actions"][t]).float().to(wp.DEV)[None]; model.begin(P0); o0 = model.occ0.clone(); pr = model.predict(a[:, None, :2], a[:, None, 2:])[-1]
                oT = wp.occ_from_particles(P1[None].float().cpu()).to(wp.DEV); v0 = float(lt(o0)[0]); pg.append(v0 - float(lt(pr)[0])); tg.append(v0 - float(lt(oT)[0]))
                hg.append(1000 * (hungarian_cost(np.array(e["states"][t])[:, :2], SL[g]) - hungarian_cost(np.array(e["states"][t + 1])[:, :2], SL[g])))
        pg, tg, hg = map(np.array, (pg, tg, hg)); d = dict(n=len(pg), pred=float(pg.mean()), true=float(tg.mean()), optimism=float((pg - tg).mean()), hung_mm=float(hg.mean()), true_pos=float((tg > 0).mean()), pred_pos=float((pg > 0).mean()), corr=float(np.corrcoef(pg, tg)[0, 1]))
        out["diag"][c] = d; md.append(f"| {c} | {d['n']} | {d['pred']:+.4f} | {d['true']:+.4f} | {d['optimism']:+.4f} | {d['hung_mm']:+.2f} | {d['true_pos']:.2f} | {d['pred_pos']:.2f} |"); print(c, d, flush=True)
(R / "summary.md").write_text("\n".join(md) + "\n"); (R / "summary.json").write_text(json.dumps(out, indent=1)); print("\n".join(md))
