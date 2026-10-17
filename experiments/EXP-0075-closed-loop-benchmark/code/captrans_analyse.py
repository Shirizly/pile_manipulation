"""EXP-0075 capacity-transport test analysis (2026-10-11; queue_captrans.sh; results/captrans/). Cells ct_w<W>_H<H>_<obj>_<prop>. Reuses analyse_fixes.episode for the standard
closed-loop metrics (rel with the width's normaliser, done@0.9, cubes in / entering / leaving, wasted 7-10, skeleton coverage@10, episodes with coverage >= 0.8, fallbacks, shifted,
optimism of the executed push, plan time) and adds: transport cost per push (mean over cubes of the rectangular-Hungarian distance to the capacity slots on the TRUE state, mm; at push
10 and mean over pushes 1-10), GD time per decision, proposal time, the crowding-transport term per decision (excess part / off-slot part; at the current state and at the executed
push's predicted state), free slots per decision, and paired bootstrap differences (objective effect, proposal effect, horizon effect) with 95 % CIs over the 32 (goal, start) episodes.
usage: PYTHONPATH=. python -u captrans_analyse.py [--dir results/captrans]"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE.parents[1]))
from analyse_fixes import episode, wilson
from captrans import capacity_slots, transport_cost

PAT = re.compile(r"ct_w(\d+)_H(\d)_(\w+?)_(fetch|captrans)$")


def boot(dl, rng, B=2000):
    bs = [rng.choice(dl, len(dl)).mean() for _ in range(B)]; return float(dl.mean()), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(HERE / "results/captrans")); a = ap.parse_args(); R = Path(a.dir)
    cells, raw = {}, {}
    for p in sorted(R.glob("*.json")):
        if p.name in ("summary.json", "diag.json"):
            continue
        for e in json.loads(p.read_text())["episodes"]:
            if e.get("complete") and e.get("states") and PAT.match(e["cell"]):
                x = episode(e); S = np.array(e["states"])[:, :, :2]; sl = capacity_slots(e["goal"])
                x["tc"] = np.array([1000 * transport_cost(S[k], sl) for k in range(len(S))]); cells.setdefault(e["cell"], {})[(e["goal"], e["start"])] = x; raw.setdefault(e["cell"], {})[(e["goal"], e["start"])] = e
    slots = {w: {L: len(capacity_slots(f"letter_{L}_w{w}")) for L in "OTSX"} for w in (20, 15, 10)}
    out = {"slots": slots, "cells": {}, "paired": []}
    md = ["# EXP-0075 capacity-transport objective / proposals on thin letters (O/T/S/X w15 and w10, DS-0006 starts 40-47 = 32 paired episodes per cell, ens2bal, 10 pushes, full push range, legal scoring, fetch/captrans extras n_extra 64, GD 20 steps, 1,280 evaluations / decision, planner seed 0; H=2 cells use cumspill)\n",
          "Cells ct_w<W>_H<H>_<obj>_<prop>: obj cur = lyapunov - in-goal fraction; captrans = + crowding-transport term (captrans.CrowdTransport, m_cube = pi r^2 = 4.91 raster units, d_free from the true current state, 3.5 mm slot occupancy radius). prop fetch = fetch extras (control); captrans = capacity-transport extras (pushes along the Hungarian displacement to the capacity slots and +-10 / +-20 deg; H=2: distinct-cube suffixes from the same rows, pool_mode flot). rel uses the width's own normaliser (EXP-0067 vstar_width.json). transport = mean cube distance (mm) to its rectangular-Hungarian capacity slot on the TRUE state.\n",
          "Capacity slots S per goal (5 mm lattice, 5 x 5 mm footprint inside the mask): " + "; ".join(f"w{w}: " + ", ".join(f"{L} {n}" for L, n in d.items()) for w, d in slots.items()) + " (all >= 20 cubes: no goal is over capacity).\n"]
    hdr = ["| cell | n | rel@10 | mean rel 1-10 | done@0.9 | cubes in @10 | entering / leaving per ep | wasted 7-10 | coverage@10 | cov>=0.8 | transport mm @10 | transport mean 1-10 | fallbacks | shifted | optimism exec | plan s/dec | GD s/dec | proposals ms | crowd term exec: excess / off-slot (x1e-3) | crowd now | free slots |", "|" + "---|" * 21]
    groups = sorted({(int(m.group(1)), int(m.group(2))) for c in cells for m in [PAT.match(c)]}, key=lambda t: (-t[0], t[1]))
    for W, H in groups:
        rows = []
        for c in sorted(cells, key=lambda c: (("cur", "captrans").index(PAT.match(c).group(3)), PAT.match(c).group(4))):
            m = PAT.match(c)
            if (int(m.group(1)), int(m.group(2))) != (W, H):
                continue
            El = list(cells[c].values()); Rw = list(raw[c].values()); n = len(El); Rm = np.stack([x["rel"] for x in El]); C = np.stack([x["cov"] for x in El]); T = np.stack([x["tc"] for x in El])
            d9 = int((Rm.max(1) >= 0.9).sum()); c8 = int((C[:, -1] >= 0.8).sum()); mean = lambda k: float(np.nanmean([x[k] for x in El]))
            fl = lambda key: [v for e in Rw for v in e.get(key, [])]
            gd = float(np.mean(fl("gd_time_s"))) if fl("gd_time_s") else float("nan"); tp = 1000 * float(np.mean(fl("t_captrans_s") or fl("t_fetch_s") or [np.nan]))
            ce = np.array(fl("crowd_exec")) if fl("crowd_exec") else np.full((1, 2), np.nan); cn = np.array(fl("crowd_now")) if fl("crowd_now") else np.full((1, 2), np.nan); nf = float(np.mean(fl("n_free_slots"))) if fl("n_free_slots") else float("nan")
            o = dict(W=W, H=H, n=n, rel10=float(Rm[:, -1].mean()), mean_1_10=float(Rm[:, 1:].mean()), done09=d9, wilson09=wilson(d9, n), cubes10=mean("cubes10"), entering=mean("entering"), leaving=mean("leaving"), wasted_late=mean("wasted_late"),
                     cov10=float(C[:, -1].mean()), cov08=c8, cov_curve=[round(float(v), 3) for v in C.mean(0)], tc10=float(T[:, -1].mean()), tc_mean_1_10=float(T[:, 1:].mean()), tc_curve=[round(float(v), 2) for v in T.mean(0)],
                     fallbacks=int(sum(x["fb"] for x in El)), shifted=int(sum(x["shift"] for x in El)), opt_exec=mean("opt_exec"), plan_s=mean("t"), gd_s=gd, prop_ms=tp, crowd_exec_excess=float(np.nanmean(ce[:, 0])), crowd_exec_off=float(np.nanmean(ce[:, 1])),
                     crowd_now_excess=float(np.nanmean(cn[:, 0])), crowd_now_off=float(np.nanmean(cn[:, 1])), crowd_exec_max=float(np.nanmax(ce.sum(1))), n_free=nf, rel_curve=[round(float(v), 3) for v in Rm.mean(0)])
            out["cells"][c] = o
            rows.append(f"| {c} | {n} | {o['rel10']:.3f} | {o['mean_1_10']:.3f} | {d9}/{n} {wilson(d9, n)} | {o['cubes10']:.1f} | {o['entering']:.1f} / {o['leaving']:.1f} | {100 * o['wasted_late']:.0f} % | {o['cov10']:.3f} | {c8}/{n} | {o['tc10']:.1f} | {o['tc_mean_1_10']:.1f} | {o['fallbacks']} | {o['shifted']} | {o['opt_exec']:+.4f} | {o['plan_s']:.2f} | {gd:.2f} | {tp:.1f} | {1e3 * o['crowd_exec_excess']:.3f} / {1e3 * o['crowd_exec_off']:.1f} | {1e3 * (o['crowd_now_excess'] + o['crowd_now_off']):.1f} | {nf:.1f} |")
        md += [f"\n## w{W}, H={H}\n"] + hdr + rows
    rng = np.random.default_rng(0)
    P = ["| effect | a | b | n | d rel@10 [CI] | d mean rel 1-10 [CI] | d coverage@10 [CI] | d transport@10 mm [CI] | d transport mean 1-10 [CI] | d leaving [CI] | cov better / worse |", "|" + "---|" * 11]
    pairs = []
    for W in (15, 10):
        t = lambda H, o, p: f"ct_w{W}_H{H}_{o}_{p}"
        pairs += [("proposals (H1)", t(1, "cur", "fetch"), t(1, "cur", "captrans")), ("objective + proposals (H1)", t(1, "cur", "fetch"), t(1, "captrans", "captrans")), ("objective given captrans proposals (H1)", t(1, "cur", "captrans"), t(1, "captrans", "captrans")),
                  ("objective + proposals (H2)", t(2, "cur", "fetch"), t(2, "captrans", "captrans")), ("horizon, captrans", t(1, "captrans", "captrans"), t(2, "captrans", "captrans")), ("horizon, control", t(1, "cur", "fetch"), t(2, "cur", "fetch"))]
    for name, ca, cb in pairs:
        common = sorted(set(cells.get(ca, {})) & set(cells.get(cb, {})))
        if not common:
            continue
        row = dict(effect=name, a=ca, b=cb, n=len(common)); txt = []
        for nm, f in (("rel10", lambda x: x["rel"][-1]), ("mean_1_10", lambda x: x["rel"][1:].mean()), ("cov10", lambda x: x["cov"][-1]), ("tc10", lambda x: x["tc"][-1]), ("tc_mean", lambda x: x["tc"][1:].mean()), ("leaving", lambda x: x["leaving"])):
            dl = np.array([f(cells[cb][k]) - f(cells[ca][k]) for k in common]); d, ci = boot(dl, rng); row[nm] = dict(diff=d, ci95=ci, better=int((dl > 0).sum()), worse=int((dl < 0).sum())); txt.append(f"{d:+.3f} {ci}" if nm not in ("tc10", "tc_mean") else f"{d:+.2f} {ci}")
        out["paired"].append(row); P.append(f"| {name} | {ca} | {cb} | {len(common)} | " + " | ".join(txt) + f" | {row['cov10']['better']} / {row['cov10']['worse']} |")
    md += ["\n## Paired differences (b - a), bootstrap 95 % CI over the common (goal, start) episodes (coverage / rel: higher is better; transport / leaving: lower is better)\n"] + P
    # per-goal coverage@10 and transport@10 (objective arms vs control), H=1
    G = ["| width | goal | S | cov@10 cur_fetch | cov@10 captrans_captrans | transport@10 cur_fetch | transport@10 captrans_captrans | rel@10 cur_fetch | rel@10 captrans_captrans |", "|" + "---|" * 9]
    for W in (15, 10):
        ca, cb = f"ct_w{W}_H1_cur_fetch", f"ct_w{W}_H1_captrans_captrans"
        for L in "OTSX":
            g = f"letter_{L}_w{W}"; ka = [k for k in cells.get(ca, {}) if k[0] == g]; kb = [k for k in cells.get(cb, {}) if k[0] == g]
            if ka and kb:
                f = lambda c, ks, key, i: np.mean([cells[c][k][key][i] for k in ks])
                G.append(f"| w{W} | {L} | {slots[W][L]} | {f(ca, ka, 'cov', -1):.3f} | {f(cb, kb, 'cov', -1):.3f} | {f(ca, ka, 'tc', -1):.1f} | {f(cb, kb, 'tc', -1):.1f} | {f(ca, ka, 'rel', -1):.3f} | {f(cb, kb, 'rel', -1):.3f} |")
    md += ["\n## Per goal, H=1 (means over the 8 starts)\n"] + G
    dg = R / "diag.json"
    if dg.exists():                      # captrans_diag.py: can the crowd term steer the choice among candidates at one state?
        D = json.loads(dg.read_text()); out["diag"] = {w: d["mean"] | {"n_states": d["n_states"]} for w, d in D.items()}
        md += ["\n## Candidate-level diagnostic (captrans_diag.py: visited states of the captrans_captrans H=1 cells, pushes 3 / 6 / 9, up to 512 legal captrans + fetch proposals scored under both objectives)\n",
               "| width | states | std base cost | std crowd term | crowd range (max - min) | Spearman(base, captrans) | same argmin | top-8 overlap | max excess part | mean off-slot part |", "|" + "---|" * 10]
        for w, d in D.items():
            m = d["mean"]; md.append(f"| {w} | {d['n_states']} | {m['std_base']:.4f} | {m['std_crowd']:.4f} | {m['range_crowd']:.4f} | {m['spearman']:.4f} | {100 * m['argmin_same']:.0f} % | {100 * m['top8_overlap']:.0f} % | {m['excess_max']:.4f} | {m['off_mean']:.4f} |")
    (R / "summary.md").write_text("\n".join(md) + "\n"); (R / "summary.json").write_text(json.dumps(out, indent=1)); print("\n".join(md))


if __name__ == "__main__":
    main()
