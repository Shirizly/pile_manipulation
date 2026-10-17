"""EXP-0075 gate tests G2 (vanilla64 CEM budget ladder) and G1 (model x planner at 1,280 evals), RAL run plan Tier 1; pre-registration LOG.md 2026-10-15 01:44.
Per cell (episode metrics from analyse_fixes.episode): mean rel 1-10, rel@10, done@0.8/0.9, optimism_exec (harness per-push field, pred - true dV of the executed
push; negative = over-promise), entering / leaving, bank_reach_frac, n_eval per decision, measured plan s/decision (NON-CLEAN, informational only).
Paired contrasts b - a over common (goal, start), bootstrap 95 % CI (2,000 resamples, rng 0). Optional reference cell base1F from results/fix_H1full.json.
Writes results/gates/summary.md and summary.json atomically (re-run after every finished cell).
usage: python -u analyse_gates.py"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code"))
from analyse_fixes import episode
from analyse import wilson

G = HERE / "results/gates"
PAIRS = [("g2_v64_b01", "g2_v64_b03", "G2 budget 0.1 -> 0.3 s"), ("g2_v64_b03", "g2_v64_b10", "G2 RISK TEST budget 0.3 -> 1.0 s"),
         ("g2_v64_b01", "g2_v64_b10", "G2 budget 0.1 -> 1.0 s"), ("g2_v64_b03", "g2_v64_b03_bank8k", "G2 bank control 4096 -> 8192 at 0.3 s"),
         ("g2_v64_b03_bank8k", "g2_v64_b10", "G2 budget 0.3 -> 1.0 s, both bank 8192"), ("g1_v64_cem", "g2_v64_b01", "vanilla64 cem 1,280 -> 0.1 s (2,627)"),
         ("g1_v64_rank", "g1_v64_cem", "G1 planner gain vanilla64 (cem - rank)"), ("g1_ens_rank", "g1_ens_cem", "G1 planner gain ens128 (cem - rank)"),
         ("g1_v64_cem", "g1_v64_cemgd", "G1 GD gain vanilla64 (cem_gd - cem)"), ("g1_ens_cem", "g1_ens_cemgd", "G1 GD gain ens128 (cem_gd - cem)"),
         ("g1_v64_cem", "g1_ens_cem", "ens128 - vanilla64 under cem 1,280"), ("g1_v64_rank", "g1_ens_rank", "ens128 - vanilla64 under rank 1,280"),
         ("g1_v64_cemgd", "g1_ens_cemgd", "ens128 - vanilla64 under cem_gd 1,280"), ("base1F_2026-10-11", "g1_ens_cem", "base1F reproducibility (re-run - 10-11)"),
         # G3 seed gate (LOG 2026-10-15 02:44)
         ("g3_v128_s0", "g1_ens_cem", "G3 ens - vanilla128, seed 0"), ("g3_v128_s1", "g3_ens_s1", "G3 ens - vanilla128, seed 1"),
         ("g3_z128_s0", "g1_ens_cem", "G3 ens - zoom128, seed 0"), ("g3_z128_s1", "g3_ens_s1", "G3 ens - zoom128, seed 1"),
         ("g3_v128_s0", "g3_z128_s0", "G3 zoom128 - vanilla128, seed 0"), ("g3_v128_s1", "g3_z128_s1", "G3 zoom128 - vanilla128, seed 1"),
         ("g3_v128_s0", "g3_v128_s1", "G3 seed spread vanilla128 (s1 - s0)"), ("g3_z128_s0", "g3_z128_s1", "G3 seed spread zoom128 (s1 - s0)"),
         ("g1_ens_cem", "g3_ens_s1", "G3 seed spread ens128 (s1 - s0)"), ("g3_v128_s0", "g3_ens_s1", "G3 EMBARRASSING: ens s1 - vanilla128 s0"),
         ("g1_v64_cem", "g3_ens_s1", "G3 ens s1 - vanilla64 (cem 1,280)"), ("g1_v64_cem", "g3_v128_s0", "vanilla128 s0 - vanilla64 (cem 1,280)"),
         # P0-2 ladders
         ("p02_ens_b01", "p02_ens_b03", "P0-2 ens cem 0.1 -> 0.3 s"), ("p02_ens_b03", "p02_ens_b10", "P0-2 ens cem 0.3 -> 1.0 s"),
         ("p02_ens_b10", "p02_ens_b30", "P0-2 ens cem 1.0 -> 3.0 s"), ("p02_ens_b03", "p02_ens_b30", "P0-2 (i) DECISIVE ens cem 0.3 -> 3.0 s"),
         ("p02_ens_b10", "g1_ens_cem", "ens cem 1.0 s (1,368) vs 1,280 (repeat-like control)"),
         ("p02_ens_gd_b03", "p02_ens_gd_b10", "P0-2 ens cem_gd 0.3 -> 1.0 s"), ("p02_ens_gd_b10", "p02_ens_gd_b30", "P0-2 ens cem_gd 1.0 -> 3.0 s"),
         ("p02_ens_gd_b03", "p02_ens_gd_b30", "P0-2 ens cem_gd 0.3 -> 3.0 s"),
         ("p02_v64_gd_b01", "p02_v64_gd_b03", "P0-2 v64 cem_gd 0.1 -> 0.3 s"), ("p02_v64_gd_b03", "p02_v64_gd_b10", "P0-2 v64 cem_gd 0.3 -> 1.0 s"),
         ("p02_lf_b01", "p02_lf_b03", "P0-2 lf cem 0.1 -> 0.3 s"), ("p02_lf_b03", "p02_lf_b10", "P0-2 lf cem 0.3 -> 1.0 s"),
         ("p02_ens_b03", "p02_ens_gd_b03", "GD gain ens at 0.3 s (equal time)"), ("p02_ens_b10", "p02_ens_gd_b10", "GD gain ens at 1.0 s (equal time)"),
         ("p02_ens_b30", "p02_ens_gd_b30", "GD gain ens at 3.0 s (equal time)"),
         ("g2_v64_b01", "p02_v64_gd_b01", "GD gain v64 at 0.1 s (equal time)"), ("g2_v64_b03", "p02_v64_gd_b03", "GD gain v64 at 0.3 s (equal time)"),
         ("g2_v64_b10", "p02_v64_gd_b10", "GD gain v64 at 1.0 s (equal time)"),
         ("g2_v64_b01", "p02_ens_b01", "(ii) ens - v64 cem at 0.1 s"), ("g2_v64_b03", "p02_ens_b03", "(ii) ens - v64 cem at 0.3 s"),
         ("g2_v64_b10", "p02_ens_b10", "(ii) ens - v64 cem at 1.0 s (v64 bank 8192)"), ("g2_v64_b10", "p02_ens_b30", "(ii) ens cem 3.0 s - v64 cem 1.0 s"),
         ("g2_v64_b01", "p02_ens_b30", "EMBARRASSING: ens cem 3.0 s - v64 cem 0.1 s"),
         ("p02_v64_gd_b03", "p02_ens_gd_b03", "(ii) ens - v64 cem_gd at 0.3 s"), ("p02_v64_gd_b10", "p02_ens_gd_b10", "(ii) ens - v64 cem_gd at 1.0 s"),
         ("p02_v64_gd_b10", "p02_ens_gd_b30", "(ii) ens cem_gd 3.0 s - v64 cem_gd 1.0 s"),
         ("g2_v64_b01", "p02_lf_b01", "lf - v64 cem at 0.1 s"), ("g2_v64_b10", "p02_lf_b10", "lf - v64 cem at 1.0 s"),
         # G5 proposals
         ("g1_v64_cem", "g5_v64_fl", "G5 fl - base, vanilla64"), ("g1_ens_cem", "g5_ens_fl", "G5 fl - base, ens128 (same session)"),
         ("base1F_2026-10-11", "fl1F_2026-10-11", "fl - base, ens128 (2026-10-11 record)"), ("fl1F_2026-10-11", "g5_ens_fl", "fl reproducibility (re-run - 10-11)"),
         ("g5_ens_fl", "g5_ens_fl_spill", "G5 spill filter 1.0 (on fl, ens128)"), ("g5_v64_fl", "g5_ens_fl", "ens - v64 under fl 1,280"),
         # G6 model effect with headroom (LOG 2026-10-15 05:45); rel normalised by each width's own V*
         ("g6_w10_v64_base", "g6_w10_ens_base", "G6 DECISIVE ens - v64, w10 base cem 1,280"), ("g6_w10_v64_fl", "g6_w10_ens_fl", "G6 DECISIVE ens - v64, w10 fl 1,280"),
         ("g6_w15_v64_base", "g6_w15_ens_base", "G6 ens - v64, w15 base cem 1,280"), ("g6_w15_v64_fl", "g6_w15_ens_fl", "G6 ens - v64, w15 fl 1,280"),
         ("g6_w10_v64_b01", "g6_w10_ens_base", "G6 equal time w10: ens base 1,280 (~1 s) - v64 cem 0.1 s"), ("g6_w10_v64_base", "g6_w10_v64_b01", "G6 v64 w10 cem 1,280 -> 0.1 s (2,627)"),
         ("g6_w10_v64_base", "g6_w10_v64_fl", "G6 fl - base, v64 w10"), ("g6_w10_ens_base", "g6_w10_ens_fl", "G6 fl - base, ens w10"),
         ("g6_w15_v64_base", "g6_w15_v64_fl", "G6 fl - base, v64 w15"), ("g6_w15_ens_base", "g6_w15_ens_fl", "G6 fl - base, ens w15"),
         # G6 planner-seed-1 replication of the fl cells (LOG 2026-10-15 07:18)
         ("g6s1_w10_v64_fl", "g6s1_w10_ens_fl", "G6s1 DECISIVE ens - v64, w10 fl, planner seed 1"), ("g6s1_w15_v64_fl", "g6s1_w15_ens_fl", "G6s1 DECISIVE ens - v64, w15 fl, planner seed 1"),
         ("g6_w10_v64_fl", "g6s1_w10_v64_fl", "G6 seed spread v64 w10 fl (s1 - s0)"), ("g6_w10_ens_fl", "g6s1_w10_ens_fl", "G6 seed spread ens w10 fl (s1 - s0)"),
         ("g6_w15_v64_fl", "g6s1_w15_v64_fl", "G6 seed spread v64 w15 fl (s1 - s0)"), ("g6_w15_ens_fl", "g6s1_w15_ens_fl", "G6 seed spread ens w15 fl (s1 - s0)")]


def load():
    cells, meta = {}, {}
    srcs = sorted(p for p in G.glob("*.json") if p.name[:3] in ("g1_", "g2_", "g3_", "g5_", "g6_", "g6s", "p02")) + [HERE / "results/fix_H1full.json"]
    for p in srcs:
        if p.name.startswith("summary"):
            continue
        for e in json.loads(p.read_text())["episodes"]:
            if not (e.get("complete") and e.get("states")):
                continue
            c = e["cell"] if p.parent == G else {"base1F": "base1F_2026-10-11", "fl1F": "fl1F_2026-10-11"}.get(e["cell"])
            if c is None:
                continue
            x = episode(e)
            ox = np.array(e.get("optimism_exec", []), float); x["opt_h"] = float(np.nanmean(ox)) if ox.size else np.nan
            x["opt_h_curve"] = ox
            x["reach"] = float(np.nanmean(e["bank_reach_frac"])) if e.get("bank_reach_frac") else np.nan
            x["n_eval"] = float(np.mean(e["n_evals"]))
            br = np.array(e.get("bank_reach_frac") or [np.nan] * 10, float); x["reach_late"] = float(np.nanmean(br[6:10])) if br.size >= 7 else np.nan
            if p.parent == G:
                cc = json.loads(json.loads(p.read_text())["config"]["cells"]).get(c, {})
                x["gd_steps"] = (e.get("budget") or {}).get("gd_steps", cc.get("gd_steps", 0)) if cc.get("planner") in ("cem_gd", "cem+gd") or cc.get("gd_steps") else 0
            else:
                x["gd_steps"] = 0
            cells.setdefault(c, {})[(e["goal"], e["start"])] = x
            meta.setdefault(c, dict(budget=e.get("budget"), bank_per_env=json.loads(p.read_text())["config"].get("bank_per_env") if p.parent == G else 4096))
    return cells, meta


def main():
    cells, meta = load(); out = {"cells": {}, "paired": []}
    L = ["| cell | n | mean rel 1-10 | rel@10 | done@0.9 | done@0.8 | optimism exec (mean / pushes 1-3 / 8-10) | entering / leaving per ep | bank reach frac (all / pushes 7-10) | n_eval / decision | gd_steps | bank | plan s/decision (NON-CLEAN) |", "|" + "---|" * 13]
    for c in sorted(cells):
        E = list(cells[c].values()); R = np.stack([x["rel"] for x in E]); n = len(E)
        d9, d8 = int((R.max(1) >= 0.9).sum()), int((R.max(1) >= 0.8).sum())
        m = lambda k: float(np.nanmean([x[k] for x in E]))
        oc = [x["opt_h_curve"] for x in E if len(x["opt_h_curve"]) == 10]
        o13 = float(np.mean([o[:3].mean() for o in oc])) if oc else np.nan; o810 = float(np.mean([o[7:].mean() for o in oc])) if oc else np.nan
        opt = m("opt_h") if np.isfinite(m("opt_h")) else m("opt_exec")
        o = dict(n=n, mean_1_10=float(R[:, 1:].mean()), rel10=float(R[:, -1].mean()), done09=d9, done08=d8, wilson09=wilson(d9, n), wilson08=wilson(d8, n),
                 optimism_exec=opt, optimism_p1_3=o13, optimism_p8_10=o810, entering=m("entering"), leaving=m("leaving"), bank_reach_frac=m("reach"), bank_reach_late=m("reach_late"), gd_steps=m("gd_steps"), n_eval=m("n_eval"),
                 plan_s_nonclean=m("t"), fallbacks=int(sum(x["fb"] for x in E)), bank_per_env=meta[c]["bank_per_env"], budget=meta[c]["budget"])
        out["cells"][c] = o
        L.append(f"| {c} | {n} | {o['mean_1_10']:.3f} | {o['rel10']:.3f} | {d9}/{n} | {d8}/{n} | {opt:+.4f} / {o13:+.4f} / {o810:+.4f} | {o['entering']:.1f} / {o['leaving']:.2f} | "
                 f"{o['bank_reach_frac']:.3f} / {o['bank_reach_late']:.3f} | {o['n_eval']:.0f} | {o['gd_steps']:.0f} | {o['bank_per_env']} | {o['plan_s_nonclean']:.2f} |")
    rng = np.random.default_rng(0)
    P = ["| contrast | a | b | n | d mean rel 1-10 [CI] (b>/b<) | d rel@10 [CI] (b>/b<) | d optimism exec [CI] | d leaving [CI] |", "|---|---|---|---|---|---|---|---|"]
    for ca, cb, lab in PAIRS:
        common = sorted(set(cells.get(ca, {})) & set(cells.get(cb, {})))
        if not common:
            continue
        row = dict(label=lab, a=ca, b=cb, n=len(common)); txt = []
        for nm, f in (("mean_1_10", lambda x: x["rel"][1:].mean()), ("rel10", lambda x: x["rel"][-1]), ("optimism_exec", lambda x: x["opt_h"] if np.isfinite(x["opt_h"]) else x["opt_exec"]),
                      ("leaving", lambda x: x["leaving"])):
            dl = np.array([f(cells[cb][k]) - f(cells[ca][k]) for k in common]); dl = dl[np.isfinite(dl)]
            if not len(dl):
                txt.append("n/a"); continue
            bs = [rng.choice(dl, len(dl)).mean() for _ in range(2000)]
            ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
            row[nm] = dict(diff=float(dl.mean()), ci95=ci, better=int((dl > 0).sum()), worse=int((dl < 0).sum()))
            txt.append(f"{dl.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] ({row[nm]['better']}/{row[nm]['worse']})" if nm in ("mean_1_10", "rel10") else f"{dl.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        out["paired"].append(row); P.append(f"| {lab} | {ca} | {cb} | {len(common)} | " + " | ".join(txt) + " |")
    # interaction (difference of paired gains, per episode): does the gain depend on the model?
    out["interaction"] = {}; P += ["\n## Interaction: (gain for ens128) - (gain for vanilla64), per-episode difference of paired gains, bootstrap 95 % CI\n", "| contrast | metric | diff [CI] |", "|---|---|---|"]
    for lab, (a1, b1, a2, b2) in {"GD gain ens - GD gain v64": ("g1_ens_cem", "g1_ens_cemgd", "g1_v64_cem", "g1_v64_cemgd"),
                                  "planner gain ens - v64": ("g1_ens_rank", "g1_ens_cem", "g1_v64_rank", "g1_v64_cem"),
                                  "G5 fl gain ens - fl gain v64": ("g1_ens_cem", "g5_ens_fl", "g1_v64_cem", "g5_v64_fl"),
                                  "equal-time GD gain ens - v64 at 0.3 s": ("p02_ens_b03", "p02_ens_gd_b03", "g2_v64_b03", "p02_v64_gd_b03"),
                                  "equal-time GD gain ens - v64 at 1.0 s": ("p02_ens_b10", "p02_ens_gd_b10", "g2_v64_b10", "p02_v64_gd_b10"),
                                  "G6 w10 fl gain ens - fl gain v64": ("g6_w10_ens_base", "g6_w10_ens_fl", "g6_w10_v64_base", "g6_w10_v64_fl"),
                                  "G6 w15 fl gain ens - fl gain v64": ("g6_w15_ens_base", "g6_w15_ens_fl", "g6_w15_v64_base", "g6_w15_v64_fl")}.items():
        if not all(c in cells for c in (a1, b1, a2, b2)):
            continue
        ks = sorted(set(cells[a1]) & set(cells[b1]) & set(cells[a2]) & set(cells[b2]))
        for nm, f in (("mean_1_10", lambda x: x["rel"][1:].mean()), ("rel10", lambda x: x["rel"][-1])):
            d = np.array([(f(cells[b1][k]) - f(cells[a1][k])) - (f(cells[b2][k]) - f(cells[a2][k])) for k in ks]); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; out["interaction"][f"{lab} | {nm}"] = dict(diff=float(d.mean()), ci95=ci, n=len(ks))
            P.append(f"| {lab} | {nm} | {d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    # G3 seed spread (|s0 - s1| on mean rel 1-10) and P0-2 equal-wall-time curves t = 10 x (b + 1 s)
    out["seed_spread"] = {}; P += ["\n## G3 seed spread: |s0 - s1| on mean rel 1-10 (rel@10)\n", "| model | s0 cell | s1 cell | s0 | s1 | abs diff mean rel 1-10 | abs diff rel@10 |", "|---|---|---|---|---|---|---|"]
    for mdl, a, b in (("vanilla128", "g3_v128_s0", "g3_v128_s1"), ("zoom128", "g3_z128_s0", "g3_z128_s1"), ("ens128", "g1_ens_cem", "g3_ens_s1")):
        if a in out["cells"] and b in out["cells"]:
            A, B = out["cells"][a], out["cells"][b]; out["seed_spread"][mdl] = dict(s0=A["mean_1_10"], s1=B["mean_1_10"], abs_mean=abs(A["mean_1_10"] - B["mean_1_10"]), abs_rel10=abs(A["rel10"] - B["rel10"]))
            P.append(f"| {mdl} | {a} | {b} | {A['mean_1_10']:.3f} | {B['mean_1_10']:.3f} | {abs(A['mean_1_10'] - B['mean_1_10']):.3f} | {abs(A['rel10'] - B['rel10']):.3f} |")
    # G6 seed-pooled gap (LOG 2026-10-15 07:18): per episode, ens - v64 averaged over planner seeds 0 and 1; bootstrap over episodes (seed pairs kept together)
    out["g6_pooled"] = {}; P += ["\n## G6 fl: ens - v64 pooled over planner seeds 0 + 1 (64 episode-seed pairs; bootstrap over the 32 episodes, seed pairs kept) and planner-seed spread |s0 - s1|\n",
                                 "| width | metric | n episodes | gap s0 | gap s1 | pooled gap [CI] | |s0-s1| v64 | |s0-s1| ens |", "|---|---|---|---|---|---|---|---|"]
    for w in ("w10", "w15"):
        cs = [f"g6_{w}_v64_fl", f"g6_{w}_ens_fl", f"g6s1_{w}_v64_fl", f"g6s1_{w}_ens_fl"]
        if not all(c in cells for c in cs):
            continue
        ks = sorted(set.intersection(*[set(cells[c]) for c in cs]))
        for nm, f in (("mean_1_10", lambda x: x["rel"][1:].mean()), ("rel10", lambda x: x["rel"][-1])):
            v = {c: np.array([f(cells[c][k]) for k in ks]) for c in cs}
            g0 = v[cs[1]] - v[cs[0]]; g1 = v[cs[3]] - v[cs[2]]; d = (g0 + g1) / 2
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]; ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            sv, se = abs(v[cs[2]].mean() - v[cs[0]].mean()), abs(v[cs[3]].mean() - v[cs[1]].mean())
            out["g6_pooled"][f"{w} | {nm}"] = dict(n=len(ks), gap_s0=float(g0.mean()), gap_s1=float(g1.mean()), pooled=float(d.mean()), ci95=ci, spread_v64=float(sv), spread_ens=float(se))
            P.append(f"| {w} | {nm} | {len(ks)} | {g0.mean():+.3f} | {g1.mean():+.3f} | {d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {sv:.3f} | {se:.3f} |")
    curves = {"vanilla64 cem": [(0.1, "g2_v64_b01"), (0.3, "g2_v64_b03"), (1.0, "g2_v64_b10")], "ens128 cem": [(0.1, "p02_ens_b01"), (0.3, "p02_ens_b03"), (1.0, "p02_ens_b10"), (3.0, "p02_ens_b30")],
              "vanilla64 cem_gd": [(0.1, "p02_v64_gd_b01"), (0.3, "p02_v64_gd_b03"), (1.0, "p02_v64_gd_b10")], "ens128 cem_gd": [(0.3, "p02_ens_gd_b03"), (1.0, "p02_ens_gd_b10"), (3.0, "p02_ens_gd_b30")],
              "lf cem": [(0.1, "p02_lf_b01"), (0.3, "p02_lf_b03"), (1.0, "p02_lf_b10")]}
    out["curves"] = {}; P += ["\n## P0-2 budget curves: mean rel 1-10 (rel@10) at budget b s/decision, episode wall time t = 10 x (b + 1 s); GD share of time +-15-30 % (timing caveat)\n",
                              "| curve | b=0.1 (t=11) | b=0.3 (t=13) | b=1.0 (t=20) | b=3.0 (t=40) |", "|---|---|---|---|---|"]
    for nm, pts in curves.items():
        got = {b: out["cells"][c] for b, c in pts if c in out["cells"]}; out["curves"][nm] = {str(b): dict(cell=c, t=10 * (b + 1), mean_1_10=out["cells"][c]["mean_1_10"], rel10=out["cells"][c]["rel10"]) for b, c in pts if c in out["cells"]}
        P.append(f"| {nm} | " + " | ".join(f"{got[b]['mean_1_10']:.3f} ({got[b]['rel10']:.3f})" if b in got else "-" for b in (0.1, 0.3, 1.0, 3.0)) + " |")
    out["crossing"] = {}
    for lab, (ca, cb) in {"cem": ("vanilla64 cem", "ens128 cem"), "cem_gd": ("vanilla64 cem_gd", "ens128 cem_gd")}.items():
        for met in ("mean_1_10", "rel10"):
          for flat in (False, True):
            A = {float(b): v[met] for b, v in out["curves"].get(ca, {}).items()}; B = {float(b): v[met] for b, v in out["curves"].get(cb, {}).items()}
            if len(A) < 2 or len(B) < 2:
                continue
            if flat:   # ASSUMPTION (labelled): v64 held flat beyond its last measured budget (G2: flat 0.1 -> 1.0 s), extended to ens's last budget
                if max(B) <= max(A):
                    continue
                A[max(B)] = A[max(A)]; lab2 = lab + " v64-held-flat"
            else:
                lab2 = lab
            lo, hi = max(min(A), min(B)), min(max(A), max(B)); grid = np.exp(np.linspace(np.log(10 * (lo + 1)), np.log(10 * (hi + 1)), 200))
            it = lambda D: np.interp(np.log(grid), np.log([10 * (b + 1) for b in sorted(D)]), [D[b] for b in sorted(D)])
            d = it(B) - it(A); sc = [float(grid[i + 1]) for i in range(len(d) - 1) if np.sign(d[i]) != np.sign(d[i + 1])]
            out["crossing"][f"{lab2} | {met}"] = dict(t_range=[float(grid[0]), float(grid[-1])], diff_at_start=float(d[0]), diff_at_end=float(d[-1]), crossings_t_s=sc,
                                                     ens_max_minus_v64_max=float(max(B.values()) - max(A.values())))
            P.append(f"\ncrossing ({lab2}, {met}; ens - v64 interpolated in log t over t {grid[0]:.0f}-{grid[-1]:.0f} s): diff at start {d[0]:+.3f}, at end {d[-1]:+.3f}, sign changes at t = {[round(x, 1) for x in sc]}; best ens - best v64 {max(B.values()) - max(A.values()):+.3f}")
    md = ["# EXP-0075 gate tests G2 + G1 + G3 + P0-2 remainder + G5 + G6 (base1F harness: H=1, full range, pile-aware bank, no fetch/legal_score, legal fallback; O/T/S/X w20 x starts 40-47; 10 pushes; planner seed 0; g6_* cells at w10 / w15 with rel vs each width's own V*)\n",
          "vanilla64 = vanilla64bal (w64_s0_ms4), ens128 = ens2bal. optimism exec = pred - true dV of the executed push (negative = over-promise). plan s is NON-CLEAN (informational only).\n"] + L + \
         ["\n## Paired differences (b - a), bootstrap 95 % CI over (goal, start); (b>/b<) = episodes where b is better / worse\n"] + P
    if (G / "VERDICTS.md").exists():   # hand-written verdicts against the pre-registrations (LOG.md), appended verbatim
        md.append((G / "VERDICTS.md").read_text())
    for name, txt in (("summary.md", "\n".join(md) + "\n"), ("summary.json", json.dumps(out, indent=1, default=float))):
        tmp = G / (name + ".tmp"); tmp.write_text(txt); os.replace(tmp, G / name)
    print("\n".join(md))


if __name__ == "__main__":
    main()
