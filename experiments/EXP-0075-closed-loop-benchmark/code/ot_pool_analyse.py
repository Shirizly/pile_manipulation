"""EXP-0075 OT-seeded candidate pool test (2026-10-11; queue_ot_pool.sh; results/ot_pool/). Reuses analyse_fixes.episode for the standard closed-loop table
(rel@10, mean rel 1-10, done@0.9 / 0.8, cubes in, entering / leaving, wasted pushes 7-10, skeleton coverage@10, fallbacks, shifted, plan s / decision) and adds
GD time per decision, the proposal-generation times, the best-predicted-cost trace by stage (initial pool / CEM iteration 1-4 / GD), the initial pool's best by
source (bank / fetch / OT), the share of chosen first pushes by source and by stage, paired bootstrap differences (ot - fl, mix - fl per (H, budget); the GD cells
against the recorded no-GD cells fl1F / cs2F_fl), and the 'faster' reading: the evaluation count at which fl's trace first reaches the ot pool's initial best
(paired at push 1, where the two cells see the identical start state).
usage: PYTHONPATH=. python -u ot_pool_analyse.py [--extra results/fix_H1full.json results/fix_H2.json]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE.parents[1]))
from analyse_fixes import episode, wilson

GROUPS = [("H1_E320", 1, 320), ("H2_E1280", 2, 1280), ("H4_E1280", 4, 1280), ("H1_E1280", 1, 1280)]
MODES = ("fl", "ot", "mix")


def load(paths):
    cells, raw = {}, {}
    for p in paths:
        if not Path(p).exists():
            continue
        for e in json.loads(Path(p).read_text())["episodes"]:
            if e.get("complete") and e.get("states"):
                cells.setdefault(e["cell"], {})[(e["goal"], e["start"])] = episode(e); raw.setdefault(e["cell"], {})[(e["goal"], e["start"])] = e
    return cells, raw


def boot(dl, rng, B=2000):
    bs = [rng.choice(dl, len(dl)).mean() for _ in range(B)]; return float(dl.mean()), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extra", nargs="*", default=[str(HERE / "results/fix_H1full.json"), str(HERE / "results/fix_H2.json")]); ap.add_argument("--dir", default=str(HERE / "results/ot_pool")); a = ap.parse_args()
    R = Path(a.dir); cells, raw = load([str(R / f"{g}.json") for g, _, _ in GROUPS] + a.extra)
    out, md = {"cells": {}, "paired": [], "trace": {}, "faster": {}}, ["# EXP-0075 OT-seeded candidate pool (letters O/T/S/X w20, DS-0006 starts 40-47, ens2bal, 10 pushes, planner seed 0, fetch + legal scoring + GD 20 steps in every arm)\n",
                                                                     "Cells: ot_<mode>_H<h>_E<evals>; mode fl = today's pool (bank + fetch, control), ot = OT rows + fetch + bank with OT-structured later pushes, mix = bank + fetch + OT rows with random later pushes. OT target = uniform_layout (Lloyd) slots over the letter.\n"]
    hdr = ["| cell | n | rel@10 | mean rel 1-10 | done@0.9 | done@0.8 | cubes in @10 | entering / leaving per ep | wasted 7-10 | coverage@10 | fallbacks | shifted | plan s/dec | GD s/dec | proposals ms (fetch / OT) |", "|" + "---|" * 15]
    for gname, H, E in GROUPS:
        rows = []
        for m in MODES:
            c = f"ot_{m}_H{H}_E{E}"
            if c not in cells:
                continue
            Ep = cells[c]; Rw = raw[c]; Elist = list(Ep.values()); n = len(Elist); Rm = np.stack([x["rel"] for x in Elist]); C = np.stack([x["cov"] for x in Elist])
            d9, d8 = int((Rm.max(1) >= 0.9).sum()), int((Rm.max(1) >= 0.8).sum()); mean = lambda k: float(np.nanmean([x[k] for x in Elist]))
            gd = float(np.mean([v for e in Rw.values() for v in e.get("gd_time_s", [np.nan])])); tf = 1000 * float(np.mean([v for e in Rw.values() for v in e.get("t_fetch_s", [np.nan])])); to = 1000 * float(np.mean([v for e in Rw.values() for v in e.get("t_ot_s", [np.nan])])) if m != "fl" else float("nan")
            o = dict(n=n, rel10=float(Rm[:, -1].mean()), mean_1_10=float(Rm[:, 1:].mean()), done09=d9, done08=d8, cubes10=mean("cubes10"), entering=mean("entering"), leaving=mean("leaving"), wasted_late=mean("wasted_late"),
                     cov10=float(C[:, -1].mean()), fallbacks=int(sum(x["fb"] for x in Elist)), shifted=int(sum(x["shift"] for x in Elist)), plan_s=mean("t"), gd_s=gd, t_fetch_ms=tf, t_ot_ms=to, rel_curve=[round(float(v), 3) for v in Rm.mean(0)])
            out["cells"][c] = o
            rows.append(f"| {c} | {n} | {o['rel10']:.3f} | {o['mean_1_10']:.3f} | {d9}/{n} {wilson(d9, n)} | {d8}/{n} | {o['cubes10']:.1f} | {o['entering']:.1f} / {o['leaving']:.1f} | {100 * o['wasted_late']:.0f} % | {o['cov10']:.3f} | {o['fallbacks']} | {o['shifted']} | {o['plan_s']:.2f} | {gd:.2f} | {tf:.1f} / {to:.1f} |")
        if rows:
            md += [f"\n## H={H}, {E} evaluations per decision\n"] + hdr + rows
    # paired differences
    rng = np.random.default_rng(0); P = ["| a | b | n | d rel@10 [CI] | d mean rel 1-10 [CI] | d coverage@10 [CI] | d leaving [CI] | better / worse (mean rel) |", "|---|---|---|---|---|---|---|---|"]
    pairs = [(f"ot_fl_H{H}_E{E}", f"ot_{m}_H{H}_E{E}") for _, H, E in GROUPS for m in ("ot", "mix")] + [("fl1F", "ot_fl_H1_E1280"), ("cs2F_fl", "ot_fl_H2_E1280"), ("fl1F", "ot_ot_H1_E1280"), ("ot_fl_H1_E1280", "ot_fl_H1_E320"), ("ot_fl_H1_E1280", "ot_fl_H2_E1280"), ("ot_fl_H1_E1280", "ot_fl_H4_E1280"), ("ot_fl_H2_E1280", "ot_fl_H4_E1280")]
    for ca, cb in pairs:
        common = sorted(set(cells.get(ca, {})) & set(cells.get(cb, {})))
        if not common:
            continue
        row = dict(a=ca, b=cb, n=len(common)); txt = []
        for nm, f in (("rel10", lambda x: x["rel"][-1]), ("mean_1_10", lambda x: x["rel"][1:].mean()), ("cov10", lambda x: x["cov"][-1]), ("leaving", lambda x: x["leaving"])):
            dl = np.array([f(cells[cb][k]) - f(cells[ca][k]) for k in common]); d, ci = boot(dl, rng); row[nm] = dict(diff=d, ci95=ci, better=int((dl > 0).sum()), worse=int((dl < 0).sum())); txt.append(f"{d:+.3f} {ci}")
        out["paired"].append(row); P.append(f"| {ca} | {cb} | {len(common)} | " + " | ".join(txt) + f" | {row['mean_1_10']['better']} / {row['mean_1_10']['worse']} |")
    md += ["\n## Paired differences (b - a), bootstrap 95 % CI over the common (goal, start) episodes\n"] + P
    # trace table
    T = ["| cell | decisions | pool best: bank / fetch / OT | best after pool | iter 1 | iter 2 | iter 3 | iter 4 | after GD | GD gain | GD wins | chosen src bank / fetch / OT | chosen stage pool / cem / gd |", "|" + "---|" * 14]
    for gname, H, E in GROUPS:
        for m in MODES:
            c = f"ot_{m}_H{H}_E{E}"
            if c not in raw:
                continue
            es = list(raw[c].values()); tr = np.array([t for e in es for t in e["trace"]], dtype=float); nd = len(tr)
            pb = {s: float(np.mean([v for e in es for v in e.get(f"pool_best_{s}", []) if np.isfinite(v)])) if any(f"pool_best_{s}" in e for e in es) else float("nan") for s in ("bank", "fetch", "ot")}   # finite only: a source with no legal row that decision scores +inf
            src = [v for e in es for v in e["chosen_src"]]; st = [v for e in es for v in e["best_stage"]]; gg = np.array([v for e in es for v in e["gd_gain"]])
            o = dict(decisions=nd, pool_best=pb, trace_mean=[float(v) for v in np.nanmean(tr, 0)], gd_gain=float(gg.mean()), gd_wins=float((gg > 0).mean()),
                     src_share={s: src.count(s) / len(src) for s in ("bank", "fetch", "ot")}, stage_share={"pool": st.count("pool") / len(st), "cem": sum(s.startswith("cem") for s in st) / len(st), "gd": st.count("gd") / len(st)},
                     n_ot_mean=float(np.mean([v for e in es for v in e.get("n_ot", [])])) if m != "fl" else None)
            out["trace"][c] = o; tm = o["trace_mean"]
            T.append(f"| {c} | {nd} | {pb['bank']:.3f} / {pb['fetch']:.3f} / {pb['ot']:.3f} | " + " | ".join(f"{v:.3f}" for v in tm[:6]) + f" | {o['gd_gain']:.3f} | {100 * o['gd_wins']:.0f} % | " + " / ".join(f"{100 * o['src_share'][s]:.0f}" for s in ("bank", "fetch", "ot")) + " % | " + " / ".join(f"{100 * o['stage_share'][s]:.0f}" for s in ("pool", "cem", "gd")) + " % |")
    md += ["\n## Optimiser trace (mean predicted cost of the best-so-far sequence over all decisions; lower is better; 'pool best' = best initial-pool row by source; chosen src = source of the initial-pool row nearest the executed first push)\n"] + T
    # faster reading
    F = ["| H / evals | n | fl pool best (push 1) | ot pool best (push 1) | fl evals to reach ot pool best: median [n reached by pool / by CEM / by GD / never] | all-decision: fl mean trace vs ot mean pool best |", "|---|---|---|---|---|---|"]
    for gname, H, E in GROUPS:
        cf, co = f"ot_fl_H{H}_E{E}", f"ot_ot_H{H}_E{E}"
        if cf not in raw or co not in raw:
            continue
        common = sorted(set(raw[cf]) & set(raw[co])); n0 = E // 5; ev, cat = [], {"pool": 0, "cem": 0, "gd": 0, "never": 0}
        for k in common:
            tf, to = raw[cf][k]["trace"][0], raw[co][k]["trace"][0]; target = to[0]
            idx = next((i for i, v in enumerate(tf) if v <= target), None)
            if idx is None: cat["never"] += 1; ev.append(np.inf)
            else:
                ev.append(n0 * (idx + 1) if idx <= 4 else E); cat["pool" if idx == 0 else "cem" if idx <= 4 else "gd"] += 1
        trf = np.nanmean(np.array([t for e in raw[cf].values() for t in e["trace"]], dtype=float), 0); pbo = float(np.mean([t[0] for e in raw[co].values() for t in e["trace"]]))
        reach = next((f"{n0 * (i + 1)} evals" if i <= 4 else "GD" for i, v in enumerate(trf) if v <= pbo), "never")
        out["faster"][gname] = dict(n=len(common), fl_pool_push1=float(np.mean([raw[cf][k]["trace"][0][0] for k in common])), ot_pool_push1=float(np.mean([raw[co][k]["trace"][0][0] for k in common])), evals_median=float(np.median(ev)), cat=cat, all_decision_reach=reach, fl_trace_mean=[float(v) for v in trf], ot_pool_mean=pbo)
        F.append(f"| H={H} / {E} | {len(common)} | {out['faster'][gname]['fl_pool_push1']:.3f} | {out['faster'][gname]['ot_pool_push1']:.3f} | {np.median(ev):.0f} [{cat['pool']} / {cat['cem']} / {cat['gd']} / {cat['never']}] | fl reaches {pbo:.3f} at {reach} |")
    md += ["\n## 'Faster' reading: evaluations fl needs to reach the ot arm's initial-pool best (paired at push 1: identical start state; a GD entry counts as the full budget)\n"] + F
    (R / "summary.md").write_text("\n".join(md) + "\n"); (R / "summary.json").write_text(json.dumps(out, indent=1)); print("\n".join(md))


if __name__ == "__main__":
    main()
