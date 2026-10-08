"""EXP-0075 budget study analysis -> results/budget_study_summary.md and figures/budget_study_*.png.  Paired by task (goal,start); 95% bootstrap CI over tasks (10000 resamples).  Value = true terminal value change in the simulator after 4 pushes (lower = better)."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1] / "results"; F = Path(__file__).resolve().parents[1] / "figures"; BS = R / "budget_study"
rng = np.random.default_rng(0)
rows = []
for f in sorted(glob.glob(str(BS / "replay_*.json"))): rows += json.load(open(f))
for r in rows:
    r["pred"] = r["pred_terminal"] if r.get("pred_terminal") is not None else (float(np.sum(r["pred_step_values"])) if r.get("pred_step_values") else None)
    r["key"] = (r["goal"], r["start"])
# existing 20k+GD ceiling plans (gd_budget study, ens128, same 10 tasks)
for r in json.load(open(R / "gd_budget_sim.json")):
    if r["plan"] in ("gd_ref", "gd5", "gd10"): rows.append(dict(goal=r["goal"], start=r["start"], model="ens128", planner="h4_" + r["plan"], B=None, true_terminal=r["true_terminal"], pred=r["pred_terminal"], key=(r["goal"], r["start"]), shifted_pushes=r["shifted_pushes"]))
for r in rows:
    if r["planner"].startswith("greedy_pool"): r["B"] = None


def sel(model=None, planner=None, B=None, pool=None):
    return {r["key"]: r for r in rows if (model is None or r["model"] == model) and (planner is None or r["planner"] == planner) and (B is None or r.get("B") == B) and (pool is None or r.get("pool") == pool)}


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan, np.nan, len(x))
    b = rng.choice(x, (10000, len(x))).mean(1); return (x.mean(), *np.percentile(b, [2.5, 97.5]), len(x))


def fmt(c): return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}] (n={c[3]})"


def vals(d, k="true_terminal"): return {t: r[k] for t, r in d.items() if r.get(k) is not None}


def paired(a, b, k="true_terminal"):
    ks = sorted(set(a) & set(b)); return ci([a[t][k] - b[t][k] for t in ks])


MODS = ["ens128", "zoom128", "vanilla128", "zoom64", "vanilla64"]; BUD = [1, 3, 10]; out = ["# EXP-0075 budget study (H=4 plans, 4 pushes, simulator replay; 10 tasks, 95 % bootstrap CI over tasks)\n"]
out.append("Value = TRUE terminal value change after 4 pushes (lower = better). Budget B = planning time on an exclusive GPU, converted to evaluation counts with results/timing_units.md.\n")
# 1 grid
out.append("## 1. True value by model / planner / budget\n"); out.append("| model | planner | B=1 s | B=3 s | B=10 s |"); out.append("|---|---|---|---|---|")
for m in MODS:
    for pl in ("cem", "cem_gd", "rand_gd"):
        cells = [sel(m, pl, B) for B in BUD]
        if not any(cells): continue
        out.append(f"| {m} | {pl} | " + " | ".join(fmt(ci(list(vals(c).values()))) if c else "-" for c in cells) + " |")
# 2 optimism
out.append("\n## 2. Optimism = predicted - true (negative = model over-promised), mean [CI]\n"); out.append("| model | planner | B=1 | B=3 | B=10 |"); out.append("|---|---|---|---|---|")
for m in MODS:
    for pl in ("cem", "cem_gd", "rand_gd"):
        cells = [sel(m, pl, B) for B in BUD]
        if not any(cells): continue
        out.append(f"| {m} | {pl} | " + " | ".join(fmt(ci([r["pred"] - r["true_terminal"] for r in c.values() if r["pred"] is not None])) if c else "-" for c in cells) + " |")
# 3 ceilings
out.append("\n## 3. Ceilings (ens128; same 10 tasks unless n says otherwise)\n"); out.append("| plan | n | true | predicted | optimism |"); out.append("|---|---|---|---|---|")
ceil = {}
for name, d in (("H=4 CEM(1,280)+GD", sel("ens128", "h4_gd_ref")), ("H=4 CEM(10,000)+GD", sel("ens128", "h4_gd5")), ("H=4 CEM(20,000)+GD  <- H=4 ceiling", sel("ens128", "h4_gd10")),
                ("H=1 greedy open loop, pool 1,280 + GD", sel("ens128", "greedy_pool1280")), ("H=1 greedy open loop, pool 10,000 + GD", sel("ens128", "greedy_pool10000")),
                ("H=1 closed loop, pool 1,280 + GD", sel("ens128", "h1_closed_ceil", pool="pool1280")), ("H=1 closed loop, pool 10,000 + GD", sel("ens128", "h1_closed_ceil", pool="pool10000"))):
    if not d: continue
    ceil[name] = d; out.append(f"| {name} | {len(d)} | {fmt(ci(list(vals(d).values())))} | {fmt(ci([r['pred'] for r in d.values() if r['pred'] is not None]))} | {fmt(ci([r['pred'] - r['true_terminal'] for r in d.values() if r['pred'] is not None]))} |")
H4c = ceil.get("H=4 CEM(20,000)+GD  <- H=4 ceiling")
if H4c:
    out.append("\nPaired on common tasks: plan minus the H=4 ceiling plan (negative = better than the H=4 ceiling):\n")
    for name, d in ceil.items():
        if d is not H4c: out.append(f"* {name}: {fmt(paired(d, H4c))}")
# 4 fraction of ceiling
if H4c:
    out.append("\n## 4. Fraction of the H=4 ceiling captured (true value / ceiling value on the same tasks; mean of ratios, CI over tasks)\n"); out.append("| model | planner | B=1 | B=3 | B=10 |"); out.append("|---|---|---|---|---|")
    for m in MODS:
        for pl in ("cem", "cem_gd"):
            cells = []
            for B in BUD:
                c = sel(m, pl, B); ks = sorted(set(c) & set(H4c)); cells.append(ci([c[t]["true_terminal"] / H4c[t]["true_terminal"] for t in ks]) if ks else None)
            out.append(f"| {m} | {pl} | " + " | ".join(f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]" if x else "-" for x in cells) + " |")
# 5 decomposition
out.append("\n## 5. CEM vs GD (paired differences in true value; negative = first better)\n")
for m in ("ens128", "vanilla64"):
    for B in BUD:
        a, b, c = sel(m, "cem", B), sel(m, "cem_gd", B), sel(m, "rand_gd", B)
        if a and b: out.append(f"* {m} B={B}: GD adds to CEM (cem_gd - cem): {fmt(paired(b, a))}" + (f"; CEM adds to GD (cem_gd - rand_gd): {fmt(paired(b, c))}" if c else ""))
# 6 cheap vs expensive
out.append("\n## 6. Crossover: ens128 minus cheaper model at equal planning time (cem_gd; negative = ensemble better)\n"); out.append("| cheaper model | B=1 | B=3 | B=10 |"); out.append("|---|---|---|---|")
for m in MODS[1:]:
    out.append(f"| {m} | " + " | ".join(fmt(paired(sel("ens128", "cem_gd", B), sel(m, "cem_gd", B))) if sel(m, "cem_gd", B) and sel("ens128", "cem_gd", B) else "-" for B in BUD) + " |")
out.append("\nSame comparison, CEM only:\n"); out.append("| cheaper model | B=1 | B=3 | B=10 |"); out.append("|---|---|---|---|")
for m in MODS[1:]:
    out.append(f"| {m} | " + " | ".join(fmt(paired(sel("ens128", "cem", B), sel(m, "cem", B))) if sel(m, "cem", B) and sel("ens128", "cem", B) else "-" for B in BUD) + " |")
# 7 H=1 closed equal time vs H=4
out.append("\n## 7. Equal planning time per 4 pushes: H=1 closed loop (B/4 per push) vs H=4 open loop (cem_gd); difference H1 - H4 (negative = H=1 better)\n"); out.append("| model | B=1 | B=3 | B=10 | H=1 value B=1 / 3 / 10 |"); out.append("|---|---|---|---|---|")
for m in MODS:
    cells, v1 = [], []
    for B in BUD:
        a = sel(m, "h1_closed_eq", B); b = sel(m, "cem_gd", B); v1.append(fmt(ci(list(vals(a).values()))) if a else "-"); cells.append(fmt(paired(a, b)) if a and b else "-")
    out.append(f"| {m} | " + " | ".join(cells) + " | " + " / ".join(v1) + " |")
open(R / "budget_study_summary.md", "w").write("\n".join(out)); print("\n".join(out))
# figures
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6)); col = dict(zip(MODS, ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]))
for m in MODS:
    for pl, ls in (("cem", ":"), ("cem_gd", "-")):
        xs, ys, lo, hi = [], [], [], []
        for B in BUD:
            c = sel(m, pl, B)
            if c: v = ci(list(vals(c).values())); xs.append(B); ys.append(v[0]); lo.append(v[1]); hi.append(v[2])
        if xs: ax[0].plot(xs, ys, ls, marker="o", color=col[m], label=f"{m} {pl}"); ax[0].fill_between(xs, lo, hi, color=col[m], alpha=0.07 if pl == "cem" else 0.15)
    xs, ys = [], []
    for B in BUD:
        c = sel(m, "cem_gd", B)
        if c: xs.append(B); ys.append(np.mean([r["pred"] - r["true_terminal"] for r in c.values() if r["pred"] is not None]))
    if xs: ax[1].plot(xs, ys, "o-", color=col[m], label=m)
    xs, ys = [], []
    for B in BUD:
        c = sel(m, "h1_closed_eq", B)
        if c: xs.append(B); ys.append(np.mean(list(vals(c).values())))
    if xs: ax[2].plot(xs, ys, "o-", color=col[m], label=m + " H=1 closed")
for name, d in ceil.items():
    if d: ax[0].axhline(np.mean(list(vals(d).values())), color="k", ls="--", lw=0.6)
ax[0].set_xscale("log"); ax[0].set_xlabel("planning budget per 4 pushes (s, exclusive GPU)"); ax[0].set_ylabel("true value change after 4 pushes (lower = better)"); ax[0].set_title("H=4 open-loop plans (dashed = ceilings)"); ax[0].legend(fontsize=6)
ax[1].set_xscale("log"); ax[1].axhline(0, color="k", lw=0.5); ax[1].set_title("optimism (predicted - true), cem_gd"); ax[1].set_xlabel("budget (s)"); ax[1].legend(fontsize=7)
ax[2].set_xscale("log"); ax[2].set_title("H=1 closed loop, equal time"); ax[2].set_xlabel("budget per 4 pushes (s)"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(F / "budget_study_summary.png", dpi=90)
