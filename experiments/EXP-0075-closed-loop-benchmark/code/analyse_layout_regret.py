"""Summarise results/layout_regret/states.json (layout_regret.py). Gains in mm of Hungarian transport to the 20 uniform_layout slots (positive = layout improved). Means with bootstrap 95 % CI over states.
S0 = non-refine, non-exec candidates (planner-like set); 'pick' = argmin of the model objective over S0; 'exec' = the recorded executed push replayed in the sim.
usage: python -u analyse_layout_regret.py [results/layout_regret/states.json]"""
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parents[1]; rng = np.random.default_rng(0)
p = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/layout_regret/states.json"
S = json.loads(p.read_text())["states"]


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return f"{x.mean():+.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]"


def share(x):
    x = np.asarray(x, float); b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return f"{x.mean():.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]"


rows = []
for s in S:
    g = np.array(s["gain"]); grp = np.array(s["grp"]); src = np.array(s["src"]); cost = np.array(s["cost"]); dcov = np.array(s["dcov"]); moved = np.array(s["moved"])
    s0 = np.where((grp != "refine") & (grp != "exec"))[0]; ie = int(np.where(grp == "exec")[0][0]); pk = s0[int(np.argmin(cost[s0]))]; bs = s0[int(np.argmax(g[s0]))]; ba = int(np.argmax(g))
    lp, lt = np.array(s["lay_pred"]), np.array(s["lay_true"]); dB = np.array(s["dB"]); dA = np.array(s["dA"]); tails = np.where(dB > 10)[0]
    sp = spearmanr(cost[s0], -g[s0]).correlation
    clean = {}
    for i in tails:
        imp = dB[i] - dA[:, i]; oth = np.delete(dA - dB[None], i, axis=1).max(1); ok = (imp >= 1) & (oth <= 1)
        clean[int(i)] = dict(any=bool(ok.any()), any_net=bool(((imp >= 1) & (g > 0)).any()), best_imp=float(imp.max()), exec_imp=float(imp[ie]), exec_clean=bool(ok[ie]),
                             grp=Counter(grp[ok]).most_common(1)[0][0] if ok.any() else None, rank=int((dB > dB[i]).sum()), in_refine_only=bool(ok.any() and not ok[grp != "refine"].any()))
    rows.append(dict(w=s["w"], k=s["k"], goal=s["goal"], exec=g[ie], pick=g[pk], bset=g[bs], ball=g[ba], bset_src=src[bs], ball_src=src[ba], ball_grp=grp[ba], bset_moved=int(moved[bs]), ball_moved=int(moved[ba]), pick_moved=int(moved[pk]),
                     exec_moved=int(moved[ie]), spear=sp, randbank=g[grp == "randbank"].mean(), grpbest={k: g[grp == k].max() for k in set(grp) if k != "exec"}, replay=g[ie] - s["rec_next_gain"],
                     dcov_exec=dcov[ie], dcov_pick=dcov[pk], dcov_best=dcov[ba], dcov_atbest=dcov[ba], lay_gap_exec=lp[ie] - lt[ie], lay_gap_pick=lp[pk] - lt[pk], lay_true_exec=-lt[ie], lay_pred_exec=-lp[ie],
                     lay_corr=float(np.corrcoef(lp[s0], lt[s0])[0, 1]), hung_vs_lay=spearmanr(lt[s0], -g[s0]).correlation, tails=clean, n_tail=len(tails), nfrac_pos=float((g[s0] > 0).mean()), n=len(g),
                     pick_regret_in_set=g[bs] - g[pk], pickfreq=float(s["cost"][pk] <= np.min(cost[s0]) + 1e-9)))
f = lambda key, sel=None: [r[key] for r in rows if sel is None or sel(r)]
md = [f"# EXP-0075 layout regret decomposition ({p.name}; {len(rows)} states, lam=2 H=1 ens2bal; gains in mm of Hungarian transport to slots, positive = better)\n"]
md.append("| subset | n | executed (replayed) | model pick (S0) | best in S0 | best overall (+refine) | best in S0 - executed | best overall - executed | pick - executed | share best-in-S0 > exec+1 mm | share best-overall > exec+1 mm | share pick >= best-in-S0 - 0.5 | Spearman(model cost, true) |")
md.append("|" + "---|" * 13)
for name, sel in [("all", None), ("w15", lambda r: r["w"] == "w15"), ("w10", lambda r: r["w"] == "w10")] + [(f"k={k}", (lambda k: lambda r: r["k"] == k)(k)) for k in sorted({r["k"] for r in rows})]:
    R = [r for r in rows if sel is None or sel(r)]
    if not R: continue
    e = np.array([r["exec"] for r in R]); pk = np.array([r["pick"] for r in R]); bs = np.array([r["bset"] for r in R]); ba = np.array([r["ball"] for r in R])
    md.append(f"| {name} | {len(R)} | {ci(e)} | {ci(pk)} | {ci(bs)} | {ci(ba)} | {ci(bs - e)} | {ci(ba - e)} | {ci(pk - e)} | {share(bs - e > 1)} | {share(ba - e > 1)} | {share(pk >= bs - 0.5)} | {np.nanmean([r['spear'] for r in R]):+.2f} |")
md.append("\n## Best candidate by source (overall best, incl. refinement) and by group (mean over states of the best gain within the group)\n")
md.append("best-in-S0 source: " + str(dict(Counter(f("bset_src")))) + "\nbest-overall group: " + str(dict(Counter(f("ball_grp")))) + "\nbest-overall source: " + str(dict(Counter(f("ball_src")))))
gk = sorted({k for r in rows for k in r["grpbest"]}); md.append("\n| group | mean best gain mm |\n|---|---|"); [md.append(f"| {k} | {ci([r['grpbest'][k] for r in rows if k in r['grpbest']])} |") for k in gk]
md.append(f"| random bank (mean over 16) | {ci(f('randbank'))} |")
md.append(f"\ncubes moved > 2 mm: executed {np.mean(f('exec_moved')):.2f}, model pick {np.mean(f('pick_moved')):.2f}, best-in-S0 {np.mean(f('bset_moved')):.2f}, best-overall {np.mean(f('ball_moved')):.2f}; share of best-overall with 1 cube moved {np.mean(np.array(f('ball_moved')) <= 1):.2f}, >= 3 cubes {np.mean(np.array(f('ball_moved')) >= 3):.2f}")
md.append(f"share of S0 candidates with positive true gain: {np.mean(f('nfrac_pos')):.2f}")
md.append(f"\n## Execution / model-vs-sim (c)\nreplay fidelity (replayed exec gain - recorded next-state gain): {ci(f('replay'))} mm, SD {np.std(f('replay')):.2f} (sim re-run noise + legaliser; n states)\n"
          f"executed push Sinkhorn-layout gain (lam=1 units): pred {ci(f('lay_pred_exec'))}, true {ci(f('lay_true_exec'))}; gap (pred-true change) exec {ci(f('lay_gap_exec'))}, pick {ci(f('lay_gap_pick'))}\n"
          f"within-state corr(pred layout change, true layout change) over S0: mean {np.nanmean(f('lay_corr')):.2f}; Spearman(true Sinkhorn change, true Hungarian) {np.nanmean(f('hung_vs_lay')):.2f}\n"
          f"skeleton coverage change: executed {ci(f('dcov_exec'))}, model pick {ci(f('dcov_pick'))}, candidate with best Hungarian gain {ci(f('dcov_best'))}")
T = [(r, i, d) for r in rows for i, d in r["tails"].items()]
md.append(f"\n## Tail cubes (> 10 mm from assigned slot): {len(T)} cubes in {sum(1 for r in rows if r['n_tail'])} of {len(rows)} states\n")
if T:
    md.append(f"- some simulated candidate moves it >= 1 mm closer with no other cube worse by > 1 mm ('clean'): {share([d['any'] for _, _, d in T])}; ... and net Hungarian gain > 0: {share([d['any_net'] for _, _, d in T])}")
    md.append(f"- the clean push exists only among refinement rows: {np.mean([d['in_refine_only'] for _, _, d in T if d['any']]) if any(d['any'] for _, _, d in T) else float('nan'):.2f} of clean cases")
    md.append(f"- best single-cube improvement over all candidates: {ci([d['best_imp'] for _, _, d in T])} mm; improvement by the executed push: {ci([d['exec_imp'] for _, _, d in T])} mm; executed push is clean for it: {share([d['exec_clean'] for _, _, d in T])}")
    for lo, hi in [(0, 3), (3, 99)]:
        Ts = [d for _, _, d in T if lo <= d["rank"] < hi]
        if Ts: md.append(f"- tail rank {lo}-{hi - 1} (0 = worst; the worst 3 had targeted OT rows added): n {len(Ts)}, clean candidate exists {share([d['any'] for d in Ts])}")
    md.append("- group providing a clean push: " + str(dict(Counter(d['grp'] for _, _, d in T if d['any']))))
out = HERE / "results/layout_regret/summary.md"; out.write_text("\n".join(md) + "\n"); print("\n".join(md))
