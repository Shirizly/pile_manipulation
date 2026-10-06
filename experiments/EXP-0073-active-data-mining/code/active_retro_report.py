"""PILOT: aggregate the retrospective-loop evals (runs/active_pilot/retro/eval_<arm>_s<seed>.npz)."""
import sys, glob, json, numpy as np
from scipy.stats import rankdata
RR = "experiments/EXP-0073-active-data-mining/runs/active_pilot/retro/"; AP = "experiments/EXP-0073-active-data-mining/runs/active_pilot/"
arms = ["base", "rand", "rand1000", "ens", "gray", "epi", "dec", "div", "oracle"]
E = {a: [np.load(f) for f in sorted(glob.glob(RR + f"eval_{a}_s*.npz"))] for a in arms}; E = {a: v for a, v in E.items() if v}
red = np.load(AP + "reducible_rows.npz"); n_i = red["n_i"]; B = len(n_i)
base17 = np.mean([e["ds17_e_sw"] for e in E["base"]], 0); hard17 = base17 >= np.quantile(base17, .8)
base16 = np.mean([e["ds16_e_sw"] for e in E["base"]], 0); hard16 = base16 >= np.quantile(base16, .8)
rk = rankdata(n_i) / B; bucket = np.digitize(rk, [1 / 3, 2 / 3]); hn = n_i >= np.quantile(n_i, .9)
ms = lambda v: f"{np.mean(v):.4f}+-{np.std(v, ddof=1) if len(v) > 1 else 0:.4f}"
print("DS-0017 (selection set), mean+-sd over retrain seeds, n seeds in []  [slateN first]")
print(f"{'arm':9s} n   rows  slateN(win)        acc1(win,pools)    e_sw all         e_sw hard-20%      | DS-0016: slateN        e_sw all        hard-20%")
out = {}
for a, v in E.items():
    s17 = [e["ds17_slateN"] for e in v]; a17 = [e["ds17_acc1"] for e in v]; e17 = [e["ds17_e_sw"].mean() for e in v]; h17 = [e["ds17_e_sw"][hard17].mean() for e in v]
    s16 = [e["ds16_slateN"] for e in v]; e16 = [e["ds16_e_sw"].mean() for e in v]; h16 = [e["ds16_e_sw"][hard16].mean() for e in v]
    print(f"{a:9s} [{len(v)}] {int(v[0]['n_rows'])}  {ms(s17)}  {ms(a17)}  {ms(e17)}  {ms(h17)} | {ms(s16)}  {ms(e16)}  {ms(h16)}")
    out[a] = dict(n=len(v), slateN17=float(np.mean(s17)), e17=float(np.mean(e17)), hard17=float(np.mean(h17)), e16=float(np.mean(e16)), hard16=float(np.mean(h16)))
print("\nPaired vs 'rand' (same retrain seed): delta e_sw (negative = better), mean over seeds [and per-seed values]")
for a in E:
    if a in ("rand",) or "rand" not in E: continue
    n = min(len(E[a]), len(E["rand"]))
    for nm, key, msk in (("ds17 all", "ds17_e_sw", slice(None)), ("ds17 hard", "ds17_e_sw", hard17), ("ds16 all", "ds16_e_sw", slice(None)), ("ds16 hard", "ds16_e_sw", hard16)):
        d = [E[a][i][key][msk].mean() - E["rand"][i][key][msk].mean() for i in range(n)]
        out.setdefault(a, {})["d_" + nm.replace(" ", "_")] = float(np.mean(d))
    d17 = [E[a][i]["ds17_slateN"] - E["rand"][i]["ds17_slateN"] for i in range(n)]
    print(f"{a:9s} dslateN17 {np.mean(d17):+.4f}  d e_sw ds17 all {out[a]['d_ds17_all']:+.3f} hard {out[a]['d_ds17_hard']:+.3f} | ds16 all {out[a]['d_ds16_all']:+.3f} hard {out[a]['d_ds16_hard']:+.3f}")
print("\nDS-0016 e_sw by NOISE-FLOOR bucket of the row (n_i terciles; top-10% noisiest) -- seed-mean per arm")
print(f"{'arm':9s} low      mid      high     noisiest10%")
for a, v in E.items():
    e = np.mean([x["ds16_e_sw"] for x in v], 0); print(f"{a:9s} " + "  ".join(f"{e[bucket == i].mean():.3f}" for i in range(3)) + f"   {e[hn].mean():.3f}")
print("\nDS-0016 e_RED-style bucket (e_sw - n_i/K proxy not recomputed; shown: share of seed-mean e_sw in noisiest-30% rows)")
json.dump(out, open("experiments/EXP-0073-active-data-mining/results/active_retro.json", "w"), indent=1)
