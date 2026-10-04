"""EXP-0039 RUN-0003 analysis (pre-registered in EXPERIMENT.md): clump (DS-0016 pools 16-23) vs scatter (pools 0-7)
start regimes. Per regime: model scores (analyse_run0002.episode_scores), Holm-resolved pairs; Kendall tau between the
regimes' model orders; resolved pairs that reverse sign; S1 = mean over models of (clump - scatter) score with an
episode-bootstrap CI. -> results/run0003_analysis.{json,md}"""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_run0002 as A2

HERE = A2.HERE; R = HERE / "results/run0003"; rng = np.random.default_rng(0)
REG = {"scatter": set(range(0, 8)), "clump": set(range(16, 24))}


def pairs(models, X):
    pv = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            d = X[i] - X[j]; bm = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
            pv.append((float(min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))), models[i], models[j], float(d.mean())))
    pv.sort(); n = len(pv); out = {}; still = True
    for r, (p, a, b, dm) in enumerate(pv):
        still = still and p * (n - r) < 0.05
        out[(a, b)] = dict(diff=dm, p=p, resolved=bool(still))
    return out


eps = {Path(f).stem[len("hm_cem_"):]: A2.episode_scores(f) for f in sorted(glob.glob(str(R / "hm_cem_*.json")))}
eps = {m: v for m, v in eps.items() if v}
models = sorted(eps)
res = {"models": models, "regimes": {}}
X = {}
for rg, st in REG.items():
    keys = sorted(set.intersection(*[{k for k in eps[m] if k[1] in st} for m in models]))
    X[rg] = np.array([[eps[m][k]["score"] for k in keys] for m in models])
    pr = pairs(models, X[rg])
    res["regimes"][rg] = dict(n_episodes=len(keys), score={m: float(X[rg][i].mean()) for i, m in enumerate(models)},
                              resolved=[f"{a} {'>' if v['diff'] > 0 else '<'} {b} ({v['diff']:+.3f})" for (a, b), v in pr.items() if v["resolved"]])
    res["regimes"][rg]["_pairs"] = pr
sc, cl = res["regimes"]["scatter"], res["regimes"]["clump"]
tau = kendalltau([sc["score"][m] for m in models], [cl["score"][m] for m in models])[0]
rev = [f"{a} vs {b}: scatter {sc['_pairs'][(a, b)]['diff']:+.3f} ({'res' if sc['_pairs'][(a, b)]['resolved'] else 'n.s.'}), clump {cl['_pairs'][(a, b)]['diff']:+.3f} ({'res' if cl['_pairs'][(a, b)]['resolved'] else 'n.s.'})"
       for (a, b) in sc["_pairs"] if np.sign(sc["_pairs"][(a, b)]["diff"]) != np.sign(cl["_pairs"][(a, b)]["diff"])
       and (sc["_pairs"][(a, b)]["resolved"] or cl["_pairs"][(a, b)]["resolved"])]
rev_both = [r for r in rev if r.count("(res)") == 2]
d = X["clump"].mean(0) - X["scatter"].mean(0) if X["clump"].shape == X["scatter"].shape else None
s1 = float(X["clump"].mean() - X["scatter"].mean())
bs = [X["clump"][:, rng.integers(0, X["clump"].shape[1], X["clump"].shape[1])].mean() - X["scatter"][:, rng.integers(0, X["scatter"].shape[1], X["scatter"].shape[1])].mean() for _ in range(4000)]
res.update(kendall_tau_regimes=float(tau), reversals_one_side_resolved=rev, reversals_both_resolved=rev_both,
           S1_clump_minus_scatter=[s1, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           S2_verdict=("supported" if (rev_both or tau <= 0.43) else ("refuted" if (tau >= 0.71 and not rev_both) else "inconclusive")))
for rg in REG:
    res["regimes"][rg].pop("_pairs")
(HERE / "results/run0003_analysis.json").write_text(json.dumps(res, indent=1))
L = [f"# EXP-0039 RUN-0003 analysis (code/analyse_run0003.py)\n", "| model | scatter | clump |", "|---|---|---|"]
L += [f"| {m} | {sc['score'][m]:.3f} | {cl['score'][m]:.3f} |" for m in sorted(models, key=lambda m: -sc['score'][m])]
L += [f"\nKendall tau (scatter order vs clump order): {tau:+.2f}", f"S1 clump - scatter (mean over models): {s1:+.3f} [{res['S1_clump_minus_scatter'][1]:+.3f}, {res['S1_clump_minus_scatter'][2]:+.3f}]",
      f"resolved pairs: scatter {len(sc['resolved'])}, clump {len(cl['resolved'])}", "reversals (resolved in >= 1 regime): " + ("; ".join(rev) or "none"),
      "reversals resolved in BOTH: " + ("; ".join(rev_both) or "none"), f"S2 verdict (pre-registered rule): {res['S2_verdict']}"]
(HERE / "results/run0003_analysis.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
