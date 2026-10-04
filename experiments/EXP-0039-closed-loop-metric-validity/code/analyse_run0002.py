"""EXP-0039 RUN-0002 analysis (pre-registered design: runs/RUN-0002-legal-headroom-rerun/DESIGN.md).

Closed loop: per (arm, model) the score of an episode = mean over pushes 4..16 of in-goal mass fraction / the
goal's optimum (EXP-0046 vstar.json `mass_frac_best_placement`) -- EXP-0051 analyse.py's exact mass formula.
Pairwise paired differences (paired by goal x start), bootstrap over episodes, Holm over all pairs.
Offline (DS-0016 test pools, legal by construction): accuracy_1, slateN, slateN_tough from the scored metrics;
within-pool Spearman(vp, vt), optimism at the pick (vt - vp at argmin vp, cost sense, > 0 = over-predicted
improvement) and top-1 regret (vt[pick] - min vt) from slateN_tough_raw, averaged over (pool, goal).
Per metric: Spearman rho with the closed-loop model means (episode-bootstrap CI) and agreement with the
Holm-resolved pairs (higher-is-better metrics as is; optimism and regret flipped).
Usage: python code/analyse_run0002.py [--pilot]   -> results/run0002_analysis.{json,md}
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask  # noqa: E402
from Baselines.common.goals import mass_in_region  # noqa: E402
from simple_mpc.adapters import occ_for_scoring  # noqa: E402

HERE = REPO / "experiments/EXP-0039-closed-loop-metric-validity"
R = HERE / "results/run0002"
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
K0, K1 = 4, 16
rng = np.random.default_rng(0)


def episode_scores(path):
    out = {}
    for e in json.loads(Path(path).read_text())["episodes"]:
        if not e.get("complete") or not e.get("states"):
            continue
        S = torch.tensor(e["states"]).float()
        occ = occ_for_scoring(S)
        mf = (mass_in_region(occ, torch.from_numpy(goal_mask(e["goal"]))) / occ.reshape(len(S), -1).sum(1)).numpy()
        rel = mf / VS.get(e["goal"], {}).get("mass_frac_best_placement", 1.0)
        k1 = min(K1, len(rel) - 1)
        out[(e["goal"], e["start"])] = dict(score=float(rel[K0:k1 + 1].mean()), k8=float(rel[min(8, len(rel) - 1)]),
                                            k16=float(rel[k1]), shifted=float(np.mean([x > 0 for x in e.get("legal_shift_m", [0])])))
    return out


def offline_metrics(models):
    src = {}
    for f in (REPO / "experiments/EXP-0059-retrieval-transition-model/results/test_v2.json", R / "offline_ds0016_scores.json"):
        if f.exists():
            src.update({k: v for k, v in json.loads(f.read_text()).items() if isinstance(v, dict) and "metrics" in v})
    out = {}
    for m in models:
        if m not in src:
            continue
        met = src[m]["metrics"]; raw = src[m]["raw"]["slateN_tough_raw"]
        sp, op, rg = [], [], []
        for g, pools in raw.items():
            for p in pools:
                vt, vp = np.asarray(p["vt"]), np.asarray(p["vp"])
                if np.ptp(vp) > 0 and np.ptp(vt) > 0:
                    sp.append(spearmanr(vp, vt)[0])
                i = int(np.argmin(vp)); op.append(vt[i] - vp[i]); rg.append(vt[i] - vt.min())
        out[m] = dict(accuracy_1=met.get("accuracy_1"), slateN=met.get("slateN"), slateN_tough=met.get("slateN_tough"),
                      spearman=float(np.mean(sp)), optimism=float(np.mean(op)), top1_regret=float(np.mean(rg)))
    return out


HIGHER = dict(accuracy_1=True, slateN=True, slateN_tough=True, spearman=True, optimism=False, top1_regret=False)


def analyse_arm(arm, files):
    eps = {Path(f).stem.split(f"hm_{arm}_", 1)[1]: episode_scores(f) for f in files}
    eps = {m: v for m, v in eps.items() if v}
    models = sorted(eps)
    keys = sorted(set.intersection(*[set(v) for v in eps.values()])) if eps else []
    X = np.array([[eps[m][k]["score"] for k in keys] for m in models])           # (M, E)
    res = dict(models=models, n_episodes=len(keys), per_model={}, pairs=[], metrics={})
    for i, m in enumerate(models):
        bs = rng.choice(X[i], (2000, len(keys))).mean(1)
        res["per_model"][m] = dict(score=float(X[i].mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                   k8=float(np.mean([eps[m][k]["k8"] for k in keys])), k16=float(np.mean([eps[m][k]["k16"] for k in keys])),
                                   legal_shift_rate=float(np.mean([eps[m][k]["shifted"] for k in keys])))
    pv = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            d = X[i] - X[j]
            idx = rng.integers(0, len(d), (4000, len(d))); bm = d[idx].mean(1)
            p = float(min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean())))
            pv.append((p, i, j, float(d.mean()), [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]))
    order = sorted(range(len(pv)), key=lambda t: pv[t][0]); n = len(pv); still = True
    for r, t in enumerate(order):
        p, i, j, dm, ci = pv[t]
        still = still and p * (n - r) < 0.05
        res["pairs"].append(dict(a=models[i], b=models[j], diff=dm, ci=ci, p=p, holm_resolved=bool(still)))
    off = offline_metrics(models)
    res["offline"] = off
    resolved = [q for q in res["pairs"] if q["holm_resolved"]]
    res["n_resolved"] = len(resolved)
    for met, hib in HIGHER.items():
        ms = [m for m in models if m in off and off[m].get(met) is not None]
        if len(ms) < 4:
            continue
        mv = np.array([off[m][met] for m in ms]) * (1 if hib else -1)
        cl = np.array([res["per_model"][m]["score"] for m in ms])
        rho = spearmanr(mv, cl)[0]
        ii = [models.index(m) for m in ms]
        boots = []
        for _ in range(1000):
            e = rng.integers(0, len(keys), len(keys))
            boots.append(spearmanr(mv, X[ii][:, e].mean(1))[0])
        agree = [((off[q["a"]][met] - off[q["b"]][met]) * (1 if hib else -1) * q["diff"]) > 0
                 for q in resolved if q["a"] in off and q["b"] in off]
        res["metrics"][met] = dict(n_models=len(ms), rho=float(rho), rho_ci=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                                   pair_agreement=float(np.mean(agree)) if agree else None, n_pairs=len(agree))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true"); a = ap.parse_args()
    out = {}
    for arm in ("cem", "rank"):
        files = sorted(glob.glob(str(R / f"hm_{arm}_*.json")))
        if files:
            out[arm] = analyse_arm(arm, files)
    tmp = HERE / "results/run0002_analysis.json.tmp"; tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(HERE / "results/run0002_analysis.json")
    L = ["# EXP-0039 RUN-0002 analysis (generated by code/analyse_run0002.py)\n"]
    for arm, r in out.items():
        L.append(f"\n## arm {arm}: {len(r['models'])} models, {r['n_episodes']} paired episodes, {r['n_resolved']} Holm-resolved pairs\n")
        L.append("| model | score (mass/opt, k4-16) [CI] | k8 | k16 | legal shift rate | " + " | ".join(HIGHER) + " |")
        L.append("|---" * (5 + len(HIGHER)) + "|")
        for m, v in sorted(r["per_model"].items(), key=lambda x: -x[1]["score"]):
            o = r["offline"].get(m, {})
            L.append(f"| {m} | {v['score']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {v['k8']:.3f} | {v['k16']:.3f} | {v['legal_shift_rate']:.2f} | "
                     + " | ".join(f"{o[k]:.4f}" if o.get(k) is not None else "-" for k in HIGHER) + " |")
        L.append("\n| offline metric | n models | Spearman rho with closed loop [episode-bootstrap CI] | agreement with resolved pairs |\n|---|---|---|---|")
        for k, v in r["metrics"].items():
            pa = f"{v['pair_agreement']:.2f} of {v['n_pairs']}" if v["pair_agreement"] is not None else "-"
            L.append(f"| {k} | {v['n_models']} | {v['rho']:+.2f} [{v['rho_ci'][0]:+.2f}, {v['rho_ci'][1]:+.2f}] | {pa} |")
        L.append("\nresolved pairs: " + "; ".join(f"{q['a']} {'>' if q['diff'] > 0 else '<'} {q['b']} ({q['diff']:+.3f})" for q in r["pairs"] if q["holm_resolved"]))
    (HERE / "results/run0002_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
