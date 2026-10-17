"""EXP-0039 RUN-0003 noise null (2026-10-04, zero-simulation, CPU; EXPLORATORY, not pre-registered). Provenance: grew out
of the audit recompute script (`recompute.py`, session scratchpad 2026-10-04). Reads results/run0003/hm_cem_*.json and
results/run0002_analysis.json only (analyse_run0002.episode_scores = in-goal mass / optimum over pushes 4..16).

Question: is RUN-0003's between-regime Kendall tau (+0.64, 'inconclusive' by the registered rule) distinguishable from the
tau two DISJOINT halves of the SAME regime produce with 8 starts per regime?
  * split-half null: for each regime, all 35 balanced 4/4 splits of its 8 starts -> tau(order on half A, order on half B);
  * episode bootstrap of the between-regime tau (resample the 64 episodes within each regime);
  * tau of RUN-0002's CEM arm (DS-0006 scatter starts, 128 eps) vs RUN-0003 scatter and clump;
  * per-model clump - scatter with episode-bootstrap CI (is the regime effect model-dependent?);
  * executed un-legalisable pushes (legal_ok == False -> executed unchanged) per model per regime, and the order with
    those episodes dropped; per-goal per-regime scores.
Usage: python -u code/analyse_run0003_noise_null.py -> results/run0003_noise_null.{json,md}
"""
import glob, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_run0002 as A2  # noqa: E402
from analyse_run0002_extended import jsonable, CL_LETTERS  # noqa: E402

HERE = A2.HERE; R = HERE / "results/run0003"; rng = np.random.default_rng(0)
REG = {"scatter": list(range(0, 8)), "clump": list(range(16, 24))}


def main():
    files = sorted(glob.glob(str(R / "hm_cem_*.json")))
    eps, bad = {}, {}
    for f in files:
        m = Path(f).stem[len("hm_cem_"):]; eps[m] = A2.episode_scores(f); bad[m] = {}
        for e in json.loads(Path(f).read_text())["episodes"]:
            if e.get("complete"):
                bad[m][(e["goal"], e["start"])] = sum(not x for x in e["legal_ok"])
    models = sorted(eps)
    X, K = {}, {}
    for rg, st in REG.items():
        K[rg] = sorted(set.intersection(*[{k for k in eps[m] if k[1] in st} for m in models]))
        X[rg] = np.array([[eps[m][k]["score"] for k in K[rg]] for m in models])
    out = dict(models=models, n_episodes={rg: len(K[rg]) for rg in REG}, score={rg: {m: float(X[rg][i].mean()) for i, m in enumerate(models)} for rg in REG})
    tau_obs = float(kendalltau(X["scatter"].mean(1), X["clump"].mean(1))[0]); out["tau_scatter_vs_clump"] = tau_obs

    def half(rg, starts):
        return X[rg][:, [i for i, k in enumerate(K[rg]) if k[1] in starts]].mean(1)
    sh = {}
    for rg, st in REG.items():
        taus = []
        for comb in itertools.combinations(st, 4):
            a = set(comb); b = set(st) - a; taus.append(kendalltau(half(rg, a), half(rg, b))[0])
        taus = np.array(taus)
        sh[rg] = dict(n_splits=len(taus), mean=float(taus.mean()), min=float(taus.min()), max=float(taus.max()),
                      q05=float(np.percentile(taus, 5)), q95=float(np.percentile(taus, 95)),
                      frac_splits_tau_le_observed=float(np.mean(taus <= tau_obs + 1e-12)), frac_splits_tau_le_0p43=float(np.mean(taus <= 0.43)),
                      frac_splits_tau_ge_0p71=float(np.mean(taus >= 0.71)), taus=taus.tolist())
    out["split_half_within_regime"] = sh
    # bootstrap tau between regimes
    bt = np.array([kendalltau(X["scatter"][:, rng.integers(0, X["scatter"].shape[1], X["scatter"].shape[1])].mean(1),
                              X["clump"][:, rng.integers(0, X["clump"].shape[1], X["clump"].shape[1])].mean(1))[0] for _ in range(4000)])
    out["bootstrap_tau_between_regimes"] = dict(mean=float(bt.mean()), ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                                                frac_le_0p43=float(np.mean(bt <= 0.43)), frac_ge_0p71=float(np.mean(bt >= 0.71)))
    # vs RUN-0002 CEM (DS-0006 scatter)
    ana = json.loads((HERE / "results/run0002_analysis.json").read_text())
    cl2 = np.array([ana["cem"]["per_model"][m]["score"] for m in models])
    out["tau_run0002_cem_vs"] = {rg: float(kendalltau(cl2, X[rg].mean(1))[0]) for rg in REG}
    # per-model clump - scatter
    pm = {}
    for i, m in enumerate(models):
        d = X["clump"][i].mean() - X["scatter"][i].mean()
        b = [X["clump"][i][rng.integers(0, 64, 64)].mean() - X["scatter"][i][rng.integers(0, 64, 64)].mean() for _ in range(2000)]
        pm[m] = dict(diff=float(d), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    out["per_model_clump_minus_scatter"] = pm
    out["clump_minus_scatter_spread_over_models"] = float(max(v["diff"] for v in pm.values()) - min(v["diff"] for v in pm.values()))
    # executed un-legalisable pushes and order robustness per regime
    il, rob = {}, {}
    for rg in REG:
        il[rg] = {}
        for m in models:
            nb = sum(bad[m][k] for k in K[rg]); il[rg][m] = dict(n_ok_false=int(nb), n_pushes=16 * len(K[rg]), rate=float(nb / (16 * len(K[rg]))))
        tot = sum(v["n_ok_false"] for v in il[rg].values()); il[rg]["_total"] = dict(n_ok_false=int(tot), n_pushes=16 * len(K[rg]) * len(models), rate=float(tot / (16 * len(K[rg]) * len(models))))
        badk = {k for m in models for k in K[rg] if bad[m][k] > 0}; keep = [i for i, k in enumerate(K[rg]) if k not in badk]
        clean = X[rg][:, keep].mean(1)
        pmc = np.array([np.mean([X[rg][i, j] for j, k in enumerate(K[rg]) if bad[m][k] == 0]) for i, m in enumerate(models)])
        rob[rg] = dict(n_kept=len(keep), n_dropped=len(K[rg]) - len(keep), score_clean={m: float(clean[i]) for i, m in enumerate(models)},
                       tau_vs_full=float(kendalltau(X[rg].mean(1), clean)[0]),
                       per_model_drop_clean={m: float(pmc[i]) for i, m in enumerate(models)}, per_model_drop_tau_vs_full=float(kendalltau(X[rg].mean(1), pmc)[0]))
    out["illegal_executed"] = il; out["order_dropping_illegal_episodes"] = rob
    keep_s = [i for i, k in enumerate(K["scatter"]) if all(bad[m][k] == 0 for m in models)]; keep_c = [i for i, k in enumerate(K["clump"]) if all(bad[m][k] == 0 for m in models)]
    out["tau_scatter_vs_clump_clean"] = float(kendalltau(X["scatter"][:, keep_s].mean(1), X["clump"][:, keep_c].mean(1))[0])
    out["tau_scatter_vs_clump_clean_per_model_drop"] = float(kendalltau([rob["scatter"]["per_model_drop_clean"][m] for m in models], [rob["clump"]["per_model_drop_clean"][m] for m in models])[0])
    # per goal per regime
    pg = {}
    for rg in REG:
        pg[rg] = {}
        for g in CL_LETTERS:
            gi = [i for i, k in enumerate(K[rg]) if k[0] == g]
            if gi:
                s = X[rg][:, gi].mean(1)
                pg[rg][g] = dict(n=len(gi), mean_over_models=float(s.mean()), scores={m: float(s[i]) for i, m in enumerate(models)},
                                 tau_vs_regime_pooled=float(kendalltau(s, X[rg].mean(1))[0]))
    out["per_goal"] = pg
    out["per_goal_clump_minus_scatter"] = {g: float(pg["clump"][g]["mean_over_models"] - pg["scatter"][g]["mean_over_models"]) for g in pg["scatter"]}
    out = jsonable(out)
    tmp = HERE / "results/run0003_noise_null.json.tmp"; tmp.write_text(json.dumps(out, indent=1)); tmp.replace(HERE / "results/run0003_noise_null.json")
    L = ["# EXP-0039 RUN-0003 noise null (generated by code/analyse_run0003_noise_null.py, 2026-10-04; exploratory, not pre-registered)\n",
         f"Observed Kendall tau scatter-order vs clump-order: {tau_obs:+.2f} (registered thresholds: supported <= 0.43, refuted >= 0.71).\n",
         "| regime | 4/4 split-half tau: mean | min | 5 % | 95 % | max | frac splits <= observed | frac <= 0.43 | frac >= 0.71 |", "|---|---|---|---|---|---|---|---|---|"]
    for rg, v in sh.items():
        L.append(f"| {rg} | {v['mean']:+.2f} | {v['min']:+.2f} | {v['q05']:+.2f} | {v['q95']:+.2f} | {v['max']:+.2f} | {v['frac_splits_tau_le_observed']:.2f} | {v['frac_splits_tau_le_0p43']:.2f} | {v['frac_splits_tau_ge_0p71']:.2f} |")
    b = out["bootstrap_tau_between_regimes"]
    L += [f"\nepisode-bootstrap of the between-regime tau: mean {b['mean']:+.2f}, 95 % CI [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}]; P(tau <= 0.43) {b['frac_le_0p43']:.2f}, P(tau >= 0.71) {b['frac_ge_0p71']:.2f}",
          f"tau RUN-0002 CEM (DS-0006 scatter, 128 eps) vs RUN-0003 scatter {out['tau_run0002_cem_vs']['scatter']:+.2f}, vs RUN-0003 clump {out['tau_run0002_cem_vs']['clump']:+.2f}",
          f"between-regime tau with illegal-push episodes dropped: paired drop {out['tau_scatter_vs_clump_clean']:+.2f}, per-model drop {out['tau_scatter_vs_clump_clean_per_model_drop']:+.2f}\n",
          "| model | scatter | clump | clump - scatter [CI] | ok=False scatter | ok=False clump | clean scatter (paired / per-model drop) | clean clump (paired / per-model drop) |", "|---|---|---|---|---|---|---|---|"]
    for m in sorted(models, key=lambda m: -out["score"]["scatter"][m]):
        p = pm[m]
        L.append(f"| {m} | {out['score']['scatter'][m]:.3f} | {out['score']['clump'][m]:.3f} | {p['diff']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] | "
                 f"{il['scatter'][m]['n_ok_false']}/{il['scatter'][m]['n_pushes']} | {il['clump'][m]['n_ok_false']}/{il['clump'][m]['n_pushes']} | {rob['scatter']['score_clean'][m]:.3f} / {rob['scatter']['per_model_drop_clean'][m]:.3f} | {rob['clump']['score_clean'][m]:.3f} / {rob['clump']['per_model_drop_clean'][m]:.3f} |")
    for rg in REG:
        t, r = il[rg]["_total"], rob[rg]
        L.append(f"\n{rg}: executed un-legalisable pushes {t['n_ok_false']}/{t['n_pushes']} = {100 * t['rate']:.1f} %; paired drop of {r['n_dropped']} of {len(K[rg])} episodes: tau vs full order {r['tau_vs_full']:+.2f}; per-model drop: tau {r['per_model_drop_tau_vs_full']:+.2f}")
    L += ["\n| goal | scatter mean | clump mean | clump - scatter | scatter tau vs pooled | clump tau vs pooled |", "|---|---|---|---|---|---|"]
    for g in pg["scatter"]:
        L.append(f"| {g} | {pg['scatter'][g]['mean_over_models']:.3f} | {pg['clump'][g]['mean_over_models']:.3f} | {out['per_goal_clump_minus_scatter'][g]:+.3f} | {pg['scatter'][g]['tau_vs_regime_pooled']:+.2f} | {pg['clump'][g]['tau_vs_regime_pooled']:+.2f} |")
    (HERE / "results/run0003_noise_null.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
