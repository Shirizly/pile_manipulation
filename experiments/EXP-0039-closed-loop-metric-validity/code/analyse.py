"""EXP-0039 analysis (DESIGN.md addendum 3: 4 models, paired primary analysis).

Closed-loop score per episode: improvement V0 - V8 (soft lyapunov; higher = better).
X[model, planner, episode] over the 16 (goal, start) episodes, identical across models.

1. per (model, planner): mean improvement, sd, evaluations per decision;
2. PRIMARY: per planner, all 6 model pairs, paired over the 16 episodes (bootstrap CI,
   sign-flip p, Holm within planner); the seed pair (nfd_3ch_randlen vs seed1) is the
   closed-loop seed floor; planner pairs per model likewise;
3. offline metrics per model -- DS-0006 whole-pool slateN and pool spearman (lyapunov,
   30 goals; cached predictions, linear_switched_soft predicted here), grad_capture
   (EXP-0037; not for seed1), accuracy (randlen_test; EXP-0038 / EXP-0036) -- and, for
   every pair x planner, whether the offline metric orders the pair as closed loop does
   (descriptive, n = 4 models).
-> results/analysis.json (rewritten after each block).
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0036-seed-noise-floor/code"))
import score_seeds as ss
from Baselines.common.paired_stats import paired_comparison

HERE = REPO / "experiments/EXP-0039-closed-loop-metric-validity"
RES, ART = HERE / "results/analysis.json", HERE / "artifacts/RUN-0001"
MODELS = ["nfd_3ch_randlen", "nfd_3ch_randlen_seed1", "linear_switched_soft", "nfd_residual_worldframe_noaug_ep43"]
PLANNERS = ["rank", "gd", "cem"]
SEED_PAIR = ("nfd_3ch_randlen", "nfd_3ch_randlen_seed1")


def save(out):
    RES.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(RES) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); tmp.replace(RES)


def offline_ds0006():
    """Whole-pool slateN and pool spearman, lyapunov, 30 goals, DS-0006."""
    path, src = ss.CORPORA["DS-0006"]
    rows = ss.load_rows(path)
    slate = rows.slate_idx.long().numpy()
    idx = [np.nonzero(slate == s)[0] for s in range(int(slate.max()) + 1)]
    vt = torch.load(src / "truth.pt", weights_only=False)["dv"].numpy()[..., 0]
    files = {"nfd_3ch_randlen": src / "pred_nfd_3ch_randlen.pt",
             "nfd_residual_worldframe_noaug_ep43": src / "pred_nfd_residual_worldframe_noaug_ep43.pt",
             "nfd_3ch_randlen_seed1": ss.ART / "pred_DS-0006_nfd_3ch_randlen_seed1.pt",
             "linear_switched_soft": ART / "pred_DS-0006_linear_switched_soft.pt"}
    if not files["linear_switched_soft"].exists():
        ART.mkdir(parents=True, exist_ok=True)
        acts = torch.cat([rows.p_starts[:, :2], rows.p_stops[:, :2]], 1).float()
        masks, dists = ss.goal_tensors("cpu")
        ss.save_atomic({"dv": ss.predict("linear_switched_soft", rows, idx, acts, masks, dists, vt.shape[1]),
                        "model": "linear_switched_soft"}, files["linear_switched_soft"])
    out = {}
    for m, f in files.items():
        vp = torch.load(f, weights_only=False)["dv"].numpy()[..., 0]
        cap = ss.per_state_capture(vp, vt, idx, hib=False)
        rho = np.nanmean([[stats.spearmanr(vp[i, g], vt[i, g]).statistic for g in range(vt.shape[1])] for i in idx])
        out[m] = dict(slateN=float(np.nanmean(cap)), spearman=float(rho))
    return out


def main():
    d = json.loads((HERE / "results/episodes.json").read_text())
    E = [e for e in d["episodes"] if e.get("complete")]
    combos = sorted({(e["goal"], e["start"]) for e in E})
    X = np.full((len(MODELS), len(PLANNERS), len(combos)), np.nan)
    N = np.full_like(X, np.nan)
    for e in E:
        if e["model"] in MODELS:
            i, p, c = MODELS.index(e["model"]), PLANNERS.index(e["planner"]), combos.index((e["goal"], e["start"]))
            X[i, p, c] = e["values"][0] - e["values"][-1]
            N[i, p, c] = np.mean(e["n_evals"])
    assert not np.isnan(X).any(), "missing episodes"
    out = dict(n_episodes=len(E), combos=[list(c) for c in combos], models=MODELS, planners=PLANNERS,
               closed_loop={m: {pl: dict(mean=float(X[i, p].mean()), sd=float(X[i, p].std(ddof=1)),
                                         evals_per_decision=float(N[i, p].mean()))
                                for p, pl in enumerate(PLANNERS)} for i, m in enumerate(MODELS)})
    # 2. paired model comparisons per planner; planner comparisons per model
    out["model_pairs"] = {pl: paired_comparison(X[:, p], MODELS) for p, pl in enumerate(PLANNERS)}
    out["planner_pairs"] = {m: paired_comparison(X[i], PLANNERS) for i, m in enumerate(MODELS)}
    out["seed_gap"] = {pl: next(r for r in out["model_pairs"][pl]
                                if {r["a"], r["b"]} == set(SEED_PAIR)) for pl in PLANNERS}
    save(out)
    # 3. offline metrics and pairwise order agreement
    off = offline_ds0006()
    g37 = json.loads((REPO / "experiments/EXP-0037-gradient-benchmark-ds0006/results/analysis.json").read_text())
    t38 = json.loads((REPO / "experiments/EXP-0038-offline-metric-table/results/metric_table.json").read_text())
    for m in MODELS:
        off[m]["grad_capture"] = g37["grad_capture"]["means"].get(m)
        off[m]["accuracy"] = t38.get(m, {}).get("accuracy_randlen_test")
    acc_seed = REPO / "experiments/EXP-0036-seed-noise-floor/results/randlen_test/seed1.json"
    if acc_seed.exists():
        off["nfd_3ch_randlen_seed1"]["accuracy"] = json.loads(acc_seed.read_text())["randlen_test"]["nfd_randlen"]["accuracy"]
    for i, m in enumerate(MODELS):
        off[m]["evals_per_decision"] = {pl: float(N[i, p].mean()) for p, pl in enumerate(PLANNERS)}
    out["offline"] = off
    agree = {}
    for pl in PLANNERS:
        rows = []
        for r in out["model_pairs"][pl]:
            a, b = r["a"], r["b"]
            row = dict(pair=f"{a} - {b}", closed_loop=r["mean_diff"], p_holm=r["p_holm"])
            for k in ("slateN", "spearman", "grad_capture", "accuracy"):
                va, vb = off[a].get(k), off[b].get(k)
                row[k + "_agrees"] = None if va is None or vb is None else bool(np.sign(va - vb) == np.sign(r["mean_diff"]))
            rows.append(row)
        agree[pl] = rows
    out["order_agreement"] = agree
    out["model_order"] = {pl: sorted(MODELS, key=lambda m: -out["closed_loop"][m][pl]["mean"]) for pl in PLANNERS}
    out["offline_order"] = {k: sorted([m for m in MODELS if off[m].get(k) is not None], key=lambda m: -off[m][k])
                            for k in ("slateN", "spearman", "grad_capture", "accuracy")}
    save(out)
    # print
    print("closed-loop improvement V0-V8, mean (sd) [evals/decision]:")
    for m in MODELS:
        print(f"  {m:38s} " + "  ".join(f"{pl} {out['closed_loop'][m][pl]['mean']:+.3f} ({out['closed_loop'][m][pl]['sd']:.3f}) "
                                          f"[{out['closed_loop'][m][pl]['evals_per_decision']:.0f}]" for pl in PLANNERS))
    for pl in PLANNERS:
        print(f"\n== {pl}: model pairs")
        for r in out["model_pairs"][pl]:
            print(f"  {r['a']:36s} - {r['b']:36s} {r['mean_diff']:+.4f} {[round(r['ci_lo'], 4), round(r['ci_hi'], 4)]} p {r['p_signflip']:.3f} Holm {r['p_holm']:.3f}")
    for m in MODELS:
        print(f"\n== {m}: planner pairs")
        for r in out["planner_pairs"][m]:
            print(f"  {r['a']} - {r['b']} {r['mean_diff']:+.4f} {[round(r['ci_lo'], 4), round(r['ci_hi'], 4)]} Holm {r['p_holm']:.3f}")
    print("\noffline:", json.dumps({m: {k: v for k, v in o.items() if k != 'evals_per_decision'} for m, o in off.items()}, indent=0))
    print("closed-loop order:", out["model_order"])
    print("offline order:", out["offline_order"])
    for pl in PLANNERS:
        print(f"\n== agreement {pl}")
        for r in agree[pl]:
            print("  ", r)


if __name__ == "__main__":
    main()
