"""EXP-0026 A2: paired, slate-resampled power analysis of the eval_report
slateN harness (L20mm / L40mm / randlen_test, 3 goals x 3 value fns).

Unit = slate. Primary quantity: per-slate slateN averaged over the 3 goals
(value fn `lyapunov`, the column the reference table leads with); the other
two value fns are secondary. Also: how much independent information the 3
goals add (variance of paired differences split into between-slate and
within-slate-across-goal parts), which sets whether "more goals" substitutes
for "more slates".

Known-number checks first: (1) the per-slate lists must average back to the
report's own per-goal means (NaN-dropped) to 1e-9; (2) nfd_randlen's
goal-averaged values must match Baselines/common/runs/cross_corpus_report.json.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Baselines.common.paired_stats import (friedman, paired_comparison, power_table,
                                           rank_stability, variance_components)

GOALS = ("random_quadrant", "ring_O", "T")
VFS = ("lyapunov", "mass_in_region", "signed_mass")
REF = ("persistence", "random")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = json.load(open(args.report))
    old = json.load(open(REPO / "Baselines/common/runs/cross_corpus_report.json"))

    # ---- known-number checks ------------------------------------------------
    for c, models in rep.items():
        for m, row in models.items():
            cap = row["capture"]
            if "per_slate" not in cap:
                continue
            for g in GOALS:
                for v in VFS:
                    xs = np.array(cap["per_slate"][g][v], dtype=float)
                    assert abs(np.nanmean(xs) - cap["per_goal"][g][v]) < 1e-9, (c, m, g, v)
        if "nfd_randlen" in models and "nfd_randlen" in old.get(c, {}):
            for v in VFS:
                a = models["nfd_randlen"]["capture"]["averaged_over_goals"][v]
                b = old[c]["nfd_randlen"]["capture"]["averaged_over_goals"][v]
                print(f"known-number {c}/nfd_randlen/{v}: new {a:.6f} old {b:.6f} diff {a-b:+.2e}")

    out = {"report": args.report, "corpora": {}}
    for c, models in rep.items():
        names = [m for m in models if m not in REF]
        S = models[names[0]]["capture"]["n_slates"]
        cube = np.array([[[models[m]["capture"]["per_slate"][g][v] for g in GOALS]
                          for v in VFS] for m in names], dtype=float)   # (M, V, G, S)
        out["corpora"][c] = {"n_slates": S, "models": names, "per_vf": {}}
        for vi, v in enumerate(VFS):
            X = np.nanmean(cube[:, vi], axis=1)        # goal-averaged per slate (M, S)
            pairs = paired_comparison(X, names)
            # goal replication: for every pair, split var of d[g, s] into
            # between-slate (mean over goals) and within-slate parts.
            icc = []
            for i, j in itertools.combinations(range(len(names)), 2):
                D = cube[i, vi] - cube[j, vi]            # (G, S)
                D = D[:, ~np.isnan(D).any(0)]
                ms_between = len(GOALS) * D.mean(0).var(ddof=1)
                ms_within = ((D - D.mean(0)) ** 2).sum() / (D.shape[1] * (len(GOALS) - 1))
                s2_slate = max((ms_between - ms_within) / len(GOALS), 0.0)
                icc.append((s2_slate, ms_within))
            icc = np.array(icc)
            s2s, s2w = np.median(icc[:, 0]), np.median(icc[:, 1])
            small = [p for p in pairs if abs(p["mean_diff"]) < 0.03]
            res = dict(
                means={m: float(np.nanmean(X[k])) for k, m in enumerate(names)},
                variance_components=variance_components(X), friedman=friedman(X),
                pairs=pairs,
                power=power_table(pairs, [0.01, 0.02, 0.03, 0.05]),
                rank_stability=rank_stability(X, names),
                goal_replication=dict(
                    median_var_between_slate=float(s2s), median_var_within_slate=float(s2w),
                    icc=float(s2s / (s2s + s2w)) if s2s + s2w > 0 else float("nan"),
                    note="sd_d^2 for a design with S slates x G goals ~= (s2_slate + s2_within/G)/S"),
                n_pairs=len(pairs), n_resolved_ci=sum(p["resolved_ci"] for p in pairs),
                n_holm_05=sum(p["p_holm"] < 0.05 for p in pairs),
                small_margin=dict(n=len(small),
                                  n_unresolved=sum(not p["resolved_ci"] for p in small)),
            )
            out["corpora"][c]["per_vf"][v] = res

    # ---- does the ranking transfer across corpora? ----------------------------
    common = sorted(set.intersection(*[set(d["models"]) for d in out["corpora"].values()]))
    out["cross_corpus_kendall"] = {}
    for v in VFS:
        for c1, c2 in itertools.combinations(out["corpora"], 2):
            a = [out["corpora"][c1]["per_vf"][v]["means"][m] for m in common]
            b = [out["corpora"][c2]["per_vf"][v]["means"][m] for m in common]
            t, p = stats.kendalltau(a, b)
            out["cross_corpus_kendall"][f"{v}:{c1}~{c2}"] = dict(tau=float(t), p=float(p))

    Path(args.out).write_text(json.dumps(out, indent=2))

    # ---- print ----------------------------------------------------------------
    for c, dc in out["corpora"].items():
        print(f"\n######## {c}  (n_slates={dc['n_slates']}, {len(dc['models'])} models)")
        for v, r in dc["per_vf"].items():
            vc, fr, pw, gr = r["variance_components"], r["friedman"], r["power"], r["goal_replication"]
            print(f"== {v}: resolved CI {r['n_resolved_ci']}/{r['n_pairs']}, Holm<.05 "
                  f"{r['n_holm_05']}/{r['n_pairs']}; |d|<0.03 unresolved "
                  f"{r['small_margin']['n_unresolved']}/{r['small_margin']['n']}; Friedman p={fr['p']:.2g} "
                  f"W={fr['kendall_w']:.2f}")
            print(f"   sd model-means {vc['model_sd_of_means']:.3f} state-means {vc['state_sd_of_means']:.3f} "
                  f"residual {vc['residual_sd']:.3f}; median sd_diff {pw['median_sd_diff']:.3f}; "
                  f"goal ICC {gr['icc']:.2f}")
            print("   required slates: " + "  ".join(
                f"d={k}: {x['alpha_05']} (Bonf {x['bonferroni']})" for k, x in pw["required_n"].items()))
            if v == "lyapunov":
                rs = r["rank_stability"]
                for m in sorted(r["means"], key=lambda m: -r["means"][m]):
                    print(f"     {m:40s} {r['means'][m]:.3f}  rank95 [{rs[m]['rank_lo']},{rs[m]['rank_hi']}]"
                          f"  P1={rs[m]['p_first']:.2f}")
    print("\ncross-corpus Kendall tau of model means:")
    for k, x in out["cross_corpus_kendall"].items():
        print(f"   {k:45s} tau={x['tau']:+.2f} p={x['p']:.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
