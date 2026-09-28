"""EXP-0028: the eval_report harness under IMAGE vs SOFT ground-truth scoring,
same models / slates / goal sets. Image-scored reports: EXP-0026 RUN-0001
(default goals) and EXP-0027 RUN-0001 (many goals). Soft: this record's RUN-0001.

Per corpus x value fn x goal set: model means under both, rank agreement
(Kendall tau), mean |shift|, paired noise (median per-pair sd of the per-slate
difference), resolved pairs, and per-model paired CIs under soft scoring.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from scipy import stats
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.paired_stats import paired_comparison, rank_stability

SRC = {"default": ("experiments/EXP-0026-benchmark-power/artifacts/RUN-0001-slaten-per-slate/report.json",
                   "experiments/EXP-0028-soft-truth-scoring/artifacts/RUN-0001-soft-default/report.json"),
       "many": ("experiments/EXP-0027-many-goal-benchmarks/artifacts/RUN-0001-slaten-many-goals/report.json",
                "experiments/EXP-0028-soft-truth-scoring/artifacts/RUN-0001-soft-many/report.json")}
VFS = ("lyapunov", "mass_in_region", "signed_mass")
REF = ("persistence", "random")
out = {}
for gs, (pi, ps) in SRC.items():
    di, ds = json.load(open(REPO / pi)), json.load(open(REPO / ps))
    for c in ds:
        names = [m for m in ds[c] if m not in REF and m in di[c]]
        assert all(ds[c][m]["capture"].get("truth_scoring") == "soft" for m in names)
        for v in VFS:
            X = {}
            for tag, d in (("image", di), ("soft", ds)):
                goals = list(d[c][names[0]]["capture"]["per_slate"])
                X[tag] = np.array([np.nanmean([d[c][m]["capture"]["per_slate"][g][v] for g in goals], 0)
                                   for m in names])
            mi, ms = X["image"].mean(1), X["soft"].mean(1)
            pi_, ps_ = paired_comparison(X["image"], names, n_boot=3000), paired_comparison(X["soft"], names, n_boot=3000)
            rs = rank_stability(X["soft"], names)
            out[f"{gs}|{c}|{v}"] = dict(
                kendall_tau=float(stats.kendalltau(mi, ms).statistic),
                mean_abs_shift=float(np.abs(ms - mi).mean()), mean_shift=float((ms - mi).mean()),
                median_sd_diff_image=float(np.median([p["sd_diff"] for p in pi_])),
                median_sd_diff_soft=float(np.median([p["sd_diff"] for p in ps_])),
                resolved_image=sum(p["resolved_ci"] for p in pi_), resolved_soft=sum(p["resolved_ci"] for p in ps_),
                n_pairs=len(ps_),
                table={m: dict(image=float(mi[k]), soft=float(ms[k]), rank_lo=rs[m]["rank_lo"], rank_hi=rs[m]["rank_hi"],
                               p_first=rs[m]["p_first"]) for k, m in enumerate(names)})
            r = out[f"{gs}|{c}|{v}"]
            print(f"{gs:7s} {c:12s} {v:14s} tau {r['kendall_tau']:+.2f}  shift {r['mean_shift']:+.3f} (|{r['mean_abs_shift']:.3f}|)  "
                  f"paired sd {r['median_sd_diff_image']:.3f}->{r['median_sd_diff_soft']:.3f}  resolved {r['resolved_image']}->{r['resolved_soft']}/{r['n_pairs']}")
(REPO / "experiments/EXP-0028-soft-truth-scoring/results").mkdir(exist_ok=True)
json.dump(out, open(REPO / "experiments/EXP-0028-soft-truth-scoring/results/compare.json", "w"), indent=1)
