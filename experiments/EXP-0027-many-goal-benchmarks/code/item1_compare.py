"""EXP-0027 item 1: the eval_report harness under its default 3 goals (EXP-0026
RUN-0001) vs the 30-goal "many" set (RUN-0001 here), same slates and models.

C1 difficulty match (|mean_many - mean_default| <= 0.05 per model/corpus),
C2 resolved-pair count ratio (goal-averaged lyapunov), plus goal ICC, sd shrink
and the re-issued reference table with per-model bootstrap CIs and rank
intervals. Unit = slate throughout; goals are averaged within slate.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Baselines.common.paired_stats import paired_comparison, rank_stability, required_n

VFS = ("lyapunov", "mass_in_region", "signed_mass")
REF = ("persistence", "random")


def cube(rep_c, names, v):
    goals = list(rep_c[names[0]]["capture"]["per_slate"])
    return np.array([[rep_c[m]["capture"]["per_slate"][g][v] for g in goals] for m in names],
                    dtype=float)                                   # (M, G, S)


def icc_median(C):
    out = []
    for i, j in itertools.combinations(range(C.shape[0]), 2):
        D = C[i] - C[j]
        D = D[:, ~np.isnan(D).any(0)]
        G, S = D.shape
        if S < 3:
            continue
        msb = G * D.mean(0).var(ddof=1)
        msw = ((D - D.mean(0)) ** 2).sum() / (S * (G - 1))
        s2 = max((msb - msw) / G, 0.0)
        out.append(s2 / (s2 + msw) if s2 + msw > 0 else np.nan)
    return float(np.nanmedian(out)) if out else float("nan")


def boot_ci(x, n=10000, seed=0):
    x = x[~np.isnan(x)]
    r = np.random.default_rng(seed)
    b = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--default", required=True)
    ap.add_argument("--many", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d0, d1 = json.load(open(a.default)), json.load(open(a.many))
    res = {}
    for c in d1:
        names = [m for m in d1[c] if m not in REF]
        assert d0[c][names[0]]["capture"]["slate_ids"] == d1[c][names[0]]["capture"]["slate_ids"]
        rc = {}
        for v in VFS:
            C0, C1 = cube(d0[c], names, v), cube(d1[c], names, v)
            X0, X1 = np.nanmean(C0, 1), np.nanmean(C1, 1)          # (M, S)
            p0, p1 = paired_comparison(X0, names), paired_comparison(X1, names)
            n0, n1 = sum(p["resolved_ci"] for p in p0), sum(p["resolved_ci"] for p in p1)
            h0, h1 = sum(p["p_holm"] < .05 for p in p0), sum(p["p_holm"] < .05 for p in p1)
            sd0 = float(np.median([p["sd_diff"] for p in p0])); sd1 = float(np.median([p["sd_diff"] for p in p1]))
            means0 = {m: float(np.nanmean(X0[k])) for k, m in enumerate(names)}
            means1 = {m: float(np.nanmean(X1[k])) for k, m in enumerate(names)}
            rs0, rs1 = rank_stability(X0, names), rank_stability(X1, names)
            rc[v] = dict(
                n_pairs=len(p0), resolved_default=n0, resolved_many=n1,
                ratio=(n1 / n0 if n0 else float("inf")), holm_default=h0, holm_many=h1,
                median_sd_default=sd0, median_sd_many=sd1, sd_shrink=sd0 / sd1,
                icc_many=icc_median(C1), nan_frac_many=float(np.isnan(C1).mean()),
                req_slates_d02_default=required_n(sd0, 0.02), req_slates_d02_many=required_n(sd1, 0.02),
                max_abs_mean_shift=max(abs(means1[m] - means0[m]) for m in names),
                table={m: dict(default=means0[m], default_ci=boot_ci(X0[k]),
                               default_rank=(rs0[m]["rank_lo"], rs0[m]["rank_hi"]),
                               many=means1[m], many_ci=boot_ci(X1[k]),
                               many_rank=(rs1[m]["rank_lo"], rs1[m]["rank_hi"]),
                               p_first_many=rs1[m]["p_first"])
                       for k, m in enumerate(names)},
                pairs_many=p1)
        res[c] = rc
    Path(a.out).write_text(json.dumps(res, indent=2))
    for c, rc in res.items():
        print(f"\n######## {c}")
        for v, r in rc.items():
            print(f"== {v}: resolved {r['resolved_default']} -> {r['resolved_many']} /{r['n_pairs']} "
                  f"(x{r['ratio']:.2f}); Holm {r['holm_default']} -> {r['holm_many']}; sd "
                  f"{r['median_sd_default']:.3f} -> {r['median_sd_many']:.3f} (shrink {r['sd_shrink']:.2f}x); "
                  f"ICC(30) {r['icc_many']:.3f}; NaN {r['nan_frac_many']:.3f}; slates for d=0.02 "
                  f"{r['req_slates_d02_default']} -> {r['req_slates_d02_many']}; max |mean shift| "
                  f"{r['max_abs_mean_shift']:.3f}")
            if v == "lyapunov":
                for m in sorted(r["table"], key=lambda m: -r["table"][m]["many"]):
                    t = r["table"][m]
                    print(f"     {m:38s} 3-goal {t['default']:.3f} [{t['default_ci'][0]:.3f},{t['default_ci'][1]:.3f}] "
                          f"rank {t['default_rank'][0]}-{t['default_rank'][1]:<3d} | 30-goal {t['many']:.3f} "
                          f"[{t['many_ci'][0]:.3f},{t['many_ci'][1]:.3f}] rank {t['many_rank'][0]}-{t['many_rank'][1]}"
                          f"  P1={t['p_first_many']:.2f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
