"""A2: population statistics over every slate in a dV cache/goal -- find the
pools worth looking at individually with `pool_inspect.py`, rather than
trusting an aggregate ranking metric to say everything.

Reports, per model:

  * agreement rate     -- fraction of slates where the model's top-1 pick IS
                           the oracle's top-1 pick (argmin dv_pred ==
                           argmin dv_true).
  * disagreement, split into "near-tie" (model and oracle pick DIFFERENT
    actions but the realised values are close) vs "genuine, large-loss"
    disagreement. "Close" is defined relative to the POOL'S OWN SPREAD
    (max - min of dv_true in that slate), not a fixed Lyapunov-unit cutoff,
    because the spread itself varies by orders of magnitude across cells
    (see the pool-spread distribution below and C-2's L10mm check). This
    script does not hard-code which fraction counts as "close": it reports
    the full distribution of loss/spread among disagreement slates, and the
    prevalence at several candidate thresholds (2%, 5%, 10%, 20%, 30%), so
    the report can point at where the distribution actually breaks rather
    than asserting a number.
  * the pool-spread distribution itself (max-min and mean-min of dv_true per
    slate) -- every normalised capture number in this project divides by one
    of these, so this is the number that says whether a slate can
    discriminate models at all.
  * slates ranked by |R_K| (docs/experiments/METRICS.md's M_k/P_k, raw
    Lyapunov units, via pool_common.rk_curve) at a chosen K, so the worst and
    most-typical pools can be handed to pool_inspect.py by slate id.

Usage
-----
    PYTHONPATH=. python scripts/probes/pool_survey.py \\
        runs_exp0026/dv_cache_sharp.pt --goal corner \\
        --models persistence,mean-delta,linear,UNet \\
        --k-rank 4 --out runs_exp0026/survey_corner.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from scripts.probes.pool_common import admissible_slates, load_cache, rk_curve

THRESHOLDS = [0.02, 0.05, 0.10, 0.20, 0.30]


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    qs = [0, 10, 25, 50, 75, 90, 100]
    out = {f"p{q}": float(np.percentile(x, q)) for q in qs}
    out["mean"] = float(x.mean())
    out["sd"] = float(x.std(ddof=1)) if x.size > 1 else 0.0
    out["sem"] = out["sd"] / np.sqrt(x.size) if x.size > 1 else float("nan")
    out["n"] = int(x.size)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache")
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--models", default="persistence,mean-delta,linear,UNet")
    ap.add_argument("--min-slate", type=int, default=8)
    ap.add_argument("--k-rank", type=int, default=4,
                     help="K at which slates are ranked by |R_K| for handoff "
                          "to pool_inspect.py")
    ap.add_argument("--top-n", type=int, default=8,
                     help="how many worst-|R_K| slate ids to print/save")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache = load_cache(args.cache)
    goal = args.goal
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ep = cache["ep"]
    dv = cache["dv"][goal]
    slates = admissible_slates(cache, goal, args.min_slate)
    print(f"cache={args.cache} goal={goal} n_slates={len(slates)} "
          f"provenance={cache['provenance']['commit']} "
          f"dirty={cache['provenance']['dirty']}")

    # ---- pool-spread distribution (shared across models) -------------------
    max_min, mean_min = [], []
    for e in slates:
        sel = (ep == e).nonzero(as_tuple=True)[0]
        t = dv["dv_true"][sel]
        max_min.append(float(t.max() - t.min()))
        mean_min.append(float(t.mean() - t.min()))
    q_maxmin, q_meanmin = quantiles(max_min), quantiles(mean_min)
    print("\n=== pool spread of dv_true (Lyapunov units), n=%d slates ===" % len(slates))
    print(f"  max-min:  mean={q_maxmin['mean']:.5f} median={q_maxmin['p50']:.5f} "
          f"p10={q_maxmin['p10']:.5f} p90={q_maxmin['p90']:.5f}")
    print(f"  mean-min: mean={q_meanmin['mean']:.5f} median={q_meanmin['p50']:.5f} "
          f"p10={q_meanmin['p10']:.5f} p90={q_meanmin['p90']:.5f}")

    out = {"cache": args.cache, "goal": goal, "n_slates": len(slates),
           "provenance": cache["provenance"],
           "pool_spread": {"max_min": q_maxmin, "mean_min": q_meanmin},
           "k_rank": args.k_rank, "per_model": {}}

    for m in models:
        pred = dv["dv_true"].clone() if m == "oracle" else dv[m]
        agree, loss, ratio_all = [], [], []
        rk_abs_by_slate = {}
        for e in slates:
            sel = (ep == e).nonzero(as_tuple=True)[0]
            p, t = pred[sel], dv["dv_true"][sel]
            i_model = int(torch.argmin(p))
            i_oracle = int(torch.argmin(t))
            a = (i_model == i_oracle)
            spread = float(t.max() - t.min())
            l = float(t[i_model] - t[i_oracle])  # >= 0, Lyapunov units
            agree.append(a)
            loss.append(l)
            ratio_all.append(l / spread if spread > 1e-12 else 0.0)
            Mk, Pk = rk_curve(p, t, [args.k_rank])
            if args.k_rank in Mk:
                rk_abs_by_slate[e] = abs(Mk[args.k_rank] - Pk[args.k_rank])

        agree = np.array(agree)
        loss = np.array(loss)
        ratio = np.array(ratio_all)
        disagree_ratio = ratio[~agree]

        prevalence = {"exact_agreement": float(agree.mean())}
        for thr in THRESHOLDS:
            near_tie = (~agree) & (ratio <= thr)
            large_loss = (~agree) & (ratio > thr)
            prevalence[f"near_tie@{thr}"] = float(near_tie.mean())
            prevalence[f"large_loss@{thr}"] = float(large_loss.mean())

        ranked = sorted(rk_abs_by_slate.items(), key=lambda kv: kv[1], reverse=True)
        worst = ranked[:args.top_n]
        med_val = float(np.median(list(rk_abs_by_slate.values()))) if ranked else float("nan")
        typical = min(rk_abs_by_slate.items(), key=lambda kv: abs(kv[1] - med_val)) \
            if ranked else (None, None)
        # a disagreement slate with small value loss, for the "near-tie" figure
        near_tie_candidates = [(int(e), float(r), float(l)) for e, r, l, a in
                                zip(slates, ratio, loss, agree) if not a]
        near_tie_candidates.sort(key=lambda x: x[1])  # smallest ratio first
        near_tie_example = near_tie_candidates[0] if near_tie_candidates else None

        print(f"\n--- model: {m} ---")
        print(f"  agreement rate: {prevalence['exact_agreement']:.3f} "
              f"({int(agree.sum())}/{len(slates)} slates)")
        print(f"  loss|top1 (Lyapunov units): mean={loss.mean():.5f} "
              f"sem={loss.std(ddof=1)/np.sqrt(len(loss)) if len(loss)>1 else float('nan'):.5f} "
              f"median={np.median(loss):.5f} max={loss.max():.5f}")
        if disagree_ratio.size:
            dq = quantiles(disagree_ratio)
            print(f"  disagreement-slate ratio (loss/spread) distribution: "
                  f"median={dq['p50']:.3f} p25={dq['p25']:.3f} p75={dq['p75']:.3f} "
                  f"(n={dq['n']})")
        for thr in THRESHOLDS:
            print(f"    thr={thr:.2f}: near-tie={prevalence[f'near_tie@{thr}']:.3f} "
                  f"large-loss={prevalence[f'large_loss@{thr}']:.3f}")
        print(f"  |R_{args.k_rank}| ranked: worst slate={worst[0] if worst else None}, "
              f"typical slate={typical}")
        if near_tie_example:
            print(f"  smallest-ratio disagreement slate (candidate for the "
                  f"'near-tie' figure): {near_tie_example}")

        out["per_model"][m] = {
            "agreement_rate": prevalence["exact_agreement"],
            "loss_quantiles": quantiles(loss),
            "disagreement_ratio_quantiles": quantiles(disagree_ratio) if disagree_ratio.size else None,
            "prevalence": prevalence,
            "worst_slates_by_abs_Rk": worst,
            "typical_slate_by_abs_Rk": typical,
            "near_tie_example_slate": near_tie_example,
        }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
