"""A1: everything about ONE action pool (one same-state slate) -- a figure
plus a text block, so a reader can see what was chosen and not just its
score.

Panels
------
1. Histogram of `dv_true` across the pool's candidates, with the oracle's
   best, each model's pick (the TRUE dV of the action the model actually
   chose, i.e. dv_true[argmin(dv_pred)]), and the pool mean (what a random
   pick gets in expectation) marked.
2. The actions themselves, drawn in the workspace: the oracle's choice and
   each model's choice overlaid on the step-0 occupancy image -- start point,
   heading (the start->end segment), and the swept rectangle.
3. |R_k| vs K for each model (docs/experiments/METRICS.md's M_k/P_k, RAW
   Lyapunov units via pool_common.rk_curve). dV is a cost (lower better), so
   R_k = M_k - P_k <= 0 by construction; the panel plots |R_k|, the value
   left on the table at pool size K, and says so on the axis.
4. Ordering comparison: the model's rank vs the true rank for every
   candidate (one color per model, y=x is perfect), plus a bar chart of the
   true-value percentile of each model's top-1/2/3 picks (0 = truly best).

Usage
-----
    PYTHONPATH=. python scripts/probes/pool_inspect.py \\
        runs_exp0026/dv_cache_sharp.pt --goal corner --slate 7 \\
        --models persistence,mean-delta,linear,UNet \\
        --out reports/figs/A_corner_slate7.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

from scripts.probes.pool_common import (
    MODEL_COLORS, action_geometry, load_cache, load_occ0_for_slate, rk_curve,
    slate_rows, swept_rectangle_corners, workspace_bounds,
)

DEFAULT_KS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def true_percentile(t_sorted_by_model_rank, r, n):
    """Fraction of the n candidates strictly better (lower dv_true) than the
    model's rank-r pick -- 0 = truly best, matches exp0026_kcurve_exact's
    rank_profile convention."""
    val = t_sorted_by_model_rank[r - 1]
    return float((t_sorted_by_model_rank < val).sum()) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache")
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--slate", type=int, required=True)
    ap.add_argument("--models", default="persistence,mean-delta,linear,UNet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = load_cache(args.cache)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sel, g = slate_rows(cache, args.goal, args.slate)
    n = sel.numel()
    if n == 0:
        raise SystemExit(f"slate {args.slate} not found in goal {args.goal!r}")
    dv_true = g["dv_true"][sel]
    actions = cache["actions"][sel]
    ks = [k for k in DEFAULT_KS if k <= n]
    if ks[-1] != n:
        ks.append(n)

    preds = {m: (dv_true.clone() if m == "oracle" else g[m][sel]) for m in models}
    i_oracle = int(torch.argmin(dv_true))
    i_model = {m: int(torch.argmin(p)) for m, p in preds.items()}
    pool_mean = float(dv_true.mean())
    spread = float(dv_true.max() - dv_true.min())

    print(f"=== pool_inspect: {args.cache} goal={args.goal} slate={args.slate} "
          f"n_candidates={n} ===")
    print(f"provenance={cache['provenance']['commit']} dirty={cache['provenance']['dirty']}")
    print(f"dv_true: mean(pool)={pool_mean:+.5f}  min(oracle)={float(dv_true.min()):+.5f}  "
          f"max={float(dv_true.max()):+.5f}  spread(max-min)={spread:.5f}")
    for m in models:
        i = i_model[m]
        print(f"  {m:14s} picks candidate {i:4d}  realised dv_true={float(dv_true[i]):+.5f}  "
              f"predicted dv={float(preds[m][i]):+.5f}  "
              f"regret vs oracle={float(dv_true[i] - dv_true[i_oracle]):+.5f}")

    # ---- occ0 + workspace geometry -----------------------------------------
    occ0 = load_occ0_for_slate(cache, args.slate)
    H, W = occ0.shape[-2:]
    ws_min, ws_max = workspace_bounds(cache)
    plate_px = 0.04 / 0.128 * W
    half_width_px = 0.5 * plate_px + 2.0

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.3, 0.9])

    # --- panel 1: histogram ---
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(dv_true.numpy(), bins=min(24, max(8, n // 3)), color="#cccccc",
            edgecolor="white")
    ax.axvline(pool_mean, color="black", linestyle=":", label=f"pool mean {pool_mean:+.4f}")
    ax.axvline(float(dv_true[i_oracle]), color=MODEL_COLORS.get("oracle", "green"),
               linestyle="-", linewidth=2, label=f"oracle best {float(dv_true[i_oracle]):+.4f}")
    for m in models:
        c = MODEL_COLORS.get(m, "black")
        v = float(dv_true[i_model[m]])
        ax.axvline(v, color=c, linestyle="--", linewidth=1.5, label=f"{m} pick {v:+.4f}")
    ax.set_xlabel("dv_true (cost; lower = better)")
    ax.set_ylabel("candidate count")
    ax.set_title(f"pool {args.slate}: realised dV distribution (n={n})")
    ax.legend(fontsize=7, loc="upper right")

    # --- panel 2: workspace ---
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(occ0.numpy(), cmap="Greys", origin="upper", vmin=0, vmax=1)
    to_draw = [("oracle", i_oracle)] + [(m, i_model[m]) for m in models if m != "oracle"]
    for name, i in to_draw:
        c = MODEL_COLORS.get(name, "black")
        s_px, e_px = action_geometry(actions[i], ws_min, ws_max, (H, W))
        corners = swept_rectangle_corners(s_px, e_px, half_width_px)
        ax.add_patch(Polygon(corners, closed=True, facecolor=c, alpha=0.18,
                              edgecolor=c, linewidth=1.5))
        ax.plot(float(s_px[0]), float(s_px[1]), "o", color=c, markersize=6)
        ax.annotate("", xy=(float(e_px[0]), float(e_px[1])),
                    xytext=(float(s_px[0]), float(s_px[1])),
                    arrowprops=dict(arrowstyle="->", color=c, linewidth=2))
    handles = [plt.Line2D([0], [0], color=MODEL_COLORS.get(n, "black"), lw=2, label=n)
               for n, _ in to_draw]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_title("step-0 occupancy: oracle vs model picks (start dot -> arrow, swept rect)")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)

    # --- panel 3: |R_k| curve ---
    ax = fig.add_subplot(gs[1, :])
    for m in models:
        Mk, Pk = rk_curve(preds[m], dv_true, ks)
        xs = sorted(Mk.keys())
        ys = [abs(Mk[k] - Pk[k]) for k in xs]
        ax.plot(xs, ys, marker="o", markersize=3, color=MODEL_COLORS.get(m, None), label=m)
    ax.set_xscale("log")
    ax.set_xlabel("K (pool size)")
    ax.set_ylabel("|R_K| = |M_K - P_K|  (Lyapunov units)")
    ax.set_title("value left on the table vs K  "
                 "(dV is a cost: R_K = M_K - P_K <= 0 by construction; |R_K| plotted)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- panel 4a: rank scatter ---
    ax = fig.add_subplot(gs[2, 0])
    true_rank = torch.argsort(torch.argsort(dv_true)).numpy() + 1
    for m in models:
        model_rank = torch.argsort(torch.argsort(preds[m])).numpy() + 1
        ax.scatter(model_rank, true_rank, s=14, alpha=0.6,
                   color=MODEL_COLORS.get(m, None), label=m)
    ax.plot([1, n], [1, n], color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("model rank (1 = model's top pick)")
    ax.set_ylabel("true rank (1 = oracle's top pick)")
    ax.set_title("ordering: model rank vs true rank, every candidate")
    ax.legend(fontsize=7)

    # --- panel 4b: top-1/2/3 true percentile bar chart ---
    ax = fig.add_subplot(gs[2, 1])
    width = 0.25
    xs = np.arange(3)
    for j, m in enumerate(models):
        order = torch.argsort(preds[m])
        t_by_rank = dv_true[order].numpy()
        pct = [true_percentile(t_by_rank, r, n) for r in (1, 2, 3)]
        ax.bar(xs + j * width, pct, width=width, color=MODEL_COLORS.get(m, None), label=m)
        print(f"  {m:14s} true percentile of top-1/2/3 picks: "
              f"{pct[0]:.3f} / {pct[1]:.3f} / {pct[2]:.3f}  (0=truly best)")
    ax.set_xticks(xs + width * (len(models) - 1) / 2)
    ax.set_xticklabels(["top-1", "top-2", "top-3"])
    ax.set_ylabel("true percentile (0 = truly best)")
    ax.set_title("true-value percentile of the model's rank-1/2/3 picks")
    ax.legend(fontsize=7)

    fig.suptitle(f"{args.cache}  goal={args.goal}  slate={args.slate}  n={n}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
