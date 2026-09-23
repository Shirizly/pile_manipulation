"""EXP-0023 stage 3: metrics.

SIGN CONVENTION -- read this before reading any number.
`dv = value(after) - value(before)` is a COST (lower/more negative is better),
fixed once in `simple_mpc.adapters` and asserted by `assert_dv_convention`.
DESIGN.md defines `gradient_gain = dv_grad - dv_rank` and glosses "negative =
optimising made things WORSE", which is only true if dv were a REWARD. Under
this repo's cost convention that formula has the opposite polarity to its own
gloss, so the three difference metrics below are reported NEGATED relative to
DESIGN.md's literal formulae, preserving the design's MEANING:

    gradient_gain = dv_rank    - dv_grad     > 0  optimising helped
    pool_escape   = pool_ceil  - dv_grad     > 0  beat everything in the pool
    regret        = dv_grad    - dv_oracle   > 0  headroom left to the oracle

`capture_vs_oracle = dv_grad / dv_oracle` is left exactly as DESIGN.md writes
it: both are costs measured from the same v0, so the ratio already reads as
"fraction of the oracle's improvement captured" with no sign change.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

CACHE = "experiments/temp/binned-pools/dv_cache_corner.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s1 = torch.load(args.stage1, map_location="cpu", weights_only=False)
    s2 = torch.load(args.stage2, map_location="cpu", weights_only=False)
    cache = torch.load(REPO / CACHE, map_location="cpu", weights_only=False)
    dv_cached = cache["dv"]["corner"]["dv_true"]

    arms = s2["arms"]
    slates = sorted(s2["dv"].keys())

    # ── known-number reproduction: re-executed pool rows vs the cache ──────
    val = []
    for s in slates:
        rec = s2["dv"][s]
        for (kind, tag), d in zip(rec["labels"], rec["dv_all"]):
            if kind == "poolval":
                val.append((s, int(tag), float(d), float(dv_cached[int(tag)])))
    v = np.array([(a[2], a[3]) for a in val])
    rep = {
        "n": len(val),
        "genesis_mean": float(v[:, 0].mean()), "cache_mean": float(v[:, 1].mean()),
        "pearson_r": float(np.corrcoef(v[:, 0], v[:, 1])[0, 1]),
        "mae": float(np.abs(v[:, 0] - v[:, 1]).mean()),
        "rms": float(np.sqrt(((v[:, 0] - v[:, 1]) ** 2).mean())),
        "cache_sd": float(v[:, 1].std()),
    }

    # ── per-(state, arm) metrics ──────────────────────────────────────────
    rows = []
    for s in slates:
        rec = s2["dv"][s]
        lab, dv = rec["labels"], rec["dv_all"]
        dv_oracle = float(dv[[i for i, l in enumerate(lab) if l[0] == "oracle"][0]])
        pool_true = dv_cached[s1["pool_rows"][s]]
        pool_ceil = float(pool_true.min())          # cost -> best = min
        for a in arms:
            i_r = [i for i, l in enumerate(lab) if l == ("rank", a)][0]
            i_g = [i for i, l in enumerate(lab) if l == ("grad", a)][0]
            dv_rank, dv_grad = float(dv[i_r]), float(dv[i_g])
            hits = s1["arms"][a]["bound_hit_frac"][s]
            rows.append(dict(
                slate=s, arm=a, dv_rank=dv_rank, dv_grad=dv_grad,
                dv_oracle=dv_oracle, pool_ceiling=pool_ceil,
                gradient_gain=dv_rank - dv_grad,
                pool_escape=pool_ceil - dv_grad,
                regret_vs_oracle=dv_grad - dv_oracle,
                capture_vs_oracle=(dv_grad / dv_oracle) if abs(dv_oracle) > 1e-4 else float("nan"),
                dv_pred_rank=s1["arms"][a]["dv_pred_rank"][s],
                dv_pred_grad=s1["arms"][a]["dv_pred_grad"][s],
                bound_hit_box=float(hits[0]), bound_hit_len=float(hits[1]),
            ))

    # ── floors, from the TRUE dv of the shared seed pool ──────────────────
    floors = {}
    for s in slates:
        pt = dv_cached[s1["pool_rows"][s]]
        floors[s] = {"random_mean": float(pt.mean()), "random_sd": float(pt.std()),
                     "pool_ceiling": float(pt.min()), "pool_worst": float(pt.max()),
                     "frac_improving": float((pt < 0).float().mean())}

    # ── power check: between-arm vs between-state variation ───────────────
    G = np.array([[r["gradient_gain"] for r in rows if r["arm"] == a] for a in arms])  # (A,S)
    arm_means, state_means = G.mean(axis=1), G.mean(axis=0)
    power = {
        "between_arm_sd_of_means": float(arm_means.std(ddof=1)),
        "between_state_sd_of_means": float(state_means.std(ddof=1)),
        "residual_sd": float((G - arm_means[:, None] - state_means[None, :] + G.mean()).std(ddof=1)),
        "verdict": None,
    }
    power["verdict"] = ("between-arm > between-state: the design can separate arms"
                        if power["between_arm_sd_of_means"] > power["between_state_sd_of_means"]
                        else "between-arm < between-state: DESIGN LACKS POWER to answer its "
                             "own question at this n; per-state values dominate arm identity")

    summary = {}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        def col(k): return np.array([r[k] for r in rs], dtype=float)
        gg = col("gradient_gain")
        summary[a] = {
            "dv_rank_mean": float(col("dv_rank").mean()),
            "dv_grad_mean": float(col("dv_grad").mean()),
            "dv_oracle_mean": float(col("dv_oracle").mean()),
            "gradient_gain_mean": float(gg.mean()),
            "gradient_gain_sd": float(gg.std(ddof=1)),
            "gradient_gain_min": float(gg.min()), "gradient_gain_max": float(gg.max()),
            "gradient_gain_n_negative": int((gg < 0).sum()),
            "gradient_gain_sem": float(gg.std(ddof=1) / np.sqrt(len(gg))),
            "pool_escape_mean": float(col("pool_escape").mean()),
            "pool_escape_n_positive": int((col("pool_escape") > 0).sum()),
            "capture_vs_oracle_mean": float(np.nanmean(col("capture_vs_oracle"))),
            "regret_vs_oracle_mean": float(col("regret_vs_oracle").mean()),
            "bound_hit_box_mean": float(col("bound_hit_box").mean()),
            "bound_hit_len_mean": float(col("bound_hit_len").mean()),
            "n_states_any_bound": int(((col("bound_hit_box") + col("bound_hit_len")) > 0).sum()),
        }

    out = {"floors": floors, "rows": rows, "summary": summary, "power": power,
           "cache_reproduction": rep, "arms": arms, "slates": slates,
           "sign_convention": "dv = value(after)-value(before), a COST; "
                              "gradient_gain/pool_escape/regret negated vs DESIGN.md "
                              "literal formulae to preserve its 'positive = better' gloss"}
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("== cache reproduction (Genesis re-execution vs binned_pool_cache dv_true) ==")
    print(json.dumps(rep, indent=2))
    print("\n== per-arm ==")
    hdr = f"{'arm':24s} {'dv_rank':>9s} {'dv_grad':>9s} {'dv_orac':>9s} {'grad_gain':>10s} {'sd':>8s} {'neg':>4s} {'escape':>9s} {'esc+':>5s} {'capt':>7s} {'bound':>6s}"
    print(hdr)
    for a, r in summary.items():
        print(f"{a:24s} {r['dv_rank_mean']:+9.5f} {r['dv_grad_mean']:+9.5f} "
              f"{r['dv_oracle_mean']:+9.5f} {r['gradient_gain_mean']:+10.5f} "
              f"{r['gradient_gain_sd']:8.5f} {r['gradient_gain_n_negative']:4d} "
              f"{r['pool_escape_mean']:+9.5f} {r['pool_escape_n_positive']:5d} "
              f"{r['capture_vs_oracle_mean']:7.3f} {r['n_states_any_bound']:6d}")
    print("\n== floors (true dv over the shared 100-action seed pool) ==")
    fr = np.array([[floors[s][k] for k in ("random_mean","pool_ceiling","pool_worst")] for s in slates])
    print(f"  random (pool mean) {fr[:,0].mean():+.5f}   pool_ceiling {fr[:,1].mean():+.5f}   "
          f"pool_worst {fr[:,2].mean():+.5f}")
    print("\n== power ==");  print(json.dumps(power, indent=2))
    print("\n== per-state gradient_gain ==")
    print("arm".ljust(24) + "".join(f"{s:>9d}" for s in slates))
    for i, a in enumerate(arms):
        print(a.ljust(24) + "".join(f"{g:+9.4f}" for g in G[i]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
