"""EXP-0026 A3: pool size vs number of states, on DS-0001's cached pools.

DS-0001 (n20_scatter_s20a1000_L20-70mm): 20 slates x 1000 simulated candidates,
true lyapunov(corner) dv plus cached predictions from nfd (MODEL-0003),
visual-switched (MODEL-0001), descriptor (MODEL-0002)
(experiments/temp/binned-pools/dv_cache_corner.pt, binned_pool_cache.py).

For K in {25, 50, 100, 250, 500} draw R random size-K subpools per slate
(without replacement within a subpool) and score slateN capture per subpool.
For each model pair, decompose the per-(slate, subpool) paired difference into
  s2_slate : variance of the slate-level mean difference (the K->inf limit is
             the K=1000 full-pool value)
  s2_pool  : within-slate variance across subpools (pool-sampling noise).
A Genesis budget of B simulated actions spent as S states x K candidates then
has Var(mean d) ~= s2_slate/S + s2_pool(K)/S (one pool per state) -- this
tells whether a fixed simulation budget is better spent on more states or on
bigger pools. Also reports the top-1 regret (true best - true[chosen]) in dv
units, which does not change scale with K the way slateN's normaliser does.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Baselines.common.goals import slate_n_capture, higher_is_better_for

CACHE = REPO / "experiments/temp/binned-pools/dv_cache_corner.pt"
MODELS = ("nfd", "visual-switched", "descriptor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    c = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert c["config"]["value_fn"] == "lyapunov"
    hib = higher_is_better_for("lyapunov")
    dv = c["dv"]["corner"]
    ep = c["ep"].numpy()
    slates = np.unique(ep)
    rng = np.random.default_rng(args.seed)
    Ks = [25, 50, 100, 250, 500, 1000]

    # cap[K][model] -> (S, R) ; reg likewise (top-1 regret in dv units, lower better)
    cap = {K: {m: np.full((len(slates), args.reps), np.nan) for m in MODELS} for K in Ks}
    reg = {K: {m: np.full((len(slates), args.reps), np.nan) for m in MODELS} for K in Ks}
    for si, s in enumerate(slates):
        rows = np.nonzero(ep == s)[0]
        vt_all = dv["dv_true"][rows]
        for K in Ks:
            R = 1 if K >= len(rows) else args.reps
            for r in range(R):
                sub = torch.from_numpy(rng.choice(len(rows), size=min(K, len(rows)), replace=False))
                vt = vt_all[sub]
                for m in MODELS:
                    vp = dv[m][rows][sub]
                    cap[K][m][si, r] = slate_n_capture(vp, vt, higher_is_better=hib)
                    pick = int(torch.argmin(vp)) if not hib else int(torch.argmax(vp))
                    reg[K][m][si, r] = float(vt[pick] - vt.min()) if not hib else float(vt.max() - vt[pick])
    out = {"cache": str(CACHE.relative_to(REPO)), "n_slates": int(len(slates)),
           "reps": args.reps, "per_K": {}}
    for K in Ks:
        rowK = {"mean_capture": {m: float(np.nanmean(cap[K][m])) for m in MODELS},
                "mean_regret": {m: float(np.nanmean(reg[K][m])) for m in MODELS}, "pairs": {}}
        for a, b in itertools.combinations(MODELS, 2):
            for name, arr, sign in (("capture", cap, 1.0), ("regret", reg, -1.0)):
                D = sign * (arr[K][a] - arr[K][b])            # (S, R), higher = a better
                if K == 1000:
                    D = D[:, :1]
                slate_mean = np.nanmean(D, axis=1)
                s2_pool = float(np.nanmean(np.nanvar(D, axis=1, ddof=1))) if D.shape[1] > 1 else 0.0
                # sd of the paired difference for ONE pool per state -- what a
                # design with S states x one K-pool actually sees
                one_pool_sd = float(np.nanstd(D[:, 0], ddof=1))
                rowK["pairs"][f"{name}:{a}-{b}"] = dict(
                    mean=float(np.nanmean(D)), s2_slate_mean=float(np.nanvar(slate_mean, ddof=1)),
                    s2_pool=s2_pool, sd_one_pool_per_state=one_pool_sd)
        out["per_K"][str(K)] = rowK
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"slates={len(slates)} reps={args.reps}")
    print("mean capture:")
    for K in Ks:
        print(f"  K={K:5d} " + "  ".join(f"{m}={out['per_K'][str(K)]['mean_capture'][m]:+.3f}" for m in MODELS)
              + "   regret: " + "  ".join(f"{m}={out['per_K'][str(K)]['mean_regret'][m]:.4f}" for m in MODELS))
    for name in ("capture", "regret"):
        print(f"\n{name}: paired difference variance split (slate-level var | within-slate pool var)")
        for p in [k for k in out["per_K"]["100"]["pairs"] if k.startswith(name)]:
            print("  " + p)
            for K in Ks:
                q = out["per_K"][str(K)]["pairs"][p]
                print(f"     K={K:5d} mean {q['mean']:+.4f}  s2_slate(mean over pools) {q['s2_slate_mean']:.2e}"
                      f"  s2_pool {q['s2_pool']:.2e}  sd(one pool/state) {q['sd_one_pool_per_state']:.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
