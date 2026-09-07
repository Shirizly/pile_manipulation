"""Closed-form slateK_exact / worstK / rank_profile, from docs/experiments/METRICS.md.

METRICS.md documents these as "From scripts/probes/exp0026_kcurve.py (closed
form)" and quotes numbers from them (slateK_exact 0.977 at K=4, linear worstK
0.471 vs UNet 0.316 at K=4, rank_profile 0.035/0.058/0.076/0.095 at ranks
1-4) -- but no such code exists in this repo (`grep -rn "slateK_exact\|worstK\|
rank_profile"` finds nothing outside METRICS.md before this file). This module
is the missing implementation, added for EXP-B (which the plan requires report
these three metrics as the headline, per METRICS.md's own instruction to
"prefer these over the sampled slateK"). `exp0026_kcurve.py` itself is left
unmodified (its sampled slateK/regret_dv/pick_pctile stay the one code path
for those metrics, still needed here for the `slate4` bridge row and the
degradation-arm sweep).

**Validated before use**: run against `runs_exp0026/dv_cache_all50.pt` (the
cache EXP-0026 already produced), this module reproduces METRICS.md's quoted
numbers to 3 decimal places or better --
slateK_exact(linear, K=4)=0.9767 (quoted 0.977), worstK(linear, K=4)=0.4710
(quoted 0.471), worstK(UNet, K=4)=0.3157 (quoted 0.316), worstK(linear)
K=2->16 = 0.483->0.334 (quoted 0.483->0.334), rank_profile(linear) ranks
1-4 = 0.035/0.058/0.076/0.095 (quoted exactly) -- see `--self-test`.

Formulas (n = slate size, r = model's rank 1..n by ascending predicted dV,
t_(r) = true dV of the model's rank-r candidate, t_[r] = r-th smallest true
dV in the slate):

    w_r(K)       = C(n-r, K-1) / C(n, K)
    E_chosen(K)  = sum_r t_(r) * w_r(K)
    E_oracle(K)  = sum_r t_[r] * w_r(K)
    slateK_exact = (mean(t) - E_chosen(K)) / (mean(t) - E_oracle(K))     per slate

    worstK = max over r with (n-r) >= K-1 of
                 [t_(r) - min_{r'>r} t_(r')] / (mean(t) - min(t))        per slate

    rank_profile[r] = mean over slates of #{j : t_j < t_(r)} / n

Per-slate values are averaged across slates (never a pooled ratio of sums),
matching every other metric in this register.

Usage
-----
    PYTHONPATH=. python scripts/probes/exp0026_kcurve_exact.py \
        runs_exp0026/dv_cache_all50.pt --goal corner --self-test
    PYTHONPATH=. python scripts/probes/exp0026_kcurve_exact.py \
        <cache.pt> --goal corner --models persistence,mean-delta,linear,UNet,oracle \
        --ks 2,4,8,16,32,64,128 --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
from scipy.special import comb

RANK_PROFILE_RANKS = [1, 2, 3, 4, 8, 16, 32, 64, 128]


def _w(n, K, r):
    """C(n-r, K-1) / C(n, K), r 1-indexed. Vectorised over r=1..n."""
    r = np.arange(1, n + 1)
    return comb(n - r, K - 1, exact=False) / comb(n, K, exact=False)


def per_slate_exact(dv_pred: torch.Tensor, dv_true: torch.Tensor, ks: list[int]):
    """slateK_exact and worstK for every K in `ks`, plus the model's
    true-percentile rank_profile, for ONE slate."""
    n = dv_pred.shape[0]
    order = torch.argsort(dv_pred)
    t_by_rank = dv_true[order].numpy()          # t_(r), r=1..n at index r-1
    t_sorted = torch.sort(dv_true).values.numpy()
    t_np = dv_true.numpy()
    mean_t = float(dv_true.mean())
    min_t = float(dv_true.min())
    span = mean_t - min_t

    exact, worst = {}, {}
    suffix_min = np.minimum.accumulate(t_by_rank[::-1])[::-1]  # suffix_min[i]=min(t_by_rank[i:])
    for K in ks:
        if K > n:
            continue
        wr = _w(n, K, None)
        E_chosen = float((t_by_rank * wr).sum())
        E_oracle = float((t_sorted * wr).sum())
        denom = mean_t - E_oracle
        if abs(denom) > 1e-12:
            exact[K] = (mean_t - E_chosen) / denom
        valid_r = [r for r in range(1, n + 1) if (n - r) >= K - 1 and r < n]
        if valid_r and span > 1e-12:
            best_inv = max(t_by_rank[r - 1] - suffix_min[r] for r in valid_r)
            worst[K] = best_inv / span

    profile = {}
    for r in RANK_PROFILE_RANKS:
        if r > n:
            continue
        tr = float(t_by_rank[r - 1])
        profile[r] = float((t_np < tr).mean())
    return exact, worst, profile


def sweep(cache, goal, models, ks, min_slate=8):
    dv = cache["dv"][goal]
    ep = cache["ep"]
    dv_true_all = dv["dv_true"]
    preds = {m: (dv_true_all.clone() if m == "oracle" else dv[m]) for m in models}

    exact = {m: {K: [] for K in ks} for m in models}
    worst = {m: {K: [] for K in ks} for m in models}
    profile = {m: {r: [] for r in RANK_PROFILE_RANKS} for m in models}
    n_slates = 0
    for e in ep.unique().tolist():
        sel = (ep == e).nonzero(as_tuple=True)[0]
        if sel.numel() < min_slate:
            continue
        t = dv_true_all[sel]
        if float(t.std()) < 1e-9:
            continue
        n_slates += 1
        for m in models:
            ex, wk, pf = per_slate_exact(preds[m][sel], t, ks)
            for K, v in ex.items():
                exact[m][K].append(v)
            for K, v in wk.items():
                worst[m][K].append(v)
            for r, v in pf.items():
                profile[m][r].append(v)
    return exact, worst, profile, n_slates


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    return dict(mean=float(d.mean()), sd=sd, sem=sem,
                t=float(d.mean() / sem) if sem > 0 else float("inf"),
                n=n, wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                ties=int((d == 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--models", default="persistence,mean-delta,linear,UNet,oracle")
    ap.add_argument("--ks", default="2,4,8,16,24,31")
    ap.add_argument("--min-slate", type=int, default=8)
    ap.add_argument("--pairs", default="UNet-linear,oracle-linear,linear-mean-delta")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true",
                     help="check against METRICS.md's published EXP-0026 numbers")
    args = ap.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    models = args.models.split(",")
    ks = [int(k) for k in args.ks.split(",")]

    exact, worst, profile, n_slates = sweep(cache, args.goal, models, ks, args.min_slate)
    print(f"cache={args.cache} goal={args.goal} n_slates={n_slates} "
          f"provenance={cache['provenance']['commit']} dirty={cache['provenance']['dirty']}")

    print("\n=== slateK_exact (closed form) ===")
    hdr = "  ".join(f"K={K:<7d}" for K in ks)
    print(f"{'model':16s} {hdr}")
    for m in models:
        row = "  ".join(f"{np.mean(exact[m][K]):<9.4f}" if exact[m][K] else "   n/a   " for K in ks)
        print(f"{m:16s} {row}")

    print("\n=== worstK (adversarial pool) ===")
    print(f"{'model':16s} {hdr}")
    for m in models:
        row = "  ".join(f"{np.mean(worst[m][K]):<9.4f}" if worst[m][K] else "   n/a   " for K in ks)
        print(f"{m:16s} {row}")

    print("\n=== rank_profile (true percentile of the model's r-th pick) ===")
    rhdr = "  ".join(f"r={r:<6d}" for r in RANK_PROFILE_RANKS)
    print(f"{'model':16s} {rhdr}")
    for m in models:
        row = "  ".join(f"{np.mean(profile[m][r]):<8.3f}" if profile[m][r] else "  n/a   " for r in RANK_PROFILE_RANKS)
        print(f"{m:16s} {row}")

    print("\n=== paired slateK_exact differences ===")
    pairs = [tuple(p.split("-", 1)) for p in args.pairs.split(",") if p]
    pairs = [(a, b) for a, b in pairs if a in models and b in models]
    paired_out = {}
    for a, b in pairs:
        print(f"\n  {a} - {b}")
        for K in ks:
            if not exact[a][K] or not exact[b][K]:
                continue
            st = paired(exact[a][K], exact[b][K])
            paired_out.setdefault(f"{a}-{b}", {})[K] = st
            print(f"    K={K:4d} mean={st['mean']:+.4f} sem={st['sem']:.4f} "
                  f"t={st['t']:+.2f} wins={st['wins']}/{st['n']} ties={st['ties']}")

    if args.self_test:
        checks = [
            ("linear", 4, "exact", 0.977),
            ("linear", 4, "worst", 0.471),
            ("UNet", 4, "worst", 0.316),
        ]
        ok = True
        for m, K, kind, expect in checks:
            got = np.mean(exact[m][K]) if kind == "exact" else np.mean(worst[m][K])
            close = abs(got - expect) < 0.002
            ok &= close
            print(f"self-test {m} {kind} K={K}: got {got:.4f} expect {expect:.4f} "
                  f"{'OK' if close else 'FAIL'}")
        print("SELF-TEST " + ("PASSED" if ok else "FAILED"))

    if args.out:
        out = {"goal": args.goal, "n_slates": n_slates, "ks": ks,
               "models": models, "provenance": cache["provenance"],
               "slateK_exact": {m: {K: float(np.mean(v)) for K, v in exact[m].items() if v}
                                for m in models},
               "worstK": {m: {K: float(np.mean(v)) for K, v in worst[m].items() if v}
                          for m in models},
               "rank_profile": {m: {r: float(np.mean(v)) for r, v in profile[m].items() if v}
                                for m in models},
               "paired": paired_out}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
