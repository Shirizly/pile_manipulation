"""EXP-0026 stage 2: the selection-pressure curve — slateK for K = 2..n.

Reads the dV cache written by `exp0026_selection_pressure.py` and sweeps the
candidate-slate size K, which is the one axis every control number in this
register has held fixed at 4.

Three things this does differently from `control_utility_test.rank_metrics`:

1. **Subsets are drawn without replacement.** `rank_metrics` uses
   `torch.randint`, so its K=16 slate is 16 draws *with* replacement from ~32
   candidates -- roughly 13 distinct actions, and at K=n it is not the full
   slate at all. Under-replacement is the whole point here, so the sampler is
   `torch.randperm`-based and K=n_s is the exact top-1-of-everything test.
2. **Every model sees the identical subsets** (one generator per slate,
   reset per model), so the model comparison is paired at the level of the
   individual draw, not just the slate.
3. **Two unbounded companions are reported alongside the bounded capture
   fraction**, because a bounded score's *difference* is forced toward zero
   and `slateK`'s own denominator grows with K:
     * `regret_dv`   -- chosen minus best, in Lyapunov units (lower better);
     * `pick_pctile` -- fraction of the subset strictly better than the pick
                        (0 = picked the best available, lower better).

Usage
-----
    PYTHONPATH=. python scripts/probes/exp0026_kcurve.py \
        runs_exp0026/dv_cache_all50.pt --goal corner --out runs_exp0026/kcurve.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

BASE_MODELS = ["mean-delta", "linear", "UNet", "oracle", "persistence"]


def _subsets(n, K, n_draw, gen):
    """(n_draw, K) index matrix, each row K DISTINCT candidates of n."""
    if K >= n:
        return torch.arange(n).unsqueeze(0)
    return torch.stack([torch.randperm(n, generator=gen)[:K] for _ in range(n_draw)])


def slate_metrics(dv_pred, dv_true, idx):
    """capture / regret / percentile for one model on one subset matrix.

    capture = (rand - chosen) / (rand - oracle), averaged over draws -- the
    mean of per-draw ratios, exactly `rank_metrics`' own convention for
    slate4, so K=4 here is directly comparable with every slate4 in the
    register.
    """
    ct, cp = dv_true[idx], dv_pred[idx]
    chosen = ct.gather(1, cp.argmin(dim=1, keepdim=True)).squeeze(1)
    oracle = ct.min(dim=1).values
    rand = ct.mean(dim=1)
    denom = rand - oracle
    ok = denom > 1e-9
    if not bool(ok.any()):
        return None
    capture = ((rand - chosen)[ok] / denom[ok]).mean().item()
    regret = (chosen - oracle).mean().item()
    pctile = (ct < chosen.unsqueeze(1)).float().mean(dim=1).mean().item()
    return capture, regret, pctile


def paired(a, b):
    """Paired mean difference a-b with the sem, t, and win count."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    sd = float(d.std(ddof=1))
    sem = sd / math.sqrt(n)
    return dict(mean=float(d.mean()), sd=sd, sem=sem,
                t=float(d.mean() / sem) if sem > 0 else float("inf"),
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), n=n)


def boot_ci(vals, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals)
    m = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def gaussian_capture(rho, K, n_sim=200000, seed=0):
    """Expected capture at slate size K under a bivariate-normal model.

    A model whose predicted dV correlates rho with the true dV, both
    marginally normal within a slate. Used ONLY to extrapolate the measured
    curve past the largest K the data can reach (n=32) -- validated against
    the measured K<=31 points before any extrapolated number is quoted.
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_sim, K, generator=g)
    e = torch.randn(n_sim, K, generator=g)
    p = rho * z + math.sqrt(max(1.0 - rho * rho, 0.0)) * e
    chosen = z.gather(1, p.argmin(dim=1, keepdim=True)).squeeze(1)
    oracle = z.min(dim=1).values
    rand = z.mean(dim=1)
    return float(((rand - chosen) / (rand - oracle)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--ks", default="2,4,8,16,24,31")
    ap.add_argument("--n-draw", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-slate", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=None,
                    help="comma-separated model/arm names from the cache "
                         "(default: the five reference models). Pass 'all' to "
                         "include every degradation arm the cache holds.")
    ap.add_argument("--pairs", default="UNet-linear,oracle-linear,linear-mean-delta",
                    help="comma-separated 'a-b' paired comparisons to report")
    ap.add_argument("--only-files", default=None,
                    help="comma-separated slate file names to keep (for the "
                         "EXP-0024 49-slate reproduction)")
    args = ap.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    dv = cache["dv"][args.goal]
    ep = cache["ep"]
    files = cache.get("slate_files", [])
    dv_true = dv["dv_true"]
    if args.models == "all":
        MODELS = ["oracle"] + [k for k in dv if k not in ("dv_true", "v0")]
    elif args.models:
        MODELS = args.models.split(",")
    else:
        MODELS = list(BASE_MODELS)
    preds = {m: (dv_true.clone() if m == "oracle" else dv[m]) for m in MODELS}
    ks = [int(k) for k in args.ks.split(",")]

    keep_files = set(args.only_files.split(",")) if args.only_files else None

    print(f"cache: {args.cache}  goal={args.goal}  "
          f"provenance={cache['provenance']['commit']} "
          f"dirty={cache['provenance']['dirty']}")

    # ---- per-slate, per-K metrics -----------------------------------------
    per = {m: {K: [] for K in ks} for m in MODELS}
    slate_ids, slate_ns = [], []
    for e in ep.unique().tolist():
        sel = (ep == e).nonzero(as_tuple=True)[0]
        if sel.numel() < args.min_slate:
            continue
        if keep_files is not None and files and files[e] not in keep_files:
            continue
        t = dv_true[sel]
        if float(t.std()) < 1e-9:
            continue
        slate_ids.append(e)
        slate_ns.append(int(sel.numel()))
        for K in ks:
            gen = torch.Generator().manual_seed(args.seed * 100003 + e)
            idx = _subsets(sel.numel(), K, args.n_draw, gen)
            for m in MODELS:
                r = slate_metrics(preds[m][sel], t, idx)
                per[m][K].append(r[0] if r else float("nan"))
                if m == "linear":
                    pass
        # unbounded companions, stored separately to keep the loop readable
    print(f"{len(slate_ids)} slates used (sizes: "
          f"{np.bincount(np.array(slate_ns)).nonzero()[0].tolist()})")

    # second pass for regret / percentile (same subsets, same seeds)
    reg = {m: {K: [] for K in ks} for m in MODELS}
    pct = {m: {K: [] for K in ks} for m in MODELS}
    for e in slate_ids:
        sel = (ep == e).nonzero(as_tuple=True)[0]
        t = dv_true[sel]
        for K in ks:
            gen = torch.Generator().manual_seed(args.seed * 100003 + e)
            idx = _subsets(sel.numel(), K, args.n_draw, gen)
            for m in MODELS:
                r = slate_metrics(preds[m][sel], t, idx)
                reg[m][K].append(r[1] if r else float("nan"))
                pct[m][K].append(r[2] if r else float("nan"))

    # ---- report ------------------------------------------------------------
    out = {"goal": args.goal, "n_slates": len(slate_ids), "ks": ks,
           "provenance": cache["provenance"], "cache": args.cache,
           "capture": {}, "regret_dv": {}, "pick_pctile": {}, "paired": {},
           "gaussian": {}}

    print("\n=== slateK: fraction of the oracle's advantage over a random pick ===")
    hdr = "  ".join(f"K={K:<7d}" for K in ks)
    print(f"{'model':22s} {hdr}")
    for m in MODELS:
        row = []
        for K in ks:
            v = np.array(per[m][K])
            lo, hi = boot_ci(v)
            row.append(f"{v.mean():.3f}±{(hi - lo) / 2:.3f}")
            out["capture"].setdefault(m, {})[K] = dict(
                mean=float(v.mean()), ci=[lo, hi],
                sem=float(v.std(ddof=1) / math.sqrt(len(v))))
        print(f"{m:22s} " + "  ".join(f"{r:<9s}" for r in row))

    print("\n=== regret_dv (chosen - best, Lyapunov units; lower better) ===")
    print(f"{'model':22s} {hdr}")
    for m in MODELS:
        row = []
        for K in ks:
            v = np.array(reg[m][K])
            row.append(f"{v.mean():.4f}")
            out["regret_dv"].setdefault(m, {})[K] = float(v.mean())
        print(f"{m:22s} " + "  ".join(f"{r:<9s}" for r in row))

    print("\n=== pick_pctile (fraction of the subset better than the pick) ===")
    print(f"{'model':22s} {hdr}")
    for m in MODELS:
        row = []
        for K in ks:
            v = np.array(pct[m][K])
            row.append(f"{v.mean():.3f}")
            out["pick_pctile"].setdefault(m, {})[K] = float(v.mean())
        print(f"{m:22s} " + "  ".join(f"{r:<9s}" for r in row))

    print("\n=== paired differences (per slate; sem is the floor, not the sd) ===")
    pairs = [tuple(p.rsplit("-", 1)) if p.count("-") == 1 else (p.split("-", 1)[0], p.split("-", 1)[1])
             for p in args.pairs.split(",") if p]
    pairs = [(a, b) for a, b in pairs if a in MODELS and b in MODELS]
    for a, b in pairs:
        print(f"\n  {a} - {b}")
        print(f"    {'K':>4s} {'mean':>9s} {'sd':>8s} {'sem':>8s} {'t':>7s} "
              f"{'wins':>8s} {'ties':>5s}")
        for K in ks:
            st = paired(per[a][K], per[b][K])
            out["paired"].setdefault(f"{a}-{b}", {})[K] = st
            print(f"    {K:4d} {st['mean']:+9.4f} {st['sd']:8.4f} {st['sem']:8.4f} "
                  f"{st['t']:7.2f} {st['wins']:4d}/{st['n']:<3d} {st['ties']:5d}")

    # ---- Gaussian-copula extrapolation ------------------------------------
    print("\n=== bivariate-normal extrapolation (validate on K<=31 first) ===")
    rhos = {}
    for m in MODELS:
        rs = []
        for e in slate_ids:
            sel = (ep == e).nonzero(as_tuple=True)[0]
            p, t = preds[m][sel], dv_true[sel]
            if float(p.std()) < 1e-12:
                continue
            pc = ((p - p.mean()) * (t - t.mean())).mean() / (p.std() * t.std())
            rs.append(float(pc))
        rhos[m] = float(np.mean(rs)) if rs else float("nan")
    print(f"  mean within-slate Pearson rho: "
          + ", ".join(f"{m}={rhos[m]:+.3f}" for m in MODELS if not math.isnan(rhos[m])))
    big_ks = ks + [64, 128, 256, 1000]
    print(f"\n  {'model':22s} " + "  ".join(f"K={K:<7d}" for K in big_ks))
    for m in MODELS:
        if math.isnan(rhos[m]) or m == "persistence":
            continue
        row = []
        for K in big_ks:
            c = 1.0 if m == "oracle" else gaussian_capture(rhos[m], K)
            row.append(f"{c:.3f}")
            out["gaussian"].setdefault(m, {})[K] = c
        out["gaussian"].setdefault(m, {})["rho"] = rhos[m]
        print(f"  {m:22s} " + "  ".join(f"{r:<9s}" for r in row))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
