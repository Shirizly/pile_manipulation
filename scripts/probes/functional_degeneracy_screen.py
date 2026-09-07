"""Degeneracy screen for a (dataset, functional) cell, run BEFORE any
slateK/slateK_exact number is computed or reported.

Motivation (EXP-0024_v2/EXP-0026_v2): C-040 is the precedent -- a centred
convex target gives dV=0 identically for a centred pile, and EXP-0024/0026
wasted cells discovering it only after computing full metric suites. A sharp
functional on a SMALL target is at risk of the opposite failure: a single push
moves only 1-2 cubes across the target boundary, so dV may be exactly zero for
most candidates and take only a few discrete values for the rest, which can
make slateK_exact numerically well-defined but not meaningfully informative
(same handful of achievable dV values recur across candidates).

For every (dataset, functional) cell, this reports, from the dv cache alone
(no K-sweep needed):
  * dv_true mean / sd, pooled over every candidate in the cell
  * % helpful (dv_true < 0)
  * % of candidates with |dv_true| < 1e-9 (exactly-zero pushes under this
    functional)
  * # of distinct dv_true values pooled (a small count is the discrete-value
    symptom described above)
  * # / total slates clearing the same per-slate variation threshold
    exp0026_kcurve_exact.py itself uses to skip a slate (std(dv_true) < 1e-9)

A cell that fails this screen (near-0% helpful, near-100% exact-zero, or most
slates failing the variation threshold) is reported as degenerate and
EXCLUDED from any conclusion -- never silently dropped, per the task's
explicit instruction.

Usage
-----
    PYTHONPATH=. python scripts/probes/functional_degeneracy_screen.py \
        runs_exp0024_v2/dv_cache_A.pt --goals corner,ind-square8,... \
        --min-slate 8
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def screen_one(cache, goal, min_slate=8, var_eps=1e-9, zero_eps=1e-9):
    dv = cache["dv"][goal]
    dv_true = dv["dv_true"]
    ep = cache["ep"]

    n = dv_true.shape[0]
    mean_ = float(dv_true.mean())
    sd_ = float(dv_true.std())
    pct_helpful = 100.0 * float((dv_true < 0).float().mean())
    pct_zero = 100.0 * float((dv_true.abs() < zero_eps).float().mean())
    n_distinct = int(torch.unique((dv_true / max(zero_eps, 1e-12)).round()).numel())

    n_slates_total = 0
    n_slates_clear = 0
    for e in ep.unique().tolist():
        sel = (ep == e).nonzero(as_tuple=True)[0]
        if sel.numel() < min_slate:
            continue
        n_slates_total += 1
        t = dv_true[sel]
        if float(t.std()) >= var_eps:
            n_slates_clear += 1

    degenerate = (pct_helpful < 1.0 or pct_helpful > 99.0 or pct_zero > 90.0
                  or (n_slates_total > 0 and n_slates_clear / n_slates_total < 0.5))
    return dict(goal=goal, n=n, mean=mean_, sd=sd_, pct_helpful=pct_helpful,
                pct_zero=pct_zero, n_distinct=n_distinct,
                n_slates_total=n_slates_total, n_slates_clear=n_slates_clear,
                degenerate=degenerate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--goals", required=True)
    ap.add_argument("--min-slate", type=int, default=8)
    args = ap.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    goals = [g.strip() for g in args.goals.split(",") if g.strip()]
    print(f"cache={args.cache} provenance={cache['provenance']['commit']} "
          f"dirty={cache['provenance']['dirty']}")
    print(f"{'goal':24s} {'n':>6s} {'mean':>10s} {'sd':>10s} {'%helpful':>9s} "
          f"{'%|dv|<eps':>10s} {'#distinct':>9s} {'slates clear':>13s} {'status':>12s}")
    for g in goals:
        if g not in cache["dv"]:
            print(f"{g:24s}  -- not in cache --")
            continue
        r = screen_one(cache, g, args.min_slate)
        clear_str = f"{r['n_slates_clear']}/{r['n_slates_total']}"
        status = "DEGENERATE" if r["degenerate"] else "ok"
        print(f"{g:24s} {r['n']:6d} {r['mean']:+10.6f} {r['sd']:10.6f} "
              f"{r['pct_helpful']:8.1f}% {r['pct_zero']:9.1f}% {r['n_distinct']:9d} "
              f"{clear_str:>13s} {status:>12s}")


if __name__ == "__main__":
    main()
