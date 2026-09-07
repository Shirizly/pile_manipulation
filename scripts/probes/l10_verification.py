"""C2: is L10mm's accuracy=-0.44 / slateK_exact=0.91 dissociation real, or a
symptom of a near-degenerate pool spread inflating a ratio metric?

L10mm's `dv_true` sd is only ~0.005 (n20_L10mm_dv_cache.pt, goal=corner) --
two orders of magnitude smaller than the swept-region push itself, so every
`slateK_exact`/capture-fraction number there divides by a SMALL per-slate
denominator (`mean(t) - E_oracle(K)`, essentially `mean - min` of a nearly
flat pool). A ratio with a small denominator is unstable: the same absolute
skill produces a much larger reported fraction than it would at a cell with
real spread, which would look like "near-oracle ranking" for a reason that
has nothing to do with ranking skill.

This checks three things directly rather than assuming either story:

  (a) does per-slate spread (max-min, mean-min of dv_true) correlate with
      per-slate slateK_exact capture? A correlation between spread and the
      reported capture fraction would point at ratio instability (the same
      absolute skill reading out as a different fraction depending on how
      small the denominator is); no correlation is consistent with the
      capture number reflecting real skill regardless of scale.
  (b) recompute dv_pred (linear operator) for this cell through an
      INDEPENDENT code path -- a fresh fit via `dmdc_baseline.
      load_transition_arrays` (not the cache-building script's build_dataset
      stacking loop), occ0 reloaded via `pool_common.load_occ0_for_slate`
      (a third loading route), predict_world/lyapunov applied to that -- and
      compares against the cached "linear" dv_pred row for row.
  (c) reports the RAW, un-normalised |R_K| (Lyapunov units) beside the
      normalised slateK_exact capture fraction, and beside L20mm/L40mm's own
      |R_K| (from their published kcurve_exact caches) for scale context --
      a large FRACTION of a tiny available gain is not the same achievement
      as the same fraction of a large one.

Usage
-----
    PYTHONPATH=. python scripts/probes/l10_verification.py \\
        runs_expB/n20_L10mm_dv_cache.pt --goal corner \\
        --models linear,UNet --k 4,128 \\
        --context runs_expB/n20_L20mm_kcurve_exact.json,runs_expB/n20_L40mm_kcurve_exact.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy import stats

from control_utility_test import lyapunov, lyapunov_weights
from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import actions_to_pixels, canonicalise, fit_operator, predict_world
from scripts.probes.exp0026_kcurve_exact import per_slate_exact
from scripts.probes.pool_common import (
    action_geometry, admissible_slates, load_cache, load_occ0_for_slate, rk_curve,
)

R, CR, RIDGE = 64, 1.0, 1.0


def part_a(cache, goal, models, ks):
    ep = cache["ep"]
    dv = cache["dv"][goal]
    dv_true_all = dv["dv_true"]
    print(f"\n=== (a) spread vs. capture, goal={goal} ===")
    print(f"overall dv_true: sd={float(dv_true_all.std()):.6f} "
          f"mean={float(dv_true_all.mean()):+.6f} "
          f"|mean/sd|={abs(float(dv_true_all.mean()))/float(dv_true_all.std()):.3f}")

    slates = admissible_slates(cache, goal, min_slate=8)
    max_min, mean_min = [], []
    per_slate = {m: {K: [] for K in ks} for m in models}
    for e in slates:
        sel = (ep == e).nonzero(as_tuple=True)[0]
        t = dv_true_all[sel]
        max_min.append(float(t.max() - t.min()))
        mean_min.append(float(t.mean() - t.min()))
        for m in models:
            p = t.clone() if m == "oracle" else dv[m][sel]
            exact, _worst, _prof = per_slate_exact(p, t, ks)
            for K in ks:
                if K in exact:
                    per_slate[m][K].append(exact[K])
                else:
                    per_slate[m][K].append(float("nan"))
    max_min, mean_min = np.array(max_min), np.array(mean_min)
    print(f"n_slates={len(slates)}")
    print(f"pool spread (max-min): mean={max_min.mean():.6f} median={np.median(max_min):.6f} "
          f"min={max_min.min():.6f} max={max_min.max():.6f}")
    print(f"pool spread (mean-min): mean={mean_min.mean():.6f} median={np.median(mean_min):.6f} "
          f"min={mean_min.min():.6f} max={mean_min.max():.6f}")

    for m in models:
        for K in ks:
            vals = np.array(per_slate[m][K])
            ok = np.isfinite(vals)
            if ok.sum() < 3:
                continue
            r_mm, p_mm = stats.pearsonr(max_min[ok], vals[ok])
            rs_mm, ps_mm = stats.spearmanr(max_min[ok], vals[ok])
            r_mn, p_mn = stats.pearsonr(mean_min[ok], vals[ok])
            print(f"  model={m:10s} K={K:4d}  slateK_exact mean={vals[ok].mean():.4f} "
                  f"sd={vals[ok].std(ddof=1):.4f}  "
                  f"corr(spread=max-min): pearson r={r_mm:+.3f} (p={p_mm:.3f}) "
                  f"spearman rho={rs_mm:+.3f} (p={ps_mm:.3f})  "
                  f"corr(spread=mean-min): pearson r={r_mn:+.3f} (p={p_mn:.3f})")
    return slates, per_slate, max_min, mean_min


def part_b(cache, goal, train_cfg, n_check_slates):
    print(f"\n=== (b) independent recompute of dv_pred (linear), goal={goal} ===")
    print(f"independent fit via dmdc_baseline.load_transition_arrays({train_cfg!r}), "
          f"fresh fit_operator(ridge={RIDGE}, toward_identity=True), occ0 reloaded via "
          f"pool_common.load_occ0_for_slate (build_dataset call, independent of the "
          f"cache-building script's own stacking loop)")
    data_tr = load_transition_arrays(train_cfg, split="train")
    H, W = data_tr.occ_t.shape[-2:]
    s_tr, e_tr = actions_to_pixels(data_tr.actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    n_tr = data_tr.occ_t.shape[0]
    Y0 = canonicalise(data_tr.occ_t, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    Y1 = canonicalise(data_tr.occ_t1, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    A = fit_operator(Y0, Y1, ridge=RIDGE, toward_identity=True)
    print(f"  independent fit: {n_tr} train transitions, operator {A.shape}")

    dw = lyapunov_weights((H, W), goal, "cpu")
    ep = cache["ep"]
    dv = cache["dv"][goal]
    slates = admissible_slates(cache, goal, min_slate=8)[:n_check_slates]

    cached, recomputed = [], []
    for sid in slates:
        occ0 = load_occ0_for_slate(cache, sid)
        sel = (ep == sid).nonzero(as_tuple=True)[0]
        actions = cache["actions"][sel]
        s_px, e_px = actions_to_pixels(actions, data_tr.workspace_min,
                                       data_tr.workspace_max, (H, W))
        occ0_batch = occ0.unsqueeze(0).expand(sel.numel(), -1, -1)
        pred1 = predict_world(A, occ0_batch, s_px, e_px, R, (H, W), CR)
        v0 = lyapunov(occ0_batch, dw)
        dv_pred_indep = lyapunov(pred1, dw) - v0
        cached.append(dv["linear"][sel].numpy())
        recomputed.append(dv_pred_indep.numpy())
    cached = np.concatenate(cached)
    recomputed = np.concatenate(recomputed)
    diff = recomputed - cached
    r, p = stats.pearsonr(cached, recomputed)
    print(f"  checked {len(cached)} rows across {len(slates)} slates")
    print(f"  cached vs independent dv_pred: pearson r={r:.5f} (p={p:.2e})")
    print(f"  mean diff={diff.mean():+.3e}  sd diff={diff.std():.3e}  "
          f"max|diff|={np.abs(diff).max():.3e}  "
          f"(cached sd={cached.std():.3e}, so max|diff|/sd={np.abs(diff).max()/cached.std():.3f})")
    return r, diff


def part_c(cache, goal, models, ks, context_paths):
    print(f"\n=== (c) raw |R_K| (Lyapunov units) beside capture fraction, goal={goal} ===")
    ep = cache["ep"]
    dv = cache["dv"][goal]
    slates = admissible_slates(cache, goal, min_slate=8)
    for m in models:
        for K in ks:
            rk_abs, exact_vals = [], []
            for sid in slates:
                sel = (ep == sid).nonzero(as_tuple=True)[0]
                t = dv["dv_true"][sel]
                p = t.clone() if m == "oracle" else dv[m][sel]
                Mk, Pk = rk_curve(p, t, [K])
                if K not in Mk:
                    continue
                rk_abs.append(abs(Mk[K] - Pk[K]))
                exact, _w, _pr = per_slate_exact(p, t, [K])
                if K in exact:
                    exact_vals.append(exact[K])
            rk_abs = np.array(rk_abs)
            print(f"  model={m:10s} K={K:4d}  mean|R_K|={rk_abs.mean():.6f} "
                  f"(sd={rk_abs.std(ddof=1):.6f}, n={len(rk_abs)})  "
                  f"mean slateK_exact={np.mean(exact_vals):.4f}")
    for path in context_paths:
        if not path:
            continue
        try:
            j = json.loads(open(path).read())
        except FileNotFoundError:
            print(f"  (context file not found: {path})")
            continue
        print(f"  context: {path} slateK_exact linear @K=4/K=128: "
              f"{j['slateK_exact']['linear'].get('4')}/{j['slateK_exact']['linear'].get('128')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache")
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--models", default="linear,UNet")
    ap.add_argument("--k", default="4,128")
    ap.add_argument("--train-cfg", default="configs/dataset/genesis_slates_multistep_n20_L10mm_train.yaml")
    ap.add_argument("--n-check-slates", type=int, default=6)
    ap.add_argument("--context", default="")
    args = ap.parse_args()

    cache = load_cache(args.cache)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ks = [int(k) for k in args.k.split(",")]
    print(f"=== l10_verification: {args.cache} goal={args.goal} ===")
    print(f"provenance={cache['provenance']['commit']} dirty={cache['provenance']['dirty']}")

    part_a(cache, args.goal, models, ks)
    part_b(cache, args.goal, args.train_cfg, args.n_check_slates)
    context_paths = [p.strip() for p in args.context.split(",") if p.strip()]
    part_c(cache, args.goal, models, ks, context_paths)


if __name__ == "__main__":
    main()
