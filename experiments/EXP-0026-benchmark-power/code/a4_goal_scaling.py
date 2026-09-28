"""EXP-0026 A4: do extra GOALS (free) substitute for extra STATES (costly)?

Same corpora / models / step-0 pools as eval_report. Per slate, score slateN
under G_MAX random goals (disks and axis-aligned rectangles at random
positions/sizes, seeded, identical for every model), for lyapunov (distance
field of the mask) and mass_in_region (the mask). Then for G in {1,3,6,12,24}
average per-slate slateN over random goal subsets and measure the median
per-pair sd of the paired difference, required slates, and resolved pairs.
Only per-(model, slate, goal) capture is saved; predictions are regenerable.
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from control_utility_test import lyapunov
from Baselines.common import eval_report as er
from Baselines.common.goals import (dist_field_from_mask, higher_is_better_for,
                                    mass_in_region, slate_n_capture)
from Baselines.common.eval_baseline import _predictor_batch
from Baselines.common.paired_stats import paired_comparison, required_n

G_MAX = 24
GS = [1, 3, 6, 12, 24]


def random_goals(H, W, n, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    out = []
    while len(out) < n:
        if len(out) % 2 == 0:       # disk
            r = rng.uniform(0.08, 0.2) * H
            cy, cx = rng.uniform(r, H - r), rng.uniform(r, W - r)
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        else:                       # rectangle
            h, w = rng.uniform(0.15, 0.45) * H, rng.uniform(0.15, 0.45) * W
            y0, x0 = rng.uniform(0, H - h), rng.uniform(0, W - w)
            m = (yy >= y0) & (yy < y0 + h) & (xx >= x0) & (xx < x0 + w)
        if 0.02 < m.mean() < 0.3:
            out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--corpora", default="L20mm,L40mm,randlen_test")
    ap.add_argument("--subsets", type=int, default=200)
    args = ap.parse_args()
    models = args.models.split(",")
    res = {"G_MAX": G_MAX, "models": models, "corpora": {}}
    for cname in args.corpora.split(","):
        cell = er._load_cell(er.CORPORA[cname], tag=cname)
        step0 = cell.step_idx == 0
        sids = cell.slate_idx[step0]
        slates = sids.unique().tolist()
        true_s0 = cell.occ1[step0].float()
        H, W = cell.occ0.shape[-2:]
        goals = random_goals(H, W, G_MAX, seed=12345)
        masks = [torch.from_numpy(g.astype(np.float32)) for g in goals]
        dists = [torch.from_numpy(dist_field_from_mask(g)).float() for g in goals]
        cap = {v: np.full((len(models), len(slates), G_MAX), np.nan)
               for v in ("lyapunov", "mass_in_region")}
        for mi, m in enumerate(models):
            t0 = time.time()
            pred = er._load_predictor(er.MODELS[m])
            batch = _predictor_batch(cell) if hasattr(cell, "states") else cell
            p = pred.predict_occ(batch).float().cpu()[step0.cpu() if hasattr(step0, "cpu") else step0]
            for si, s in enumerate(slates):
                rows = (sids == s).nonzero(as_tuple=True)[0]
                tp, pp = true_s0[rows].cpu(), p[rows]
                for gi in range(G_MAX):
                    cap["lyapunov"][mi, si, gi] = slate_n_capture(
                        lyapunov(pp, dists[gi]), lyapunov(tp, dists[gi]),
                        higher_is_better=higher_is_better_for("lyapunov"))
                    cap["mass_in_region"][mi, si, gi] = slate_n_capture(
                        mass_in_region(pp, masks[gi]), mass_in_region(tp, masks[gi]),
                        higher_is_better=higher_is_better_for("mass_in_region"))
            print(f"[{cname}/{m}] {time.time()-t0:.1f}s  lyap mean {np.nanmean(cap['lyapunov'][mi]):.3f}",
                  flush=True)
            del pred
        rc = {"n_slates": len(slates), "per_vf": {}}
        rng = np.random.default_rng(0)
        for v, C in cap.items():
            rv = {"per_G": {}}
            for G in GS:
                sds, nres = [], []
                for _ in range(args.subsets if G < G_MAX else 1):
                    gsub = rng.choice(G_MAX, size=G, replace=False)
                    X = np.nanmean(C[:, :, gsub], axis=2)
                    for i, j in itertools.combinations(range(len(models)), 2):
                        d = X[i] - X[j]; d = d[~np.isnan(d)]
                        sds.append(d.std(ddof=1))
                sd = float(np.median(sds))
                X = np.nanmean(C[:, :, rng.choice(G_MAX, size=G, replace=False)], axis=2)
                pairs = paired_comparison(X, models, n_boot=2000)
                rv["per_G"][str(G)] = dict(median_sd_diff=sd,
                                           required_slates_d02=required_n(sd, 0.02),
                                           required_slates_d05=required_n(sd, 0.05),
                                           n_resolved_ci_one_subset=sum(p["resolved_ci"] for p in pairs))
            # ICC over all 24 goals, median over pairs
            iccs = []
            for i, j in itertools.combinations(range(len(models)), 2):
                D = (C[i] - C[j]).T                     # (G, S)
                D = D[:, ~np.isnan(D).any(0)]
                G = D.shape[0]
                msb = G * D.mean(0).var(ddof=1)
                msw = ((D - D.mean(0)) ** 2).sum() / (D.shape[1] * (G - 1))
                s2 = max((msb - msw) / G, 0.0)
                iccs.append(s2 / (s2 + msw) if s2 + msw > 0 else np.nan)
            rv["icc_24_median"] = float(np.nanmedian(iccs))
            rv["shrink_sd_G3_to_G24"] = rv["per_G"]["3"]["median_sd_diff"] / rv["per_G"]["24"]["median_sd_diff"]
            rc["per_vf"][v] = rv
            print(f"  {cname} {v}: ICC24 {rv['icc_24_median']:.3f}  sd shrink G3->G24 "
                  f"{rv['shrink_sd_G3_to_G24']:.2f}x  " + "  ".join(
                      f"G={G}: sd {x['median_sd_diff']:.3f} nS(0.02)={x['required_slates_d02']} "
                      f"res={x['n_resolved_ci_one_subset']}" for G, x in rv["per_G"].items()), flush=True)
        rc["capture"] = {v: C.tolist() for v, C in cap.items()}
        res["corpora"][cname] = rc
        Path(args.out).write_text(json.dumps(res))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
