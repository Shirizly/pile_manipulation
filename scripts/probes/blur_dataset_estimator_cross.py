"""C-021: does blur's sign on the density view come from the DATASET
(single-pile pile20 vs varied-starts) or the ESTIMATOR (linear-nonneg vs
ridge-toward-identity)?

docs/sand_manipulation.md Sec.6 (pile20, crop 1.0 / res 64, linear-nonneg,
density) says blur HURTS: 50.9% -> 54.5%.
docs/experiments/EXP-0003-blur-vs-view.md (varied, crop 1.0 / res 32/64,
ridge-toward-identity, density) says blur HELPS: 62.5% -> 33.9%.

Both factors moved between the two reports, so this crosses dataset x
estimator x blur x seed, view held at density throughout (that is what C-021
is about), at ONE resolution (res=32, crop=1.0) chosen for CPU cost -- see
provenance.runtime in the EXP record for why res=64 was not affordable for
linear-nonneg on this machine.
"""
import argparse
import itertools

import torch

from fit_linear_foresight import (actions_to_pixels, canonicalise,
                                   fit_operator, fit_operator_nonneg,
                                   metrics, swept_region_mask)
from occupancy_foresight import BOUNDS, load_transition_fields

ap = argparse.ArgumentParser()
ap.add_argument("--pile20-glob", required=True)
ap.add_argument("--varied-glob", required=True)
ap.add_argument("--varied-max-episodes", type=int, default=10)
ap.add_argument("--grid", type=int, default=64)
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--crop", type=float, default=1.0)
ap.add_argument("--min-push-mm", type=float, default=19.9)
ap.add_argument("--ridge", type=float, default=1.0)
ap.add_argument("--nonneg-iters", type=int, default=4000)
ap.add_argument("--blurs", type=float, nargs="+", default=[0.0, 1.0])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--estimators", nargs="+", default=["ridge", "nonneg"])
a = ap.parse_args()
H = W = a.grid

datasets = {
    "pile20": (a.pile20_glob, None),
    "varied": (a.varied_glob, a.varied_max_episodes),
}

print(f"{'dataset':8s} {'estimator':8s} {'blur':>5s} {'seed':>5s} "
      f"{'linear %chg':>12s} {'mean-delta %chg':>16s} {'identity %chg':>14s}")

rows = []
for dname, (glob, max_ep) in datasets.items():
    # Load once per (dataset, blur); reused across estimators and seeds.
    cache = {}
    for blur in a.blurs:
        o0, o1, act, ep, _, _ = load_transition_fields(
            glob, a.grid, blur, "mean", a.min_push_mm, "cpu", view="density",
            max_episodes=max_ep)
        s_px, e_px = actions_to_pixels(
            act, (BOUNDS["x_min"], BOUNDS["y_min"]),
            (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
        cache[blur] = (o0, o1, s_px, e_px, ep)

    for blur, seed in itertools.product(a.blurs, a.seeds):
        o0, o1, s_px, e_px, ep = cache[blur]
        eps = ep.unique()
        g = torch.Generator().manual_seed(seed)
        val = set(eps[torch.randperm(len(eps), generator=g)]
                  [:max(1, len(eps) // 4)].tolist())
        te = torch.tensor([int(e) in val for e in ep])
        tr = ~te
        Y0 = canonicalise(o0[tr], s_px[tr], e_px[tr], a.res, a.crop
                           ).reshape(int(tr.sum()), -1).T
        Y1 = canonicalise(o1[tr], s_px[tr], e_px[tr], a.res, a.crop
                           ).reshape(int(tr.sum()), -1).T

        ote, o1te, ste, ete = o0[te], o1[te], s_px[te], e_px[te]
        plate = 0.04 / 0.128 * W
        region = swept_region_mask(ste, ete, (H, W), 0.5 * plate + 2.0,
                                    0.5 * plate)
        base = metrics(ote, o1te, ote, region=region)["rms"]

        def pct(P):
            return 100 * metrics(P, o1te, ote, region=region)["rms"] / base

        from transforms.functional import (blend_push_prediction,
                                            from_push_frame,
                                            push_frame_validity_mask)
        Y0te = canonicalise(ote, ste, ete, a.res, a.crop
                             ).reshape(int(te.sum()), -1).T

        def to_world(Yp):
            back = from_push_frame(Yp.T.reshape(-1, a.res, a.res), ste, ete,
                                    (H, W), a.crop)
            m = push_frame_validity_mask(ste, ete, (H, W), (a.res, a.res),
                                          a.crop)
            return blend_push_prediction(back, ote, m).clamp_min(0.0)

        bmd = (Y1 - Y0).mean(1, keepdim=True)
        p_md = pct(to_world(Y0te + bmd))
        p_id = pct(to_world(Y0te))

        for est in a.estimators:
            if est == "ridge":
                A = fit_operator(Y0, Y1, a.ridge, toward_identity=True)
            elif est == "nonneg":
                A = fit_operator_nonneg(Y0, Y1, max_iter=a.nonneg_iters,
                                         ridge=a.ridge, toward_identity=True)
            else:
                raise SystemExit(f"unknown estimator {est}")
            p_lin = pct(to_world(A @ Y0te))
            print(f"{dname:8s} {est:8s} {blur:5.1f} {seed:5d} "
                  f"{p_lin:11.1f}% {p_md:15.1f}% {p_id:13.1f}%")
            rows.append((dname, est, blur, seed, p_lin, p_md, p_id))

print("\n--- summary: mean +/- sd over seeds, linear %chg ---")
print(f"{'dataset':8s} {'estimator':8s} {'blur':>5s} {'mean':>8s} {'sd':>6s} n")
import statistics as stats
for (dname, est, blur), grp in itertools.groupby(
        sorted(rows, key=lambda r: (r[0], r[1], r[2])),
        key=lambda r: (r[0], r[1], r[2])):
    vals = [r[4] for r in grp]
    m = stats.mean(vals)
    sd = stats.stdev(vals) if len(vals) > 1 else float("nan")
    print(f"{dname:8s} {est:8s} {blur:5.1f} {m:8.1f} {sd:6.2f} {len(vals)}")
