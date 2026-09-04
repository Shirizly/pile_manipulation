"""density-vs-mask, both under the SAME blur, on the repo's own metric.

docs/sand_manipulation.md compares a blurred mask against an UNBLURRED density
map, and concludes the mask view wins. Blur is not held fixed across that
comparison, so the view effect and the blur effect are confounded.
"""
import argparse, itertools, torch
from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import (canonicalise, fit_operator, metrics,
                                  predict_world, swept_region_mask,
                                  actions_to_pixels)

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--cube-size", type=float, default=None)
ap.add_argument("--grid", type=int, default=64)
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--crop", type=float, default=1.0)
ap.add_argument("--min-grains", type=float, default=2.0)
ap.add_argument("--max-episodes", type=int, default=None)
ap.add_argument("--ridge", type=float, default=1.0)
ap.add_argument("--min-push-mm", type=float, default=19.9)
a = ap.parse_args()
H = W = a.grid
print(f"\n### {a.label or a.glob}   res={a.res} crop={a.crop}")
print(f"{'view':10s} {'blur':>5s} {'linear %chg':>12s} {'mean-delta %chg':>16s} "
      f"{'identity %chg':>14s} {'operator margin':>16s}")
for view, blur in itertools.product(["mask", "density"], [0.0, 0.5, 1.0, 1.5]):
    o0, o1, act, ep, _, _ = load_transition_fields(
        a.glob, a.grid, blur, "mean", a.min_push_mm, "cpu", view=view,
        min_grains=a.min_grains, cube_size=a.cube_size,
        max_episodes=a.max_episodes)
    M = o0.shape[0]
    s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                   (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    eps = ep.unique()
    g = torch.Generator().manual_seed(0)
    val = set(eps[torch.randperm(len(eps), generator=g)][:max(1, len(eps)//4)].tolist())
    te = torch.tensor([int(e) in val for e in ep]); tr = ~te
    Y0 = canonicalise(o0[tr], s_px[tr], e_px[tr], a.res, a.crop).reshape(int(tr.sum()), -1).T
    Y1 = canonicalise(o1[tr], s_px[tr], e_px[tr], a.res, a.crop).reshape(int(tr.sum()), -1).T
    A = fit_operator(Y0, Y1, a.ridge, toward_identity=True)
    D = Y0.shape[0]
    Amd = torch.eye(D)                       # mean-delta as an affine map
    bmd = (Y1 - Y0).mean(1, keepdim=True)
    ote, o1te, ste, ete = o0[te], o1[te], s_px[te], e_px[te]
    plate = 0.04 / 0.128 * W
    region = swept_region_mask(ste, ete, (H, W), 0.5 * plate + 2.0, 0.5 * plate)
    base = metrics(ote, o1te, ote, region=region)["rms"]
    def pct(P): return 100 * metrics(P, o1te, ote, region=region)["rms"] / base
    from transforms.functional import (blend_push_prediction, from_push_frame,
                                       push_frame_validity_mask)
    Y0te = canonicalise(ote, ste, ete, a.res, a.crop).reshape(int(te.sum()), -1).T
    def to_world(Yp):
        back = from_push_frame(Yp.T.reshape(-1, a.res, a.res), ste, ete, (H, W), a.crop)
        m = push_frame_validity_mask(ste, ete, (H, W), (a.res, a.res), a.crop)
        return blend_push_prediction(back, ote, m).clamp_min(0.0)
    p_lin, p_md = pct(to_world(A @ Y0te)), pct(to_world(Y0te + bmd))
    p_id = pct(to_world(Y0te))
    print(f"{view:10s} {blur:5.1f} {p_lin:11.1f}% {p_md:15.1f}% {p_id:13.1f}% "
          f"{p_md - p_lin:15.1f}")
