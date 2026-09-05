"""C-004 re-run on corrected grids (post aac084e3): fit_operator_nonneg vs
fit_operator at several ridge strengths, swept-region metric, on >=2 (res,
crop) configurations. One code path (load_transition_fields) for both the
scattered and piled dataset, so a difference between them is data not
rasteriser (EXP-0001/EXP-0002).

COST WARNING (measured, see EXP-0010): fit_operator_nonneg is O(D^3) per FISTA
iteration. D=1024 (res 32) ~51s per fit at 4000 iters; D=4096 (res 64) is
projected at over an hour and is NOT run here -- res 32 only, per the budget.
"""
import argparse, time, torch

from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import (canonicalise, fit_operator, fit_operator_nonneg,
                                  metrics, swept_region_mask, actions_to_pixels)
from transforms.functional import (blend_push_prediction, from_push_frame,
                                   push_frame_validity_mask)

ap = argparse.ArgumentParser()
ap.add_argument("--blur", type=float, default=1.0)
ap.add_argument("--ridge-sweep", default="0.1,1,10,100")
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--max-episodes", type=int, default=None,
               help="cap episode FILES per dataset -- particles_to_occupancy's "
                    "footprint_radius path costs ~60ms/transition (measured, "
                    "an unrelated finding), so bound the load for a fast run.")
a = ap.parse_args()
H = W = 64
RIDGES = [float(x) for x in a.ridge_sweep.split(",") if x.strip()]


_CACHE = {}


def load_cached(glob, cube_size, min_push_mm, min_grains):
    key = (glob, cube_size, min_push_mm, min_grains)
    if key not in _CACHE:
        print(f"  loading {glob} ...", flush=True)
        _CACHE[key] = load_transition_fields(
            glob, 64, a.blur, "mean", min_push_mm, "cpu",
            view="mask", min_grains=min_grains, cube_size=cube_size,
            max_episodes=a.max_episodes)
        print(f"  loaded {_CACHE[key][0].shape[0]} transitions", flush=True)
    return _CACHE[key]


def run(dataset_name, glob, cube_size, min_push_mm, min_grains, res, crop):
    o0, o1, act, ep, _, _ = load_cached(glob, cube_size, min_push_mm, min_grains)
    s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                   (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    eps = ep.unique()
    g = torch.Generator().manual_seed(0)
    val = set(eps[torch.randperm(len(eps), generator=g)][:max(1, len(eps)//4)].tolist())
    te = torch.tensor([int(e) in val for e in ep]); tr = ~te

    Y0 = canonicalise(o0[tr], s_px[tr], e_px[tr], res, crop).reshape(int(tr.sum()), -1).T
    Y1 = canonicalise(o1[tr], s_px[tr], e_px[tr], res, crop).reshape(int(tr.sum()), -1).T
    D, M = Y0.shape
    bmd = (Y1 - Y0).mean(1, keepdim=True)

    ote, o1te, ste, ete = o0[te], o1[te], s_px[te], e_px[te]
    plate = 0.04 / 0.128 * W
    region = swept_region_mask(ste, ete, (H, W), 0.5*plate + 2.0, 0.5*plate)
    pers_rms = metrics(ote, o1te, ote, region=region)["rms"]

    Y0te = canonicalise(ote, ste, ete, res, crop).reshape(int(te.sum()), -1).T
    msk = push_frame_validity_mask(ste, ete, (H, W), (res, res), crop)

    def world(Yp):
        back = from_push_frame(Yp.T.reshape(-1, res, res), ste, ete, (H, W), crop)
        return blend_push_prediction(back, ote, msk).clamp_min(0.0)

    def rms_of(pred):
        return metrics(pred, o1te, ote, region=region)["rms"]

    md_rms = rms_of(world(Y0te + bmd))

    print(f"\n=== {dataset_name}  res={res} crop={crop} blur={a.blur} "
          f"D={D} M={M} ({'OVER' if M > D else 'UNDER'}determined) "
          f"N_test={int(te.sum())} ===")
    print(f"{'model':16s} {'rms':>9s} {'%persist':>9s} {'%meandelta':>11s} {'fit s':>7s}")
    print(f"{'persistence':16s} {pers_rms:9.5f} {100.0:9.1f} "
          f"{100*pers_rms/md_rms:11.1f} {'-':>7s}")
    print(f"{'mean-delta':16s} {md_rms:9.5f} {100*md_rms/pers_rms:9.1f} "
          f"{100.0:11.1f} {'-':>7s}")

    rows = []
    for lam in RIDGES:
        t = time.time()
        A = fit_operator(Y0, Y1, lam, toward_identity=True)
        dt = time.time() - t
        r = rms_of(world(A @ Y0te))
        rows.append((f"ridge{lam:g}", r, dt))
    t = time.time()
    A = fit_operator_nonneg(Y0, Y1, max_iter=a.iters, ridge=0.0, toward_identity=True)
    dt = time.time() - t
    r = rms_of(world(A @ Y0te))
    rows.append(("nonneg", r, dt))

    for name, r, dt in sorted(rows, key=lambda x: x[1]):
        print(f"{name:16s} {r:9.5f} {100*r/pers_rms:9.1f} {100*r/md_rms:11.1f} {dt:7.1f}")
    best = min(rows, key=lambda x: x[1])
    print(f"winner: {best[0]}")
    return dataset_name, res, crop, rows, pers_rms, md_rms


results = []
for res, crop in [(32, 1.0), (32, 0.5)]:
    results.append(run("SCATTERED (L040, n50)", "Genesis/data/foresight/L040/**/*_data.pt",
                       0.005, 39.0, 1.0, res, crop))
    results.append(run("PILED (n20, 2-layer)", "Genesis/data/cube_spectrum/n20/*_data.pt",
                       0.005, 19.9, 1.0, res, crop))

print("\n\n=== SUMMARY: winner per configuration ===")
for name, res, crop, rows, pers, md in results:
    best = min(rows, key=lambda x: x[1])
    nonneg_row = [r for r in rows if r[0] == "nonneg"][0]
    best_ridge = min([r for r in rows if r[0] != "nonneg"], key=lambda x: x[1])
    print(f"{name:24s} res={res} crop={crop}: winner={best[0]:10s}  "
          f"nonneg={100*nonneg_row[1]/pers:.1f}%  best-ridge({best_ridge[0]})="
          f"{100*best_ridge[1]/pers:.1f}%")
