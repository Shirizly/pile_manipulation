"""
EXP-0018 probe: is the C-001 reversal (linear operator beats persistence and
mean-delta) a property of one crop/res cell, or does it hold across the
configuration space?

Loads genesis_foresight_L040 ONCE, builds ONE seed-0 episode split, then loops
over a (res, crop) grid, canonicalising and fitting ols + ridge(1.0) fresh at
each cell (nonneg is intentionally NOT swept here -- EXP-0009 already showed
ols/ridge/nonneg agree to ~0.1pt at res=8, and nonneg's O(D^3) FISTA is the
part of this codebase that is infeasible to sweep on CPU; see EXP-0009
Threats). Reuses fit_linear_foresight.py's own canonicalise/fit_operator/
predict_world/metrics/swept_region_mask/actions_to_pixels and exp0009_rerun's
mean-delta predictor verbatim -- no new rasteriser, no new warp, no new
estimator.

Usage
-----
    OMP_NUM_THREADS=4 PYTHONPATH=. python -u scripts/probes/exp0018_config_sweep.py \
        configs/dataset/genesis_foresight_L040.yaml --blur 1.0 --split-seed 0
"""
from __future__ import annotations

import argparse
import time

import torch

from dmdc_baseline import load_transition_arrays, split_by_episode
from transforms.functional import (
    blend_push_prediction, from_push_frame, push_frame_validity_mask,
    to_push_frame,
)
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, metrics, predict_world,
    swept_region_mask, verify_pixel_mapping,
)


def predict_meandelta(bmd, occ, start_px, end_px, res, grid_res, scale=1.0,
                       batch=256):
    H, W = grid_res
    outs = []
    for i in range(0, occ.shape[0], batch):
        sl = slice(i, i + batch)
        o, s, e = occ[sl], start_px[sl], end_px[sl]
        canon = to_push_frame(o, s, e, (res, res), scale)
        pred_c = (canon.reshape(canon.shape[0], -1) + bmd.reshape(1, -1)
                 ).reshape(-1, res, res)
        back = from_push_frame(pred_c, s, e, (H, W), scale)
        mask = push_frame_validity_mask(s, e, (H, W), (res, res), scale)
        outs.append(blend_push_prediction(back, o, mask).clamp_(0.0, 1.0))
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_cfg")
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--res-list", default="8,16,32,64")
    ap.add_argument("--crop-list", default="0.25,0.5,1.0")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    t0 = time.time()
    data = load_transition_arrays(args.dataset_cfg, split="train", max_samples=None)
    data.cfg_path, data.split = args.dataset_cfg, "train"
    H, W = data.occ_t.shape[-2:]
    print(f"loaded {data.occ_t.shape[0]} transitions, grid {H}x{W}, "
          f"{time.time() - t0:.1f}s")

    if args.blur > 0:
        sig = args.blur
        k = int(2 * round(3 * sig) + 1)
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-ax ** 2 / (2 * sig * sig))
        g = (g / g.sum())
        def _blur(x):
            x = torch.nn.functional.conv2d(x.unsqueeze(1), g.view(1, 1, 1, -1),
                                           padding=(0, k // 2))
            return torch.nn.functional.conv2d(x, g.view(1, 1, -1, 1),
                                              padding=(k // 2, 0)).squeeze(1)
        data.occ_t, data.occ_t1 = _blur(data.occ_t), _blur(data.occ_t1)

    start_px, end_px = actions_to_pixels(
        data.actions, data.workspace_min, data.workspace_max, (H, W))
    d_mid, _ = verify_pixel_mapping(data, start_px, end_px, (H, W))
    print(f"pixel-mapping check: median {d_mid:.2f} px from midpoint")
    if not (d_mid < 6.0):
        print("  !! mapping looks wrong -- aborting")
        return

    torch.manual_seed(args.split_seed)
    m_tr, m_te = split_by_episode(data, holdout_frac=args.holdout_frac)
    print(f"split: {int(m_tr.sum())} train / {int(m_te.sum())} test "
          f"(seed {args.split_seed})\n")

    dev = args.device
    occ_tr, occ1_tr = data.occ_t[m_tr].to(dev), data.occ_t1[m_tr].to(dev)
    occ_te, occ1_te = data.occ_t[m_te].to(dev), data.occ_t1[m_te].to(dev)
    s_tr, e_tr = start_px[m_tr].to(dev), end_px[m_tr].to(dev)
    s_te, e_te = start_px[m_te].to(dev), end_px[m_te].to(dev)
    plate_px = 0.04 / 0.128 * W
    M_tr = occ_tr.shape[0]

    res_list = [int(x) for x in args.res_list.split(",")]
    crop_list = [float(x) for x in args.crop_list.split(",")]

    print(f"{'res':>4s} {'crop':>5s} {'D':>6s} {'M/D':>7s} {'pct_persist(ols)':>17s} "
          f"{'pct_persist(ridge1)':>19s} {'pct_meandelta(ridge1)':>21s} "
          f"{'cov_swept_in_crop':>18s} {'t(s)':>6s}")
    for res in res_list:
        for crop in crop_list:
            t = time.time()
            D = res * res
            region = swept_region_mask(s_te, e_te, (H, W),
                                       half_width_px=0.5 * plate_px + 2.0,
                                       pad_px=0.5 * plate_px)
            valid = push_frame_validity_mask(s_te, e_te, (H, W), (res, res), crop)
            cov = float((region * valid).sum() / region.sum().clamp_min(1.0))

            Y0 = canonicalise(occ_tr, s_tr, e_tr, res, crop).reshape(M_tr, -1).T
            Y1 = canonicalise(occ1_tr, s_tr, e_tr, res, crop).reshape(M_tr, -1).T

            def expl(pred, base_rms):
                m = metrics(pred, occ1_te, occ_te, region=region)
                return m["rms"], 100 * m["rms"] / base_rms

            rp = metrics(occ_te, occ1_te, occ_te, region=region)["rms"]
            bmd = (Y1 - Y0).mean(dim=1)
            rmd, pct_md = expl(predict_meandelta(bmd, occ_te, s_te, e_te, res,
                                                 (H, W), crop), rp)

            A_ols = fit_operator(Y0, Y1, 0.0)
            r_ols, pct_ols = expl(predict_world(A_ols, occ_te, s_te, e_te, res,
                                                (H, W), crop), rp)
            A_ridge = fit_operator(Y0, Y1, args.ridge)
            r_ridge, pct_ridge = expl(predict_world(A_ridge, occ_te, s_te, e_te,
                                                    res, (H, W), crop), rp)
            pct_ridge_vs_md = 100 * r_ridge / rmd

            dt = time.time() - t
            print(f"{res:4d} {crop:5.2f} {D:6d} {M_tr/D:7.2f} {pct_ols:17.1f} "
                  f"{pct_ridge:19.1f} {pct_ridge_vs_md:21.1f} {cov:18.3f} {dt:6.1f}")


if __name__ == "__main__":
    main()
