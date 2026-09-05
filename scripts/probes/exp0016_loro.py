"""
EXP-0016 probe: leave-one-run-out at crop 0.5 (res 16, blur 1.0, bins 3) on
L040, to settle C-008 -- EXP-0015 found switched-nonneg beat single ridge1 by
+0.0069 explained (swept region) on ONE seed-0 split, but borrowed its noise
floor from a different design (~0.03, from linear_foresight_report.md's
sigma-1 blur/rms fold sweep). This probe MEASURES the floor on the same
design: 8 folds, one run held out each time (L040 has exactly 8 runs of 320
transitions each -- episode_ids IS the run index, confirmed empirically).

Reuses fit_linear_foresight.py's own functions verbatim (canonicalise,
contact_score, fit_operator, fit_operator_nonneg, predict_world, metrics,
swept_region_mask, actions_to_pixels) plus exp0009_rerun.py's mean-delta
predictor, so the code path matches EXP-0015/EXP-0009 exactly (no new
rasteriser, no new warp).

Usage
-----
    OMP_NUM_THREADS=4 python -u scripts/probes/exp0016_loro.py \
        configs/dataset/genesis_foresight_L040.yaml \
        --res 16 --crop 0.5 --blur 1.0 --bins 3 --ridge 1.0
"""
from __future__ import annotations

import argparse
import time

import torch

from dmdc_baseline import load_transition_arrays
from transforms.functional import (
    blend_push_prediction, from_push_frame, push_frame_validity_mask,
    to_push_frame,
)
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, contact_score, fit_operator,
    fit_operator_nonneg, metrics, predict_world, swept_region_mask,
    verify_pixel_mapping,
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
    ap.add_argument("--res", type=int, default=16)
    ap.add_argument("--crop", type=float, default=0.5)
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"=== EXP-0016 LORO: {args.dataset_cfg} res={args.res} "
          f"crop={args.crop} blur={args.blur} bins={args.bins} "
          f"ridge={args.ridge} ===")
    t0 = time.time()
    data = load_transition_arrays(args.dataset_cfg, split="train",
                                  max_samples=None)
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
    print(f"pixel-mapping check: median {d_mid:.2f} px from midpoint "
          f"(transposed/flipped scores >= 9.5 px here)")
    if not (d_mid < 6.0):
        print("  !! mapping looks wrong -- aborting")
        return

    ids = data.episode_ids.unique().tolist()
    print(f"runs (episode_ids): {ids}  -- {len(ids)} folds\n")

    R, CR, dev = args.res, args.crop, args.device
    D = R * R
    plate_px = 0.04 / 0.128 * W

    names3 = {2: ["low", "high"], 3: ["barely", "mildly", "significantly"]
              }.get(args.bins, [f"bin{i}" for i in range(args.bins)])

    fold_rows = []
    for run_id in ids:
        m_te = (data.episode_ids == run_id)
        m_tr = ~m_te

        occ_tr, occ1_tr = data.occ_t[m_tr].to(dev), data.occ_t1[m_tr].to(dev)
        occ_te, occ1_te = data.occ_t[m_te].to(dev), data.occ_t1[m_te].to(dev)
        s_tr, e_tr = start_px[m_tr].to(dev), end_px[m_tr].to(dev)
        s_te, e_te = start_px[m_te].to(dev), end_px[m_te].to(dev)

        region = swept_region_mask(s_te, e_te, (H, W),
                                   half_width_px=0.5 * plate_px + 2.0,
                                   pad_px=0.5 * plate_px)

        Y0 = canonicalise(occ_tr, s_tr, e_tr, R, CR).reshape(occ_tr.shape[0], -1).T
        Y1 = canonicalise(occ1_tr, s_tr, e_tr, R, CR).reshape(occ_tr.shape[0], -1).T
        M_tr = occ_tr.shape[0]

        def record_expl(pred):
            return metrics(pred, occ1_te, occ_te, region=region)["explained"]

        # persistence
        expl_persist = record_expl(occ_te)

        # mean-delta
        bmd = (Y1 - Y0).mean(dim=1)
        expl_meandelta = record_expl(
            predict_meandelta(bmd, occ_te, s_te, e_te, R, (H, W), CR))

        # single operator, ridge=args.ridge
        A_single = fit_operator(Y0, Y1, args.ridge)
        expl_single = record_expl(
            predict_world(A_single, occ_te, s_te, e_te, R, (H, W), CR))

        # switched operator, one nonneg fit per contact-score bin (edges from
        # this fold's TRAIN split only, exactly as fit_linear_foresight.py does)
        c_tr = contact_score(occ_tr, s_tr, e_tr, (H, W), plate_px)
        c_te = contact_score(occ_te, s_te, e_te, (H, W), plate_px)
        qs = torch.linspace(0, 1, args.bins + 1, device=dev)[1:-1]
        edges = torch.quantile(c_tr, qs)
        b_tr = torch.bucketize(c_tr, edges)
        b_te = torch.bucketize(c_te, edges)

        pred_sw = occ_te.clone()
        bin_info = []
        for b in range(args.bins):
            mtr, mte = (b_tr == b), (b_te == b)
            n_tr_b, n_te_b = int(mtr.sum()), int(mte.sum())
            bin_info.append((names3[b], n_tr_b, n_te_b))
            if n_tr_b < 10 or n_te_b == 0:
                continue
            Yb0 = canonicalise(occ_tr[mtr], s_tr[mtr], e_tr[mtr], R, CR
                               ).reshape(n_tr_b, -1).T
            Yb1 = canonicalise(occ1_tr[mtr], s_tr[mtr], e_tr[mtr], R, CR
                               ).reshape(n_tr_b, -1).T
            Ab = fit_operator_nonneg(Yb0, Yb1)
            pred_sw[mte] = predict_world(Ab, occ_te[mte], s_te[mte],
                                         e_te[mte], R, (H, W), CR)
        expl_switched = record_expl(pred_sw)

        diff = expl_switched - expl_single
        fold_rows.append(dict(run=int(run_id), switched=expl_switched,
                              single=expl_single, persistence=expl_persist,
                              mean_delta=expl_meandelta, diff=diff))
        print(f"fold run={int(run_id)}  M_tr={M_tr:4d} (M/D={M_tr/D:.2f})  "
              f"switched={expl_switched:+.4f}  single={expl_single:+.4f}  "
              f"persistence={expl_persist:+.4f}  mean_delta={expl_meandelta:+.4f}  "
              f"diff(sw-single)={diff:+.4f}   bins={bin_info}")

    diffs = torch.tensor([r["diff"] for r in fold_rows])
    n = len(diffs)
    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(unbiased=True))
    sem_d = sd_d / (n ** 0.5)

    print(f"\n=== paired per-fold diff (switched - single), n={n} folds ===")
    for r in fold_rows:
        print(f"  run {r['run']}: {r['diff']:+.4f}")
    print(f"\nmean paired diff   = {mean_d:+.4f}")
    print(f"sd across folds    = {sd_d:.4f}   <-- MEASURED noise floor")
    print(f"sem (sd/sqrt({n}))   = {sem_d:.4f}")
    print(f"mean / sem         = {mean_d / sem_d if sem_d > 0 else float('nan'):.2f}"
          f"  (>~2 => survives; EXP-0015's +0.0069 point estimate for reference)")


if __name__ == "__main__":
    main()
