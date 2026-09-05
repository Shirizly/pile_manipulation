"""
EXP-0009 probe: re-run the pixel-operator comparison (C-001, C-008) on the
now-fixed grid convention (`grid-convention`/`rasteriser-identity` fixed in
commit aac084e3).

Reuses fit_linear_foresight.py's own functions verbatim (canonicalise,
fit_operator, fit_operator_nonneg, predict_world, swept_region_mask, metrics,
contact_score, actions_to_pixels) so the code path is identical to the one the
original (invalidated) report and EXP-0001 used -- the only additions are:

  1. a mean-delta baseline (C-011: the baseline that matters), built with the
     exact same warp/blend pipeline `predict_world` uses;
  2. a configurable nonneg FISTA iteration cap, because fit_operator_nonneg's
     default max_iter=4000 costs O(D^3) per iteration and is infeasible at
     D=res*res=4096 (res=64): measured ~3.1 s/iter under this run's CPU
     contention -> 4000 iters would be ~3.4 h. This is a disclosed, measured
     deviation, not a silent shortcut (see the record's Threats section).

Usage
-----
    PYTHONPATH=. python scripts/probes/exp0009_rerun.py \
        configs/dataset/genesis_foresight_L040.yaml \
        --res 64 --crop 0.5 --blur 1.0 --nonneg-max-iter 4000 \
        --ridge-sweep 1e-2,1e-1,1,10,100,1000

    # contact-switched operator (C-008):
    PYTHONPATH=. python scripts/probes/exp0009_rerun.py \
        configs/dataset/genesis_foresight_L040.yaml \
        --res 16 --crop 0.25 --blur 1.0 --bins 3
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
    actions_to_pixels, canonicalise, contact_score, fit_operator,
    fit_operator_nonneg, metrics, predict_world, swept_region_mask,
    verify_pixel_mapping,
)


def predict_meandelta(bmd, occ, start_px, end_px, res, grid_res, scale=1.0,
                       batch=256):
    """Same warp/apply/unwarp/blend pipeline as predict_world, but the
    "operator" is a constant additive canonical-frame vector rather than a
    matrix -- this IS the mean-delta baseline (C-011), round-tripped through
    the identical pipeline so it pays the same warp cost every other model
    pays."""
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
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--crop", type=float, default=1.0)
    ap.add_argument("--bins", type=int, default=0)
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--ridge-sweep", default="1e-2")
    ap.add_argument("--nonneg-max-iter", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"=== EXP-0009 rerun: {args.dataset_cfg} res={args.res} "
          f"crop={args.crop} blur={args.blur} bins={args.bins} ===")
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

    torch.manual_seed(args.split_seed)
    m_tr, m_te = split_by_episode(data, holdout_frac=args.holdout_frac)
    n_runs = int(data.episode_ids.unique().numel())
    print(f"split: {int(m_tr.sum())} train / {int(m_te.sum())} test "
          f"({n_runs} runs total)")

    dev = args.device
    occ_tr, occ1_tr = data.occ_t[m_tr].to(dev), data.occ_t1[m_tr].to(dev)
    occ_te, occ1_te = data.occ_t[m_te].to(dev), data.occ_t1[m_te].to(dev)
    s_tr, e_tr = start_px[m_tr].to(dev), end_px[m_tr].to(dev)
    s_te, e_te = start_px[m_te].to(dev), end_px[m_te].to(dev)

    R, CR = args.res, args.crop
    D = R * R
    M_tr = occ_tr.shape[0]
    print(f"\ncanonicalising ({R}x{R}, crop={CR}) ... D={D}, M_train={M_tr}, "
          f"M/D={M_tr / D:.2f}")
    Y0 = canonicalise(occ_tr, s_tr, e_tr, R, CR).reshape(M_tr, -1).T
    Y1 = canonicalise(occ1_tr, s_tr, e_tr, R, CR).reshape(M_tr, -1).T

    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_te, e_te, (H, W),
                               half_width_px=0.5 * plate_px + 2.0,
                               pad_px=0.5 * plate_px)
    print(f"swept-region mask covers {100 * float(region.mean()):.1f}% of "
          f"the grid")

    results, results_sw = {}, {}

    def record(name, pred):
        results[name] = metrics(pred, occ1_te, occ_te)
        results_sw[name] = metrics(pred, occ1_te, occ_te, region=region)

    record("persistence", occ_te)
    eye = torch.eye(D, device=dev)
    record("identity(warp only)",
           predict_world(eye, occ_te, s_te, e_te, R, (H, W), CR))
    bmd = (Y1 - Y0).mean(dim=1)
    record("mean-delta", predict_meandelta(bmd, occ_te, s_te, e_te, R,
                                           (H, W), CR))

    fits = [("ols", lambda: fit_operator(Y0, Y1, 0.0))]
    for lam in [float(x) for x in args.ridge_sweep.split(",") if x.strip()]:
        fits.append((f"ridge{lam:g}", lambda lam=lam: fit_operator(Y0, Y1, lam)))

    for name, fit in fits:
        t = time.time()
        A = fit()
        dt = time.time() - t
        pred = predict_world(A, occ_te, s_te, e_te, R, (H, W), CR)
        record(name, pred)
        print(f"  fit {name:>10s}: {dt:.1f}s")

    t = time.time()
    A = fit_operator_nonneg(Y0, Y1, max_iter=args.nonneg_max_iter,
                            report=True)
    dt = time.time() - t
    pred = predict_world(A, occ_te, s_te, e_te, R, (H, W), CR)
    record("nonneg", pred)
    print(f"  fit {'nonneg':>10s}: {dt:.1f}s ({args.nonneg_max_iter} iters cap)")

    # ---- switched operator: one per contact-score bin --------------------
    if args.bins > 1:
        c_tr = contact_score(occ_tr, s_tr, e_tr, (H, W), plate_px)
        c_te = contact_score(occ_te, s_te, e_te, (H, W), plate_px)
        qs = torch.linspace(0, 1, args.bins + 1, device=dev)[1:-1]
        edges = torch.quantile(c_tr, qs)
        b_tr = torch.bucketize(c_tr, edges)
        b_te = torch.bucketize(c_te, edges)
        names = ({2: ["low", "high"],
                  3: ["barely", "mildly", "significantly"]}
                 .get(args.bins, [f"bin{i}" for i in range(args.bins)]))
        print(f"\ncontact-score bins (train quantile edges "
              f"{[round(float(x), 2) for x in edges]}):")
        pred_sw = occ_te.clone()
        for b in range(args.bins):
            mtr, mte = (b_tr == b), (b_te == b)
            n_tr_b, n_te_b = int(mtr.sum()), int(mte.sum())
            print(f"  {names[b]:>14s}: {n_tr_b:5d} train / {n_te_b:4d} test "
                  f"(M/D={n_tr_b / D:.2f})")
            if n_tr_b < 10 or n_te_b == 0:
                print("    too few samples, skipped")
                continue
            Yb0 = canonicalise(occ_tr[mtr], s_tr[mtr], e_tr[mtr], R, CR
                               ).reshape(n_tr_b, -1).T
            Yb1 = canonicalise(occ1_tr[mtr], s_tr[mtr], e_tr[mtr], R, CR
                               ).reshape(n_tr_b, -1).T
            Ab = fit_operator_nonneg(Yb0, Yb1, max_iter=args.nonneg_max_iter)
            pred_sw[mte] = predict_world(Ab, occ_te[mte], s_te[mte],
                                         e_te[mte], R, (H, W), CR)
        record("switched-nonneg", pred_sw)

    print("\n=== whole-image ===")
    for name, m in results.items():
        print(f"  {name:>20s}  rms={m['rms']:.5f}  "
              f"pct_persist={100 * m['rms'] / results['persistence']['rms']:.1f}%  "
              f"pct_meandelta={100 * m['rms'] / results['mean-delta']['rms']:.1f}%")

    print("\n=== swept region ===")
    rp = results_sw["persistence"]["rms"]
    rm = results_sw["mean-delta"]["rms"]
    for name, m in results_sw.items():
        print(f"  {name:>20s}  rms={m['rms']:.5f}  "
              f"pct_persist={100 * m['rms'] / rp:.1f}%  "
              f"pct_meandelta={100 * m['rms'] / rm:.1f}%  "
              f"explained_vs_persist={1 - m['rms'] / rp:+.3f}  "
              f"explained_vs_meandelta={1 - m['rms'] / rm:+.3f}")


if __name__ == "__main__":
    main()
