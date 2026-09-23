"""EXP-0022 A3 -- wall-proximity stratification: is the warped NFD's deficit
concentrated near workspace walls, or flat?

Compares two checkpoints, on the SAME transitions, per-transition swept-region
`accuracy` (fit_linear_foresight.metrics's row-wise formula, kept per-row
instead of averaged):

    baseline : world-frame NFD,  Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth
    warped   : push-frame NFD,   Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth (RUN-0005)

Primary corpus: overnight_randlen_test (widest push-length/position range).
Optionally also L20mm/L40mm eval cells if time allows (--corpora flag).

Wall-proximity statistic (PRIMARY, defined precisely because the task asks for
exactly one): the Euclidean-axis-aligned distance from the push END point
(p_stop = (ex, ey), world metres -- where the plate is at the second action
render, i.e. where material gets shoved to) to the NEAREST workspace boundary:

    d_wall = min(ex - x_min, x_max - ex, ey - y_min, y_max - ey)

Why the END point and not the swept-rectangle nearest-boundary distance or a
margin-fraction statistic: it is the simplest statistic that is monotonically
related to "how much of the push's aftermath sits near a wall", requires only
the action tuple already loaded (no plate-half-width geometry, no swept-mask
integration), and is exactly the point most likely to interact with a wall
after a push directed toward one. It does NOT subtract the plate's own
half-width, so it is a slightly optimistic (upward-biased) proxy for the true
minimum clearance of the swept rectangle -- stated as a simplification, not
hidden.

Confound guard: also stratifies by push length (mm), and by wall-proximity
WITHIN one push-length band, per the task brief -- wall proximity plausibly
correlates with push length (a push that runs to the wall is long; a push
that's cut short by contact is a separate confound already flagged in
PLAN.md as "early contact").

Runs entirely on CPU (device asserted below); does not train; does not touch
Baselines/common/eval_report.py.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import torch

from Baselines.common.randlen_data import load_randlen_cell
from Baselines.common.data import load_cell
from Baselines.NFD.predictor import NFDPredictor
from model.warped_nfd.predictor import WarpedNFDPredictor
from fit_linear_foresight import swept_region_mask
from transforms.functional import to_push_frame, push_frame_roundtrip

DEVICE = "cpu"
torch.set_num_threads(4)

BASELINE_CKPT = "Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth"
WARPED_CKPT = "Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth"

CELLS = {
    "randlen_test": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_all.yaml",
    ),
    "L20mm": dict(
        kind="slate",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json",
    ),
    "L40mm": dict(
        kind="slate",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L40mm/manifest.json",
    ),
}


def load_one_cell(name):
    spec = CELLS[name]
    if spec["kind"] == "randlen":
        return load_randlen_cell(spec["cfg"], "train", tag=name, need_step_idx=True)
    return load_cell(spec["eval_cfg"], "train", manifest_path=spec["manifest"], tag=name)


def run_predictor(predictor, cell, chunk=256):
    n = cell.occ0.shape[0]
    outs = []
    dev_seen = set()
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        batch = SimpleNamespace(
            occ0=cell.occ0[sl].to(DEVICE), p_start=cell.p_start[sl].to(DEVICE),
            p_stop=cell.p_stop[sl].to(DEVICE), angle=cell.angle[sl].to(DEVICE),
            raw=cell.raw, H=cell.H, W=cell.W,
        )
        pred = predictor.predict_occ(batch)
        dev_seen.add(str(pred.device))
        outs.append(pred)
    return torch.cat(outs, dim=0), dev_seen


def run_predictor_canonical(predictor, cell, chunk=256):
    """Native canonical-frame prediction, per-chunk, for a predictor exposing
    `predict_occ_canonical` (only `WarpedNFDPredictor`). Mirrors
    `eval_report.py::_accuracy_canonical`'s native branch exactly (same
    `canon_res=H`/`scale=1.0` convention), just batched over chunks instead
    of the whole corpus at once, for memory parity with `run_predictor`.
    Returns `(pred_canon [N,res,res], start_px [N,2], end_px [N,2])` --
    start_px/end_px are the predictor's OWN pixel derivation, to be checked
    against `actions_to_pixels`' derivation by the caller (RUN-0011's parity
    assert), not assumed equal here."""
    n = cell.occ0.shape[0]
    preds, starts, ends = [], [], []
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        batch = SimpleNamespace(
            occ0=cell.occ0[sl].to(DEVICE), p_start=cell.p_start[sl].to(DEVICE),
            p_stop=cell.p_stop[sl].to(DEVICE), angle=cell.angle[sl].to(DEVICE),
            raw=cell.raw, H=cell.H, W=cell.W,
        )
        pred_canon, s_px, e_px, canon_res, scale = predictor.predict_occ_canonical(batch)
        preds.append(pred_canon)
        starts.append(s_px)
        ends.append(e_px)
    return torch.cat(preds, dim=0), torch.cat(starts, dim=0), torch.cat(ends, dim=0), canon_res, scale


def run_predictor_degraded_input(baseline, warped_for_pixels, cell, chunk=256):
    """RUN-0014 control: feed the WORLD-FRAME BASELINE an `occ0` that has
    already suffered the same warp-and-back round trip the warped model's
    canonical `occ0` input effectively suffers, then run the UNMODIFIED
    baseline checkpoint on it. Only `occ0` is degraded -- the two action
    (plate) channels the baseline builds from `batch.p_start`/`p_stop` are
    left untouched, so this isolates "the input was resampled" from "the
    model predicts in a rotated frame."

    Pixel derivation: reuses `warped_for_pixels._build_fn`'s NATIVE
    start_px/end_px (the exact ones `WarpedNFDPredictor` uses to warp its
    own `occ0` -- see WARPED_NFD_NOTES.md's (col,row) vs (row,col) trap),
    not `actions_to_pixels`' world derivation, so the degradation applied
    here is bit-for-bit the transform the warped model's input actually
    goes through, not a re-derived approximation of it.

    `push_frame_roundtrip` with `fn = identity`, `canon_res` = the warped
    model's own (batch's H, since `WARPED_CKPT` was trained with
    `canon_res=None` -> batch resolution), `scale=1.0`, `blend=True`
    (default) -- one warp in, identity, one warp back, blended with the
    original `occ0` outside the round trip's validity mask, exactly
    `push_frame_roundtrip`'s documented contract.
    """
    n = cell.occ0.shape[0]
    outs = []
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        batch = SimpleNamespace(
            occ0=cell.occ0[sl].to(DEVICE), p_start=cell.p_start[sl].to(DEVICE),
            p_stop=cell.p_stop[sl].to(DEVICE), angle=cell.angle[sl].to(DEVICE),
            raw=cell.raw, H=cell.H, W=cell.W,
        )
        _, occ0, start_px, end_px, canon_res = warped_for_pixels._build_fn(batch)
        occ0_degraded = push_frame_roundtrip(
            lambda c: c, occ0, start_px, end_px, canon_res, scale=1.0, blend=True)
        batch2 = SimpleNamespace(
            occ0=occ0_degraded, p_start=batch.p_start, p_stop=batch.p_stop,
            angle=batch.angle, raw=batch.raw, H=batch.H, W=batch.W,
        )
        pred = baseline.predict_occ(batch2)
        outs.append(pred)
    return torch.cat(outs, dim=0)


def per_transition_errors(pred, truth, occ_prev, region):
    """Row-wise RMS prediction error and row-wise RMS persistence error, in
    the swept region -- the two ingredients of fit_linear_foresight.metrics()
    ['accuracy'] = 1 - mean(err_pred)/mean(err_pers), kept UN-RATIOED per row.

    IMPORTANT: `accuracy` is a ratio of population means, not a mean of a
    per-row ratio. A per-row ratio err_pred_i/err_pers_i blows up whenever a
    single row's persistence error is near zero (e.g. a push that barely
    moved anything in the swept band) -- found empirically here: taking the
    ratio per row before aggregating produced accuracy values in the
    millions. Aggregating err_pred and err_pers SEPARATELY first (as
    `metrics()` does at the whole-population level) and dividing the MEANS
    is the numerically stable form used everywhere else in this repo; this
    function returns the two per-row ingredients so that aggregation, and
    only that aggregation, ever forms the ratio.
    """
    n = pred.shape[0]
    w = region.reshape(n, -1)
    npix = w.sum(dim=1).clamp_min(1.0)
    d = (pred - truth) * region
    flat = d.reshape(n, -1)
    tr_ = (truth.reshape(n, -1) * w)
    pv = (occ_prev.reshape(n, -1) * w)
    err_pred = (flat.pow(2).sum(dim=1) / npix).sqrt()
    err_pers = ((tr_ - pv).pow(2).sum(dim=1) / npix).sqrt()
    return err_pred, err_pers


def accuracy_from_errors(err_pred, err_pers):
    return 1.0 - float(err_pred.mean()) / float(err_pers.mean().clamp_min(1e-9))


def bootstrap_delta(err_pred_base, err_pred_warp, err_pers, n_boot=2000, seed=0):
    """Bootstrap the warped-minus-baseline accuracy delta over rows (with
    replacement), returning (mean, sem, ci_lo, ci_hi [2.5/97.5 pct]).
    `accuracy` is a ratio of means, so its sampling distribution is bootstrapped
    directly rather than approximated by a per-row-ratio SEM (see
    `per_transition_errors`'s docstring for why the naive per-row ratio is
    unstable in the first place)."""
    n = err_pers.shape[0]
    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    ep_b, ep_w, ers = err_pred_base[idx], err_pred_warp[idx], err_pers[idx]
    acc_b = 1.0 - ep_b.mean(dim=1) / ers.mean(dim=1).clamp_min(1e-9)
    acc_w = 1.0 - ep_w.mean(dim=1) / ers.mean(dim=1).clamp_min(1e-9)
    d = acc_w - acc_b
    lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975]))
    return float(d.mean()), float(d.std(unbiased=True)), float(lo), float(hi)


def wall_distance_mm(actions, ws_min, ws_max):
    x_min, y_min = float(ws_min[0]), float(ws_min[1])
    x_max, y_max = float(ws_max[0]), float(ws_max[1])
    ex, ey = actions[:, 2], actions[:, 3]
    d = torch.minimum(torch.minimum(ex - x_min, x_max - ex),
                       torch.minimum(ey - y_min, y_max - ey))
    return d * 1000.0


def push_length_mm(actions):
    sx, sy, ex, ey = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3]
    return torch.hypot(ex - sx, ey - sy) * 1000.0


def quantile_bin(x, n_bins):
    qs = torch.linspace(0, 1, n_bins + 1)
    edges = torch.quantile(x, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    idx = torch.bucketize(x, edges[1:-1])
    return idx, edges


def strat_table(err_pred_base, err_pred_warp, err_pers, bin_idx, n_bins, stat, stat_name):
    rows = []
    for b in range(n_bins):
        m = bin_idx == b
        n = int(m.sum())
        if n == 0:
            rows.append(dict(bin=b, n=0))
            continue
        acc_b = accuracy_from_errors(err_pred_base[m], err_pers[m])
        acc_w = accuracy_from_errors(err_pred_warp[m], err_pers[m])
        d_mean, d_sem, d_lo, d_hi = bootstrap_delta(err_pred_base[m], err_pred_warp[m], err_pers[m])
        rows.append(dict(
            bin=b, n=n,
            stat_lo=float(stat[m].min()), stat_hi=float(stat[m].max()),
            stat_mean=float(stat[m].mean()),
            baseline_acc=acc_b, warped_acc=acc_w,
            delta_mean=d_mean, delta_boot_sem=d_sem, delta_ci_lo=d_lo, delta_ci_hi=d_hi,
        ))
    return rows


def print_table(rows, stat_name):
    print(f"\n  bin   n     {stat_name:>16s}      base_acc  warp_acc   delta       boot_sem  95%CI")
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['bin']:>3d}   0     (empty)")
            continue
        print(f"  {r['bin']:>3d}  {r['n']:>4d}   [{r['stat_lo']:7.1f},{r['stat_hi']:7.1f}] "
              f"mean={r['stat_mean']:7.1f}   {r['baseline_acc']:8.4f}  {r['warped_acc']:8.4f}  "
              f"{r['delta_mean']:+8.4f}  {r['delta_boot_sem']:8.4f}  "
              f"[{r['delta_ci_lo']:+.4f},{r['delta_ci_hi']:+.4f}]")


def bootstrap_delta3(err_pred_base, err_pred_degraded, err_pred_warp, err_pers,
                      n_boot=2000, seed=0):
    """Same bootstrap as `bootstrap_delta`, but over three arms at once so
    all three accuracies and both deltas (degraded-base, warp-base) share the
    same resample indices, and the fraction-of-deficit-accounted statistic
    (degraded delta / warp delta) is computed per-resample, not from the
    point estimates alone."""
    n = err_pers.shape[0]
    if n < 4:
        nan = float("nan")
        return dict(acc_b=nan, acc_d=nan, acc_w=nan,
                     d_deg=(nan, nan, nan, nan), d_warp=(nan, nan, nan, nan),
                     frac=(nan, nan, nan, nan))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    ep_b = err_pred_base[idx].mean(dim=1)
    ep_d = err_pred_degraded[idx].mean(dim=1)
    ep_w = err_pred_warp[idx].mean(dim=1)
    ers = err_pers[idx].mean(dim=1).clamp_min(1e-9)
    acc_b, acc_d, acc_w = 1.0 - ep_b / ers, 1.0 - ep_d / ers, 1.0 - ep_w / ers
    d_deg, d_warp = acc_d - acc_b, acc_w - acc_b
    # fraction of the warped model's deficit the degraded-input baseline
    # accounts for; only defined where the warped deficit is non-trivial.
    frac = d_deg / d_warp.where(d_warp.abs() > 1e-6, torch.full_like(d_warp, float("nan")))

    def summarize(x):
        lo, hi = torch.quantile(x[~x.isnan()], torch.tensor([0.025, 0.975])) \
            if (~x.isnan()).any() else (float("nan"), float("nan"))
        return float(x[~x.isnan()].mean()) if (~x.isnan()).any() else float("nan"), \
            float(x[~x.isnan()].std(unbiased=True)) if (~x.isnan()).any() else float("nan"), \
            float(lo), float(hi)

    return dict(
        acc_b=float(acc_b.mean()), acc_d=float(acc_d.mean()), acc_w=float(acc_w.mean()),
        d_deg=summarize(d_deg), d_warp=summarize(d_warp), frac=summarize(frac),
    )


def strat_table3(err_pred_base, err_pred_degraded, err_pred_warp, err_pers,
                  bin_idx, n_bins, stat, stat_name):
    rows = []
    for b in range(n_bins):
        m = bin_idx == b
        n = int(m.sum())
        if n == 0:
            rows.append(dict(bin=b, n=0))
            continue
        s = bootstrap_delta3(err_pred_base[m], err_pred_degraded[m], err_pred_warp[m], err_pers[m])
        rows.append(dict(
            bin=b, n=n,
            stat_lo=float(stat[m].min()), stat_hi=float(stat[m].max()),
            stat_mean=float(stat[m].mean()),
            **s,
        ))
    return rows


def print_table3(rows):
    print(f"\n  bin   n     len mm (mean)   base_acc  degr_acc  warp_acc   "
          f"d_degraded (95%CI)          d_warped (95%CI)            frac_of_deficit (95%CI)")
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['bin']:>3d}   0     (empty)")
            continue
        d_deg, d_warp, frac = r["d_deg"], r["d_warp"], r["frac"]
        print(f"  {r['bin']:>3d}  {r['n']:>4d}   {r['stat_mean']:9.1f}      "
              f"{r['acc_b']:8.4f}  {r['acc_d']:8.4f}  {r['acc_w']:8.4f}   "
              f"{d_deg[0]:+.4f} [{d_deg[2]:+.4f},{d_deg[3]:+.4f}]   "
              f"{d_warp[0]:+.4f} [{d_warp[2]:+.4f},{d_warp[3]:+.4f}]   "
              f"{frac[0]:+.3f} [{frac[2]:+.3f},{frac[3]:+.3f}]")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="randlen_test")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--frame", choices=["world", "canonical"], default="world",
                     help="RUN-0013: score accuracy in the WORLD frame (RUN-0012's original "
                          "behaviour, default, unchanged) or in the CANONICAL push frame "
                          "(EXP-0022 A1's convention: nfd_warped_randlen scored natively via "
                          "predict_occ_canonical, zero extra resamplings; nfd_randlen's "
                          "already-computed world prediction warped ONCE via to_push_frame, "
                          "same asymmetry eval_report.py::_accuracy_canonical documents).")
    ap.add_argument("--out-prefix",
                     default="experiments/EXP-0022-warped-nfd-push-frame/artifacts/RUN-0012-wall-proximity/wall_proximity")
    ap.add_argument("--degraded-baseline", action="store_true",
                     help="RUN-0014: add a third arm -- the world-frame baseline run on an "
                          "occ0 that has been round-tripped through the warped model's own "
                          "warp-and-back (push_frame_roundtrip, identity fn), to isolate input "
                          "resampling from the push-frame prediction itself. World frame only "
                          "(--frame canonical is not supported with this flag).")
    ap.add_argument("--warped-ckpt", default=WARPED_CKPT,
                     help="RUN-0018/0020: which warped checkpoint to stratify. Defaults to "
                          "RUN-0005 (nfd_warped_randlen), the arm RUN-0012/0013 measured. Point "
                          "it at RUN-0010 (nfd_warped_randlen_flipaug) to test PLAN.md's standing "
                          "hypothesis that the short-push deficit is partly the x8-augmentation "
                          "handicap itself and should FLATTEN under flip-only augmentation.")
    ap.add_argument("--warped-name", default="nfd_warped_randlen",
                     help="label for the warped arm in the output JSON.")
    args = ap.parse_args()
    if args.degraded_baseline:
        assert args.frame == "world", "--degraded-baseline only supported with --frame world"

    print(f"loading baseline predictor: {BASELINE_CKPT}")
    baseline = NFDPredictor(BASELINE_CKPT, channels=3, name="nfd_baseline_world")
    print(f"loading warped predictor:   {args.warped_ckpt}")
    warped = WarpedNFDPredictor(args.warped_ckpt, plate_mode="canonical", wall_channel=False,
                                 canon_res=None, scale=1.0, name=args.warped_name)
    for m, nm in [(baseline.model, "baseline"), (warped.model, "warped")]:
        dev = next(m.parameters()).device
        print(f"  {nm} model params on device: {dev}")
        assert str(dev) == "cpu", f"{nm} model unexpectedly on {dev}, must be CPU"

    all_out = {}
    for corpus in args.corpora.split(","):
        corpus = corpus.strip()
        print(f"\n=== corpus: {corpus} ===")
        t0 = time.time()
        cell = load_one_cell(corpus)
        n = cell.occ0.shape[0]
        print(f"  loaded {n} transitions, grid {cell.H}x{cell.W}, {time.time()-t0:.1f}s")

        t0 = time.time()
        pred_base, dev_base = run_predictor(baseline, cell)
        print(f"  baseline predict: {time.time()-t0:.1f}s, devices seen: {dev_base}")
        t0 = time.time()
        pred_warp, dev_warp = run_predictor(warped, cell)
        print(f"  warped   predict: {time.time()-t0:.1f}s, devices seen: {dev_warp}")

        pred_base_degraded = None
        if args.degraded_baseline:
            t0 = time.time()
            pred_base_degraded = run_predictor_degraded_input(baseline, warped, cell)
            print(f"  degraded-input baseline predict: {time.time()-t0:.1f}s")

        from fit_linear_foresight import actions_to_pixels
        start_px, end_px = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max,
                                              (cell.H, cell.W))
        plate_px = 0.04 / 0.128 * cell.W
        region = swept_region_mask(start_px, end_px, (cell.H, cell.W),
                                    half_width_px=0.5 * plate_px + 2.0, pad_px=0.5 * plate_px)

        occ1 = cell.occ1.to(torch.float32)
        occ0 = cell.occ0.to(torch.float32)

        if args.frame == "world":
            pred_base_scored, pred_warp_scored = pred_base, pred_warp
            truth_scored, prev_scored, region_scored = occ1, occ0, region
        else:
            # RUN-0013 (A1 x A3): score in the CANONICAL push frame instead,
            # exactly eval_report.py::_accuracy_canonical's convention --
            # nfd_warped_randlen (native) via predict_occ_canonical, zero
            # extra resamplings; nfd_randlen's world prediction warped ONCE.
            canon_res = cell.H
            scale = 1.0
            pred_warp_canon, s_px_n, e_px_n, canon_res, scale = run_predictor_canonical(
                warped, cell)
            max_px_diff = float((s_px_n - start_px).abs().max())
            assert max_px_diff < 1.0, (
                f"canonical-frame pixel convention mismatch: {max_px_diff:.3f}px "
                f"(native predict_occ_canonical vs actions_to_pixels) -- would "
                f"silently misalign the canonical truth/region")
            # BUG FOUND HERE (RUN-0013, debugged against eval_report.py's
            # 0.4536 reference): even though max_px_diff < 1px, the world
            # (actions_to_pixels) convention and the native
            # (predict_occ_canonical) convention differ by a SYSTEMATIC
            # ~0.5px offset (not noise) -- using the world start_px/end_px to
            # warp truth/region/prev instead of the native s_px_n/e_px_n
            # dropped accuracy_canonical from 0.452 to 0.295 on this exact
            # corpus/model. eval_report.py::_accuracy_canonical reassigns
            # s_px,e_px = s_px_n,e_px_n for exactly this reason -- do the same.
            start_px, end_px = s_px_n, e_px_n
            pred_base_canon = to_push_frame(pred_base.to(torch.float32), start_px, end_px,
                                             (canon_res, canon_res), scale)
            region_canon = (to_push_frame(region.to(torch.float32), start_px, end_px,
                                           (canon_res, canon_res), scale) > 0.5).to(torch.float32)
            truth_canon = to_push_frame(occ1, start_px, end_px, (canon_res, canon_res), scale)
            prev_canon = to_push_frame(occ0, start_px, end_px, (canon_res, canon_res), scale)
            pred_base_scored, pred_warp_scored = pred_base_canon, pred_warp_canon
            truth_scored, prev_scored, region_scored = truth_canon, prev_canon, region_canon

        err_pred_base, err_pers_base = per_transition_errors(
            pred_base_scored, truth_scored, prev_scored, region_scored)
        err_pred_warp, err_pers_warp = per_transition_errors(
            pred_warp_scored, truth_scored, prev_scored, region_scored)
        # err_pers is identical for both models (same occ0/occ1/region) up to
        # float noise -- assert rather than silently using two slightly
        # different denominators.
        assert torch.allclose(err_pers_base, err_pers_warp, atol=1e-5), \
            "persistence error differs between the two model passes -- region/occ mismatch"
        err_pers = err_pers_base

        err_pred_degraded = None
        if pred_base_degraded is not None:
            # world frame only (asserted earlier); scored against the same
            # truth/prev/region as pred_base_scored / pred_warp_scored above.
            err_pred_degraded, err_pers_degraded = per_transition_errors(
                pred_base_degraded.to(torch.float32), truth_scored, prev_scored, region_scored)
            assert torch.allclose(err_pers_degraded, err_pers, atol=1e-5), \
                "persistence error differs for the degraded-input pass -- region/occ mismatch"

        d_wall = wall_distance_mm(cell.actions, cell.workspace_min, cell.workspace_max)
        p_len = push_length_mm(cell.actions)

        print(f"  wall-distance mm: min {float(d_wall.min()):.1f} p10 {float(d_wall.quantile(0.1)):.1f} "
              f"p50 {float(d_wall.quantile(0.5)):.1f} p90 {float(d_wall.quantile(0.9)):.1f} "
              f"max {float(d_wall.max()):.1f}")
        print(f"  push-length  mm: min {float(p_len.min()):.1f} p10 {float(p_len.quantile(0.1)):.1f} "
              f"p50 {float(p_len.quantile(0.5)):.1f} p90 {float(p_len.quantile(0.9)):.1f} "
              f"max {float(p_len.max()):.1f}")
        corr = float(torch.corrcoef(torch.stack([d_wall, p_len]))[0, 1])
        print(f"  Pearson corr(wall_distance_mm, push_length_mm) = {corr:.3f}")

        n_bins = args.n_bins
        wall_bin, wall_edges = quantile_bin(d_wall, n_bins)
        len_bin, len_edges = quantile_bin(p_len, n_bins)

        print(f"\n  --- stratified by WALL DISTANCE (mm), {n_bins} quantile bins ---")
        wall_rows = strat_table(err_pred_base, err_pred_warp, err_pers, wall_bin, n_bins, d_wall, "wall_dist_mm")
        print_table(wall_rows, "wall_dist_mm")

        print(f"\n  --- stratified by PUSH LENGTH (mm), {n_bins} quantile bins (confound check) ---")
        len_rows = strat_table(err_pred_base, err_pred_warp, err_pers, len_bin, n_bins, p_len, "push_len_mm")
        print_table(len_rows, "push_len_mm")

        len_rows_3way = None
        if err_pred_degraded is not None:
            print(f"\n  --- RUN-0014 3-WAY, stratified by PUSH LENGTH (mm), {n_bins} quantile "
                  f"bins (SAME bins as above -- baseline-clean / baseline-degraded-input / "
                  f"warped) ---")
            len_rows_3way = strat_table3(err_pred_base, err_pred_degraded, err_pred_warp,
                                          err_pers, len_bin, n_bins, p_len, "push_len_mm")
            print_table3(len_rows_3way)

        # within-band wall-proximity stratification: restrict to the MIDDLE
        # push-length quantile bin (band 2 of 0..n_bins-1, i.e. the median
        # band), then re-stratify by wall distance within it.
        mid_band = n_bins // 2
        m_band = len_bin == mid_band
        if int(m_band.sum()) == 0:
            # Degenerate length distribution (e.g. L20mm/L40mm cells are a
            # single nominal push length, so quantile edges collide and one
            # bucket can land empty) -- fall back to the most populated bin.
            counts = torch.tensor([int((len_bin == b).sum()) for b in range(n_bins)])
            mid_band = int(counts.argmax())
            m_band = len_bin == mid_band
        band_desc = (f"range [{float(p_len[m_band].min()):.1f},{float(p_len[m_band].max()):.1f}] mm"
                     if int(m_band.sum()) else "EMPTY")
        print(f"\n  --- WITHIN push-length band {mid_band} (n={int(m_band.sum())}, "
              f"{band_desc}), re-stratified by wall distance ---")
        if int(m_band.sum()) >= 3 * n_bins:
            wb_bin, _ = quantile_bin(d_wall[m_band], n_bins)
            wb_rows = strat_table(err_pred_base[m_band], err_pred_warp[m_band], err_pers[m_band],
                                   wb_bin, n_bins, d_wall[m_band], "wall_dist_mm")
            print_table(wb_rows, "wall_dist_mm")
        else:
            wb_rows = []
            print("    too few rows in this band, skipped")

        overall_acc_base = accuracy_from_errors(err_pred_base, err_pers)
        overall_acc_warp = accuracy_from_errors(err_pred_warp, err_pers)
        od_mean, od_sem, od_lo, od_hi = bootstrap_delta(err_pred_base, err_pred_warp, err_pers)
        print(f"\n  overall: baseline_acc={overall_acc_base:.4f} "
              f"warped_acc={overall_acc_warp:.4f} "
              f"delta={od_mean:+.4f} boot_sem={od_sem:.4f} 95%CI=[{od_lo:+.4f},{od_hi:+.4f}]")

        overall_acc_degraded = None
        overall_3way = None
        if err_pred_degraded is not None:
            overall_acc_degraded = accuracy_from_errors(err_pred_degraded, err_pers)
            overall_3way = bootstrap_delta3(err_pred_base, err_pred_degraded, err_pred_warp, err_pers)
            print(f"  overall (RUN-0014): baseline_clean_acc={overall_acc_base:.4f} "
                  f"baseline_degraded_acc={overall_acc_degraded:.4f} warped_acc={overall_acc_warp:.4f} "
                  f"d_degraded={overall_3way['d_deg'][0]:+.4f} d_warped={overall_3way['d_warp'][0]:+.4f} "
                  f"frac_of_deficit={overall_3way['frac'][0]:+.3f}")

        all_out[corpus] = dict(
            n=n, frame=args.frame, corr_wall_len=corr,
            wall_dist_mm_quantiles={str(round(float(q), 2)): float(v) for q, v in
                                     zip([0, .1, .25, .5, .75, .9, 1], torch.quantile(
                                         d_wall, torch.tensor([0, .1, .25, .5, .75, .9, 1])))},
            push_len_mm_quantiles={str(round(float(q), 2)): float(v) for q, v in
                                    zip([0, .1, .25, .5, .75, .9, 1], torch.quantile(
                                        p_len, torch.tensor([0, .1, .25, .5, .75, .9, 1])))},
            wall_bins=wall_rows, len_bins=len_rows, within_band_wall_bins=wb_rows,
            mid_band_len_range=[float(p_len[m_band].min()), float(p_len[m_band].max())] if int(m_band.sum()) else None,
            overall_delta_mean=od_mean, overall_delta_boot_sem=od_sem,
            overall_delta_ci=[od_lo, od_hi],
            overall_baseline_acc=overall_acc_base, overall_warped_acc=overall_acc_warp,
            overall_baseline_degraded_acc=overall_acc_degraded, overall_3way=overall_3way,
            len_bins_3way=len_rows_3way,
            per_row=dict(
                d_wall_mm=d_wall.tolist(), push_len_mm=p_len.tolist(),
                err_pred_baseline=err_pred_base.tolist(), err_pred_warped=err_pred_warp.tolist(),
                err_persistence=err_pers.tolist(),
                **({"err_pred_baseline_degraded": err_pred_degraded.tolist()}
                   if err_pred_degraded is not None else {}),
            ),
        )

    out_path = f"{args.out_prefix}.json"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_out, f)
    print(f"\nwrote raw output -> {out_path}")


if __name__ == "__main__":
    main()
