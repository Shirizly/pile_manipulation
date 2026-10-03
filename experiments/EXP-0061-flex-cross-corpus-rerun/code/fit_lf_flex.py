"""EXP-0061 Task B -- switched-linear visual foresight on FleX image masks.

Fit on DS-0020 TRAIN (trajectory split, splits.json), select the ridge lambda
(toward identity) on DS-0020 VAL (the SAME trajectory-level split -- whole
trajectories held out, never row-random), EXP-0003 bin scheme (6 equal-width
push-length bins over [0, max train length], MIN_ROWS_PER_BIN = 50 else
identity), plus one single (unswitched) operator with its own lambda.
Input and target are the binary image masks (occ_source="image_mask").

Writes the operator bundle in the format Baselines/LinearForesight/predictor.py
reads (operators, bin_edges, counts, single_operator, mean_delta, res, crop,
constraint, ridge, n_bins, train_cfg, provenance) + a JSON of the sweep,
rewritten after every lambda.

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/fit_lf_flex.py \
        --out weights/MODEL-XXXX-linear-foresight-flex-mask/operators.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fit_linear_foresight import (  # noqa: E402
    actions_to_pixels, canonicalise, fit_operator, metrics, plate_width_px, predict_world,
    swept_region_mask,
)
from utils import git_provenance  # noqa: E402
from FlexData.dataset import load_flex_cell  # noqa: E402
from Baselines.LinearForesight.model import bin_index, predict_switched, push_length_m  # noqa: E402
from Baselines.LinearForesight.fit_switched import MIN_ROWS_PER_BIN  # noqa: E402

DS20 = "datasets/DS-0020-training-data-flex-N864/old_data/_ported_v1/config.yaml"
LAMBDAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]


def solve(G, C, lam):
    """== fit_operator(Y0, Y1, ridge=lam, toward_identity=True) given G=Y0Y0^T, C=Y1Y0^T."""
    eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return torch.linalg.solve((G + lam * eye).T, (C + lam * eye).T).T


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, R, crop = args.device, args.res, 1.0
    out_json = args.out.rsplit(".", 1)[0] + "_fit.json"

    tr = load_flex_cell(DS20, "train", tag="train", occ_source="image_mask")
    va = load_flex_cell(DS20, "val", tag="val", occ_source="image_mask")
    H, W = tr.H, tr.W
    L_tr, L_va = push_length_m(tr.actions), push_length_m(va.actions)
    edges = torch.linspace(0.0, float(L_tr.max()), args.n_bins + 1)
    b_tr, b_va = bin_index(L_tr, edges), bin_index(L_va, edges)
    counts = [int((b_tr == b).sum()) for b in range(args.n_bins)]
    counts_va = [int((b_va == b).sum()) for b in range(args.n_bins)]
    print(f"train {len(L_tr)} rows (len {float(L_tr.min()):.3f}-{float(L_tr.max()):.3f}), "
          f"val {len(L_va)} rows; edges {[round(float(e), 3) for e in edges]}", flush=True)
    print(f"rows/bin train {counts}  val {counts_va}  (D = {R * R})", flush=True)

    s_tr, e_tr = actions_to_pixels(tr.actions, tr.workspace_min, tr.workspace_max, (H, W))
    s_va, e_va = actions_to_pixels(va.actions, va.workspace_min, va.workspace_max, (H, W))
    plate_px = plate_width_px(va.raw, W)
    region = swept_region_mask(s_va, e_va, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    occ0_va, occ1_va = va.occ0.float(), va.occ1.float()

    def canon(occ, s, e):
        return canonicalise(occ.float().to(dev), s.to(dev), e.to(dev), R, crop).reshape(occ.shape[0], -1).T

    # Gram matrices per bin + global (canonicalise once)
    G_b, C_b = [], []
    G_all = torch.zeros(R * R, R * R, device=dev); C_all = torch.zeros_like(G_all)
    dsum = torch.zeros(R * R, device=dev)
    for b in range(args.n_bins):
        m = (b_tr == b).nonzero(as_tuple=True)[0]
        G = torch.zeros(R * R, R * R, device=dev); C = torch.zeros_like(G)
        for i in range(0, len(m), 2048):
            mm = m[i:i + 2048]
            Y0 = canon(tr.occ0[mm], s_tr[mm], e_tr[mm]); Y1 = canon(tr.occ1[mm], s_tr[mm], e_tr[mm])
            G += Y0 @ Y0.T; C += Y1 @ Y0.T; dsum += (Y1 - Y0).sum(1)
        G_b.append(G); C_b.append(C); G_all += G; C_all += C
    bmd = (dsum / len(L_tr)).cpu()

    log = dict(train_rows=len(L_tr), val_rows=len(L_va), bin_edges=[float(e) for e in edges],
               counts_train=counts, counts_val=counts_va, D=R * R, res=R, plate_px=plate_px,
               persistence_val=metrics(occ0_va, occ1_va, occ0_va, region=region),
               sweep={"single": {}, "switched": {}}, per_bin_val={})
    eye = torch.eye(R * R)
    for lam in LAMBDAS:
        t0 = time.time()
        A1 = solve(G_all, C_all, lam).cpu()
        ops = [solve(G_b[b], C_b[b], lam).cpu() if counts[b] >= MIN_ROWS_PER_BIN else eye
               for b in range(args.n_bins)]
        p1 = predict_world(A1, occ0_va, s_va, e_va, R, (H, W), crop)
        ps = predict_switched(edges, ops, occ0_va, s_va, e_va, L_va, R, (H, W), crop)
        a1 = metrics(p1, occ1_va, occ0_va, region=region)["accuracy"]
        a_s = metrics(ps, occ1_va, occ0_va, region=region)["accuracy"]
        pb = {}
        for b in range(args.n_bins):
            m = b_va == b
            if int(m.sum()):
                pb[b] = dict(single=metrics(p1[m], occ1_va[m], occ0_va[m], region=region[m])["accuracy"],
                             switched=metrics(ps[m], occ1_va[m], occ0_va[m], region=region[m])["accuracy"])
        log["sweep"]["single"][str(lam)] = a1
        log["sweep"]["switched"][str(lam)] = a_s
        log["per_bin_val"][str(lam)] = pb
        print(f"lambda {lam:>7g}: val acc single {a1:.4f} switched {a_s:.4f} ({time.time() - t0:.0f}s) "
              f"per-bin switched {[round(v['switched'], 3) for v in pb.values()]}", flush=True)
        write_json(out_json, log)

    lam_sw = max(LAMBDAS, key=lambda l: log["sweep"]["switched"][str(l)])
    lam_1 = max(LAMBDAS, key=lambda l: log["sweep"]["single"][str(l)])
    # final operators through the project's own fit routine (same math as solve())
    ops = [solve(G_b[b], C_b[b], lam_sw).cpu() if counts[b] >= MIN_ROWS_PER_BIN else eye
           for b in range(args.n_bins)]
    A1 = solve(G_all, C_all, lam_1).cpu()
    log.update(lambda_switched=lam_sw, lambda_single=lam_1,
               val_accuracy_switched=log["sweep"]["switched"][str(lam_sw)],
               val_accuracy_single=log["sweep"]["single"][str(lam_1)])
    # equivalence check against fit_operator on one small bin (cheap)
    bmin = min((b for b in range(args.n_bins) if counts[b] >= MIN_ROWS_PER_BIN), key=lambda b: counts[b])
    m = (b_tr == bmin).nonzero(as_tuple=True)[0]
    A_ref = fit_operator(canon(tr.occ0[m], s_tr[m], e_tr[m]), canon(tr.occ1[m], s_tr[m], e_tr[m]),
                         ridge=lam_sw, toward_identity=True).cpu()
    log["fit_operator_equivalence_maxabs"] = float((A_ref - ops[bmin]).abs().max())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "operators": ops, "bin_edges": edges, "counts": counts,
        "single_operator": A1, "mean_delta": bmd, "res": R, "crop": crop, "constraint": "ridge",
        "ridge": lam_sw, "ridge_single": lam_1, "n_bins": args.n_bins,
        "train_cfg": f"{DS20} split=train occ_source=image_mask",
        "units": "flex (push length bins in FleX units, not metres)",
        "provenance": git_provenance(),
    }, args.out)
    write_json(out_json, log)
    print(f"chosen lambda switched {lam_sw} (val {log['val_accuracy_switched']:.4f}), single {lam_1} "
          f"(val {log['val_accuracy_single']:.4f}); fit_operator equivalence {log['fit_operator_equivalence_maxabs']:.2e}; "
          f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
