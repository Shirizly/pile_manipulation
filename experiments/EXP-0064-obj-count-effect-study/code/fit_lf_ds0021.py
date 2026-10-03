"""EXP-0064 RUN-0013 -- EXP-0062 fit_lf_v2.py recipe copied VERBATIM with the train corpus swapped to DS-0021
(count-group FleX carrots; per-group 90/10 state split). Only D20V2/RUNCFG paths and the 0-99 leakage assert changed.


= EXP-0061's `code/fit_lf_flex.py` (MODEL-0004) with the v2 dataset and a bin-scheme
comparison:

  (a) "equal":      EXP-0003 scheme, 6 equal-width bins over [0, max TRAIN length]
  (b) "collection": the collector's own length-bin edges (run_config.json
                    `bins.edges_abs`, shared by DS-0019): 0.9617 ... 9.6167; rows outside
                    are merged into the end bins (bin_index clamps).

Fit on DS-0020 v2 TRAIN (trajectory split, splits.json; trajectories 0-99 excluded),
ridge toward identity, res 64. (scheme, lambda) picked on DS-0020 v2 VAL -- whole
trajectories held out, never row-random. The single (unswitched) operator gets its own
lambda on the same val split. Final per-bin operators are re-fit with
`Baselines/LinearForesight/fit_switched.py::fit_bins` (the project fit routine) at the
chosen (scheme, lambda) and checked equal to the Gram-matrix solve used for the sweep.
Input and target = binary image masks (occ_source="image_mask").

    python -u experiments/EXP-0062-flex-v2-train-rerun/code/fit_lf_v2.py \
        --out weights/MODEL-0009-linear-foresight-flex-mask-v2/checkpoint.pt
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
os.chdir(REPO)

from fit_linear_foresight import (  # noqa: E402
    actions_to_pixels, canonicalise, metrics, plate_width_px, predict_world, swept_region_mask,
)
from utils import git_provenance  # noqa: E402
from FlexData.dataset import load_flex_cell  # noqa: E402
from Baselines.LinearForesight.model import bin_index, predict_switched, push_length_m  # noqa: E402
from Baselines.LinearForesight.fit_switched import MIN_ROWS_PER_BIN, fit_bins  # noqa: E402

D20V2 = "datasets/DS-0021-flex-carrots-countgroups-train/config.yaml"
RUNCFG = "datasets/DS-0021-flex-carrots-countgroups-train/data/run_config.json"
LAMBDAS = [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scheme", default=None, choices=[None, "equal", "collection"],
                    help="force this bin scheme (lambda still picked on val within it); default: pick on val")
    args = ap.parse_args()
    dev, R, crop, NB = args.device, args.res, 1.0, 6
    out_json = str(Path(args.out).with_name("fit.json"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tr = load_flex_cell(D20V2, "train", tag="train", occ_source="image_mask")
    va = load_flex_cell(D20V2, "val", tag="val", occ_source="image_mask")
    assert set(torch.unique(tr.occ0).tolist()) <= {0.0, 1.0} and set(torch.unique(va.occ1).tolist()) <= {0.0, 1.0}
    tr_traj, va_traj = set(tr.run_idx.tolist()), set(va.run_idx.tolist())
    assert not (tr_traj & va_traj), "train/val share trajectories"
    # (DS-0021 has no leakage exclusion; split = EXP-0064 per-group 90/10)
    H, W = tr.H, tr.W
    L_tr, L_va = push_length_m(tr.actions), push_length_m(va.actions)
    print(f"train {len(L_tr)} rows / {len(tr_traj)} traj, val {len(L_va)} rows / {len(va_traj)} traj "
          f"({time.time() - t0:.0f}s); len train {float(L_tr.min()):.3f}-{float(L_tr.max()):.3f}", flush=True)

    edges_abs = json.load(open(RUNCFG))["transitions"]["bins"]["edges_abs"]
    schemes = {"equal": torch.linspace(0.0, float(L_tr.max()), NB + 1),
               "collection": torch.tensor(edges_abs, dtype=torch.float32)}

    s_tr, e_tr = actions_to_pixels(tr.actions, tr.workspace_min, tr.workspace_max, (H, W))
    s_va, e_va = actions_to_pixels(va.actions, va.workspace_min, va.workspace_max, (H, W))
    plate_px = plate_width_px(va.raw, W)
    region = swept_region_mask(s_va, e_va, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    occ0_va, occ1_va = va.occ0.float(), va.occ1.float()

    def canon(occ, s, e):
        return canonicalise(occ.float().to(dev), s.to(dev), e.to(dev), R, crop).reshape(occ.shape[0], -1).T

    # canonicalise train once, accumulate Gram matrices over a fine partition (rows -> bin per scheme)
    bins_tr = {k: bin_index(L_tr, ed) for k, ed in schemes.items()}
    G = {k: [torch.zeros(R * R, R * R, device=dev) for _ in range(NB)] for k in schemes}
    C = {k: [torch.zeros(R * R, R * R, device=dev) for _ in range(NB)] for k in schemes}
    dsum = torch.zeros(R * R, device=dev)
    t0 = time.time()
    for i in range(0, len(L_tr), 2048):
        sl = slice(i, i + 2048)
        Y0 = canon(tr.occ0[sl], s_tr[sl], e_tr[sl]); Y1 = canon(tr.occ1[sl], s_tr[sl], e_tr[sl])
        dsum += (Y1 - Y0).sum(1)
        for k in schemes:
            bb = bins_tr[k][sl].to(dev)
            for b in range(NB):
                m = bb == b
                if int(m.sum()):
                    y0, y1 = Y0[:, m], Y1[:, m]
                    G[k][b] += y0 @ y0.T; C[k][b] += y1 @ y0.T
    G_all = sum(G["equal"]); C_all = sum(C["equal"])
    bmd = (dsum / len(L_tr)).cpu()
    print(f"Gram matrices ({time.time() - t0:.0f}s)", flush=True)

    log = dict(train_rows=len(L_tr), val_rows=len(L_va), n_traj_train=len(tr_traj), n_traj_val=len(va_traj),
               D=R * R, res=R, plate_px=plate_px, lambdas=LAMBDAS, min_rows_per_bin=MIN_ROWS_PER_BIN,
               persistence_val=metrics(occ0_va, occ1_va, occ0_va, region=region),
               schemes={}, sweep_single={}, provenance=git_provenance())
    for k, ed in schemes.items():
        bva = bin_index(L_va, ed)
        log["schemes"][k] = dict(bin_edges=[float(e) for e in ed],
                                 counts_train=[int((bins_tr[k] == b).sum()) for b in range(NB)],
                                 counts_val=[int((bva == b).sum()) for b in range(NB)],
                                 sweep={}, per_bin_val={})
        print(k, log["schemes"][k]["bin_edges"], "train", log["schemes"][k]["counts_train"],
              "val", log["schemes"][k]["counts_val"], flush=True)
    eye = torch.eye(R * R)
    for lam in LAMBDAS:
        t0 = time.time()
        A1 = solve(G_all, C_all, lam).cpu()
        p1 = predict_world(A1.to(dev), occ0_va.to(dev), s_va.to(dev), e_va.to(dev), R, (H, W), crop).cpu()
        log["sweep_single"][str(lam)] = metrics(p1, occ1_va, occ0_va, region=region)["accuracy"]
        msg = f"lambda {lam:>7g}: single {log['sweep_single'][str(lam)]:.4f}"
        for k, ed in schemes.items():
            cnt = log["schemes"][k]["counts_train"]
            ops = [solve(G[k][b], C[k][b], lam) if cnt[b] >= MIN_ROWS_PER_BIN else eye.to(dev) for b in range(NB)]
            ps = predict_switched(ed, ops, occ0_va.to(dev), s_va.to(dev), e_va.to(dev), L_va.to(dev),
                                  R, (H, W), crop).cpu()
            a = metrics(ps, occ1_va, occ0_va, region=region)["accuracy"]
            bva = bin_index(L_va, ed)
            pb = {}
            for b in range(NB):
                m = bva == b
                if int(m.sum()):
                    pb[b] = dict(n=int(m.sum()),
                                 single=metrics(p1[m], occ1_va[m], occ0_va[m], region=region[m])["accuracy"],
                                 switched=metrics(ps[m], occ1_va[m], occ0_va[m], region=region[m])["accuracy"])
            log["schemes"][k]["sweep"][str(lam)] = a
            log["schemes"][k]["per_bin_val"][str(lam)] = pb
            msg += f" | {k} {a:.4f} per-bin {[round(v['switched'], 3) for v in pb.values()]}"
            del ops
        print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
        write_json(out_json, log)

    cands = [(k, l) for k in schemes for l in LAMBDAS if args.scheme in (None, k)]
    sch, lam_sw = max(cands, key=lambda kl: log["schemes"][kl[0]]["sweep"][str(kl[1])])
    lam_1 = max(LAMBDAS, key=lambda l: log["sweep_single"][str(l)])
    log.update(forced_scheme=args.scheme, chosen_scheme=sch, lambda_switched=lam_sw, lambda_single=lam_1,
               val_accuracy_switched=log["schemes"][sch]["sweep"][str(lam_sw)],
               val_accuracy_single=log["sweep_single"][str(lam_1)],
               best_per_scheme={k: dict(lam=max(LAMBDAS, key=lambda l: log["schemes"][k]["sweep"][str(l)]),
                                        acc=max(log["schemes"][k]["sweep"].values())) for k in schemes})
    edges = schemes[sch]
    # final per-bin operators through the project's fit routine (fit_switched.fit_bins)
    G_all, C_all = G_all.cpu(), C_all.cpu()
    ops_gram = [solve(G[sch][b], C[sch][b], lam_sw).cpu() for b in range(NB)]
    del G, C
    torch.cuda.empty_cache()
    ops, counts = fit_bins(tr.occ0, tr.occ1, s_tr, e_tr, L_tr, edges, R, crop, "ridge", lam_sw, dev)
    log["fit_bins_vs_gram_maxabs"] = max(float((a - b).abs().max()) for a, b in zip(ops, ops_gram)
                                         if not torch.equal(a, torch.eye(R * R)))
    A1 = solve(G_all.to(dev), C_all.to(dev), lam_1).cpu()
    torch.save({
        "operators": ops, "bin_edges": edges, "counts": counts,
        "single_operator": A1, "mean_delta": bmd, "res": R, "crop": crop, "constraint": "ridge",
        "ridge": lam_sw, "ridge_single": lam_1, "n_bins": NB, "bin_scheme": sch,
        "train_cfg": f"{D20V2} split=train occ_source=image_mask (DS-0021, EXP-0064 RUN-0013)",
        "units": "flex (push length bins in FleX units, not metres)",
        "provenance": log["provenance"],
    }, args.out)
    write_json(out_json, log)
    print(f"chosen scheme {sch} lambda {lam_sw} (val {log['val_accuracy_switched']:.4f}); single lambda {lam_1} "
          f"(val {log['val_accuracy_single']:.4f}); fit_bins vs gram {log['fit_bins_vs_gram_maxabs']:.2e}; "
          f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
