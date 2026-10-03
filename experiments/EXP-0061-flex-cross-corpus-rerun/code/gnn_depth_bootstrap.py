"""Paired bootstrap (over val rows, 2000 resamples) of the swept-region mask accuracy
difference between GNN depth variants saved by gnn_check.py --save-pred all.

    python -u .../gnn_depth_bootstrap.py results/gnn_val_full.json n200_true n200_plane0.24 n200_plane0
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_linear_foresight import actions_to_pixels, plate_width_px, swept_region_mask  # noqa: E402
from FlexData.dataset import load_flex_cell  # noqa: E402
from gnn_check import DS20, slice_cell  # noqa: E402


def per_row(pred, truth, prev, region):
    n = pred.shape[0]
    w = region.reshape(n, -1); npix = w.sum(1).clamp_min(1)
    num = ((((pred - truth) * region).reshape(n, -1) ** 2).sum(1) / npix).sqrt()
    den = ((((truth - prev) * region).reshape(n, -1) ** 2).sum(1) / npix).sqrt()
    return num, den


def main():
    js, names = sys.argv[1], sys.argv[2:]
    cell_all = load_flex_cell(DS20, "val", occ_source="image_mask")
    P = {n: torch.load(js.replace(".json", f"_{n}_pred.pt")) for n in names}
    rows = P[names[0]]["rows"]
    cell, _ = slice_cell(cell_all, rows)
    H, W = cell.H, cell.W
    s, e = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    pp = plate_width_px(cell.raw, W)
    reg = swept_region_mask(s, e, (H, W), 0.5 * pp + 2.0, 0.5 * pp)
    o0, o1 = cell.occ0.float(), cell.occ1.float()
    nd = {n: per_row(P[n]["pred"].float(), o1, o0, reg) for n in names}
    den = nd[names[0]][1]
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(rows), (2000, len(rows)), generator=g)
    acc = {n: 1 - nd[n][0][idx].mean(1) / den[idx].mean(1) for n in names}
    out = {"n_rows": len(rows)}
    for n in names:
        a0 = float(1 - nd[n][0].mean() / den.mean())
        out[n] = dict(accuracy=a0, ci95=[float(acc[n].quantile(0.025)), float(acc[n].quantile(0.975))])
    ref = names[0]
    for n in names[1:]:
        d = acc[n] - acc[ref]
        out[f"{n}_minus_{ref}"] = dict(mean=float(d.mean()), ci95=[float(d.quantile(0.025)), float(d.quantile(0.975))])
    print(json.dumps(out, indent=1))
    Path(js.replace(".json", "_bootstrap.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
