"""EXP-0061 Task A diagnostics on DS-0020 val (image masks):

1. renderer/representation BOUND: carry the input mask by each node's TRUE
   displacement (node tracked to its nearest GT particle, table XY) through the
   same `FlexGNNPredictor.render` -- the best mask any dynamics model could get
   through N nodes + nearest-node carry. Compared with the GNN's own accuracy on
   the same rows.
2. figures: predicted-vs-true masks for a few rows -> figures/gnn_check/.

    python -u .../gnn_bound_and_figs.py --cfg n50_plane0.24 --rows 300 --out .../results/gnn_bound.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_linear_foresight import actions_to_pixels, metrics, plate_width_px, swept_region_mask  # noqa: E402
from FlexData.dataset import load_flex_cell  # noqa: E402
from Baselines.GNN.flex_predictor import GLOBAL_SCALE, FlexGNNPredictor  # noqa: E402
from gnn_check import DS20, parse_cfg, slice_cell, subset  # noqa: E402

FIG = REPO / "experiments/EXP-0061-flex-cross-corpus-rerun/figures/gnn_check"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", nargs="+", required=True)
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-cfg", default=None)
    ap.add_argument("--n-fig", type=int, default=8)
    args = ap.parse_args()
    cell_all = load_flex_cell(DS20, "val", tag="val", occ_source="image_mask")
    rows = subset(cell_all, args.rows)
    cell, _ = slice_cell(cell_all, rows)
    raw = cell.raw
    H, W = cell.H, cell.W
    to_pxl, ctr = float(raw.to_pxl), float(raw.ctr_in_PXL[0])
    s_px, e_px = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    plate_px = plate_width_px(raw, W)
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    occ0, occ1 = cell.occ0.float(), cell.occ1.float()
    res = json.loads(Path(args.out).read_text()) if os.path.exists(args.out) else {}
    for name in args.cfgs:
        p = FlexGNNPredictor(**parse_cfg(name))
        pred = p.predict_occ(cell)
        oracle = torch.zeros_like(pred)
        nodes_all = []
        for j in range(len(rows)):
            i = int(rows[j])          # dataset index (particles_before resolves flags itself)
            assert raw.get_run_index(i) == int(cell.run_idx[j]) and raw.get_step_index(i) == int(cell.step_idx[j])
            st = p._state(raw, int(cell.run_idx[j]), int(cell.step_idx[j]))
            nodes = torch.from_numpy(st["nodes"]).to(p.device)
            cur = st["nodes"][:, :2] * GLOBAL_SCALE
            cur = np.stack([cur[:, 0], -cur[:, 1]], 1)
            P0 = raw.particles_before(i).numpy().astype(np.float32)
            P1 = raw.particles_after(i).numpy().astype(np.float32)
            ok = np.isfinite(P0).all(1) & np.isfinite(P1).all(1)
            P0, P1 = P0[ok], P1[ok]
            nn = np.argmin(((cur[:, None] - P0[None]) ** 2).sum(-1), 1)
            d = torch.from_numpy(P1[nn] - P0[nn]).to(p.device)[None]
            oracle[j] = p.render(st, d, nodes, H, W, to_pxl, ctr)[0].cpu()
            nodes_all.append(cur)
        a_m = metrics(pred, occ1, occ0, region=region)["accuracy"]
        a_o = metrics(oracle, occ1, occ0, region=region)["accuracy"]
        res[name] = dict(n_rows=len(rows), accuracy_model=a_m, accuracy_oracle_node_carry=a_o)
        print(f"[{name}] model {a_m:.4f}  oracle-node-carry bound {a_o:.4f}  ({len(rows)} rows)", flush=True)
        tmp = args.out + ".tmp"; Path(tmp).write_text(json.dumps(res, indent=1)); os.replace(tmp, args.out)

        if name == args.fig_cfg:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            FIG.mkdir(parents=True, exist_ok=True)
            # rows with the largest true change inside the region, plus a few typical ones
            chg = ((occ1 - occ0).abs() * region).sum((1, 2))
            order = torch.argsort(chg, descending=True)
            pick = list(order[:args.n_fig // 2].tolist()) + list(order[len(order) // 2:len(order) // 2 + args.n_fig // 2].tolist())
            fig, axs = plt.subplots(len(pick), 4, figsize=(12, 3 * len(pick)))
            for r, j in enumerate(pick):
                acc_j = metrics(pred[j:j + 1], occ1[j:j + 1], occ0[j:j + 1], region=region[j:j + 1])["accuracy"]
                acc_o = metrics(oracle[j:j + 1], occ1[j:j + 1], occ0[j:j + 1], region=region[j:j + 1])["accuracy"]
                ims = [occ0[j], occ1[j], pred[j], oracle[j]]
                titles = [f"input mask (traj {int(cell.run_idx[j])} push {int(cell.step_idx[j])})",
                          "true next mask", f"GNN pred (acc {acc_j:.2f})", f"true-node-disp carry (acc {acc_o:.2f})"]
                for c in range(4):
                    ax = axs[r, c]
                    ax.imshow(ims[c].numpy(), cmap="gray_r", vmin=0, vmax=1, origin="upper")
                    ax.contour(region[j].numpy(), levels=[0.5], colors="tab:orange", linewidths=0.8)
                    ax.annotate("", xy=(float(e_px[j, 0]), float(e_px[j, 1])), xytext=(float(s_px[j, 0]), float(s_px[j, 1])),
                                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5))
                    if c in (0, 2):
                        npx = nodes_all[j] * to_pxl + ctr
                        ax.scatter(npx[:, 1], npx[:, 0], s=6, c="tab:blue")
                    ax.set_title(titles[c], fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"DS-0020 val, {name}: grid row = X (down), col = Y (right); orange = swept region; blue = GNN nodes", fontsize=9)
            fig.tight_layout()
            out = FIG / f"pred_vs_true_{name}.png"
            fig.savefig(out, dpi=80); plt.close(fig)
            print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
