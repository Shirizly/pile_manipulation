"""EXP-0061 Task A -- sanity check of the original dyn-res-pile-manip GNN checkpoint
on DS-0020 VAL through Baselines/GNN/flex_predictor.py (image-mask input/truth).

For each config (particle_num, depth_mode, plane_y, carry_k) it reports
  * mask accuracy (fit_linear_foresight.metrics, swept region with the 2.4-unit
    plate -> px), persistence = 0 by construction, plus raw rms of model and
    persistence inside the region;
  * node-level check (the training loss's own quantity, independent of the
    renderer): each node is tracked to its nearest GT particle (table XY) before
    the push; RMSE of predicted vs GT XY DISPLACEMENT, vs persistence (node does not
    move), over all nodes and over nodes whose GT particle moved > 0.25 units.
Results are rewritten atomically after every config.

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/gnn_check.py \
        --configs n50_true n50_plane0 ... --rows 600 --out .../results/gnn_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fit_linear_foresight import actions_to_pixels, metrics, plate_width_px, swept_region_mask  # noqa: E402
from FlexData.dataset import load_flex_cell  # noqa: E402
from Baselines.GNN.flex_predictor import GLOBAL_SCALE, FlexGNNPredictor  # noqa: E402

DS20 = "datasets/DS-0020-training-data-flex-N864/old_data/_ported_v1/config.yaml"


def parse_cfg(name: str) -> dict:
    """n50_true | n50_plane0.24 | n50_plane0_k4"""
    m = re.fullmatch(r"n(\d+)_(true|plane([0-9.]+))(?:_k(\d+))?", name)
    assert m, name
    return dict(particle_num=int(m.group(1)), depth_mode="true" if m.group(2) == "true" else "plane",
                plane_y=float(m.group(3) or 0.0), carry_k=int(m.group(4) or 1))


def subset(cell, n: int | None):
    if n is None or n >= cell.occ0.shape[0]:
        return torch.arange(cell.occ0.shape[0])
    g = torch.Generator().manual_seed(0)
    return torch.randperm(cell.occ0.shape[0], generator=g)[:n].sort().values


def slice_cell(cell, rows):
    import dataclasses
    upd = {f.name: getattr(cell, f.name)[rows] for f in dataclasses.fields(cell)
           if torch.is_tensor(getattr(cell, f.name)) and getattr(cell, f.name).dim() > 0
           and getattr(cell, f.name).shape[0] == cell.occ0.shape[0]}
    return dataclasses.replace(cell, **upd), rows


def node_check(pred: FlexGNNPredictor, cell, rows_cell, cell_rows_raw):
    """RMSE (FleX units, table XY) of predicted node positions vs tracked GT particles."""
    raw = cell.raw
    e_m, e_p, e_m_mv, e_p_mv = [], [], [], []
    for j in range(len(rows_cell)):
        i = int(cell_rows_raw[j])
        assert raw.get_run_index(i) == int(cell.run_idx[j]) and raw.get_step_index(i) == int(cell.step_idx[j])
        st = pred._state(raw, int(cell.run_idx[j]), int(cell.step_idx[j]))
        s_cur, s_pred = pred.predict_nodes([st], cell.actions[j:j + 1])
        cur = s_cur[0, :, :2].cpu().numpy() * GLOBAL_SCALE
        prd = s_pred[0, :, :2].cpu().numpy() * GLOBAL_SCALE
        cur = np.stack([cur[:, 0], -cur[:, 1]], 1)        # cam (x, y) -> table (X, Y)
        prd = np.stack([prd[:, 0], -prd[:, 1]], 1)
        P0 = raw.particles_before(i).numpy().astype(np.float32)
        P1 = raw.particles_after(i).numpy().astype(np.float32)
        ok = np.isfinite(P0).all(1) & np.isfinite(P1).all(1)
        P0, P1 = P0[ok], P1[ok]
        nn = np.argmin(((cur[:, None] - P0[None]) ** 2).sum(-1), 1)
        g0, g1 = P0[nn], P1[nn]
        em = (((prd - cur) - (g1 - g0)) ** 2).sum(1)    # predicted vs GT DISPLACEMENT
        ep = ((g1 - g0) ** 2).sum(1)                     # persistence: zero displacement
        mv = np.linalg.norm(g1 - g0, axis=1) > 0.25
        e_m.append(em); e_p.append(ep); e_m_mv.append(em[mv]); e_p_mv.append(ep[mv])
    f = lambda xs: float(np.sqrt(np.concatenate(xs).mean())) if sum(len(x) for x in xs) else float("nan")
    out = dict(node_rmse_model=f(e_m), node_rmse_persist=f(e_p),
               node_rmse_model_moved=f(e_m_mv), node_rmse_persist_moved=f(e_p_mv),
               n_nodes_moved=int(sum(len(x) for x in e_m_mv)), n_nodes=int(sum(len(x) for x in e_m)))
    out["node_accuracy"] = 1 - out["node_rmse_model"] / out["node_rmse_persist"]
    out["node_accuracy_moved"] = 1 - out["node_rmse_model_moved"] / out["node_rmse_persist_moved"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--rows", type=int, default=None, help="random val subset (seed 0); default all")
    ap.add_argument("--node-rows", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-pred", default=None, help="config name whose predictions are saved (.pt)")
    ap.add_argument("--node-only", action="store_true", help="only the node-level check (key <cfg>__node)")
    args = ap.parse_args()

    cell_all = load_flex_cell(DS20, "val", tag="ds0020_val", occ_source="image_mask")
    rows = subset(cell_all, args.rows)
    cell, _ = slice_cell(cell_all, rows)
    # FlexPileData.particles_before/after take a DATASET index (they resolve flags themselves)
    raw_rows = [int(r) for r in rows]
    H, W = cell.H, cell.W
    s_px, e_px = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    plate_px = plate_width_px(cell.raw, W)
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    occ0, occ1 = cell.occ0.float(), cell.occ1.float()
    base = metrics(occ0, occ1, occ0, region=region)
    print(f"val rows {len(rows)} / {cell_all.occ0.shape[0]}; plate_px {plate_px:.2f}; persistence rms {base['rms']:.4f}", flush=True)

    res = {}
    if os.path.exists(args.out):
        res = json.loads(Path(args.out).read_text())
    res["_meta"] = dict(n_rows=len(rows), n_val=int(cell_all.occ0.shape[0]), plate_px=plate_px,
                        persistence=base, rows_seed=0, occ_source="image_mask")
    for name in args.configs:
        kw = parse_cfg(name)
        t0 = time.time()
        p = FlexGNNPredictor(**kw)
        if args.node_only:
            nrows = min(args.node_rows, len(rows))
            nc = node_check(p, *slice_cell(cell, torch.arange(nrows)), raw_rows[:nrows])
            res[name + "__node"] = dict(cfg=kw, node=nc, n_node_rows=nrows, seconds=time.time() - t0)
            print(f"[{name} node-only] node_acc={nc['node_accuracy']:.3f} moved={nc['node_accuracy_moved']:.3f} "
                  f"(rmse {nc['node_rmse_model_moved']:.3f} vs {nc['node_rmse_persist_moved']:.3f}, "
                  f"{nc['n_nodes_moved']} moved / {nc['n_nodes']} nodes) ({time.time() - t0:.0f}s)", flush=True)
            Path(args.out + ".tmp").write_text(json.dumps(res, indent=1)); os.replace(args.out + ".tmp", args.out)
            continue
        # renderer identity check: zero displacement must reproduce the input mask
        st = p._state(cell.raw, int(cell.run_idx[0]), int(cell.step_idx[0]))
        z = p.render(st, torch.zeros(1, st["nodes"].shape[0], 2, device=p.device),
                     torch.from_numpy(st["nodes"]).to(p.device), H, W, float(cell.raw.to_pxl),
                     float(cell.raw.ctr_in_PXL[0]))[0].cpu()
        ident = float((z != occ0[0]).float().sum())
        pred = p.predict_occ(cell)
        m = metrics(pred, occ1, occ0, region=region)
        nrows = min(args.node_rows, len(rows))
        nc = node_check(p, *slice_cell(cell, torch.arange(nrows)), raw_rows[:nrows])
        res[name] = dict(cfg=kw, accuracy=m["accuracy"], rms=m["rms"], soft_iou=m["soft_iou"],
                         renderer_identity_mismatch_px=ident, node=nc, n_node_rows=nrows,
                         seconds=time.time() - t0,
                         mean_den=float(np.mean([s["den"] for s in p._cache.values()])))
        print(f"[{name}] acc={m['accuracy']:.4f} rms={m['rms']:.4f} identity_mismatch={ident:.0f}px "
              f"node_acc={nc['node_accuracy']:.3f} moved={nc['node_accuracy_moved']:.3f} "
              f"(rmse {nc['node_rmse_model_moved']:.3f} vs {nc['node_rmse_persist_moved']:.3f}, "
              f"{nc['n_nodes_moved']} moved nodes) ({time.time() - t0:.0f}s)", flush=True)
        if args.save_pred in (name, "all"):
            torch.save(dict(rows=rows, pred=pred.to(torch.uint8), cfg=kw), args.out.replace(".json", f"_{name}_pred.pt"))
        tmp = args.out + ".tmp"
        Path(tmp).write_text(json.dumps(res, indent=1))
        os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
