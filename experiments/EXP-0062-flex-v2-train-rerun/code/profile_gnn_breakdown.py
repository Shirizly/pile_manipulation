"""EXP-0062 RUN-0003 -- where the GNN's per-slate time goes, stage by stage (the GNN's own
perception INCLUDED), on the DS-0019 slates, each slate's whole pool as one batch.

Stages (sync after each, so the total is inflated slightly vs the un-staged call; the staged
output is asserted == predictor.predict_occ):
  imread (colour PNG, 720x720)  | segment + back-project onto the plane (image_pointcloud)
  voxel downsample (np.unique)  | FPS (python loop over N)  | recenter (python loop over N)
  s_delta (batched torch)       | graph build (adjacency: radius + top-10, Rr/Rs incl. the per-sample loop)
  GNN forward (PropModuleDiffDen) | render (nearest-node carry of every fg pixel + binning; per-row loop)
`--fast` uses the vectorised graph build + render (bit-identical; asserted).

    python -u experiments/EXP-0062-flex-v2-train-rerun/code/profile_gnn_breakdown.py --ckpt weights/MODEL-0010-*/checkpoint.pth --out results/timing_gnn_parts/breakdown.json [--fast]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Baselines.common import eval_report as er  # noqa: E402
import Baselines.GNN.flex_predictor as fp  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from time_inference import sub_batch  # noqa: E402


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def graph_build(model, a_cur, s_cur, s_delta, vec):
    """predict_one_step's adjacency part (source code path, or the vectorised one)."""
    B, N = a_cur.size()
    if vec:
        x = s_cur + s_delta
        dis = torch.sum((x[:, None, :, :] - x[:, :, None, :]) ** 2, -1)
        thr = model.adj_thresh ** 2
        topk_idx = torch.topk(dis, k=min(10, N), dim=2, largest=False).indices
        adj = ((dis - thr) < 0).float() * torch.zeros_like(dis).scatter_(2, topk_idx, 1)
        rels = adj.nonzero()
        n_rels = torch.bincount(rels[:, 0], minlength=B)
        n_rel = int(n_rels.max().item())
        rels_idx = torch.arange(rels.shape[0], device=s_cur.device) - (torch.cumsum(n_rels, 0) - n_rels)[rels[:, 0]]
    else:   # model/gnn_dyn.py predict_one_step, verbatim logic
        s_receiv = (s_cur + s_delta)[:, :, None, :].repeat(1, 1, N, 1)
        s_sender = (s_cur + s_delta)[:, None, :, :].repeat(1, N, 1, 1)
        threshold = model.adj_thresh * model.adj_thresh
        dis = torch.sum((s_sender - s_receiv) ** 2, -1)
        topk_idx = torch.topk(dis, k=min(10, N), dim=2, largest=False).indices
        topk_bin_mat = torch.zeros_like(dis, dtype=torch.float32, device=dis.device)
        topk_bin_mat.scatter_(2, topk_idx, 1)
        adj = ((dis - threshold) < 0).float() * topk_bin_mat
        n_rels = adj.sum(dim=(1, 2))
        n_rel = n_rels.max().long().item()
        rels_idx = torch.hstack([torch.arange(n_rels[i]) for i in range(B)]).to(device=s_cur.device, dtype=torch.long)
        rels = adj.nonzero()
    Rr = torch.zeros((B, n_rel, N), device=s_cur.device, dtype=s_cur.dtype)
    Rs = torch.zeros((B, n_rel, N), device=s_cur.device, dtype=s_cur.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    return Rr, Rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    import cv2
    cell = er._load_cell(er.CORPORA["flex_ds0019_mask"], tag="flex_ds0019_mask")
    pr = fp.FlexGNNPredictor(ckpt_path=a.ckpt, particle_num=a.n, depth_mode="plane", plane_y=0.24, carry_k=1,
                             device=DEV, fast_graph=a.fast, vector_render=a.fast, cache_states=False)
    raw = cell.raw
    H, W = cell.H, cell.W
    to_pxl, ctr = float(raw.to_pxl), float(raw.ctr_in_PXL[0])
    root, idx = pr._image_paths(raw)
    sids = cell.slate_idx.unique().tolist()
    rows_of = {s: (cell.slate_idx == s).nonzero(as_tuple=True)[0] for s in sids}
    stages = ["imread", "segment_backproject", "voxel", "fps", "recenter", "s_delta", "graph_build",
              "gnn_forward", "render"]
    T = {k: {s: [] for s in sids} for k in stages}
    total = {s: [] for s in sids}
    for p in range(a.passes + 1):                     # pass 0 = warm-up
        for s in sids:
            rr = rows_of[s]
            acts = cell.actions[rr].to(DEV)
            sync(); t = [time.perf_counter()]
            color = cv2.imread(str(root / idx["states"][str(s)]["initial_color"]), cv2.IMREAD_COLOR); t.append(time.perf_counter())
            v, u, pts, h = fp.image_pointcloud(color, None, 0.24); t.append(time.perf_counter())
            ds = fp._voxel_down(pts, fp.VOXEL); t.append(time.perf_counter())
            assert len(ds) > a.n, "fallback state: not profiled separately"
            rng = np.random.default_rng(zlib.crc32(f"{raw.dataset_id}/{s}/0".encode()))
            nodes, r = fp._fps(ds, a.n, int(rng.integers(len(ds)))); t.append(time.perf_counter())
            nodes = fp._recenter(ds, nodes, min(0.02, 0.5 * r)); t.append(time.perf_counter())
            st = dict(nodes=nodes.astype(np.float32), den=1.0 / (r * r), pts=pts.astype(np.float32),
                      h=h.astype(np.float32), v=v, u=u)
            s_cur = torch.from_numpy(np.stack([st["nodes"]] * len(rr))).to(DEV)
            dens = torch.full((len(rr),), st["den"], dtype=torch.float32, device=DEV)
            sd = fp.s_delta_train(s_cur, acts); sync(); t.append(time.perf_counter())
            a0 = torch.zeros(s_cur.shape[:2], device=DEV)
            Rr, Rs = graph_build(pr.model, a0, s_cur, sd, a.fast); sync(); t.append(time.perf_counter())
            with torch.no_grad():
                s_pred = pr.model.model.forward(a0, s_cur, sd, Rr, Rs, dens)
            sync(); t.append(time.perf_counter())
            d = (s_pred - s_cur)[..., :2] * fp.GLOBAL_SCALE
            dw = torch.stack([d[..., 0], -d[..., 1]], -1)
            out = pr.render(st, dw, s_cur[0], H, W, to_pxl, ctr).cpu(); t.append(time.perf_counter())
            if p == 0:
                ref = pr.predict_occ(sub_batch(cell, rr.tolist(), DEV))
                assert torch.equal(ref, out), f"staged output != predict_occ on slate {s}"
                continue
            for k, name in enumerate(stages):
                T[name][s].append(1e3 * (t[k + 1] - t[k]))
            total[s].append(1e3 * (t[-1] - t[0]))
    med = {k: float(np.median([np.median(v) for v in T[k].values()])) for k in stages}
    res = dict(ckpt=a.ckpt, n=a.n, fast=a.fast, device=DEV, threads=torch.get_num_threads(), gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None, passes=a.passes,
               n_slates=len(sids), stage_ms_median_over_slates=med,
               stage_share={k: med[k] / sum(med.values()) for k in stages},
               total_ms_median=float(np.median([np.median(v) for v in total.values()])),
               perception_ms=sum(med[k] for k in stages[:5]),
               model_ms=sum(med[k] for k in stages[5:8]), render_ms=med["render"],
               staged_output_equals_predict_occ=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
