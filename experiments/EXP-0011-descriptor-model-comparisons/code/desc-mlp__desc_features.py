"""experiments/temp/desc-mlp/desc_features.py -- build the D-all-local (94-dim)
push-frame descriptor + action-feature tensors for a set of raw
overnight_randlen `_*_data.pt` files.

Reuses, unmodified:
  - transforms.functional.particles_to_occupancy (rasteriser, same convention
    as experiments/temp/dmdc-lenbins/descriptors.py)
  - experiments/temp/dmdc-lenbins/descriptors_d.py's push_frame_full_descriptors
    / slices_d (the SAME 94-dim basis weights/MODEL-0002-descriptor-only-D-all-local
    was fit on, so the switched-linear baseline and the new MLP are scored on
    an identical basis)
  - descriptors_b.world_to_pushframe_px for the pixel-frame transform

Action features per the task spec: [x, y, cos(theta), sin(theta), length] --
x,y = push START point (world metres), theta = blade yaw (`angles` field),
length = ||p_stop_xy - p_start_xy||.
"""
from __future__ import annotations

import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
from transforms.functional import particles_to_occupancy  # noqa: E402
from descriptors_d import push_frame_full_descriptors, slices_d  # noqa: E402
from descriptors_b import world_to_pushframe_px  # noqa: E402

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH


def list_files(root: str) -> list[str]:
    return sorted(glob.glob(f"{root}/*/_*_data.pt"))


def build_from_files(files: list[str], verbose: bool = True):
    """-> dict(phi_t[N,94], phi_t1[N,94], length_m[N], x[N], y[N], angle[N])."""
    phi_t_all, phi_t1_all, len_all, x_all, y_all, ang_all = [], [], [], [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        states = d["states"][:, :, :3]
        states_ = d["states_"][:, :, :3]
        occ0 = particles_to_occupancy(states, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
        occ1 = particles_to_occupancy(states_, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)

        H, W = occ0.shape[-2:]
        global_mass0 = occ0.sum(dim=(-2, -1)) / (H * W)
        global_mass1 = occ1.sum(dim=(-2, -1)) / (H * W)

        p_start = d["p_starts"][:, :2]
        p_stop = d["p_stops"][:, :2]
        length_m = (p_stop - p_start).norm(dim=-1)
        start_px = world_to_pushframe_px(p_start)
        end_px = world_to_pushframe_px(p_stop)

        local0 = push_frame_full_descriptors(occ0, start_px, end_px)
        local1 = push_frame_full_descriptors(occ1, start_px, end_px)

        phiD0 = torch.cat([global_mass0.unsqueeze(1), length_m.unsqueeze(1), local0], dim=1)
        phiD1 = torch.cat([global_mass1.unsqueeze(1), length_m.unsqueeze(1), local1], dim=1)

        phi_t_all.append(phiD0.numpy().astype(np.float32))
        phi_t1_all.append(phiD1.numpy().astype(np.float32))
        len_all.append(length_m.numpy().astype(np.float32))
        x_all.append(p_start[:, 0].numpy().astype(np.float32))
        y_all.append(p_start[:, 1].numpy().astype(np.float32))
        ang_all.append(d["angles"].numpy().astype(np.float32))
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(files)}] {time.time() - t0:.1f}s")
    if verbose:
        print(f"done: {len(files)} files, {len(files) and 0 or 0}{time.time() - t0:.1f}s")
    return dict(
        phi_t=np.concatenate(phi_t_all),
        phi_t1=np.concatenate(phi_t1_all),
        length_m=np.concatenate(len_all),
        x=np.concatenate(x_all),
        y=np.concatenate(y_all),
        angle=np.concatenate(ang_all),
    )


def action_features(d: dict) -> np.ndarray:
    """[x, y, cos(theta), sin(theta), length] -> [N,5] float32."""
    return np.stack([d["x"], d["y"], np.cos(d["angle"]), np.sin(d["angle"]),
                      d["length_m"]], axis=1).astype(np.float32)


if __name__ == "__main__":
    import os

    outdir = "experiments/temp/desc-mlp/cache"
    os.makedirs(outdir, exist_ok=True)
    for split, root in [("train", "Genesis/data/overnight_randlen_train"),
                        ("test", "Genesis/data/overnight_randlen_test")]:
        files = list_files(root)
        print(f"{split}: {len(files)} files")
        d = build_from_files(files)
        np.savez(f"{outdir}/desc_{split}.npz", **d)
        print(f"wrote {outdir}/desc_{split}.npz: {d['phi_t'].shape[0]} rows")
