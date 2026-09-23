"""STEP 1 gate: ground-truth flow reconstruction ceiling, EXP-0025 extension.

Builds a per-pixel backward-displacement target field from particle
correspondence (states/states_ share indexing), warps occ0 through it, and
scores swept-region accuracy against occ1 -- model-free. This is both a
sign/indexing check (a wrong convention scores near/below zero) and the
ceiling for any supervised-flow model using this parameterisation.
"""
import sys
sys.path.insert(0, "/home/alon/Code/pile_manipulation")
import torch
import torch.nn.functional as F
import numpy as np

from Baselines.common.data import load_cell
from fit_linear_foresight import actions_to_pixels, metrics, swept_region_mask

torch.manual_seed(0)

CORPUS = dict(
    eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
    manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json",
)

print("loading L20mm eval cell...")
cell = load_cell(CORPUS["eval_cfg"], "train", manifest_path=CORPUS["manifest"], tag="L20mm_eval")
N, H, W = cell.occ0.shape
print(f"N={N} H={H} W={W}")

raw = cell.raw
to_pxl = float(raw.to_pxl)
ctr = raw.ctr_in_PXL.to(torch.float32)  # (3,)

states = cell.states    # (N,20,7) world metres
states_ = cell.states_  # (N,20,7)

# pixel-space centers, matching Genesis/training/dataset.py::_extract_sample_in_pxl
# particles[:, :3] = particles[:, :3] * to_pxl + ctr_in_PXL
# resulting grid axis convention (per _draw_particle_grid docstring, pinned by
# tests/test_grid_convention.py): final occ grid dim0 = world_x, dim1 = world_y,
# and particles[:,0] IS that dim0 pixel coord, particles[:,1] IS that dim1 coord
# (no further swap needed -- the cv2 transpose dance in _draw_particle_grid
# nets out to identity on the (x_pixel, y_pixel) -> (dim0, dim1) mapping).
src_row = states[:, :, 0] * to_pxl + ctr[0]   # (N,20) -> dim0 (world x)
src_col = states[:, :, 1] * to_pxl + ctr[1]   # (N,20) -> dim1 (world y)
dst_row = states_[:, :, 0] * to_pxl + ctr[0]
dst_col = states_[:, :, 1] * to_pxl + ctr[1]

occ1 = cell.occ1.to(torch.float32)  # (N,H,W)
occ0 = cell.occ0.to(torch.float32)

# per-pixel grid
rows = torch.arange(H, dtype=torch.float32)
cols = torch.arange(W, dtype=torch.float32)
grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")  # (H,W) each

flow_target = torch.zeros(N, 2, H, W, dtype=torch.float32)  # [dcol, drow]
mask = torch.zeros(N, H, W, dtype=torch.float32)

for n in range(N):
    fg = occ1[n] > 0.5  # (H,W) destination foreground
    if not bool(fg.any()):
        continue
    fg_idx = fg.nonzero(as_tuple=False)  # (K,2) row,col
    fg_row = fg_idx[:, 0].float()
    fg_col = fg_idx[:, 1].float()
    # distance from each foreground pixel to each of 20 destination centers
    dr = fg_row[:, None] - dst_row[n][None, :]   # (K,20)
    dc = fg_col[:, None] - dst_col[n][None, :]
    d2 = dr * dr + dc * dc
    nearest = d2.argmin(dim=1)  # (K,)
    dcol = src_col[n][nearest] - fg_col - (dst_col[n][nearest] - fg_col)  # placeholder, fixed below
    # Correct: target flow at destination pixel x uses the PARTICLE's own
    # (src - dst) displacement (constant per particle, not per-pixel offset
    # from the particle center) -- assign the WHOLE particle's rigid
    # displacement to every foreground pixel nearest that particle.
    dcol = (src_col[n][nearest] - dst_col[n][nearest])
    drow = (src_row[n][nearest] - dst_row[n][nearest])
    flow_target[n, 0][fg] = dcol
    flow_target[n, 1][fg] = drow
    mask[n][fg] = 1.0

print("built target flow field. nonzero-flow frac:", float((mask > 0).float().mean()))
print("displacement stats (px), foreground only:")
disp_mag = (flow_target[:, 0] ** 2 + flow_target[:, 1] ** 2).sqrt()
fg_mag = disp_mag[mask > 0]
print(f"  mean={fg_mag.mean():.3f}  median={fg_mag.median():.3f}  p95={fg_mag.quantile(0.95):.3f}  max={fg_mag.max():.3f}")

# ---- backward warp occ0 through the GT flow field ----
def base_grid(B, H, W, device, dtype):
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    g = torch.stack([gx, gy], dim=-1)
    return g.unsqueeze(0).expand(B, -1, -1, -1)

disp_x_norm = flow_target[:, 0] * (2.0 / max(W - 1, 1))  # dcol -> x/col axis
disp_y_norm = flow_target[:, 1] * (2.0 / max(H - 1, 1))  # drow -> y/row axis
disp_norm = torch.stack([disp_x_norm, disp_y_norm], dim=-1)  # (N,H,W,2)

bgrid = base_grid(N, H, W, occ0.device, occ0.dtype)
sampling_grid = bgrid + disp_norm

pred = F.grid_sample(occ0.unsqueeze(1), sampling_grid, mode="bilinear",
                      padding_mode="zeros", align_corners=True).squeeze(1)
pred = pred.clamp(0.0, 1.0)

# ---- score: swept-region accuracy, same harness as everything else ----
s_px, e_px = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
plate_px = 0.04 / 0.128 * W
region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

m = metrics(pred, occ1, occ0, region=region)
print("\nGROUND-TRUTH FLOW RECONSTRUCTION (ceiling):")
print(f"  accuracy = {m['accuracy']:.4f}")
print(m)

torch.save(dict(flow_target=flow_target, mask=mask, pred=pred, occ0=occ0, occ1=occ1,
                 region=region, s_px=s_px, e_px=e_px),
           "/tmp/claude-1000/-home-alon-Code-pile-manipulation/0200a1cb-398e-4e01-9553-2dce13cb540b/scratchpad/gate_gt_flow_cache.pt")
print("saved cache")
