import sys; sys.path.insert(0, "/home/alon/Code/pile_manipulation")
import torch
import torch.nn.functional as F
from model.flow_nfd.supervised import _rotate_flow_components

d = torch.load("/home/alon/Code/pile_manipulation/experiments/EXP-0025-flow-warp-nfd-pilot/"
               "runs/RUN-0011-gt-flow-ceiling-gate/gate_gt_flow_cache.pt", weights_only=False)
occ0, flow_target, pred = d["occ0"], d["flow_target"], d["pred"]
N, H, W = occ0.shape

def base_grid(B, H, W, device, dtype):
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

def warp(occ, flow):
    B, H, W = occ.shape
    disp_x = flow[:, 0] * (2.0 / max(W - 1, 1))
    disp_y = flow[:, 1] * (2.0 / max(H - 1, 1))
    grid = base_grid(B, H, W, occ.device, occ.dtype) + torch.stack([disp_x, disp_y], dim=-1)
    return F.grid_sample(occ.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True).squeeze(1).clamp(0, 1)

idxs = [0, 1, 2, 100, 500]
worst = 0.0
for i in idxs:
    occ0_i = occ0[i:i+1]
    flow_i = flow_target[i:i+1]
    pred_i = pred[i:i+1]
    for k in range(4):
        for flipped in (False, True):
            def sp(x):
                y = torch.rot90(x, k, dims=(-2, -1))
                if flipped:
                    y = torch.flip(y, dims=[-1])
                return y
            occ0_aug = sp(occ0_i)
            f_sp = sp(flow_i)
            dc_o, dr_o = _rotate_flow_components(f_sp[:, 0], f_sp[:, 1], k, flipped)
            flow_aug = torch.stack([dc_o, dr_o], dim=1)

            pred_aug = warp(occ0_aug, flow_aug)
            pred_expected = sp(pred_i)  # spatially transform the ORIGINAL reconstruction
            diff = (pred_aug - pred_expected).abs().max().item()
            worst = max(worst, diff)
            print(f"i={i} k={k} flip={flipped}  max_abs_diff={diff:.6f}")

print("\nWORST max_abs_diff across all checked (sample, k, flip):", worst)
print("PASS" if worst < 1e-4 else "FAIL")

# Negative control: skip the vector-component rotation (only spatially permute,
# as a naive 'carry along like an image' bug would) -- should show a LARGE
# mismatch, proving the check has power to catch the bug being guarded against.
print("\n--- negative control: vectors carried WITHOUT rotating components ---")
worst_bug = 0.0
for i in idxs[:2]:
    occ0_i = occ0[i:i+1]; flow_i = flow_target[i:i+1]; pred_i = pred[i:i+1]
    for k in range(4):
        for flipped in (False, True):
            def sp(x):
                y = torch.rot90(x, k, dims=(-2, -1))
                if flipped:
                    y = torch.flip(y, dims=[-1])
                return y
            occ0_aug = sp(occ0_i)
            flow_aug_buggy = sp(flow_i)  # BUG: no component rotation
            pred_aug = warp(occ0_aug, flow_aug_buggy)
            pred_expected = sp(pred_i)
            diff = (pred_aug - pred_expected).abs().max().item()
            worst_bug = max(worst_bug, diff)
print("worst max_abs_diff with the bug (should be LARGE):", worst_bug)
