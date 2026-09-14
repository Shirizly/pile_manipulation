"""GO/NO-GO probe for docs/experimental_design/hybrid_linear_latent.md, Stage 2.

Question: can a FiLM-conditioned latent predictor beat the trivial dz=0
(identity) baseline on the spec's Stage-2 target -- the noise-free, purely
geometric analytic push transform T_a(X)?

Data: a small subsample (6 files, ~512 transitions each) of
Genesis/data/overnight_randlen/piled/cube/n20/size0.005/layers4 -- raw
*_data.pt schema (states, states_, p_starts, p_stops, angles), NOT the
registry/PileSweepData path (that rasteriser draws cv2 boxes at a
resolution_scale-dependent pixel scale; the task asks for stage-1/2's
particles_to_occupancy + footprint_radius convention at grid 64, bounds
+-0.064 m, matching occupancy_foresight.py's cube view exactly, so a probe
result is comparable to that existing convention rather than a third one).

T_a(X): "purely geometric" push transform. Rather than a chained
to_push_frame -> shift-along-canonical-x -> from_push_frame (which is
mathematically a translation of X by the world push vector, since the
canonical frame's +x axis IS the push direction by construction -- shifting
canonical content by the push length along canonical +x is identical to
translating the original image by length*(cos phi, sin phi) = end-start),
this computes that same translation directly with one warp_affine_occ call:
less code, identical result, no rotation/corner-loss caveat to worry about
because there IS no rotation in a pure translation. Invalid (shifted-out)
region is masked the same way push_frame_validity_mask does it for the
canonical round trip: warp a ones-field through the same transform and
threshold.

Action encoding a_t = [x_p, y_p, sin(theta_p), cos(theta_p)], per spec section 5,
using START-OF-PUSH as the reference point (p_starts), and the dataset's own
recorded plate `angles` field as theta_p (this IS the plate pose the spec's
a_t=(x_p,y_p,theta_p) means -- more accurate than re-deriving it from
[sx,sy,ex,ey] via action_to_pose, which only APPROXIMATES the plate yaw from
push direction).
"""
from __future__ import annotations

import glob
import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/alon/Code/pile_manipulation")

from transforms.functional import particles_to_occupancy
from model.NFDUNetFilm import FiLMConvBlock

torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 64
BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
CUBE_SIZE = 0.005  # from the data path: size0.005
DATA_GLOB = "Genesis/data/overnight_randlen/piled/cube/n20/size0.005/layers4/*_data.pt"
N_FILES = 6
N_STEPS = 400
BATCH = 64
LATENT_CH = 32


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_transitions(n_files=N_FILES):
    files = sorted(glob.glob(DATA_GLOB))[:n_files]
    assert files, f"no files matched {DATA_GLOB}"
    S0, S1, PS, PE, ANG = [], [], [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S0.append(d["states"][..., :3])
        S1.append(d["states_"][..., :3])
        PS.append(d["p_starts"])
        PE.append(d["p_stops"])
        ANG.append(d["angles"])
    s0 = torch.cat(S0).float()
    s1 = torch.cat(S1).float()
    ps = torch.cat(PS).float()
    pe = torch.cat(PE).float()
    ang = torch.cat(ANG).float()

    length_m = (pe[:, :2] - ps[:, :2]).norm(dim=-1)
    keep = length_m > 1e-4
    s0, s1, ps, pe, ang = s0[keep], s1[keep], ps[keep], pe[keep], ang[keep]
    print(f"loaded {len(files)} files, {len(keep)} transitions, kept {int(keep.sum())} "
          f"(dropped {int((~keep).sum())} degenerate pushes)")
    return files, s0, s1, ps, pe, ang


def actions_to_pixels(ps, pe, ws_min, ws_max, grid_res):
    """[sx,sy]/[ex,ey] metres -> (start_px, end_px) in (col, row) pixels.

    Same convention as fit_linear_foresight.py::actions_to_pixels: dim0 (row)
    <- world_x, dim1 (col) <- world_y, matching particles_to_occupancy's own
    axis assignment (grid axis 0 = 'x', stored as occ dim1; grid axis 1 =
    'y', stored as occ dim2 -- i.e. occ's "row" is x, occ's "col" is y).
    """
    H, W = int(grid_res[0]), int(grid_res[1])
    x_min, y_min = float(ws_min[0]), float(ws_min[1])
    x_max, y_max = float(ws_max[0]), float(ws_max[1])

    def to_px(vx, vy):
        row = (vx - x_min) / (x_max - x_min) * H - 0.5
        col = (vy - y_min) / (y_max - y_min) * W - 0.5
        return torch.stack([col, row], dim=-1)

    return to_px(ps[:, 0], ps[:, 1]), to_px(pe[:, 0], pe[:, 1])


def translate_occ(occ: torch.Tensor, start_px: torch.Tensor, end_px: torch.Tensor):
    """T_a(X): rigid translation of the whole occupancy field by the push
    vector (col, row) pixel displacement. Returns (T_a(X), validity_mask)."""
    B, H, W = occ.shape
    d = (end_px - start_px)  # (B,2) = (dcol, drow)
    dx_norm = 2.0 * d[:, 0] / W   # affine_grid x <-> col <-> W
    dy_norm = 2.0 * d[:, 1] / H   # affine_grid y <-> row <-> H
    theta = torch.zeros(B, 2, 3, device=occ.device, dtype=occ.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -dx_norm
    theta[:, 1, 2] = -dy_norm
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    warped = F.grid_sample(occ.unsqueeze(1), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False).squeeze(1)
    ones = torch.ones_like(occ)
    valid = F.grid_sample(ones.unsqueeze(1), grid, mode="nearest",
                           padding_mode="zeros", align_corners=False).squeeze(1)
    return warped, (valid > 0.5).float()


# ---------------------------------------------------------------------------
# Model: small encoder + FiLM residual predictor. No decoder (probe only).
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """3 downsampling stages, reduced width (8->16->32) vs spec's 32->64->128->256.
    64x64x1 -> 8x8xLATENT_CH."""
    def __init__(self, latent_ch=LATENT_CH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # 64->32
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 32->16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_ch, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 16->8
        )

    def forward(self, x):
        return self.net(x)


class FiLMPredictor(nn.Module):
    """Residual latent transition: zhat' = z + P(z, a). Reuses FiLMConvBlock."""
    def __init__(self, latent_ch=LATENT_CH, cond_dim=4, hidden=64):
        super().__init__()
        self.block1 = FiLMConvBlock(latent_ch, latent_ch, cond_dim, hidden)
        self.block2 = FiLMConvBlock(latent_ch, latent_ch, cond_dim, hidden)
        self.out = nn.Conv2d(latent_ch, latent_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # zero-init residual: P(z,a)=0 at start

    def forward(self, z, a):
        h = self.block1(z, a)
        h = self.block2(h, a)
        return self.out(h)


def main():
    files, s0, s1, ps, pe, ang = load_transitions(N_FILES)

    pitch = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
    radius = 0.5 * CUBE_SIZE / pitch

    print("rasterising occupancy...")
    X0 = particles_to_occupancy(s0.to(DEVICE), BOUNDS, (GRID, GRID), footprint_radius=radius)
    start_px, end_px = actions_to_pixels(ps, pe, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                          (BOUNDS["x_max"], BOUNDS["y_max"]), (GRID, GRID))
    start_px, end_px = start_px.to(DEVICE), end_px.to(DEVICE)
    Ta_X0, valid = translate_occ(X0, start_px, end_px)

    action_vec = torch.stack([
        ps[:, 0] / 0.064, ps[:, 1] / 0.064, torch.sin(ang), torch.cos(ang),
    ], dim=-1).to(DEVICE)

    n = X0.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_holdout = max(1, int(0.2 * n))
    holdout_idx = idx[:n_holdout]
    train_idx = idx[n_holdout:]
    print(f"n={n} train={len(train_idx)} holdout={len(holdout_idx)}")

    enc = Encoder().to(DEVICE)
    pred = FiLMPredictor().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)

    def step_batch(bidx, train=True):
        x = X0[bidx].unsqueeze(1)
        tx = Ta_X0[bidx].unsqueeze(1)
        a = action_vec[bidx]
        z = enc(x)
        with torch.set_grad_enabled(train):
            zt = enc(tx)  # tx already has invalid (shifted-out) pixels zeroed by grid_sample's zero padding
        dz = pred(z, a)
        zhat = z + dz
        loss_model = F.mse_loss(zhat, zt)
        loss_base = F.mse_loss(z, zt)
        # Anti-collapse regulariser (VICReg-style variance hinge over the batch,
        # per channel): without this the encoder is free to minimise
        # loss_model AND loss_base to ~0 together by collapsing to a near-constant
        # z, which is exactly the trap the task asked us to guard against (see
        # RESULTS.md). This is a probe-only patch to get a MEANINGFUL ratio, not
        # part of the original spec.
        std_z = z.flatten(2).std(dim=0).mean()
        loss_var = F.relu(0.1 - std_z)
        return loss_model, loss_base, z, loss_var

    print("training...")
    for step in range(N_STEPS):
        opt.zero_grad()
        bidx = train_idx[torch.randint(0, len(train_idx), (BATCH,))]
        loss_model, loss_base, z, loss_var = step_batch(bidx, train=True)
        (loss_model + 1.0 * loss_var).backward()
        opt.step()
        if step % 50 == 0 or step == N_STEPS - 1:
            print(f"step {step:4d}  loss_model={loss_model.item():.6f}  "
                  f"loss_base(dz=0)={loss_base.item():.6f}  ratio={loss_model.item()/max(loss_base.item(),1e-12):.4f}")

    with torch.no_grad():
        train_model, train_base, z_train, _ = step_batch(train_idx, train=False)
        hold_model, hold_base, z_hold, _ = step_batch(holdout_idx, train=False)
        z_all = torch.cat([z_train, z_hold], dim=0)
        z_std = z_all.std(dim=0).mean().item()
        z_meanabs = z_all.abs().mean().item()

    results = {
        "n_transitions_total": int(n),
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
        "n_files": len(files),
        "encoder_width": "8->8->16->16->32->32 (3 downsample stages, 64x64->8x8x32); "
                          "reduced from spec's 32->64->128->256 for probe speed",
        "latent_shape": list(z_all.shape[1:]),
        "footprint_radius_px": float(radius),
        "train": {
            "loss_model": train_model.item(),
            "loss_baseline_dz0": train_base.item(),
            "ratio": train_model.item() / max(train_base.item(), 1e-12),
        },
        "holdout": {
            "loss_model": hold_model.item(),
            "loss_baseline_dz0": hold_base.item(),
            "ratio": hold_model.item() / max(hold_base.item(), 1e-12),
        },
        "latent_collapse_diagnostic": {
            "z_std_mean_over_channels": z_std,
            "z_meanabs": z_meanabs,
            "note": "if z_std collapses toward 0, both loss_model and loss_baseline "
                    "go to 0 together and the ratio is meaningless (encoder collapse).",
        },
    }
    with open("/home/alon/Code/pile_manipulation/experiments/temp/latent-killprobe/results_probe.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
