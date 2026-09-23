"""EXP-0025 extension -- STEP 2: train a flow head with DIRECT particle-
correspondence supervision on top of the photometric loss, instead of the
photometric loss alone (`model/flow_nfd/lib.py`, the original pilot).

Standalone script (not the generic Trainer): the target/mask construction
needs raw `states`/`states_` per sample, which the registered
`nfd-genesis-3ch` dataset type does not expose downstream of
`EulerianDatasetWrapper` (see `docs/CODEMAP.md`'s `Baselines/common/data.py`
entry -- the same "get both views from the same raw PileSweepData index"
pattern is reused here, not re-derived).

Two cells, run with --cell:
  supervised           -- photometric MSE + masked direct flow supervision.
  supervised_masked    -- + a magnitude penalty shrinking flow toward zero
                           where there is no material, weighted by
                           (1 - occ0) (task brief's literal suggestion),
                           itself masked (background only) so it cannot
                           fight the foreground supervision term.

Same subset/recipe as every other EXP-0025 cell (see EXPERIMENT.md):
n20_L20mm_train_pilotsubset, 40 epochs, batch 32, Adam lr 1e-4, grad clip
1.0. Augmentation is DROPPED here (deviation, noted in results/
supervised_flow.md) to keep this pilot script small; all EXP-0025 photometric
cells used x8 flip/rotation augmentation, so this comparison is directional,
not a like-for-like rerun of those numbers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/alon/Code/pile_manipulation")

from Baselines.NFD.nfd_lib import PileSweepData3Ch
from model.flow_nfd.supervised import (
    build_flow_targets_for_split, augment_flow_batch_x8,
)
from model.UNetModels_modular import UNet
from physics.normalization import PhysicsBounds

PATHS = ["slates_multistep/n20_L20mm_train_pilotsubset"]
MAX_DISP_PX = 24.0


def _base_grid(B, H, W, device, dtype):
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    g = torch.stack([gx, gy], dim=-1)
    return g.unsqueeze(0).expand(B, -1, -1, -1)


def load_split(split: str):
    raw = PileSweepData3Ch(
        paths=PATHS, split=split, val_pct=15, test_pct=15,
        resolution_scale=0.5, physics_bounds=PhysicsBounds.default(),
        min_push_length_m=0.0198,
    )
    n = len(raw)
    inputs = torch.stack([raw[i][0][0] for i in range(n)])   # (n,3,H,W)
    occ1 = torch.stack([raw[i][1] for i in range(n)])         # (n,H,W)
    states = torch.empty((n, 20, 7), dtype=torch.float32)
    states_ = torch.empty((n, 20, 7), dtype=torch.float32)
    for i in range(n):
        idx = raw._resolve_idx(i)
        r = raw._run_lookup[idx]
        s = idx - raw._offsets[r]
        run = raw.runs[r]
        states[i] = run["states"][s]
        states_[i] = run["states_"][s]
    flow_target, mask = build_flow_targets_for_split(
        states, states_, occ1, float(raw.to_pxl), raw.ctr_in_PXL)
    occ0 = inputs[:, 0]
    return dict(inputs=inputs, occ0=occ0, occ1=occ1, flow_target=flow_target, mask=mask)


def build_model():
    structure = dict(
        features=[4, 8, 16], in_channels=3, out_channels=2,
        kernel_size=3, final_kernel_size=1, activation="relu",
        residual=False, bottleneck_type="None", bottleneck_kwargs={},
    )
    return UNet(structure)


def forward_flow(model, inputs, occ0):
    B, H, W = occ0.shape
    raw = model(inputs)                       # (B,2,H,W)
    disp_px = torch.tanh(raw) * MAX_DISP_PX    # ch0=dcol(x/col), ch1=drow(y/row)
    disp_x_norm = disp_px[:, 0] * (2.0 / max(W - 1, 1))
    disp_y_norm = disp_px[:, 1] * (2.0 / max(H - 1, 1))
    disp_norm = torch.stack([disp_x_norm, disp_y_norm], dim=-1)
    grid = _base_grid(B, H, W, occ0.device, occ0.dtype) + disp_norm
    occ_pred = F.grid_sample(occ0.unsqueeze(1), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=True).squeeze(1)
    return occ_pred.clamp(0.0, 1.0), disp_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["supervised", "supervised_masked"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda-flow", type=float, default=1.0,
                    help="weight of the direct flow-supervision term (in px^2 units)")
    ap.add_argument("--lambda-bg", type=float, default=0.05,
                    help="cell 'supervised_masked' only: background magnitude penalty weight")
    ap.add_argument("--augment", action="store_true",
                    help="apply the x8 rotation/flip augmentation to the TRAIN split only "
                         "(matching EXP-0025's photometric cells), with the flow-target "
                         "VECTOR correctly rotated per component -- see "
                         "model/flow_nfd/supervised.py::augment_flow_batch_x8 and "
                         "code/verify_flow_augmentation.py for the equivariance check.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    print("[data] loading train split...", flush=True)
    train = load_split("train")
    print(f"[data] {train['inputs'].shape[0]} train samples (pre-augmentation)", flush=True)
    if args.augment:
        aug = augment_flow_batch_x8(train["inputs"], train["occ1"],
                                     train["flow_target"], train["mask"])
        train = dict(inputs=aug["inputs"], occ0=aug["inputs"][:, 0],
                     occ1=aug["occ1"], flow_target=aug["flow_target"], mask=aug["mask"])
        print(f"[data] {train['inputs'].shape[0]} train samples (x8 augmented)", flush=True)
    print("[data] loading val split...", flush=True)
    val = load_split("val")
    print(f"[data] {val['inputs'].shape[0]} val samples", flush=True)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_train = train["inputs"].shape[0]
    batch_size = 32
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    def run_epoch(split_data, train_mode: bool):
        n = split_data["inputs"].shape[0]
        idx_order = torch.randperm(n) if train_mode else torch.arange(n)
        total_photo, total_flow, total_bg, total_n = 0.0, 0.0, 0.0, 0
        model.train(train_mode)
        for start in range(0, n, batch_size):
            sel = idx_order[start:start + batch_size]
            inp = split_data["inputs"][sel].to(device)
            occ0 = split_data["occ0"][sel].to(device)
            occ1 = split_data["occ1"][sel].to(device)
            ftgt = split_data["flow_target"][sel].to(device)
            fmask = split_data["mask"][sel].to(device)

            with torch.set_grad_enabled(train_mode):
                occ_pred, disp_px = forward_flow(model, inp, occ0)
                photo_loss = F.mse_loss(occ_pred, occ1)

                fmask_sum = fmask.sum().clamp_min(1.0)
                flow_err = (disp_px - ftgt) ** 2
                flow_loss = (flow_err.sum(dim=1) * fmask).sum() / fmask_sum / (24.0 ** 2)

                bg_loss = torch.zeros((), device=device)
                if args.cell == "supervised_masked":
                    bg_w = (1.0 - occ0)
                    bg_mag = (disp_px ** 2).sum(dim=1) * bg_w
                    bg_loss = bg_mag.sum() / bg_w.sum().clamp_min(1.0) / (24.0 ** 2)

                loss = photo_loss + args.lambda_flow * flow_loss + args.lambda_bg * bg_loss

            if train_mode:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            bs = inp.shape[0]
            total_photo += float(photo_loss) * bs
            total_flow += float(flow_loss) * bs
            total_bg += float(bg_loss) * bs
            total_n += bs
        return total_photo / total_n, total_flow / total_n, total_bg / total_n

    t0 = time.time()
    for epoch in range(args.epochs):
        tr_photo, tr_flow, tr_bg = run_epoch(train, True)
        with torch.no_grad():
            va_photo, va_flow, va_bg = run_epoch(val, False)
        va_total = va_photo + args.lambda_flow * va_flow + args.lambda_bg * va_bg
        print(f"[epoch {epoch+1}/{args.epochs}] "
              f"train photo={tr_photo:.5f} flow={tr_flow:.5f} bg={tr_bg:.5f} | "
              f"val photo={va_photo:.5f} flow={va_flow:.5f} bg={va_bg:.5f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if va_total < best_val:
            best_val = va_total
            torch.save(model.state_dict(), out_dir / "unet_best.pth")
    torch.save(model.state_dict(), out_dir / "unet_last.pth")
    print(f"[done] best_val={best_val:.5f} saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
