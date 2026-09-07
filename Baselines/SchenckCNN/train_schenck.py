"""Baselines/SchenckCNN/train_schenck.py -- standalone training loop for the
Schenck single-net ablation.

Not routed through `training.trainer.Trainer` (unlike NFD's `train_nfd.py`,
which registers into the shared dataset/model registries because it needed
`unet-modular`'s existing Trainer plumbing): SchenckCNN is a small
from-scratch architecture with a plain L2 loss and no physics conditioning,
so a ~60-line manual loop is simpler and avoids registry surface area this
baseline doesn't need. `Baselines/common/data.py::load_cell` (registry-backed
occupancy view, byte-identical to the register's own numbers) and
`Baselines/SchenckCNN/action_encoding.py` (reused NFD action-rasterisation
primitive) are still reused unchanged -- only the training loop itself is
new code.

Action channels are precomputed ONCE for the whole pooled train set (a
single batched `draw_plate_soft` call over all ~23k transitions is cheap and
avoids re-rasterising every epoch).

Usage:
    PYTHONPATH=. python Baselines/SchenckCNN/train_schenck.py --epochs 3
    PYTHONPATH=. python Baselines/SchenckCNN/train_schenck.py --epochs 100 \\
        --out Baselines/SchenckCNN/runs/schenck.pth
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from Baselines.common.data import load_cell
from Baselines.SchenckCNN.action_encoding import build_input
from Baselines.SchenckCNN.model import SchenckCNN

TRAIN_CFG = "configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the Schenck single-net ablation.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)  # paper's stated value, SPEC.md sec 3
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", type=str, default="Baselines/SchenckCNN/runs/schenck.pth")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    print(f"device={device}")

    t0 = time.time()
    cell = load_cell(TRAIN_CFG, "train")
    H, W = cell.H, cell.W
    print(f"loaded pooled train: {cell.occ0.shape[0]} transitions, grid {H}x{W} "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    x = build_input(cell.occ0, cell.p_start, cell.p_stop, cell.angle, cell.raw, H, W)
    y = cell.occ1
    print(f"built inputs: x={tuple(x.shape)} y={tuple(y.shape)} ({time.time() - t0:.1f}s)")

    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SchenckCNN(in_channels=x.shape[1], width=args.width, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} params, in_channels={x.shape[1]}, "
          f"{args.n_layers} conv layers @ width {args.width}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  loss={avg:.6f}  ({dt:.1f}s)")

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.out)
            print(f"  wrote checkpoint -> {args.out}")

    print("done.")


if __name__ == "__main__":
    main()
