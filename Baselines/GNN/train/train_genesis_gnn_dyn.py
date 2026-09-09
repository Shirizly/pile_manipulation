"""Baselines/GNN/train/train_genesis_gnn_dyn.py -- single-step training loop
for the dynamic-resolution GNN (`model/gnn_dyn.py::PropNetDiffDenModel`) on
our pooled Genesis slates_multistep data (N=20 fixed, no resolution
regressor). See SPEC.md sections 6-8 for the recipe this follows.

Does NOT touch `Baselines/GNN/train/train_gnn_dyn.py` (vendored,
PyFlex/FlexEnv-specific) -- this is a new script for a fundamentally
different I/O shape.

Usage (through the GPU lock, per ORCHESTRATION_LOG.md hardware constraint):

    Baselines/common/gpu_lock.sh env PYTHONPATH=. python \\
        Baselines/GNN/train/train_genesis_gnn_dyn.py \\
        --epochs 300 --batch-size 128 --out-dir Baselines/GNN/runs
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Baselines.GNN.dataset.dataset_genesis_gnn import GenesisGNNDataset, collate
from Baselines.GNN.geometry import ADJ_THRESH, PARTICLE_DENS, compute_s_delta, model_config
from model.gnn_dyn import PropNetDiffDenModel

TRAIN_ROOTS = [
    "Genesis/data/slates_multistep/n20_L20mm_train",
    "Genesis/data/slates_multistep/n20_L40mm_train",
]
VAL_ROOTS = [
    "Genesis/data/slates_multistep/n20_L20mm_eval",
    "Genesis/data/slates_multistep/n20_L40mm_eval",
]


def run_epoch(model, loader, device, adj_thresh, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, total_n = 0.0, 0
    for batch in loader:
        s_cur = batch["s_cur"].to(device)
        s_next = batch["s_next"].to(device)
        p_start = batch["p_start"].to(device)
        p_stop = batch["p_stop"].to(device)
        B, N, _ = s_cur.shape

        s_delta = compute_s_delta(s_cur, p_start, p_stop)
        a_cur = torch.zeros(B, N, device=device)
        dens = torch.full((B,), PARTICLE_DENS, device=device)

        with torch.set_grad_enabled(train):
            s_pred = model.predict_one_step(a_cur, s_cur, s_delta, dens)
            loss = F.mse_loss(s_pred, s_next)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach()) * B
        total_n += B
    return total_loss / max(total_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adj-thresh", type=float, default=ADJ_THRESH)
    ap.add_argument("--out-dir", default="Baselines/GNN/runs")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                     help="tiny run (2 epochs) to check the pipeline end-to-end before "
                          "committing to a long run, per ORCHESTRATION_LOG.md priority order.")
    ap.add_argument("--train-roots", default=None,
                     help="comma-separated list of directories of _*_data.pt files, "
                          "overriding the module-level TRAIN_ROOTS default (L20mm+L40mm "
                          "pooled). Added for EXP-0030 (overnight_randlen corpus) without "
                          "changing the default behaviour for any existing invocation.")
    ap.add_argument("--val-roots", default=None,
                     help="comma-separated list, overriding VAL_ROOTS. See --train-roots.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    train_roots = args.train_roots.split(",") if args.train_roots else TRAIN_ROOTS
    val_roots = args.val_roots.split(",") if args.val_roots else VAL_ROOTS
    train_ds = GenesisGNNDataset(train_roots)
    val_ds = GenesisGNNDataset(val_roots)
    print(f"train examples: {len(train_ds)}  val examples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=0)

    cfg = model_config(adj_thresh=args.adj_thresh)
    model = PropNetDiffDenModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}  adj_thresh: {args.adj_thresh}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = 2 if args.smoke else args.epochs
    best_val = float("inf")
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        tr_loss = run_epoch(model, train_loader, device, args.adj_thresh, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, args.adj_thresh, optimizer=None)
        scheduler.step()
        dt = time.time() - t0
        print(f"epoch {epoch:4d}  train_mse {tr_loss:.6e}  val_mse {val_loss:.6e}  "
              f"({dt:.1f}s elapsed)")
        history.append({"epoch": epoch, "train_mse": tr_loss, "val_mse": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "val_mse": val_loss}, os.path.join(args.out_dir, "ckpt_best.pth"))

        if epoch % args.ckpt_every == 0 or epoch == epochs - 1:
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "val_mse": val_loss}, os.path.join(args.out_dir, "ckpt_last.pth"))
            with open(os.path.join(args.out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"done. best val_mse={best_val:.6e}  total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
