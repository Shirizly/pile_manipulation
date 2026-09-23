"""experiments/temp/desc-mlp/train_mlp.py -- train ONE small MLP over D-all-local
(94-dim) push-frame descriptors + action features [x,y,cos,sin,length] to
predict the next-step descriptor vector, replacing the per-push-length-bin
switched-linear family (weights/MODEL-0002-descriptor-only-D-all-local) with
a single network.

Strict capacity ceiling: params <= n_train_samples / 10. n_train = 98304
(Genesis/data/overnight_randlen_train, D-all-local built by desc_features.py),
so the ceiling is 9830 params. A single hidden layer H=48 net (99->48->94,
with biases) has 194*48+94 = 9406 params -- under budget, chosen up front
rather than swept up to the ceiling (a smaller net that does as well is
preferred per task instructions).

Trains in z-scored target space (train phi_t1 mean/std, exactly
`descriptor_accuracy`'s own normalisation) so the loss does not let
high-variance blocks (DFT) dominate low-variance ones (mass) -- the
"pooling across unequal scales" trap this repo has hit before.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from metric import descriptor_accuracy, fit_stats, zscore

CACHE = "experiments/temp/desc-mlp/cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def action_features(d):
    return np.stack([d["x"], d["y"], np.cos(d["angle"]), np.sin(d["angle"]),
                      d["length_m"]], axis=1).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, act="relu"):
        super().__init__()
        act_layer = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[act]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act_layer(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def load_split(split):
    d = np.load(f"{CACHE}/desc_{split}.npz")
    phi_t = d["phi_t"].astype(np.float32)
    phi_t1 = d["phi_t1"].astype(np.float32)
    act = action_features(d)
    return phi_t, phi_t1, act


def train_one(hidden, act_name, weight_decay, lr, epochs, seed,
              Xtr, Ytr_z, Xte, phi_t_te, phi_t1_te, mu_y, sigma_y, in_dim, out_dim,
              patience=15, verbose=False):
    torch.manual_seed(seed)
    model = MLP(in_dim, hidden, out_dim, act_name).to(DEVICE)
    n_p = n_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr_z).to(DEVICE)
    Xte_t = torch.from_numpy(Xte).to(DEVICE)
    n = Xtr_t.shape[0]
    batch = 4096
    best_acc, best_state, best_epoch = -1e9, None, -1
    bad = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tot_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            opt.step()
            tot_loss += float(loss) * xb.shape[0]
        tot_loss /= n

        model.eval()
        with torch.no_grad():
            pred_z_te = model(Xte_t).cpu().numpy()
        pred_raw_te = pred_z_te * sigma_y + mu_y
        acc = descriptor_accuracy(pred_raw_te, phi_t1_te, phi_t_te, mu_y, sigma_y)
        if verbose and ep % 10 == 0:
            print(f"    ep {ep:3d} train_z_mse {tot_loss:.4f} test_descacc {acc:.4f}")
        if acc > best_acc:
            best_acc, best_epoch = acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    return model, n_p, best_acc, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    print("loading caches...")
    phi_t_tr, phi_t1_tr, act_tr = load_split("train")
    phi_t_te, phi_t1_te, act_te = load_split("test")
    n_train = phi_t_tr.shape[0]
    max_params = n_train // 10
    print(f"n_train={n_train}  n_test={phi_t_te.shape[0]}  max_params(1:10)={max_params}")

    Xtr_raw = np.concatenate([phi_t_tr, act_tr], axis=1)
    Xte_raw = np.concatenate([phi_t_te, act_te], axis=1)
    in_dim, out_dim = Xtr_raw.shape[1], phi_t1_tr.shape[1]

    mu_x, sigma_x = fit_stats(Xtr_raw)
    mu_y, sigma_y = fit_stats(phi_t1_tr)
    Xtr = zscore(Xtr_raw, mu_x, sigma_x).astype(np.float32)
    Xte = zscore(Xte_raw, mu_x, sigma_x).astype(np.float32)
    Ytr_z = zscore(phi_t1_tr, mu_y, sigma_y).astype(np.float32)

    # persistence / random baselines on the SAME held-out test rows
    persist_acc = descriptor_accuracy(phi_t_te, phi_t1_te, phi_t_te, mu_y, sigma_y)
    rng = np.random.default_rng(0)
    rand_pred = phi_t1_tr[rng.integers(0, n_train, size=phi_t_te.shape[0])]
    random_acc = descriptor_accuracy(rand_pred, phi_t1_te, phi_t_te, mu_y, sigma_y)
    print(f"baseline descriptor_accuracy: persistence={persist_acc:.4f} (0 by construction) "
          f"random={random_acc:.4f}")

    # ---- small sweep: hidden size / activation / weight decay ----
    configs = [
        dict(hidden=48, act_name="relu", weight_decay=1e-4, lr=1e-3, tag="H48_relu_wd1e-4"),
        dict(hidden=48, act_name="tanh", weight_decay=1e-4, lr=1e-3, tag="H48_tanh_wd1e-4"),
        dict(hidden=32, act_name="relu", weight_decay=1e-4, lr=1e-3, tag="H32_relu_wd1e-4"),
        dict(hidden=48, act_name="relu", weight_decay=1e-3, lr=1e-3, tag="H48_relu_wd1e-3"),
        dict(hidden=48, act_name="relu", weight_decay=0.0, lr=1e-3, tag="H48_relu_wd0"),
    ]
    sweep_results = []
    best = None
    t0 = time.time()
    for cfg in configs:
        model, n_p, acc, ep = train_one(
            cfg["hidden"], cfg["act_name"], cfg["weight_decay"], cfg["lr"], args.epochs,
            seed=0, Xtr=Xtr, Ytr_z=Ytr_z, Xte=Xte, phi_t_te=phi_t_te, phi_t1_te=phi_t1_te,
            mu_y=mu_y, sigma_y=sigma_y, in_dim=in_dim, out_dim=out_dim, verbose=True)
        assert n_p <= max_params, f"{cfg['tag']}: {n_p} params exceeds ceiling {max_params}"
        print(f"[{cfg['tag']}] params={n_p} best_epoch={ep} test_descriptor_accuracy={acc:.4f} "
              f"({time.time() - t0:.1f}s elapsed)")
        sweep_results.append(dict(tag=cfg["tag"], n_params=n_p, best_epoch=ep, test_descriptor_accuracy=acc))
        if best is None or acc > best[0]:
            best = (acc, cfg, model, n_p)

    best_acc, best_cfg, best_model, best_n_p = best
    print(f"\nBEST: {best_cfg['tag']}  params={best_n_p}  test_descriptor_accuracy={best_acc:.4f}")

    ckpt = {
        "state_dict": best_model.state_dict(),
        "config": best_cfg,
        "in_dim": in_dim, "out_dim": out_dim,
        "mu_x": mu_x, "sigma_x": sigma_x, "mu_y": mu_y, "sigma_y": sigma_y,
        "n_train": n_train, "n_test": phi_t_te.shape[0], "n_params": best_n_p,
        "max_params_1to10": max_params,
        "descriptor_basis": "D-all-local (94-dim, push-frame; experiments/temp/dmdc-lenbins/descriptors_d.py::slices_d), "
                             "IDENTICAL to weights/MODEL-0002-descriptor-only-D-all-local's basis",
        "action_features": "[x, y, cos(theta), sin(theta), length_m] (push start world xy, blade yaw, push length)",
        "note": "single global MLP (not per-bin), trained on overnight_randlen_train, "
                "model-selected + reported on overnight_randlen_test.",
    }
    torch.save(ckpt, "experiments/temp/desc-mlp/mlp_checkpoint.pt")
    print("wrote experiments/temp/desc-mlp/mlp_checkpoint.pt")

    with open("experiments/temp/desc-mlp/results_train.json", "w") as fh:
        json.dump({
            "n_train": n_train, "n_test": int(phi_t_te.shape[0]), "max_params_1to10": int(max_params),
            "persistence_descriptor_accuracy": persist_acc,
            "random_descriptor_accuracy": random_acc,
            "sweep": sweep_results,
            "best": {"tag": best_cfg["tag"], "n_params": int(best_n_p), "descriptor_accuracy": best_acc},
        }, fh, indent=2)
    print("wrote experiments/temp/desc-mlp/results_train.json")


if __name__ == "__main__":
    main()
