"""Stage 3 / Stage A: encoder E_phi + decoder D_omega by reconstruction.

Trains X -> D(E(X)) on the 32x32 canonical occupancy grids already cached by
stage 2 (experiments/temp/hybrid-vis-desc/cache/{train,test}_cache.pt --
canon0/canon1 fields, `fit_linear_foresight.canonicalise` convention, same
80/20 file split seed 0 stratified by spawn mode as stages 1-2). Reusing that
cache (rather than rebuilding from Genesis/data/overnight_randlen) saves the
~80s+ rasterisation pass and guarantees byte-identical canonical frames to
what stage 2 scored, which matters for a fair before/after comparison.

Reduced width per stage3_plan.md's GPU sizing note (halve channels rather
than drop resolution, to preserve push_frame_transform's frame geometry):
encoder 1->16->32->64 (32x32 -> 4x4x64), FC bottleneck to a POOLED latent
vector of dim LATENT_DIM (default 64) -- much smaller than the 1024-dim raw
pixel state stage 2 used, by design (that's the point of a learned latent).
Decoder is the mirror: FC LATENT_DIM -> 4x4x64 -> deconv -> 32x32x1.

Anti-collapse: reconstruction loss itself is a strong anti-collapse signal
(a constant latent cannot reconstruct varying inputs), but per
latent-killprobe/RESULTS.md's finding that plain L_latent objectives CAN
collapse, we still add the VICReg-style per-channel variance hinge on z
(weight 0.1, secondary to reconstruction) and report z_std every log step as
the diagnostic requested.
"""
from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc/cache"
OUT_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/latent-switched"

LATENT_DIM = 64
BATCH = 256
N_STEPS = 6000
LR = 1e-3
VIC_WEIGHT = 0.1
VIC_TARGET_STD = 0.1


class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 32->16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16->8
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8->4
        )
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        # x: (N,32,32) -> (N,1,32,32)
        h = self.conv(x.unsqueeze(1))
        h = h.flatten(1)
        z = self.fc(h)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 4->8
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 8->16
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),                          # 16->32
        )

    def forward(self, z):
        h = self.fc(z).reshape(-1, 64, 4, 4)
        out = self.deconv(h)
        return out.squeeze(1)  # (N,32,32) logits


def vicreg_hinge(z, target_std=VIC_TARGET_STD):
    std = z.std(dim=0)
    return F.relu(target_std - std).mean(), std.mean().item()


def main():
    t0 = time.time()
    train = torch.load(f"{CACHE_DIR}/train_cache.pt", map_location="cpu")
    test = torch.load(f"{CACHE_DIR}/test_cache.pt", map_location="cpu")

    # pool canon0 and canon1 together as the reconstruction corpus (autoencoder,
    # not action-dependent -- both timesteps are just "an occupancy grid").
    X_train = torch.cat([train["canon0"], train["canon1"]], dim=0)
    X_test = torch.cat([test["canon0"], test["canon1"]], dim=0)
    print(f"reconstruction corpus: train {X_train.shape[0]} holdout {X_test.shape[0]}  "
          f"({time.time()-t0:.1f}s to load)")

    E = Encoder().to(DEVICE)
    D = Decoder().to(DEVICE)
    opt = torch.optim.Adam(list(E.parameters()) + list(D.parameters()), lr=LR)

    N = X_train.shape[0]
    X_train_dev = X_train.to(DEVICE)
    log = []
    for step in range(N_STEPS):
        idx = torch.randint(0, N, (BATCH,), device=DEVICE)
        x = X_train_dev[idx]
        z = E(x)
        xhat = D(z)
        recon_loss = F.binary_cross_entropy_with_logits(xhat, x)
        vic_loss, z_std = vicreg_hinge(z)
        loss = recon_loss + VIC_WEIGHT * vic_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0 or step == 0:
            with torch.no_grad():
                mse = F.mse_loss(torch.sigmoid(xhat), x).item()
            print(f"step {step+1}/{N_STEPS}  recon_bce {recon_loss.item():.4f}  "
                  f"mse {mse:.4f}  z_std {z_std:.4f}  ({time.time()-t0:.1f}s)", flush=True)
            log.append(dict(step=step + 1, recon_bce=recon_loss.item(), mse=mse, z_std=z_std))

    # final eval on holdout
    E.eval(); D.eval()
    with torch.no_grad():
        Xt = X_test.to(DEVICE)
        recon_mse_list, z_std_list = [], []
        for i in range(0, Xt.shape[0], 2048):
            xb = Xt[i:i + 2048]
            zb = E(xb)
            xhatb = torch.sigmoid(D(zb))
            recon_mse_list.append(F.mse_loss(xhatb, xb, reduction="sum").item())
            z_std_list.append(zb.std(dim=0).mean().item() * xb.shape[0])
        holdout_mse = sum(recon_mse_list) / (Xt.numel())
        holdout_zstd = sum(z_std_list) / Xt.shape[0]

        # also report train-set z_std/mse consistently
        Xtr = X_train.to(DEVICE)
        tr_mse_list, tr_zstd_list = [], []
        for i in range(0, Xtr.shape[0], 2048):
            xb = Xtr[i:i + 2048]
            zb = E(xb)
            xhatb = torch.sigmoid(D(zb))
            tr_mse_list.append(F.mse_loss(xhatb, xb, reduction="sum").item())
            tr_zstd_list.append(zb.std(dim=0).mean().item() * xb.shape[0])
        train_mse = sum(tr_mse_list) / (Xtr.numel())
        train_zstd = sum(tr_zstd_list) / Xtr.shape[0]

    print(f"\nFINAL train recon MSE {train_mse:.5f}  z_std {train_zstd:.5f}")
    print(f"FINAL holdout recon MSE {holdout_mse:.5f}  z_std {holdout_zstd:.5f}")

    ckpt_path = f"{OUT_DIR}/encoder_decoder.pt"
    torch.save(dict(encoder=E.state_dict(), decoder=D.state_dict(),
                     latent_dim=LATENT_DIM), ckpt_path)
    print(f"wrote {ckpt_path}")

    with open(f"{OUT_DIR}/stage_a_log.json", "w") as f:
        json.dump(dict(log=log, train_mse=train_mse, train_zstd=train_zstd,
                        holdout_mse=holdout_mse, holdout_zstd=holdout_zstd,
                        latent_dim=LATENT_DIM, n_steps=N_STEPS, batch=BATCH),
                   f, indent=2)
    print(f"total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
