"""Stage 3 / Stages B+C: switched-linear operator on
state = [learned encoder latent z_g || 94-dim analytic descriptors d],
decoded back to image space through the frozen Stage-A decoder and scored
with the repo's real `accuracy` metric, exactly mirroring
experiments/temp/hybrid-vis-desc/fit_hybrid.py's cells/mask/metric so the
two stages are directly comparable.

Reuses the SAME stage-2 cache (canon0/1, desc0/1, occ0/1, action, length_m --
80/20 file split seed 0 stratified by spawn mode, same 6-bin EXP-0003
scheme) so the only thing that changes between stage 2 and stage 3 is the
visual representation: 1024-dim raw canonical pixels -> LATENT_DIM-dim
learned latent from the frozen Stage-A encoder/decoder
(encoder_decoder.pt).

Cells:
  1. latent        -- z_g only (learned latent, no descriptors)
  2. latent+desc94 -- z_g || 94-dim descriptors (HEADLINE cell)
  3. desc94         -- descriptors only, image accuracy = persistence by
                       construction (no visual channel to reconstruct at all --
                       same argument as stage 2's desc-only cells)
  4. persistence    -- mandatory baseline (identical cache/mask to stage 2,
                       numbers should reproduce exactly)
  5. stage 2's visual operator number (0.1697 holdout switched) -- QUOTED,
                       not refit
  6. unswitched (global) operator, computed for both latent and latent+desc94

Ridge: same fit_operator(..., ridge=lam, toward_identity=True) as stage 2,
lam swept over {0.1, 1.0, 10.0}. toward_identity keeps the same "nothing
changed" degrade-to prior; whether this is the right prior for a LEARNED
latent (vs. stage 2's occupancy-valued state) is an open question -- flagged
in RESULTS.md rather than silently assumed correct.
"""
from __future__ import annotations

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc")

from fit_linear_foresight import (  # noqa: E402
    actions_to_pixels, fit_operator, metrics, swept_region_mask,
)
from transforms.functional import (  # noqa: E402
    blend_push_prediction, from_push_frame, push_frame_validity_mask,
)
from descriptors import BOUNDS, GRID  # noqa: E402
from train_encoder import Encoder, Decoder  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BINS = 6
MIN_ROWS_PER_BIN = 50
LAMS = [0.1, 1.0, 10.0]
CACHE_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc/cache"
OUT_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/latent-switched"
RES = 32  # decoder output resolution (matches stage-2 canonical grid)

WS_MIN = torch.tensor([BOUNDS["x_min"], BOUNDS["y_min"]])
WS_MAX = torch.tensor([BOUNDS["x_max"], BOUNDS["y_max"]])

CELLS = {
    "latent":        dict(use_latent=True,  desc_dim=0),
    "latent+desc94": dict(use_latent=True,  desc_dim=94),
    "desc94":        dict(use_latent=False, desc_dim=94),
}
UNSWITCHED_CELLS = {"latent", "latent+desc94"}


def bin_index(lengths_m, bin_edges):
    return torch.bucketize(lengths_m, bin_edges[1:-1].to(lengths_m.device))


def load_encoder_decoder():
    ckpt = torch.load(f"{OUT_DIR}/encoder_decoder.pt", map_location=DEVICE)
    latent_dim = ckpt["latent_dim"]
    E = Encoder(latent_dim).to(DEVICE)
    D = Decoder(latent_dim).to(DEVICE)
    E.load_state_dict(ckpt["encoder"])
    D.load_state_dict(ckpt["decoder"])
    E.eval(); D.eval()
    for p in E.parameters():
        p.requires_grad_(False)
    for p in D.parameters():
        p.requires_grad_(False)
    return E, D, latent_dim


@torch.no_grad()
def encode_all(E, canon, batch=4096):
    out = []
    for i in range(0, canon.shape[0], batch):
        out.append(E(canon[i:i + batch].to(DEVICE)).cpu())
    return torch.cat(out, dim=0)


def load_cache(split, E):
    d = torch.load(f"{CACHE_DIR}/{split}_cache.pt", map_location="cpu")
    s_px, e_px = actions_to_pixels(d["action"], WS_MIN, WS_MAX, (GRID, GRID))
    d["start_px"], d["end_px"] = s_px, e_px
    d["z0"] = encode_all(E, d["canon0"])
    d["z1"] = encode_all(E, d["canon1"])
    return d


def build_state(cache, cell, which):
    z = cache[f"z{which}"]
    desc = cache[f"desc{which}"]
    dd = cell["desc_dim"]
    if cell["use_latent"] and dd > 0:
        return torch.cat([z, desc[:, :dd]], dim=1)
    if cell["use_latent"]:
        return z
    return desc[:, :dd]


def fit_one_bin(X0, X1, lam, D):
    if X0.shape[0] < MIN_ROWS_PER_BIN:
        return torch.eye(D)
    Y0, Y1 = X0.to(DEVICE).T, X1.to(DEVICE).T
    A = fit_operator(Y0, Y1, ridge=lam, toward_identity=True)
    return A.cpu()


def fit_cell(cell, X0_tr, X1_tr, lengths_tr, bin_edges, lam):
    D = X0_tr.shape[1]
    bins = bin_index(lengths_tr, bin_edges)
    ops = [fit_one_bin(X0_tr[bins == b], X1_tr[bins == b], lam, D) for b in range(N_BINS)]
    A_single = fit_one_bin(X0_tr, X1_tr, lam, D)
    return ops, A_single


def predict_image(cell, A_or_ops, switched, cache, bin_edges, decoder, latent_dim, batch=512):
    occ0 = cache["occ0"].float()
    if not cell["use_latent"]:
        return occ0.clone()

    z0 = cache["z0"]
    desc0 = cache["desc0"]
    s_px, e_px = cache["start_px"], cache["end_px"]
    lengths = cache["length_m"]
    N = occ0.shape[0]
    H, W = 64, 64
    dd = cell["desc_dim"]
    out = torch.empty(N, H, W)
    bins = bin_index(lengths, bin_edges) if switched else None
    for i in range(0, N, batch):
        sl = slice(i, i + batch)
        zc = z0[sl].to(DEVICE)
        if dd > 0:
            x0 = torch.cat([zc, desc0[sl, :dd].to(DEVICE)], dim=1)
        else:
            x0 = zc
        s, e = s_px[sl].to(DEVICE), e_px[sl].to(DEVICE)
        o0 = occ0[sl].to(DEVICE)
        if switched:
            pred_full = torch.empty(x0.shape[0], x0.shape[1], device=DEVICE)
            b_sl = bins[sl]
            for b in range(N_BINS):
                m = b_sl == b
                if not bool(m.any()):
                    continue
                A = A_or_ops[b].to(DEVICE)
                pred_full[m] = (A @ x0[m].T).T
        else:
            A = A_or_ops.to(DEVICE)
            pred_full = (A @ x0.T).T
        z1_hat = pred_full[:, :latent_dim]
        with torch.no_grad():
            canon1_hat = torch.sigmoid(decoder(z1_hat))
        back = from_push_frame(canon1_hat, s, e, (H, W), 1.0)
        mask = push_frame_validity_mask(s, e, (H, W), (RES, RES), 1.0)
        pred = blend_push_prediction(back, o0, mask).clamp_(0.0, 1.0)
        out[sl] = pred.cpu()
    return out


def eval_accuracy(pred_img, cache, region):
    return metrics(pred_img, cache["occ1"].float(), cache["occ0"].float(), region=region)


def per_bin_accuracy(pred_img, cache, region, bin_edges):
    bins = bin_index(cache["length_m"], bin_edges)
    out = {}
    for b in range(N_BINS):
        m = bins == b
        n = int(m.sum())
        if n == 0:
            out[b] = None
            continue
        out[b] = dict(n=n, **eval_accuracy(pred_img[m], {k: v[m] for k, v in cache.items()
                                                          if k in ("occ0", "occ1")}, region[m]))
    return out


def main():
    t0 = time.time()
    E, D, latent_dim = load_encoder_decoder()
    print(f"loaded encoder/decoder, latent_dim={latent_dim}")

    print("loading caches + encoding...")
    train = load_cache("train", E)
    test = load_cache("test", E)
    print(f"train N={train['canon0'].shape[0]}, test N={test['canon0'].shape[0]}, "
          f"({time.time()-t0:.1f}s)")

    hi = float(train["length_m"].max())
    bin_edges = torch.linspace(0.0, hi, N_BINS + 1)

    plate_px = 0.04 / 0.128 * 64
    region_tr = swept_region_mask(train["start_px"], train["end_px"], (64, 64),
                                   0.5 * plate_px + 2.0, 0.5 * plate_px)
    region_te = swept_region_mask(test["start_px"], test["end_px"], (64, 64),
                                   0.5 * plate_px + 2.0, 0.5 * plate_px)

    results = {"bin_edges_mm": (bin_edges * 1000).tolist(), "latent_dim": latent_dim,
               "stage2_visual_holdout_switched_QUOTED": 0.1697, "cells": {}}

    pers_tr = eval_accuracy(train["occ0"].float(), train, region_tr)
    pers_te = eval_accuracy(test["occ0"].float(), test, region_te)
    results["cells"]["persistence"] = {"train": pers_tr, "holdout": pers_te}
    print(f"persistence: train acc {pers_tr['accuracy']:.4f}  holdout acc {pers_te['accuracy']:.4f}")

    for cell_name, cell in CELLS.items():
        results["cells"][cell_name] = {}
        X0_tr = build_state(train, cell, "0")
        X1_tr = build_state(train, cell, "1")
        for lam in LAMS:
            t1 = time.time()
            ops, A_single = fit_cell(cell, X0_tr, X1_tr, train["length_m"], bin_edges, lam)
            entry = {}
            for split_name, cache, region in [("train", train, region_tr),
                                               ("holdout", test, region_te)]:
                pred_sw = predict_image(cell, ops, True, cache, bin_edges, D, latent_dim)
                acc_sw = eval_accuracy(pred_sw, cache, region)
                pb_sw = per_bin_accuracy(pred_sw, cache, region, bin_edges)
                entry[split_name] = {"switched": acc_sw, "per_bin_switched": pb_sw}

                if cell_name in UNSWITCHED_CELLS:
                    pred_gl = predict_image(cell, A_single, False, cache, bin_edges, D, latent_dim)
                    acc_gl = eval_accuracy(pred_gl, cache, region)
                    pb_gl = per_bin_accuracy(pred_gl, cache, region, bin_edges)
                    entry[split_name]["global"] = acc_gl
                    entry[split_name]["per_bin_global"] = pb_gl
            results["cells"][cell_name][str(lam)] = entry
            hs = entry["holdout"]["switched"]["accuracy"]
            ts = entry["train"]["switched"]["accuracy"]
            print(f"  [{cell_name} lam={lam}] train acc(sw)={ts:.4f} holdout acc(sw)={hs:.4f} "
                  f"({time.time()-t1:.1f}s)")

    out_path = f"{OUT_DIR}/results_stage3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
