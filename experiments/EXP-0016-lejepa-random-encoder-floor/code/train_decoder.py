"""EXP-0016 / RUN-0003: post-hoc occupancy decoder on the FROZEN random
encoder.  z is loaded from disk (already detached), and the encoder object is
never constructed here at all -- so no gradient can reach it by construction.

Reported:
  recon_accuracy  1 - rms(D(z0) - occ0) / rms(0 - occ0)     (autoencoding)
  accuracy        1 - rms(D(z0+dz_pred) - occ1) / rms(occ0 - occ1)
                  per `experiments/METRICS.md`, persistence (occ0) as baseline,
                  reported both full-frame and over `swept_region_mask`.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_linear_foresight import swept_region_mask, actions_to_pixels   # noqa: E402
from model import OccDecoder, SwitchedLinearDynamics, ACTION_DIM        # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WS_MIN = torch.tensor([-0.064, -0.064]); WS_MAX = torch.tensor([0.064, 0.064])


def acc(pred, true, base, mask=None):
    if mask is None:
        num = (pred - true).pow(2).mean().sqrt(); den = (base - true).pow(2).mean().sqrt()
    else:
        m = mask.float(); w = m.sum().clamp_min(1)
        num = (((pred - true) ** 2) * m).sum().div(w).sqrt()
        den = (((base - true) ** 2) * m).sum().div(w).sqrt()
    return float(1 - num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--dyn-dir", required=True)
    ap.add_argument("--ks", default="1,8")
    ap.add_argument("--dyn-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = Path(args.latents)
    tr = torch.load(L / "latents_train.pt"); te = torch.load(L / "latents_test.pt")
    otr = torch.load(L / "occ_sample_train.pt"); ote = torch.load(L / "occ_sample_test.pt")
    ntr, nte = otr["occ0"].shape[0], ote["occ0"].shape[0]
    print(f"decoder train on {ntr} frozen-z/occ pairs, test {nte}", flush=True)

    ztr = tr["z0"][:ntr].to(DEV); occ_tr = otr["occ0"].float().to(DEV)
    zte = te["z0"][:nte].to(DEV); occ0_te = ote["occ0"].float().to(DEV)
    occ1_te = ote["occ1"].float().to(DEV)
    assert ztr.device.type == DEV.split(":")[0] and occ_tr.device.type == DEV.split(":")[0]
    assert not ztr.requires_grad, "z must be detached -- no path into the encoder"
    mu, sd = tr["z0"].mean(0, keepdim=True).to(DEV), float(tr["z0"].std())

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dec = OccDecoder(latent_dim=ztr.shape[1], out_resolution=occ_tr.shape[-1]).to(DEV)
    assert all(p.requires_grad for p in dec.parameters())
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((ntr + args.bs - 1) // args.bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(ntr, device=DEV); tot = 0.0
        for lo in range(0, ntr, args.bs):
            i = perm[lo:lo + args.bs]
            loss = (dec((ztr[i] - mu) / sd) - occ_tr[i]).pow(2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            tot += float(loss.detach()) * i.numel()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  dec ep{ep:3d} mse {tot/ntr:.6g}", flush=True)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else float("nan")
    dec.eval()
    torch.save({"state_dict": dec.state_dict(), "latent_dim": int(ztr.shape[1]),
                "out_resolution": int(occ_tr.shape[-1]), "mu": mu.cpu(), "sd": sd,
                "args": vars(args), "encoder": "encoder_seed0.pt (random, frozen)"},
               Path(args.out) / "decoder.pt")

    with torch.no_grad():
        rec = torch.cat([dec((zte[i:i+512] - mu) / sd) for i in range(0, nte, 512)])
    out = {"decoder_wall_s": wall, "decoder_peak_mib": peak, "n_train": ntr,
           "n_test": nte, "recon_mse": float((rec - occ0_te).pow(2).mean()),
           "recon_accuracy_vs_zero": acc(rec, occ0_te, torch.zeros_like(occ0_te)),
           "cells": []}
    print(f"autoencoding: recon accuracy vs empty-frame {out['recon_accuracy_vs_zero']:+.4f}",
          flush=True)

    # swept-region mask needs the raw push endpoints; reconstruct them from the
    # encoded action (encode_action scales by 0.064 and appends the displacement)
    a_te = te["a"][:nte]
    s_xy = a_te[:, :2] * 0.064
    e_xy = s_xy + a_te[:, 4:6] * 0.064
    s_px, e_px = actions_to_pixels(torch.cat([s_xy, e_xy], 1), WS_MIN, WS_MAX,
                                   (occ0_te.shape[-1],) * 2)
    mask = swept_region_mask(s_px.to(DEV), e_px.to(DEV), (occ0_te.shape[-1],) * 2,
                             half_width_px=6.0, pad_px=3.0)
    print(f"swept mask covers {float(mask.float().mean()):.3f} of the frame", flush=True)

    for K in [int(x) for x in args.ks.split(",")]:
        ck = torch.load(Path(args.dyn_dir) / f"dyn_K{K}_seed{args.dyn_seed}.pt")
        dyn = SwitchedLinearDynamics(**ck["config"]).to(DEV); dyn.load_state_dict(ck["state_dict"]); dyn.eval()
        with torch.no_grad():
            zp = torch.cat([dyn((zte[i:i+512] - mu) / sd, te["a"][i:i+512].to(DEV))[1]
                            for i in range(0, nte, 512)]) + zte
            pred = torch.cat([dec((zp[i:i+512] - mu) / sd) for i in range(0, nte, 512)])
        row = {"cell": f"K={K}", "K": K, "dyn_seed": args.dyn_seed,
               "accuracy_full": acc(pred, occ1_te, occ0_te),
               "accuracy_swept": acc(pred, occ1_te, occ0_te, mask),
               "oracle_decode_accuracy_full": None}
        # ceiling: decode the TRUE next latent -- separates dynamics error from
        # decoder error.
        with torch.no_grad():
            zt1 = te["z1"][:nte].to(DEV)
            po = torch.cat([dec((zt1[i:i+512] - mu) / sd) for i in range(0, nte, 512)])
        row["oracle_decode_accuracy_full"] = acc(po, occ1_te, occ0_te)
        row["oracle_decode_accuracy_swept"] = acc(po, occ1_te, occ0_te, mask)
        print(f"  K={K}: accuracy full {row['accuracy_full']:+.4f} swept "
              f"{row['accuracy_swept']:+.4f} | decode-true-z ceiling full "
              f"{row['oracle_decode_accuracy_full']:+.4f} swept "
              f"{row['oracle_decode_accuracy_swept']:+.4f}", flush=True)
        out["cells"].append(row)

    # persistence through the decoder: D(z0) as the prediction of occ1
    out["persistence_through_decoder_full"] = acc(rec, occ1_te, occ0_te)
    json.dump(out, open(Path(args.out) / "decoder_metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
