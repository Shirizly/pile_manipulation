"""EXP-0016 / RUN-0002: fit residual switched-linear latent dynamics on the
FROZEN random encoder's latents, for several K, several seeds.

The kill-gate cell: if K=8 beats K=1 on a randomly-initialised encoder, then
"switched beats single" is not evidence about LeJEPA.

Normalised latent error is reported as R^2 against the `Delta z = 0`
(persistence-in-latent) baseline *in the same latent space*:

    R2 = 1 - mse(dz_pred - dz_true) / mse(dz_true)

which is exactly 0 for the Delta z = 0 baseline by construction.  This number
is comparable ONLY WITHIN one encoder -- any later cell that fine-tunes the
encoder moves the target and the denominator with it.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import SwitchedLinearDynamics, ACTION_DIM        # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def effective_rank(z, n=20000):
    """exp(entropy of the normalised eigenvalue spectrum of cov(z)) -- the
    standard participation-ratio-style effective rank (Roy & Vetterli 2007)."""
    x = z[:n].double()
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = (s ** 2) / (s ** 2).sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum())), s.float().tolist()


def r2(pred, true):
    return 1.0 - float(((pred - true) ** 2).mean() / (true ** 2).mean())


def r2_perdim(pred, true):
    num = ((pred - true) ** 2).mean(0)
    den = (true ** 2).mean(0)
    return float((1.0 - num / den).mean())


def fit(K, seed, ztr, atr, dztr, zte, ate, dzte, epochs, bs, lr, mu, sd, wd):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dyn = SwitchedLinearDynamics(latent_dim=ztr.shape[1], K=K, act_dim=ACTION_DIM).to(DEV)
    opt = torch.optim.AdamW(dyn.parameters(), lr=lr, weight_decay=wd)
    n = ztr.shape[0]
    steps = epochs * ((n + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for lo in range(0, n, bs):
            idx = perm[lo:lo + bs]
            zi = (ztr[idx] - mu) / sd
            _, dz = dyn(zi, atr[idx])
            loss = ((dz - dztr[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            tot += float(loss) * idx.numel()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    K={K} s={seed} ep{ep:3d} train_mse {tot/n:.6g}", flush=True)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else float("nan")
    dyn.eval()
    with torch.no_grad():
        pr_te = torch.cat([dyn((zte[i:i+4096] - mu) / sd, ate[i:i+4096])[1]
                           for i in range(0, zte.shape[0], 4096)])
        pr_tr = torch.cat([dyn((ztr[i:i+4096] - mu) / sd, atr[i:i+4096])[1]
                           for i in range(0, ztr.shape[0], 4096)])
        g = torch.cat([dyn((zte[i:i+4096] - mu) / sd, ate[i:i+4096], return_gate=True)[2]
                       for i in range(0, zte.shape[0], 4096)])
    gm = g.mean(0)
    gate_entropy = float(-(gm * gm.clamp_min(1e-12).log()).sum())
    return dyn, dict(
        K=K, seed=seed, wall_s=wall, peak_mib=peak,
        r2_test=r2(pr_te, dzte), r2_train=r2(pr_tr, dztr),
        r2_test_perdim=r2_perdim(pr_te, dzte), r2_train_perdim=r2_perdim(pr_tr, dztr),
        gate_mean=gm.tolist(), gate_entropy_nats=gate_entropy,
        gate_max_usage=float(gm.max()),
        gate_frac_confident=float((g.max(1).values > 0.9).float().mean()),
        n_params=sum(p.numel() for p in dyn.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--ks", default="1,4,8")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = Path(args.latents)
    tr = torch.load(L / "latents_train.pt"); te = torch.load(L / "latents_test.pt")
    ztr, atr = tr["z0"].to(DEV), tr["a"].to(DEV)
    zte, ate = te["z0"].to(DEV), te["a"].to(DEV)
    dztr = (tr["z1"] - tr["z0"]).to(DEV); dzte = (te["z1"] - te["z0"]).to(DEV)
    assert ztr.device.type == DEV.split(":")[0]
    mu, sd = ztr.mean(0, keepdim=True), ztr.std()

    er, sv = effective_rank(tr["z0"])
    stats = dict(
        n_train=int(ztr.shape[0]), n_test=int(zte.shape[0]),
        z_perdim_std_mean=float(tr["z0"].std(0).mean()),
        z_perdim_std_min=float(tr["z0"].std(0).min()),
        z_perdim_std_max=float(tr["z0"].std(0).max()),
        z_global_mean_norm=float(tr["z0"].mean(0).norm()),
        z_centred_rms=float((tr["z0"] - tr["z0"].mean(0)).pow(2).mean().sqrt()),
        dz_perdim_std_mean=float(dztr.std(0).mean().cpu()),
        dz_rms_over_z_centred_rms=float(
            dztr.pow(2).mean().sqrt().cpu() / (tr["z0"] - tr["z0"].mean(0)).pow(2).mean().sqrt()),
        effective_rank=er, latent_dim=int(ztr.shape[1]),
        singular_values_top20=sv[:20], singular_values_bottom5=sv[-5:],
    )
    print("latent stats:", json.dumps({k: v for k, v in stats.items()
                                       if not k.startswith("singular")}, indent=2), flush=True)

    # Mandatory "do nothing" baseline: Delta z = 0, in the same latent space.
    base = dict(cell="dz=0", r2_test=r2(torch.zeros_like(dzte), dzte),
                r2_test_perdim=r2_perdim(torch.zeros_like(dzte), dzte))
    print("dz=0 baseline:", base, flush=True)

    rows, out = [base], Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for K in [int(x) for x in args.ks.split(",")]:
        for s in [int(x) for x in args.seeds.split(",")]:
            dyn, m = fit(K, s, ztr, atr, dztr, zte, ate, dzte,
                         args.epochs, args.bs, args.lr, mu, sd, args.wd)
            m["cell"] = f"K={K}"
            print(f"  -> K={K} seed={s} test R2 {m['r2_test']:+.4f} "
                  f"(perdim {m['r2_test_perdim']:+.4f}) train {m['r2_train']:+.4f} "
                  f"{m['wall_s']:.1f}s peak {m['peak_mib']:.0f} MiB", flush=True)
            rows.append(m)
            torch.save({"state_dict": dyn.state_dict(), "config": dyn.cfg,
                        "metrics": m, "mu": mu.cpu(), "sd": float(sd),
                        "encoder": "encoder_seed0.pt (random, frozen)",
                        "fit_args": vars(args)}, out / f"dyn_K{K}_seed{s}.pt")
    json.dump({"latent_stats": stats, "cells": rows, "args": vars(args)},
              open(out / "dynamics_metrics.json", "w"), indent=2)
    print("\n== summary (test R2 vs dz=0, mean +/- sd over seeds) ==", flush=True)
    for K in [int(x) for x in args.ks.split(",")]:
        v = torch.tensor([r["r2_test"] for r in rows if r.get("K") == K])
        print(f"  K={K}: {v.mean():+.4f} +/- {v.std(unbiased=True):.4f}  n={len(v)}", flush=True)


if __name__ == "__main__":
    main()
