"""GATE: does an encoder trained through TRANSITION GRADIENTS collapse?

This is Stage 3 of `docs/experimental_design/jepa_based_encoder.md`, which has
never been run in this repo. The one prior attempt to push transition gradients
into an encoder here (EXP-0007, FiLM architecture, `hybrid_linear_latent.md`
section 9) COLLAPSED: z_std fell to ~2e-6 with both model and baseline losses
converging to ~1e-9 together. That is invariant `hybrid-latent-stage2-anticollapse`,
status `broken`. A per-channel variance hinge fixed it there.

The question this run answers, and the ONLY thing it is for:
  does the same objective collapse when SIGReg(z) is already present?

Design, deliberately the design doc's Stage 3 as literally written so the
failure mode stays interpretable:

    z0      = E(occ_t)
    z1_tgt  = stopgrad(E(occ_{t+1}))            <- target detached
    z1_hat  = z0 + W_b(len) @ [z0, a, 1]        <- 6 hard push-length bins,
                                                   LEARNED jointly (not closed form)
    L       = ||z1_hat - z1_tgt||^2 / ||z1_tgt - z0||^2      (scale-free)
              + lambda_sigreg * SIGReg(z)

The normaliser makes the loss a latent-R^2-like quantity rather than something
a shrinking latent can win by scale alone -- collapse must then show up in the
MONITORS, not be hidden by the loss going to zero.

Stop-gradient on the target is what makes this Stage 3 rather than a joint
autoencoder: gradient flows to the encoder only through z0 and through the
operator's input, exactly as the doc specifies.

Reported every epoch (the collapse signature, same monitors as EXP-0019/0020):
per-dim std of z, min per-dim std, effective rank, ||mean z||, SIGReg(z), and
held-out latent R^2 against the `dz = 0` baseline -- so "did it collapse" and
"did it learn anything" are answered separately.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "le-wm"))
sys.path.insert(0, str(REPO / "experiments/EXP-0016-lejepa-random-encoder-floor/code"))
sys.path.insert(0, str(REPO / "experiments/EXP-0019-lejepa-encoder-pushlen-switched/code"))

from module import SIGReg                                     # noqa: E402
from model import ResCNNEncoder                               # noqa: E402
from raster import rasterise, BOUNDS, CUBE_SIZE               # noqa: E402
from data import randlen_files, load_transitions              # noqa: E402
from Baselines.LinearForesight.model import bin_index         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BINS = 6
EDGES = torch.linspace(0.0, 0.080, N_BINS + 1)


def radius_for(grid):
    return 0.5 * CUBE_SIZE / ((BOUNDS["x_max"] - BOUNDS["x_min"]) / grid)


def action_feats(ps, pe):
    d = pe - ps
    th = torch.atan2(d[:, 1], d[:, 0])
    return torch.stack([ps[:, 0], ps[:, 1], torch.sin(th), torch.cos(th),
                        d[:, 0], d[:, 1]], 1)


@torch.no_grad()
def monitors(enc, pts, n_real, grid, rad, n=8192, bs=512):
    enc.eval(); zs = []
    for lo in range(0, min(n, len(pts)), bs):
        p = pts[lo:lo + bs].to(DEV); nr = n_real[lo:lo + bs].to(DEV)
        valid = torch.arange(p.shape[1], device=DEV)[None, :] < nr[:, None]
        zs.append(enc(rasterise(p, rad, grid, BOUNDS, valid=valid)[:, None]).float())
    z = torch.cat(zs); zc = z - z.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(zc.double())
    p_ = (sv ** 2) / (sv ** 2).sum().clamp_min(1e-30)
    er = float(torch.exp(-(p_ * (p_ + 1e-30).log()).sum()))
    enc.train()
    return dict(perdim_std_mean=float(z.std(0).mean()), perdim_std_min=float(z.std(0).min()),
                effective_rank=er, mean_norm=float(z.mean(0).norm()))


@torch.no_grad()
def heldout_r2(enc, W, tr, grid, rad, bs=512, n=6144):
    """latent R^2 vs the dz=0 baseline, in THIS encoder's own latent space."""
    enc.eval(); num = den = 0.0
    for lo in range(0, min(n, len(tr["pts0"])), bs):
        sl = slice(lo, lo + bs)
        p0 = tr["pts0"][sl].to(DEV); p1 = tr["pts1"][sl].to(DEV)
        nr = tr["n_real"][sl].to(DEV)
        v = torch.arange(p0.shape[1], device=DEV)[None, :] < nr[:, None]
        z0 = enc(rasterise(p0, rad, grid, BOUNDS, valid=v)[:, None])
        z1 = enc(rasterise(p1, rad, grid, BOUNDS, valid=v)[:, None])
        a = action_feats(tr["p_start"][sl].to(DEV), tr["p_stop"][sl].to(DEV))
        b = bin_index(tr["len_m"][sl], EDGES).clamp(0, N_BINS - 1).to(DEV)
        f = torch.cat([z0, a, torch.ones(len(z0), 1, device=DEV)], 1)
        dz_hat = torch.einsum("bi,bio->bo", f, W[b])
        num += float(((z0 + dz_hat - z1) ** 2).sum()); den += float(((z1 - z0) ** 2).sum())
    enc.train()
    return 1.0 - num / max(den, 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--n-res-blocks", type=int, default=2)
    ap.add_argument("--lambda-sigreg", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps-per-epoch", type=int, default=250)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-files", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detach-denom", action="store_true",
                    help="detach the scale normaliser. WITHOUT this the encoder "
                         "controls both numerator and denominator and can lower "
                         "the loss by INFLATING dz_true along one direction -- "
                         "the suspected cause of the rank->1 collapse.")
    ap.add_argument("--standardise-z", action="store_true",
                    help="per-dimension batch-standardise z (gradients flow through, "
                         "BatchNorm-style) BEFORE the transition loss, using ONE set of "
                         "stats shared by z0 and the target. Makes the loss invariant to "
                         "the encoder's scale, so neither shrinking (detached-denominator "
                         "failure) nor inflating (attached-denominator failure) changes it. "
                         "NOTE: this removes SCALE as a free parameter but not CORRELATION "
                         "-- 256 unit-variance dims can still be perfectly correlated, i.e. "
                         "rank 1. If rank still collapses, a decorrelation/whitening term "
                         "is needed, not just standardisation.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    grid, rad = args.res, radius_for(args.res)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tr = load_transitions(randlen_files("train")[:args.max_train_files])
    te = load_transitions(randlen_files("test")[:12])
    for d in (tr, te):
        d["len_m"] = (d["p_stop"] - d["p_start"]).norm(dim=1)
    M = len(tr["pts0"])
    print(f"[data] {M} train transitions, {len(te['pts0'])} test; "
          f"grid={grid} radius={rad:.4f} vox", flush=True)

    enc = ResCNNEncoder(args.res, args.latent_dim, args.n_res_blocks).to(DEV)
    D = args.latent_dim
    W = nn.Parameter(torch.zeros(N_BINS, D + 7, D, device=DEV))   # init at dz=0
    sig = SIGReg().to(DEV)
    opt = torch.optim.AdamW([{"params": enc.parameters()}, {"params": [W]}],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * args.steps_per_epoch)

    m0 = monitors(enc, tr["pts0"], tr["n_real"], grid, rad)
    print(f"[epoch -1 at init] {json.dumps(m0)}", flush=True)
    hist = [dict(epoch=-1, **m0)]
    t0 = time.time()
    for ep in range(args.epochs):
        acc = dict(trans=0.0, sigreg=0.0)
        for _ in range(args.steps_per_epoch):
            idx = torch.randint(0, M, (args.bs,))
            p0 = tr["pts0"][idx].to(DEV); p1 = tr["pts1"][idx].to(DEV)
            nr = tr["n_real"][idx].to(DEV)
            v = torch.arange(p0.shape[1], device=DEV)[None, :] < nr[:, None]
            o0 = rasterise(p0, rad, grid, BOUNDS, valid=v)[:, None]
            o1 = rasterise(p1, rad, grid, BOUNDS, valid=v)[:, None]
            z0 = enc(o0)
            with torch.no_grad():
                z1_tgt = enc(o1)                       # STOP-GRADIENT target
            if args.standardise_z:
                # ONE set of stats over both endpoints, so the transformation is
                # identical for z0 and the target and dz is not distorted.
                both = torch.cat([z0, z1_tgt], 0)
                mu_b = both.mean(0, keepdim=True)
                sd_b = both.std(0, keepdim=True).clamp_min(1e-5)
                z0 = (z0 - mu_b) / sd_b
                z1_tgt = (z1_tgt - mu_b) / sd_b
            a = action_feats(tr["p_start"][idx].to(DEV), tr["p_stop"][idx].to(DEV))
            b = bin_index(tr["len_m"][idx], EDGES).clamp(0, N_BINS - 1).to(DEV)
            f = torch.cat([z0, a, torch.ones(len(z0), 1, device=DEV)], 1)
            dz_hat = torch.einsum("bi,bio->bo", f, W[b])
            dz_true = z1_tgt - z0
            # scale-free: a shrinking latent cannot win by scale alone
            den = (dz_true ** 2).sum()
            den = den.detach() if args.detach_denom else den
            l_tr = ((z0 + dz_hat - z1_tgt) ** 2).sum() / den.clamp_min(1e-8)
            sg = sig(z0.unsqueeze(0))
            loss = l_tr + args.lambda_sigreg * sg
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            acc["trans"] += float(l_tr); acc["sigreg"] += float(sg)
        m = monitors(enc, tr["pts0"], tr["n_real"], grid, rad)
        r2 = heldout_r2(enc, W.detach(), te, grid, rad)
        row = dict(epoch=ep, trans=acc["trans"] / args.steps_per_epoch,
                   sigreg=acc["sigreg"] / args.steps_per_epoch, heldout_latent_r2=r2, **m)
        hist.append(row)
        print(f"ep {ep:2d} trans {row['trans']:.4f} sigreg {row['sigreg']:.2f} | "
              f"std {m['perdim_std_mean']:.4f} (min {m['perdim_std_min']:.4f}) | "
              f"eff_rank {m['effective_rank']:.1f} | heldout latent R2 {r2:+.4f} | "
              f"{time.time()-t0:.0f}s", flush=True)

    torch.save({"state_dict": enc.state_dict(), "config": dict(
        input_resolution=args.res, latent_dim=args.latent_dim,
        n_res_blocks=args.n_res_blocks), "W": W.detach().cpu(),
        "bin_edges": EDGES, "train_args": vars(args), "history": hist},
        out / "encoder_transition.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    init_rank, final_rank = hist[0]["effective_rank"], hist[-1]["effective_rank"]
    print(f"\nGATE: std {hist[0]['perdim_std_mean']:.4f} -> {hist[-1]['perdim_std_mean']:.4f} "
          f"(min {hist[-1]['perdim_std_min']:.2e}); eff_rank {init_rank:.1f} -> {final_rank:.1f}; "
          f"heldout latent R2 {hist[-1]['heldout_latent_r2']:+.4f}")
    shrunk = hist[-1]["perdim_std_min"] < 1e-3
    rank_lost = final_rank < 0.1 * init_rank
    print(f"VERDICT: variance-collapse={'YES' if shrunk else 'no'} "
          f"rank-collapse={'YES' if rank_lost else 'no'} "
          f"({init_rank:.1f} -> {final_rank:.1f} of {args.latent_dim})")


if __name__ == "__main__":
    main()
