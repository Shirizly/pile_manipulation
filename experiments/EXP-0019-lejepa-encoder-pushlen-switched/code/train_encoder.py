"""EXP-0019 Stage 1: actually TRAIN the ResCNN occupancy encoder with the
LeJEPA objective (design doc sections 6 and 13 Stage 1).

  L = ||p1 - p2||^2  +  lambda_sigreg * SIGReg([p1;p2])

`SIGReg` is the vendored dependency-free Epps-Pulley sketch from
`le-wm/module.py` (torch + einops only; `stable_pretraining` is NOT needed).
The architecture is EXP-0016's `model.py::ResCNNEncoder` verbatim -- only the
projector and the training loop are new.

Collapse is the known failure mode (INVARIANTS `hybrid-latent-stage2-anticollapse`:
an unregularised latent objective drove per-dim std to ~2e-6).  Per-dimension
std and effective rank of z (NOT of the projection p) are logged every epoch.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "le-wm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments/EXP-0016-lejepa-random-encoder-floor/code"))

from module import SIGReg                                   # noqa: E402  (le-wm)
from model import ResCNNEncoder                             # noqa: E402  (EXP-0016)
from raster import make_views, rasterise, BASE_RADIUS       # noqa: E402
from data import randlen_files, sean_files, load_states     # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Projector(nn.Module):
    """LeJEPA projector P: z -> p, used ONLY for the pretraining objective
    (design doc 6.1: the dynamics model operates on z, not p)."""
    def __init__(self, latent_dim=256, hidden=1024, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.BatchNorm1d(hidden),
                                 nn.SiLU(), nn.Linear(hidden, out_dim))
        self.cfg = dict(latent_dim=latent_dim, hidden=hidden, out_dim=out_dim)
    def forward(self, z):
        return self.net(z)


sigreg_probe = None   # set in main(); SIGReg applied to z for DIAGNOSIS only


@torch.no_grad()
def latent_stats(enc, pts, n_real, n=8192, bs=512):
    enc.eval()
    zs = []
    for lo in range(0, min(n, pts.shape[0]), bs):
        p = pts[lo:lo+bs].to(DEV); nr = n_real[lo:lo+bs].to(DEV)
        valid = torch.arange(p.shape[1], device=DEV)[None, :] < nr[:, None]
        occ = rasterise(p, BASE_RADIUS, valid=valid)
        assert occ.device.type == DEV, occ.device
        zs.append(enc(occ[:, None]).float())
    z = torch.cat(zs)
    zc = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(zc.double())
    pw = (s ** 2) / (s ** 2).sum(); pw = pw[pw > 0]
    er = float(torch.exp(-(pw * pw.log()).sum()))
    enc.train()
    with torch.no_grad():
        sg_z = float(sigreg_probe(z[:1024][None]))   # SIGReg statistic OF z itself
    return dict(sigreg_of_z=sg_z, perdim_std_mean=float(z.std(0).mean()), perdim_std_min=float(z.std(0).min()),
                perdim_std_max=float(z.std(0).max()), mean_norm=float(z.mean(0).norm()),
                centred_rms=float(zc.pow(2).mean().sqrt()), effective_rank=er,
                latent_dim=int(z.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lambda-sigreg", type=float, default=0.02)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--n-res-blocks", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--steps-per-epoch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sean-files", type=int, default=0, help="n Sean files to add")
    ap.add_argument("--sigreg-on-z", type=float, default=0.0,
                    help="ALSO apply SIGReg directly to z with this weight. The design doc "
                         "puts SIGReg on the PROJECTION p only; this flag tests whether that "
                         "placement is what lets z collapse in rank.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    files = randlen_files("train")
    if args.sean_files:
        sf = sean_files()
        step = max(1, len(sf) // args.sean_files)
        files = files + sf[::step][:args.sean_files]
    pts, n_real = load_states(files)
    pts = pts.to(DEV); n_real = n_real.to(DEV)
    assert pts.device.type == DEV, pts.device
    M = pts.shape[0]
    print(f"train states: {M} (both states and states_ of each transition) "
          f"from {len(files)} files; device {pts.device}", flush=True)

    enc = ResCNNEncoder(args.res, args.latent_dim, args.n_res_blocks).to(DEV)
    proj = Projector(args.latent_dim, 1024, args.proj_dim).to(DEV)
    sig = SIGReg().to(DEV)
    globals()['sigreg_probe'] = SIGReg().to(DEV)
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total = args.epochs * args.steps_per_epoch
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total)
    gen = torch.Generator(device=DEV); gen.manual_seed(args.seed)
    print(f"encoder {sum(p.numel() for p in enc.parameters())} params, "
          f"projector {sum(p.numel() for p in proj.parameters())}, device "
          f"{next(enc.parameters()).device}", flush=True)

    st0 = latent_stats(enc, pts, n_real)
    print("epoch -1 (at init):", json.dumps(st0), flush=True)
    curves = [dict(epoch=-1, **st0)]
    torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
    t0 = time.time()
    for ep in range(args.epochs):
        acc = dict(align=0.0, sigreg=0.0, loss=0.0)
        for it in range(args.steps_per_epoch):
            idx = torch.randint(0, M, (args.bs,), device=DEV, generator=gen)
            v1, v2 = make_views(pts[idx], n_real[idx], gen)
            assert v1.device.type == DEV, v1.device
            z = enc(torch.cat([v1, v2])[:, None])
            p = proj(z)
            p1, p2 = p[:args.bs], p[args.bs:]
            align = (p1 - p2).pow(2).mean()
            sg = sig(torch.stack([p1, p2], dim=0))      # (T=2, B, D)
            loss = align + args.lambda_sigreg * sg
            if args.sigreg_on_z > 0:
                sgz = sig(torch.stack([z[:args.bs], z[args.bs:]], dim=0))
                loss = loss + args.sigreg_on_z * sgz
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            acc["align"] += float(align.detach()); acc["sigreg"] += float(sg.detach()); acc["loss"] += float(loss.detach())
        n = args.steps_per_epoch
        st = latent_stats(enc, pts, n_real)
        row = dict(epoch=ep, align=acc["align"]/n, sigreg=acc["sigreg"]/n,
                   loss=acc["loss"]/n, wall_s=time.time()-t0, **st)
        curves.append(row)
        print(f"ep{ep:3d} align {row['align']:.5g} sigreg {row['sigreg']:.5g} "
              f"| z perdim_std mean {st['perdim_std_mean']:.4g} min {st['perdim_std_min']:.3g} "
              f"| eff_rank {st['effective_rank']:.1f} | ||mean z|| {st['mean_norm']:.3f} "
              f"centred_rms {st['centred_rms']:.4f} | sigreg(z) {st['sigreg_of_z']:.4g} | {row['wall_s']:.0f}s", flush=True)
        torch.save({"state_dict": enc.state_dict(), "config": enc.config(),
                    "projector_state_dict": proj.state_dict(), "projector_config": proj.cfg,
                    "train_args": vars(args), "epoch": ep, "curves": curves,
                    "views": "particle dropout U(0.80,1.0) + footprint-radius jitter "
                             "x U(0.85,1.15) + additive occ noise std 0.02, clamped; "
                             "NO rotations/flips/crops",
                    "note": "LeJEPA-trained (view-alignment + SIGReg), EXP-0019 Stage 1"},
                   out / "encoder_lejepa.pt")
        json.dump(curves, open(out / "train_curves.json", "w"), indent=2)
    peak = torch.cuda.max_memory_allocated()/2**20 if DEV == "cuda" else float("nan")
    print(f"done {time.time()-t0:.0f}s peak {peak:.0f} MiB", flush=True)
    json.dump({"curves": curves, "args": vars(args), "peak_mib": peak},
              open(out / "train_curves.json", "w"), indent=2)


if __name__ == "__main__":
    main()
