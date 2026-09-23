"""EXP-0016 / RUN-0001: freeze a randomly-initialised ResCNNEncoder and encode
the pre-split `overnight_randlen_{train,test}` transition corpus.

Fast path only: glob `*_data.pt` and `torch.load` directly (docs/CODEMAP.md
LOADER TRAP -- `load_randlen_cell()` takes >10 min per call).  `*_failed.pt`
and `.zip` are excluded by the glob pattern.

Outputs (artifacts/RUN-0001/):
  encoder_seed<SEED>.pt   the frozen random encoder state_dict + its config
  latents_train.pt        z0, z1, a, length_m, group_id
  latents_test.pt         same, held-out files
  occ_sample_{train,test}.pt  occupancy pairs for decoder training (subset)
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transforms.functional import particles_to_occupancy      # noqa: E402
from model import ResCNNEncoder, encode_action                # noqa: E402

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
RADIUS = 0.5 * CUBE_SIZE / ((BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GROUPS = ["mixed_n20", "piled_n20", "piled_n50", "scattered_n20", "scattered_n50"]


def occ_of(states):
    return particles_to_occupancy(states[..., :3].to(DEV), BOUNDS, (GRID, GRID),
                                  footprint_radius=RADIUS)


def run_split(root, enc, batch, occ_keep):
    files = sorted(glob.glob(f"{root}/*/*_data.pt"))
    assert files, root
    Z0, Z1, A, L, G, occ0_keep, occ1_keep = [], [], [], [], [], [], []
    t0 = time.time()
    n_rows = 0
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        st, st_ = d["states"].float(), d["states_"].float()
        ps, pe, ang = d["p_starts"][:, :2].float(), d["p_stops"][:, :2].float(), d["angles"].float()
        g = GROUPS.index(Path(f).parent.name)
        n = st.shape[0]
        for lo in range(0, n, batch):
            hi = min(lo + batch, n)
            o0, o1 = occ_of(st[lo:hi]), occ_of(st_[lo:hi])
            assert o0.device.type == DEV.split(":")[0], o0.device
            with torch.no_grad():
                z0 = enc(o0[:, None]); z1 = enc(o1[:, None])
            Z0.append(z0.cpu()); Z1.append(z1.cpu())
            if len(occ0_keep) * batch < occ_keep:
                occ0_keep.append(o0.half().cpu()); occ1_keep.append(o1.half().cpu())
        A.append(encode_action(ps, pe, ang))
        L.append((pe - ps).norm(dim=-1))
        G.append(torch.full((n,), g, dtype=torch.long))
        n_rows += n
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(files)}] {n_rows} rows {time.time()-t0:.1f}s", flush=True)
    dt = time.time() - t0
    print(f"{root}: {len(files)} files, {n_rows} rows, {dt:.1f}s", flush=True)
    out = dict(z0=torch.cat(Z0), z1=torch.cat(Z1), a=torch.cat(A),
               length_m=torch.cat(L), group=torch.cat(G),
               n_files=len(files), wall_s=dt)
    occ = dict(occ0=torch.cat(occ0_keep)[:occ_keep], occ1=torch.cat(occ1_keep)[:occ_keep])
    return out, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--n-res-blocks", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--occ-keep-train", type=int, default=24576)
    ap.add_argument("--occ-keep-test", type=int, default=6144)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    enc = ResCNNEncoder(args.res, args.latent_dim, args.n_res_blocks).to(DEV).eval()
    for p in enc.parameters():
        p.requires_grad_(False)          # FROZEN at random init, throughout
    n_par = sum(p.numel() for p in enc.parameters())
    print(f"encoder: {n_par} params, device {next(enc.parameters()).device}, "
          f"cfg {enc.config()}, seed {args.seed}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": enc.state_dict(), "config": enc.config(),
                "seed": args.seed, "n_params": n_par,
                "note": "randomly initialised, NEVER trained"},
               out / f"encoder_seed{args.seed}.pt")

    torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
    tr, occ_tr = run_split(str(REPO / "Genesis/data/overnight_randlen_train"),
                           enc, args.batch, args.occ_keep_train)
    te, occ_te = run_split(str(REPO / "Genesis/data/overnight_randlen_test"),
                           enc, args.batch, args.occ_keep_test)
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else float("nan")
    print(f"peak GPU MiB (rasterise+encode, batch={args.batch}): {peak:.0f}", flush=True)

    tr["peak_mib"] = peak; tr["batch"] = args.batch
    torch.save(tr, out / "latents_train.pt")
    torch.save(te, out / "latents_test.pt")
    torch.save(occ_tr, out / "occ_sample_train.pt")
    torch.save(occ_te, out / "occ_sample_test.pt")

    z = tr["z0"]
    print(f"train z0 {tuple(z.shape)}  per-dim std: mean {z.std(0).mean():.4f} "
          f"min {z.std(0).min():.4g} max {z.std(0).max():.4f}", flush=True)
    dz = tr["z1"] - tr["z0"]
    print(f"train dz per-dim std: mean {dz.std(0).mean():.4g}  "
          f"||dz||/||z|| = {dz.norm(dim=1).mean()/z.norm(dim=1).mean():.4f}", flush=True)
    json.dump({"encoder_params": n_par, "peak_mib": peak, "batch": args.batch,
               "train_wall_s": tr["wall_s"], "test_wall_s": te["wall_s"],
               "n_train": int(z.shape[0]), "n_test": int(te["z0"].shape[0])},
              open(out / "encode_cost.json", "w"), indent=2)


if __name__ == "__main__":
    main()
