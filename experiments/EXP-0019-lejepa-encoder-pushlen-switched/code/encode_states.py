"""EXP-0019 Stage 2a: encode the transition corpus with a TRAINED encoder.

Same output contract as EXP-0016's `encode_corpus.py` (z0, z1, a, length_m),
but loads an encoder checkpoint instead of freezing a random init, and uses
the verified-equivalent fast rasteriser (`raster.py`, exact match to
`particles_to_occupancy`, 77x faster at B=64).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments/EXP-0016-lejepa-random-encoder-floor/code"))
from model import ResCNNEncoder, encode_action              # noqa: E402
from raster import rasterise, BASE_RADIUS                   # noqa: E402
from data import randlen_files, load_transitions            # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def enc_all(enc, pts, n_real, bs=1024):
    zs = []
    for lo in range(0, pts.shape[0], bs):
        p = pts[lo:lo+bs].to(DEV); nr = n_real[lo:lo+bs].to(DEV)
        valid = torch.arange(p.shape[1], device=DEV)[None, :] < nr[:, None]
        occ = rasterise(p, BASE_RADIUS, valid=valid)
        assert occ.device.type == DEV, occ.device
        zs.append(enc(occ[:, None]).float().cpu())
    return torch.cat(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=1024)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    enc = ResCNNEncoder(**ck["config"]).to(DEV).eval()
    enc.load_state_dict(ck["state_dict"])
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f"encoder cfg {ck['config']} from {args.ckpt} (epoch {ck.get('epoch')}), "
          f"device {next(enc.parameters()).device}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        t0 = time.time()
        d = load_transitions(randlen_files(split))
        z0 = enc_all(enc, d["pts0"], d["n_real"], args.bs)
        z1 = enc_all(enc, d["pts1"], d["n_real"], args.bs)
        a = encode_action(d["p_start"], d["p_stop"], d["angle"])
        L = (d["p_stop"] - d["p_start"]).norm(dim=-1)
        torch.save(dict(z0=z0, z1=z1, a=a, length_m=L, ckpt=args.ckpt,
                        encoder_config=ck["config"]), out / f"latents_{split}.pt")
        print(f"{split}: {z0.shape[0]} rows, push length "
              f"{L.min()*1000:.2f}-{L.max()*1000:.2f} mm, {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
