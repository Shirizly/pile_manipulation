"""Latent ranking arms scored through a VALUE READOUT, with no decoder.

Per direction 2026-09-17: the LeJEPA arms are not to be scored by decoding a
latent back to an occupancy image. The earlier decoder-based numbers
(randenc +0.139, lejepa +0.070 mean slateN) put a reconstruction step in the
measurement path and are superseded by this arm; they are not the latent
arms' result.

What this does instead, per candidate:

    z0      = E(occ0)                       encoder, frozen
    z1_hat  = z0 + W_bin(length) @ [z0, a, 1]   latent dynamics, closed-form ridge
    dv      = h(z1_hat, z_g) - h(z0, z_g)   fitted ValueReadout, no reconstruction

`h` is a `ValueReadout` fitted per value function (EXP-0017's instrument, the
repo's action-ranking readout). `z_g` is the encoder applied to K legal
configurations of the cell's goal mask, averaged -- a goal is the shape, not
one particular arrangement.

SIGN: `dv` above is already the readout's own value scale. The builder's cache
is a COST throughout, so the returned quantity is multiplied by `cell.sign`
(+1 where the value function is cost-native, -1 where it is value-native).
Getting this wrong silently inverts every metric downstream.

NOTE ON COMPARABILITY: this arm and the `descriptor-readout` arm are now
scored through the SAME mechanism -- a trained readout on an embedding -- so
they are a clean comparison of EMBEDDINGS. `visual-switched` and `nfd` score
through a predicted image and are not commensurable with them in the same way.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
for p in ("", "experiments/EXP-0016-lejepa-random-encoder-floor/code",
          "experiments/EXP-0019-lejepa-encoder-pushlen-switched/code",
          "experiments/EXP-0017-value-readout-instrument/code"):
    sys.path.insert(0, str(REPO / p) if p else str(REPO))

from model import ResCNNEncoder                                    # noqa: E402
from raster import rasterise                                       # noqa: E402
from value_readout import ValueReadout                             # noqa: E402
from Baselines.common.goal_configs import mask_to_configuration    # noqa: E402
from Baselines.LinearForesight.model import bin_index              # noqa: E402

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
CUBE = 0.005
K_CFG = 3
LAT = REPO / "experiments/temp/exp0021-latent/artifacts"
RO_DIR = REPO / "experiments/temp/exp0021-latent-readout"

ARMS = ["randenc-s0", "randenc-s1", "randenc-s2",
        "lejepa-s0", "lejepa-s1", "lejepa-s2"]


def _radius(grid):
    """Footprint radius is PHYSICAL (half a cube) -- it rescales with pitch."""
    return 0.5 * CUBE / ((BOUNDS["x_max"] - BOUNDS["x_min"]) / grid)


def _encode(enc, occ, device, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(occ), bs):
            out.append(enc(occ[i:i + bs].to(device)[:, None]).cpu())
    return torch.cat(out)


def _action_feats(ctx):
    """EXP-0016 encode_action: [x_s, y_s, sin, cos, dx, dy] -- must match the
    6 action columns the operators were FITTED with (W is (256+6+1, 256))."""
    d = ctx.p_stop - ctx.p_start
    th = torch.atan2(d[:, 1], d[:, 0])
    return torch.stack([ctx.p_start[:, 0], ctx.p_start[:, 1],
                        torch.sin(th), torch.cos(th), d[:, 0], d[:, 1]], 1).float()


def _make_arm(arm, device, capacity="mlp"):
    ops = torch.load(LAT / f"operators/{arm}/operators_grouped.pt",
                     map_location="cpu", weights_only=False)
    ck = torch.load(ops["encoder_ckpt"], map_location="cpu", weights_only=False)
    enc = ResCNNEncoder(**ck["config"]).to(device).eval()
    enc.load_state_dict(ck["state_dict"])
    grid = int(ck["config"].get("input_resolution", 64))
    W_bins = [w.float() for w in ops["W_bins"]]
    W_glob = ops["W_global"].float()
    edges, mu, centred = ops["bin_edges"], ops["mu"].float(), bool(ops["centred"])

    def fn(ctx, cell):
        ro_p = RO_DIR / f"readout__{arm}__{capacity}__{cell.value_fn}.joblib"
        if not ro_p.exists():
            raise FileNotFoundError(f"no fitted readout for {arm}/{cell.value_fn}: {ro_p}")
        ro = ValueReadout.load(ro_p)

        key = f"z0::{arm}"
        if key not in ctx.cache:
            occ0 = ctx.occ0.cpu() if grid == ctx.occ0.shape[-1] else \
                rasterise(ctx.states[..., :2].to(device).float(), _radius(grid),
                          grid, BOUNDS).cpu()
            ctx.cache[key] = _encode(enc, occ0, device)
        z0 = ctx.cache[key]

        a = _action_feats(ctx)
        zc = (z0 - mu) if centred else z0
        feats = torch.cat([zc, a, torch.ones(len(zc), 1)], 1)
        b = bin_index(ctx.length_m, edges).clamp(0, len(W_bins) - 1)
        dz = torch.empty_like(z0)
        for k in range(len(W_bins)):
            m = (b == k)
            if m.any():
                dz[m] = feats[m] @ W_bins[k]
        z1 = z0 + dz

        # goal embedding: K legal configurations of this cell's mask, averaged
        gk = f"zg::{arm}::{cell.goal}"
        if gk not in ctx.cache:
            rng = np.random.default_rng(0)
            mask = cell.mask.detach().cpu().numpy().astype(bool)
            poses = np.stack([mask_to_configuration(mask, 20, seed=rng).poses
                              for _ in range(K_CFG)])
            gocc = rasterise(torch.as_tensor(poses[..., :2], dtype=torch.float32,
                                             device=device), _radius(grid), grid, BOUNDS)
            ctx.cache[gk] = _encode(enc, gocc.cpu(), device).mean(0, keepdim=True)
        z_g = ctx.cache[gk]

        dv = ro.dv(z0.numpy(), z1.numpy(), z_g.numpy())   # single goal row; dv broadcasts
        return cell.sign * torch.as_tensor(np.asarray(dv, dtype=np.float32))

    return fn


def register(builder, device):
    for arm in ARMS:
        if (LAT / f"operators/{arm}/operators_grouped.pt").exists():
            builder.register_score_arm(f"{arm}-vh", _make_arm(arm, device))
