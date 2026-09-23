"""EXP-0016 architecture: residual-CNN encoder (design doc §3.1), action
encoder (§5), residual switched-linear latent dynamics (§8-10), and a post-hoc
occupancy decoder that must NOT backprop into the encoder.

Experiment-local by design (`experiment-log`: a new composition of existing
capability stays local).  Nothing here is imported by any project module.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- encoder ---
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.silu(x + h)


class ResCNNEncoder(nn.Module):
    """64x64 (or 32x32) single-channel occupancy -> z in R^latent_dim.

    Stages 32/64/128/256 channels at 64/32/16/8 px, residual blocks at each,
    global average pool, linear projection.  At input_resolution=32 the first
    stage is dropped (design doc: "one fewer spatial stage").
    """

    def __init__(self, input_resolution=64, latent_dim=256, n_res_blocks=2,
                 in_channels=1, channels=(32, 64, 128, 256)):
        super().__init__()
        if input_resolution == 32:
            channels = channels[1:]
        self.input_resolution = int(input_resolution)
        self.latent_dim = int(latent_dim)
        self.n_res_blocks = int(n_res_blocks)
        self.channels = tuple(channels)

        layers = [nn.Conv2d(in_channels, channels[0], 3, padding=1),
                  nn.GroupNorm(8, channels[0]), nn.SiLU()]
        for _ in range(n_res_blocks):
            layers.append(ResBlock(channels[0]))
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(cin, cout, 3, stride=2, padding=1))   # downsample
            layers.append(nn.GroupNorm(8, cout))
            layers.append(nn.SiLU())
            for _ in range(n_res_blocks):
                layers.append(ResBlock(cout))
        self.body = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], latent_dim)

    def forward(self, x):                       # x: (B,1,H,W)
        h = self.body(x)
        h = h.mean(dim=(2, 3))                  # GAP
        return self.proj(h)

    def config(self):
        return dict(input_resolution=self.input_resolution,
                    latent_dim=self.latent_dim, n_res_blocks=self.n_res_blocks,
                    channels=list(self.channels))


# ----------------------------------------------------------------- action ---
def encode_action(p_start_xy, p_stop_xy, angle):
    """Design doc §5: a = [x, y, sin(theta), cos(theta)], where (x, y) is the
    tool position.  §5 also says to PRESERVE any extra variables the existing
    API carries rather than rewrite it -- this corpus's action is a *push*
    (start, stop, angle), so the displacement is appended as extra dims:

        a = [x_s, y_s, sin(th), cos(th), dx, dy]

    Positions are in metres and scaled by 1/WS_HALF so every component is O(1).
    """
    s = p_start_xy / 0.064
    d = (p_stop_xy - p_start_xy) / 0.064
    return torch.cat([s, torch.sin(angle)[:, None], torch.cos(angle)[:, None], d], dim=1)


ACTION_DIM = 6


class ActionEncoder(nn.Module):
    """Small MLP, per design doc §5 ("mapped through a small MLP")."""

    def __init__(self, in_dim=ACTION_DIM, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, out_dim))
        self.out_dim = out_dim

    def forward(self, a):
        return self.net(a)


# --------------------------------------------------- switched-linear model ---
class SwitchedLinearDynamics(nn.Module):
    """Residual switched-linear latent dynamics, design doc §8-10:

        z' = z + sum_k g_k(z,a) [ A_k z + B_k a_e + c_k ],
        g  = softmax(G(z, a_e))          (soft gate, 2-layer MLP)

    K=1 reduces exactly to a single residual linear model (the softmax over one
    logit is identically 1), so K=1 vs K>1 is a clean single-vs-switched test
    with every other part of the model held fixed.  Descriptors in the gate are
    out of scope for this record.
    """

    def __init__(self, latent_dim=256, K=8, act_dim=ACTION_DIM, act_emb=32,
                 gate_hidden=128, gate_layers=2, a_scale=0.01):
        super().__init__()
        self.latent_dim, self.K = int(latent_dim), int(K)
        self.act = ActionEncoder(act_dim, 64, act_emb)
        self.A = nn.Parameter(torch.randn(K, latent_dim, latent_dim) * a_scale / math.sqrt(latent_dim))
        self.B = nn.Parameter(torch.randn(K, act_emb, latent_dim) * a_scale)
        self.c = nn.Parameter(torch.zeros(K, latent_dim))
        gin = latent_dim + act_emb
        if gate_layers == 1:
            self.gate = nn.Linear(gin, K)
        else:
            self.gate = nn.Sequential(nn.Linear(gin, gate_hidden), nn.SiLU(),
                                      nn.Linear(gate_hidden, K))
        self.cfg = dict(latent_dim=latent_dim, K=K, act_dim=act_dim,
                        act_emb=act_emb, gate_hidden=gate_hidden,
                        gate_layers=gate_layers, a_scale=a_scale)

    def forward(self, z, a, return_gate=False):
        ae = self.act(a)                                   # (B, act_emb)
        g = F.softmax(self.gate(torch.cat([z, ae], dim=1)), dim=1)   # (B,K)
        # (B,K,D): per-mode delta
        dz_k = torch.einsum('bd,kde->bke', z, self.A) + \
               torch.einsum('bm,kme->bke', ae, self.B) + self.c[None]
        dz = (g[:, :, None] * dz_k).sum(1)
        if return_gate:
            return z + dz, dz, g
        return z + dz, dz


# ---------------------------------------------------------------- decoder ---
class OccDecoder(nn.Module):
    """z -> (B,1,res,res) occupancy logits.  Trained on a FROZEN encoder; the
    caller must pass a detached z (see train_decoder.py, which also asserts the
    encoder receives no gradient)."""

    def __init__(self, latent_dim=256, out_resolution=64, base_ch=256):
        super().__init__()
        self.out_resolution = out_resolution
        self.fc = nn.Linear(latent_dim, base_ch * 4 * 4)
        self.base_ch = base_ch
        chs = [base_ch, 128, 64, 32, 16]        # 4->8->16->32->64
        n_up = int(math.log2(out_resolution // 4))
        blocks = []
        for i in range(n_up):
            cin, cout = chs[i], chs[i + 1]
            blocks += [nn.Upsample(scale_factor=2, mode='nearest'),
                       nn.Conv2d(cin, cout, 3, padding=1),
                       nn.GroupNorm(8, cout), nn.SiLU(),
                       nn.Conv2d(cout, cout, 3, padding=1),
                       nn.GroupNorm(8, cout), nn.SiLU()]
        self.up = nn.Sequential(*blocks)
        self.out = nn.Conv2d(chs[n_up], 1, 3, padding=1)

    def forward(self, z):
        h = self.fc(z).view(-1, self.base_ch, 4, 4)
        return self.out(self.up(h))[:, 0]       # (B,H,W) logits
