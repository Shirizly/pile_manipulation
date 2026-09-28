"""Goal-aware candidate selection for closed-loop MPC (EXP-0056).

The pile-aware sampler (`SandboxManipulation.generate_action_samples(pile_aware=True)`)
places the blade against the pile but is blind to the goal: it proposes pushes that
move correctly placed cubes, or carry cubes into goal parts that are already full.
These functions take a bank of pile-aware pushes (m, 4) = [sx, sy, ex, ey] (blade-centre
start/stop, metres; the blade is 40 mm wide and perpendicular to the push) for ONE env,
its particle xy (n, 2), and the goal, and return the n_out candidates the planner starts
from. They use a geometric carry model only (no learned model):

  carry: a cube inside the swath (lateral |v| <= 20 mm + half a cube, along 0..L) ends at
         the stop line, lateral position unchanged.

  * `select_misplaced` (S1): weight each push by the misplaced mass in its swath -- cubes
    outside the goal (weight 1) or the above-capacity share of a crowded goal pixel.
  * `select_deposit`   (S2): score each push by the change in placement quality of the cubes
    it carries, q(dest) - q(origin), q = 1 - over-capacity share inside the goal and
    -min(1, 5 * distance field) outside; softmax-sample.
  * `ot_proposals`     (S3): pushes along the entropic OT displacement (current image ->
    uniform goal target) of a misplaced cube, blade just behind it; mixed with pile-aware.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from simple_mpc.adapters import OCC_BOUNDS, occ_for_scoring
from simple_mpc.value_functions import _gauss_blur, _normalise, entropic_plan, uniform_target, _grid_xy

HALF_BLADE, HALF_CUBE = 0.020, 0.0025
LO, HI = OCC_BOUNDS["x_min"], OCC_BOUNDS["x_max"]


def _on_cpu(fn):
    """Host-side bookkeeping: Genesis sets torch's default device to cuda, so every
    factory call here is pinned to the CPU explicitly."""
    import functools

    @functools.wraps(fn)
    def w(*a, **k):
        with torch.device("cpu"):
            return fn(*a, **k)
    return w


def _pix(xy):
    return ((xy - LO) / (HI - LO) * 63).round().long().clamp(0, 63)


def carry(xy: torch.Tensor, acts: torch.Tensor):
    """xy (n, 2), acts (m, 4) -> in_swath (m, n) bool, dest (m, n, 2)."""
    s, e = acts[:, :2], acts[:, 2:]
    d = e - s; L = d.norm(dim=1, keepdim=True).clamp_min(1e-9); u = d / L
    nrm = torch.stack([-u[:, 1], u[:, 0]], 1)
    rel = xy[None] - s[:, None]                                          # (m, n, 2)
    al = (rel * u[:, None]).sum(-1); lat = (rel * nrm[:, None]).sum(-1)
    inside = (lat.abs() <= HALF_BLADE + HALF_CUBE) & (al >= -HALF_CUBE) & (al <= L + HALF_CUBE)
    dest = s[:, None] + (L[:, None] + HALF_CUBE) * u[:, None] + lat[..., None] * nrm[:, None]
    return inside, torch.where(inside[..., None], dest, xy[None].expand_as(dest))


def _excess_share(xy, mask):
    """Over-capacity share at each cube's pixel: relu(pb - rho*) / pb (0 where not crowded)."""
    occ = occ_for_scoring(torch.cat([xy, torch.full_like(xy[:, :1], HALF_CUBE)], 1)[None])
    pb = _gauss_blur(_normalise(occ), 1.25)[0]
    rho = 1.0 / float(mask.sum())
    return lambda p: (torch.relu(pb - rho) / pb.clamp_min(1e-9))[p[..., 0], p[..., 1]]


@_on_cpu
def select_misplaced(bank, xy, mask, dist, n_out, gen=None, floor=0.05):
    m = torch.as_tensor(np.asarray(mask)).bool()
    p = _pix(xy); ex = _excess_share(xy, mask)(p)
    w = torch.where(m[p[:, 0], p[:, 1]], ex, torch.ones_like(ex))             # (n,)
    ins, _ = carry(xy, bank)
    score = (ins.float() * w[None]).sum(1) + floor
    return bank[torch.multinomial(score, n_out, replacement=False, generator=gen)]


@_on_cpu
def select_deposit(bank, xy, mask, dist, n_out, gen=None, tau=0.5):
    m = torch.as_tensor(np.asarray(mask)).bool()
    exf = _excess_share(xy, mask)

    def q(pts):
        p = _pix(pts)
        inside = m[p[..., 0], p[..., 1]]
        return torch.where(inside, 1.0 - exf(p), -(5 * dist[p[..., 0], p[..., 1]]).clamp(max=1.0))
    ins, dest = carry(xy, bank)
    score = ((q(dest) - q(xy)[None]) * ins.float()).sum(1)
    return bank[torch.multinomial(torch.softmax(score / tau, 0), n_out, replacement=False, generator=gen)]


@_on_cpu
def ot_proposals(xy, mask, n_out, gen=None, ang_sd_deg=10.0, eps=1.0):
    """n_out pushes along the OT displacement of misplaced cubes (see module doc). The blade
    starts one cube-width behind the chosen cube; starts whose blade footprint lands on a
    cube are redrawn (4x oversampling), the rest filled with the valid ones repeated."""
    occ = occ_for_scoring(torch.cat([xy, torch.full_like(xy[:, :1], HALF_CUBE)], 1)[None])[0]
    tgt = uniform_target(mask)
    si, ti, f, g, C = entropic_plan(occ, tgt, eps=eps)
    a = _normalise(occ).reshape(-1)[si]; b = tgt.reshape(-1)[ti]
    P = torch.exp((f[:, None] + g[None] - C) / eps) * a[:, None] * b[None]
    XY = _grid_xy(occ.device)
    bary = (P @ XY[ti]) / P.sum(1, keepdim=True).clamp_min(1e-12)           # (ns, 2) px
    disp = torch.zeros(64 * 64, 2); disp[si] = (bary - XY[si])
    p = _pix(xy); dcube = disp[p[:, 0] * 64 + p[:, 1]] * (HI - LO) / 63       # (n, 2) metres
    mag = dcube.norm(dim=1)
    if float(mag.max()) < 1e-4:
        return None
    k = 4 * n_out
    idx = torch.multinomial(mag + 1e-6, k, replacement=True, generator=gen)
    th = torch.atan2(dcube[idx, 1], dcube[idx, 0]) + torch.randn(k, generator=gen) * math.radians(ang_sd_deg)
    u = torch.stack([th.cos(), th.sin()], 1); nrm = torch.stack([-u[:, 1], u[:, 0]], 1)
    lat = (torch.rand(k, generator=gen) - 0.5) * 0.024
    s = xy[idx] - u * (2 * HALF_CUBE + 0.003) + nrm * lat[:, None]
    L = (mag[idx] + 0.005).clamp(0.020, 0.070)
    acts = torch.cat([s, s + u * L[:, None]], 1)
    rel = xy[None] - s[:, None]
    al = (rel * u[:, None]).sum(-1); lt = (rel * nrm[:, None]).sum(-1)
    hit = ((lt.abs() <= HALF_BLADE + HALF_CUBE) & (al.abs() <= HALF_CUBE + 0.002)).any(1)
    inb = (s.abs() <= HI - 0.01).all(1)
    ok = acts[~hit & inb]
    if len(ok) == 0:
        return None
    return ok[torch.arange(n_out) % len(ok)]
