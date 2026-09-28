"""Capacity-aware value functions for goal-shape MPC (EXP-0055) and the coverage metric.

Lyapunov (`learned_mpc.lyap`) is <image, goal distance field>: linear in the image, i.e.
the transport cost to the goal SET with unlimited capacity, so a cube arriving in an
already-full stroke scores the same as one filling an empty part. Everything here
compares the (mass-normalised) image with the UNIFORM distribution over the goal mask
instead, which has finite capacity per pixel.

All images are (B, 64, 64) on the planner grid (`adapters.occ_from_particles` /
`occ_for_scoring` convention: dim 0 = world x). Every value is a COST (lower = better),
mass-normalised, so it drops into `ModelObjective` in place of lyapunov.

  * `SlicedEMD`        - V1: sliced Wasserstein-1 to the uniform target (fixed pixel
                          projections, optional average-pool downsampling for cost).
  * `linearised_emd_field` - V2: entropic OT dual potential from the CURRENT image to the
                          target, evaluated at every pixel (soft c-transform); used as a
                          drop-in distance field, <image, phi>, same cost as lyapunov.
  * `CrowdingPenalty`  - V3: lyapunov + lam * sum_goal relu(blur(p) - rho*)^2 / rho*, a
                          Gaussian over-capacity repulsion that acts only inside the goal;
                          `floor_single` keeps rho* >= one isolated cube's density.
  * `coverage_metrics` - evaluation only: sliced EMD (64 px, 64 dirs) to the target and
                          the covered-area fraction; `uniform_layout` gives their ceiling.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

H = 64


def uniform_target(mask) -> torch.Tensor:
    """(64, 64) bool mask -> (64, 64) float distribution, uniform over the mask, sums to 1."""
    m = torch.as_tensor(np.asarray(mask)).float()
    return m / m.sum()


def _normalise(occ):
    return occ / occ.sum((-2, -1), keepdim=True).clamp_min(1e-9)


class SlicedEMD:
    """Sliced W1 between normalised images and a fixed target, on a `res`-px grid (average
    pooled from 64). Distances in 64-px pixel units. Differentiable in the image."""

    def __init__(self, target: torch.Tensor, res: int = 32, n_dirs: int = 16, device="cuda"):
        self.pool = H // res
        self.res = res
        ys, xs = torch.meshgrid(torch.arange(res, device=device).float(),
                                torch.arange(res, device=device).float(), indexing="ij")
        C = torch.stack([ys.reshape(-1), xs.reshape(-1)], 1) * self.pool      # (N, 2) in 64-px units
        ang = torch.arange(n_dirs, device=device).float() * math.pi / n_dirs
        P = C @ torch.stack([ang.cos(), ang.sin()], 0)                        # (N, K)
        sp, order = torch.sort(P, 0)
        self.order = order.T.contiguous()                                     # (K, N)
        self.deltas = (sp[1:] - sp[:-1]).T.contiguous()                       # (K, N-1)
        t = self._down(target.to(device)[None])[0].reshape(-1)
        self.t_sorted = t[self.order]                                         # (K, N)

    def _down(self, occ):
        occ = _normalise(occ.float())
        if self.pool > 1:
            occ = F.avg_pool2d(occ[:, None], self.pool)[:, 0] * self.pool ** 2
        return occ

    def __call__(self, occ):
        a = self._down(occ).reshape(occ.shape[0], -1)                        # (B, N)
        g = a[:, self.order] - self.t_sorted[None]                           # (B, K, N)
        return (torch.cumsum(g, 2)[:, :, :-1].abs() * self.deltas[None]).sum(2).mean(1)


def _grid_xy(device):
    ys, xs = torch.meshgrid(torch.arange(H, device=device).float(),
                            torch.arange(H, device=device).float(), indexing="ij")
    return torch.stack([ys.reshape(-1), xs.reshape(-1)], 1)                   # (4096, 2)


def entropic_plan(src: torch.Tensor, target: torch.Tensor, eps: float = 1.0, n_iter: int = 300):
    """Log-domain Sinkhorn with Euclidean (W1) cost in pixels between two (64, 64) images
    (normalised here), restricted to their supports. Returns (src_idx, tgt_idx, f, g, eps):
    flat pixel indices of both supports and the dual potentials on them."""
    dev = src.device
    a = _normalise(src.float()).reshape(-1); b = target.to(dev).float().reshape(-1)
    si = torch.nonzero(a > 1e-6 * a.max())[:, 0]; ti = torch.nonzero(b > 0)[:, 0]
    XY = _grid_xy(dev)
    C = torch.cdist(XY[si], XY[ti])                                            # (ns, nt) px
    la, lb = (a[si] / a[si].sum()).log(), (b[ti] / b[ti].sum()).log()
    f = torch.zeros(len(si), device=dev); g = torch.zeros(len(ti), device=dev)
    for _ in range(n_iter):
        f = -eps * torch.logsumexp(lb[None] + (g[None] - C) / eps, 1)
        g = -eps * torch.logsumexp(la[:, None] + (f[:, None] - C) / eps, 0)
    return si, ti, f, g, C


def linearised_emd_field(occ0: torch.Tensor, target: torch.Tensor, eps: float = 1.0,
                         n_iter: int = 300) -> torch.Tensor:
    """V2 field: the source-side dual potential of OT(current image -> target), extended to
    every pixel by the soft c-transform phi(x) = -eps log sum_y b_y exp((g_y - |x-y|)/eps).
    <p, phi> is then the first-order (subgradient) model of W1(p, target) around the current
    state: pixels in over-supplied goal parts get a HIGHER potential than empty goal parts,
    and outside the goal it grows like the distance to where mass is still missing.
    Returned in 64-px units, shifted to min 0, as a (64, 64) field."""
    dev = occ0.device
    si, ti, f, g, _ = entropic_plan(occ0.reshape(H, H), target, eps, n_iter)
    b = target.to(dev).float().reshape(-1)[ti]; lb = (b / b.sum()).log()
    XY = _grid_xy(dev)
    Cx = torch.cdist(XY, XY[ti])                                               # (4096, nt)
    phi = -eps * torch.logsumexp(lb[None] + (g[None] - Cx) / eps, 1)
    return (phi - phi.min()).reshape(H, H)


def _gauss_blur(p, sigma_px):
    r = int(math.ceil(3 * sigma_px))
    x = torch.arange(-r, r + 1, device=p.device).float()
    k = torch.exp(-x ** 2 / (2 * sigma_px ** 2)); k = k / k.sum()
    q = p[:, None]
    q = F.conv2d(F.pad(q, (0, 0, r, r), mode="constant"), k.view(1, 1, -1, 1))
    q = F.conv2d(F.pad(q, (r, r, 0, 0), mode="constant"), k.view(1, 1, 1, -1))
    return q[:, 0]


class CrowdingPenalty:
    """V3 over-capacity term (added to lyapunov by the caller): sum over goal pixels of
    relu(blur_sigma(p) - rho*)^2 / rho*, p the mass-normalised image and rho* = 1/|mask|
    the uniform target density. Zero while every goal part is at or under capacity, so
    bringing mass into an EMPTY part of the goal is never penalised; crowding cubes into
    a full stroke is, like a Gaussian repulsion that only switches on above capacity."""

    def __init__(self, mask, sigma_px: float = 1.25, device="cuda", floor_single: bool = False,
                 n_particles: int = 20):
        self.m = torch.as_tensor(np.asarray(mask)).float().to(device)
        self.rho = 1.0 / float(self.m.sum())
        self.sigma = sigma_px
        if floor_single:
            # capacity never below ONE isolated cube's blurred peak density (mass 1/n): for a
            # goal much larger than the cubes' area (quadrants), 1/|mask| is below a lone
            # cube, so without the floor every cube inside the goal is penalised and cubes
            # flee it (EXP-0055 RUN-0001, quadrant_0). With it only neighbours crowding each
            # other are -- the pairwise-repulsion reading of the term.
            from simple_mpc.adapters import occ_for_scoring
            one = occ_for_scoring(torch.tensor([[[0.0, 0.0, 0.0025]]], device="cpu")).to(device)
            self.rho = max(self.rho, float(_gauss_blur(_normalise(one), sigma_px).max()) / n_particles)

    def __call__(self, occ):
        pb = _gauss_blur(_normalise(occ.float()), self.sigma)
        return ((F.relu(pb - self.rho) ** 2) * self.m[None]).sum((1, 2)) / self.rho


def coverage_metrics(occ: torch.Tensor, mask, sigma_px: float = 1.25, n_dirs: int = 64):
    """Evaluation metrics (METRICS.md `coverage_emd`, `covered_frac`) for (B, 64, 64) SCORING
    images: sliced W1 (64 px, 64 directions, 64-px pixel units) to the uniform target over
    the mask, and the fraction of goal pixels whose blurred normalised density reaches half
    the uniform target density."""
    dev = occ.device
    t = uniform_target(mask).to(dev)
    emd = SlicedEMD(t, res=64, n_dirs=n_dirs, device=dev)(occ)
    m = torch.as_tensor(np.asarray(mask)).to(dev).bool()
    pb = _gauss_blur(_normalise(occ.float()), sigma_px)
    cov = ((pb >= 0.5 * float(t[m][0])) & m[None]).sum((1, 2)).float() / m.sum()
    return emd, cov


def uniform_layout(mask, n: int = 20, iters: int = 60, seed: int = 0) -> np.ndarray:
    """Lloyd relaxation of n cube centres over the mask pixels (the 'nice uniform
    distribution' for n cubes): returns (n, 2) world-metre xy. Rasterised with
    `occ_for_scoring` it gives the coverage ceiling for that goal and cube count."""
    from simple_mpc.adapters import OCC_BOUNDS
    m = np.asarray(mask).astype(bool)
    pix = np.argwhere(m).astype(float)                                         # (M, 2) (i=x, j=y)
    rng = np.random.default_rng(seed)
    c = pix[rng.choice(len(pix), n, replace=len(pix) < n)] + rng.normal(0, 0.1, (n, 2))
    for _ in range(iters):
        lab = ((pix[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for k in range(n):
            if (lab == k).any():
                c[k] = pix[lab == k].mean(0)
    lo, hi = OCC_BOUNDS["x_min"], OCC_BOUNDS["x_max"]
    return lo + c / (H - 1) * (hi - lo)
