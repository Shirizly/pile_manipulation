"""EXP-0075 layout-transport objective (2026-10-11, results/layout/). Crowding term = balanced entropic-OT (Sinkhorn, log domain) cost between the MODEL's predicted
64x64 occupancy (normalised to unit mass, pixel centres LO + i*PITCH) and the n uniform_layout slots (equal mass 1/n, EXP-0067 optimum), Euclidean cost, in lyapunov units
(metres / d_max, d_max = farthest mask distance in metres as in captrans.CrowdTransport). Differentiable in occ; the slot set is the same one wide_planner.ot_candidates uses.
  LayoutTransport(slots, mask64, lam=1.0, eps=0.02, iters=30)(occ (N,64,64)) -> (N,) ; .parts(occ) -> (term, zeros) for the crowd_parts logging interface
  hungarian_cost(xy, slots) -> exact balanced W1 (mean |d|, metres) of n cubes to the n slots (analysis metric 'transport to layout')."""
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt
from simple_mpc.adapters import OCC_BOUNDS, OCC_GRID
LO = OCC_BOUNDS["x_min"]; PITCH = (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"]) / (OCC_GRID - 1)


def hungarian_cost(xy, slots):
    P = np.asarray(xy, float); S = np.asarray(slots, float)
    r, c = linear_sum_assignment(np.linalg.norm(P[:, None] - S[None], axis=-1)); return float(np.linalg.norm(P[r] - S[c], axis=1).mean())


class LayoutTransport:
    def __init__(self, slots, mask64, lam=1.0, eps=0.02, iters=25, pool=2, device="cuda"):
        S = np.asarray(slots, float); self.pool = int(pool); ctr = LO + np.arange(OCC_GRID) * PITCH; ctr = ctr.reshape(-1, self.pool).mean(1); gx, gy = np.meshgrid(ctr, ctr, indexing="ij"); pix = np.stack([gx.ravel(), gy.ravel()], 1)   # pool x pool average-pooled grid
        self.d_max = float(distance_transform_edt(~np.asarray(mask64).astype(bool)).max()) * PITCH
        self.C = torch.as_tensor(np.linalg.norm(pix[:, None] - S[None], axis=-1) / self.d_max, dtype=torch.float32, device=device)   # (4096, n)
        self.n = len(S); self.lam, self.eps, self.iters = float(lam), float(eps), int(iters); self.logb = -float(np.log(self.n))

    def raw(self, occ):
        N = occ.shape[0]; occ = torch.nn.functional.avg_pool2d(occ[:, None], self.pool)[:, 0] if self.pool > 1 else occ; f = occ.reshape(N, -1); a = f / f.sum(1, keepdim=True).clamp_min(1e-6)
        loga = torch.log(a.clamp_min(1e-12)); K = -self.C / self.eps; v = torch.zeros(N, self.n, device=occ.device)
        for _ in range(self.iters):                                   # log-domain Sinkhorn: u over pixels, v over slots
            u = loga - torch.logsumexp(K[None] + v[:, None, :], 2)
            v = self.logb - torch.logsumexp(K[None] + u[:, :, None], 1)
        P = torch.exp(K[None] + u[:, :, None] + v[:, None, :])
        return (P * self.C[None]).sum((1, 2))

    def parts(self, occ):
        t = self.lam * self.raw(occ); return t, torch.zeros_like(t)

    def __call__(self, occ):
        return self.lam * self.raw(occ)
