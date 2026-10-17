"""EXP-0075 capacity-transport objective and proposals (2026-10-11, results/captrans/). Thin-letter test (w15 / w10) of the hypothesis that the OBJECTIVE
(in-goal mass) is what stops the planner from drawing the letter.
  capacity_slots(goal)      -> (S, 2) world metres: 5 mm lattice (aligned to the world origin) of cube centres whose 5 x 5 mm footprint fits inside the goal mask
                               (letter_goal_mask at 128 px = 1 mm / px, EDT(mask) >= 2.5 px).
  capacity_plan(xy, slots)  -> rectangular Hungarian (Euclidean, W1-type: common mass stays put) assignment of the n cubes to the S slots; displacement (n, 2) and
                               the assigned slot index. S < n (over-capacity goal): the slot set is tiled so every cube gets a slot (duplicates = unavoidable crowding).
  transport_cost(xy, slots) -> mean |displacement| (metres) of that plan on a TRUE state: the analysis metric (0 = letter filled without crowding).
  captrans_candidates(...)  -> pushes along each displaced cube's displacement and at +-10 / +-20 deg around it (blade CUBE_HALF + BLADE_HW + gap behind the cube,
                               length |d| + gap + extra, clamped, projected); replaces the fetch extras in the 'captrans proposals' arms.
  CrowdTransport            -> the crowding-transport term added to SeqObjective._value (objective 'captrans'): sum_s relu(m_s - m_cube) d_free(s) + off-slot in-mask
                               mass x d_free(pixel), over total mass x d_max (lyapunov units); d_free from the TRUE current state once per decision."""
import math, numpy as np, torch
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from Baselines.common.goals import named_goal_mask as letter_goal_mask   # == letter_goal_mask on letter names; also disc_* (T0b)
from simple_mpc.adapters import OCC_BOUNDS, OCC_GRID, OCC_FOOTPRINT_RADIUS
from simple_mpc.learned_mpc import project_push, L_MIN, L_MAX

LO = np.array([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]]); HI = np.array([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]])
PITCH = (HI[0] - LO[0]) / (OCC_GRID - 1)                      # 64-grid pixel-centre pitch (2.03 mm): pixel i centre = LO + i * PITCH (occ_from_particles convention)
CUBE_HALF, BLADE_HW = 0.0025, 0.001
M_CUBE = math.pi * OCC_FOOTPRINT_RADIUS ** 2                   # one cube's raster mass (occ_for_scoring per-particle constant, 4.91 pixel units)
SLOT_PITCH, SLOT_HALF = 0.005, 0.0025
_SLOTS = {}


def capacity_slots(goal, res=128, min_edt_px=2.5):
    if goal in _SLOTS:
        return _SLOTS[goal]
    m = letter_goal_mask(goal, res, res).astype(bool)          # mask[x_px, y_px]
    ok = distance_transform_edt(m) >= min_edt_px
    ax = np.arange(-0.060, 0.0601, SLOT_PITCH); gx, gy = np.meshgrid(ax, ax, indexing="ij"); pts = np.stack([gx.ravel(), gy.ravel()], 1)
    px = np.rint((pts - LO) / (HI - LO) * (res - 1)).astype(int).clip(0, res - 1)
    _SLOTS[goal] = pts[ok[px[:, 0], px[:, 1]]].copy()
    return _SLOTS[goal]


def capacity_plan(xy, slots):
    """xy (n, 2), slots (S, 2) numpy -> displacement (n, 2), assigned slot index (n,) (into the tiled slot set), tiled slots."""
    P = np.asarray(xy, dtype=float); S = np.asarray(slots, dtype=float); n = len(P)
    if len(S) < n:
        S = np.tile(S, (int(math.ceil(n / len(S))), 1))
    r, c = linear_sum_assignment(np.linalg.norm(P[:, None] - S[None], axis=-1))
    d = np.zeros_like(P); idx = np.zeros(n, dtype=int); d[r] = S[c] - P[r]; idx[r] = c
    return d, idx, S


def transport_cost(xy, slots):
    d, _, _ = capacity_plan(xy, slots); return float(np.linalg.norm(d, axis=1).mean())


def captrans_candidates(xy, slots, gaps=(0.004, 0.008), extras=(0.008, 0.015, 0.025), angles_deg=(0.0, 10.0, -10.0, 20.0, -20.0), min_disp=0.005, lo=None, hi=None):
    """xy (n, 2) world on DEV -> (rows (M, 4) on xy.device, cube index (M,) long). M = 0 when no cube is displaced by more than min_disp."""
    dev = xy.device; d, _, _ = capacity_plan(xy.detach().cpu().double().numpy(), slots)
    d = torch.as_tensor(d, dtype=torch.float32, device=dev); n = d.norm(dim=-1); keep = n > min_disp
    idx = keep.nonzero().flatten(); d, n = d[keep], n[keep]
    if len(idx) == 0:
        return torch.zeros(0, 4, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    u0 = d / n[:, None]; A = torch.tensor(angles_deg, device=dev) * math.pi / 180; ca, sa = torch.cos(A), torch.sin(A)
    u = torch.stack([u0[:, None, 0] * ca - u0[:, None, 1] * sa, u0[:, None, 0] * sa + u0[:, None, 1] * ca], -1)        # (m, a, 2)
    G = torch.tensor(gaps, device=dev); E = torch.tensor(extras, device=dev); na, ng, ne = len(A), len(G), len(E)
    uu = u[:, :, None, None, :]
    s0 = xy[idx].float()[:, None, None, None, :] - uu * (CUBE_HALF + BLADE_HW + G)[None, None, :, None, None]          # (m, a, g, 1, 2)
    L = (n[:, None, None, None] + G[None, None, :, None] + E[None, None, None, :]).clamp(lo or L_MIN, hi or L_MAX)     # (m, a, g, e)
    s0 = s0.expand(-1, na, ng, ne, 2); e0 = s0 + uu * L[..., None]
    rows = project_push(torch.cat([s0, e0], -1).reshape(-1, 4), lo or L_MIN, hi or L_MAX)[0]
    return rows, idx[:, None, None, None].expand(-1, na, ng, ne).reshape(-1)


class CrowdTransport:
    """term(occ (N, 64, 64)) -> (N,), lyapunov units, differentiable in occ. Built once per decision from the TRUE current cube centres xy0 (n, 2)."""

    def __init__(self, xy0, slots, mask64, dist64, device="cuda", occ_thresh=0.0035, m_cube=M_CUBE):
        S = np.asarray(slots, dtype=float); self.S = S; xy0 = np.asarray(xy0, dtype=float)
        occupied = (np.linalg.norm(S[:, None] - xy0[None], axis=-1) < occ_thresh).any(1); free = S[~occupied]; self.n_free = int(len(free))
        d_px = distance_transform_edt(~np.asarray(mask64).astype(bool)); self.d_max = float(d_px.max()) * PITCH          # lyapunov normaliser in metres
        ctr = LO[0] + np.arange(OCC_GRID) * PITCH; gx, gy = np.meshgrid(ctr, ctr, indexing="ij"); pix = np.stack([gx.ravel(), gy.ravel()], 1)   # (4096, 2) pixel centres (mask[x, y])
        dfree = lambda p: np.linalg.norm(p[:, None] - free[None], axis=-1).min(1) if len(free) else np.full(len(p), self.d_max)
        ds, dq = dfree(S) / self.d_max, dfree(pix) / self.d_max
        ox = np.clip(np.minimum(S[:, None, 0] + SLOT_HALF, ctr[None] + PITCH / 2) - np.maximum(S[:, None, 0] - SLOT_HALF, ctr[None] - PITCH / 2), 0, PITCH) / PITCH   # (S, 64) overlap fraction along x
        oy = np.clip(np.minimum(S[:, None, 1] + SLOT_HALF, ctr[None] + PITCH / 2) - np.maximum(S[:, None, 1] - SLOT_HALF, ctr[None] - PITCH / 2), 0, PITCH) / PITCH
        W = (ox[:, :, None] * oy[:, None, :]).reshape(len(S), -1)                                                           # (S, 4096) pixel-area fraction inside slot s
        mk = np.asarray(mask64).astype(float).ravel(); off = np.clip(mk - W.sum(0), 0, 1)                                   # in-mask area off every slot footprint
        self.W = torch.as_tensor(W, dtype=torch.float32, device=device); self.ds = torch.as_tensor(ds, dtype=torch.float32, device=device)
        self.off_w = torch.as_tensor(off * dq, dtype=torch.float32, device=device); self.m_cube = float(m_cube)

    def slot_mass(self, occ):
        return occ.reshape(occ.shape[0], -1) @ self.W.T                                                                      # (N, S)

    def parts(self, occ):
        """(excess part, off-slot part), each (N,) in lyapunov units."""
        f = occ.reshape(occ.shape[0], -1); m = f @ self.W.T; tot = f.sum(1).clamp_min(1e-6)
        return ((m - self.m_cube).clamp_min(0) * self.ds).sum(1) / tot, (f @ self.off_w) / tot

    def __call__(self, occ):
        a, b = self.parts(occ); return a + b
