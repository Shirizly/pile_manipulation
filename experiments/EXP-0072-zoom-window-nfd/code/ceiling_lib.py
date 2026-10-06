"""Scoring helpers for the ceiling study: frames, pooled accuracy, pool capture matrix."""
import glob, sys, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0043-batched-closed-loop/code"); sys.path.insert(0, "experiments/EXP-0053-narrow-domain-models/code")
from ceiling_common import *
from batched_closed_loop import goal_mask
from Baselines.common.goals import dist_field_from_mask
from simple_mpc.adapters import occ_from_particles, occ_for_scoring
from simple_mpc.learned_mpc import lyap
from eval_narrow import swept_region, acc, GOALS
from model.zoom_nfd.window import WindowSpec, window_batch, swept_region_window, paste_window_into_world
from model.zoom_nfd.world128 import raster128, LO, PX, RES

DIST = {g: torch.from_numpy(dist_field_from_mask(goal_mask(g))).float() for g in GOALS}
ZERO = torch.zeros(64, 64)


def region128(act):
    """swept region on the 128 world raster: world64 region sampled at 128 pixel centres (nearest)."""
    R = swept_region(act, "cpu")                                   # (B,64,64) bool
    c = (LO + (np.arange(128) + 0.5) * PX); i = np.clip(np.round((c + 0.064) / 0.128 * 63).astype(int), 0, 63)
    return R[:, i][:, :, i]


def paste_delta(pred_w, state_w, o0, p0, p1, spec):
    return (o0 + paste_window_into_world(pred_w - state_w, ZERO, p0, p1, spec)).clamp(0, 1)


def pool_caps(P, S0p, S1p):
    """P: (npool*64,64,64) predicted pasted/world64 next-state rasters; S0p: (npool,20,7) start states; S1p: (npool*64,20,7) true next.
    returns caps (npool, 13) with NaN for degenerate pool/goal (same den<=1e-9 skip rule as score_zoom) + truth dv list."""
    npool = len(S0p); caps = np.full((npool, len(GOALS)), np.nan)
    for p in range(npool):
        sl = slice(p * 64, (p + 1) * 64); o0 = occ_from_particles(S0p[p][None]).cpu(); t0 = occ_for_scoring(S0p[p][None][:, :, :3]); T1 = occ_for_scoring(S1p[sl][:, :, :3])
        for gi, g in enumerate(GOALS):
            vt = (lyap(T1, DIST[g]) - lyap(t0, DIST[g])).numpy(); den = vt.mean() - vt.min()
            if den <= 1e-9: continue
            vp = (lyap(P[sl].float(), DIST[g]) - lyap(o0.expand(64, -1, -1), DIST[g])).numpy()
            caps[p, gi] = (vt.mean() - vt[vp.argmin()]) / den
    return caps


def load_pools():
    """returns S0p (32,20,7), S1 (2048,20,7), P0,P1 (2048,2), same order as ceiling_resim / ceiling_model_preds"""
    S0p, S1, P0, P1 = [], [], [], []
    for f in sorted(glob.glob(D + "test_pools_v2/pools_*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for p in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == p)[:, 0]
            S0p.append(d["states"][ix[0]].float()); S1.append(d["states_"][ix].float()); P0.append(d["p_starts"][ix, :2]); P1.append(d["p_stops"][ix, :2])
    return torch.stack(S0p), torch.cat(S1), torch.cat(P0).numpy(), torch.cat(P1).numpy()
