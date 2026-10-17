"""Genesis-free helpers for oracle.py --legal (RAL plan G4, 2026-10-15). Imported by oracle.py; unit-tested in
tests/test_oracle_legal.py on CPU tensors.

Touchdown legality exactly as the EXP-0075 harness: Genesis/action_sampling.py::legalize_pushes_torch with the same
cube / blade constants, offset grid, nearest-first order and tray box as wide_planner.make_legal_scorer (planner
candidates) and closed_loop.py's executed-push legaliser (box = OCC_BOUNDS x_min/x_max +- XY_MARGIN on both axes).
The constants are literals here because importing wide_planner pulls in Genesis (via eval_wide); oracle.py asserts at
runtime that they equal wide_planner's (Genesis is loaded there anyway)."""
import torch

CUBE_HALF, BLADE_HL, BLADE_HW = 0.0025, 0.02, 0.001   # == wide_planner.CUBE_HALF, BLADE_HL, BLADE_HW (asserted in oracle.py)


def legal_box():
    from simple_mpc.adapters import OCC_BOUNDS
    from simple_mpc.learned_mpc import XY_MARGIN
    return (OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN)


def make_legaliser(state):
    """state (n, 7) [x, y, z, qw, qx, qy, qz] -> callable acts (N, 4) -> (acts', shift_m (N,), ok (N,)) on acts.device.
    Same call as wide_planner.make_legal_scorer, but runs on whatever device the candidates are on (CPU in the oracle)."""
    from Genesis.action_sampling import legalize_pushes_torch
    from model.retrieval.frame import yaw_from_quat
    xy = state[None, :, :2].float(); yaw = yaw_from_quat(state[:, 3:7].float())[None]
    box = legal_box()
    return lambda acts: legalize_pushes_torch(acts.float(), xy.to(acts.device), yaw.to(acts.device), CUBE_HALF, BLADE_HL, BLADE_HW, box=box)


def masked_values(v_legal, ok):
    """Values of the legal (simulated) rows scattered into a full-population vector; un-legalisable rows get +inf."""
    v = torch.full((len(ok),), float("inf"), device="cpu")   # explicit: Genesis sets torch's default device to cuda
    v[ok] = v_legal.float().cpu()
    return v


def cem_refit(pop, v, n_elite, n_new, gen, l_min, l_max):
    """Gaussian refit to the best FINITE-cost rows of pop (N, 4) under v (N,) (inf = illegal, never an elite).
    Returns the new projected population (n_new, 4), or None when no row is finite (caller resamples pile-aware).
    One finite elite: sd = 5 mm on every coordinate (std of a single row is undefined)."""
    from simple_mpc.learned_mpc import project_push
    v = v.cpu(); fin = torch.isfinite(v)
    if not bool(fin.any()):
        return None
    idx = v.argsort()[:min(n_elite, int(fin.sum()))]
    el = pop[idx]
    mu = el.mean(0)
    sd = el.std(0).clamp_min(1e-3) if len(el) > 1 else torch.full((4,), 0.005, device="cpu")
    return project_push(mu + sd * torch.randn(n_new, 4, device="cpu", generator=gen), l_min=l_min, l_max=l_max)[0]
