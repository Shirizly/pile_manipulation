"""Why doesn't the sand settle? Sweep the material knobs on short settle-only sims.

Real dry sand is high-friction and dissipative: dropped in a heap it stops
almost at once and holds a cone at its angle of repose (~30-35 deg for rounded
grains, up to ~45 for angular). Our setup does not do that -- the median grain
reaches rest but the top 0.5% creeps indefinitely at ~0.4 um/step, and a
post-spawn settle never passes the rest criterion (EXP-0007).

The prime suspect is stiffness. `configs/sand.yaml` sets E = 1e5 Pa against
Genesis' own MPM.Sand default of 1e6, lowered to save substeps. But the CFL
bound it was lowered for is not tight at 1e5: with dx = 4 mm and rho = 1500,
the elastic wave speed sqrt(E/rho) is 8.2 m/s, so a stable substep is ~4.9e-4 s
and only ~8 substeps are needed where we run 30. So the medium was softened for
margin that was already there, and a soft continuum relaxes viscoelastically
under its own weight instead of locking up.

Each row is a spawn plus a settle -- no pushes, a few seconds of sim -- and
reports the two things that decide physicality:

  steps to rest   how long until q0.995 grain speed < the rest threshold, or
                  'never' within the cap. Sand should be fast.
  repose angle    the free-surface slope of the settled heap. This is the
                  physical fingerprint: too soft or too low-friction and the
                  pile pancakes to a shallow angle; correct and it holds
                  30-45 deg.

    python scripts/sand_material_sweep.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# (label, E, friction_angle, substeps). Row 1 is the current config.
SETUPS = [
    ("current  E1e5 f45 s30", 1.0e5, 45.0, 30),
    ("default  E1e6 f45 s30", 1.0e6, 45.0, 30),
    ("stiffer  E1e6 f45 s60", 1.0e6, 45.0, 60),
    ("stiff    E3e6 f45 s60", 3.0e6, 45.0, 60),
    ("stiffest E1e7 f45 s120", 1.0e7, 45.0, 120),
    ("lowfric  E1e6 f30 s30", 1.0e6, 30.0, 30),
]


def repose_angle(pos, floor_z, n_bins=12):
    """Free-surface slope of a settled heap, in degrees.

    Bin grains by radius, take the top of each bin, and fit a line to the outer
    flank. A cone of sand gives its angle of repose; a pancake gives ~0.
    """
    xy = pos[..., :2].reshape(-1, 2)
    r = xy.norm(dim=-1)
    z = pos[..., 2].reshape(-1) - floor_z
    if r.numel() < 50:
        return float("nan"), float("nan")
    r_max = float(r.quantile(0.99))
    edges = torch.linspace(0.0, r_max, n_bins + 1, device=r.device)
    rs, zs = [], []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if int(m.sum()) < 8:
            continue
        rs.append(0.5 * float(edges[i] + edges[i + 1]))
        zs.append(float(z[m].quantile(0.95)))       # surface, not interior
    if len(rs) < 4:
        return float("nan"), float("nan")
    rt = torch.tensor(rs); zt = torch.tensor(zs)
    # Fit the OUTER half: the crown is flat on a frustum and would bias the
    # slope toward zero.
    k = len(rs) // 2
    ro, zo = rt[k:], zt[k:]
    A = torch.stack([ro, torch.ones_like(ro)], dim=1)
    slope = float(torch.linalg.lstsq(A, zo.unsqueeze(1)).solution[0, 0])
    return math.degrees(math.atan(abs(slope))), 1000 * float(zt.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sand.yaml")
    ap.add_argument("--cap", type=int, default=2000, help="settle step cap")
    ap.add_argument("--check-every", type=int, default=25)
    args = ap.parse_args()

    from Genesis.sand_manipulation import SandManipulation, load_sand_config

    print("\nSAND MATERIAL SWEEP -- spawn + settle only, no pushes")
    print("real dry sand: stops fast, holds 30-45 deg of repose\n")
    print(f"{'setup':24s} {'to rest':>9s} {'resid':>8s} {'repose':>8s} "
          f"{'height':>8s} {'mass':>7s} {'wall s':>7s}")
    print("-" * 78)

    import time
    for label, E, fric, substeps in SETUPS:
        cfg = load_sand_config(args.config)
        cfg["sand"]["E"] = E
        cfg["sand"]["friction_angle"] = fric
        cfg["simulation"]["substeps_mpm"] = substeps
        cfg.setdefault("data_collection", {})["sand"] = True
        # Keep the cap generous so "never" means never, not under-budgeted.
        cfg["simulation"]["settle_steps"] = args.cap
        cfg["simulation"]["settle_check_every"] = args.check_every

        sim = SandManipulation(config=cfg, n_envs=1, debug=False, viewer_type=None)
        sim.build()
        floor_z = float(sim._wall_thickness) / 2.0
        thr = float(sim._settle_vel_threshold)

        sim.shuffle_particles()
        t0 = time.time()
        reached = None
        for step in range(1, args.cap + 1):
            sim._step_scene()
            if step % args.check_every == 0:
                v = sim.sand.get_particles_vel().norm(dim=-1).reshape(-1)
                if float(v.quantile(0.995)) < thr:
                    reached = step
                    break
        wall = time.time() - t0

        v = sim.sand.get_particles_vel().norm(dim=-1).reshape(-1)
        resid = 1000 * float(v.quantile(0.995))
        pos = sim._get_particle_positions()
        ang, h = repose_angle(pos, floor_z)
        from transforms.sand_occupancy import sand_mass
        B = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
        pb = pos[None] if pos.dim() == 2 else pos
        mass = float(sand_mass(pb, B).mean())

        print(f"{label:24s} {(str(reached) if reached else 'never'):>9s} "
              f"{resid:7.2f}m {ang:7.1f}d {h:7.1f}m {mass:7.4f} {wall:7.1f}",
              flush=True)
        sim.destroy()

    print("\nto rest = settle steps until q0.995 < 1.0 mm/s (cap "
          f"{args.cap}); resid in mm/s; repose in degrees; height in mm")


if __name__ == "__main__":
    main()
