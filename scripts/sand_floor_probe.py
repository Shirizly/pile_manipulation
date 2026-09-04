"""Can the sand rest on the FRICTIONAL tray floor instead of a frictionless boundary?

The pile pancakes to a 0.2-3.2 degree slope and never holds an angle of repose,
and stiffer sand pancakes MORE (height 5.8 -> 0.3 mm as E goes 1e5 -> 3e6),
which rules out stiffness and points at the base. The cause is in Genesis'
`CubeBoundary.impose_pos_vel`: on contact it does `vel[i] *= -restitution` for
the crossed axis ONLY and leaves both tangential components untouched. With
restitution 0 that is a perfectly free-slip floor, and a frictional medium on a
frictionless base cannot hold a repose angle -- the bottom layer has no shear
resistance, so the heap spreads until lateral stress vanishes.

`sand_manipulation.py` puts the MPM domain boundary exactly AT the tray floor on
the finding that fixed rigid geoms do not couple to MPM. But the rigid material
defaults are `needs_coup=True` and `coup_friction=0.1`, and the Legacy coupler
does implement Coulomb friction for rigid-MPM
(`legacy_coupler.py`: rvel_tan + rvel_normal * coup_friction). So the geometry
may be masking the coupling rather than the coupling being absent: particles are
clamped by the boundary before they ever reach the floor's SDF.

This drops the domain floor below the tray floor and asks which one catches the
sand:

  rests at ~10 mm   the rigid floor couples -> friction is available, and
                    coup_friction becomes the knob for the repose angle
  falls to the drop  fixed geoms really do not couple -> the frictionless
                    boundary is unavoidable and the sand is a puddle by
                    construction

    python scripts/sand_floor_probe.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sand_material_sweep import repose_angle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sand.yaml")
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--pile-radius", type=float, default=None,
                    help="override sand.pile_radius. To measure a real angle of "
                         "repose the pile must be too TALL to stand: a placed "
                         "cylinder that barely relaxes just reports its own "
                         "softened edge, not the material's repose angle.")
    ap.add_argument("--pile-height", type=float, default=None)
    ap.add_argument("--coup-friction", type=float, nargs="+", default=[None],
                    help="tray coup_friction (MPM-vs-rigid friction, NOT the "
                         "rigid-rigid `friction`). Genesis defaults to 0.1, "
                         "which is far too slippery to hold a repose angle.")
    ap.add_argument("--drops", type=float, nargs="+", default=[0.0, 0.03],
                    help="metres to lower the MPM domain floor below the tray "
                         "floor. 0 = current setup (boundary AT the floor).")
    a = ap.parse_args()

    from Genesis.sand_manipulation import SandManipulation, load_sand_config
    from transforms.sand_occupancy import sand_mass
    B = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}

    print("\nSAND FLOOR PROBE -- does the sand rest on the tray or the boundary?")
    print("tray floor is at z = 10.0 mm; 'rest z' is the 2nd-percentile grain\n")
    print(f"{'drop (mm)':>10s} {'coupF':>6s} {'rest z':>8s} {'to rest':>8s} "
          f"{'resid':>8s} {'repose':>8s} {'height':>8s} {'mass':>7s} {'wall s':>7s}")
    print("-" * 82)

    combos = [(d, cf) for d in a.drops for cf in a.coup_friction]
    for drop, coup_f in combos:
        cfg = load_sand_config(a.config)
        cfg.setdefault("data_collection", {})["sand"] = True
        cfg["simulation"]["settle_steps"] = a.cap
        if coup_f is not None:
            cfg["box"]["coup_friction"] = coup_f
        if a.pile_radius is not None:
            cfg["sand"]["pile_radius"] = a.pile_radius
        if a.pile_height is not None:
            cfg["sand"]["pile_height"] = a.pile_height
        w = float(cfg["box"]["wall_thickness"])
        width, depth, height = cfg["box"]["vol"]
        # Reproduce the default bound arithmetic, then lower only z.
        ps = cfg["mpm_options"]["particle_size"]
        dx = 2.0 * ps
        bpad = 3.0 * dx
        cfg["mpm_options"]["lower_bound"] = [-width / 2 - bpad,
                                             -depth / 2 - bpad,
                                             w / 2 - bpad - drop]
        cfg["mpm_options"]["upper_bound"] = [width / 2 + bpad,
                                            depth / 2 + bpad, height + bpad]

        sim = SandManipulation(config=cfg, n_envs=1, debug=False, viewer_type=None)
        sim.build()
        floor_z = float(sim._wall_thickness) / 2.0
        thr = float(sim._settle_vel_threshold)
        sim.shuffle_particles()

        t0 = time.time()
        reached = None
        for step in range(1, a.cap + 1):
            sim._step_scene()
            if step % 25 == 0:
                v = sim.sand.get_particles_vel().norm(dim=-1).reshape(-1)
                if float(v.quantile(0.995)) < thr:
                    reached = step
                    break
        wall_s = time.time() - t0

        v = sim.sand.get_particles_vel().norm(dim=-1).reshape(-1)
        pos = sim._get_particle_positions()
        rest_z = 1000 * float(pos[..., 2].reshape(-1).quantile(0.02))
        ang, h = repose_angle(pos, floor_z)
        pb = pos[None] if pos.dim() == 2 else pos
        print(f"{1000 * drop:10.0f} "
              f"{('def' if coup_f is None else f'{coup_f:.2f}'):>6s} {rest_z:7.1f}m "
              f"{(str(reached) if reached else 'never'):>8s} "
              f"{1000 * float(v.quantile(0.995)):7.2f}m {ang:7.1f}d {h:7.1f}m "
              f"{float(sand_mass(pb, B).mean()):7.4f} {wall_s:7.1f}", flush=True)
        sim.destroy()

    print("\nrest z ~10 mm => the rigid floor caught it (coupling works, and\n"
          "coup_friction is then the repose knob). rest z ~= 10 - drop => the\n"
          "frictionless domain boundary caught it and fixed geoms do not couple.")


if __name__ == "__main__":
    main()
