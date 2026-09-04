"""Does material BEHIND the blade get dragged along with the sweep?

Observed on video: sand behind the plate follows it, which reads as adhesion.
The sand model has none. Genesis implements the Klar et al. Drucker-Prager
return mapping, and its tensile branch (`tr >= 0`) leaves S at the identity,
releasing all elastic deformation -- constitutive tensile strength is exactly
zero. So any pulling is numerical, and there is one obvious mechanism: the MPM
grid transfer does not know the blade exists. Particles on opposite sides of a
thin body share grid nodes, so momentum crosses it. CPIC (Compatible
Particle-In-Cell) exists precisely to introduce that displacement discontinuity;
Genesis ships it as `MPMOptions.enable_CPIC`, off by default.

The measurement has to separate dragging from ordinary pushing, so grains are
classified by which side of the blade they start on:

    front   ahead of the blade along the push direction -- SHOULD move
    behind  behind the blade's start plane -- should NOT move; whatever it does
            is the artifact

and the reported number is displacement projected onto the push direction, so
material merely slumping sideways does not count as drag. `behind/front` is the
headline: 0 means the blade is a clean barrier, and large means the sweep is
towing a wake.

    python scripts/sand_drag_probe.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# (label, enable_CPIC, particle_size, grid dx multiple of particle_size)
ARMS = [
    ("CPIC off  dx=2ps", False, 0.002, 2.0),
    ("CPIC ON   dx=2ps", True, 0.002, 2.0),
    ("CPIC ON   dx=1ps", True, 0.002, 1.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sand.yaml")
    ap.add_argument("--push-length", type=float, default=0.02)
    ap.add_argument("--pushes", type=int, default=3)
    a = ap.parse_args()

    from Genesis.sand_manipulation import SandManipulation, load_sand_config

    print("\nDRAG PROBE -- is the blade a barrier, or does it tow a wake?")
    print("displacement is projected on the push direction; behind/front is the "
          "artifact ratio\n")
    print(f"{'arm':18s} {'front':>8s} {'behind':>8s} {'ratio':>7s} "
          f"{'up-drag':>8s} {'cells':>13s} {'s/push':>7s}")
    print("-" * 76)

    for label, cpic, ps, dxmul in ARMS:
        cfg = load_sand_config(a.config)
        cfg.setdefault("data_collection", {})["sand"] = True
        cfg["mpm_options"]["particle_size"] = ps
        cfg["mpm_options"]["enable_CPIC"] = cpic
        # grid_density is cells per METRE; dx = dxmul * particle_size.
        cfg["mpm_options"]["grid_density"] = int(round(1.0 / (dxmul * ps)))

        sim = SandManipulation(config=cfg, n_envs=1, debug=False, viewer_type=None)
        sim.build()
        sim.shuffle_particles()
        sim.update_material_state()

        fronts, behinds, ups = [], [], []
        t0 = time.time()
        for _ in range(a.pushes):
            s, e, ang = sim.generate_action_samples(
                1, pile_aware=True, push_length=a.push_length,
                perpendicular_pushes=True, min_swath_particles=3)
            # Re-anchor the sweep to START AT THE PILE CENTROID, keeping the
            # sampler's direction, length, height and yaw. The pile-aware start
            # sits at the pile's EDGE by construction, so nothing is behind the
            # blade and the drag question cannot be asked at all -- the first
            # version of this probe reported all zeros for exactly that reason.
            # Ploughing out from the middle is also what the video shows.
            cur = sim._get_particle_positions().reshape(-1, 3)
            centroid = cur[:, :2].mean(0)
            shift = centroid - s[0, 0, :2]
            s = s.clone(); e = e.clone()
            s[0, 0, :2] += shift
            e[0, 0, :2] += shift
            p0, p1 = s[0, 0, :2], e[0, 0, :2]
            d = p1 - p0
            n = d / (d.norm() + 1e-9)               # push direction, unit

            before = sim._get_particle_positions().clone().reshape(-1, 3)
            # Signed distance along the push direction from the blade's start
            # plane. Negative = behind the blade at the moment of the sweep.
            proj0 = (before[:, :2] - p0.unsqueeze(0)) @ n
            sim.execute_action(s[:, 0, :], e[:, 0, :], ang[:, 0])
            sim.update_material_state()
            after = sim._get_particle_positions().reshape(-1, 3)

            moved = ((after[:, :2] - before[:, :2]) @ n) * 1000.0   # mm along push
            behind_m = proj0 < -0.004      # a blade-thickness clear of the plane
            front_m = proj0 > 0.0
            if int(behind_m.sum()) and int(front_m.sum()):
                fronts.append(float(moved[front_m].mean()))
                behinds.append(float(moved[behind_m].mean()))
                ups.append(1000.0 * float(
                    (after[behind_m, 2] - before[behind_m, 2]).mean()))
        dt = (time.time() - t0) / max(a.pushes, 1)

        f = sum(fronts) / max(len(fronts), 1)
        b = sum(behinds) / max(len(behinds), 1)
        u = sum(ups) / max(len(ups), 1)
        cells = sim._scene.sim.mpm_solver.grid_res if hasattr(
            sim._scene.sim, "mpm_solver") else None
        print(f"{label:18s} {f:7.2f}m {b:7.2f}m "
              f"{(b / f if abs(f) > 1e-6 else float('nan')):7.2f} "
              f"{u:7.2f}m {str(cells):>13s} {dt:7.1f}", flush=True)
        sim.destroy()

    print("\nfront/behind in mm along the push direction, averaged over "
          f"{a.pushes} pushes.\nratio 0 = clean barrier; the sand model's own "
          "tensile strength is exactly zero,\nso anything above 0 is the grid "
          "transfer crossing the blade.")


if __name__ == "__main__":
    main()
