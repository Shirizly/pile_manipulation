"""How much does the blade's grip on the sand (plate coup_friction) matter?

The tray's `coup_friction` turned out to decide whether a pile stands up at all
(EXP-0008). The plate carries the same parameter, unset, so it runs at Genesis'
default of 0.1 -- a nearly slippery blade on the one interface the action acts
through. Whether that is actually wrong depends on what it changes, so this
measures a single push at several values instead of assuming a higher number is
better.

Mean displacement alone would not settle it: a slippery blade could move the
same mass a shorter way, or shear a thinner layer off the top. So this also
reports the fraction of grains the push engaged at all, and the change in pile
height -- which is what separates ploughing a heap from skimming its surface.

    python scripts/sand_plate_friction_probe.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sand.yaml")
    ap.add_argument("--values", type=float, nargs="+", default=[0.1, 0.3, 0.6])
    ap.add_argument("--push-length", type=float, default=0.02)
    a = ap.parse_args()

    from Genesis.sand_manipulation import SandManipulation, load_sand_config

    print("\nPLATE coup_friction -- one 20 mm push on a settled pile\n")
    print(f"{'coupF':>6s} {'mean disp':>10s} {'p95 disp':>9s} {'engaged':>9s} "
          f"{'dz top':>8s}")
    print("-" * 48)
    for cf in a.values:
        cfg = load_sand_config(a.config)
        cfg.setdefault("data_collection", {})["sand"] = True
        cfg["plate"]["coup_friction"] = cf

        sim = SandManipulation(config=cfg, n_envs=1, debug=False, viewer_type=None)
        sim.build()
        sim.shuffle_particles()
        sim.update_material_state()

        before = sim._get_particle_positions().clone()
        z_before = 1000 * float(before[..., 2].reshape(-1).quantile(0.98))

        s, e, ang = sim.generate_action_samples(
            1, pile_aware=True, push_length=a.push_length,
            perpendicular_pushes=True, min_swath_particles=3)
        sim.execute_action(s[:, 0, :], e[:, 0, :], ang[:, 0])
        sim.update_material_state()

        after = sim._get_particle_positions()
        d = (after - before)[..., :2].norm(dim=-1).reshape(-1) * 1000
        z_after = 1000 * float(after[..., 2].reshape(-1).quantile(0.98))

        print(f"{cf:6.2f} {float(d.mean()):9.2f}m {float(d.quantile(0.95)):8.2f}m "
              f"{100 * float((d > 1.0).float().mean()):8.1f}% "
              f"{z_after - z_before:7.2f}m", flush=True)
        sim.destroy()

    print("\ndisp in mm; 'engaged' is the fraction of grains that moved more "
          "than 1 mm;\n'dz top' is the change in 98th-percentile height "
          "(negative = the pile got flatter)")


if __name__ == "__main__":
    main()
