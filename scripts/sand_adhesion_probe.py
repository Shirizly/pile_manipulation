"""Two direct tests of "sand should have near-zero adhesion and high friction".

The bulk drag average (scripts/sand_drag_probe.py) came out near zero, but it
averages over ALL material behind the blade, most of which is metres of pile
away and never moving. What is visible on video is the near wake -- the band
immediately behind the blade. So this measures that band specifically, and adds
the cleanest adhesion test there is:

  near wake   grains 4-12 mm behind the blade's start plane, displacement
              projected on the push direction. A barrier leaves them; adhesion
              tows them.

  lift test   lower the blade INTO the settled pile and take it straight back
              out with no sweep at all (a zero-length push does exactly
              lower-then-lift). Sand with no adhesion stays put. Anything that
              rises with the blade is the artifact, and it is measured as the
              mean z gain of the grains nearest the blade plus how many rise by
              more than a grain diameter.

Both are reported with CPIC off and on, because the grid transfer crossing a
thin body is the mechanism that would produce either.

    python scripts/sand_adhesion_probe.py
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
    ap.add_argument("--push-length", type=float, default=0.02)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--cpic", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()

    from Genesis.sand_manipulation import SandManipulation, load_sand_config

    print("\nADHESION / NEAR-WAKE PROBE")
    print("near wake: grains 4-12 mm BEHIND the blade, mm along the push")
    print("lift test: zero-length push (lower then lift), mm of z the sand "
          "gains\n")
    print(f"{'CPIC':>5s} {'front':>8s} {'near wake':>10s} {'wake/front':>11s} "
          f"{'lift dz':>8s} {'lifted>2mm':>11s}")
    print("-" * 60)

    for cp in a.cpic:
        cfg = load_sand_config(a.config)
        cfg.setdefault("data_collection", {})["sand"] = True
        cfg["mpm_options"]["enable_CPIC"] = bool(cp)
        ps = float(cfg["mpm_options"]["particle_size"])

        sim = SandManipulation(config=cfg, n_envs=1, debug=False, viewer_type=None)
        sim.build()
        sim.shuffle_particles()
        sim.update_material_state()

        fronts, wakes, lifts, lifted = [], [], [], []
        for t in range(a.trials):
            s, e, ang = sim.generate_action_samples(
                1, pile_aware=True, push_length=a.push_length,
                perpendicular_pushes=True, min_swath_particles=3)
            # Plough out from the middle so there IS material behind the blade.
            cur = sim._get_particle_positions().reshape(-1, 3)
            shift = cur[:, :2].mean(0) - s[0, 0, :2]
            s = s.clone(); e = e.clone()
            s[0, 0, :2] += shift
            e[0, 0, :2] += shift
            p0 = s[0, 0, :2]
            n = (e[0, 0, :2] - p0)
            n = n / (n.norm() + 1e-9)

            # ---- lift test: same touchdown, NO sweep -----------------------
            before = sim._get_particle_positions().clone().reshape(-1, 3)
            near = ((before[:, :2] - p0.unsqueeze(0)) @ n).abs() < 0.006
            sim.execute_action(s[:, 0, :], s[:, 0, :], ang[:, 0])   # p_stop = p_start
            after_lift = sim._get_particle_positions().reshape(-1, 3)
            dz = 1000.0 * (after_lift[:, 2] - before[:, 2])
            if int(near.sum()):
                lifts.append(float(dz[near].mean()))
                lifted.append(100.0 * float((dz > 2.0).float().mean()))
            sim.update_material_state()

            # ---- near wake: a real sweep ------------------------------------
            before = sim._get_particle_positions().clone().reshape(-1, 3)
            proj0 = (before[:, :2] - p0.unsqueeze(0)) @ n
            sim.execute_action(s[:, 0, :], e[:, 0, :], ang[:, 0])
            sim.update_material_state()
            after = sim._get_particle_positions().reshape(-1, 3)
            moved = ((after[:, :2] - before[:, :2]) @ n) * 1000.0
            wake_m = (proj0 < -0.004) & (proj0 > -0.012)
            front_m = proj0 > 0.0
            if int(wake_m.sum()) and int(front_m.sum()):
                fronts.append(float(moved[front_m].mean()))
                wakes.append(float(moved[wake_m].mean()))

        f = sum(fronts) / max(len(fronts), 1)
        w = sum(wakes) / max(len(wakes), 1)
        lz = sum(lifts) / max(len(lifts), 1)
        lp = sum(lifted) / max(len(lifted), 1)
        print(f"{('on' if cp else 'off'):>5s} {f:7.2f}m {w:9.2f}m "
              f"{(w / f if abs(f) > 1e-6 else float('nan')):11.3f} "
              f"{lz:7.2f}m {lp:10.1f}%", flush=True)
        sim.destroy()

    print(f"\ngrain diameter is {1000 * 0.002:.0f} mm; 'lifted>2mm' is the "
          "percentage of ALL grains raised by\nmore than one diameter during a "
          "pure lower-and-lift. For cohesionless sand both\nthe wake ratio and "
          "the lift should be ~0.")


if __name__ == "__main__":
    main()
