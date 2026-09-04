"""Does an MPM sand pile ever reach the rigid path's rest threshold?

The settle criterion (`settle_velocity_threshold = 1e-3` m/s at the q=0.995
quantile) was chosen for rigid cubes -- it "matches Genesis' hibernation
default". Sand does not meet it: at a 100-step cap the q=0.995 grain speed was
7.91 mm/s, and raising the cap 25x to 2500 only brought it to 2.22 mm/s.

Two possibilities, with different consequences:

  decaying   the pile is still relaxing and simply needs a bigger budget, in
             which case settle_steps should go up and the recorded states are
             mid-motion by a knowable amount.
  floored    the residual is a numerical floor MPM will not go below, in which
             case no budget fixes it, the threshold is the wrong test for a
             continuum, and what matters instead is whether the residual is
             small relative to the motion a push causes (~19 mm).

This measures the curve instead of guessing: step a settled pile and record the
speed quantiles as a function of step count.

    python scripts/sand_settle_decay.py --steps 8000
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
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--push-first", action="store_true",
                    help="apply one push before measuring, so the curve is the "
                         "post-PUSH settle rather than the post-spawn one")
    a = ap.parse_args()

    from Genesis.sand_manipulation import SandManipulation, load_sand_config

    cfg = load_sand_config(a.config)
    cfg.setdefault("data_collection", {}).update({"sand": True})
    sim = SandManipulation(config=cfg, n_envs=1, debug=False, viewer_type=None)
    sim.build()
    sim.shuffle_particles()

    if a.push_first:
        sim.update_material_state()
        s, e, ang = sim.generate_action_samples(
            1, pile_aware=True, push_length=0.02, perpendicular_pushes=True,
            min_swath_particles=3)
        sim.execute_action(s[:, 0, :], e[:, 0, :], ang[:, 0])

    thr = 1000 * float(sim._settle_vel_threshold)
    print(f"\nsettle decay: threshold {thr:.2f} mm/s at q=0.995, "
          f"{'post-push' if a.push_first else 'post-spawn'}\n")
    # `moved since 100` is the point of the whole probe: every transition in
    # the collected sand datasets was recorded at step 100, because a duplicate
    # YAML key silently reverted settle_steps from 2500 to the code default.
    # This column is therefore the BIAS in every recorded s' -- how much further
    # the pile would have moved had the settle been allowed to finish. Judge it
    # against the 8-15 mm a push moves the pile.
    print(f"{'step':>7s} {'q0.50':>8s} {'q0.90':>8s} {'q0.995':>8s} {'max':>8s} "
          f"{'drift/step(um)':>14s} {'moved since 100 (mm)':>21s}")
    prev = sim._get_particle_positions().clone()
    at_100 = None
    for i in range(0, a.steps + 1, a.every):
        if i:
            for _ in range(a.every):
                sim._step_scene()
        v = sim.sand.get_particles_vel().norm(dim=-1).reshape(-1)
        pos = sim._get_particle_positions()
        drift = 1e6 * float((pos - prev).norm(dim=-1).mean()) / max(a.every, 1)
        prev = pos.clone()
        if i >= 100 and at_100 is None:
            at_100 = pos.clone()
        since = ("" if at_100 is None else
                 f"{1000 * float((pos - at_100).norm(dim=-1).mean()):21.3f}")
        q = [1000 * float(v.quantile(x)) for x in (0.5, 0.9, 0.995)]
        print(f"{i:7d} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f} "
              f"{1000 * float(v.max()):8.3f} {drift:14.2f} {since}", flush=True)

    print("\nread it as: if q0.995 keeps falling, raise settle_steps; if it "
          "plateaus above the threshold, the threshold is the wrong test for a\n"
          "continuum and the residual should be judged against the ~19 mm a "
          "push moves.", flush=True)
    sim.destroy()


if __name__ == "__main__":
    main()
