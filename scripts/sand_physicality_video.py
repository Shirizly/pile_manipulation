"""Is the sand simulation physical? Record it and measure it, per push.

Why both
--------
The sand setup has already produced data that looked fine and was not: the
post-push settle silently never ran (the pile was recorded mid-motion), and
before the MPM domain was resized to coincide with the tray, grains fell
straight through a floor at z=+10 mm and mass dropped to 0.65
(docs/sand_manipulation.md section 3). Neither was visible in the fitted
numbers. One was obvious on sight; the other was only obvious as a number.

So this records a video AND asserts the quantities a video cannot show:

  mass in tray      grains inside the tray bounds / total. Must stay 1.0000.
                    A drop means grains are leaving through a wall or floor.
  z floor           min grain z against the tray floor. Below it = leaking.
  rest speed        the q=0.995 grain speed AFTER the settle. If this is not
                    near zero the recorded state is mid-motion, and because
                    each transition's s is the previous s', that propagates.
  displacement      mean grain motion per push, against the 20 mm push. Zero
                    means the blade is missing the pile; huge means it is
                    launching grains.
  pile extent/height  the pile should spread and flatten gradually, not jump.

Two cameras side by side: an oblique view, which is where flow, blade contact
and any explosion are visible, and the overhead view, which is what the model
actually consumes.

    python scripts/sand_physicality_video.py --episodes 3 --pushes 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sand.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--pushes", type=int, default=5)
    ap.add_argument("--push-length", type=float, default=0.02)
    ap.add_argument("--out", default="outputs/sand_physicality")
    ap.add_argument("--res", type=int, default=480, help="per-camera square px")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame-every", type=int, default=2,
                    help="record every Nth step during lower/sweep/lift")
    ap.add_argument("--settle-frame-every", type=int, default=6,
                    help="every Nth settle step; the settle is long and slow")
    ap.add_argument("--state-library", default=None, metavar="GLOB",
                    help="draw a different start pile per episode from a "
                         "previous collection. Without it every episode starts "
                         "from the identical as-sampled pile, which is fine for "
                         "judging physics and poor for judging variety.")
    return ap.parse_args()


def main():
    args = parse_args()
    import cv2
    import genesis as gs

    from Genesis.sand_manipulation import SandManipulation, load_sand_config
    from transforms.sand_occupancy import sand_mass
    from utils import write_video_frame

    BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = load_sand_config(args.config)
    cfg.setdefault("data_collection", {}).update({
        "sand": True, "pushes_per_episode": args.pushes,
        "push_length": args.push_length, "pile_aware": True,
        "perpendicular_pushes": True, "min_swath_particles": 3,
        "mpm_particle_size": cfg.get("mpm_options", {}).get("particle_size"),
        "sand_material": cfg.get("sand", {}),
    })

    # One env: this is an inspection run, and a batch would only render env 0
    # anyway while making every step cost the batch's worth of solve.
    sim = SandManipulation(config=cfg, n_envs=1, debug=False, viewer_type=None)

    floor_z = float(sim._wall_thickness) / 2.0
    # Cameras must be added BEFORE build(). Oblique first: it is the one that
    # shows whether the material behaves like sand -- angle of repose, flow
    # around the blade, grains launching -- none of which reads from directly
    # above. The overhead camera is the model's-eye view, kept alongside so the
    # two can be compared frame by frame.
    cam_obl = sim._scene.add_camera(res=(args.res, args.res),
                                    pos=(0.17, -0.17, 0.15),
                                    lookat=(0.0, 0.0, floor_z + 0.005),
                                    up=(0.0, 0.0, 1.0), fov=40.0,
                                    GUI=False, env_idx=0)
    cam_top = sim._scene.add_camera(res=(args.res, args.res),
                                    pos=(0.0, 0.0, 0.30),
                                    lookat=(0.0, 0.0, floor_z),
                                    up=(0.0, 1.0, 0.0), fov=45.0,
                                    GUI=False, env_idx=0)
    sim.build()

    lib = None
    if args.state_library:
        from Genesis.sand_state_library import build_sand_state_library
        lib = build_sand_state_library(
            args.state_library, box_vol=cfg["box"]["vol"],
            n_states=max(8, args.episodes), skip_first_push=8,
            particle_size=cfg.get("mpm_options", {}).get("particle_size", 0.002),
            device="cpu")
        sim.set_state_library(lib)

    def frame():
        """One side-by-side BGR frame: oblique | overhead."""
        rgb_o, _, _, _ = cam_obl.render(rgb=True)
        rgb_t, _, _, _ = cam_top.render(rgb=True)
        pair = np.concatenate([np.asarray(rgb_o, dtype=np.uint8)[..., :3],
                               np.asarray(rgb_t, dtype=np.uint8)[..., :3]], axis=1)
        return (pair.astype(np.float32) / 255.0)

    def grain_speed_q(q=0.995):
        v = sim.sand.get_particles_vel()
        return float(v.norm(dim=-1).reshape(-1).quantile(q))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    report = []

    for ep in range(args.episodes):
        path = out / f"sand_episode{ep + 1}.mp4"
        writer = cv2.VideoWriter(str(path), fourcc, args.fps,
                                 (2 * args.res, args.res))
        sim.shuffle_particles()
        sim.update_material_state(
            on_step=lambda i: (i % args.settle_frame_every == 0)
            and write_video_frame(frame(), writer))

        s_prev = sim._get_particle_positions().clone()
        m0 = float(sand_mass(s_prev[None] if s_prev.dim() == 2 else s_prev,
                             BOUNDS).mean())
        print(f"\n=== episode {ep + 1}/{args.episodes} -> {path.name} ===",
              flush=True)
        print(f"    start: mass {m0:.4f}, rest speed "
              f"{1000 * grain_speed_q():.2f} mm/s", flush=True)

        for k in range(args.pushes):
            starts, stops, angles = sim.generate_action_samples(
                1, pile_aware=True, push_length=args.push_length,
                perpendicular_pushes=True, min_swath_particles=3,
                pile_clearance=cfg.get("mpm_options", {}).get("particle_size", 0.002))
            sim.execute_action(
                starts[:, 0, :], stops[:, 0, :], angles[:, 0],
                on_step=lambda phase, i: (i % args.frame_every == 0)
                and write_video_frame(frame(), writer))
            sim.update_material_state(
                on_step=lambda i: (i % args.settle_frame_every == 0)
                and write_video_frame(frame(), writer))

            s = sim._get_particle_positions()
            sb = s[None] if s.dim() == 2 else s
            mass = float(sand_mass(sb, BOUNDS).mean())
            disp = 1000.0 * float((s - s_prev)[..., :2].norm(dim=-1).mean())
            zmin, zmax = 1000 * float(s[..., 2].min()), 1000 * float(s[..., 2].max())
            xy = s[..., :2].reshape(-1, 2)
            ext = 1000 * float((xy.quantile(0.95, 0) - xy.quantile(0.05, 0)).max())
            rest = 1000 * grain_speed_q()
            push_mm = 1000 * float((stops[0, 0, :2] - starts[0, 0, :2]).norm())
            print(f"    push {k + 1}: len {push_mm:5.1f} mm | mass {mass:.4f} | "
                  f"z {zmin:5.1f}..{zmax:5.1f} mm | extent {ext:5.1f} mm | "
                  f"disp {disp:5.2f} mm | rest {rest:6.2f} mm/s", flush=True)
            report.append(dict(ep=ep + 1, push=k + 1, mass=mass, zmin=zmin,
                               zmax=zmax, extent=ext, disp=disp, rest=rest,
                               push_mm=push_mm))
            s_prev = s.clone()

        writer.release()
        print(f"    wrote {path}", flush=True)

    # ---- verdict ----------------------------------------------------------
    print("\n=== physicality summary ===", flush=True)
    mass = [r["mass"] for r in report]
    zmin = [r["zmin"] for r in report]
    rest = [r["rest"] for r in report]
    disp = [r["disp"] for r in report]
    checks = [
        ("mass conserved", min(mass) >= 0.999,
         f"min {min(mass):.4f} over {len(mass)} pushes (want 1.0000)"),
        ("stays above the floor", min(zmin) >= 1000 * floor_z - 0.5,
         f"min grain z {min(zmin):.1f} mm (floor {1000 * floor_z:.1f})"),
        ("settles to rest", max(rest) < 5.0,
         f"worst post-settle q=0.995 speed {max(rest):.2f} mm/s (threshold 5)"),
        # Condition on the push actually having length. The pile-aware sampler
        # can emit a ZERO-length push when it cannot find room -- observed once
        # in 15 -- and a 0 mm push correctly moves nothing, so scoring it as a
        # blade failure is a bug in the check, not a finding. Collections drop
        # these via --min-push-mm, so they never reach a fit.
        ("blade moves sand (full-length pushes)",
         (min([r["disp"] for r in report if r["push_mm"] >= 19.9] or [0]) > 0.2),
         f"smallest mean displacement {min([r['disp'] for r in report if r['push_mm'] >= 19.9] or [0]):.2f} mm "
         f"over {sum(1 for r in report if r['push_mm'] >= 19.9)}/{len(report)} "
         f"full-length pushes"),
        ("no launching", max(r["zmax"] for r in report) < 1000 * floor_z + 40,
         f"highest grain {max(r['zmax'] for r in report):.1f} mm"),
    ]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} {detail}", flush=True)
    print(f"\nvideos in {out}/  (left: oblique, right: overhead/model view)",
          flush=True)
    sim.destroy()
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
