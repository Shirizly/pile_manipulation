"""Find a pyramid spawn that gives a natural two-layer heap, at n=20/50/80.

Why this probe exists
---------------------
The pyramid spawn was added because a *dropped* pile cannot be more than one
layer deep (measured: 90-94% of cubes in layer 0). But the pyramid it produces
is CRYSTALLINE: as-placed and after-settling came out byte-identical, so every
episode starts from the same stepped lattice. For a foresight dataset that is
the same defect the first sand dataset had -- a start distribution with no
variety -- and the fix is the same in spirit: perturb the start until it
collapses into something irregular, while keeping the depth that only placement
can give.

So this sweeps the three shaping knobs against what we actually want:

    ~2 layers        mean layer index around 0.5-1.0 (0 = flat monolayer,
                     1.0 = two full layers), with real L1 occupancy
    not too uniform  column heights should VARY (high CV), and the settled
                     state should differ from the placed one (collapse > 0)
    still a pile     footprint must stay compact -- a setup that scatters the
                     cubes across the tray has just reinvented the drop

Run before collecting; the winning row goes into the collection config.

    python scripts/probe_pyramid_setups.py --size 0.003
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# (gap, pos_jitter, yaw_jitter, lift, stagger). pos_jitter/lift in cube widths,
# yaw in radians, stagger in pitches. Row A is the crystalline default, kept as
# the control.
#
# A-E vary the perturbation of a COLUMN-stacked pyramid and all came back with
# the settled layer occupancy exactly equal to the placed plan -- the stack
# slides but never restructures, because each upper cube rests squarely on one
# lower cube. F-H stagger alternate layers by half a pitch so upper cubes bridge
# a seam and have somewhere to fall.
SETUPS = {
    "A crystalline":  (1.15, 0.08, 0.00, 0.00, 0.0),
    "B mild":         (1.10, 0.25, 0.15, 0.00, 0.0),
    "C tilt+drop":    (1.15, 0.35, 0.25, 0.30, 0.0),
    "D loose":        (1.20, 0.50, 0.35, 0.60, 0.0),
    "E full drop":    (1.15, 0.30, 0.20, 1.00, 0.0),
    "F stagger":      (1.15, 0.15, 0.15, 0.10, 0.5),
    "G stagger+drop": (1.15, 0.30, 0.25, 0.40, 0.5),
    "H stagger loose":(1.20, 0.40, 0.30, 0.60, 0.5),
}


def report(state, size, label, ref=None):
    """Layer structure, compactness and irregularity of one settled pile."""
    z = state[..., 2]
    z0 = float(z.reshape(-1).quantile(0.02))
    lay = ((z - z0) / size).round().clamp(min=0)
    occ = {L: float((lay == L).float().mean()) for L in range(6)}
    occ = {L: v for L, v in occ.items() if v > 0.02}
    mean_layer = sum(L * v for L, v in occ.items())

    xy = state[..., :2]
    foot = 1000 * float(xy.reshape(-1, 2).max(0).values
                        .sub(xy.reshape(-1, 2).min(0).values).max())

    # Irregularity: bin cubes into columns one cube wide and look at how uneven
    # the column heights are. A perfect stepped pyramid is smooth and gives a
    # LOW cv; a collapsed heap has some columns two deep next to empty floor.
    cv = []
    for e in range(state.shape[0]):
        cell = (xy[e] / size).round().long()
        key = cell[:, 0] * 1000 + cell[:, 1]
        cnt = torch.bincount(key - key.min()).float()
        cnt = cnt[cnt > 0]
        cv.append(float(cnt.std() / cnt.mean()) if cnt.numel() > 1 else 0.0)
    cv = sum(cv) / len(cv)

    coll = ("" if ref is None else
            f"  collapse {1000 * float((state - ref).norm(dim=-1).mean()):4.1f} mm")
    print(f"    {label:18s} mean layer {mean_layer:4.2f}  footprint {foot:5.1f} mm  "
          f"col-CV {cv:4.2f}{coll}  " +
          " ".join(f"L{L}={100 * v:.0f}%" for L, v in sorted(occ.items())),
          flush=True)
    return mean_layer, cv, foot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[20, 50, 80])
    ap.add_argument("--size", type=float, default=0.003)
    ap.add_argument("--density", type=float, default=4000.0,
                    help="3 mm cubes at 1000 kg/m3 are light enough that the "
                         "solver warns about mass ratios; 4000 is what the "
                         "dense-pile probe settled on.")
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--push", action="store_true",
                    help="also apply one push, to check the structure survives")
    ap.add_argument("--setups", nargs="+", default=None)
    ap.add_argument("--layout", default="pyramid", choices=["pyramid", "heap"],
                    help="'heap' redraws irregular two-layer sites every "
                         "respawn; 'pyramid' is the stepped lattice")
    ap.add_argument("--heap-base-frac", type=float, default=0.6)
    args = ap.parse_args()

    import yaml
    from Genesis.sandbox_manipulation_clean import SandboxManipulation

    setups = {k: v for k, v in SETUPS.items()
              if args.setups is None or k.split()[0] in args.setups}

    print(f"PYRAMID SETUP PROBE  size={1000 * args.size:.0f}mm  "
          f"density={args.density}  friction={args.friction}\n"
          "want: mean layer ~0.5-1.0 (two layers), HIGH col-CV (irregular), "
          "compact footprint, collapse > 0\n", flush=True)

    best = {}
    for n in args.counts:
        print(f"=== n={n} ===", flush=True)
        for name, (gap, jit, yaw, lift, stag) in setups.items():
            cfg = yaml.safe_load(open("Genesis/configs/basic.yaml"))
            cfg["material"].update({"shape": "cube", "particle_size": args.size,
                                    "n_particles": n, "density": args.density,
                                    "particle_friction": args.friction})
            cfg["box"]["friction"] = args.friction
            # The repo's measured rule; oversizing costs parallelism, see
            # docs/scaling_to_200_objects.md 1.5.
            cfg.setdefault("rigid_options", {})["max_collision_pairs"] = \
                max(150, n // 2)
            cfg["spawn"] = {"mode": args.layout, "pyramid_gap": gap,
                            "heap_base_frac": args.heap_base_frac,
                            "pyramid_pos_jitter": jit, "pyramid_yaw_jitter": yaw,
                            "pyramid_lift": lift, "pyramid_stagger": stag}

            sim = SandboxManipulation(config=cfg, n_envs=args.envs, debug=False,
                                      viewer_type=None)
            sim.build()
            sim.set_material_properties({"particle_friction": args.friction,
                                         "particle_density": args.density,
                                         "box_friction": args.friction,
                                         "sampled_particle_friction": None,
                                         "sampled_particle_density": None})
            t = time.time()
            sim.shuffle_particles()
            # Read the placed poses from the SIMULATOR, not from
            # `_particle_state`: `_spawn_pyramid` writes poses into the scene
            # but leaves the cached state untouched (only
            # `update_material_state` refreshes it), so the cache still holds
            # the previous episode's settled pile and the collapse figure came
            # out nonsense -- 13 mm of "collapse" in a 10 mm footprint.
            placed = sim._get_particle_positions()[..., :3].clone()
            sim.update_material_state()
            m, cv, foot = report(sim._particle_state[..., :3], args.size,
                                 name, ref=placed)
            if args.push:
                try:
                    s, e, a = sim.generate_action_samples(
                        1, pile_aware=True, push_length=0.02,
                        min_swath_particles=3)
                    sim.execute_action(s[:, 0, :], e[:, 0, :], a[:, 0])
                    sim.update_material_state()
                    report(sim._particle_state[..., :3], args.size, "  + 1 push")
                except Exception as exc:                       # noqa: BLE001
                    print(f"      push failed: {exc}", flush=True)
            best[(n, name)] = (m, cv, foot, time.time() - t)
            sim.destroy()
        print(flush=True)

    print("summary (mean layer / col-CV / footprint mm):", flush=True)
    for name in setups:
        row = " | ".join(f"n={n}: {best[(n, name)][0]:.2f}/{best[(n, name)][1]:.2f}/"
                         f"{best[(n, name)][2]:.0f}" for n in args.counts)
        print(f"  {name:18s} {row}", flush=True)


if __name__ == "__main__":
    main()
