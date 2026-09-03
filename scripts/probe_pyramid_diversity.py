"""Do successive pyramid episodes actually start from DIFFERENT piles?

`probe_pyramid_setups.py` reports the mean shape of a settled pyramid, and at
n=20 every setup returned the same layer occupancy (75/20/5, i.e. the placed
plan [15,4,1]) -- the stack shifts sideways under perturbation but does not
lose its layering. A mean shape that is stable across setups says nothing about
whether individual episodes differ, and that is the property the dataset needs:
the first sand dataset restarted from an identical pile every time, which
capped its state diversity at ~14 dimensions and made the low-rank result
partly a statement about the data (docs/sand_manipulation.md §7).

So this measures spread rather than shape, over many respawns:

    centroid spread   std of the pile's xy centre across episodes. The sand
                      state-library rebuild moved this from 0.00 to 15.82 mm,
                      which is the yardstick.
    pairwise distance mean per-cube distance between two independent respawns,
                      in cube widths. 0 = identical start every episode.
    height-field std  std of the occupancy column-count map across episodes,
                      i.e. how much the SHAPE moves, not just its position.

    python scripts/probe_pyramid_diversity.py --n 50 --episodes 12
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--size", type=float, default=0.003)
    ap.add_argument("--density", type=float, default=4000.0)
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--gap", type=float, default=1.15)
    ap.add_argument("--pos-jitter", type=float, default=0.35)
    ap.add_argument("--yaw-jitter", type=float, default=0.25)
    ap.add_argument("--lift", type=float, default=0.30)
    ap.add_argument("--stagger", type=float, default=0.5)
    ap.add_argument("--layout", default="heap", choices=["pyramid", "heap"])
    ap.add_argument("--heap-base-frac", type=float, default=0.6)
    args = ap.parse_args()

    import yaml
    from Genesis.sandbox_manipulation_clean import SandboxManipulation

    cfg = yaml.safe_load(open("Genesis/configs/basic.yaml"))
    cfg["material"].update({"shape": "cube", "particle_size": args.size,
                            "n_particles": args.n, "density": args.density,
                            "particle_friction": args.friction})
    cfg["box"]["friction"] = args.friction
    cfg.setdefault("rigid_options", {})["max_collision_pairs"] = max(250, 12 * args.n)
    cfg["spawn"] = {"mode": args.layout, "pyramid_gap": args.gap,
                    "heap_base_frac": args.heap_base_frac,
                    "pyramid_pos_jitter": args.pos_jitter,
                    "pyramid_yaw_jitter": args.yaw_jitter,
                    "pyramid_lift": args.lift,
                    "pyramid_stagger": args.stagger}

    sim = SandboxManipulation(config=cfg, n_envs=args.envs, debug=False,
                              viewer_type=None)
    sim.build()
    sim.set_material_properties({"particle_friction": args.friction,
                                 "particle_density": args.density,
                                 "box_friction": args.friction,
                                 "sampled_particle_friction": None,
                                 "sampled_particle_density": None})

    print(f"\nPYRAMID DIVERSITY  n={args.n}  size={1000*args.size:.0f}mm  "
          f"gap={args.gap} jitter={args.pos_jitter} yaw={args.yaw_jitter} "
          f"lift={args.lift} stagger={args.stagger}\n{args.episodes} episodes x {args.envs} envs\n",
          flush=True)

    starts, fields = [], []
    for ep in range(args.episodes):
        sim.shuffle_particles()
        sim.update_material_state()
        st = sim._particle_state[..., :3].clone()
        starts.append(st)
        # Column-count map, one cube wide, on a fixed grid so fields from
        # different episodes are comparable cell by cell.
        g = 32
        cell = ((st[..., :2] + 0.048) / (0.096 / g)).long().clamp(0, g - 1)
        f = torch.zeros((st.shape[0], g, g), device=st.device)
        for e in range(st.shape[0]):
            f[e].index_put_((cell[e, :, 0], cell[e, :, 1]),
                            torch.ones(st.shape[1], device=st.device),
                            accumulate=True)
        fields.append(f)

    S = torch.cat(starts)                              # (E*envs, n, 3)
    F = torch.cat(fields)
    cen = S[..., :2].mean(dim=1)
    spread = 1000 * float(cen.std(dim=0).mean())
    # Pairwise distance between independent starts, in cube widths. Cube
    # identity is meaningless across respawns (cube 3 is not "the same" cube),
    # so compare the SORTED position sets, not index by index.
    a, b = S[:len(S) // 2], S[len(S) // 2: 2 * (len(S) // 2)]
    sa = a[..., :2].flatten(1).sort(dim=1).values
    sb = b[..., :2].flatten(1).sort(dim=1).values
    pair = float((sa - sb).abs().mean()) / args.size
    fstd = float(F.std(dim=0).mean())

    print(f"  centroid spread    {spread:6.2f} mm   "
          f"(sand: 0.00 single-pile -> 15.82 with the state library)")
    print(f"  pairwise distance  {pair:6.2f} cube widths  "
          f"(0 = identical start every episode)")
    print(f"  height-field std   {fstd:6.3f} cubes/cell")
    z = S[..., 2]
    z0 = float(z.reshape(-1).quantile(0.02))
    lay = ((z - z0) / args.size).round().clamp(min=0)
    occ = {L: float((lay == L).float().mean()) for L in range(5)}
    print("  layers             " +
          " ".join(f"L{L}={100*v:.0f}%" for L, v in occ.items() if v > 0.02))
    per_ep = [1000 * float(s[..., :2].mean(dim=1).std(dim=0).mean())
              for s in starts]
    print(f"  within-episode env spread {sum(per_ep)/len(per_ep):.2f} mm "
          f"(envs differ from each other too)")
    sim.destroy()


if __name__ == "__main__":
    main()
