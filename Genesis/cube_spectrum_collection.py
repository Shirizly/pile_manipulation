"""
Genesis/cube_spectrum_collection.py — cube-count spectrum for model comparison.

    python -m Genesis.cube_spectrum_collection --counts 20 50 80

Why
---
Sand and cubes gave opposite verdicts on the switched-linear operator. On
identical code and identical action sampling, the linear model BEAT the
mean-delta baseline on sand (+0.106 explained variance) and LOST to it on cubes
(-0.003) — docs/sand_manipulation.md §6. Two very different explanations fit
that equally well:

  * granularity — a few large cubes move as individuals, so a fixed operator
    cannot represent "this cube tipped and that one did not", while a continuum
    averages into something a linear map can capture;
  * pile depth — the cube datasets were monolayers, sand was a heap, so the
    difference is what the field IS rather than what it is made of.

One dataset cannot separate them; a *spectrum* can. This collects the same
amount of data (5200 transitions, matching the first sand set) at n=20, 50 and
80 small cubes, all piled and all pushed the same way, so the model comparison
can be run as a function of particle count with sand as the continuum limit at
the far end.

Design choices that matter for the comparison
---------------------------------------------
Same action sampling as sand: pile-aware touchdown, perpendicular pushes, a
SINGLE fixed push length, five sequential pushes per episode. Everything the
operator fit sees is therefore identical in structure across the spectrum; only
the material changes. Anything else and the comparison measures the pipeline
instead of the physics.

Piled like the sand pile, not dropped. A dropped spawn cannot make a pile deeper
than one layer (measured: 90-94% of cubes in layer 0), so it would confound
granularity with depth — exactly the thing this dataset exists to separate.

The pile is an irregular two-layer HEAP, not a stepped pyramid, and that choice
is a measurement rather than a preference. A pyramid gives the depth but always
the same shape: probed at 3 mm across eight setups spanning lateral jitter, cube
yaw up to 0.35 rad, a full-cube drop and a half-pitch brick bond, the settled
layer occupancy came out EXACTLY equal to the placed layer plan every time
(75/20/5 at n=20, 72/18/8 at n=50) — the stack slid a millimetre or three as a
unit and never restructured. That is the physics of flat-faced rigid cubes, not
a tuning failure: they stack stably in almost any arrangement, and a cube
bridging two supports is as stable as one on a single support. Sand collapses
into an irregular heap by itself; a cube pile does not. So the irregularity is
PLACED — layer 0 takes a random subset of lattice sites and the rest go on top
of randomly chosen occupied ones, redrawn every episode (`heap_positions`).

Start diversity is therefore still free here, unlike sand. MPM samples its pile
once at entity creation, so varied sand starts had to be reconstructed from a
state library; a heap is simply redrawn.

5 mm cubes, not 3 mm, and that is a measurement rather than a preference. 3 mm
simulates fine (at 4000 kg/m3 — 1000 is light enough that the solver warns about
mass ratios), but the shared observation model runs at 2.0 mm per canonical
pixel, where a 3 mm cube covers **~1 pixel**:

    n=20 @ 3 mm    19 active px of 4096   0.95 px/cube
    n=50 @ 3 mm    53                     1.06 px/cube
    n=80 @ 3 mm    88                     1.10 px/cube
    sand          529                     --

A one-pixel object cannot express partial displacement — it sits still and then
jumps a whole pixel — so quantisation would dominate the very margin the
spectrum exists to measure. 3 mm also spans only 15-29 mm against sand's ~40 mm,
which would reintroduce a pile-EXTENT confound alongside granularity, the exact
kind of thing this dataset is built to remove.

5 mm gives 2.5 px/cube, heaps of 25-46 mm comparable to the sand pile, and the
same material as `configs/collection_foresight_single_operator.yaml` (density
1000, friction 0.3) — so the spectrum connects to the existing cube result
rather than sitting beside it. Push length stays at sand's 20 mm, not that
config's 40 mm, because the action sampling is what must match across the
spectrum.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="Genesis/configs/basic.yaml")
    ap.add_argument("--counts", type=int, nargs="+", default=[20, 50, 80],
                    help="cube counts to sweep; one dataset per count")
    ap.add_argument("--size", type=float, default=0.005,
                    help="cube edge, metres. 5 mm, matching the existing cube "
                         "foresight dataset -- see the resolution note in the "
                         "module docstring for why 3 mm is a bad choice here "
                         "even though it simulates fine.")
    ap.add_argument("--density", type=float, default=1000.0,
                    help="matches configs/collection_foresight_single_operator")
    ap.add_argument("--friction", type=float, default=0.3,
                    help="matches configs/collection_foresight_single_operator")
    ap.add_argument("--n-envs", type=int, default=32,
                    help="Measured OOM ceilings on this GPU are 128 envs at "
                         "n=50 and 64 at n=100 (docs/scaling_to_200_objects.md "
                         "3.1), so 64 is safe across this whole spectrum and "
                         "roughly halves wall time -- 32 is the conservative "
                         "default because those ceilings assume nothing else "
                         "is using the GPU.")
    ap.add_argument("--transitions", type=int, default=5200,
                    help="target per count. 5200 matches the first sand "
                         "dataset, so the model comparison is not confounded "
                         "by dataset size.")
    ap.add_argument("--pushes", type=int, default=5,
                    help="sequential pushes per episode; each is one transition")
    ap.add_argument("--push-length", type=float, default=0.02,
                    help="metres, a SINGLE fixed length -- same as sand")
    ap.add_argument("--min-swath-particles", type=int, default=3)
    # Pyramid shaping. Defaults are the setup chosen by
    # scripts/probe_pyramid_setups.py; see docs for the measured rows.
    ap.add_argument("--spawn-mode", default="heap",
                    choices=["heap", "pyramid", "drop"],
                    help="'heap' redraws an irregular two-layer pile every "
                         "episode (the default, and the only one of the three "
                         "that is both deep AND varied); 'pyramid' is the same "
                         "stepped lattice every time; 'drop' cannot exceed one "
                         "layer.")
    ap.add_argument("--heap-base-frac", type=float, default=0.6,
                    help="fraction of cubes in layer 0; 0.6 gives a lumpy "
                         "two-layer heap rather than a paved floor")
    ap.add_argument("--pyramid-gap", type=float, default=1.15)
    ap.add_argument("--pyramid-pos-jitter", type=float, default=0.35)
    ap.add_argument("--pyramid-yaw-jitter", type=float, default=0.25)
    ap.add_argument("--pyramid-lift", type=float, default=0.30)
    ap.add_argument("--pyramid-stagger", type=float, default=0.5,
                    help="offset alternate layers by this many pitches. "
                         "Without it the pyramid is a stack of columns that "
                         "slides but never restructures, so every episode "
                         "starts from the same lattice.")
    ap.add_argument("--output-root", default="data/cube_spectrum")
    ap.add_argument("--debug", action="store_true",
                    help="verbose sim logs. NOTE: the simulator reads debug as "
                         "'show viewer', which renders every step and destroys "
                         "any timing measurement.")
    return ap.parse_args()


def main():
    args = parse_args()
    from .sandbox_manipulation_clean import SandboxManipulation

    episodes = max(1, round(args.transitions / (args.pushes * args.n_envs)))
    total = episodes * args.pushes * args.n_envs
    print(f"cube spectrum: counts {args.counts}, {1000 * args.size:.0f} mm cubes, "
          f"{episodes} episodes x {args.pushes} pushes x {args.n_envs} envs = "
          f"{total} transitions each, push {1000 * args.push_length:.0f} mm",
          flush=True)

    t_all = time.time()
    for n in args.counts:
        cfg = yaml.safe_load(open(args.config))
        cfg["material"].update({"shape": "cube", "particle_size": args.size,
                                "n_particles": n, "density": args.density,
                                "particle_friction": args.friction})
        cfg["box"]["friction"] = args.friction
        # The repo's measured rule (docs/scaling_to_200_objects.md 1.5): the
        # actual requirement is ~0.26 * n_particles, i.e. 5-21 across this
        # spectrum, and OVERSIZING is not free -- the constraint Jacobian is
        # O(max_collision_pairs * contacts_per_pair * n_dofs * n_envs) while
        # raw step time is independent of the cap, so a too-large value
        # converts directly into lost parallelism.
        cfg.setdefault("rigid_options", {})["max_collision_pairs"] = max(150, n // 2)
        cfg["spawn"] = {"mode": args.spawn_mode,
                        "heap_base_frac": args.heap_base_frac,
                        "pyramid_gap": args.pyramid_gap,
                        "pyramid_pos_jitter": args.pyramid_pos_jitter,
                        "pyramid_yaw_jitter": args.pyramid_yaw_jitter,
                        "pyramid_lift": args.pyramid_lift,
                        "pyramid_stagger": args.pyramid_stagger}
        cfg.setdefault("data_collection", {}).update({
            "cube_spectrum": True, "n_cubes": n, "cube_size": args.size,
            "pushes_per_episode": args.pushes, "push_length": args.push_length,
            "pile_aware": True, "perpendicular_pushes": True,
            "min_swath_particles": args.min_swath_particles,
            "spawn": dict(cfg["spawn"]),
        })

        out_path = f"{args.output_root}/n{n}"
        print(f"\n=== n={n} -> Genesis/{out_path} ===", flush=True)
        t0 = time.time()
        sim = SandboxManipulation(config=cfg, n_envs=args.n_envs,
                                  debug=args.debug, viewer_type=None)
        sim.build()
        sim.set_material_properties({"particle_friction": args.friction,
                                     "particle_density": args.density,
                                     "box_friction": args.friction,
                                     "sampled_particle_friction": None,
                                     "sampled_particle_density": None})
        print(f"  build: {time.time() - t0:.0f}s", flush=True)

        for ep in range(episodes):
            t = time.time()
            # Fresh pyramid, fresh jitter, fresh collapse -- this is where the
            # start diversity comes from, so it must happen every episode.
            sim.shuffle_particles()
            sim.update_material_state()
            try:
                sim.collect_data_samples(
                    n_samples=args.pushes,
                    path=out_path,
                    placement_aware=False,
                    shared_travel_distance=False,
                    perpendicular_pushes=True,
                    push_length=args.push_length,
                    pile_aware=True,
                    min_swath_particles=args.min_swath_particles,
                )
            except RuntimeError as exc:
                print(f"  episode {ep + 1} failed: {exc}", flush=True)
                continue
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  episode {ep + 1}/{episodes} "
                      f"({time.time() - t:.0f}s)", flush=True)
        print(f"  n={n} done in {time.time() - t0:.0f}s", flush=True)
        sim.destroy()

    print(f"\nall counts done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
