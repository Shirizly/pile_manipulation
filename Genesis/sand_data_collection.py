"""
Genesis/sand_data_collection.py — collect sand push transitions.

    python -m Genesis.sand_data_collection --episodes 40 --n-envs 8

Deliberately a thin driver. The episode loop, the buffers, the on-disk schema
and the pile-aware action sampling all come from `SandboxManipulation`
unchanged — only the material differs (see `Genesis/sand_manipulation.py`), so
the sand dataset lands in the *same* format as every cube dataset and every
existing loader and analysis script reads it without modification. That is worth
more than a bespoke format: the whole point is to compare sand against the cube
results, and a shared schema makes that a config change rather than a port.

What the run produces
---------------------
One episode = the pile restored to its as-sampled state, then ``--pushes``
sequential pile-aware pushes of a single fixed length. Each push is one
transition ``(s, u, s')``, with ``s`` the grain positions before and ``s'``
after the post-push settle.

State is stored in the parent's ``(N, 7)`` layout with identity quaternions.
That wastes 4 floats per grain, and it is the right trade: the alternative is a
second schema that every downstream tool would have to learn.

Sizing
------
The default pile (20 mm radius, 12 mm tall, 2 mm grains) samples to roughly two
thousand particles, so a transition is ~110 kB and a thousand transitions ~110
MB. `--episodes 40 --n-envs 8 --pushes 5` gives 1600 transitions, which is the
same order as the cube datasets the visual-foresight numbers were computed on
(1400-2600), i.e. enough to run `variance_decomposition.py` and
`fit_linear_foresight.py` and compare like with like.

Occupancy
---------
Convert with `transforms/sand_occupancy.py`, **not** `particles_to_occupancy` —
the latter clamps to a binary silhouette and would throw away the depth that is
the only reason to simulate a continuum.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sand.yaml",
                    help="config under Genesis/, relative")
    ap.add_argument("--n-envs", type=int, default=8,
                    help="parallel envs. MPM allocates a dense background grid "
                         "per env, so this costs far more memory than the rigid "
                         "path at the same count -- raise it only after watching "
                         "one run.")
    ap.add_argument("--episodes", type=int, default=40,
                    help="pile restored to its initial state at the start of each")
    ap.add_argument("--pushes", type=int, default=5,
                    help="sequential pushes per episode; each is one transition")
    ap.add_argument("--push-length", type=float, default=0.02,
                    help="metres. A SINGLE fixed length, so the dataset supports "
                         "one switched-linear operator without binning (see "
                         "docs/linear_foresight_findings.md).")
    ap.add_argument("--min-swath-particles", type=int, default=3)
    ap.add_argument("--pile-clearance", type=float, default=None,
                    help="blade-to-pile gap at push start, metres. Defaults to "
                         "one MPM particle size.")
    ap.add_argument("--output-root", default="data/sand/pile")
    ap.add_argument("--state-library", default=None, metavar="GLOB",
                    help="glob of a previous sand collection to build varied "
                         "episode starts from. Without it every episode "
                         "restarts from the identical as-sampled pile, which "
                         "measurably caps the state diversity (~14 dims for 90%% "
                         "of variance) and therefore the rank any fitted "
                         "operator can need.")
    ap.add_argument("--library-size", type=int, default=4000)
    ap.add_argument("--library-jitter", type=float, default=0.15,
                    help="grain jitter as a fraction of grain diameter")
    ap.add_argument("--particle-size", type=float, default=None,
                    help="override mpm_options.particle_size; drives both the "
                         "grain scale and the particle count")
    ap.add_argument("--debug", action="store_true",
                    help="verbose sim logs. NOTE: the simulator treats debug as "
                         "'show viewer', which renders every step and destroys "
                         "any timing measurement -- never pass it to time a run.")
    return ap.parse_args()


def main():
    args = parse_args()
    from .sand_manipulation import SandManipulation, load_sand_config

    cfg = load_sand_config(args.config)
    if args.particle_size is not None:
        cfg.setdefault("mpm_options", {})["particle_size"] = args.particle_size

    cfg.setdefault("data_collection", {}).update({
        "sand": True,
        "pushes_per_episode": args.pushes,
        "push_length": args.push_length,
        "pile_aware": True,
        "perpendicular_pushes": True,
        "min_swath_particles": args.min_swath_particles,
        "mpm_particle_size": cfg.get("mpm_options", {}).get("particle_size"),
        "sand_material": cfg.get("sand", {}),
    })

    print(f"sand collection: {args.episodes} episodes x {args.pushes} pushes "
          f"x {args.n_envs} envs = {args.episodes * args.pushes * args.n_envs} "
          f"transitions, push {1000 * args.push_length:.0f} mm", flush=True)

    t0 = time.time()
    sim = SandManipulation(config=cfg, n_envs=args.n_envs,
                           debug=args.debug, viewer_type=None)
    sim.build()
    print(f"  build+sample: {time.time() - t0:.0f}s", flush=True)

    if args.state_library:
        from .sand_state_library import build_sand_state_library
        lib = build_sand_state_library(
            args.state_library,
            box_vol=cfg["box"]["vol"],
            n_states=args.library_size,
            skip_first_push=args.n_envs,
            jitter_frac=args.library_jitter,
            particle_size=cfg.get("mpm_options", {}).get("particle_size", 0.002),
            device="cpu")
        sim.set_state_library(lib)
        cfg["data_collection"]["state_library"] = {
            "source": args.state_library, "size": int(lib.shape[0]),
            "jitter_frac": args.library_jitter}

    clearance = (args.pile_clearance if args.pile_clearance is not None
                 else cfg.get("mpm_options", {}).get("particle_size", 0.002))

    out_path = args.output_root
    for ep in range(args.episodes):
        t = time.time()
        # Restores the as-sampled pile. Cheap for sand -- a pose write, not a
        # re-settle -- which is why episodes can be short and numerous.
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
                pile_clearance=clearance,
                min_swath_particles=args.min_swath_particles,
            )
        except RuntimeError as exc:
            print(f"  episode {ep + 1} failed: {exc}", flush=True)
            continue
        print(f"  episode {ep + 1}/{args.episodes} "
              f"({time.time() - t:.0f}s)", flush=True)

    print(f"done in {time.time() - t0:.0f}s -> Genesis/{out_path}", flush=True)
    sim.destroy()


if __name__ == "__main__":
    main()
