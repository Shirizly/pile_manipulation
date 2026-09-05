"""
Genesis/same_state_slate_collection.py — same-state candidate slates.

    python -m Genesis.same_state_slate_collection --n-states 50 --n-envs 32

Why
---
Every action-ranking result in this repo (`reports/linear_foresight_report.md`
sec 2.4, EXP-0008) draws its candidate actions from DIFFERENT states, because
`data_collection_clean.py` and `cube_spectrum_collection.py` both call
``shuffle_particles()`` (or ``state_library.apply()``, which shares one state
across envs but changes it every batch) once per material batch and then let
every env sample its OWN action independently -- so by the time a push
finishes, env-to-env variation is a mix of "different action" and "different
starting pile". That conflates "this action is good" with "this state was
easy", and it is specifically what makes EXP-0008's headline mechanism
(high-frequency prediction noise destroys ranking because it is independent
per candidate) possibly an artefact: in a real MPC loop, one state's
candidates share a prediction context, so the noise need not be independent
the way it is when the candidates themselves came from different states.

This module collects the fix directly: settle ONE pile, broadcast that
IDENTICAL state to every env via ``StateLibrary.apply_per_env`` with the SAME
index repeated for every env, let each env draw its own action (already how
``collect_data_samples(pile_aware=True)`` works -- unchanged), execute one
push per env, and record. Repeat for many distinct settled states, each its
own on-disk batch. No new sim wrapper: this is `cube_spectrum_collection.py`'s
exact material/spawn/action configuration (20 cubes, 5 mm, heap spawn, fixed
20 mm contact-aware pushes) plus `state_library.py`'s existing
``apply_per_env``, called from a thin loop.

On-disk schema is unchanged: ``sandbox_manipulation_clean.collect_data_samples``
still writes ``_{k}_data.pt`` / ``_{k}_failed.pt`` / ``_{k}_config.yaml`` via
its own auto-incrementing batch counter (files in the output dir / 3). Because
this driver calls it once per state with n_samples=1, batch index k IS the
slate index for free -- every row inside ``_{k}_data.pt`` is one candidate
action pushed from the same start state. A ``manifest.json`` is written
alongside for explicit traceability (batch index -> state-library index used,
plus the run config), so grouping does not rely on an implicit convention
surviving a re-read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="Genesis/configs/basic.yaml")
    ap.add_argument("--n-cubes", type=int, default=20)
    ap.add_argument("--size", type=float, default=0.005, help="cube edge, metres")
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--friction", type=float, default=0.3)
    ap.add_argument("--n-envs", type=int, default=32,
                    help="candidate actions per slate (one push per env)")
    ap.add_argument("--n-states", type=int, default=50,
                    help="distinct settled start states (slates) to collect")
    ap.add_argument("--push-length", type=float, default=0.02, help="metres")
    ap.add_argument("--min-swath-particles", type=int, default=3)
    ap.add_argument("--spawn-mode", default="heap", choices=["heap", "pyramid", "drop"])
    ap.add_argument("--heap-base-frac", type=float, default=0.6)
    ap.add_argument("--n-settles", type=int, default=None,
                    help="fresh piles to settle for the state library "
                         "(default: ceil(n_states / n_envs) + 1, so the "
                         "library has more distinct base states than needed "
                         "without relying on symmetry-augmented duplicates)")
    ap.add_argument("--augment", action="store_true",
                    help="expand the settled library by the box's symmetry "
                         "group (x8 for a square tray). Off by default here: "
                         "with n-settles already sized to cover n-states from "
                         "genuinely independent settles, augmentation would "
                         "just add symmetric duplicates to the pool drawn "
                         "from, and this collection cares about state "
                         "diversity more than settle cost.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs-build", type=int, default=None,
                    help="override n-envs used only while settling the state "
                         "library (default: same as --n-envs)")
    ap.add_argument("--output-root", default="data/slates",
                    help="relative to Genesis/, matching cube_spectrum_collection.py's "
                         "own convention -- collect_data_samples resolves `path` as "
                         "Path(__file__).parent / path where __file__ is inside Genesis/")
    ap.add_argument("--tag", default="n20_heap_5mm")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    from .sandbox_manipulation_clean import SandboxManipulation
    from .state_library import build_state_library

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    cfg = yaml.safe_load(open(args.config))
    cfg["material"].update({"shape": "cube", "particle_size": args.size,
                            "n_particles": args.n_cubes, "density": args.density,
                            "particle_friction": args.friction})
    cfg["box"]["friction"] = args.friction
    cfg.setdefault("rigid_options", {})["max_collision_pairs"] = max(150, args.n_cubes // 2)
    cfg["spawn"] = {"mode": args.spawn_mode, "heap_base_frac": args.heap_base_frac}

    out_path = f"{args.output_root}/{args.tag}"
    full_out = Path(__file__).parent / out_path
    full_out.mkdir(parents=True, exist_ok=True)

    n_envs = args.n_envs
    cfg.setdefault("data_collection", {}).update({
        "same_state_slate": True, "n_cubes": args.n_cubes, "cube_size": args.size,
        "push_length": args.push_length, "pile_aware": True,
        "perpendicular_pushes": True, "min_swath_particles": args.min_swath_particles,
        "spawn": dict(cfg["spawn"]), "n_envs_per_slate": n_envs,
        "n_states_target": args.n_states,
    })

    t0 = time.time()
    sim = SandboxManipulation(config=cfg, n_envs=n_envs, debug=args.debug, viewer_type=None)
    sim.build()
    sim.set_material_properties({"particle_friction": args.friction,
                                 "particle_density": args.density,
                                 "box_friction": args.friction,
                                 "sampled_particle_friction": None,
                                 "sampled_particle_density": None})
    print(f"build: {time.time() - t0:.0f}s", flush=True)

    n_settles = args.n_settles
    if n_settles is None:
        n_settles = int(np.ceil(args.n_states / n_envs)) + 1
    t_lib = time.time()
    lib = build_state_library(sim, n_settles=n_settles, augment=args.augment,
                              damping=0.0, verbose=True)
    print(f"state library: {len(lib)} states ({n_settles} settles x {n_envs} envs"
          f"{' x symmetries' if args.augment else ''}) in {time.time() - t_lib:.0f}s",
          flush=True)

    if len(lib) < args.n_states:
        print(f"WARNING: library only has {len(lib)} states, fewer than the "
              f"{args.n_states} requested; collecting {len(lib)} slates instead.",
              flush=True)
    n_states = min(args.n_states, len(lib))
    chosen = rng.choice(len(lib), size=n_states, replace=False)
    chosen.sort()  # no information encoded in order; sorted for readable logs

    manifest = {
        "n_cubes": args.n_cubes, "size": args.size, "density": args.density,
        "friction": args.friction, "n_envs": n_envs, "push_length": args.push_length,
        "spawn_mode": args.spawn_mode, "seed": args.seed,
        "n_settles": n_settles, "library_size": len(lib), "augmented": args.augment,
        "n_states_requested": args.n_states, "n_states_collected": n_states,
        "batches": [],  # filled in below: {batch_idx, state_library_index}
    }

    t_collect = time.time()
    for i, state_idx in enumerate(chosen):
        indices = [int(state_idx)] * n_envs
        used = lib.apply_per_env(sim, indices=indices)
        assert used == indices

        # batch index this write will get = current file count / 3 (see
        # collect_data_samples' own n_runs computation) -- read it BEFORE the
        # call so the manifest entry is unambiguous even if a run is resumed
        # into a partially-populated directory.
        import os
        batch_idx = int(len([n for n in os.listdir(full_out)
                             if os.path.isfile(os.path.join(full_out, n))]) / 3)

        sim.collect_data_samples(
            n_samples=1, path=out_path, placement_aware=False,
            shared_travel_distance=False, perpendicular_pushes=True,
            push_length=args.push_length, pile_aware=True,
            min_swath_particles=args.min_swath_particles,
        )
        manifest["batches"].append({"batch_idx": batch_idx,
                                    "state_library_index": int(state_idx)})
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t_collect
            print(f"  slate {i + 1}/{n_states} (state {state_idx}, batch "
                  f"{batch_idx}) -- {elapsed:.0f}s elapsed, "
                  f"{elapsed / (i + 1):.1f}s/slate", flush=True)

    with open(full_out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"done: {n_states} slates x up to {n_envs} candidates in "
          f"{time.time() - t_collect:.0f}s (+ {time.time() - t0 - (time.time() - t_collect):.0f}s "
          f"build/library) -> Genesis/{out_path}", flush=True)
    sim.destroy()


if __name__ == "__main__":
    main()
