"""
Genesis/same_state_slate_collection.py — same-state candidate slates.

    python -m Genesis.same_state_slate_collection --n-states 50 --n-envs 32

    # multi-step (3-push sequences), placement-aware start sampling:
    python -m Genesis.same_state_slate_collection --n-states 50 --n-envs 128 \\
        --n-steps 3 --placement-aware --no-pile-aware --push-length 0.02 \\
        --output-root data/slates_multistep --tag n20_L20mm

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
this driver calls it once per state (or once per state PER STEP -- see below)
with n_samples=1, batch index k IS the slate (and step) index for free --
every row inside ``_{k}_data.pt`` is one candidate action pushed from the same
start state. A ``manifest.json`` is written alongside for explicit
traceability (batch index -> slate index, state-library index, step index and
env count), so grouping does not rely on an implicit convention surviving a
re-read.

Multi-step sequences (``--n-steps > 1``)
-----------------------------------------
For scoring action SEQUENCES against an oracle, one push per candidate is not
enough -- each of a state's X envs needs to execute its own independent
3-push sequence, with the pile evolving between pushes. That is a straight
extension of the loop above: ``collect_data_samples(n_samples=1, ...)`` is
called ``n_steps`` times in a row for the same broadcast state WITHOUT
re-applying the state library between calls, so the second and third calls
see whatever the sim's own settle already left after the previous push (the
same way ``pile_aware`` collection already re-draws its action per push from
the CURRENT pile -- see the comment at the top of
``SandboxManipulation.collect_data_samples`` about stale-pile aiming). Each
call still writes its own ``_{k}_data.pt``, so a step is a batch file exactly
like a single-push slate is, and the manifest's ``step_idx`` field is what
lets a reader regroup three consecutive batch files back into one sequence
per (slate, env).

Sampling mode for multi-step collection
----------------------------------------
The single-push slates above use ``pile_aware=True`` (blade starts already
touching the pile). That code path OWNS the push geometry end-to-end --
``generate_action_samples`` returns immediately once ``pile_aware`` branch
runs (see ``Genesis/sandbox_manipulation_clean.py::generate_action_samples``)
-- so passing ``placement_aware=True`` at the same time is not a refinement,
it is dead: the placement-aware branch never executes. Multi-step collection
therefore uses ``placement_aware=True`` with ``pile_aware=False`` instead
(see ``--placement-aware`` / ``--pile-aware`` below); if both are requested
this module logs a warning and forces ``pile_aware=False``, i.e. it prefers
placement-aware. ``perpendicular_pushes`` and a fixed ``push_length`` compose
fine with ``placement_aware`` (they run afterwards, in
``_constrain_push_geometry``, and only touch direction/length, not the
touchdown point chosen by placement-aware sampling).

Two extra guarantees multi-step collection enforces that the single-push
slates did not need (see ``_draw_validated_actions``):

  wall-clipping   ``constrain_push`` already retries the +/- blade-normal sign
                  to avoid a push that cannot reach the target length from a
                  wall-adjacent start, but can still report a genuinely
                  infeasible action (via the length/realized mismatch this
                  module checks). Any such draw is re-sampled from scratch
                  BEFORE simulating, never truncated and recorded as if it
                  had reached the requested length.
  uniqueness      independently-sampled starts can coincide (see
                  ``action_sampling.duplicate_action_mask``); any two draws
                  that land within tolerance of each other are both
                  re-sampled, so a slate's X sequences never contain the same
                  (start, heading) twice at a given step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager, nullcontext
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
                    help="candidate actions (or action SEQUENCES, if "
                         "--n-steps > 1) per slate")
    ap.add_argument("--n-states", type=int, default=50,
                    help="distinct settled start states (slates) to collect")
    ap.add_argument("--n-steps", type=int, default=1,
                    help="pushes per candidate sequence. 1 reproduces the "
                         "original single-push same-state slate; >1 collects "
                         "independent multi-step sequences, one per env, "
                         "from the same broadcast start state")
    ap.add_argument("--push-length", type=float, default=0.02, help="metres")
    ap.add_argument("--min-swath-particles", type=int, default=3)
    ap.add_argument("--spawn-mode", default="heap", choices=["heap", "pyramid", "drop"])
    ap.add_argument("--heap-base-frac", type=float, default=0.6)
    ap.add_argument("--pile-aware", action=argparse.BooleanOptionalAction, default=True,
                    help="start every push in contact with the pile (see "
                         "docs/piled_collection.md). Mutually exclusive with "
                         "--placement-aware in effect -- see module docstring")
    ap.add_argument("--placement-aware", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="draw touchdown poses from the tool's free "
                         "configuration space instead of blindly. Needed for "
                         "multi-step collection since --pile-aware bypasses "
                         "it entirely")
    ap.add_argument("--pos-tol", type=float, default=0.001,
                    help="metres; two candidates' starts closer than this "
                         "AND within --angle-tol-deg of the same heading "
                         "count as a duplicate action and are re-sampled")
    ap.add_argument("--angle-tol-deg", type=float, default=5.0,
                    help="degrees; see --pos-tol")
    ap.add_argument("--max-resample-tries", type=int, default=30,
                    help="redraw attempts for wall-clipped/duplicate actions "
                         "before giving up and reporting the residual count")
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


def _draw_validated_actions(sim, n_envs, push_length, pos_tol, angle_tol,
                            max_tries, min_swath_particles, log):
    """One placement-aware, perpendicular, fixed-length action per env.

    Redraws (whole-batch, then splices only the still-bad entries) until every
    env's action both reaches the commanded length inside the workspace box
    and is not a duplicate of another env's (start, heading) -- or until
    ``max_tries`` is exhausted, in which case the residual bad envs are
    reported and returned anyway (their action is the best-effort last draw).

    Reads/writes nothing on ``sim`` except calling its own
    ``generate_action_samples`` -- this runs entirely on already-settled
    state, no simulation step is taken here, which is the point: wall-clipping
    and duplication are rejected BEFORE anything is simulated, not after.

    Returns (starts, stops, angles, still_bad) shaped (n_envs, 1, 3) /
    (n_envs, 1, 3) / (n_envs, 1) / (n_envs,).
    """
    from .action_sampling import duplicate_action_mask

    device = sim._particle_state.device
    still_bad = torch.ones(n_envs, dtype=torch.bool, device=device)
    best_starts = best_stops = best_angles = None
    tries_used = 0

    for attempt in range(max_tries):
        tries_used = attempt + 1
        starts, stops, angles = sim.generate_action_samples(
            1, placement_aware=True, perpendicular_pushes=True,
            push_length=push_length, pile_aware=False,
            shared_travel_distance=False,
            min_swath_particles=min_swath_particles)

        if best_starts is None:
            best_starts, best_stops, best_angles = starts, stops, angles
        else:
            m2 = still_bad.view(-1, 1, 1)
            best_starts = torch.where(m2, starts, best_starts)
            best_stops = torch.where(m2, stops, best_stops)
            best_angles = torch.where(still_bad.view(-1, 1), angles, best_angles)

        s2 = best_starts[:, 0, :2]
        e2 = best_stops[:, 0, :2]
        realized = (e2 - s2).norm(dim=-1)
        wall_ok = (realized - push_length).abs() < 1e-6

        heading = torch.atan2(e2[:, 1] - s2[:, 1], e2[:, 0] - s2[:, 0])
        dup = duplicate_action_mask(s2, heading, pos_tol, angle_tol)

        still_bad = (~wall_ok) | dup
        if not bool(still_bad.any()):
            break

    n_bad = int(still_bad.sum())
    if n_bad:
        n_wall = int((~wall_ok).sum())
        n_dup = int(dup.sum())
        log(f"WARNING action resampling: {n_bad}/{n_envs} actions still bad "
            f"after {tries_used} tries ({n_wall} wall-clipped/infeasible, "
            f"{n_dup} duplicate) -- proceeding with the best draw found. "
            f"These rows are NOT guaranteed at the requested length or unique.")
    elif tries_used > 1:
        log(f"action resampling: resolved in {tries_used} tries")

    return best_starts, best_stops, best_angles, still_bad


@contextmanager
def _patched_action_sampler(sim, starts, stops, angles):
    """Make ``sim.generate_action_samples`` return pre-validated actions once.

    ``collect_data_samples`` always draws its own actions internally; this is
    the seam that lets the driver substitute actions that have already been
    checked for wall-clipping and within-slate uniqueness (see
    ``_draw_validated_actions``) without duplicating any of
    ``collect_data_samples``'s buffer/save/statistics logic. Restores the
    original bound method on exit regardless of how the block exits.
    """
    original = sim.generate_action_samples

    def _fixed(*_args, **_kwargs):
        return starts, stops, angles

    sim.generate_action_samples = _fixed
    try:
        yield
    finally:
        sim.generate_action_samples = original


def main():
    args = parse_args()
    from .sandbox_manipulation_clean import SandboxManipulation
    from .state_library import build_state_library

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    pile_aware = args.pile_aware
    placement_aware = args.placement_aware
    if pile_aware and placement_aware:
        print("WARNING --pile-aware and --placement-aware were both requested; "
              "pile_aware's action-sampling branch returns before "
              "placement_aware's ever runs (see generate_action_samples), so "
              "they cannot compose. Preferring placement-aware: forcing "
              "pile_aware=False.", flush=True)
        pile_aware = False

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
        "push_length": args.push_length, "pile_aware": pile_aware,
        "placement_aware": placement_aware,
        "perpendicular_pushes": True, "min_swath_particles": args.min_swath_particles,
        "spawn": dict(cfg["spawn"]), "n_envs_per_slate": n_envs,
        "n_states_target": args.n_states, "n_steps": args.n_steps,
        "pos_tol": args.pos_tol, "angle_tol_deg": args.angle_tol_deg,
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
        "n_steps": args.n_steps, "pile_aware": pile_aware,
        "placement_aware": placement_aware,
        "spawn_mode": args.spawn_mode, "seed": args.seed,
        "n_settles": n_settles, "library_size": len(lib), "augmented": args.augment,
        "n_states_requested": args.n_states, "n_states_collected": n_states,
        "pos_tol": args.pos_tol, "angle_tol_deg": args.angle_tol_deg,
        "worst_state_broadcast_spread_m": 0.0,
        # one entry per WRITTEN batch file:
        # {batch_idx, slate_idx, state_library_index, step_idx, env_count}
        "batches": [],
    }

    def log(msg):
        print(msg, flush=True)

    angle_tol = math.radians(args.angle_tol_deg)
    t_collect = time.time()
    for i, state_idx in enumerate(chosen):
        indices = [int(state_idx)] * n_envs
        used = lib.apply_per_env(sim, indices=indices)
        assert used == indices

        # Verify the broadcast landed bit-identical across envs -- this reads
        # the state exactly as apply_per_env wrote it (no settle in between,
        # which would be wrong here: library states are already at rest).
        state_now = sim._particle_state.detach()
        spread = float((state_now.amax(dim=0) - state_now.amin(dim=0)).abs().max())
        assert spread < 1e-9, (
            f"state broadcast spread {spread:.3e} m across envs exceeds 1e-9 m "
            f"at slate {i} (state {state_idx})")
        manifest["worst_state_broadcast_spread_m"] = max(
            manifest["worst_state_broadcast_spread_m"], spread)

        for step in range(args.n_steps):
            # The resample-for-wall-clip/uniqueness guarantee is only
            # meaningful (and only correct) for the placement-aware path:
            # pile_aware's OWN action generation already owns the geometry
            # end-to-end (see the module docstring) and must not be replaced
            # here, or the original single-push pile_aware behaviour this
            # driver has always had would silently change underneath it.
            if placement_aware:
                starts, stops, angles, still_bad = _draw_validated_actions(
                    sim, n_envs, args.push_length, args.pos_tol, angle_tol,
                    args.max_resample_tries, args.min_swath_particles, log)
                patch_ctx = _patched_action_sampler(sim, starts, stops, angles)
                unresolved = int(still_bad.sum())
            else:
                patch_ctx = nullcontext()
                unresolved = None

            batch_idx = int(len([n for n in os.listdir(full_out)
                                 if os.path.isfile(os.path.join(full_out, n))]) / 3)

            with patch_ctx:
                sim.collect_data_samples(
                    n_samples=1, path=out_path, placement_aware=placement_aware,
                    shared_travel_distance=False, perpendicular_pushes=True,
                    push_length=args.push_length, pile_aware=pile_aware,
                    min_swath_particles=args.min_swath_particles,
                )

            data_path = full_out / f"_{batch_idx}_data.pt"
            env_count = int(torch.load(data_path, weights_only=False)["p_starts"].shape[0]) \
                if data_path.exists() else 0
            manifest["batches"].append({
                "batch_idx": batch_idx, "slate_idx": int(i),
                "state_library_index": int(state_idx), "step_idx": int(step),
                "env_count": env_count,
                "unresolved_after_resampling": unresolved,
            })

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t_collect
            print(f"  slate {i + 1}/{n_states} (state {state_idx}) -- "
                  f"{elapsed:.0f}s elapsed, {elapsed / (i + 1):.1f}s/slate "
                  f"({args.n_steps} step(s) each)", flush=True)

    with open(full_out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"done: {n_states} slates x up to {n_envs} candidates x {args.n_steps} "
          f"step(s) in {time.time() - t_collect:.0f}s (+ "
          f"{time.time() - t0 - (time.time() - t_collect):.0f}s build/library) "
          f"-> Genesis/{out_path}; worst broadcast spread "
          f"{manifest['worst_state_broadcast_spread_m']:.3e} m", flush=True)
    sim.destroy()


if __name__ == "__main__":
    main()
