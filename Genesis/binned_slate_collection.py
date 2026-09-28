"""
Genesis/binned_slate_collection.py — multi-step same-state slates with
length-binned actions.

    python -u -m Genesis.binned_slate_collection \\
        --n-states 8 --n-actions 20 --n-steps 3 --n-envs 128 \\
        --output-root data/slates_binned --tag n20_b5_L20-70mm

What this collects
------------------
``n_states`` distinct settled start states. From each, ``n_actions``
independent action *chains* of ``n_steps`` pushes — so
``n_states * n_actions * n_steps`` simulated pushes, laid out as
``n_states`` slates of ``n_actions`` candidate sequences, exactly the
structure an MPC action-ranker is scored against (the true result of every candidate, and of
every candidate *sequence*, is known because it was simulated).

The difference from ``same_state_slate_collection.py`` is the action-length
distribution. That module fixes ONE push length per corpus, which left four of
six length bins with zero data (see ``docs/CODEMAP.md``). Here the push length
is drawn uniformly inside one of five bins spanning 20–70 mm, and a chain
*mixes* bins across its steps, so a single corpus supports fitting and scoring
length-conditioned operators over the whole range.

The two things that make this non-trivial, and how they are handled
-------------------------------------------------------------------

**1. Sampling is separated from simulation.** ``execute_action`` sweeps all
``n_envs`` in lockstep and the sweep is sized by the LONGEST travel in the
batch, so mixing a 20 mm and a 70 mm push in one batch makes the short one pay
for the long one. Every simulated batch therefore contains actions of one bin
only. That cannot be arranged while sampling, because pile-aware sampling is
*state-dependent*: it reads the current pile, and the length it actually
achieves is capped by how far the blade can travel inside the tray from the
contact point it found. So each step runs in two passes:

    sampling pass   load each chain's current state, draw its action at a
                    target length from its requested bin (retrying — see 2),
                    take no simulation steps. Cheap: a pose write plus tensor
                    maths.
    execution pass  regroup the drawn actions by their REALIZED length bin,
                    and simulate one ``n_envs``-sized batch per (bin, chunk).

Because the regrouping happens between the passes, the bin counts a batch
actually receives are whatever the sampler produced — they are *not* the
configured counts. Every loop here is driven by the realized grouping; nothing
assumes ``n_actions / n_bins`` per bin.

**2. A chain's later steps need states this run produced.** A chain's step-2
start state is its step-1 result, and its step-1 and step-2 bins differ. So
the post-push states of one sweep are carried in memory and written back into
the sim (``set_particle_state``, no re-settle — they are already at rest) as
the next sweep's initial states. Step ``k`` is fully collected before step
``k+1`` begins, which is also what keeps the bin regrouping global rather than
per-chunk.

Bin retry policy (``--max-length-tries``)
-----------------------------------------
Pile-aware sampling places the blade against the pile and sweeps into it; from
some contact points the tray wall is closer than the requested length, and
``_pile_aware_stops`` shortens the push rather than lying about it. Redrawing
until every env hits its bin would bias the action set toward pile
configurations that happen to admit long pushes. Instead each env gets up to
``--max-length-tries`` independent redraws, keeps the LONGEST push it saw, and
whatever that is gets binned by its realized length. Short pushes therefore
land in a lower bin instead of being discarded, and the corpus keeps the
state-dependence of the action distribution instead of filtering it out.

A push that comes in under the LOWEST edge is labelled ``UNDERFLOW_BIN`` (-1)
rather than folded into bin 0: it is not a sample of the 20-30 mm operator and
must not be fitted as one. It is still simulated and recorded — which pile
configurations force a short push is itself information, and dropping them at
collection time would bias the corpus toward piles that happen to sit where
long pushes fit.

On-disk schema
--------------
``{output_root}/{tag}/step{k}.pt``, one file per step, every tensor indexed by
a flat chain id ``c = slate_idx * n_actions + action_idx`` (so
``n_chains = n_states * n_actions`` rows, identical row ordering in every step
file — row ``c`` of ``step1.pt`` continues row ``c`` of ``step0.pt``):

    states            (n_chains, n_particles, 7)  pre-push particle pose (xyz + wxyz quat)
    states_           (n_chains, n_particles, 7)  post-push, after settling
    p_starts/p_stops  (n_chains, 3)     blade start / stop as actually executed
                                 (``execute_action``'s final pose, which can
                                 differ from the sampled stop if the sweep was
                                 cut short; ``len_realized`` below is the
                                 SAMPLED geometry, and is what binning used)
    angles            (n_chains,)       blade yaw, radians
    len_target        (n_chains,)       length asked of the sampler, metres
    len_realized      (n_chains,)       ||p_stop - p_start||, metres
    bin_requested     (n_chains,)       int, index into the bin edges
    bin_realized      (n_chains,)       int, bin len_realized actually fell in, or
                                 ``UNDERFLOW_BIN`` (-1) if it fell below the
                                 lowest edge
    slate_idx         (n_chains,)       int, which start state
    reached_goal      (n_chains,)       bool, execute_action's own success flag

plus ``manifest.json`` (full resolved config, bin edges, the state-library
indices used, and per-step requested-vs-realized bin histograms).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .binned_slate_dataset import UNDERFLOW_BIN, realized_bin


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="Genesis/configs/basic.yaml")
    ap.add_argument("--n-cubes", type=int, default=20)
    ap.add_argument("--size", type=float, default=0.005, help="cube edge, metres")
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--friction", type=float, default=0.3,
                    help="particle friction (and box friction unless --box-friction is given). "
                         "NOTE: overnight_randlen / Sean (the training corpora) use particle 0.7, "
                         "box 0.5, density 450 -- the 0.3 / 1000 defaults do NOT match them")
    ap.add_argument("--box-friction", type=float, default=None,
                    help="box (tray) friction; default = --friction (the historical behaviour)")
    ap.add_argument("--settle-steps", type=int, default=None,
                    help="override simulation.settle_steps (a cap: settling stops early once still)")
    ap.add_argument("--safety-margin", type=float, default=None,
                    help="override the top-level safety_margin (action-sampling margin)")
    ap.add_argument("--n-envs", type=int, default=128,
                    help="environments simulated concurrently. Unrelated to "
                         "the slate layout: chains are packed into batches of "
                         "this size regardless of which slate they belong to")
    ap.add_argument("--n-states", type=int, default=8,
                    help="distinct settled start states (one slate each)")
    ap.add_argument("--n-actions", type=int, default=20,
                    help="action chains per start state; their step-0 "
                         "lengths are split as evenly as possible across the bins")
    ap.add_argument("--n-steps", type=int, default=3,
                    help="pushes per chain (1 = single-push slates)")
    ap.add_argument("--bin-edges", type=float, nargs="+",
                    default=[0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                    help="metres; n_bins = len(edges) - 1. Lengths are drawn "
                         "uniformly inside the requested bin")
    ap.add_argument("--max-length-tries", type=int, default=10,
                    help="redraws allowed for an env whose pile-aware push "
                         "cannot reach its target length; the longest draw is "
                         "kept and binned by what it actually achieved")
    ap.add_argument("--min-swath-particles", type=int, default=3)
    ap.add_argument("--pile-clearance", type=float, default=None,
                    help="blade-to-pile gap at the start, metres (default: one "
                         "particle size)")
    ap.add_argument("--spawn-mode", default="heap", choices=["heap", "pyramid", "drop"],
                    help="how the start states are built. heap/pyramid are "
                         "PILE spawns (multi-layer); drop is a SCATTER spawn "
                         "(single layer -- dropped cubes bounce outward, so "
                         "90-94%% end up in layer 0)")
    ap.add_argument("--heap-base-frac", type=float, default=0.6)
    ap.add_argument("--n-settles", type=int, default=None,
                    help="fresh piles to settle for the state library "
                         "(default: ceil(n_states / n_envs) + 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-root", default="data/slates_binned",
                    help="relative to Genesis/, matching the other collection "
                         "drivers' convention")
    ap.add_argument("--tag", default="n20_binned")
    ap.add_argument("--checkpoint-every", type=int, default=5,
                    help="write step{k}.partial.pt + manifest.json every N simulated batches "
                         "(atomic), so a cut-off run keeps everything simulated so far")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------
# bin bookkeeping
# --------------------------------------------------------------------------

def balanced_bins(n: int, n_bins: int, rng) -> np.ndarray:
    """``n`` bin ids, as evenly split across ``n_bins`` as ``n`` allows, shuffled.

    Used per (slate, step): every step of every slate asks for a balanced
    spread of lengths, and the shuffle is what makes a chain's own schedule
    mix bins across its steps rather than repeat one.
    """
    ids = np.arange(n) % n_bins
    rng.shuffle(ids)
    return ids


def chunks(indices: np.ndarray, size: int):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


# --------------------------------------------------------------------------
# the two passes
# --------------------------------------------------------------------------

def draw_binned_actions(sim, target_len, max_tries, min_swath, pile_clearance):
    """One pile-aware push per env, aimed at ``target_len[env]`` metres.

    Takes no simulation steps — it reads whatever pile state is currently
    loaded. Redraws only the envs that fell short of their target, keeping the
    longest draw seen per env (see the module docstring on why the shortfall is
    kept rather than filtered out).

    Returns ``(starts, stops, angles, realized)`` shaped (E, 1, 3) / (E, 1, 3)
    / (E, 1) / (E,).
    """
    best_s = best_e = best_a = None
    best_len = None

    for _ in range(max_tries):
        s, e, a = sim.generate_action_samples(
            1, pile_aware=True, push_length=target_len,
            pile_clearance=pile_clearance, min_swath_particles=min_swath)
        length = (e[:, 0, :2] - s[:, 0, :2]).norm(dim=-1)

        if best_s is None:
            best_s, best_e, best_a, best_len = s, e, a, length
        else:
            better = (length > best_len).view(-1)
            best_s = torch.where(better.view(-1, 1, 1), s, best_s)
            best_e = torch.where(better.view(-1, 1, 1), e, best_e)
            best_a = torch.where(better.view(-1, 1), a, best_a)
            best_len = torch.where(better, length, best_len)

        if bool((best_len >= target_len.view(-1) - 1e-6).all()):
            break

    return best_s, best_e, best_a, best_len


def load_states(sim, states, idx):
    """Write chain states ``idx`` into the sim, one per env, padding to n_envs.

    Padding repeats the last index; padded envs are simulated (they must be,
    the batch steps in lockstep) and their results are never read.
    Returns the padded index array.
    """
    n_envs = int(sim._n_envs)
    padded = np.concatenate([idx, np.repeat(idx[-1:], n_envs - len(idx))]) \
        if len(idx) < n_envs else idx
    batch = states[padded].to(sim._particle_state.device)
    sim.set_particle_state(batch[..., 0:3].contiguous(), batch[..., 3:7].contiguous())
    return padded


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_manifest(manifest: dict, out_dir: Path) -> None:
    """Rewritten after every checkpoint and every step (atomic): `in_progress`
    says how far a running/cut-off step got, `complete` marks a finished run."""
    tmp = out_dir / "manifest.json.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, out_dir / "manifest.json")


def main():
    args = parse_args()
    from .sandbox_manipulation_clean import SandboxManipulation
    from .state_library import build_state_library

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    edges = np.asarray(args.bin_edges, dtype=np.float64)
    n_bins = len(edges) - 1
    assert n_bins >= 1 and np.all(np.diff(edges) > 0), "--bin-edges must be increasing"

    n_states, n_actions = args.n_states, args.n_actions
    n_steps, n_envs = args.n_steps, args.n_envs
    n_chains = n_states * n_actions

    cfg = yaml.safe_load(open(args.config))
    cfg["material"].update({"shape": "cube", "particle_size": args.size,
                            "n_particles": args.n_cubes, "density": args.density,
                            "particle_friction": args.friction})
    box_friction = args.friction if args.box_friction is None else args.box_friction
    cfg["box"]["friction"] = box_friction
    if args.settle_steps is not None:
        cfg.setdefault("simulation", {})["settle_steps"] = args.settle_steps
    if args.safety_margin is not None:
        cfg["safety_margin"] = args.safety_margin
    cfg.setdefault("rigid_options", {})["max_collision_pairs"] = max(150, args.n_cubes // 2)
    cfg["spawn"] = {"mode": args.spawn_mode, "heap_base_frac": args.heap_base_frac}
    cfg.setdefault("data_collection", {}).update({
        "binned_slate": True, "n_cubes": args.n_cubes, "cube_size": args.size,
        "pile_aware": True, "perpendicular_pushes": True,
        "bin_edges": [float(x) for x in edges], "n_states": n_states,
        "n_actions_per_state": n_actions, "n_steps": n_steps,
        "min_swath_particles": args.min_swath_particles,
        "max_length_tries": args.max_length_tries, "spawn": dict(cfg["spawn"]),
    })

    out_dir = Path(__file__).parent / args.output_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sim = SandboxManipulation(config=cfg, n_envs=n_envs, debug=args.debug, viewer_type=None)
    sim.build()
    sim.set_material_properties({"particle_friction": args.friction,
                                 "particle_density": args.density,
                                 "box_friction": box_friction,
                                 "sampled_particle_friction": None,
                                 "sampled_particle_density": None})
    print(f"build: {time.time() - t0:.0f}s", flush=True)

    n_settles = args.n_settles or int(np.ceil(n_states / n_envs)) + 1
    t_lib = time.time()
    lib = build_state_library(sim, n_settles=n_settles, augment=False,
                              damping=0.0, verbose=True)
    print(f"state library: {len(lib)} states in {time.time() - t_lib:.0f}s", flush=True)
    if len(lib) < n_states:
        print(f"WARNING library holds {len(lib)} states < {n_states} requested; "
              f"collecting {len(lib)} slates instead.", flush=True)
        n_states = min(n_states, len(lib))
        n_chains = n_states * n_actions
    chosen = np.sort(rng.choice(len(lib), size=n_states, replace=False))

    # Bin schedule, shaped (n_states, n_actions, n_steps). Balanced across
    # chains within every (slate, step) and independently shuffled per step,
    # which is what mixes lengths ALONG a chain.
    schedule = np.stack([
        np.stack([balanced_bins(n_actions, n_bins, rng) for _ in range(n_steps)], axis=-1)
        for _ in range(n_states)], axis=0)
    slate_idx = np.repeat(np.arange(n_states), n_actions)

    # cur[c] is chain c's current settled state; every chain of a slate starts
    # from the same one, which is the same-state slate property.
    # (n_chains, n_particles, 7)
    cur = lib.states[list(chosen)].cpu().repeat_interleave(n_actions, dim=0).clone()
    n_particles = cur.shape[1]

    manifest = {"n_cubes": args.n_cubes, "size": args.size, "density": args.density,
                "friction": args.friction, "box_friction": box_friction,
                "settle_steps": cfg.get("simulation", {}).get("settle_steps"),
                "safety_margin": cfg.get("safety_margin"), "n_envs": n_envs, "n_states": n_states,
                "n_actions_per_state": n_actions, "n_steps": n_steps, "n_chains": n_chains,
                "bin_edges": [float(x) for x in edges],
                "max_length_tries": args.max_length_tries,
                "min_swath_particles": args.min_swath_particles,
                # Full spawn dict, not just the mode: heap_base_frac changes
                # the pile's footprint, so a corpus that records only "heap" is
                # not reproducible from its own manifest.
                "spawn_mode": args.spawn_mode, "spawn": dict(cfg["spawn"]),
                "n_particles": args.n_cubes, "particle_shape": "cube",
                "seed": args.seed,
                "n_settles": n_settles, "library_size": len(lib),
                "state_library_indices": [int(i) for i in chosen],
                "pile_aware": True, "steps": []}

    t_collect = time.time()
    for k in range(n_steps):
        req_bin = schedule[:, :, k].reshape(n_chains)
        lo, hi = edges[req_bin], edges[req_bin + 1]
        target = torch.from_numpy(rng.uniform(lo, hi)).float()

        # ---- sampling pass: no simulation steps, only pose writes ----------
        # Explicit device: Genesis sets torch's default device to cuda, and
        # these are host-side bookkeeping buffers indexed by chain id.
        starts = torch.zeros(n_chains, 3, device="cpu")
        stops = torch.zeros(n_chains, 3, device="cpu")
        angles = torch.zeros(n_chains, device="cpu")
        realized = torch.zeros(n_chains, device="cpu")
        for chunk in chunks(np.arange(n_chains), n_envs):
            padded = load_states(sim, cur, chunk)
            tgt = target[padded].to(sim._particle_state.device).view(-1, 1)
            s, e, a, L = draw_binned_actions(
                sim, tgt, args.max_length_tries, args.min_swath_particles,
                args.pile_clearance)
            n = len(chunk)
            starts[chunk] = s[:n, 0].cpu()
            stops[chunk] = e[:n, 0].cpu()
            angles[chunk] = a[:n, 0].cpu()
            realized[chunk] = L[:n].cpu()

        got_bin = realized_bin(realized.numpy(), edges)

        # ---- execution pass: one batch per (realized bin, chunk) -----------
        nxt = torch.zeros_like(cur)
        reached = torch.zeros(n_chains, dtype=torch.bool, device="cpu")
        # UNDERFLOW_BIN first: it is a real group of pushes that must still be
        # simulated, just not ones that reached the lowest bin.
        # The batch count is known up front because the grouping is already
        # decided -- which is what lets a long single-step run report progress
        # at all, since nothing else in the loop prints until the step ends.
        t_exec = time.time()
        n_batches = sum(int(np.ceil(int((got_bin == b).sum()) / n_envs))
                        for b in [UNDERFLOW_BIN, *range(n_bins)])
        done_batches = 0
        done_mask = torch.zeros(n_chains, dtype=torch.bool, device="cpu")

        def _step_payload(done):
            return {"states": cur.clone(), "states_": nxt.clone(),
                    "p_starts": starts, "p_stops": stops, "angles": angles,
                    "len_target": target, "len_realized": realized,
                    "bin_requested": torch.from_numpy(req_bin).long(),
                    "bin_realized": torch.from_numpy(got_bin).long(),
                    "slate_idx": torch.from_numpy(slate_idx).long(),
                    "reached_goal": reached, "simulated": done.clone()}

        for b in [UNDERFLOW_BIN, *range(n_bins)]:
            members = np.flatnonzero(got_bin == b)
            if len(members) == 0:
                continue
            for chunk in chunks(members, n_envs):
                padded = load_states(sim, cur, chunk)
                dev = sim._particle_state.device
                s = starts[padded].to(dev)
                e = stops[padded].to(dev)
                a = angles[padded].to(dev)
                ok, final = sim.execute_action(s, e, a)
                sim.update_material_state()
                n = len(chunk)
                nxt[chunk] = sim._particle_state[:n].detach().cpu()
                stops[chunk] = final[:n].cpu()      # realized stop, post-clamp
                reached[chunk] = ok[:n].cpu().bool()

                done_batches += 1
                done_mask[chunk] = True
                if done_batches % args.checkpoint_every == 0 or done_batches == n_batches:
                    _save_atomic(_step_payload(done_mask), out_dir / f"step{k}.partial.pt")
                    manifest["in_progress"] = {"step": k, "done_batches": done_batches,
                                               "n_batches": n_batches,
                                               "n_chains_done": int(done_mask.sum()),
                                               "partial_file": f"step{k}.partial.pt",
                                               "elapsed_s": round(time.time() - t_collect, 1)}
                    _write_manifest(manifest, out_dir)
                rate = (time.time() - t_exec) / done_batches
                print(f"  step {k + 1}/{n_steps} bin {b}: batch "
                      f"{done_batches}/{n_batches} ({n} pushes) -- "
                      f"{rate:.0f}s/batch, ~{rate * (n_batches - done_batches) / 60:.0f}"
                      f" min left in this step", flush=True)

        payload = _step_payload(done_mask)
        payload.pop("simulated")               # a finished step file is complete by definition
        _save_atomic(payload, out_dir / f"step{k}.pt")
        (out_dir / f"step{k}.partial.pt").unlink(missing_ok=True)
        manifest.pop("in_progress", None)

        hist_req = np.bincount(req_bin, minlength=n_bins).tolist()
        hist_got = [int((got_bin == b).sum()) for b in range(n_bins)]
        n_under = int((got_bin == UNDERFLOW_BIN).sum())
        on_target = int((realized.numpy() >= edges[req_bin] - 1e-6).sum())
        manifest["steps"].append({
            "step": k, "requested_bin_counts": hist_req,
            "realized_bin_counts": hist_got, "n_underflow": n_under,
            "n_reaching_requested_bin": on_target, "n_chains": n_chains,
            "len_realized_m": {"min": float(realized.min()),
                               "mean": float(realized.mean()),
                               "max": float(realized.max())},
            "elapsed_s": round(time.time() - t_collect, 1)})
        print(f"step {k + 1}/{n_steps}: requested {hist_req} -> realized {hist_got} "
              f"(+{n_under} underflow); {on_target}/{n_chains} reached their bin; "
              f"{time.time() - t_collect:.0f}s elapsed", flush=True)

        _write_manifest(manifest, out_dir)
        cur = nxt

    manifest["complete"] = True
    _write_manifest(manifest, out_dir)
    print(f"done: {n_states} slates x {n_actions} chains x {n_steps} steps "
          f"= {n_chains * n_steps} pushes ({n_particles} particles) in "
          f"{time.time() - t_collect:.0f}s -> {out_dir}", flush=True)
    sim.destroy()


if __name__ == "__main__":
    main()
