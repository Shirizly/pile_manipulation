"""EXP-0059 R0 addition (coordinator, 2026-09-28): perturbed-simulator zoo members scored
through the FULL metric set on DS-0009, including slateN/slateN_tough -- the "clean, physically
feasible, state-uncertain" predictor family that is the key reference for the user's headline
question (does a ranking model's advantage come from real transition information, or just from
adding the right amount of noise/blur?).

For each perturbation level in {0, 0.5, 1, 2} mm (+ yaw jitter for levels > 0): for EVERY one of
DS-0009's 32 test_pools, perturb that pool's ONE shared start state once, re-simulate ALL 64 of
that pool's recorded actions from the perturbed state (same seam as
`chaos_floor.py`/`Genesis/chain_collection.py`: sim.set_particle_state + sim.execute_action +
sim.update_material_state, TRAINING_PHYSICS, no re-settle), and score the resulting 64-candidate
"prediction" against DS-0009's own recorded true outcomes with the IDENTICAL slateN/slateN_tough
code `eval_extended.py`'s occ-model path uses (`slate_capture`, imported not reimplemented).
Also computes BLURRED slateN (sigma 1, 2 px) for the 0.5mm/1mm members and for `nfd_3ch_narrow_l20`
(the frozen narrow NFD's own predictions, reusing its already-registered OCC_ADAPTER -- no
retraining), so sharpness is varied independently of information content, on the same pools.

Deliberately NOT run automatically by this file's import -- launch explicitly (heavy Genesis
job, ~25-30 min GPU at n_envs=64). Checkpoints atomically per (perturbation level, blur sigma)
cell, so a cut-off run loses at most one cell.

Usage (once the GPU is free -- see EXP-0059/LOG.md for why this was deferred):
    python -u perturbed_sim_zoo.py --levels 0 0.5 1 2 --blur-sigmas 0 1 2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
import eval_narrow as EN  # noqa: E402
from eval_extended import (GOALS_13, GOALS_TOUGH, _dist_fields, slate_capture,  # noqa: E402
                           gaussian_blur_occ)
from simple_mpc.adapters import occ_from_particles, occ_for_scoring, make_occ_adapter  # noqa: E402

D = EN.D
RES = Path(__file__).resolve().parents[1] / "results" / "perturbed_sim_zoo.json"


def _load_pools():
    import glob
    cpu = lambda d: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
    return [cpu(torch.load(f, map_location="cpu", weights_only=False))
           for f in sorted(glob.glob(str(D / "test_pools/pools_*.pt")))]


def perturb(states, xy_std_m, yaw_std_rad, rng):
    from model.retrieval.frame import yaw_from_quat, yaw_to_quat
    out = states.clone()
    if xy_std_m > 0:
        out[..., 0:2] += torch.from_numpy(rng.normal(0, xy_std_m, size=out[..., 0:2].shape)).float()
    if yaw_std_rad > 0:
        yaw0 = yaw_from_quat(out[..., 3:7])
        yaw1 = yaw0 + torch.from_numpy(rng.normal(0, yaw_std_rad, size=yaw0.shape)).float()
        out[..., 3:7] = yaw_to_quat(yaw1)
    return out


def score_pools(pools, Dist13, DistTough, per_pool_occ_fn, blur_sigmas):
    """per_pool_occ_fn(s0 (1,n,7), states_pool (64,n,7), p_starts (64,3), p_stops (64,3),
    angles (64,)) -> po (64,64,64) predicted occupancy, o0 (1,64,64) predicted start occupancy.
    Returns {sigma: {goal: [capture,...]}} pooled over every DS-0009 pool."""
    caps = {s: {"13": {g: [] for g in GOALS_13}, "tough": {g: [] for g in GOALS_TOUGH}}
           for s in blur_sigmas}
    for d in pools:
        for pi in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            s0 = d["states"][ix[0]][None].float()
            truth_occ = occ_for_scoring(d["states_"][ix, :, :3].float())
            t0 = occ_for_scoring(s0[:, :, :3])
            po, o0 = per_pool_occ_fn(s0, d["states"][ix].float(), d["p_starts"][ix].float(),
                                     d["p_stops"][ix].float(), d["angles"][ix].float())
            for sigma in blur_sigmas:
                out13 = slate_capture(po, o0, truth_occ, t0, Dist13, blur_sigma=sigma)
                outT = slate_capture(po, o0, truth_occ, t0, DistTough, blur_sigma=sigma)
                for g, v in out13.items():
                    if v is not None:
                        caps[sigma]["13"][g].append(v)
                for g, v in outT.items():
                    if v is not None:
                        caps[sigma]["tough"][g].append(v)
    return caps


def summarise(caps):
    out = {}
    for sigma, d in caps.items():
        s13 = {g: float(np.mean(v)) for g, v in d["13"].items() if v}
        sT = {g: float(np.mean(v)) for g, v in d["tough"].items() if v}
        out[f"blur{sigma}"] = dict(
            slateN=float(np.mean(list(s13.values()))) if s13 else None,
            slateN_tough=float(np.mean(list(sT.values()))) if sT else None,
            slateN_per_goal=s13, slateN_tough_per_goal=sT,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--blur-sigmas", type=float, nargs="*", default=[0.0, 1.0, 2.0])
    ap.add_argument("--yaw-jitter-deg", type=float, default=1.0)
    ap.add_argument("--also-model", default="nfd_3ch_narrow_l20",
                   help="also compute blurred slateN for this registered OCC_ADAPTERS model on "
                        "the same pools, for a sharpness-vs-information control (set '' to skip)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Dist13, DistTough = _dist_fields(GOALS_13), _dist_fields(GOALS_TOUGH)
    pools = _load_pools()
    out = json.loads(RES.read_text()) if RES.exists() else {}
    RES.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # --- the frozen NFD control (no Genesis needed) -------------------------------------
    if a.also_model and a.also_model not in out.get("model_control", {}):
        ad = make_occ_adapter(a.also_model, dev, "corner")

        def occ_fn_model(s0, states_pool, p_starts, p_stops, angles):
            o0 = occ_from_particles(s0, dev)
            act = torch.cat([p_starts[:, :2], p_stops[:, :2]], 1)
            with torch.no_grad():
                po = ad.predict_step(o0.expand(len(act), -1, -1).contiguous(), act.to(dev)).float().cpu()
            return po, o0.cpu()

        caps = score_pools(pools, Dist13, DistTough, occ_fn_model, a.blur_sigmas)
        out.setdefault("model_control", {})[a.also_model] = summarise(caps)
        tmp = str(RES) + ".tmp"; Path(tmp).write_text(json.dumps(out, indent=1)); os.replace(tmp, RES)
        print(f"[model control {a.also_model}] " +
             "  ".join(f"blur{s}:{out['model_control'][a.also_model][f'blur{s}']['slateN_tough']:.3f}"
                       for s in a.blur_sigmas), flush=True)

    # --- perturbed-simulator zoo (needs Genesis) -----------------------------------------
    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.learned_mpc import TRAINING_PHYSICS, apply_physics, oracle_config_with_physics
    from simple_mpc.oracle_mpc import load_oracle_config
    import genesis as gs

    physics = dict(TRAINING_PHYSICS)
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")), physics)
    cfg["dataset"]["record_transitions"] = False
    cfg["mpc"]["n_envs"] = 64
    env = GenesisOracleEnv(cfg, n_envs=64)
    apply_physics(env, physics)
    sim = env._sim
    sim._settle_steps = env._real_settle_steps
    sim._clearance_ctrl_steps = env._real_clearance_steps
    torch.set_default_device("cpu")  # see chaos_floor.py's identical fix / LOG.md

    out.setdefault("levels_mm", {})
    for lvl_mm in a.levels:
        key = f"{lvl_mm:g}mm"
        if key in out["levels_mm"]:
            print("skip (done):", key); continue
        t0 = time.time()
        yaw_std = np.deg2rad(a.yaw_jitter_deg) if lvl_mm > 0 else 0.0

        def occ_fn_sim(s0, states_pool, p_starts, p_stops, angles, _lvl=lvl_mm, _yaw=yaw_std):
            s0_pert = perturb(s0, _lvl / 1000.0, _yaw, rng)
            o0 = occ_from_particles(s0_pert)
            sim.set_particle_state(s0_pert.expand(64, -1, -1)[:, :, :3].to(gs.device),
                                   s0_pert.expand(64, -1, -1)[:, :, 3:7].to(gs.device))
            sim.update_material_state()
            sim.execute_action(p_starts.to(gs.device), p_stops.to(gs.device), angles.to(gs.device))
            sim.update_material_state()
            pred = sim._particle_state.detach().cpu().float().clone()
            return occ_from_particles(pred), o0

        caps = score_pools(pools, Dist13, DistTough, occ_fn_sim, a.blur_sigmas)
        out["levels_mm"][key] = summarise(caps)
        out["levels_mm"][key]["wall_s"] = time.time() - t0
        tmp = str(RES) + ".tmp"; Path(tmp).write_text(json.dumps(out, indent=1)); os.replace(tmp, RES)
        print(f"{key:8s} " + "  ".join(f"blur{s}:{out['levels_mm'][key][f'blur{s}']['slateN_tough']:.3f}"
                                       for s in a.blur_sigmas) + f"  ({out['levels_mm'][key]['wall_s']:.0f}s)",
             flush=True)

    env.destroy()
    print("wrote", RES)


if __name__ == "__main__":
    main()
