"""EXP-0059 R0 task 3: chaos floor / state-uncertainty ceiling.

Takes a sample of DS-0009 test_chains transitions (real state0, action,
recorded true state1), re-simulates each one from state0 (a) unperturbed and
(b) with the initial cube positions perturbed by 0.5/1/2 mm (+ small yaw
jitter), using the EXACT seam Genesis/chain_collection.py uses to produce
DS-0009 itself (TRAINING_PHYSICS, sim.set_particle_state + execute_action +
update_material_state, no re-settle). Each re-simulated outcome is then
scored AS IF it were a prediction of the ORIGINAL recorded DS-0009 outcome,
through the same accuracy / blurred-accuracy / mm metrics eval_extended.py
uses for every model -- so this is the ceiling for how well ANY model, even
a perfect dynamics function fed a slightly-wrong initial state, could do.

Particle order is preserved by the sim (no reordering), so per-cube mm error
is a direct correspondence, no Hungarian matching needed.

Usage:
    python -u chaos_floor.py --n-sample 64 --levels 0 0.5 1 2
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
import eval_narrow as EN  # noqa: E402
from simple_mpc.adapters import occ_from_particles  # noqa: E402
from eval_extended import blurred_acc, GOALS_TOUGH  # noqa: E402

D = EN.D
OUT = Path(__file__).resolve().parents[1] / "results" / "chaos_floor.json"


def sample_rows(n_sample, seed=0):
    import glob
    files = sorted(glob.glob(str(D / "test_chains/_*_data.pt")))
    chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    # flatten with a global row index -> (chunk_i, row_i)
    all_states, all_states1, all_ps, all_pe, all_ang, all_kind = [], [], [], [], [], []
    for d in chunks:
        all_states.append(d["states"]); all_states1.append(d["states_"])
        all_ps.append(d["p_starts"]); all_pe.append(d["p_stops"]); all_ang.append(d["angles"])
        all_kind += list(d["start_kind"])
    states = torch.cat(all_states).float(); states1 = torch.cat(all_states1).float()
    ps = torch.cat(all_ps).float(); pe = torch.cat(all_pe).float(); ang = torch.cat(all_ang).float()
    kind = np.array(all_kind)
    rng = np.random.default_rng(seed)
    idx_scatter = np.nonzero(kind == "scatter")[0]
    idx_clump = np.nonzero(kind == "clump")[0]
    n_half = n_sample // 2
    pick = np.concatenate([
        rng.choice(idx_scatter, min(n_half, len(idx_scatter)), replace=False),
        rng.choice(idx_clump, min(n_sample - n_half, len(idx_clump)), replace=False),
    ])
    pick = pick[:n_sample]
    return states[pick], states1[pick], ps[pick], pe[pick], ang[pick], kind[pick]


def perturb(states, xy_std_m, yaw_std_rad, rng):
    """states (B,n,7) = [x,y,z,qw,qx,qy,qz]. Perturb xy by iid gaussian noise,
    and yaw by a small jitter applied via quaternion (z-axis rotation compose,
    valid for flat single-layer cubes where yaw is the only orientation dof
    that matters -- consistent with model/retrieval/frame.py's yaw-only
    treatment of the same corpus)."""
    from model.retrieval.frame import yaw_from_quat, yaw_to_quat
    out = states.clone()
    if xy_std_m > 0:
        out[..., 0:2] += torch.from_numpy(rng.normal(0, xy_std_m, size=out[..., 0:2].shape)).float()
    if yaw_std_rad > 0:
        yaw0 = yaw_from_quat(out[..., 3:7])
        yaw1 = yaw0 + torch.from_numpy(rng.normal(0, yaw_std_rad, size=yaw0.shape)).float()
        out[..., 3:7] = yaw_to_quat(yaw1)
    return out


def score(pred_states, true_states, prev_states, act):
    """Mirrors eval_extended.py's per-row scoring: image accuracy (blurred at
    sigma 0/1/2), plus mean/rms per-cube mm position error (direct
    correspondence, particle order preserved by the sim)."""
    P = occ_from_particles(pred_states)
    T = occ_from_particles(true_states)
    O = occ_from_particles(prev_states)
    act2 = torch.cat([act[:, :2], act[:, 2:4]], 1) if act.shape[1] >= 4 else act
    R = EN.swept_region(act, "cpu")
    out = {}
    out["accuracy_1"] = EN.acc(P, T, O, R)
    for sigma in (1, 2):
        out[f"accuracy_1_blur{sigma}"] = blurred_acc(P, T, O, R, sigma)
    d = (pred_states[..., :2] - true_states[..., :2]).norm(dim=-1) * 1000.0  # mm, per cube
    out["mm_mean"] = float(d.mean())
    out["mm_rms"] = float((d ** 2).mean().sqrt())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=64)
    ap.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--yaw-jitter-deg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.learned_mpc import TRAINING_PHYSICS, apply_physics, oracle_config_with_physics
    from simple_mpc.oracle_mpc import load_oracle_config
    import genesis as gs

    states0, states1_true, p_starts, p_stops, angles, kinds = sample_rows(a.n_sample, seed=a.seed)
    n_envs = len(states0)
    print(f"sampled {n_envs} rows ({(kinds=='scatter').sum()} scatter, {(kinds=='clump').sum()} clump)", flush=True)

    physics = dict(TRAINING_PHYSICS)
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")), physics)
    cfg["dataset"]["record_transitions"] = False
    cfg["mpc"]["n_envs"] = n_envs
    t0 = time.time()
    env = GenesisOracleEnv(cfg, n_envs=n_envs)
    apply_physics(env, physics)
    sim = env._sim
    sim._settle_steps = env._real_settle_steps
    sim._clearance_ctrl_steps = env._real_clearance_steps
    print(f"[build] {time.time()-t0:.1f}s", flush=True)
    # Genesis sets torch's default device to cuda as a side effect of gs.init()
    # (data-collection skill's documented trap) -- eval_narrow.swept_region's
    # internal torch.linspace calls then default to cuda while our own
    # CPU-loaded `act`/state tensors stay cpu, a device mismatch. All further
    # sim calls in this script explicitly .to(gs.device) their own tensors, so
    # resetting the default back to cpu here only affects OUR bookkeeping code.
    torch.set_default_device("cpu")

    rng = np.random.default_rng(a.seed)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("n_sample", n_envs)
    out.setdefault("levels_mm", {})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    act = torch.cat([p_starts[:, :2], p_stops[:, :2]], 1).float()

    for lvl_mm in a.levels:
        key = f"{lvl_mm:g}mm"
        if key in out["levels_mm"]:
            print("skip (done):", key); continue
        t1 = time.time()
        yaw_std = np.deg2rad(a.yaw_jitter_deg) if lvl_mm > 0 else 0.0
        st_pert = perturb(states0, lvl_mm / 1000.0, yaw_std, rng)
        sim.set_particle_state(st_pert[:, :, :3].to(gs.device), st_pert[:, :, 3:7].to(gs.device))
        sim.update_material_state()
        sim.execute_action(p_starts.to(gs.device), p_stops.to(gs.device), angles.to(gs.device))
        sim.update_material_state()
        pred = sim._particle_state.detach().cpu().float().clone()
        r = score(pred, states1_true, states0, act)
        for k_ in ("scatter", "clump"):
            ix = torch.from_numpy(np.nonzero(kinds == k_)[0])
            if len(ix):
                r[f"accuracy_1_{k_}"] = EN.acc(occ_from_particles(pred[ix]), occ_from_particles(states1_true[ix]),
                                              occ_from_particles(states0[ix]), EN.swept_region(act[ix], "cpu"))
        r["n"] = n_envs
        r["wall_s"] = time.time() - t1
        out["levels_mm"][key] = r
        tmp = str(OUT) + ".tmp"; Path(tmp).write_text(json.dumps(out, indent=1)); os.replace(tmp, OUT)
        print(f"{key:8s} acc1 {r['accuracy_1']:+.3f}  blur1 {r['accuracy_1_blur1']:+.3f}  blur2 {r['accuracy_1_blur2']:+.3f}  "
             f"mm_mean {r['mm_mean']:.3f}  mm_rms {r['mm_rms']:.3f}  ({r['wall_s']:.1f}s)", flush=True)

    env.destroy()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
