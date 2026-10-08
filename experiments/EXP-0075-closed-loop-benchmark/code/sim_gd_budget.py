"""EXP-0075: execute the gd_budget plans (10 tasks x plans ref / gd_ref / gd5 / gd10 / cem / gd_full) open loop in the simulator (each push legalised against the live cubes), record true value after every push.
One replica per plan (replica sd was <= 0.02 in diag_sim_intensive). 32 envs per batch, 2 batches.  usage: PYTHONPATH=. python -u sim_gd_budget.py"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask, goal_dist
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, lyap
from simple_mpc.adapters import occ_for_scoring
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose
K = 32; OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"; PLANS = ["ref", "gd_ref", "gd5", "gd10", "cem", "gd_full"]


def main():
    tasks = [json.load(open(f)) for f in sorted((OUT / "gd_budget").glob("*.json"))]; specs = [(t["goal"], t["start"], p, t[p]["seq"], t[p]["cost"]) for t in tasks for p in PLANS]; print(len(tasks), "tasks", len(specs), "plans", flush=True)
    rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0); starts = {s: rows.states[(rows.slate_idx == s).nonzero()[0, 0]].float() for s in {x[1] for x in specs}}
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))); cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    dists = {g: goal_dist(g) for g in {x[0] for x in specs}}; masks = {g: torch.from_numpy(goal_mask(g)).float().reshape(-1) for g in dists}

    def legalize(acts):
        from Genesis.action_sampling import legalize_pushes
        from model.retrieval.frame import yaw_from_quat
        from simple_mpc.adapters import OCC_BOUNDS
        from simple_mpc.learned_mpc import XY_MARGIN
        st = sim._particle_state.detach().cpu().float()
        return legalize_pushes(acts, st[:, :, :2], yaw_from_quat(st[:, :, 3:7]), 0.0025, 0.02, 0.001, box=(OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN))

    out = []
    for b0 in range(0, len(specs), K):
        batch = specs[b0:b0 + K]; n = len(batch); pad = batch + [batch[0]] * (K - n)
        def value(parts):
            occ = occ_for_scoring(parts[:, :, :3].float()).cpu(); f = occ.reshape(K, -1)
            return np.array([float(lyap(occ[k:k + 1], dists[pad[k][0]])[0] - (f[k] * masks[pad[k][0]]).sum() / f[k].sum()) for k in range(K)])
        st = torch.stack([starts[x[1]] for x in pad]); sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device)); sim.update_material_state()
        p = sim._particle_state[:, :, :7].detach().cpu().float().clone(); vals = [value(p)]; parts = [p.numpy()]; shifts = []; seqs = torch.tensor([x[3] for x in pad]).float()
        for j in range(4):
            a, shift, ok = legalize(seqs[:, j]); shifts.append(shift.cpu().numpy()); sx, sy, ex, ey, ang = action_to_pose(a.to(gs.device).float()); z = torch.full_like(sx, float(sim._operation_height))
            sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); sim.update_material_state()
            p = sim._particle_state[:, :, :7].detach().cpu().float().clone(); vals.append(value(p)); parts.append(p.numpy())
        vals = np.stack(vals, 1); shifts = np.stack(shifts, 1); parts = np.stack(parts, 1)
        for k in range(n):
            g, s, pl, sq, pc = batch[k]; out.append(dict(goal=g, start=s, plan=pl, pred_terminal=pc, true_terminal=float(vals[k, -1] - vals[k, 0]), true_per_push=(vals[k, 1:] - vals[k, 0]).tolist(), shift_mm=float(shifts[k].mean() * 1000), shifted_pushes=int((shifts[k] > 0).sum())))
        json.dump(out, open(OUT / "gd_budget_sim.json", "w"), indent=1); np.savez_compressed(OUT / f"gd_budget_sim_parts_{b0}.npz", parts=parts[:n], specs=np.array([f"{x[0]}|{x[1]}|{x[2]}" for x in batch])); print("batch", b0, "done", flush=True)
    env.destroy()


if __name__ == "__main__":
    main()
