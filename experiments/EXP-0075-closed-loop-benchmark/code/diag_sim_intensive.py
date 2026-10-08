"""EXP-0075: execute the six INTENSIVE-optimisation sequences (results/intensive.json; letter_T_w20, start 40) open loop in the simulator -- each pushed legalised against the live cubes -- 5 replicas each (32 envs = 5 x 6 sequences + 2 spare),
record the TRUE objective value after every push (soft truth) and compare with the model's predicted value change; save true occupancies for figures.  usage: PYTHONPATH=. python -u diag_sim_intensive.py"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask, goal_dist
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, lyap
from simple_mpc.adapters import occ_for_scoring, occ_from_particles
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose
K = 32; GOAL, START = "letter_T_w20", 40; OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"
NAMES = ["ref", "random", "cem10k", "greedy", "beam", "gd"]


def main():
    res = json.load(open(OUT / "intensive.json")); S0 = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
    S0 = S0.states[(S0.slate_idx == START).nonzero()[0, 0]].float(); dist, mask = goal_dist(GOAL), torch.from_numpy(goal_mask(GOAL)).float()
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))); cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    which = [i % len(NAMES) for i in range(K)]; seqs = torch.tensor([res[NAMES[w]]["seq"] for w in which]).float()          # (K, 4, 4)

    def legalize(acts):
        from Genesis.action_sampling import legalize_pushes
        from model.retrieval.frame import yaw_from_quat
        from simple_mpc.adapters import OCC_BOUNDS
        from simple_mpc.learned_mpc import XY_MARGIN
        st = sim._particle_state.detach().cpu().float()
        return legalize_pushes(acts, st[:, :, :2], yaw_from_quat(st[:, :, 3:7]), 0.0025, 0.02, 0.001, box=(OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN))

    HARD, PARTS = [], []

    def value(parts):
        occ = occ_for_scoring(parts[:, :, :3].float()).cpu(); f = occ.reshape(K, -1); PARTS.append(parts[:, :, :7].float().cpu().numpy()); HARD.append(occ_from_particles(parts[:, :, :3].float()).cpu().numpy().astype(np.float16)); return (lyap(occ, dist) - (f * mask.reshape(1, -1)).sum(1) / f.sum(1)).numpy(), occ.numpy()

    st = S0[None].expand(K, -1, -1).contiguous(); sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device)); sim.update_material_state()
    p = sim._particle_state[:, :, :7].detach().cpu().float().clone(); v, o = value(p); vals, occs, shifts, oks = [v], [o], [], []
    for j in range(4):
        a, shift, ok = legalize(seqs[:, j]); shifts.append(shift.cpu().numpy()); oks.append(ok.cpu().numpy())
        sx, sy, ex, ey, ang = action_to_pose(a.to(gs.device).float()); z = torch.full_like(sx, float(sim._operation_height))
        sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); sim.update_material_state()
        p = sim._particle_state[:, :, :7].detach().cpu().float().clone(); v, o = value(p); vals.append(v); occs.append(o)
    vals = np.stack(vals, 1); occs = np.stack(occs, 1).astype(np.float16); shifts = np.stack(shifts, 1); oks = np.stack(oks, 1); which = np.array(which)
    out = {}
    for i, n in enumerate(NAMES):
        m = which == i; tv = (vals[m] - vals[m][:, :1])[:, 1:]; pv = np.array(res[n]["seq_value"])
        out[n] = dict(pred=pv.tolist(), true_mean=tv.mean(0).tolist(), true_sd=tv.std(0).tolist(), true_terminal_replicas=tv[:, -1].tolist(), optimism_terminal=float(pv[-1] - tv[:, -1].mean()), n_replicas=int(m.sum()),
                      shifted_pushes_mean=float((shifts[m] > 0).sum(1).mean()), shift_mm_mean=float(shifts[m].mean() * 1000), illegal_after_legalise=int((~oks[m]).sum()))
        print(n, json.dumps(out[n]), flush=True)
    json.dump(out, open(OUT / "intensive_sim.json", "w"), indent=1); np.savez_compressed(OUT / "intensive_sim_arrays.npz", which=which, vals=vals, occs=occs, shifts=shifts, hard=np.stack(HARD, 1), parts=np.stack(PARTS, 1)); env.destroy()


if __name__ == "__main__":
    main()
