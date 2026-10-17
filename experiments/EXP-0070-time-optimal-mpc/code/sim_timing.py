"""Measure push timing in Genesis and check it against action_time.py (EXP-0070 RUN-0001).

For K envs (same DS-0006 start state, different pushes) records the plate position after
every simulation step of execute_action (lower / sweep / lift), and separately drives a
FREE-AIR transit at the clearance height with the same trapezoidal servo the sweep uses
(the move the simulator normally teleports through). Reports, per push:
  * steps x dt per phase as the simulator runs them;
  * the time the plate first comes within 1 mm of its target (sweep and transit) vs the
    analytic trapezoid duration T(d).
Output: results/sim_timing.json (written once; the job takes < 2 min).
"""
import json, math, os, sys, time
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_time import trap_time, H_SAFE
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose

OUT = REPO / "experiments/EXP-0070-time-optimal-mpc/results/sim_timing.json"
K = 6
CORPUS = "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys"

# pushes [sx, sy, ex, ey] (m): lengths 20/40/70 mm, two directions; the transit test moves
# the plate from each push's END to the next env's START (lengths 0-90 mm)
ACTS = torch.tensor([[-0.040, -0.040, -0.020, -0.040], [-0.040, 0.020, 0.000, 0.020],
                     [-0.050, 0.040, 0.020, 0.040], [0.030, -0.045, 0.030, -0.025],
                     [0.040, 0.040, 0.040, 0.000], [0.000, -0.050, 0.000, 0.020]])
TRANSIT = [(0.0, 0.0), (0.005, 0.0), (0.02, 0.0), (-0.04, 0.03), (-0.06, -0.05), (-0.05, -0.06)]  # all ends stay inside the tray


def main():
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    rows = BinnedSlateCorpus.load(str(REPO / CORPUS)).step(0)
    st = rows.states[(rows.slate_idx == 40).nonzero()[0, 0]].float()[None].expand(K, -1, -1)
    sim.set_particle_state(st[:, :, :3].contiguous().to(gs.device), st[:, :, 3:7].contiguous().to(gs.device))
    sim.update_material_state()
    dt = float(sim._scene.dt)
    log = {"lower": [], "sweep": [], "lift": []}

    def on_step(phase, step):
        log[phase].append(sim.plate.get_pos().detach().cpu().clone())

    sx, sy, ex, ey, ang = action_to_pose(ACTS.to(gs.device).float())
    z = torch.full_like(sx, float(sim._operation_height))
    p0 = torch.stack([sx, sy, z], 1); p1 = torch.stack([ex, ey, z], 1)
    t_wall = time.time()
    sim.execute_action(p0, p1, ang, on_step=on_step)
    wall_push = time.time() - t_wall
    sw = torch.stack(log["sweep"])                      # (steps, K, 3)
    err = (sw[:, :, :2] - p1[None, :, :2].cpu()).norm(dim=-1)
    L = (ACTS[:, 2:] - ACTS[:, :2]).norm(dim=1)
    res = dict(dt=dt, operation_height=float(sim._operation_height), clearance_height=float(sim._clearance_height),
               clearance_ctrl_steps=int(sim._clearance_ctrl_steps), arrival_steps=int(sim._plate_arrival_steps),
               sweep_settle_steps=int(sim._sweep_settle_steps), wall_s_one_push_K_envs=wall_push,
               steps=dict(lower=len(log["lower"]), sweep=len(log["sweep"]), lift=len(log["lift"])), pushes=[])
    for k in range(K):
        hit = (err[:, k] < 1e-3).nonzero()
        t_hit = (int(hit[0]) + 1) * dt if len(hit) else None
        res["pushes"].append(dict(L_m=float(L[k]), T_analytic=float(trap_time(L[k:k + 1])[0]),
                                  t_within_1mm=t_hit, final_err_mm=float(err[-1, k]) * 1e3))
    # descent as simulated: plate z over the lower phase
    lz = torch.stack(log["lower"])[:, :, 2]
    res["lower_z_first_last_mm"] = [float(lz[0, 0] * 1e3), float(lz[-1, 0] * 1e3)]
    res["lower_sim_s"] = len(log["lower"]) * dt
    res["T_analytic_h_safe"] = float(trap_time(torch.tensor([H_SAFE]))[0])

    # free-air transit at clearance height with the sweep's own trapezoid servo
    sim.update_material_state()
    zc = float(sim._operation_height + sim._clearance_height)
    start = torch.stack([p1[:, 0].cpu(), p1[:, 1].cpu(), torch.full((K,), zc, device="cpu")], 1)
    goal = start.clone()
    for k, (dx, dy) in enumerate(TRANSIT):
        goal[k, 0] += dx; goal[k, 1] += dy
    sim.plate.set_pos(start.to(gs.device), zero_velocity=True)
    old = sim._horizontal_dof_fix[:, 0].clone(); sim._horizontal_dof_fix[:, 0] = zc
    tr = []
    sim.plate_velocity_translation(start.to(gs.device), goal.to(gs.device), ang,
                                   on_step=lambda s, *_: tr.append(sim.plate.get_pos().detach().cpu().clone()))
    sim._horizontal_dof_fix[:, 0] = old
    tr = torch.stack(tr); err_t = (tr[:, :, :2] - goal[None, :, :2]).norm(dim=-1)
    D = goal[:, :2].sub(start[:, :2]).norm(dim=1)
    res["transit"] = []
    for k in range(K):
        hit = (err_t[:, k] < 1e-3).nonzero()
        res["transit"].append(dict(d_m=float(D[k]), T_analytic=float(trap_time(D[k:k + 1])[0]),
                                   t_within_1mm=(int(hit[0]) + 1) * dt if len(hit) else None,
                                   final_err_mm=float(err_t[-1, k]) * 1e3,
                                   max_z_dev_mm=float((tr[:, k, 2] - zc).abs().max() * 1e3)))
    tmp = Path(str(OUT) + ".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, OUT)
    print(json.dumps(res, indent=1))
    env.destroy()


if __name__ == "__main__":
    main()
