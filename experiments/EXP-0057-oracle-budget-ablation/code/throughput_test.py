"""EXP-0057 Step 1: measure simulated pushes/sec for the oracle's batched simulation at one
n_envs value (20 cubes, TRAINING_PHYSICS, oracle's settle steps), a few batched steps. Genesis
allows gs.init() once per process, so this is run once per n_envs value, each in its OWN
process (see run_throughput.sh). Appends its result into results/throughput.json (read-modify-
write, single writer at a time since the driver runs them sequentially).
"""
import argparse, json, sys, time
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, project_push
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose

HERE = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--n-envs", type=int, required=True)
ap.add_argument("--n-batches", type=int, default=3)
a = ap.parse_args()
n_envs = a.n_envs

rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
state0 = rows.states[(rows.slate_idx == 40).nonzero()[0, 0]].float()

result = dict(ok=False)
try:
    torch.cuda.reset_peak_memory_stats()
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = n_envs
    env = GenesisOracleEnv(cfg, n_envs=n_envs); apply_physics(env); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    st = state0[None].expand(n_envs, -1, -1).contiguous()
    t_sim = 0.0
    for b in range(a.n_batches):
        sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device))
        s, e, _ = sim.generate_action_samples(1, pile_aware=True, min_swath_particles=3)
        c = torch.cat([s[..., :2], e[..., :2]], -1).reshape(-1, 4).float().cpu()[:n_envs]
        act, _ = project_push(c)
        sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device))
        torch.cuda.synchronize()
        t0 = time.time()
        sx, sy, ex, ey, ang = action_to_pose(act.to(gs.device))
        z = torch.full_like(sx, float(sim._operation_height))
        sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang)
        sim.update_material_state()
        torch.cuda.synchronize()
        t_sim += time.time() - t0
    pushes_per_s = (a.n_batches * n_envs) / t_sim
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    reserved_mb = torch.cuda.max_memory_reserved() / 1e6
    result = dict(ok=True, pushes_per_s=pushes_per_s, t_sim=t_sim, peak_mb=peak_mb,
                  reserved_mb=reserved_mb, n_batches=a.n_batches)
    print(f"n_envs={n_envs}: {pushes_per_s:.3f} pushes/s, peak={peak_mb:.0f}MB reserved={reserved_mb:.0f}MB",
          flush=True)
    env.destroy()
except Exception as ex:
    result = dict(ok=False, error=str(ex))
    print(f"n_envs={n_envs}: FAILED {ex}", flush=True)

p = HERE / "results/throughput.json"; p.parent.mkdir(parents=True, exist_ok=True)
data = json.loads(p.read_text()) if p.exists() else {}
data[str(n_envs)] = result
tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(data, indent=1)); tmp.replace(p)
