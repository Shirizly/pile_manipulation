"""EXP-0048 check: does apply_physics change the built simulator? Reads geom
friction / link mass back after each switch, and rolls out 8 pushes from DS-0006
slate 100 under P_train (x2), P_binned and an extreme low-friction set."""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent))
from sensitivity import CORPUS, CFG, P_TRAIN, P_BINNED, sample_actions, save_atomic
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics
from simple_mpc.oracle_mpc import load_oracle_config
from Genesis.binned_slate_dataset import BinnedSlateCorpus
N = 8
tr = BinnedSlateCorpus.load(REPO / CORPUS).step(0)
st = tr.states[int((tr.slate_idx == 100).nonzero()[0, 0])].float().cpu()
A = sample_actions(st, np.random.default_rng(1000 + 100))[:N]
cfg = oracle_config_with_physics(load_oracle_config(str(CFG)), P_TRAIN)
cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = N
env = GenesisOracleEnv(cfg, n_envs=N)
import genesis as gs
snap = {"pos": st[None, :, :3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
LOW = dict(particle_friction=0.02, particle_density=450.0, box_friction=0.02, settle_steps=3000)
HEAVY = dict(particle_friction=0.7, particle_density=4000.0, box_friction=0.5, settle_steps=3000)
out, fins = {}, {}
for name, ph in [("train", P_TRAIN), ("train2", P_TRAIN), ("binned", P_BINNED), ("low_friction", LOW), ("heavy", HEAVY)]:
    apply_physics(env, ph); env._real_settle_steps = ph["settle_steps"]
    p = env._sim.material[0]; b = env._sim.box_parts["ground_plate"]
    rec = {"cube_geom_friction": float(p.geoms[0].friction), "floor_geom_friction": float(b.geoms[0].friction),
           "cube_mass": float(p.get_mass())}
    fin = env.rollout_candidates(A[:, None, :], snap, use_rollout_fidelity=False, record=False).float().cpu()
    fins[name] = fin
    d = (fin[..., :2] - fins["train"][..., :2]).norm(dim=-1) * 1e3
    rec["max_cube_diff_vs_train_mm"] = float(d.max()); rec["mean_cube_diff_vs_train_mm"] = float(d.mean())
    out[name] = rec; print(name, rec, flush=True)
    save_atomic(out, REPO / "experiments/EXP-0048-physics-sensitivity/results/verify_switch.json")
env.destroy()
