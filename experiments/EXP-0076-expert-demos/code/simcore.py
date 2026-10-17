"""EXP-0076 simulator core: one Genesis build with E parallel envs and n cubes, where every env can be put in its own state.
Same physics / fidelity / call sequence as the EXP-0075 harness and the EXP-0050 oracle (TRAINING_PHYSICS, real settle and clearance
budgets; set_particle_state -> execute_action -> update_material_state). Legaliser constants = EXP-0075 wide_planner.make_legal_scorer."""
import sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
for p in (REPO, REPO / "experiments/EXP-0075-closed-loop-benchmark/code", REPO / "experiments/EXP-0043-batched-closed-loop/code"):
    sys.path.insert(0, str(p))


class Sim:
    def __init__(self, n, E):
        from simple_mpc.genesis_oracle import GenesisOracleEnv
        from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics
        from simple_mpc.oracle_mpc import load_oracle_config
        cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
        cfg["dataset"]["record_transitions"] = False; cfg["dataset"]["n_particles"] = n; cfg["dataset"]["num_objects"] = n; cfg["mpc"]["n_envs"] = E
        self.env = GenesisOracleEnv(cfg, n_envs=E); apply_physics(self.env); self.sim = self.env._sim
        self.sim._settle_steps = self.env._real_settle_steps; self.sim._clearance_ctrl_steps = self.env._real_clearance_steps
        import genesis as gs
        self.gs, self.n, self.E = gs, n, E
        assert self.sim._particle_state.shape[1] == n, f"sim built with {self.sim._particle_state.shape[1]} particles, wanted {n}"
        self.t_exec = 0.0; self.n_exec = 0

    def set(self, st):
        """st (k, n, 7) cpu, k <= E; envs >= k get copies of the last row."""
        st = torch.as_tensor(st).float()
        if len(st) < self.E:
            st = torch.cat([st, st[-1:].expand(self.E - len(st), -1, -1)])
        gs = self.gs; self.sim.set_particle_state(st[:, :, :3].to(gs.device).contiguous(), st[:, :, 3:7].to(gs.device).contiguous())

    def state(self):
        return self.sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def execute(self, acts):
        """acts (E, 4) world [sx, sy, ex, ey] (rows beyond the real ones: anything legal, e.g. copies). Returns (E, n, 7) cpu after settling."""
        from transforms.functional import action_to_pose
        t0 = time.time(); gs = self.gs
        sx, sy, ex, ey, ang = action_to_pose(torch.as_tensor(acts).to(gs.device).float()); z = torch.full_like(sx, float(self.sim._operation_height))
        self.sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); self.sim.update_material_state()
        self.t_exec += time.time() - t0; self.n_exec += 1
        return self.state()

    def step_from(self, states, acts):
        """Simulate push acts[i] from states[i] (k <= E rows). Returns (k, n, 7)."""
        k = len(states); self.set(states)
        A = torch.as_tensor(acts).float()
        if k < self.E:
            A = torch.cat([A, A[-1:].expand(self.E - k, -1)])
        return self.execute(A)[:k]

    def bank(self, states, m):
        """Pile-aware candidate banks (k, m, 4) cpu for each of k states (same sampler as the EXP-0075 harness)."""
        k = len(states); self.set(states)
        s, e, _ = self.sim.generate_action_samples(m, pile_aware=True, min_swath_particles=3)
        return torch.cat([s[..., :2], e[..., :2]], -1).float().cpu()[:k]

    def destroy(self):
        self.env.destroy()


def legalize(acts, state):
    """acts (N, 4) on any device; state (n, >=7) one shared state -> (acts', shift (N,), ok (N,)) on cuda. Same constants as the harness."""
    from wide_planner import make_legal_scorer
    from model.retrieval.frame import yaw_from_quat
    st = torch.as_tensor(state).float()
    return make_legal_scorer(st[:, :2], yaw_from_quat(st[:, 3:7]))(acts)
