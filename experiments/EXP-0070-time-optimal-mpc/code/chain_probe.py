"""EXP-0070 RUN-0006 (tests A, B, C): no-lift transitions, in-place rotation, chained pushes -- Genesis
ground truth only, no learned model.

States: visited states of pilot_v0b `lyap` episodes (positions + quaternions recorded), step k >= 1, so
the push that produced them (a_{k-1}) is known: replaying a_{k-1} from s_{k-1} WITHOUT lifting
reproduces the state with the plate down at a_{k-1}'s end -- the state the new framing plans from.

A  (--mode lift): 16 (s_{k-1}, a_{k-1}) pairs x 2 replicate envs. Pass 1 = standard execute_action
   (lower, sweep, lift, settle); pass 2 = lower, sweep, SETTLE WITH PLATE DOWN (record), lift, settle
   (record). Per-cube xy differences (mm) pass1 rep a vs rep b (sim noise floor), pass1 vs no-lift,
   pass1 vs no-lift-then-lift; and the in-goal mass / lyapunov differences.
B  (--mode rotate): 4 (state, a_{k-1}) pairs replayed no-lift x 8 rotations per pass, pivot in
   {centre, +end, -end} (3 passes): dyaw in {+-10, +-20, +-40, +-60} deg at omega_sim 3 rad/s, then
   settle. Per-cube displacement caused by the rotation alone (max, n moved > 1 mm, mass-moved).
C  (--mode chain): per state, two passes from the plate-down state: F = 32 free pushes (#0 = the action
   the lyap planner actually executed from s_k, #1.. = pile-aware samples; plate lifts, transits, lowers),
   C = 32 chained pushes from the current plate pose (rotate about the blade centre by dphi, sweep L in the
   rotated direction; dphi x L grid). True dV, d mass_rel, physical time t(a) (free: up + transit + down +
   sweep; chained: t_rot(dphi) + T(L), t_rot trapezoid at omega_rot / alpha_rot, see ROT), progress rate
   -dV / (t_plan + t), and J = t + etc(V + dV) - etc(V) under etc_map_v0's power law.
Checkpoint: results/chain_probe_<mode>.json rewritten after every state/pass.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batched_closed_loop import goal_dist, goal_mask
from Baselines.common.goals import mass_in_region
from simple_mpc.adapters import occ_for_scoring
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, lyap
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose
from action_time import action_time, trap_time

HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
K = 32
ROT = dict(omega_rot=math.pi, alpha_rot=20.0)      # ASSUMED real rotation limits (rad/s, rad/s^2); sweep sensitivity in analysis
OMEGA_SIM = 3.0                                     # rotation speed simulated (blade end 60 mm/s < sweep 125 mm/s)
HALF_BLADE = 0.02


def rot_time(dphi, w=ROT["omega_rot"], a=ROT["alpha_rot"]):
    return trap_time(torch.as_tensor(np.abs(dphi), dtype=torch.float64, device="cpu"), w, a).cpu().numpy()


class Sim:
    def __init__(self):
        cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
        cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
        self.env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(self.env); self.sim = self.env._sim
        import genesis as gs
        self.gs = gs
        self.sim._settle_steps = self.env._real_settle_steps; self.sim._clearance_ctrl_steps = self.env._real_clearance_steps

    def set_state(self, xyz, quat):
        d = self.gs.device
        self.sim.set_particle_state(xyz.contiguous().to(d), quat.contiguous().to(d)); self.sim.update_material_state()

    def parts(self):
        return self.sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def pose(self, acts):
        sx, sy, ex, ey, ang = action_to_pose(acts.to(self.gs.device).float())
        z = torch.full_like(sx, float(self.sim._operation_height))
        return torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang

    def lower_sweep(self, acts):
        """execute_action's lower + sweep, NO lift; returns (final_pos, angle)."""
        s = self.sim; p0, p1, ang = self.pose(acts)
        s._reaction_reset()
        s._vertical_dof_fix[:, 0] = p0[:, 0]; s._vertical_dof_fix[:, 1] = p0[:, 1]; s._vertical_dof_fix[:, 4] = ang
        lo = p0 + s._clearance_offset
        s.plate.set_pos(lo, zero_velocity=True)
        s.plate_position_translation(lo, p0, s._clearance_ctrl_steps, phase="lower")
        _, fp = s.plate_velocity_translation(p0, p1, ang)
        return fp, ang

    def lift(self, fp):
        s = self.sim
        s._vertical_dof_fix[:, 0] = fp[:, 0]; s._vertical_dof_fix[:, 1] = fp[:, 1]
        s.plate_position_translation(fp, fp + s._clearance_offset, s._clearance_ctrl_steps, phase="lift")
        s.plate.set_pos(fp + s._lift_height_tensor, zero_velocity=True)

    def rotate(self, fp, ang0, dyaw, pivot_off, hold=12):
        """Rotate the DOWN plate in place by dyaw (K,) about the point pivot_off (K,) m along the blade
        from its centre, at OMEGA_SIM; x/y driven by the servo along the arc, yaw pinned per step."""
        s = self.sim; dev = fp.device
        dyaw = dyaw.to(dev).float(); pivot_off = pivot_off.to(dev).float(); ang0 = ang0.float()
        dt = float(s._scene.dt); n = max(1, int(math.ceil(float(dyaw.abs().max()) / OMEGA_SIM / dt)))
        axis = torch.stack([torch.cos(ang0), torch.sin(ang0)], 1)
        piv = fp[:, :2] + pivot_off[:, None] * axis
        r0 = fp[:, :2] - piv
        z = fp[:, 2:3]
        for i in range(n + hold):
            f = min(1.0, (i + 1) / n)
            th = dyaw * f; c, sn = torch.cos(th), torch.sin(th)
            r = torch.stack([c * r0[:, 0] - sn * r0[:, 1], sn * r0[:, 0] + c * r0[:, 1]], 1)
            p = torch.cat([piv + r, z], 1)
            s._horizontal_dof_fix[:, -1] = ang0 + th
            s.plate.control_dofs_position_velocity(p, torch.zeros_like(p), dofs_idx_local=[0, 1, 2])
            s.plate.set_dofs_position(s._horizontal_dof_fix, dofs_idx_local=s._horizontal_dofs_local, zero_velocity=False)
            s._step_scene()
        return torch.cat([piv + r, z], 1), ang0 + dyaw


def load_states(n, kmax=6, seed=0):
    d = json.loads((HERE / "results/pilot_v0b.json").read_text())
    E = [e for e in d["episodes"] if e.get("complete") and e["cell"] == "lyap" and "quats" in e]
    rng = np.random.default_rng(seed); out = []; seen = set()
    order = [int(i) for _ in range(kmax) for i in rng.permutation(len(E))]   # cycle episodes; distinct (episode, k)
    for i in order:
        e = E[i]; k = int(rng.integers(1, kmax + 1))
        if (i, k) in seen:
            continue
        seen.add((i, k))
        out.append(dict(goal=e["goal"], start=e["start"], k=k,
                        xyz_prev=torch.tensor(e["states"][k - 1]), q_prev=torch.tensor(e["quats"][k - 1]),
                        a_prev=torch.tensor(e["actions"][k - 1]), a_next=torch.tensor(e["actions"][k]),
                        V_rec=e["values"][k]))
        if len(out) == n:
            break
    return out


def score(parts, goal):
    occ = occ_for_scoring(parts[:, :, :3]); dist = goal_dist(goal).to(occ.device)
    V = lyap(occ, dist); m = torch.from_numpy(goal_mask(goal)).to(occ.device)
    mr = mass_in_region(occ, m) / occ.reshape(len(occ), -1).sum(1) / VS.get(goal, {}).get("mass_frac_best_placement", 1.0)
    return V.cpu().numpy(), mr.cpu().numpy()


def save(name, res):
    p = HERE / f"results/chain_probe_{name}.json"; tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, p)


def mode_lift(S):
    st = load_states(16, seed=1)
    xyz = torch.stack([s["xyz_prev"] for s in st for _ in range(2)]); q = torch.stack([s["q_prev"] for s in st for _ in range(2)])
    acts = torch.stack([s["a_prev"] for s in st for _ in range(2)])
    S.set_state(xyz, q); fp, _ = S.lower_sweep(acts); S.lift(fp); S.sim.update_material_state(); std = S.parts()
    S.set_state(xyz, q); fp, _ = S.lower_sweep(acts); S.sim.update_material_state(); nolift = S.parts()
    S.lift(fp); S.sim.update_material_state(); lifted = S.parts()
    d = lambda a, b: (a[:, :, :2] - b[:, :, :2]).norm(dim=-1) * 1e3     # (K, n) mm
    res = dict(n_pairs=16, pairs={})
    for name, (a, b) in dict(rep_noise=(std[0::2], std[1::2]), std_vs_nolift=(std, nolift),
                             std_vs_nolift_then_lift=(std, lifted), nolift_vs_lifted=(nolift, lifted)).items():
        dd = d(a, b)
        res["pairs"][name] = dict(mean_mm=float(dd.mean()), p99_mm=float(dd.quantile(.99)), max_mm=float(dd.max()),
                                  frac_cubes_gt1mm=float((dd > 1).float().mean()))
    dV, dM = [], []
    for i, s in enumerate(st):
        g = s["goal"]
        Vs, Ms = score(std[2 * i:2 * i + 1], g); Vn, Mn = score(nolift[2 * i:2 * i + 1], g); Vl, Ml = score(lifted[2 * i:2 * i + 1], g)
        Vr, _ = score(std[2 * i + 1:2 * i + 2], g)
        dV.append([float((Vn - Vs)[0]), float((Vl - Vs)[0]), float((Vr - Vs)[0])]); dM.append([float((Mn - Ms)[0]), float((Ml - Ms)[0])])
    dV = np.array(dV); dM = np.array(dM)
    res["dV_abs_mean"] = dict(nolift=float(np.abs(dV[:, 0]).mean()), nolift_then_lift=float(np.abs(dV[:, 1]).mean()),
                              replicate=float(np.abs(dV[:, 2]).mean()))
    res["dmass_rel_abs_mean"] = dict(nolift=float(np.abs(dM[:, 0]).mean()), nolift_then_lift=float(np.abs(dM[:, 1]).mean()))
    # reference: how much a push itself moves cubes
    push = d(std, xyz.cpu().float())
    res["push_itself"] = dict(mean_mm=float(push.mean()), frac_cubes_gt1mm=float((push > 1).float().mean()))
    save("lift", res); print(json.dumps(res, indent=1))


def mode_rotate(S):
    st = load_states(4, seed=2)
    dy = torch.tensor([10, -10, 20, -20, 40, -40, 60, -60], dtype=torch.float32) * math.pi / 180
    res = dict(dyaw_deg=(dy * 180 / math.pi).tolist(), omega_sim=OMEGA_SIM, by_pivot={})
    for piv_name, off in (("centre", 0.0), ("end_plus", HALF_BLADE), ("end_minus", -HALF_BLADE)):
        xyz = torch.stack([s["xyz_prev"] for s in st for _ in range(8)]); q = torch.stack([s["q_prev"] for s in st for _ in range(8)])
        acts = torch.stack([s["a_prev"] for s in st for _ in range(8)])
        S.set_state(xyz, q); fp, ang = S.lower_sweep(acts); S.sim.update_material_state(); before = S.parts()
        S.rotate(fp, ang, dy.repeat(4), torch.full((K,), off)); S.sim.update_material_state(); after = S.parts()
        dd = (after[:, :, :2] - before[:, :, :2]).norm(dim=-1) * 1e3      # (K, n)
        rows = []
        for j in range(8):
            sel = dd[j::8]
            dVs = []
            for i, s in enumerate(st):
                Vb, Mb = score(before[8 * i + j:8 * i + j + 1], s["goal"]); Va, Ma = score(after[8 * i + j:8 * i + j + 1], s["goal"])
                dVs.append([float(Va - Vb), float(Ma - Mb)])
            dVs = np.array(dVs)
            rows.append(dict(dyaw_deg=float(dy[j] * 180 / math.pi), max_mm=float(sel.max()), mean_moved_cubes_gt1mm=float((sel > 1).float().sum(1).mean()),
                             sum_disp_mm=float(sel.sum(1).mean()), dV_abs=float(np.abs(dVs[:, 0]).mean()), dmass_abs=float(np.abs(dVs[:, 1]).mean())))
        res["by_pivot"][piv_name] = rows
        save("rotate", res); print(piv_name, json.dumps(rows), flush=True)
    # reference: a 20 mm push from the same plate-down states (via pass with lift; displacement of cubes)
    print(json.dumps(res, indent=1))


def chained_grid():
    g = [(dp, L) for dp in (-45, -30, -15, 0, 15, 30, 45) for L in (0.010, 0.020, 0.035, 0.050)]
    g += [(-60, 0.02), (60, 0.02), (-60, 0.035), (60, 0.035)]
    return (torch.tensor([x[0] for x in g], dtype=torch.float32, device="cpu") * math.pi / 180,
            torch.tensor([x[1] for x in g], dtype=torch.float32, device="cpu"))


def mode_chain(S, n_states, t_plan):
    etcm = json.loads((HERE / "results/etc_map_v0.json").read_text())
    st = load_states(n_states, seed=3)
    out_p = HERE / "results/chain_probe_chain.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else dict(rot=ROT, t_plan=t_plan, states=[])
    done = {(r["goal"], r["start"], r["k"]) for r in res["states"]}
    dphi, L = chained_grid()
    for s in st:
        if (s["goal"], s["start"], s["k"]) in done:
            continue
        g = s["goal"]
        xyz = s["xyz_prev"][None].expand(K, -1, -1); q = s["q_prev"][None].expand(K, -1, -1)
        a_prev = s["a_prev"][None].expand(K, -1)
        # ---- pass F: free pushes (plate lifts first)
        S.set_state(xyz, q); fp, ang = S.lower_sweep(a_prev); S.sim.update_material_state(); base = S.parts()
        V0, M0 = score(base[:1], g)
        sm, em, _ = S.sim.generate_action_samples(K - 1, pile_aware=True, min_swath_particles=3)
        free = torch.cat([s["a_next"][None].float().cpu(), torch.cat([sm[0, :, :2], em[0, :, :2]], -1).float().cpu()], 0)
        S.lift(fp)
        p0, p1, an = S.pose(free); S.sim.execute_action(p0, p1, an); S.sim.update_material_state()
        after_f = S.parts(); Vf, Mf = score(after_f, g)
        tf = action_time(free.double(), s["a_prev"][2:4].double().cpu(), False).cpu().numpy()
        # ---- pass C: chained pushes from the current plate pose
        S.set_state(xyz, q); fp, ang = S.lower_sweep(a_prev); S.sim.update_material_state()
        d0 = (s["a_prev"][2:4] - s["a_prev"][0:2]); phi0 = math.atan2(float(d0[1]), float(d0[0]))
        cur = fp[:, :2].detach().cpu().float()
        ends = cur + L[:, None] * torch.stack([torch.cos(phi0 + dphi), torch.sin(phi0 + dphi)], 1)
        chain = torch.cat([cur, ends], 1)
        _, _, ang_new = S.pose(chain)
        dyaw = torch.remainder(ang_new.cpu() - ang.cpu() + math.pi / 2, math.pi) - math.pi / 2     # blade is pi-symmetric
        pos, _ = S.rotate(fp, ang, dyaw, torch.zeros(K))
        p0c = pos.clone(); p1c = torch.cat([ends.to(pos.device), pos[:, 2:3]], 1)
        S.sim._horizontal_dof_fix[:, -1] = ang_new
        S.sim.plate_velocity_translation(p0c, p1c, ang_new); S.sim.update_material_state()
        after_c = S.parts(); Vc, Mc = score(after_c, g)
        tc = rot_time(dphi.numpy()) + trap_time(L.double()).cpu().numpy()
        f = etcm[g]; f = f["pass2"] if f["n_completed"] < 0.5 * f["n_episodes"] else f["pass1"]
        c, p = f["power"]["c"], f["power"]["p"]
        etc = lambda V: c * np.clip(V, 1e-6, None) ** p
        rec = dict(goal=g, start=s["start"], k=s["k"], V0=float(V0[0]), M0=float(M0[0]),
                   free=dict(dV=(Vf - V0).tolist(), dM=(Mf - M0).tolist(), t=tf.tolist(), actions=free.tolist()),
                   chain=dict(dV=(Vc - V0).tolist(), dM=(Mc - M0).tolist(), t=tc.tolist(), dphi_deg=(dphi * 180 / math.pi).tolist(),
                              L=L.tolist()))
        for kind, dV, t in (("free", Vf - V0, tf), ("chain", Vc - V0, tc)):
            rec[kind]["J"] = (t + etc(V0 + dV) - etc(V0)).tolist()
            rec[kind]["rate"] = (-(dV) / (t_plan + t)).tolist()
        res["states"].append(rec); save("chain", res)
        sp = HERE / "results/chain_probe_states"; sp.mkdir(exist_ok=True)       # for model evaluation (chain_model_eval.py)
        torch.save(dict(prev=torch.cat([s["xyz_prev"], s["q_prev"]], -1), a_prev=s["a_prev"], base=base[0], free=free,
                        after_free=after_f, chain=chain, after_chain=after_c, goal=g),
                   sp / f"{g}_s{s['start']}_k{s['k']}.pt")
        bf, bc = float(np.min(Vf - V0)), float(np.min(Vc - V0))
        print(f"{g:11s} s{s['start']} k{s['k']}: best dV free {bf:+.4f} (planner {float(Vf[0] - V0[0]):+.4f}) chain {bc:+.4f} | "
              f"best J free {min(rec['free']['J']):+.2f} chain {min(rec['chain']['J']):+.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["lift", "rotate", "chain"])
    ap.add_argument("--n-states", type=int, default=16)
    ap.add_argument("--t-plan", type=float, default=0.5)
    a = ap.parse_args()
    S = Sim()
    {"lift": lambda: mode_lift(S), "rotate": lambda: mode_rotate(S), "chain": lambda: mode_chain(S, a.n_states, a.t_plan)}[a.mode]()
    S.env.destroy()


if __name__ == "__main__":
    main()
