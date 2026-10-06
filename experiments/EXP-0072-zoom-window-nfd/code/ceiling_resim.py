"""Ceiling step 1b: re-simulate DS-0016 pushes in Genesis from (perturbed) recorded start states, same seam as
Genesis/chain_collection.py / EXP-0059 chaos_floor.py (TRAINING_PHYSICS, set_particle_state + execute_action + update_material_state).
Per level (xy sigma mm; yaw 2 deg/mm), three experiments, each saved atomically to artifacts/ceiling/resim_<tag>.pt:
  onestep : 896 chain rows, each recorded start state perturbed independently, the recorded push re-run once
  chain   : 112 chains x 8 pushes, perturbed ONCE at chain start, the 8 recorded pushes re-run in sequence (no re-set between)
  pools   : 32 test_pools_v2 pools x 64 pushes, ONE perturbed start per pool shared by its 64 candidates
Usage: python -u ceiling_resim.py --levels 0 0 0.01 0.05 0.25 0.5 1   (repeated levels get distinct tags _r1, _r2 ...)"""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code")
from ceiling_common import *
ART = "experiments/EXP-0072-zoom-window-nfd/artifacts/ceiling/"; os.makedirs(ART, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--levels", type=float, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-envs", type=int, default=128)
    a = ap.parse_args()
    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.learned_mpc import TRAINING_PHYSICS, apply_physics, oracle_config_with_physics
    from simple_mpc.oracle_mpc import load_oracle_config
    import genesis as gs
    physics = dict(TRAINING_PHYSICS)
    cfg = oracle_config_with_physics(load_oracle_config("simple_mpc/config/config_oracle.yaml"), physics)
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = a.n_envs
    env = GenesisOracleEnv(cfg, n_envs=a.n_envs); apply_physics(env, physics); sim = env._sim
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    torch.set_default_device("cpu"); N = a.n_envs
    def setst(st): sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device)); sim.update_material_state()
    def push(pS, pE, an):
        sim.execute_action(pS.to(gs.device), pE.to(gs.device), an.to(gs.device)); sim.update_material_state()
        return sim._particle_state.detach().cpu().float().clone()
    def pad(x):
        n = len(x); k = (-n) % N
        return torch.cat([x, x[:k]]) if k else x, n

    c = load_chains(); ps, pe, an, S0, S1 = c["p_starts"].float(), c["p_stops"].float(), c["angles"].float(), c["states"].float(), c["states_"].float()
    pf = sorted(glob.glob(D + "test_pools_v2/pools_*.pt")); pools = [torch.load(f, map_location="cpu", weights_only=False) for f in pf]
    PS, PE, PA, PS0, PIDX = [], [], [], [], []
    for pi_, d in enumerate(pools):
        for p in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == p)[:, 0]
            PS.append(d["p_starts"][ix].float()); PE.append(d["p_stops"][ix].float()); PA.append(d["angles"][ix].float())
            PS0.append(d["states"][ix[0]].float()[None].expand(len(ix), -1, -1)); PIDX += [len(PIDX)] * len(ix) if False else [0] * 0
    PS, PE, PA, PS0 = [torch.cat(x) for x in (PS, PE, PA, PS0)]
    npool = len(PS) // 64; assert len(PS) == npool * 64, len(PS)
    print("rows", len(S0), "chains", len(S0) // 8, "pool rows", len(PS), flush=True)
    seen = {}
    for lvl in a.levels:
        r = seen.get(lvl, 0); seen[lvl] = r + 1; tag = f"{lvl:g}mm_r{r}"; fn = ART + f"resim_{tag}.pt"
        if os.path.exists(fn): print("skip", tag); continue
        rng = np.random.default_rng(1000 * (r + 1) + int(lvl * 1000) + a.seed); t0 = time.time(); out = {"level_mm": lvl, "rep": r}
        # --- one step
        Sp = perturb(S0, lvl, rng); res = []
        for i in range(0, len(S0), N):
            sl = slice(i, i + N); st, n = pad(Sp[sl]); setst(st); res.append(push(pad(ps[sl])[0], pad(pe[sl])[0], pad(an[sl])[0])[:n])
        out["onestep"] = torch.cat(res); print(tag, "onestep", time.time() - t0, flush=True)
        # --- chains (rows sorted chain, step)
        nc = len(S0) // 8; assert (c["step"].reshape(nc, 8) == torch.arange(8)).all()
        st0 = perturb(S0.reshape(nc, 8, 20, 7)[:, 0], lvl, rng); stp, n = pad(st0); setst(stp); traj = []
        for t in range(8):
            g = lambda x: pad(x.reshape(nc, 8, *x.shape[1:])[:, t])[0]
            traj.append(push(g(ps), g(pe), g(an))[:n])
        out["chain"] = torch.stack(traj, 1); print(tag, "chain", time.time() - t0, flush=True)
        # --- pools: one perturbed start per pool
        base = PS0.reshape(npool, 64, 20, 7)[:, 0]; pp = perturb(base, lvl, rng)[:, None].expand(-1, 64, -1, -1).reshape(-1, 20, 7).contiguous(); res = []
        for i in range(0, len(pp), N):
            sl = slice(i, i + N); st, n = pad(pp[sl]); setst(st); res.append(push(pad(PS[sl])[0], pad(PE[sl])[0], pad(PA[sl])[0])[:n])
        out["pools"] = torch.cat(res); out["time_s"] = time.time() - t0
        torch.save(out, fn + ".tmp"); os.replace(fn + ".tmp", fn); print(tag, "saved", time.time() - t0, flush=True)

if __name__ == "__main__": main()
