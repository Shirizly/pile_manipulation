"""EXP-0075 per-state regret decomposition (2026-10-11 bottleneck review item 4): model ranking error vs candidate-proposal error over a whole episode.
States: recorded closed-loop states (positions + quaternions) of one cell, pushes k in --ks (e.g. 2 5 8), all 32 episodes -> 96 states.
At each state: the planner's own pile-aware bank (4,096, min_swath 3, same sampler / seed convention) and the fetch proposals (wide_planner.fetch_candidates),
all legalised (legalize_pushes_torch) and scored by the cell's model + objective (single push). Candidate set C (64): model top-24 bank, top-16 fetch, 16 random bank,
8 random fetch (bank fills when fetch is short). Every candidate is executed in Genesis from the state (all 32 envs reset to that state, 2 batches) -> true value change.
Per state: true best of C (bank-only / all), true value of the model's argmax (bank-only / all), Spearman(pred, true) on C, regret of the model's pick.
Writes results/<out>.json after every state (atomic).
usage: PYTHONPATH=. python -u regret_decomp.py --src results/fix_H1full.json --cell base1F --ks 2 5 8 --out regret_base1F"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
os.chdir(REPO)
from batched_closed_loop import goal_mask, goal_dist
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, lyap, project_push, L_MIN, L_MAX
from simple_mpc.adapters import occ_for_scoring
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose
from model.retrieval.frame import yaw_from_quat
from scipy.stats import spearmanr
import wide_planner as wp
from closed_loop import MODELS

K = 32


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--cell", required=True); ap.add_argument("--ks", type=int, nargs="+", default=[2, 5, 8])
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="ens2bal"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); out_p = REPO / "experiments/EXP-0075-closed-loop-benchmark/results" / f"{a.out}.json"
    eps = [e for e in json.loads((REPO / a.src).read_text())["episodes"] if e["cell"] == a.cell and e.get("complete")]
    states = [(e, k) for k in a.ks for e in eps]
    done = json.loads(out_p.read_text()) if out_p.exists() else {"config": vars(a), "states": []}
    have = {(s["goal"], s["start"], s["k"]) for s in done["states"]}
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))); cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    model = MODELS[a.model]()
    M, D = {}, {}

    def reset(P):
        sim.set_particle_state(P[None, :, :3].expand(K, -1, -1).contiguous().to(gs.device), P[None, :, 3:7].expand(K, -1, -1).contiguous().to(gs.device)); sim.update_material_state()
        return sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def execute(acts):
        sx, sy, ex, ey, ang = action_to_pose(acts.to(gs.device).float()); z = torch.full_like(sx, float(sim._operation_height))
        sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); sim.update_material_state()
        return sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def value(parts, g):
        occ = occ_for_scoring(parts[:, :, :3].float()).cpu(); f = occ.reshape(len(occ), -1)
        return (lyap(occ, D[g]) - (f * M[g].reshape(1, -1)).sum(1) / f.sum(1)).numpy()

    for si, (e, k) in enumerate(states):
        g = e["goal"]
        if (g, e["start"], k) in have:
            continue
        t0 = time.time()
        if g not in M:
            M[g] = torch.from_numpy(goal_mask(g)).float(); D[g] = goal_dist(g).float()
        P = torch.cat([torch.tensor(e["states"][k]), torch.tensor(e["quats"][k])], 1).float()
        p0 = reset(P); V0 = value(p0[:1], g)[0]
        torch.manual_seed(a.seed * 1_000_003 + si)
        s, e_, _ = sim.generate_action_samples(4096, pile_aware=True, min_swath_particles=3); bank = torch.cat([s[0, :, :2], e_[0, :, :2]], -1).float()
        legal = wp.make_legal_scorer(P[:, :2], yaw_from_quat(P[:, 3:7]))
        bank = project_push(bank.to(wp.DEV), L_MIN, L_MAX)[0]; bank, _, okb = legal(bank); bank = bank[okb]
        F = wp.fetch_candidates(P[:, :2].to(wp.DEV), M[g])
        if len(F):
            F = project_push(F, L_MIN, L_MAX)[0]; F, _, okf = legal(F); F = F[okf]
        obj = wp.SeqObjective(model, P, D[g], M[g], 1.0)
        cb = obj.cost(bank[:, None]); cf = obj.cost(F[:, None]) if len(F) else torch.zeros(0, device=wp.DEV)
        g_ = torch.Generator(device="cpu").manual_seed(si)
        nf_top = min(16, len(F)); nf_rand = min(8, max(0, len(F) - nf_top))
        ib_top = cb.argsort()[:24 + (16 - nf_top) + (8 - nf_rand)]
        rest_b = torch.tensor(sorted(set(range(len(bank))) - set(ib_top.tolist())), dtype=torch.long, device="cpu")
        ib_rand = rest_b[torch.randperm(len(rest_b), generator=g_, device="cpu")[:16]].to(wp.DEV)
        if_top = cf.argsort()[:nf_top]
        rest_f = torch.tensor(sorted(set(range(len(F))) - set(if_top.tolist())), dtype=torch.long, device="cpu")
        if_rand = rest_f[torch.randperm(len(rest_f), generator=g_, device="cpu")[:nf_rand]].to(wp.DEV) if nf_rand else torch.zeros(0, dtype=torch.long, device=wp.DEV)
        C = torch.cat([bank[ib_top], bank[ib_rand], F[if_top], F[if_rand]]).cpu(); pred = torch.cat([cb[ib_top], cb[ib_rand], cf[if_top], cf[if_rand]]).cpu().numpy()
        src = ["bank_top"] * len(ib_top) + ["bank_rand"] * len(ib_rand) + ["fetch_top"] * len(if_top) + ["fetch_rand"] * len(if_rand)
        true = []
        for b0 in range(0, len(C), K):
            c = C[b0:b0 + K]; n = len(c); c = torch.cat([c, c[-1:].expand(K - n, -1)]) if n < K else c
            reset(P); p1 = execute(c); true += list(value(p1[:n], g) - V0)
        true = np.array(true); isb = np.array([x.startswith("bank") for x in src]); isf = ~isb
        rec = dict(goal=g, start=e["start"], k=k, n_bank_legal=int(len(bank)), n_fetch_legal=int(len(F)), n_strays=int((~wp.inside_goal(P[:, :2], M[g])).sum()),
                   src=src, pred=pred.tolist(), true=true.tolist(), best_true_bank=float(true[isb].min()), best_true_all=float(true.min()),
                   pick_bank_true=float(true[isb][pred[isb].argmin()]), pick_all_true=float(true[pred.argmin()]), pick_all_src=src[int(pred.argmin())],
                   best_true_fetch=float(true[isf].min()) if isf.any() else None, spearman=float(spearmanr(pred, true).correlation), seconds=time.time() - t0)
        done["states"].append(rec); tmp = Path(str(out_p) + ".tmp"); tmp.write_text(json.dumps(done)); os.replace(tmp, out_p)
        print(f"state {si + 1}/{len(states)} {g}/{e['start']} k={k}: strays {rec['n_strays']}, fetch {len(F)}; best true bank {rec['best_true_bank']:+.3f} all {rec['best_true_all']:+.3f}; "
              f"model pick bank {rec['pick_bank_true']:+.3f} all {rec['pick_all_true']:+.3f} ({rec['pick_all_src']}); spearman {rec['spearman']:+.2f}; {rec['seconds']:.0f}s", flush=True)
    env.destroy(); print("wrote", out_p)


if __name__ == "__main__":
    main()
