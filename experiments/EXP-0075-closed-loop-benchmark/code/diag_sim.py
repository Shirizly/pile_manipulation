"""EXP-0075 diagnosis D4 (small simulator test): at the 32 start states (letters O/T/S/X w20 x starts 40-47, full push range, legal pushes)
 (a) plan with CEM at H = 1, 2, 4 (same settings as the benchmark), execute the ENTIRE selected sequence open loop (each push legalised), and record the TRUE value after every push next to the model's predicted value and occupancy;
 (b) take the model's top-20 sequences of each plan, execute each in the simulator and compare the model's ranking of them with the true outcome (Spearman per start, regret of the model's pick among the 20).
value = lyapunov - 1.0 * in-goal fraction (the planner's objective). Saves results/diag_sim.json and results/diag_sim_arrays.npz (predicted / true occupancies for the figures).
usage: PYTHONPATH=. python -u diag_sim.py [--steps-only a]"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask, goal_dist
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, lyap
from simple_mpc.adapters import occ_for_scoring
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose
from scipy.stats import spearmanr
import wide_planner as wp
K = 32; R74 = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/"; OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"
GOALS = ["letter_O_w20", "letter_T_w20", "letter_S_w20", "letter_X_w20"]; STARTS = list(range(40, 48)); TOPK = 20
specs = [(g, s) for g in GOALS for s in STARTS]


def main():
    rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
    starts = {s: rows.states[(rows.slate_idx == s).nonzero()[0, 0]].float() for s in STARTS}
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))); cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    model = wp.WideEns([wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")], balance=True)
    dists = {g: goal_dist(g) for g in GOALS}; masks = {g: torch.from_numpy(goal_mask(g)).float() for g in GOALS}

    def reset():
        st = torch.stack([starts[s] for _, s in specs]); sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device)); sim.update_material_state()
        return sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def execute(acts):
        sx, sy, ex, ey, ang = action_to_pose(acts.to(gs.device).float()); z = torch.full_like(sx, float(sim._operation_height))
        sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); sim.update_material_state()
        return sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def legalize(acts):
        from Genesis.action_sampling import legalize_pushes
        from model.retrieval.frame import yaw_from_quat
        from simple_mpc.adapters import OCC_BOUNDS
        from simple_mpc.learned_mpc import XY_MARGIN
        st = sim._particle_state.detach().cpu().float()
        return legalize_pushes(acts, st[:, :, :2], yaw_from_quat(st[:, :, 3:7]), 0.0025, 0.02, 0.001, box=(OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN))

    def true_value(parts):          # (K,n,3+) -> (K,) value of the soft-truth occupancy, per env's own goal
        occ = occ_for_scoring(parts[:, :, :3].float()).cpu(); f = occ.reshape(K, -1)
        return np.array([float(lyap(occ[k:k + 1], dists[specs[k][0]])[0] - (f[k] * masks[specs[k][0]].reshape(-1)).sum() / f[k].sum()) for k in range(K)])

    def run_seq(seqs, H):           # seqs (K,H,4): execute open loop, legalise each push; returns values (K,H+1), occupancies (K,H+1,64,64), shifts (K,H)
        p = reset(); vals = [true_value(p)]; occs = [occ_for_scoring(p[:, :, :3]).numpy().astype(np.float16)]; sh = []
        for j in range(H):
            a, shift, ok = legalize(seqs[:, j]); sh.append(shift.cpu().numpy()); p = execute(a); vals.append(true_value(p)); occs.append(occ_for_scoring(p[:, :, :3]).cpu().numpy().astype(np.float16))
        return np.stack(vals, 1), np.stack(occs, 1), np.stack(sh, 1)

    # candidate bank at the start states (one pile-aware draw, shared by the three horizons)
    torch.manual_seed(0); p0 = reset(); s, e, _ = sim.generate_action_samples(4096, pile_aware=True, min_swath_particles=3); bank = torch.cat([s[..., :2], e[..., :2]], -1).float().cpu()
    res = {}; arrays = {}
    for H in (1, 2, 4):
        t0 = time.time(); seqs, tops, topc, pred_vals, pred_occ, pcost = [], [], [], [], [], []
        for k in range(K):
            torch.manual_seed(1000 + k); obj = wp.SeqObjective(model, p0[k], dists[specs[k][0]], masks[specs[k][0]], 1.0)
            out = wp.plan_seq(obj, bank[k], "cem", H, topk=TOPK); seqs.append(out["seq"]); tops.append(out["top_seqs"]); topc.append(out["top_costs"]); pcost.append(out["cost"])
            sq = out["seq"].to(wp.DEV)[None]; preds = model.predict(sq[..., :2], sq[..., 2:])
            pv = [float(obj._value(pr)[0] - obj.v0) for pr in preds]; pred_vals.append(pv); pred_occ.append(torch.cat(preds).cpu().numpy().astype(np.float16))
        seqs = torch.stack(seqs); tops = torch.stack(tops); topc = torch.stack(topc); plan_s = time.time() - t0
        v, o, sh = run_seq(seqs, H); tv = v - v[:, :1]
        pv = np.array(pred_vals)                                           # predicted change of value after each push of the selected sequence
        r = dict(H=H, plan_s=plan_s, pred_dvalue=pv.mean(0).tolist(), true_dvalue=tv[:, 1:].mean(0).tolist(), bias_pred_minus_true=(pv - tv[:, 1:]).mean(0).tolist(), shifted_pushes=int((sh > 0).sum()))
        # (b) ranking of the model's top-20 sequences: execute each
        true_term = np.zeros((K, TOPK));
        for c in range(TOPK):
            vv, _, _ = run_seq(tops[:, c], H); true_term[:, c] = vv[:, -1] - vv[:, 0]
        pc = topc.numpy(); sp = [spearmanr(pc[k], true_term[k])[0] for k in range(K) if np.std(pc[k]) > 1e-9 and np.std(true_term[k]) > 1e-9]
        r.update(top20_spearman_mean=float(np.nanmean(sp)), top20_pred_mean=float(pc.mean()), top20_true_mean=float(true_term.mean()), top20_bias=float((pc - true_term).mean()),
                 model_pick_true=float(true_term[:, 0].mean()), best_of_20_true=float(true_term.min(1).mean()), regret_model_pick_vs_best_of_20=float((true_term[:, 0] - true_term.min(1)).mean()))
        res[f"H{H}"] = r; print(json.dumps(r), flush=True)
        arrays.update({f"H{H}_seq": seqs.numpy(), f"H{H}_pred_occ": np.stack(pred_occ), f"H{H}_true_occ": o, f"H{H}_true_val": v, f"H{H}_pred_val": pv, f"H{H}_top_true": true_term, f"H{H}_top_pred": pc})
        json.dump(res, open(OUT / "diag_sim.json", "w"), indent=1); np.savez_compressed(OUT / "diag_sim_arrays.npz", start_occ=occ_for_scoring(p0[:, :, :3]).cpu().numpy().astype(np.float16), goals=np.array([g for g, _ in specs]), starts=np.array([s for _, s in specs]), **arrays)
    env.destroy()


if __name__ == "__main__":
    main()
