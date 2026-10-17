"""EXP-0075 layout regret decomposition (2026-10-12, LOG 'LAYOUT REGRET DECOMPOSITION'; results/layout_regret/). Adapted from regret_decomp.py.
States: recorded visited states of results/layout_lam/w{15,10}_lam2.json (cells lw_w15_lam2 / lw_w10_lam2), pushes k in --ks, starts in --starts (state k = BEFORE push k).
Per state ~96 simulated candidates (3 batches of 32): model top-24 (planner pool + CEM samples, plan_seq re-run with the cell's config), 16 random bank, OT/fetch rows top-16 + 8 random
+ OT rows aimed at the 3 worst tail cubes, the recorded executed push, local Gaussian refinement around the best-by-true. Metrics per candidate: true change of Hungarian transport to the
20 uniform_layout slots (mm), true skeleton coverage change, model objective cost, predicted vs true Sinkhorn layout change, per-cube slot distances (fixed pre-push assignment).
usage: PYTHONPATH=. python -u layout_regret.py --out layout_regret/states.json"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
os.chdir(REPO)
from batched_closed_loop import goal_mask, goal_dist
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, project_push, L_MIN, L_MAX
from simple_mpc.adapters import occ_for_scoring
from simple_mpc.oracle_mpc import load_oracle_config
from simple_mpc.value_functions import uniform_layout
from transforms.functional import action_to_pose
from model.retrieval.frame import yaw_from_quat
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import wide_planner as wp
from closed_loop import MODELS
from layout_term import LayoutTransport
from analyse_fixes import coverage

K = 32; NTOT = 96
SRCS = {"w15": "experiments/EXP-0075-closed-loop-benchmark/results/layout_lam/w15_lam2.json", "w10": "experiments/EXP-0075-closed-loop-benchmark/results/layout_lam/w10_lam2.json"}


def hung(xy, S):
    D = np.linalg.norm(xy[:, None] - S[None], axis=-1); r, c = linear_sum_assignment(D); return 1000 * D[r, c].mean(), c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="layout_regret/states.json"); ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--starts", type=int, nargs="+", default=[40, 42, 44, 46]); ap.add_argument("--model", default="ens2bal"); ap.add_argument("--max-states", type=int, default=10 ** 9)
    a = ap.parse_args(); out_p = REPO / "experiments/EXP-0075-closed-loop-benchmark/results" / a.out; out_p.parent.mkdir(parents=True, exist_ok=True)
    EP = {}
    for w, p in SRCS.items():
        for e in json.loads((REPO / p).read_text())["episodes"]:
            if e.get("complete") and e["start"] in a.starts:
                EP[(w, e["goal"], e["start"])] = e
    order = [(w, g, s, k) for s in sorted(a.starts) for L in "OTSX" for w in SRCS for g in [f"letter_{L}_{w}"] for k in a.ks if (w, g, s) in EP]   # start-major so any prefix is balanced over letters / widths / pushes
    done = json.loads(out_p.read_text()) if out_p.exists() else {"config": vars(a), "states": []}
    have = {(s["w"], s["goal"], s["start"], s["k"]) for s in done["states"]}
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))); cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    model = MODELS[a.model](); M, D, SL, LT = {}, {}, {}, {}

    def reset(P):
        sim.set_particle_state(P[None, :, :3].expand(K, -1, -1).contiguous().to(gs.device), P[None, :, 3:7].expand(K, -1, -1).contiguous().to(gs.device)); sim.update_material_state()
        return sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def execute(acts):
        sx, sy, ex, ey, ang = action_to_pose(acts.to(gs.device).float()); z = torch.full_like(sx, float(sim._operation_height))
        sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); sim.update_material_state()
        return sim._particle_state[:, :, :7].detach().cpu().float().clone()

    n_done = 0
    for si, (w, g, st, k) in enumerate(order):
        if (w, g, st, k) in have:
            continue
        if n_done >= a.max_states:
            break
        t0 = time.time(); e = EP[(w, g, st)]
        if g not in M:
            M[g] = torch.from_numpy(goal_mask(g)).float(); D[g] = goal_dist(g).float(); SL[g] = uniform_layout(M[g].numpy(), 20)
            LT[g] = LayoutTransport(SL[g], M[g].numpy(), lam=2.0); LT[g].n_free = 0
        P = torch.cat([torch.tensor(e["states"][k], device="cpu"), torch.tensor(e["quats"][k], device="cpu")], 1).float(); S = SL[g]
        reset(P); xy0 = P[:, :2].numpy(); H0, asg = hung(xy0, S); dB = 1000 * np.linalg.norm(xy0 - S[asg], axis=1); cov0 = coverage(xy0, g)
        torch.manual_seed(1_000_003 * 7 + si)
        s_, e_, _ = sim.generate_action_samples(4096, pile_aware=True, min_swath_particles=3); bank = torch.cat([s_[0, :, :2], e_[0, :, :2]], -1).float()
        xy_t = P[:, :2].to(wp.DEV); legal = wp.make_legal_scorer(P[:, :2], yaw_from_quat(P[:, 3:7]))
        obj = wp.SeqObjective(model, P, D[g], M[g], 1.0, crowd=LT[g]); torch.manual_seed(si + 17)
        F = wp.fetch_candidates(xy_t, M[g]); OTR, OTC = wp.ot_candidates(xy_t, M[g], S)
        out = wp.plan_seq(obj, bank, "cem", 1, n_pool=64, pop=64, extra=F, n_extra=64, pool_mode="ot", ot=(OTR, OTC), cube_xy=xy_t, legal=legal, gd_steps=20, topk=64)
        def clean(rows):
            if len(rows) == 0:
                return rows
            r = project_push(rows.to(wp.DEV).float(), L_MIN, L_MAX)[0]; r, _, ok = legal(r); return r[ok]
        bankp = clean(bank.to(wp.DEV)); Fp = clean(F) if len(F) else F.reshape(0, 4)
        # cand groups
        cand, src, grp = [], [], []
        def add(rows, s_, g_):
            for r in rows.cpu():
                cand.append(r); src.append(s_); grp.append(g_)
        ts, tc = out["top_seqs"][:, 0].to(wp.DEV), out["top_costs"]; fin = torch.isfinite(tc); ts = ts[fin.to(wp.DEV)]
        seen, keep = set(), []
        for i, r in enumerate(torch.cat([out["action"][None].to(wp.DEV), ts])):
            key = tuple(np.round(r.cpu().numpy(), 4))
            if key not in seen:
                seen.add(key); keep.append(r)
            if len(keep) >= 24:
                break
        A = torch.stack(keep)
        rows_all = torch.cat([OTR.to(wp.DEV), Fp, bankp]); lab_all = ["ot"] * len(OTR) + ["fetch"] * len(Fp) + ["bank"] * len(bankp)
        for r in A:
            d = (rows_all - r[None]).norm(dim=1); j = int(d.argmin()); cand.append(r.cpu()); src.append(lab_all[j] if float(d[j]) < 0.003 else "refined_" + lab_all[j]); grp.append("top")
        # random bank
        gen = torch.Generator().manual_seed(si); rb = bankp[torch.randperm(len(bankp), generator=gen, device="cpu")[:16].to(wp.DEV)]; add(rb, "bank", "randbank")
        # OT / fetch pool rows
        OTc = clean(OTR) if len(OTR) else OTR.reshape(0, 4); OF = torch.cat([OTc, Fp]); olab = ["ot"] * len(OTc) + ["fetch"] * len(Fp)
        if len(OF):
            co = obj.cost(OF[:, None]); top = co.argsort()[:16]; rest = torch.tensor(sorted(set(range(len(OF))) - set(top.tolist())), dtype=torch.long, device="cpu")
            rnd = rest[torch.randperm(len(rest), generator=gen, device="cpu")[:8]] if len(rest) else rest
            for i in top.tolist(): cand.append(OF[i].cpu()); src.append(olab[i]); grp.append("ofTop")
            for i in rnd.tolist(): cand.append(OF[i].cpu()); src.append(olab[i]); grp.append("ofRand")
        # tail-targeted OT rows
        tail = [i for i in np.argsort(-dB)[:3] if dB[i] > 10]
        if len(OTR) and tail:
            sel = torch.isin(OTC.cpu(), torch.tensor(tail, device="cpu")); add(clean(OTR[sel.to(OTR.device)]), "ot", "tailOT")
        # executed
        ex = torch.tensor(e["actions"][k], device="cpu").float(); cand.append(ex); src.append("exec"); grp.append("exec")
        nb = len(cand)
        def evaluate(rows):
            true_xy, pr = [], []
            for b0 in range(0, len(rows), K):
                c = rows[b0:b0 + K]; n = len(c); c = torch.cat([c, c[-1:].expand(K - n, -1)]) if n < K else c
                reset(P); p1 = execute(c); true_xy.append(p1[:n, :, :2].numpy()); pr.append(p1[:n])
            return np.concatenate(true_xy), torch.cat(pr)
        Cmat = torch.stack(cand).float(); XY, PP = evaluate(Cmat)
        def metrics(XY_):
            return [(hung(x, S)[0], coverage(x, g)) for x in XY_]
        mt = metrics(XY); gain = np.array([H0 - m[0] for m in mt])
        # refinement around best-by-true (excluding nothing)
        nref = max(8, ((nb + 31) // 32 + (0 if (nb % 32) else 1)) * 32 - nb) if False else max(8, NTOT - nb)
        nref = (-(-(nb + nref) // 32)) * 32 - nb
        base = Cmat[int(gain.argmax())]; sig = torch.cat([torch.full((nref // 2, 1), 0.002, device="cpu"), torch.full((nref - nref // 2, 1), 0.004, device="cpu")])
        ref = clean((base[None] + sig * torch.randn(nref, 4, device="cpu")).to(wp.DEV)).cpu(); ref = ref[: nref]
        if len(ref):
            XY2, PP2 = evaluate(ref); XY = np.concatenate([XY, XY2]); Cmat = torch.cat([Cmat, ref]); mt = mt + metrics(XY2)
            src += ["refine"] * len(ref); grp += ["refine"] * len(ref)
        gain = np.array([H0 - m[0] for m in mt]); dcov = np.array([m[1] - cov0 for m in mt])
        cost = obj.cost(Cmat.to(wp.DEV)[:, None]).cpu().numpy()
        # predicted vs true Sinkhorn layout change (lam 1 units)
        lt1 = LayoutTransport(S, M[g].numpy(), lam=1.0); occ0m = obj.model.occ0
        with torch.no_grad():
            predocc = torch.cat([obj.model._predict(Cmat[i:i + 128, None, :2].to(wp.DEV), Cmat[i:i + 128, None, 2:].to(wp.DEV))[-1] for i in range(0, len(Cmat), 128)])
            lay_pred = (lt1.raw(predocc) - lt1.raw(occ0m)).cpu().numpy()
            tocc = occ_for_scoring(torch.from_numpy(XY.astype(np.float32)).new_zeros(0)) if False else None
            true_occ = torch.cat([occ_for_scoring(torch.cat([torch.from_numpy(XY[i:i + 128]).float(), P[None, :, 2:3].expand(len(XY[i:i + 128]), -1, -1)], 2)).to(wp.DEV) for i in range(0, len(XY), 128)])
            lay_true = (lt1.raw(true_occ) - lt1.raw(occ_for_scoring(P[None, :, :3]).to(wp.DEV))).cpu().numpy()
        dA = np.stack([1000 * np.linalg.norm(x - S[asg], axis=1) for x in XY]); moved = (np.linalg.norm(XY - xy0[None], axis=2) > 0.002).sum(1)
        rec = dict(w=w, goal=g, start=st, k=k, H0=H0, cov0=cov0, dB=np.round(dB, 2).tolist(), n_tail=int((dB > 10).sum()), src=src, grp=grp, gain=np.round(gain, 3).tolist(), dcov=np.round(dcov, 4).tolist(),
                   cost=cost.tolist(), lay_pred=lay_pred.tolist(), lay_true=lay_true.tolist(), moved=moved.tolist(), dA=np.round(dA, 2).tolist(), acts=Cmat.numpy().round(5).tolist(),
                   rec_next_gain=float(H0 - hung(np.array(e["states"][k + 1])[:, :2], S)[0]), seconds=time.time() - t0, n_bank_legal=len(bankp))
        done["states"].append(rec); tmp = Path(str(out_p) + ".tmp"); tmp.write_text(json.dumps(done)); os.replace(tmp, out_p); n_done += 1
        ie = grp.index("exec"); nr = [i for i, x in enumerate(grp) if x != "refine" and x != "exec"]; pk = nr[int(np.argmin(cost[nr]))]
        print(f"{si + 1}/{len(order)} {w}/{g}/{st} k={k}: tail {rec['n_tail']}, ncand {len(src)}; exec {gain[ie]:+.2f} (replay vs rec {rec['rec_next_gain']:+.2f}), pick {gain[pk]:+.2f}, best-in-set {gain[nr].max():+.2f}, best all {gain.max():+.2f} ({src[int(gain.argmax())]}); {rec['seconds']:.0f}s", flush=True)
    env.destroy(); print("wrote", out_p)


if __name__ == "__main__":
    main()
