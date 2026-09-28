"""EXP-0042 (pilot): one-step planner benchmark -- MPC-parameter ablation and planning-
time curve (DESIGN.md). For each DS-0006 state: draw a bank of pile-aware candidates
from the Genesis sampler, run `learned_mpc.plan` for every (goal, cell), then
simulate every chosen push once (rollout_candidates, full fidelity, TRAINING_PHYSICS,
banked in DS-0004 under EXP-0037's fingerprint) and score its capture against the
state's banked 128-push pool. results/trials.json is rewritten after every state;
finished states are skipped on restart.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from simple_mpc.adapters import make_occ_adapter, occ_for_scoring
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.gt_bank import GroundTruthBank
from simple_mpc.learned_mpc import (TRAINING_PHYSICS, ModelObjective, apply_physics,
                                    oracle_config_with_physics, plan, project_push)
from simple_mpc.oracle_mpc import load_oracle_config

HERE = REPO / "experiments/EXP-0042-one-step-planner-ablation"
E37 = REPO / "experiments/EXP-0037-gradient-benchmark-ds0006/artifacts/RUN-0001"
CFG = REPO / "simple_mpc/config/config_oracle.yaml"
K, BANK_PER_ENV = 32, 1024
DEFAULT = dict(budget=1.0, n_cand=64, n_restarts=8, lr=1.5e-3, cem_pop=None, cem_elite_frac=0.125)


def cells():
    out = []
    for pl in ("rank", "gd", "cem"):
        for b in (0.1, 0.3, 1.0, 3.0):
            out.append((f"{pl}_budget{b}", pl, dict(budget=b)))
        for n in (16, 256):
            out.append((f"{pl}_pool{n}", pl, dict(n_cand=n)))
        out.append((f"{pl}_default_repeat", pl, {}))
    out += [("gd_restarts1", "gd", dict(n_restarts=1)), ("gd_restarts32", "gd", dict(n_restarts=32)),
            ("gd_lr5e-4", "gd", dict(lr=5e-4)), ("gd_lr5e-3", "gd", dict(lr=5e-3)),
            ("cem_pop256", "cem", dict(cem_pop=256)),
            ("cem_elite0.06", "cem", dict(cem_elite_frac=0.06)), ("cem_elite0.25", "cem", dict(cem_elite_frac=0.25))]
    return out


def sweep_cells():
    out = [("rank_default", "rank", {})]
    for lr in (1.5e-3, 5e-3, 1.5e-2, 5e-2):
        for r in (8, 32):
            out.append((f"gd_lr{lr:g}_r{r}", "gd", dict(lr=lr, n_restarts=r)))
    for n in (64, 256, 1024):
        for el in (0.125, 0.25):
            out.append((f"cem_n{n}_e{el:g}", "cem", dict(n_cand=n, cem_pop=n, cem_elite_frac=el)))
    out += [("gd_lr0.005_r32_b0.3", "gd", dict(lr=5e-3, n_restarts=32, budget=0.3)),
            ("cem_n256_e0.25_b0.1", "cem", dict(n_cand=256, cem_pop=256, cem_elite_frac=0.25, budget=0.1))]
    return out


def lyap_fields(occ, D):
    f = occ.reshape(len(occ), -1)
    return (f @ D.reshape(len(D), -1).T) / f.sum(1, keepdim=True).clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nfd_residual_worldframe_noaug_ep43")
    ap.add_argument("--states", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--cells", default=None, help="'sweep' = addendum 1's tuning cells; default = the pilot's")
    a = ap.parse_args()
    res_p = HERE / f"results/trials_{a.tag}.json"
    res_p.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(res_p.read_text()) if res_p.exists() else {"config": vars(a), "cells": [c[0] for c in (sweep_cells() if a.cells == "sweep" else cells())],
                                                                "states": {}}
    s1 = torch.load(E37 / "stage1.pt", weights_only=False)
    s2 = torch.load(E37 / "stage2_index.pt", weights_only=False)
    goals, D = s1["goals"], s1["goal_fields"].float()
    dev = "cuda"
    Dd = D.to(dev)
    cfg = oracle_config_with_physics(load_oracle_config(str(CFG)))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    fp = {"config_sha1": hashlib.sha1(CFG.read_bytes()).hexdigest()[:12], "use_rollout_fidelity": False,
          "n_envs": K, "physics": dict(TRAINING_PHYSICS), "n_particles": 20}
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    bank = GroundTruthBank(REPO / "datasets/DS-0004-ground-truth-bank/data")
    ad = make_occ_adapter(a.model, dev, "corner")
    C = sweep_cells() if a.cells == "sweep" else cells()
    for s in a.states:
        if str(s) in out["states"]:
            continue
        t_state = time.time()
        st = s1["states0"][s].float()
        snap = {"pos": st[None, :, :3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
        env.restore_snapshot(snap)                       # sampler reads sim._particle_state
        sp, ep, _ = sim.generate_action_samples(BANK_PER_ENV, pile_aware=True, min_swath_particles=3)
        cand_bank = torch.cat([sp[..., :2], ep[..., :2]], -1).reshape(-1, 4).float().cpu()
        rng = np.random.default_rng(1000 + s)
        trials = []
        warm = ModelObjective(ad, st, Dd[0])
        plan(warm, cand_bank[:64], "gd", 0.2); plan(warm, cand_bank[:64], "cem", 0.1)   # lazy-init warm-up
        for gi in range(len(goals)):
            obj = ModelObjective(ad, st, Dd[gi])
            for name, pl, kw in C:
                p = dict(DEFAULT, **kw)
                perm = torch.from_numpy(rng.permutation(len(cand_bank)))
                n = p["n_cand"]
                ptr = [n]

                def more():
                    i = ptr[0]; ptr[0] += n
                    return cand_bank[perm[i % len(perm):i % len(perm) + n]]
                r = plan(obj, cand_bank[perm[:n]], pl, p["budget"], n_restarts=p["n_restarts"], lr=p["lr"],
                         cem_elite_frac=p["cem_elite_frac"], cem_pop=p["cem_pop"],
                         more_candidates=more if pl == "rank" else None)
                trials.append(dict(goal=goals[gi], gi=gi, cell=name, planner=pl, action=r["action"].tolist(),
                                   pred_dv=r["pred_dv"], n_evals=r["n_evals"], n_iters=r["n_iters"],
                                   time_s=r["time_s"], max_iter_s=r["max_iter_s"]))
        t_plan = time.time() - t_state
        A = project_push(torch.tensor([t["action"] for t in trials]).float())[0]
        uniq = torch.unique(A, dim=0)

        def simulate(batch):
            order = (batch[:, 2:] - batch[:, :2]).norm(dim=1).argsort()
            fin = torch.zeros(len(batch), st.shape[0], 3, device="cpu")
            for i in range(0, len(batch), K):
                o = order[i:i + K]; m = len(o)
                chunk = batch[o]
                pad = torch.cat([chunk, chunk[:1].expand(K - m, -1)]) if m < K else chunk
                pos = env.rollout_candidates(pad[:, None, :], snap, use_rollout_fidelity=False, record=False)
                fin[o] = pos.float().cpu()[:m]
            return fin
        bank.evaluate(st, uniq, "oracle_rollout_full", fp, f"EXP-0042/{a.tag}", simulate)
        hit, fin = bank.lookup(st, A, "oracle_rollout_full"); assert hit.all()
        L = s2["labels"][s]
        pool_i = [i for i, l in enumerate(L["labels"]) if l[0] == "pool"]
        hp, fin_pool = bank.lookup(st, L["actions"][pool_i], "oracle_rollout_full"); assert hp.all()
        v0 = lyap_fields(occ_for_scoring(st[None, :, :3]), D)[0]
        dv = (lyap_fields(occ_for_scoring(fin[..., :3]), D) - v0[None]).numpy()          # (N, G), cost
        dvp = (lyap_fields(occ_for_scoring(fin_pool[..., :3]), D) - v0[None]).numpy()
        for i, t in enumerate(trials):
            pool = dvp[:, t["gi"]]
            den = pool.mean() - pool.min()
            t["true_dv"] = float(dv[i, t["gi"]])
            t["capture"] = float((pool.mean() - dv[i, t["gi"]]) / den) if den > 1e-9 else None
        out["states"][str(s)] = dict(trials=trials, plan_s=t_plan, total_s=time.time() - t_state,
                                     n_unique=len(uniq))
        tmp = Path(str(res_p) + ".tmp"); tmp.write_text(json.dumps(out)); os.replace(tmp, res_p)
        caps = {}
        for t in trials:
            caps.setdefault(t["cell"], []).append(t["capture"])
        print(f"state {s}: plan {t_plan:.0f}s, total {time.time() - t_state:.0f}s; " +
              " ".join(f"{k}={np.nanmean([c for c in v if c is not None]):.2f}" for k, v in caps.items()
                       if k.endswith("budget1.0")), flush=True)
    env.destroy()
    print("wrote", res_p)


if __name__ == "__main__":
    main()
