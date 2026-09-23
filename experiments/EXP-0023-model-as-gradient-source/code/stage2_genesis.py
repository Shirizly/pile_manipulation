"""EXP-0023 stage 2 (Genesis): execute every arm's a_rank / a_grad and run the
Genesis-CEM oracle, from the SAME 10 slate states stage 1 optimised from.

Per state:
  1. restore the slate's particle state as a frozen snapshot;
  2. CEM (`simple_mpc.sampling_optimizers.CEMOptimizer`, NOT MPPI -- one
     optimiser, held fixed) with Genesis itself as the model: iteration 0's
     population is a random 32-subset of THE SAME seed pool stage 1 ranked, so
     the oracle starts where the arms start; later iterations sample from the
     refitted Gaussian.  Candidates are pushed through the identical legality
     projection stage 1 uses, so the oracle searches the same legal set;
  3. one final FULL-FIDELITY batch of 32 executions: every arm's a_rank and
     a_grad, the CEM winner, and the remaining slots filled with seed-pool rows
     whose true dv is already cached by `scripts/probes/binned_pool_cache.py`
     -- the known-number reproduction check for this whole pipeline.

dv = lyapunov(occ_after) - lyapunov(occ_before), a COST (lower is better),
computed with `simple_mpc.adapters.occ_from_particles`, i.e. the identical
occupancy convention `binned_pool_cache.py` used.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0023-model-as-gradient-source/code"))

from control_utility_test import lyapunov, lyapunov_weights
from simple_mpc.adapters import occ_from_particles, OCC_GRID
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.oracle_mpc import load_oracle_config
from simple_mpc.sampling_optimizers import CEMOptimizer
from stage1_optimise import project, L_MIN, L_MAX, XY_MARGIN
from utils import git_provenance

BOX = 0.064 - XY_MARGIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--cem-iters", type=int, default=4)
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s1 = torch.load(args.stage1, map_location="cpu", weights_only=False)
    arms = list(s1["arms"].keys())
    slates = s1["slates"]
    K = args.n_envs

    cfg = load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))
    cfg["dataset"]["record_transitions"] = False
    cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K)
    import genesis as gs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dw = lyapunov_weights((OCC_GRID, OCC_GRID), args.goal, dev)

    def dv_of(pos_world, v0):
        occ = occ_from_particles(pos_world.float(), dev)
        return (lyapunov(occ, dw) - v0.to(dev)).detach().cpu()

    lo = np.array([-BOX] * 4, dtype=np.float32)
    hi = np.array([+BOX] * 4, dtype=np.float32)
    torch.manual_seed(args.seed)

    out = {"arms": arms, "slates": slates, "dv": {}, "cem": {}, "validation": {},
           "config": vars(args) | {"box": BOX, "L_MIN": L_MIN, "L_MAX": L_MAX,
                                   "dv": "value(after) - value(before), a COST",
                                   "optimizer": "cem"},
           "provenance": git_provenance()}

    for s in slates:
        t0 = time.time()
        st = s1["states0"][s].float()
        snap = {"pos": st[None, :, 0:3].to(gs.device),
                "quat": st[None, :, 3:7].to(gs.device)}
        env.restore_snapshot(snap)
        occ0 = occ_from_particles(st[None], dev)
        v0 = lyapunov(occ0, dw)
        # state-restore fidelity: what the env holds must match what we asked for
        back = env.current_particles_world().float().cpu()
        restore_err = float((back[0] - st[:, 0:3]).abs().max())

        pool = s1["pool_actions"][s].to(dev)

        # ── CEM oracle ────────────────────────────────────────────────────
        cem = CEMOptimizer(1, lo, hi, device=dev, **cfg["mpc"]["cem"])
        cem.reset(pool.mean(dim=0)[None])
        cem_hist = []
        for it in range(args.cem_iters):
            if it == 0:
                idx = torch.randperm(pool.shape[0])[:K]
                cand = pool[idx][:, None, :].clone()
                if cand.shape[0] < K:
                    cand = cand.repeat((K // cand.shape[0]) + 1, 1, 1)[:K]
            else:
                cand = cem.ask(K)
            cand[:, 0, :] = project(cand[:, 0, :])[0]
            pos = env.rollout_candidates(cand.cpu(), snap, use_rollout_fidelity=True,
                                         record=False)
            cost = dv_of(pos, v0)
            cem.tell(cand, cost.to(dev))
            cem_hist.append(float(cost.min()))
        a_oracle = cem.best()[0].cpu()

        # ── final full-fidelity batch ─────────────────────────────────────
        acts, labels = [], []
        for a in arms:
            acts.append(s1["arms"][a]["a_rank"][s]); labels.append(("rank", a))
            acts.append(s1["arms"][a]["a_grad"][s]); labels.append(("grad", a))
        acts.append(a_oracle); labels.append(("oracle", "cem"))
        n_val = K - len(acts)
        val_rows = s1["pool_rows"][s][:n_val]
        for j in range(n_val):
            acts.append(pool[j].cpu()); labels.append(("poolval", int(val_rows[j])))
        A = torch.stack(acts)[:, None, :]
        A[:, 0, :] = project(A[:, 0, :].to(dev))[0].cpu()
        pos = env.rollout_candidates(A, snap, use_rollout_fidelity=False, record=False)
        dv = dv_of(pos, v0)

        rec = {"restore_err_m": restore_err, "v0": float(v0),
               "a_oracle": a_oracle, "cem_hist": cem_hist,
               "labels": labels, "dv_all": dv, "val_rows": val_rows}
        out["dv"][s] = rec
        print(f"slate {s:2d}  restore_err {restore_err:.2e}  v0 {float(v0):.4f}  "
              f"oracle dv {float(dv[2*len(arms)]):+.5f}  cem {['%+.4f'%c for c in cem_hist]}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        for (kind, a), d in zip(labels, dv):
            if kind != "poolval":
                print(f"    {kind:6s} {a:24s} dv {float(d):+.5f}", flush=True)
        torch.save(out, args.out)

    env.destroy()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
