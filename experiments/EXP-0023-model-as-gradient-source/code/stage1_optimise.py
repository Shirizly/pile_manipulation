"""EXP-0023 stage 1 (model side, no Genesis): per state and per arm, pick the
rank-only action from the shared seed pool and then gradient-optimise it.

Writes `stage1_actions.pt` holding, for every (state, arm):
    a_rank   (4,)   best-in-pool by PREDICTED dv
    a_grad   (4,)   best iterate of Adam on the model's own predicted dv
plus the shared seed pool and its bookkeeping, so stage 2 (Genesis) is a pure
executor and every arm is guaranteed to have started from the same pool.

Action parameterisation: the raw 4-vector [sx,sy,ex,ey] in world metres, with
CONSTRAINTS APPLIED BY PROJECTION after each Adam step (endpoints clamped into
the workspace box, then the push length rescaled into the fitted 20-70 mm
range along the current direction, start held fixed).  Every projection that
actually moved the action is counted, so an arm that is permanently clamped is
visible rather than silently scored on the constraint.

Objective: the adapter's own `dv` = value(after) - value(before), a COST, so
the optimiser MINIMISES it (`simple_mpc.adapters.assert_dv_convention`).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import (make_occ_adapter, assert_dv_convention,
                                 occ_from_particles, OCC_BOUNDS)
from utils import git_provenance

CORPUS = "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"
XY_MARGIN = 0.004          # keep endpoints 4 mm inside the +/-64 mm workspace
L_MIN, L_MAX = 0.020, 0.070   # the corpus / fitted push-length range


def project(act):
    """Clamp into the legal set. Returns (act, hit) with `hit` a (B,2) bool of
    [box clamped, length clamped]."""
    lo = OCC_BOUNDS["x_min"] + XY_MARGIN
    hi = OCC_BOUNDS["x_max"] - XY_MARGIN
    clamped = act.clamp(lo, hi)
    hit_box = (clamped != act).any(dim=1)
    a = clamped
    d = a[:, 2:4] - a[:, 0:2]
    L = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    Lc = L.clamp(L_MIN, L_MAX)
    hit_len = (Lc != L).squeeze(-1)
    a = torch.cat([a[:, 0:2], a[:, 0:2] + d / L * Lc], dim=1)
    # re-clamp the end point after the length rescale (box wins)
    end = a[:, 2:4].clamp(lo, hi)
    hit_box = hit_box | (end != a[:, 2:4]).any(dim=1)
    return torch.cat([a[:, 0:2], end], dim=1), torch.stack([hit_box, hit_len], dim=1)


def optimise(adapter, occ0, a0, steps, lr):
    """Adam on the model's predicted dv, with projection. Returns the best
    iterate by PREDICTED dv (the only signal an MPC controller has)."""
    a = a0.clone()
    p = a.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    best = a0.clone()
    with torch.no_grad():
        best_dv = adapter.dv(occ0, a0).clone()
    hits = torch.zeros(a0.shape[0], 2, device=a0.device)
    traj = []
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        dv = adapter.dv(occ0, p)
        dv.sum().backward()
        opt.step()
        with torch.no_grad():
            proj, hit = project(p.data)
            hits += hit.float()
            p.data.copy_(proj)
            dv_now = adapter.dv(occ0, p.data)
            imp = dv_now < best_dv
            best[imp] = p.data[imp]
            best_dv[imp] = dv_now[imp]
            traj.append(float(dv_now.mean()))
    return best, best_dv, hits / steps, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", required=True)
    ap.add_argument("--n-states", type=int, default=10)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = BinnedSlateCorpus.load(str(REPO / CORPUS))
    rows = corpus.step(0)
    g = torch.Generator().manual_seed(args.seed)

    slates = list(range(args.n_states))
    pool_rows, occ0s, states0 = {}, {}, {}
    for s in slates:
        idx = (rows.slate_idx == s).nonzero(as_tuple=True)[0]
        pick = idx[torch.randperm(idx.numel(), generator=g)[:args.pool]]
        pool_rows[s] = pick
        states0[s] = rows.states[pick[0]].float()          # shared by the pool
        occ0s[s] = occ_from_particles(states0[s][None], dev)

    pool_act = {s: torch.cat([rows.p_starts[pool_rows[s], :2],
                              rows.p_stops[pool_rows[s], :2]], dim=1).float().to(dev)
                for s in slates}

    out = {"slates": slates, "pool_rows": {s: pool_rows[s] for s in slates},
           "pool_actions": {s: pool_act[s].cpu() for s in slates},
           "states0": {s: states0[s] for s in slates},
           "arms": {},
           "config": vars(args) | {"corpus": CORPUS, "xy_margin": XY_MARGIN,
                                   "L_MIN": L_MIN, "L_MAX": L_MAX,
                                   "dv": "value(after) - value(before), a COST"},
           "provenance": git_provenance()}

    for arm in args.arms:
        t0 = time.time()
        ad = make_occ_adapter(arm, dev, args.goal)
        assert_dv_convention(ad)
        rec = {"a_rank": {}, "a_grad": {}, "dv_pred_rank": {}, "dv_pred_grad": {},
               "bound_hit_frac": {}, "traj": {}}
        for s in slates:
            occ0 = occ0s[s]
            acts = pool_act[s]
            B = acts.shape[0]
            with torch.no_grad():
                dvp = torch.cat([ad.dv(occ0.expand(min(50, B - i), -1, -1).contiguous(),
                                       acts[i:i + 50])
                                 for i in range(0, B, 50)])
            k = int(torch.argmin(dvp))
            a_rank = acts[k:k + 1]
            a_grad, dv_g, hits, traj = optimise(ad, occ0, a_rank, args.steps, args.lr)
            rec["a_rank"][s] = a_rank[0].cpu()
            rec["a_grad"][s] = a_grad[0].cpu()
            rec["dv_pred_rank"][s] = float(dvp[k])
            rec["dv_pred_grad"][s] = float(dv_g[0])
            rec["bound_hit_frac"][s] = hits[0].cpu()
            rec["traj"][s] = traj
            print(f"  {arm} slate {s:2d}: pred dv {float(dvp[k]):+.5f} -> {float(dv_g[0]):+.5f} "
                  f"(bound hit box/len {float(hits[0,0]):.2f}/{float(hits[0,1]):.2f})", flush=True)
        out["arms"][arm] = rec
        print(f"[{arm}] done in {time.time()-t0:.1f}s", flush=True)
        del ad
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
