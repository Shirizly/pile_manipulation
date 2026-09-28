"""EXP-0027 item 2, stage 1 (model side, no Genesis): per (state, arm), the
rank-only pick and the gradient-optimised action for EVERY goal at once.

Reuses EXP-0023's legality projection and optimiser settings verbatim
(`project`, 20-70 mm, 4 mm margin, Adam 120 steps, lr 1.5e-3, best iterate by
PREDICTED dv). Goals are batched along the batch dimension: row g of the
action batch is optimised against goal g's distance field only, and Adam is
element-wise, so this is the same computation as G separate runs.

Pools: EXP-0023's exact generator (torch.randperm, seed 0, slates in order),
so slates 0-9 reproduce EXP-0023's pools; asserted against its artifact.

dv = lyapunov(after) - lyapunov(before) under each goal's field: a COST.
"""
from __future__ import annotations
import argparse, glob, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(next((REPO / "experiments").glob("EXP-0023-*")) / "code"))

from control_utility_test import lyapunov_weights
from Baselines.common.goals import (dist_field_from_mask, letter_mask, quadrant_mask,
                                    higher_is_better_for)
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, assert_dv_convention, occ_from_particles, OCC_GRID
from stage1_optimise import project, CORPUS, XY_MARGIN, L_MIN, L_MAX
from utils import git_provenance

GOALS = ["corner", "quadrant_0", "quadrant_1", "quadrant_2", "quadrant_3",
         "letter_O", "letter_T", "letter_A", "letter_C", "letter_H", "letter_L", "letter_S"]
ARMS = ["nfd_3ch_randlen", "nfd_3ch_finetuned", "nfd_warped_randlen",
        "nfd_residual_warped", "linear_switched_soft", "linear_switched_hard"]


def goal_fields(dev):
    """(G, H, W) lyapunov distance fields, convention A (row = world x)."""
    D = []
    for g in GOALS:
        if g == "corner":
            D.append(lyapunov_weights((OCC_GRID, OCC_GRID), "corner", dev).float())
            continue
        m = quadrant_mask(OCC_GRID, OCC_GRID, int(g[-1])) if g.startswith("quadrant_") \
            else letter_mask(g[-1], OCC_GRID, OCC_GRID)
        D.append(torch.from_numpy(dist_field_from_mask(m)).float().to(dev))
    return torch.stack(D)


def lyap_rows(occ, D, eps=1e-6):
    """Row-wise lyapunov: occ (B,H,W), D (B,H,W) -> (B,). Same formula as
    control_utility_test.lyapunov, with a per-row field."""
    f = occ.reshape(occ.shape[0], -1)
    return (f * D.reshape(D.shape[0], -1)).sum(1) / f.sum(1).clamp_min(eps)


def optimise(dv_fn, a0, steps, lr):
    """EXP-0023's `optimise`, with the objective passed in (row-separable)."""
    p = a0.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    best = a0.clone()
    with torch.no_grad():
        best_dv = dv_fn(a0).clone()
    hits = torch.zeros(a0.shape[0], 2, device=a0.device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        dv_fn(p).sum().backward()
        opt.step()
        with torch.no_grad():
            proj, hit = project(p.data)
            hits += hit.float()
            p.data.copy_(proj)
            dv_now = dv_fn(p.data)
            imp = dv_now < best_dv
            best[imp] = p.data[imp]
            best_dv[imp] = dv_now[imp]
    return best, best_dv, hits / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=20)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert not higher_is_better_for("lyapunov")      # minimise
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = BinnedSlateCorpus.load(str(REPO / CORPUS)).step(0)
    g = torch.Generator().manual_seed(args.seed)
    slates = list(range(args.n_states))
    pool_rows, states0, occ0s = {}, {}, {}
    for s in slates:                      # EXP-0023's exact draw order
        idx = (rows.slate_idx == s).nonzero(as_tuple=True)[0]
        pool_rows[s] = idx[torch.randperm(idx.numel(), generator=g)[:args.pool]]
        states0[s] = rows.states[pool_rows[s][0]].float().clone()
        occ0s[s] = occ_from_particles(states0[s][None], dev)
    ref = torch.load(glob.glob(str(REPO / "experiments/EXP-0023-*/artifacts/stage1_actions.pt"))[0],
                     map_location="cpu", weights_only=False)
    ref_slates = [s for s in ref["slates"] if s in slates]
    for s in ref_slates:
        assert torch.equal(ref["pool_rows"][s], pool_rows[s]), f"pool mismatch on slate {s}"
    print(f"pools for slates {ref_slates} reproduce EXP-0023 exactly", flush=True)
    pool_act = {s: torch.cat([rows.p_starts[pool_rows[s], :2], rows.p_stops[pool_rows[s], :2]],
                             1).float().to(dev) for s in slates}
    D = goal_fields(dev)
    G = len(GOALS)

    out = {"slates": slates, "goals": GOALS, "goal_fields": D.cpu(),
           "pool_rows": pool_rows, "pool_actions": {s: pool_act[s].cpu() for s in slates},
           "states0": states0, "arms": {},
           "config": vars(args) | {"corpus": CORPUS, "xy_margin": XY_MARGIN, "L_MIN": L_MIN,
                                   "L_MAX": L_MAX, "value_fn": "lyapunov",
                                   "dv": "value(after) - value(before), a COST"},
           "provenance": git_provenance()}
    for arm in args.arms:
        t0 = time.time()
        ad = make_occ_adapter(arm, dev, "corner")
        assert_dv_convention(ad)
        rec = {k: {} for k in ("a_rank", "a_grad", "dv_pred_rank", "dv_pred_grad",
                               "bound_hit_frac", "dv_pred_pool")}
        for s in slates:
            occ0 = occ0s[s]
            acts = pool_act[s]
            with torch.no_grad():
                occ1 = torch.cat([ad.predict_step(occ0.expand(min(50, len(acts) - i), -1, -1).contiguous(),
                                                  acts[i:i + 50]) for i in range(0, len(acts), 50)])
                P = len(acts)
                v1 = torch.stack([lyap_rows(occ1, D[k].expand(P, -1, -1)) for k in range(G)], 1)
                v0 = lyap_rows(occ0.expand(G, -1, -1), D)                       # (G,)
                dvp = v1 - v0[None]                                             # (P, G)
            k_best = dvp.argmin(0)                                              # (G,)
            a_rank = acts[k_best]                                               # (G, 4)
            occ0G = occ0.expand(G, -1, -1).contiguous()

            def dv_fn(a):
                return lyap_rows(ad.predict_step(occ0G, a), D) - v0
            a_grad, dv_g, hits = optimise(dv_fn, a_rank, args.steps, args.lr)
            rec["a_rank"][s] = a_rank.cpu(); rec["a_grad"][s] = a_grad.cpu()
            rec["dv_pred_rank"][s] = dvp[k_best, torch.arange(G)].cpu()
            rec["dv_pred_grad"][s] = dv_g.cpu(); rec["bound_hit_frac"][s] = hits.cpu()
            rec["dv_pred_pool"][s] = dvp.cpu()
        out["arms"][arm] = rec
        # continuity with EXP-0023 (corner = goal 0): a_rank must be identical
        if arm in ref["arms"]:
            same = [torch.equal(rec["a_rank"][s][0], ref["arms"][arm]["a_rank"][s]) for s in ref_slates]
            dg = max(float((rec["a_grad"][s][0] - ref["arms"][arm]["a_grad"][s]).abs().max())
                     for s in ref_slates)
            print(f"[{arm}] corner a_rank == EXP-0023 on {sum(same)}/{len(same)} slates; "
                  f"max |a_grad - EXP-0023| {dg:.2e} m", flush=True)
        print(f"[{arm}] done in {time.time() - t0:.1f}s; mean bound-hit box/len "
              f"{float(torch.stack(list(rec['bound_hit_frac'].values())).mean((0, 1))[0]):.3f}/"
              f"{float(torch.stack(list(rec['bound_hit_frac'].values())).mean((0, 1))[1]):.3f}", flush=True)
        del ad
        torch.cuda.empty_cache()
        torch.save(out, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
