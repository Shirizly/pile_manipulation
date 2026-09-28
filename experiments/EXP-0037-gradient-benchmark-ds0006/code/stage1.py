"""EXP-0037 stage 1 (models only): rank-only picks and GD restarts for every
(state, goal, arm). DESIGN.md is the plan. Checkpointed after every arm.

Reuses EXP-0027's goal fields / row-wise lyapunov / projected-Adam optimiser
(itself EXP-0023's), so the optimiser is unchanged across the three records.
"""
import sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0027-many-goal-benchmarks/code"))
sys.path.insert(0, str(next((REPO / "experiments").glob("EXP-0023-*")) / "code"))
import stage1_grad_many as s27
from stage1_grad_many import lyap_rows, optimise
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, occ_from_particles, assert_dv_convention
from utils import git_provenance

CORPUS = "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys"
OUT = REPO / "experiments/EXP-0037-gradient-benchmark-ds0006/artifacts/RUN-0001/stage1.pt"
STATES = list(range(40))
GOALS = ["quadrant_0", "quadrant_3", "letter_O", "letter_T", "letter_L", "letter_S"]
NFD = ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_warped_randlen_flipaug",
       "nfd_residual_warped_flipaug_randlen", "nfd_residual_worldframe_noaug_ep43"]
ARMS = NFD + ["linear_switched_soft", "ensemble_nfd"]
N_RESTART, STEPS, LR = 3, 120, 1.5e-3


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = BinnedSlateCorpus.load(str(REPO / CORPUS)).step(0)
    sl = rows.slate_idx.long()
    s27.GOALS = GOALS                                   # goal_fields reads the module global
    D = s27.goal_fields(dev)
    G = len(GOALS)
    pool_rows = {s: (sl == s).nonzero(as_tuple=True)[0] for s in STATES}
    states0 = {s: rows.states[pool_rows[s][0]].float().clone() for s in STATES}
    pool_act = {s: torch.cat([rows.p_starts[pool_rows[s], :2], rows.p_stops[pool_rows[s], :2]], 1).float()
                for s in STATES}
    out = torch.load(OUT, weights_only=False) if OUT.exists() else {
        "states": STATES, "goals": GOALS, "arms": {}, "goal_fields": D.cpu(),
        "pool_rows": pool_rows, "pool_actions": pool_act, "states0": states0,
        "config": dict(corpus=CORPUS, n_restart=N_RESTART, steps=STEPS, lr=LR, nfd_members=NFD),
        "provenance": git_provenance()}
    for arm in ARMS:
        if arm in out["arms"]:
            continue
        t0 = time.time()
        members = NFD if arm == "ensemble_nfd" else [arm]
        ads = []
        for m in members:
            ad = make_occ_adapter(m, dev, "corner"); assert_dv_convention(ad); ads.append(ad)

        def predict(occ, a):
            return torch.stack([ad.predict_step(occ, a) for ad in ads]).mean(0) if len(ads) > 1 \
                else ads[0].predict_step(occ, a)
        rec = {k: {} for k in ("a_rank", "pred_rank", "a_restart", "pred_restart", "a_grad", "pred_grad", "hits")}
        for s in STATES:
            occ0 = occ_from_particles(states0[s][None], dev)
            acts = pool_act[s].to(dev); P = len(acts)
            with torch.no_grad():
                if len(ads) > 1:     # ensemble of dv = mean of member dv (lyapunov is not linear in the image)
                    v0m = [lyap_rows(occ0.expand(G, -1, -1), D) for _ in ads]
                    dvp = torch.stack([torch.stack([lyap_rows(ad.predict_step(occ0.expand(P, -1, -1).contiguous(), acts),
                                                              D[k].expand(P, -1, -1)) for k in range(G)], 1) - v0m[0][None]
                                       for ad in ads]).mean(0)
                else:
                    occ1 = ads[0].predict_step(occ0.expand(P, -1, -1).contiguous(), acts)
                    v0 = lyap_rows(occ0.expand(G, -1, -1), D)
                    dvp = torch.stack([lyap_rows(occ1, D[k].expand(P, -1, -1)) for k in range(G)], 1) - v0[None]
            top = dvp.argsort(0)[:N_RESTART]                                    # (R, G)
            a0 = acts[top.T.reshape(-1)]                                        # (G*R, 4), goal-major
            Drep = D.repeat_interleave(N_RESTART, 0)
            occ0R = occ0.expand(G * N_RESTART, -1, -1).contiguous()
            v0R = lyap_rows(occ0R, Drep)

            def dv_fn(a):
                if len(ads) > 1:
                    return torch.stack([lyap_rows(ad.predict_step(occ0R, a), Drep) - v0R for ad in ads]).mean(0)
                return lyap_rows(ads[0].predict_step(occ0R, a), Drep) - v0R
            a_g, dv_g, hits = optimise(dv_fn, a0, STEPS, LR)
            a_g = a_g.view(G, N_RESTART, 4); dv_g = dv_g.view(G, N_RESTART)
            best = dv_g.argmin(1)
            rec["a_rank"][s] = acts[top[0]].cpu(); rec["pred_rank"][s] = dvp[top[0], torch.arange(G)].cpu()
            rec["a_restart"][s] = a_g.cpu(); rec["pred_restart"][s] = dv_g.cpu()
            rec["a_grad"][s] = a_g[torch.arange(G), best].cpu(); rec["pred_grad"][s] = dv_g[torch.arange(G), best].cpu()
            rec["hits"][s] = hits.view(G, N_RESTART, 2).cpu()
        out["arms"][arm] = rec
        tmp = Path(str(OUT) + ".tmp"); torch.save(out, tmp); tmp.replace(OUT)
        print(f"[{arm}] {time.time() - t0:.0f}s; mean pred dv rank {torch.stack(list(rec['pred_rank'].values())).mean():+.4f} "
              f"-> grad {torch.stack(list(rec['pred_grad'].values())).mean():+.4f}", flush=True)
        del ads; torch.cuda.empty_cache()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
