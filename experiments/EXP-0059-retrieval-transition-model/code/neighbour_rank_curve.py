"""Coordinator follow-up (2026-09-28, hard stop 08:10, offline/CPU only):
information-vs-noise control for retrieval (design doc R2). For the 25k
bank (task 1's winner) and DS-0009, transfer from the donor at distance RANK
r in {1, 5, 20, 100, 1000, "random"} (displacement-only transfer, matching
`diagnostics_r2.py`'s `transfer_displacement`; no k-way aggregation -- this
isolates "how good is the retrieved donor" from "how good is the
aggregation") and score: `slateN_tough` (DS-0009 test_pools, all 32 pools x
64 candidates), `accuracy_1` and moved-cube mm error (DS-0009 test_chains,
N=256 queries, scatter+clump balanced -- same subsample convention as
`diagnostics_r2.py`'s section 5). A monotone decline as r increases (worst
at "random") is the signature that RETRIEVED INFORMATION is doing the work,
not the aggregation machinery.

ONE distance search (chains queries + pool queries together) against the
25k bank, computed once; every r is then a cheap column-selection +
Hungarian-transfer post-process (no repeated bank search).

Usage:
    CUDA_VISIBLE_DEVICES= python -u neighbour_rank_curve.py
"""
import glob, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_narrow as EN  # noqa: E402
import eval_extended as EE  # noqa: E402
from simple_mpc.adapters import occ_from_particles, occ_for_scoring  # noqa: E402
from simple_mpc.learned_mpc import lyap  # noqa: E402
from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.distance import (DistanceConfig, query_points_and_weights,
                                      bank_points_and_weights, full_distance_matrix)  # noqa: E402
from model.retrieval.frame import push_frame_to_world  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RES = Path(__file__).resolve().parents[1] / "results" / "neighbour_rank_curve.json"
D = EN.D
_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)
BEST_CFG = DistanceConfig(**_CAPPED)
MOVED_THRESH = 0.001
RANKS = [1, 5, 20, 100, 1000]
N_CHAIN_Q = 256


def transfer_at(q_uv0, bank, donor_idx):
    donor_uv0, donor_duv, donor_moved = bank.uv0[donor_idx], bank.duv[donor_idx], bank.moved[donor_idx]
    cost = torch.cdist(q_uv0, donor_uv0).numpy()
    row, col = linear_sum_assignment(cost)
    duv = torch.zeros_like(q_uv0)
    moved_matched = donor_moved[col]
    duv[row] = torch.where(moved_matched.unsqueeze(-1), donor_duv[col], torch.zeros_like(donor_duv[col]))
    return q_uv0 + duv


def main():
    assert not torch.cuda.is_available(), "expected CUDA hidden; refusing to touch the GPU per coordinator instruction"
    bank = TransitionBank.load(str(ARTIFACTS / "bank_25k.pt"))
    bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, BEST_CFG)
    T = len(bank)
    print(f"bank_25k: {T} rows")

    # ---- chain queries (acc1, mm) ----
    ch = [torch.load(f, map_location="cpu", weights_only=False)
         for f in sorted(glob.glob(str(D / "test_chains/_*_data.pt")))]
    states0 = torch.cat([d["states"] for d in ch]).float()
    states1 = torch.cat([d["states_"] for d in ch]).float()
    p_starts = torch.cat([d["p_starts"] for d in ch]).float()
    p_stops = torch.cat([d["p_stops"] for d in ch]).float()
    kinds = np.array(sum([list(d["start_kind"]) for d in ch], []))
    rng = np.random.default_rng(1)
    idx_parts = []
    for k in ("scatter", "clump"):
        pool = np.nonzero(kinds == k)[0]
        idx_parts.append(rng.choice(pool, size=min(N_CHAIN_Q // 2, len(pool)), replace=False))
    cidx = np.sort(np.concatenate(idx_parts))
    print(f"chain queries: {len(cidx)}")

    cq_pts, cq_w, cq_uv0, _ = query_points_and_weights(states0[cidx], p_starts[cidx], p_stops[cidx], BEST_CFG)
    t0 = time.time()
    D_chain = full_distance_matrix(cq_pts, cq_w, bank_pts, bank_w, BEST_CFG, bank_valid=bank_valid)
    print(f"  chain distance matrix: {time.time()-t0:.1f}s")
    order_chain = torch.argsort(D_chain, dim=1)  # (N,T) ascending

    # ---- pool queries (slateN_tough) ----
    pools = [torch.load(f, map_location="cpu", weights_only=False)
            for f in sorted(glob.glob(str(D / "test_pools/pools_*.pt")))]
    DistTough = EE._dist_fields(EE.GOALS_TOUGH)
    pool_data = []  # (pool_uv0_all, order, s0, o0, t0_occ, truth_occ, act, p_start, p_stop)
    t0 = time.time()
    for d in pools:
        for pi in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            s0 = d["states"][ix[0]][None].float()
            o0 = occ_from_particles(s0)
            t0_occ = occ_for_scoring(s0[:, :, :3])
            truth_occ = occ_for_scoring(d["states_"][ix, :, :3].float())
            ps, pe = d["p_starts"][ix].float(), d["p_stops"][ix].float()
            q_pts, q_w, q_uv0, _ = query_points_and_weights(s0.expand(len(ix), -1, -1).contiguous(), ps, pe, BEST_CFG)
            Dp = full_distance_matrix(q_pts, q_w, bank_pts, bank_w, BEST_CFG, bank_valid=bank_valid)
            order = torch.argsort(Dp, dim=1)
            pool_data.append((q_uv0, order, s0, o0, t0_occ, truth_occ, ps, pe))
    print(f"  pool distance matrices ({len(pool_data)} pools): {time.time()-t0:.1f}s")

    out = json.loads(RES.read_text()) if RES.exists() else {}
    for rank_label in RANKS + ["random"]:
        key = f"rank_{rank_label}"
        if key in out:
            print("skip (already done):", key); continue
        t0 = time.time()
        # --- chains: mm + acc1 ---
        mms = []
        preds_particles = states0[cidx].clone()
        for row, qi in enumerate(cidx):
            q_start, q_stop = p_starts[qi:qi + 1], p_stops[qi:qi + 1]
            q_uv0 = cq_uv0[row]
            if rank_label == "random":
                donor = int(np.random.default_rng(2000 + int(qi)).integers(0, T))
            else:
                donor = int(order_chain[row, rank_label - 1])
            pred_uv1 = transfer_at(q_uv0, bank, donor)
            pred_xy1 = push_frame_to_world(pred_uv1[None], q_start, q_stop)[0]
            preds_particles[row, :, :2] = pred_xy1
            true_xy1 = states1[qi, :, :2]
            true_moved = (true_xy1 - states0[qi, :, :2]).norm(dim=-1) > MOVED_THRESH
            if bool(true_moved.any()):
                d_ = torch.cdist(true_xy1[true_moved], pred_xy1)
                mms.append(float(d_.min(dim=1).values.mean()) * 1000.0)
        act_c = torch.cat([p_starts[cidx, :2], p_stops[cidx, :2]], 1)
        Pc = occ_from_particles(preds_particles)
        Tc = occ_from_particles(states1[cidx])
        Oc = occ_from_particles(states0[cidx])
        Rc = EN.swept_region(act_c, "cpu")
        acc1 = EN.acc(Pc, Tc, Oc, Rc)

        # --- pools: slateN_tough ---
        caps = {g: [] for g in DistTough}
        for (q_uv0_all, order, s0, o0, t0_occ, truth_occ, ps, pe) in pool_data:
            n_cand = q_uv0_all.shape[0]
            pred_states = s0.expand(n_cand, -1, -1).clone()
            for b in range(n_cand):
                if rank_label == "random":
                    donor = int(np.random.default_rng(3000 + b).integers(0, T))
                else:
                    donor = int(order[b, rank_label - 1])
                pred_uv1 = transfer_at(q_uv0_all[b], bank, donor)
                pred_xy1 = push_frame_to_world(pred_uv1[None], ps[b:b + 1], pe[b:b + 1])[0]
                pred_states[b, :, :2] = pred_xy1
            po = occ_from_particles(pred_states)
            for g in DistTough:
                vt = (lyap(truth_occ, DistTough[g]) - lyap(t0_occ, DistTough[g])).numpy()
                vp = (lyap(po, DistTough[g]) - lyap(o0, DistTough[g])).numpy()
                den = vt.mean() - vt.min()
                if den > 1e-9:
                    caps[g].append(float((vt.mean() - vt[vp.argmin()]) / den))
        slateN_tough = float(np.mean([np.mean(v) for v in caps.values() if v]))

        wall = time.time() - t0
        out[key] = dict(rank=rank_label, mm_mean=float(np.mean(mms)), n_mm=len(mms),
                        accuracy_1=float(acc1), slateN_tough=slateN_tough, wall_s=wall)
        print(f"{key:12s} mm={out[key]['mm_mean']:.3f}  acc1={out[key]['accuracy_1']:+.3f}  "
             f"slateN_tough={out[key]['slateN_tough']:.3f}  ({wall:.1f}s)", flush=True)
        tmp = str(RES) + ".tmp"; RES.parent.mkdir(parents=True, exist_ok=True)
        Path(tmp).write_text(json.dumps(out, indent=1)); os.replace(tmp, RES)
    print("wrote", RES)


if __name__ == "__main__":
    main()
