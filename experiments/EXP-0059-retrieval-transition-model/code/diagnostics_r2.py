"""EXP-0059 R2 diagnostics: decompose retrieval's gap to the narrow NFD
(coordinator request, 2026-09-28). ALL OFFLINE (no Genesis), checkpointed
atomically per section. Every number here that reads DS-0009 test states is
labelled `ceiling` and MUST NOT be used as a headline result (design doc
Section 2.0's rule) -- these exist to attribute the gap, not to claim it.

Sections (see module docstrings inline for the exact method):
  1. oracle-donor ceiling: for each of a query subsample, search the WHOLE
     bank (not just the metric's top-k) for the donor whose TRANSFERRED
     prediction is closest to the TRUE outcome, separately by moved-cube mm
     error and by a swept-region proxy error, at M in {50, 500, whole bank}.
  2. same-state leave-one-out ceiling: DS-0009 pools share a start state;
     retrieve the nearest-by-ACTION other push in the SAME pool.
  3. transfer-rule comparison (displacement / paste / transport+projection)
     UNDER the oracle donor from section 1.
  4. moved-cube-in-window fraction + "predicts nothing moves" check.

Run: python -u experiments/EXP-0059-retrieval-transition-model/code/diagnostics_r2.py
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
import eval_narrow as EN  # noqa: E402

from simple_mpc.adapters import occ_from_particles, occ_for_scoring  # noqa: E402
from simple_mpc.learned_mpc import lyap  # noqa: E402
from Baselines.common.goals import dist_field_from_mask  # noqa: E402
from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.frame import world_to_push_frame, push_frame_to_world, push_frame_basis  # noqa: E402
from model.retrieval.distance import DistanceConfig, full_distance_matrix, bank_points_and_weights  # noqa: E402
from model.retrieval.predictor import RetrievalPredictor  # noqa: E402
from model.retrieval.overlap import resolve_overlaps  # noqa: E402

D = EN.D
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RES = Path(__file__).resolve().parents[1] / "results" / "diagnostics_r2.json"
MOVED_THRESH = 0.001   # metres, matches bank.py's DEFAULT_MOVED_THRESHOLD
CUBE_SIZE = 0.005

# "current best distance": the config the R1 sweep's best CLEAN variant
# (k5_cube_median) used.
BEST_CFG = DistanceConfig(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)

N_SUBSAMPLE = 40   # oracle-whole-bank search cost is O(N * bank_size) ~1ms/candidate; see report


def _atomic(obj, path):
    tmp = str(path) + ".tmp"; Path(tmp).write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def _load(out=None):
    out = out if out is not None else (json.loads(RES.read_text()) if RES.exists() else {})
    return out


def in_swept_rect(xy_world, p_start, p_stop):
    """xy_world: (n,2), p_start/p_stop: (1,>=2) -> (n,) bool, EXACTLY
    `eval_narrow.swept_region`'s rectangle (u in [-20mm, L], |v|<=24mm),
    but as a particle-space predicate instead of a rasterised mask."""
    uv = world_to_push_frame(xy_world[None], p_start, p_stop)[0]
    _, _, L = push_frame_basis(p_start, p_stop)
    return (uv[:, 0] >= -0.02) & (uv[:, 0] <= float(L[0])) & (uv[:, 1].abs() <= 0.024)


def transfer_displacement(q_uv0, donor_uv0, donor_duv, donor_moved):
    cost = torch.cdist(q_uv0, donor_uv0).numpy()
    row, col = linear_sum_assignment(cost)
    duv = torch.zeros_like(q_uv0)
    moved_matched = donor_moved[col]
    duv[row] = torch.where(moved_matched.unsqueeze(-1), donor_duv[col], torch.zeros_like(donor_duv[col]))
    return q_uv0 + duv, row, col, moved_matched


def transfer_paste(q_uv0, donor_uv0, donor_uv1, donor_moved):
    cost = torch.cdist(q_uv0, donor_uv0).numpy()
    row, col = linear_sum_assignment(cost)
    out = q_uv0.clone()
    moved_matched = donor_moved[col]
    out[row] = torch.where(moved_matched.unsqueeze(-1), donor_uv1[col], q_uv0[row])
    return out


def mm_error(true_xy1_moved, pred_xy1):
    if true_xy1_moved.shape[0] == 0:
        return None
    d = torch.cdist(true_xy1_moved, pred_xy1)
    return float(d.min(dim=1).values.mean()) * 1000.0   # mm


def swept_proxy_error(true_xy1_in_swept, pred_xy1):
    if true_xy1_in_swept.shape[0] == 0:
        return None
    d = torch.cdist(true_xy1_in_swept, pred_xy1)
    return float((d.min(dim=1).values ** 2).mean())


def section1_and_3(out):
    """Oracle-donor ceiling (whole bank / top-500 / top-50) by mm error and by
    swept-region proxy, PLUS the 3 transfer rules compared under the
    mm-error oracle donor (section 3), on the SAME N_SUBSAMPLE query rows."""
    if "section1" in out and "section3" in out:
        print("skip section1/3 (already done)"); return
    bank = TransitionBank.load(str(ARTIFACTS / "bank.pt"))
    bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, BEST_CFG)
    ch = [torch.load(f, map_location="cpu", weights_only=False)
         for f in sorted(glob.glob(str(D / "test_chains/_*_data.pt")))]
    states0 = torch.cat([d["states"] for d in ch]).float()
    states1 = torch.cat([d["states_"] for d in ch]).float()
    p_starts = torch.cat([d["p_starts"] for d in ch]).float()
    p_stops = torch.cat([d["p_stops"] for d in ch]).float()
    Qtot = states0.shape[0]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(Qtot, size=min(N_SUBSAMPLE, Qtot), replace=False))
    print(f"section1/3: {len(idx)} of {Qtot} DS-0009 chain rows, bank size {len(bank)}")

    q_pts, q_w, q_uv0_all, q_valid = _query_pw(states0[idx], p_starts[idx], p_stops[idx])
    t0 = time.time()
    # GPU is shared with concurrent Genesis collection jobs tonight (often
    # near-full) -- this matrix is small enough (N x ~12k x 20 x 20) to run
    # on CPU without meaningfully changing the runtime budget, so force it
    # rather than risk an OOM race with those jobs.
    D_full = full_distance_matrix(q_pts, q_w, bank_pts, bank_w, BEST_CFG,
                                  bank_valid=bank_valid, device="cpu")  # (N, T)
    print(f"  full distance matrix: {time.time()-t0:.1f}s")

    M_LIST = [("top50", 50), ("top500", 500), ("whole", len(bank))]
    res1 = {m: {"mm": [], "swept": []} for m, _ in M_LIST}
    res3 = {"displacement": {"mm": [], "swept": []}, "paste": {"mm": [], "swept": []},
           "transport_projected": {"mm": [], "swept": []}}
    n_zero_moved = 0
    t0 = time.time()
    for row, qi in enumerate(idx):
        q_start, q_stop = p_starts[qi:qi + 1], p_stops[qi:qi + 1]
        q_uv0 = q_uv0_all[row]
        true_xy1 = states1[qi, :, :2]
        true_moved = (true_xy1 - states0[qi, :, :2]).norm(dim=-1) > MOVED_THRESH
        if not bool(true_moved.any()):
            n_zero_moved += 1
            continue
        true_xy1_moved = true_xy1[true_moved]
        in_swept = in_swept_rect(true_xy1, q_start, q_stop)
        true_xy1_swept = true_xy1[in_swept]

        order = torch.argsort(D_full[row])
        best_mm = {m: (float("inf"), None) for m, _ in M_LIST}
        best_swept = {m: (float("inf"), None) for m, _ in M_LIST}
        best_mm_donor_overall = None
        cum_bound = 0
        for rank, di in enumerate(order.tolist()):
            di_t = int(di)
            pred_uv1, mrow, mcol, mmoved = transfer_displacement(
                q_uv0, bank.uv0[di_t], bank.duv[di_t], bank.moved[di_t])
            pred_xy1 = push_frame_to_world(pred_uv1[None], q_start, q_stop)[0]
            e_mm = mm_error(true_xy1_moved, pred_xy1)
            e_sw = swept_proxy_error(true_xy1_swept, pred_xy1) if true_xy1_swept.shape[0] else None
            for m, M in M_LIST:
                if rank < M:
                    if e_mm is not None and e_mm < best_mm[m][0]:
                        best_mm[m] = (e_mm, (di_t, pred_xy1))
                    if e_sw is not None and e_sw < best_swept[m][0]:
                        best_swept[m] = (e_sw, di_t)
            if rank + 1 >= M_LIST[-1][1]:
                pass
        for m, _ in M_LIST:
            if best_mm[m][1] is not None:
                res1[m]["mm"].append(best_mm[m][0])
            if best_swept[m][0] < float("inf"):
                res1[m]["swept"].append(best_swept[m][0])
        # --- section 3: transfer rules under the mm-error WHOLE-bank oracle donor
        _, oracle_info = best_mm["whole"]
        if oracle_info is None:
            continue
        di_t, oracle_pred_xy1_disp = oracle_info
        donor_uv0, donor_uv1 = bank.uv0[di_t], bank.uv1[di_t]
        donor_duv, donor_moved = bank.duv[di_t], bank.moved[di_t]
        # displacement (already have oracle_pred_xy1_disp)
        res3["displacement"]["mm"].append(mm_error(true_xy1_moved, oracle_pred_xy1_disp))
        res3["displacement"]["swept"].append(swept_proxy_error(true_xy1_swept, oracle_pred_xy1_disp)
                                             if true_xy1_swept.shape[0] else None)
        # paste
        pred_uv1_paste = transfer_paste(q_uv0, donor_uv0, donor_uv1, donor_moved)
        pred_xy1_paste = push_frame_to_world(pred_uv1_paste[None], q_start, q_stop)[0]
        res3["paste"]["mm"].append(mm_error(true_xy1_moved, pred_xy1_paste))
        res3["paste"]["swept"].append(swept_proxy_error(true_xy1_swept, pred_xy1_paste)
                                      if true_xy1_swept.shape[0] else None)
        # transport + 5mm non-overlap projection
        pred_xy1_proj = resolve_overlaps(oracle_pred_xy1_disp, min_dist=CUBE_SIZE, n_iters=5)
        res3["transport_projected"]["mm"].append(mm_error(true_xy1_moved, pred_xy1_proj))
        res3["transport_projected"]["swept"].append(swept_proxy_error(true_xy1_swept, pred_xy1_proj)
                                                    if true_xy1_swept.shape[0] else None)
        if (row + 1) % 25 == 0:
            print(f"  {row+1}/{len(idx)} rows, {time.time()-t0:.0f}s elapsed", flush=True)

    def _clean(v):
        return [x for x in v if x is not None]

    out["section1"] = {
        "n_rows_used": len(idx) - n_zero_moved, "n_rows_zero_moved_skipped": n_zero_moved,
        "bank_size": len(bank),
        **{f"{m}_mm_mean": float(np.mean(_clean(res1[m]["mm"]))) for m, _ in M_LIST},
        **{f"{m}_swept_mean": float(np.mean(_clean(res1[m]["swept"]))) for m, _ in M_LIST},
    }
    out["section3"] = {rule: {"mm_mean": float(np.mean(_clean(v["mm"]))),
                              "swept_mean": float(np.mean(_clean(v["swept"])))}
                       for rule, v in res3.items()}
    _atomic(out, RES)
    print("section1:", out["section1"]); print("section3:", out["section3"])


def _query_pw(states0, p_starts, p_stops):
    from model.retrieval.distance import query_points_and_weights
    return query_points_and_weights(states0, p_starts, p_stops, BEST_CFG)


def section2(out):
    """Same-state leave-one-out ceiling on DS-0009 test_pools: for each push,
    retrieve the nearest-by-ACTION other push in the SAME pool (identical
    start state, so this isolates transfer quality from retrieval-key
    quality)."""
    if "section2" in out:
        print("skip section2 (already done)"); return
    pools = [torch.load(f, map_location="cpu", weights_only=False)
            for f in sorted(glob.glob(str(D / "test_pools/pools_*.pt")))]
    preds, truths, prevs, regs = [], [], [], []
    mm_errs = []
    for d in pools:
        for pi in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            s0_all, s1_all = d["states"][ix].float(), d["states_"][ix].float()
            ps_all, pe_all = d["p_starts"][ix].float(), d["p_stops"][ix].float()
            n = len(ix)
            act_xy = torch.cat([ps_all[:, :2], pe_all[:, :2]], 1)   # (n,4): sx,sy,ex,ey
            angle = torch.atan2(pe_all[:, 1] - ps_all[:, 1], pe_all[:, 0] - ps_all[:, 0])
            L_SCALE = 0.02
            for i in range(n):
                others = torch.tensor([j for j in range(n) if j != i])
                pos_d = (act_xy[others, :2] - act_xy[i, :2]).norm(dim=-1)
                ang_d = torch.remainder(angle[others] - angle[i] + np.pi, 2 * np.pi) - np.pi
                action_d = (pos_d ** 2 + (L_SCALE * ang_d) ** 2).sqrt()
                j = others[torch.argmin(action_d)]
                q_uv0 = world_to_push_frame(s0_all[i, :, :2][None], ps_all[i:i + 1], pe_all[i:i + 1])[0]
                donor_uv0 = world_to_push_frame(s0_all[j, :, :2][None], ps_all[j:j + 1], pe_all[j:j + 1])[0]
                donor_uv1 = world_to_push_frame(s1_all[j, :, :2][None], ps_all[j:j + 1], pe_all[j:j + 1])[0]
                donor_duv = donor_uv1 - donor_uv0
                donor_moved = donor_duv.norm(dim=-1) > MOVED_THRESH
                pred_uv1, *_ = transfer_displacement(q_uv0, donor_uv0, donor_duv, donor_moved)
                pred_xy1 = push_frame_to_world(pred_uv1[None], ps_all[i:i + 1], pe_all[i:i + 1])[0]
                pred_state = s0_all[i:i + 1].clone(); pred_state[0, :, :2] = pred_xy1
                preds.append(occ_from_particles(pred_state))
                truths.append(occ_from_particles(s1_all[i:i + 1]))
                prevs.append(occ_from_particles(s0_all[i:i + 1]))
                regs.append(EN.swept_region(act_xy[i:i + 1], "cpu"))
                true_moved = (s1_all[i, :, :2] - s0_all[i, :, :2]).norm(dim=-1) > MOVED_THRESH
                if bool(true_moved.any()):
                    mm_errs.append(mm_error(s1_all[i, :, :2][true_moved], pred_xy1))
    P, T, O, R = map(torch.cat, (preds, truths, prevs, regs))
    acc1 = EN.acc(P, T, O, R)
    out["section2"] = {"acc1_leave_one_out_ceiling": acc1,
                       "mm_error_mean": float(np.mean([x for x in mm_errs if x is not None])),
                       "n_rows": int(len(P))}
    _atomic(out, RES)
    print("section2:", out["section2"])


def section4(out):
    """Moved-cube-in-window fraction + whether retrieval predicts "nothing
    moves" correctly there, on the full DS-0009 test_chains (1024 rows)."""
    if "section4" in out:
        print("skip section4 (already done)"); return
    ch = [torch.load(f, map_location="cpu", weights_only=False)
         for f in sorted(glob.glob(str(D / "test_chains/_*_data.pt")))]
    states0 = torch.cat([d["states"] for d in ch]).float()
    states1 = torch.cat([d["states_"] for d in ch]).float()
    p_starts = torch.cat([d["p_starts"] for d in ch]).float()
    p_stops = torch.cat([d["p_stops"] for d in ch]).float()
    Q = states0.shape[0]

    q_uv0 = world_to_push_frame(states0[:, :, :2], p_starts, p_stops)
    q_uv1_true = world_to_push_frame(states1[:, :, :2], p_starts, p_stops)
    true_duv = q_uv1_true - q_uv0
    true_moved = true_duv.norm(dim=-1) > MOVED_THRESH
    win_mask = (q_uv0[..., 0] >= BEST_CFG.window_u[0]) & (q_uv0[..., 0] <= BEST_CFG.window_u[1]) & \
              (q_uv0[..., 1].abs() <= BEST_CFG.window_v[1])
    zero_in_window = ~(true_moved & win_mask).any(dim=1)
    frac_zero = float(zero_in_window.float().mean())

    bank = TransitionBank.load(str(ARTIFACTS / "bank.pt"))
    predictor = RetrievalPredictor(bank, cfg=BEST_CFG, k=5, aggregation="cube_median", device="cpu")
    n_correct = 0
    n_total = int(zero_in_window.sum())
    CHUNK = 64
    zi = torch.nonzero(zero_in_window)[:, 0]
    for i in range(0, len(zi), CHUNK):
        b = zi[i:i + CHUNK]
        pred = predictor.predict_particles(states0[b], p_starts[b], p_stops[b])
        pred_uv1 = world_to_push_frame(pred[:, :, :2], p_starts[b], p_stops[b])
        pred_duv = pred_uv1 - q_uv0[b]
        pred_win_mask = win_mask[b]
        pred_moved_in_win = (pred_duv.norm(dim=-1) > MOVED_THRESH) & pred_win_mask
        n_correct += int((~pred_moved_in_win.any(dim=1)).sum())
    out["section4"] = {"n_queries": Q, "frac_zero_moved_in_window": frac_zero,
                       "n_zero_moved_rows": n_total,
                       "frac_predicted_correctly_as_zero": (n_correct / n_total) if n_total else None,
                       "predictor": "k5_cube_median (BEST_CFG)"}
    _atomic(out, RES)
    print("section4:", out["section4"])


GOALS_TOUGH = ["letter_O", "letter_T", "letter_S", "letter_X", "letter_L", "letter_I",
              "two_squares", "quadrant_0"]   # same 8 as eval_extended.py's GOALS_TOUGH


def _oracle_predict(q_uv0, cand_idx, bank):
    """cand_idx: 1-D LongTensor/array of bank row indices. -> per-candidate
    (mm_err, swept_err, pred_xy1) is NOT returned here -- caller supplies
    true_xy1_moved/true_xy1_swept and picks the argmin itself; this just
    yields (idx, pred_xy1) pairs, kept separate from error scoring so the
    same helper serves mm-err, swept-err, and "just rasterise" callers."""
    for di in cand_idx:
        di_t = int(di)
        pred_uv1, *_ = transfer_displacement(q_uv0, bank.uv0[di_t], bank.duv[di_t], bank.moved[di_t])
        yield di_t, pred_uv1


def section5_oracle_controls(out):
    """(coordinator caveat i) N=36 was tiny and best-of-top-N-by-distance
    alone cannot tell "the metric ranks well" apart from "best-of-N is
    optimistically biased whenever you search N candidates for the best of
    N noisy outcomes" -- ANY N candidates, even RANDOM ones, look better
    than a single random pick, purely from order statistics. The control:
    best-of-N-RANDOM vs best-of-top-N-BY-DISTANCE, same N, same rows. If
    top-N does not clearly beat random-N, the metric is not doing better
    than chance at surfacing good donors."""
    if "section5" in out:
        print("skip section5 (already done)"); return
    bank = TransitionBank.load(str(ARTIFACTS / "bank.pt"))
    bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, BEST_CFG)
    ch = [torch.load(f, map_location="cpu", weights_only=False)
         for f in sorted(glob.glob(str(D / "test_chains/_*_data.pt")))]
    states0 = torch.cat([d["states"] for d in ch]).float()
    states1 = torch.cat([d["states_"] for d in ch]).float()
    p_starts = torch.cat([d["p_starts"] for d in ch]).float()
    p_stops = torch.cat([d["p_stops"] for d in ch]).float()
    kinds = np.array(sum([list(d["start_kind"]) for d in ch], []))
    Qtot = states0.shape[0]

    rng = np.random.default_rng(1)
    N_PER_KIND = 128
    idx_parts = []
    for k in ("scatter", "clump"):
        pool = np.nonzero(kinds == k)[0]
        idx_parts.append(rng.choice(pool, size=min(N_PER_KIND, len(pool)), replace=False))
    idx = np.sort(np.concatenate(idx_parts))
    print(f"section5: {len(idx)} DS-0009 rows ({(kinds[idx]=='scatter').sum()} scatter, "
         f"{(kinds[idx]=='clump').sum()} clump), bank size {len(bank)}")

    q_pts, q_w, q_uv0_all, q_valid = _query_pw(states0[idx], p_starts[idx], p_stops[idx])
    t0 = time.time()
    D_full = full_distance_matrix(q_pts, q_w, bank_pts, bank_w, BEST_CFG, bank_valid=bank_valid, device="cpu")
    print(f"  full distance matrix ({len(idx)} rows): {time.time()-t0:.1f}s")
    T = len(bank)

    conditions = [("top_by_distance", 50), ("top_by_distance", 500),
                  ("random", 50), ("random", 500)]
    res = {f"{meth}_N{M}": {"mm": [], "P": [], "T": [], "O": [], "R": []} for meth, M in conditions}
    t0 = time.time()
    for row, qi in enumerate(idx):
        q_start, q_stop = p_starts[qi:qi + 1], p_stops[qi:qi + 1]
        q_uv0 = q_uv0_all[row]
        true_xy1 = states1[qi, :, :2]
        true_moved = (true_xy1 - states0[qi, :, :2]).norm(dim=-1) > MOVED_THRESH
        if not bool(true_moved.any()):
            continue
        true_xy1_moved = true_xy1[true_moved]
        order = torch.argsort(D_full[row])
        act = torch.cat([q_start[:, :2], q_stop[:, :2]], 1)
        o0 = occ_from_particles(states0[qi:qi + 1])
        o1 = occ_from_particles(states1[qi:qi + 1])
        reg = EN.swept_region(act, "cpu")
        for meth, M in conditions:
            cand = order[:M] if meth == "top_by_distance" else torch.from_numpy(
                np.random.default_rng(1000 + qi).choice(T, size=min(M, T), replace=False))
            best_err, best_pred = float("inf"), None
            for di_t, pred_uv1 in _oracle_predict(q_uv0, cand, bank):
                pred_xy1 = push_frame_to_world(pred_uv1[None], q_start, q_stop)[0]
                e = mm_error(true_xy1_moved, pred_xy1)
                if e is not None and e < best_err:
                    best_err, best_pred = e, pred_xy1
            key = f"{meth}_N{M}"
            res[key]["mm"].append(best_err)
            pred_state = states0[qi:qi + 1].clone(); pred_state[0, :, :2] = best_pred
            res[key]["P"].append(occ_from_particles(pred_state)); res[key]["T"].append(o1)
            res[key]["O"].append(o0); res[key]["R"].append(reg)
        if (row + 1) % 50 == 0:
            print(f"  {row+1}/{len(idx)} rows, {time.time()-t0:.0f}s", flush=True)

    summary = {}
    for key, v in res.items():
        if not v["mm"]:
            continue
        P, T_, O, R = map(torch.cat, (v["P"], v["T"], v["O"], v["R"]))
        summary[key] = {"mm_mean": float(np.mean(v["mm"])), "acc1": EN.acc(P, T_, O, R), "n": len(v["mm"])}
    out["section5"] = {"n_rows": len(idx), **summary}
    _atomic(out, RES)
    print("section5:", json.dumps(out["section5"], indent=1))


def section5b_slateN_tough_oracle(out):
    """Oracle slateN_tough (8-goal set): for a SUBSET of DS-0009 pools (cost
    scoped down -- see report), replace each of the pool's candidate pushes
    with its OWN oracle (best-of-N, by mm error) transferred prediction,
    then rank the pool exactly as the real slateN_tough harness does. This
    bounds what distance-based retrieval + this transfer rule could achieve
    with an ideal donor-picker -- independent of the ranking metric's own
    quality (already probed in section 1/5)."""
    if "section5b" in out:
        print("skip section5b (already done)"); return
    bank = TransitionBank.load(str(ARTIFACTS / "bank.pt"))
    bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, BEST_CFG)
    pools = [torch.load(f, map_location="cpu", weights_only=False)
            for f in sorted(glob.glob(str(D / "test_pools/pools_*.pt")))]
    Dist = {g: torch.from_numpy(dist_field_from_mask(EN.goal_mask(g))).float() for g in GOALS_TOUGH}
    T = len(bank)
    N_POOLS = 8    # of 32 -- scoped down for the ~15 min budget; stated explicitly
    N_CAND = 50    # per push; 500 not attempted here for the same reason

    conditions = [("top_by_distance", N_CAND), ("random", N_CAND)]
    caps = {f"{meth}_N{M}": {g: [] for g in GOALS_TOUGH} for meth, M in conditions}
    t0 = time.time()
    pool_ids_seen = 0
    for d in pools:
        for pi in torch.unique(d["pool_idx"]):
            if pool_ids_seen >= N_POOLS:
                break
            pool_ids_seen += 1
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            s0 = d["states"][ix[0]][None].float()
            o0 = occ_from_particles(s0)
            truth_occ = occ_for_scoring(d["states_"][ix, :, :3].float())
            t0_ = occ_for_scoring(s0[:, :, :3])
            ps_all, pe_all = d["p_starts"][ix].float(), d["p_stops"][ix].float()
            s1_all = d["states_"][ix].float()
            q_pts, q_w, q_uv0_all, _ = _query_pw(s0.expand(len(ix), -1, -1).contiguous(), ps_all, pe_all)
            D_full = full_distance_matrix(q_pts, q_w, bank_pts, bank_w, BEST_CFG,
                                          bank_valid=bank_valid, device="cpu")
            for meth, M in conditions:
                preds = []
                for i in range(len(ix)):
                    q_uv0 = q_uv0_all[i]
                    true_xy1 = s1_all[i, :, :2]
                    true_moved = (true_xy1 - s0[0, :, :2]).norm(dim=-1) > MOVED_THRESH
                    if not bool(true_moved.any()):
                        preds.append(s0[0].clone()); continue
                    true_xy1_moved = true_xy1[true_moved]
                    cand = torch.argsort(D_full[i])[:M] if meth == "top_by_distance" else torch.from_numpy(
                        np.random.default_rng(2000 + i).choice(T, size=min(M, T), replace=False))
                    best_err, best_pred = float("inf"), None
                    for di_t, pred_uv1 in _oracle_predict(q_uv0, cand, bank):
                        pred_xy1 = push_frame_to_world(pred_uv1[None], ps_all[i:i + 1], pe_all[i:i + 1])[0]
                        e = mm_error(true_xy1_moved, pred_xy1)
                        if e is not None and e < best_err:
                            best_err, best_pred = e, pred_xy1
                    st = s0[0].clone(); st[:, :2] = best_pred if best_pred is not None else s0[0, :, :2]
                    preds.append(st)
                po = occ_from_particles(torch.stack(preds))
                key = f"{meth}_N{M}"
                for g in GOALS_TOUGH:
                    vt = (lyap(truth_occ, Dist[g]) - lyap(t0_, Dist[g])).numpy()
                    vp = (lyap(po, Dist[g]) - lyap(o0, Dist[g])).numpy()
                    den = vt.mean() - vt.min()
                    if den > 1e-9:
                        caps[key][g].append(float((vt.mean() - vt[vp.argmin()]) / den))
        if pool_ids_seen >= N_POOLS:
            break
    out["section5b"] = {"n_pools": N_POOLS, "n_cand": N_CAND,
                        **{key: float(np.mean([np.mean(v) for v in gg.values() if v]))
                           for key, gg in caps.items()}}
    print(f"  section5b done in {time.time()-t0:.0f}s")
    _atomic(out, RES)
    print("section5b:", out["section5b"])


def section6_loo_action_tolerance(out):
    """(coordinator caveat ii) section2's LOO retrieved among all 63 siblings,
    whose actions can be far from the query's -- confounding "transfer
    quality" with "how much the action differs". Restrict candidates to
    siblings within 5mm start-position AND 10deg heading of the query's OWN
    action; report the qualifying fraction (some pushes may have none) and
    the LOO ceiling computed ONLY over qualifying rows."""
    if "section6" in out:
        print("skip section6 (already done)"); return
    pools = [torch.load(f, map_location="cpu", weights_only=False)
            for f in sorted(glob.glob(str(D / "test_pools/pools_*.pt")))]
    POS_TOL, ANG_TOL = 0.005, np.deg2rad(10)
    preds, truths, prevs, regs = [], [], [], []
    mm_errs = []
    n_qualify = 0
    n_total = 0
    for d in pools:
        for pi in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            s0_all, s1_all = d["states"][ix].float(), d["states_"][ix].float()
            ps_all, pe_all = d["p_starts"][ix].float(), d["p_stops"][ix].float()
            n = len(ix)
            angle = torch.atan2(pe_all[:, 1] - ps_all[:, 1], pe_all[:, 0] - ps_all[:, 0])
            for i in range(n):
                n_total += 1
                others = torch.tensor([j for j in range(n) if j != i])
                pos_d = (ps_all[others, :2] - ps_all[i, :2]).norm(dim=-1)
                ang_d = (torch.remainder(angle[others] - angle[i] + np.pi, 2 * np.pi) - np.pi).abs()
                qualify = others[(pos_d <= POS_TOL) & (ang_d <= ANG_TOL)]
                if len(qualify) == 0:
                    continue
                n_qualify += 1
                pos_dq = (ps_all[qualify, :2] - ps_all[i, :2]).norm(dim=-1)
                j = qualify[torch.argmin(pos_dq)]
                q_uv0 = world_to_push_frame(s0_all[i, :, :2][None], ps_all[i:i + 1], pe_all[i:i + 1])[0]
                donor_uv0 = world_to_push_frame(s0_all[j, :, :2][None], ps_all[j:j + 1], pe_all[j:j + 1])[0]
                donor_uv1 = world_to_push_frame(s1_all[j, :, :2][None], ps_all[j:j + 1], pe_all[j:j + 1])[0]
                donor_duv = donor_uv1 - donor_uv0
                donor_moved = donor_duv.norm(dim=-1) > MOVED_THRESH
                pred_uv1, *_ = transfer_displacement(q_uv0, donor_uv0, donor_duv, donor_moved)
                pred_xy1 = push_frame_to_world(pred_uv1[None], ps_all[i:i + 1], pe_all[i:i + 1])[0]
                pred_state = s0_all[i:i + 1].clone(); pred_state[0, :, :2] = pred_xy1
                preds.append(occ_from_particles(pred_state)); truths.append(occ_from_particles(s1_all[i:i + 1]))
                prevs.append(occ_from_particles(s0_all[i:i + 1]))
                regs.append(EN.swept_region(torch.cat([ps_all[i:i + 1, :2], pe_all[i:i + 1, :2]], 1), "cpu"))
                true_moved = (s1_all[i, :, :2] - s0_all[i, :, :2]).norm(dim=-1) > MOVED_THRESH
                if bool(true_moved.any()):
                    mm_errs.append(mm_error(s1_all[i, :, :2][true_moved], pred_xy1))
    out["section6"] = {
        "n_total_pushes": n_total, "n_qualifying_5mm_10deg": n_qualify,
        "frac_qualifying": n_qualify / n_total if n_total else None,
        "acc1_loo_action_restricted": EN.acc(*map(torch.cat, (preds, truths, prevs, regs))) if preds else None,
        "mm_error_mean": float(np.mean([x for x in mm_errs if x is not None])) if mm_errs else None,
    }
    _atomic(out, RES)
    print("section6:", out["section6"])


def main():
    out = _load()
    section1_and_3(out)
    section2(out)
    section4(out)
    section5_oracle_controls(out)
    section5b_slateN_tough_oracle(out)
    section6_loo_action_tolerance(out)
    print("wrote", RES)


if __name__ == "__main__":
    main()
