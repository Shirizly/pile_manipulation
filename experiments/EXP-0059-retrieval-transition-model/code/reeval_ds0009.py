"""Post-fix re-evaluation of DS-0009 1-step metrics (coordinator follow-up D,
2026-09-28), on ALL rows and on LEGAL-ONLY rows (ISS-010: tool-on-cube
touchdowns excluded via the `_legality.pt` companion flags written by
`audit_tool_placement.py`). Computes, for each predictor:
  * accuracy_1 (`eval_narrow.acc`, same swept-region convention as every
    other EXP-0053/EXP-0059 accuracy_1 number)
  * slateN (13-goal set, lyapunov, `eval_narrow.GOALS` -- same set
    `eval_retrieval.py` uses, for direct comparability with the frozen
    R1-sweep/CODEMAP numbers)
  * moved-cube mm error (nearest-match convention, matching
    `neighbour_rank_curve.py`/`diagnostics_r2.py`'s own mm metric -- NOT the
    index-matched convention `retrieval_debug.py` uses for its per-cube
    annotations; see that script's own docstring for why the two differ)
  * rollout_accuracy_4 -- ALL ROWS ONLY. Filtering individual rows out of a
    chain by legality breaks chain continuity (a later push's recorded
    pre-push state no longer matches the model's own composed prediction),
    so a "legal-only" rollout number would not mean what it says; skipped
    rather than silently misleading.

Predictors: `persistence`, `retrieval_1nn` (legacy bank.pt, i.e. the
distance-gate/interaction-set FIX applied but NOT bank curation -- isolates
the two effects), `retrieval_k5_cube_median` on the CURATED bank (the R1-best
config, now both fixed and curated -- the actual "after" number), and one
occupancy baseline (`nfd_3ch_narrow_l20`, "the narrow NFD" register model,
EXP-0053 C-056) for a same-subset fairness comparison.

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/reeval_ds0009.py
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
import eval_narrow as EN  # noqa: E402

from simple_mpc.adapters import (make_occ_adapter, occ_from_particles,  # noqa: E402
                                 occ_for_scoring)
from simple_mpc.learned_mpc import lyap  # noqa: E402
from Baselines.common.goals import dist_field_from_mask  # noqa: E402

from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.distance import DistanceConfig  # noqa: E402
from model.retrieval.predictor import (PersistencePredictor, NearestTransitionPredictor,  # noqa: E402
                                       RetrievalPredictor)

D = EN.D
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RES = Path(__file__).resolve().parents[1] / "results" / "reeval_ds0009_postfix.json"
MOVED_THRESH = 0.001
_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)


def _cpu(d):
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}


def _load_with_legal(pattern):
    """-> list of (dict, legal (N,) bool) aligned per source file.

    Bug found and fixed here, 2026-09-28 (EXP-0059 clean-data pass): the
    exclusion filter only skipped `_legality.pt` sidecars, not the
    `_nullflag.pt` ones `flag_null_transitions.py` also writes next to every
    `pools_<k>.pt` -- a bare `pools_*.pt` glob matches both. Not triggered
    when this script last actually ran (`pools_0_nullflag.pt`'s mtime
    postdates `results/reeval_ds0009_postfix.json`'s, confirmed via `stat`),
    so no existing recorded number here is affected, but it is live now."""
    out = []
    for f in sorted(f for f in glob.glob(pattern)
                    if "_legality" not in f and "_nullflag" not in f):
        d = _cpu(torch.load(f, map_location="cpu", weights_only=False))
        legality_path = Path(f).with_name(Path(f).stem + "_legality.pt")
        legal = ~torch.load(legality_path, map_location="cpu", weights_only=False)["illegal_0mm"]
        out.append((d, legal))
    return out


def _subset(d: dict, keep: torch.Tensor, per_row_keys) -> dict:
    """Row-subset every key that is genuinely per-row; leave anything else
    (e.g. a per-POOL `start_kind` shorter than N) untouched (unused by the
    metrics computed here)."""
    out = dict(d)
    for k in per_row_keys:
        if k in d and torch.is_tensor(d[k]) and d[k].shape[0] == keep.shape[0]:
            out[k] = d[k][keep]
        elif k in d and isinstance(d[k], list) and len(d[k]) == keep.shape[0]:
            out[k] = [x for x, m in zip(d[k], keep.tolist()) if m]
    return out


CHAIN_KEYS = ["states", "states_", "p_starts", "p_stops", "angles", "chain_env", "chain_step",
             "valid", "single_layer", "start_kind"]
POOL_KEYS = ["states", "states_", "p_starts", "p_stops", "angles", "pool_idx", "valid"]


def moved_cube_mm(pred_states, true_states0, true_states1):
    """Pooled nearest-match mm error over cubes that TRULY moved > 1mm
    (matches `neighbour_rank_curve.py`'s convention): for each row with at
    least one truly-moved cube, mean over those cubes of the distance to
    the NEAREST predicted cube (not index-matched -- robust to the
    predictor not preserving per-cube identity in the same way truth does)."""
    true_moved = (true_states1[:, :, :2] - true_states0[:, :, :2]).norm(dim=-1) > MOVED_THRESH
    mms = []
    for i in range(pred_states.shape[0]):
        if not bool(true_moved[i].any()):
            continue
        d_ = torch.cdist(true_states1[i, true_moved[i], :2], pred_states[i, :, :2])
        mms.append(float(d_.min(dim=1).values.mean()) * 1000.0)
    return float(np.mean(mms)) if mms else float("nan"), len(mms)


def eval_particle_predictor(predictor, chains, pools, Dist, legal_only: bool, chunk=128):
    r = {}
    # --- accuracy_1 + moved-cube mm, chains ---
    preds, truths, prevs, regs, kinds = [], [], [], [], []
    pred_states_all, true0_all, true1_all = [], [], []
    for d, legal in chains:
        keep = legal if legal_only else torch.ones_like(legal)
        dd = _subset(d, keep, CHAIN_KEYS)
        if dd["states"].shape[0] == 0:
            continue
        s0, s1 = dd["states"].float(), dd["states_"].float()
        p = predictor.predict_particles(s0, dd["p_starts"].float(), dd["p_stops"].float())
        pred_states_all.append(p); true0_all.append(s0); true1_all.append(s1)
        act = torch.cat([dd["p_starts"][:, :2], dd["p_stops"][:, :2]], 1).float()
        preds.append(occ_from_particles(p).float()); truths.append(occ_from_particles(s1).float())
        prevs.append(occ_from_particles(s0).float()); regs.append(EN.swept_region(act, "cpu").cpu())
        kinds += list(dd["start_kind"])
    P, T, O, R = map(torch.cat, (preds, truths, prevs, regs))
    r["accuracy_1"] = EN.acc(P, T, O, R)
    r["n_chain_rows"] = int(len(P))
    kinds = np.array(kinds)
    for k in ("scatter", "clump"):
        ix = torch.from_numpy(np.nonzero(kinds == k)[0])
        if len(ix):
            r[f"accuracy_1_{k}"] = EN.acc(P[ix], T[ix], O[ix], R[ix])
    mm, n_mm = moved_cube_mm(torch.cat(pred_states_all), torch.cat(true0_all), torch.cat(true1_all))
    r["moved_cube_mm"] = mm
    r["n_moved_rows"] = n_mm

    # --- slateN, pools (row-level filter is safe: independent 1-push candidates) ---
    caps = {g: [] for g in EN.GOALS}
    n_pools = 0
    for d, legal in pools:
        keep = legal if legal_only else torch.ones_like(legal)
        dd = _subset(d, keep, POOL_KEYS)
        for pi in torch.unique(dd["pool_idx"]):
            ix = torch.nonzero(dd["pool_idx"] == pi)[:, 0]
            if len(ix) < 2:
                continue
            n_pools += 1
            s0 = dd["states"][ix[0]][None].float()
            o0 = occ_from_particles(s0)
            truth_occ = occ_for_scoring(dd["states_"][ix, :, :3].float())
            t0 = occ_for_scoring(s0[:, :, :3])
            states0_batch = s0.expand(len(ix), -1, -1).contiguous()
            pred_states = []
            for i in range(0, len(ix), chunk):
                pred_states.append(predictor.predict_particles(
                    states0_batch[i:i + chunk], dd["p_starts"][ix][i:i + chunk].float(),
                    dd["p_stops"][ix][i:i + chunk].float()))
            po = occ_from_particles(torch.cat(pred_states)).float()
            for g in EN.GOALS:
                vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
                vp = (lyap(po, Dist[g]) - lyap(o0, Dist[g])).numpy()
                den = vt.mean() - vt.min()
                if den > 1e-9:
                    caps[g].append(float((vt.mean() - vt[vp.argmin()]) / den))
    r["slateN"] = float(np.mean([np.mean(v) for v in caps.values() if v])) if any(caps.values()) else None
    r["n_pools"] = n_pools

    # --- rollout_accuracy_4: ALL ROWS ONLY (chain continuity) ---
    if not legal_only:
        roll = {kk: [] for kk in range(1, 5)}
        for d, _legal in chains:
            E = int(d["chain_env"].max()) + 1
            for e in range(E):
                rows = [int(i) for i in torch.nonzero(d["chain_env"] == e)[:, 0]]
                rows = sorted(rows, key=lambda i: int(d["chain_step"][i]))[:4]
                cur_particles = d["states"][rows[0]][None].float()
                reg = torch.zeros(1, 64, 64, dtype=torch.bool)
                for kk, i in enumerate(rows, 1):
                    p_start = d["p_starts"][i][None].float(); p_stop = d["p_stops"][i][None].float()
                    act = torch.cat([p_start[:, :2], p_stop[:, :2]], 1)
                    cur_particles = predictor.predict_particles(cur_particles, p_start, p_stop)
                    cur_occ = occ_from_particles(cur_particles).float()
                    reg = reg | EN.swept_region(act, "cpu").cpu()
                    truth = occ_from_particles(d["states_"][i][None].float())
                    start_occ = occ_from_particles(d["states"][rows[0]][None].float())
                    roll[kk].append((cur_occ.cpu(), truth.cpu(), start_occ.cpu(), reg.clone()))
        for kk, L_ in roll.items():
            if L_:
                Pk, Tk, Ok, Rk = (torch.cat([x[j] for x in L_]) for j in range(4))
                r[f"rollout_accuracy_{kk}"] = EN.acc(Pk, Tk, Ok, Rk)
    return r


def eval_occ_model(model_id, chains, pools, Dist, legal_only: bool, dev="cpu"):
    ad = make_occ_adapter(model_id, dev, "corner")
    r = {}
    preds, truths, prevs, regs, kinds = [], [], [], [], []
    for d, legal in chains:
        keep = legal if legal_only else torch.ones_like(legal)
        dd = _subset(d, keep, CHAIN_KEYS)
        if dd["states"].shape[0] == 0:
            continue
        act = torch.cat([dd["p_starts"][:, :2], dd["p_stops"][:, :2]], 1).float()
        o0 = occ_from_particles(dd["states"].float(), dev); o1 = occ_from_particles(dd["states_"].float(), dev)
        with torch.no_grad():
            p = torch.cat([ad.predict_step(o0[i:i + 128], act[i:i + 128].to(dev)) for i in range(0, len(act), 128)])
        preds.append(p.float().cpu()); truths.append(o1.float().cpu()); prevs.append(o0.float().cpu())
        regs.append(EN.swept_region(act, "cpu").cpu()); kinds += list(dd["start_kind"])
    P, T, O, R = map(torch.cat, (preds, truths, prevs, regs))
    r["accuracy_1"] = EN.acc(P, T, O, R)
    r["n_chain_rows"] = int(len(P))

    caps = {g: [] for g in EN.GOALS}
    n_pools = 0
    for d, legal in pools:
        keep = legal if legal_only else torch.ones_like(legal)
        dd = _subset(d, keep, POOL_KEYS)
        for pi in torch.unique(dd["pool_idx"]):
            ix = torch.nonzero(dd["pool_idx"] == pi)[:, 0]
            if len(ix) < 2:
                continue
            n_pools += 1
            act_all = torch.cat([dd["p_starts"][:, :2], dd["p_stops"][:, :2]], 1).float()
            s0 = dd["states"][ix[0]][None].float()
            o0 = occ_from_particles(s0, dev)
            truth_occ = occ_for_scoring(dd["states_"][ix, :, :3].float()); t0 = occ_for_scoring(s0[:, :, :3])
            with torch.no_grad():
                po = ad.predict_step(o0.expand(len(ix), -1, -1).contiguous(), act_all[ix].to(dev)).float().cpu()
            for g in EN.GOALS:
                vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
                vp = (lyap(po, Dist[g]) - lyap(o0.cpu(), Dist[g])).numpy()
                den = vt.mean() - vt.min()
                if den > 1e-9:
                    caps[g].append(float((vt.mean() - vt[vp.argmin()]) / den))
    r["slateN"] = float(np.mean([np.mean(v) for v in caps.values() if v])) if any(caps.values()) else None
    r["n_pools"] = n_pools
    return r


def main():
    chains = _load_with_legal(str(D / "test_chains/_*_data.pt"))
    pools = _load_with_legal(str(D / "test_pools/pools_*.pt"))
    n_chain = sum(len(g) for _, g in chains); n_chain_legal = sum(int(g.sum()) for _, g in chains)
    n_pool = sum(len(g) for _, g in pools); n_pool_legal = sum(int(g.sum()) for _, g in pools)
    print(f"test_chains: {n_chain} rows, {n_chain_legal} legal ({100*n_chain_legal/n_chain:.1f}%)")
    print(f"test_pools: {n_pool} rows, {n_pool_legal} legal ({100*n_pool_legal/n_pool:.1f}%)")

    Dist = {g: torch.from_numpy(dist_field_from_mask(EN.goal_mask(g))).float() for g in EN.GOALS}

    bank_legacy = TransitionBank.load(str(ARTIFACTS / "bank.pt"))
    bank_curated = TransitionBank.load_curated(str(ARTIFACTS / "bank_curated.pt"))

    predictors = {
        "persistence": lambda: PersistencePredictor(),
        "retrieval_1nn_legacy_bank_fixed_transfer": lambda: NearestTransitionPredictor(bank_legacy),
        "retrieval_k5_cube_median_curated": lambda: RetrievalPredictor(
            bank_curated, cfg=DistanceConfig(**_CAPPED), k=5, aggregation="cube_median"),
    }

    out = json.loads(RES.read_text()) if RES.exists() else {}
    for name, factory in predictors.items():
        for legal_only in (False, True):
            key = f"{name}__{'legal' if legal_only else 'all'}"
            if key in out:
                print("skip (already scored):", key); continue
            t0 = time.time()
            predictor = factory()
            r = eval_particle_predictor(predictor, chains, pools, Dist, legal_only)
            r["wall_s"] = time.time() - t0
            out[key] = r
            RES.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(RES) + ".tmp"; Path(tmp).write_text(json.dumps(out, indent=1)); import os; os.replace(tmp, RES)
            print(f"{key:55s} acc1 {r['accuracy_1']:+.3f}  slateN "
                 f"{r['slateN'] if r['slateN'] is None else round(r['slateN'], 3)}  mm "
                 f"{r['moved_cube_mm']:.2f} (n={r['n_moved_rows']})  "
                 f"rollout4 {r.get('rollout_accuracy_4', float('nan')):+.3f}  ({r['wall_s']:.1f}s)",
                 flush=True)

    for model_id in ("nfd_3ch_narrow_l20",):
        for legal_only in (False, True):
            key = f"{model_id}__{'legal' if legal_only else 'all'}"
            if key in out:
                print("skip (already scored):", key); continue
            t0 = time.time()
            r = eval_occ_model(model_id, chains, pools, Dist, legal_only)
            r["wall_s"] = time.time() - t0
            out[key] = r
            tmp = str(RES) + ".tmp"; Path(tmp).write_text(json.dumps(out, indent=1)); import os; os.replace(tmp, RES)
            print(f"{key:55s} acc1 {r['accuracy_1']:+.3f}  slateN {round(r['slateN'], 3)}  "
                 f"({r['wall_s']:.1f}s)", flush=True)
    print("wrote", RES)


if __name__ == "__main__":
    main()
