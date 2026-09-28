"""EXP-0059 offline evaluation of PARTICLE-IN/PARTICLE-OUT predictors on the
same clean narrow-domain test set (DS-0009) and through the same metric code
`experiments/EXP-0053-*/code/eval_narrow.py` uses for every OCC_ADAPTERS
model -- so a retrieval model's numbers sit on the same scale as the
EXP-0053 register row (best NFD: accuracy_1 ~0.51, slateN ~0.77-0.80).

Why a separate script instead of extending `eval_narrow.py` in place: every
existing OCC_ADAPTERS model is driven from a RASTERISED occupancy grid
(`ad.predict_step(occ, act) -> occ`), which is deliberately what
`simple_mpc.adapters.OccupancyGradientAdapter` abstracts over. A retrieval
model needs the query's raw cube positions to retrieve against -- occupancy
has already discarded that (many distinct particle configurations rasterise
to the same 64x64 grid). Rather than bolt a second, particle-shaped code
path onto `eval_narrow.py` (risking the existing models' scoring), this
script IMPORTS `eval_narrow` and reuses its `acc`, `swept_region`, `GOALS`,
`goal_mask`, and data directory verbatim, and only replaces the model-call
site: `predictor.predict_particles(states0, p_start, p_stop) -> states1`,
rasterised with the identical `occ_from_particles` every other model is
scored with. `eval_narrow.py` itself is untouched.

A useful side effect of staying in particle space: multi-step ROLLOUT
composes the predictor's own particle-state output as the next step's
input, so a retrieval model's rollout never passes through an extra
raster/derasterise round trip the way an occupancy-in model's would if it
tried to do the same.

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/eval_retrieval.py \\
        --predictors persistence retrieval_1nn
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
import eval_narrow as EN  # noqa: E402  (reuses acc/swept_region/GOALS/goal_mask/D)

from simple_mpc.adapters import occ_from_particles, occ_for_scoring  # noqa: E402
from simple_mpc.learned_mpc import lyap  # noqa: E402
from Baselines.common.goals import dist_field_from_mask  # noqa: E402

from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.distance import DistanceConfig  # noqa: E402
from model.retrieval.predictor import (PersistencePredictor, NearestTransitionPredictor,
                                       RetrievalPredictor)  # noqa: E402

D = EN.D  # Genesis/data/narrow_l20_n20
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RES = Path(__file__).resolve().parents[1] / "results" / "offline_eval_retrieval.json"

# A few named presets for CLI convenience; the sweep script
# (`sweep_retrieval.py`) builds `RetrievalPredictor` directly with its own
# `DistanceConfig` grid instead of going through this dict.
PREDICTOR_FACTORIES = {
    "persistence": lambda bank: PersistencePredictor(),
    "retrieval_1nn": lambda bank: NearestTransitionPredictor(bank),
    "retrieval_k5_cube_median": lambda bank: RetrievalPredictor(
        bank, cfg=DistanceConfig(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0),
        k=5, aggregation="cube_median"),
    "retrieval_k5_occ_mean": lambda bank: RetrievalPredictor(
        bank, cfg=DistanceConfig(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0),
        k=5, aggregation="occ_mean"),
}


def _cpu(d):
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}


def _load_corpus():
    import glob
    ch = [_cpu(torch.load(f, map_location="cpu", weights_only=False))
         for f in sorted(glob.glob(str(D / "test_chains/_*_data.pt")))]
    pools = [_cpu(torch.load(f, map_location="cpu", weights_only=False))
            for f in sorted(glob.glob(str(D / "test_pools/pools_*.pt")))]
    return ch, pools


def _predict_occ_chunked(predictor, states0, p_start, p_stop, chunk):
    """Prefer `predictor.predict_occ` (the k-NN occupancy-hedge aggregations
    have no single well-defined particle state -- see `predictor.py`'s
    module docstring) via `hasattr`, exactly the optional-method pattern
    `WarpedNFDPredictor.predict_occ_canonical` already uses in this
    codebase; fall back to rasterising `predict_particles`'s output with
    the same `occ_from_particles` every other model is scored with."""
    has_occ = hasattr(predictor, "predict_occ")
    out = []
    for i in range(0, len(states0), chunk):
        s0, ps, pe = states0[i:i + chunk], p_start[i:i + chunk], p_stop[i:i + chunk]
        if has_occ:
            out.append(predictor.predict_occ(s0, ps, pe))
        else:
            out.append(occ_from_particles(predictor.predict_particles(s0, ps, pe)))
    return torch.cat(out)


def evaluate(predictor, ch, pools, Dist, chunk=128) -> dict:
    r = {}
    # --- 1-step accuracy on all chain rows -----------------------------
    # Hedged occupancy (`predict_occ`, when the predictor has one) is used
    # HERE ONLY (2026-09-28 user guidance): rollout and slateN below always
    # call `predict_particles`, which every predictor guarantees returns a
    # single CLEAN, committed-to state.
    preds, truths, prevs, regs, kinds = [], [], [], [], []
    for d in ch:
        act = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
        o0 = occ_from_particles(d["states"].float())
        o1 = occ_from_particles(d["states_"].float())
        p = _predict_occ_chunked(predictor, d["states"].float(), d["p_starts"].float(),
                                 d["p_stops"].float(), chunk)
        preds.append(p.float().cpu()); truths.append(o1.float().cpu()); prevs.append(o0.float().cpu())
        regs.append(EN.swept_region(act, "cpu").cpu()); kinds += list(d["start_kind"])
    P, T, O, R = map(torch.cat, (preds, truths, prevs, regs))
    kinds = np.array(kinds)
    r["accuracy_1"] = EN.acc(P, T, O, R)
    for k in ("scatter", "clump"):
        ix = torch.from_numpy(np.nonzero(kinds == k)[0])
        r[f"accuracy_1_{k}"] = EN.acc(P[ix], T[ix], O[ix], R[ix])

    # --- rollout accuracy on chains (own PARTICLE predictions fed back) -
    roll = {kk: [] for kk in range(1, 5)}
    for d in ch:
        E = int(d["chain_env"].max()) + 1
        for e in range(E):
            rows = [int(i) for i in torch.nonzero(d["chain_env"] == e)[:, 0]]
            rows = sorted(rows, key=lambda i: int(d["chain_step"][i]))[:4]
            cur_particles = d["states"][rows[0]][None].float()
            reg = torch.zeros(1, 64, 64, dtype=torch.bool)
            for kk, i in enumerate(rows, 1):
                p_start = d["p_starts"][i][None].float()
                p_stop = d["p_stops"][i][None].float()
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

    # --- slateN on pools (soft truth, lyapunov, GOALS) -------------------
    # DS-0008 (used for validation, see `model.retrieval.val_split`) has no
    # same-state action pools -- only DS-0009's `test_pools` does -- so a
    # validation call passes `pools=[]` and slateN is reported as None
    # rather than silently averaging over nothing.
    caps = {g: [] for g in EN.GOALS}
    for d in pools:
        act_all = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
        for pi in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            s0 = d["states"][ix[0]][None].float()
            o0 = occ_from_particles(s0)
            truth_occ = occ_for_scoring(d["states_"][ix, :, :3].float())
            t0 = occ_for_scoring(s0[:, :, :3])
            states0_batch = s0.expand(len(ix), -1, -1).contiguous()
            # CLEAN particles only (never the hedged `predict_occ` blend,
            # 2026-09-28 user guidance: hedging is allowed for accuracy_1
            # scoring only -- slateN's action ranking must reflect a state
            # the model actually commits to).
            pred_states = []
            for i in range(0, len(ix), chunk):
                pred_states.append(predictor.predict_particles(
                    states0_batch[i:i + chunk],
                    d["p_starts"][ix][i:i + chunk].float(),
                    d["p_stops"][ix][i:i + chunk].float()))
            po = occ_from_particles(torch.cat(pred_states)).float().cpu()
            for g in EN.GOALS:
                vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
                vp = (lyap(po, Dist[g]) - lyap(o0.cpu(), Dist[g])).numpy()
                den = vt.mean() - vt.min()
                if den > 1e-9:
                    caps[g].append(float((vt.mean() - vt[vp.argmin()]) / den))
    if pools:
        r["slateN_per_goal"] = {g: float(np.mean(v)) for g, v in caps.items() if v}
        r["slateN"] = float(np.mean([np.mean(v) for v in caps.values() if v]))
    else:
        r["slateN_per_goal"] = {}
        r["slateN"] = None
    r["n_test_rows"] = int(len(P))
    r["n_pools"] = int(sum(len(torch.unique(d["pool_idx"])) for d in pools))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictors", nargs="*", default=["persistence", "retrieval_1nn"])
    ap.add_argument("--bank", default=str(ARTIFACTS / "bank.pt"))
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()

    out = json.loads(RES.read_text()) if RES.exists() else {}
    RES.parent.mkdir(parents=True, exist_ok=True)

    ch, pools = _load_corpus()
    Dist = {g: torch.from_numpy(dist_field_from_mask(EN.goal_mask(g))).float() for g in EN.GOALS}

    bank = None
    if any(m != "persistence" for m in a.predictors):
        bank = TransitionBank.load(a.bank)
        print(f"loaded bank: {len(bank)} transitions from {a.bank}")

    for name in a.predictors:
        if name in out:
            print("skip (already scored):", name); continue
        if name not in PREDICTOR_FACTORIES:
            print("skip (unknown predictor):", name, "known:", sorted(PREDICTOR_FACTORIES)); continue
        predictor = PREDICTOR_FACTORIES[name](bank)
        t0 = time.time()
        r = evaluate(predictor, ch, pools, Dist, chunk=a.chunk)
        r["wall_s"] = time.time() - t0
        out[name] = r
        tmp = str(RES) + ".tmp"
        Path(tmp).write_text(json.dumps(out, indent=1))
        os.replace(tmp, RES)
        slate_str = f"{r['slateN']:.3f}" if r["slateN"] is not None else "n/a"
        print(f"{name:20s} acc1 {r['accuracy_1']:+.3f} (scatter {r['accuracy_1_scatter']:+.3f}, "
             f"clump {r['accuracy_1_clump']:+.3f}) rollout " +
             " ".join(f"{r.get(f'rollout_accuracy_{k}', float('nan')):+.3f}" for k in range(1, 5)) +
             f"  slateN {slate_str}  ({r['wall_s']:.1f}s)", flush=True)
    print("wrote", RES)


if __name__ == "__main__":
    main()
