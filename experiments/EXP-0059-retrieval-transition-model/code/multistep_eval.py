"""EXP-0059 task 2 (multi-step headline): for each of DS-0013's 32 pools (64
candidate 3-push SEQUENCES from one shared start state, see `datasets/
DS-0013-*/DATASET.md`), roll each model out along EVERY sequence's own
recorded ACTIONS (the model's own predictions are fed back as the state for
the next push -- never the true intermediate state), then:

  * rank the 64 sequences by predicted TERMINAL value (after push 3) and
    score terminal `slateN`/`slateN_tough` against the true terminal outcome
    -- the multi-step ranking headline;
  * per-step `slateN`/`slateN_tough` (after push 1, 2, 3) -- does ranking
    quality degrade with horizon;
  * per-step rollout accuracy (cumulative swept region, `eval_narrow.acc`,
    exactly the existing single-chain `rollout_accuracy_k` convention) --
    does 1-step accuracy translate into multi-step compounding error.

Occupancy-in/out models reuse `simple_mpc.adapters.make_occ_adapter`
unchanged; particle-in/out (retrieval) models reuse `model.retrieval.
predictor` unchanged. Both share `eval_extended.py`'s `slate_capture`/`GOALS_*`
/`_dist_fields`/`_masks` helpers verbatim -- no metric is redefined here.

Usage:
    python -u multistep_eval.py --models persistence nfd_3ch_narrow_l20 ... \\
        --particle-models retrieval_1nn --retrieval-bank artifacts/bank_50k.pt \\
        --retrieval-name k5_cube_median_50k
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_narrow as EN  # noqa: E402
import eval_extended as EE  # noqa: E402
from simple_mpc.adapters import make_occ_adapter, occ_from_particles, occ_for_scoring  # noqa: E402

DS0013_DIR = REPO / "Genesis/data/narrow_l20_n20/seqpools_dsB"
RES = Path(__file__).resolve().parents[1] / "results" / "multistep_eval.json"
N_STEPS = 3


def _atomic(obj, path):
    tmp = str(path) + ".tmp"; Path(tmp).write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def load_pools(legal_only: bool = False):
    """legal_only (2026-09-28, coordinator follow-up D / ISS-010): drop an
    entire candidate SEQUENCE (all 3 of its steps) if ANY of its 3 pushes was
    flagged `illegal_0mm` by `audit_tool_placement.py` (tool on a cube at
    touchdown) -- the whole rolled-out trajectory is suspect from that step
    on, not just the one row. Reads each source file's `_k_data_legality.pt`
    companion (must already exist -- run `audit_tool_placement.py` first)."""
    files = sorted(glob.glob(str(DS0013_DIR / "_*_data.pt")))
    pools = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        legal_row = None
        if legal_only:
            legality_path = Path(f).with_name(Path(f).stem + "_legality.pt")
            legal_row = ~torch.load(legality_path, map_location="cpu", weights_only=False)["illegal_0mm"]
        rows_by_step = {}
        keep_env = None
        for k in range(N_STEPS):
            ix = torch.nonzero(d["chain_step"] == k)[:, 0]
            order = torch.argsort(d["chain_env"][ix])
            rows_by_step[k] = ix[order]  # 64 rows, sorted by chain_env (0..63)
            if legal_row is not None:
                step_legal = legal_row[rows_by_step[k]]         # (64,) aligned to chain_env order
                keep_env = step_legal if keep_env is None else (keep_env & step_legal)
        if keep_env is not None:
            for k in range(N_STEPS):
                rows_by_step[k] = rows_by_step[k][keep_env]
        pools.append((d, rows_by_step))
    return pools


from simple_mpc.learned_mpc import lyap  # noqa: E402


def slate_capture_pool(po, o0, truth_occ, t0, Dist):
    out = {}
    for g in Dist:
        vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
        vp = (lyap(po, Dist[g]) - lyap(o0.cpu() if o0.is_cuda else o0, Dist[g])).numpy()
        den = vt.mean() - vt.min()
        out[g] = float((vt.mean() - vt[vp.argmin()]) / den) if den > 1e-9 else None
    return out


def slate_capture_pool_with_raw(po, o0, truth_occ, t0, Dist):
    """Like `slate_capture_pool` but also returns the raw {vt, vp} arrays per
    goal (2026-09-28 coordinator follow-up: needed for a pool-bootstrap CI on
    the multi-step terminal values -- same shape convention as
    `eval_extended.py`'s `raw13`/`rawT` -- {"pool": id, "vt": [...], "vp": [...]})."""
    out, raw = {}, {}
    for g in Dist:
        vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
        vp = (lyap(po, Dist[g]) - lyap(o0.cpu() if o0.is_cuda else o0, Dist[g])).numpy()
        den = vt.mean() - vt.min()
        out[g] = float((vt.mean() - vt[vp.argmin()]) / den) if den > 1e-9 else None
        raw[g] = (vt.tolist(), vp.tolist())
    return out, raw


def run_occ(model_id, dev, pools, Dist13, DistTough):
    if model_id == "persistence":
        class _Persist:
            def predict_step(self, occ, act):
                return occ.clone()
        ad = _Persist()
    else:
        ad = make_occ_adapter(model_id, dev, "corner")

    caps13 = {k: {g: [] for g in EE.GOALS_13} for k in range(1, N_STEPS + 1)}
    capsT = {k: {g: [] for g in EE.GOALS_TOUGH} for k in range(1, N_STEPS + 1)}
    roll_P, roll_T, roll_O, roll_R = ({k: [] for k in range(1, N_STEPS + 1)} for _ in range(4))
    rawT_terminal = {g: [] for g in EE.GOALS_TOUGH}
    for pool_id, (d, rows_by_step) in enumerate(pools):
        r0 = rows_by_step[0]
        n_cand = len(r0)
        if n_cand < 2:
            continue   # legal_only filtered this pool down to a degenerate (or empty) candidate set
        s0 = d["states"][r0[0]][None].float()          # shared start, (1,20,7)
        o0 = occ_from_particles(s0, dev)
        t0 = occ_for_scoring(s0[:, :, :3])
        cur = o0.expand(n_cand, -1, -1).contiguous()
        reg_cum = torch.zeros(n_cand, 64, 64, dtype=torch.bool)
        for k in range(N_STEPS):
            ix = rows_by_step[k]
            act = torch.cat([d["p_starts"][ix, :2], d["p_stops"][ix, :2]], 1).float()
            with torch.no_grad():
                cur = ad.predict_step(cur, act.to(dev)).float()
            reg_cum = reg_cum | EN.swept_region(act, "cpu").cpu()
            true_s1 = d["states_"][ix].float()          # (64,20,7), true state after step k+1
            truth_occ = occ_for_scoring(true_s1[:, :, :3])
            if k + 1 == N_STEPS:
                capT, rawT = slate_capture_pool_with_raw(cur.cpu(), o0.cpu(), truth_occ, t0, DistTough)
                for g, (vt, vp) in rawT.items():
                    rawT_terminal[g].append({"pool": pool_id, "vt": vt, "vp": vp})
            else:
                capT = slate_capture_pool(cur.cpu(), o0.cpu(), truth_occ, t0, DistTough)
            cap = slate_capture_pool(cur.cpu(), o0.cpu(), truth_occ, t0, Dist13)
            for g, v in cap.items():
                if v is not None: caps13[k + 1][g].append(v)
            for g, v in capT.items():
                if v is not None: capsT[k + 1][g].append(v)
            roll_P[k + 1].append(cur.cpu()); roll_T[k + 1].append(occ_from_particles(true_s1, "cpu"))
            roll_O[k + 1].append(o0.cpu().expand(n_cand, -1, -1)); roll_R[k + 1].append(reg_cum.clone())
    r = _finish(caps13, capsT, roll_P, roll_T, roll_O, roll_R)
    return r, {"slateN_tough_terminal_raw": rawT_terminal}


def run_particle(name, predictor, pools, Dist13, DistTough, chunk=128):
    caps13 = {k: {g: [] for g in EE.GOALS_13} for k in range(1, N_STEPS + 1)}
    capsT = {k: {g: [] for g in EE.GOALS_TOUGH} for k in range(1, N_STEPS + 1)}
    roll_P, roll_T, roll_O, roll_R = ({k: [] for k in range(1, N_STEPS + 1)} for _ in range(4))
    rawT_terminal = {g: [] for g in EE.GOALS_TOUGH}
    for pool_id, (d, rows_by_step) in enumerate(pools):
        r0 = rows_by_step[0]
        n_cand = len(r0)
        if n_cand < 2:
            continue   # legal_only filtered this pool down to a degenerate (or empty) candidate set
        s0 = d["states"][r0[0]][None].float()
        o0 = occ_from_particles(s0)
        t0 = occ_for_scoring(s0[:, :, :3])
        cur_particles = s0.expand(n_cand, -1, -1).contiguous()
        reg_cum = torch.zeros(n_cand, 64, 64, dtype=torch.bool)
        for k in range(N_STEPS):
            ix = rows_by_step[k]
            p_start = d["p_starts"][ix].float(); p_stop = d["p_stops"][ix].float()
            act = torch.cat([p_start[:, :2], p_stop[:, :2]], 1)
            preds = []
            for i in range(0, n_cand, chunk):
                preds.append(predictor.predict_particles(cur_particles[i:i + chunk],
                                                         p_start[i:i + chunk], p_stop[i:i + chunk]))
            cur_particles = torch.cat(preds)
            cur_occ = occ_from_particles(cur_particles)
            reg_cum = reg_cum | EN.swept_region(act, "cpu").cpu()
            true_s1 = d["states_"][ix].float()
            truth_occ = occ_for_scoring(true_s1[:, :, :3])
            if k + 1 == N_STEPS:
                capT, rawT = slate_capture_pool_with_raw(cur_occ, o0, truth_occ, t0, DistTough)
                for g, (vt, vp) in rawT.items():
                    rawT_terminal[g].append({"pool": pool_id, "vt": vt, "vp": vp})
            else:
                capT = slate_capture_pool(cur_occ, o0, truth_occ, t0, DistTough)
            cap = slate_capture_pool(cur_occ, o0, truth_occ, t0, Dist13)
            for g, v in cap.items():
                if v is not None: caps13[k + 1][g].append(v)
            for g, v in capT.items():
                if v is not None: capsT[k + 1][g].append(v)
            roll_P[k + 1].append(cur_occ); roll_T[k + 1].append(occ_from_particles(true_s1))
            roll_O[k + 1].append(o0.expand(n_cand, -1, -1)); roll_R[k + 1].append(reg_cum.clone())
    r = _finish(caps13, capsT, roll_P, roll_T, roll_O, roll_R)
    return r, {"slateN_tough_terminal_raw": rawT_terminal}


def _finish(caps13, capsT, roll_P, roll_T, roll_O, roll_R):
    r = {"step_slateN": {}, "step_slateN_tough": {}, "step_rollout_accuracy": {}}
    for k in range(1, N_STEPS + 1):
        r["step_slateN"][k] = float(np.mean([np.mean(v) for v in caps13[k].values() if v]))
        r["step_slateN_tough"][k] = float(np.mean([np.mean(v) for v in capsT[k].values() if v]))
        P, T, O, R = (torch.cat(x[k]) for x in (roll_P, roll_T, roll_O, roll_R))
        r["step_rollout_accuracy"][k] = EN.acc(P, T, O, R)
    r["terminal_slateN"] = r["step_slateN"][N_STEPS]
    r["terminal_slateN_tough"] = r["step_slateN_tough"][N_STEPS]
    r["n_pools"] = len(caps13[1]["letter_O"]) if "letter_O" in caps13[1] else None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--particle-models", nargs="*", default=[])
    ap.add_argument("--retrieval-bank", default=None, help="bank path for a k5_cube_median "
                    "RetrievalPredictor entry named --retrieval-name (in --particle-models)")
    ap.add_argument("--retrieval-name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--legal-only", action="store_true",
                    help="2026-09-28 (ISS-010): drop every candidate SEQUENCE with an illegal "
                         "(tool-on-cube) touchdown at ANY of its 3 steps; requires "
                         "audit_tool_placement.py to have already written seqpools_dsB's "
                         "_legality.pt companions")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res_path = Path(a.out) if a.out else RES
    raw_path = res_path.with_name(res_path.stem + "_raw.json")
    out = json.loads(res_path.read_text()) if res_path.exists() else {}
    raw_out = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    res_path.parent.mkdir(parents=True, exist_ok=True)

    print("loading DS-0013 seqpools ...", "(legal-only)" if a.legal_only else "(all rows)")
    pools = load_pools(legal_only=a.legal_only)
    print(f"  {len(pools)} pools")
    Dist13 = EE._dist_fields(EE.GOALS_13); DistTough = EE._dist_fields(EE.GOALS_TOUGH)

    def _save():
        tmp = str(res_path) + ".tmp"; Path(tmp).write_text(json.dumps(out, indent=1)); os.replace(tmp, res_path)
        tmp2 = str(raw_path) + ".tmp"; Path(tmp2).write_text(json.dumps(raw_out)); os.replace(tmp2, raw_path)

    for m in a.models:
        if m in out:
            print("skip (already scored):", m); continue
        t0 = time.time()
        r, raw = run_occ(m, dev, pools, Dist13, DistTough)
        r["wall_s"] = time.time() - t0
        out[m] = r; raw_out[m] = raw; _save()
        print(f"{m:36s} terminal slateN {r['terminal_slateN']:.3f} slateN_tough "
             f"{r['terminal_slateN_tough']:.3f} step_slateN_tough {r['step_slateN_tough']} "
             f"step_rollout {r['step_rollout_accuracy']} ({r['wall_s']:.1f}s)", flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()

    for m in a.particle_models:
        if m in out:
            print("skip (already scored):", m); continue
        from model.retrieval.predictor import PersistencePredictor, NearestTransitionPredictor
        from model.retrieval.bank import TransitionBank
        from model.retrieval.distance import DistanceConfig
        from model.retrieval.predictor import RetrievalPredictor
        if m == "persistence":
            predictor = PersistencePredictor()
        elif m == "retrieval_1nn":
            bank = TransitionBank.load(str(Path(__file__).resolve().parents[1] / "artifacts" / "bank.pt"))
            predictor = NearestTransitionPredictor(bank)
        elif a.retrieval_name is not None and m == a.retrieval_name:
            bank = TransitionBank.load(a.retrieval_bank)
            _CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)
            predictor = RetrievalPredictor(bank, cfg=DistanceConfig(**_CAPPED), k=5, aggregation="cube_median")
        else:
            print("skip (not registered):", m); continue
        t0 = time.time()
        r, raw = run_particle(m, predictor, pools, Dist13, DistTough)
        r["wall_s"] = time.time() - t0
        out[m] = r; raw_out[m] = raw; _save()
        print(f"{m:36s} terminal slateN {r['terminal_slateN']:.3f} slateN_tough "
             f"{r['terminal_slateN_tough']:.3f} step_slateN_tough {r['step_slateN_tough']} "
             f"step_rollout {r['step_rollout_accuracy']} ({r['wall_s']:.1f}s)", flush=True)
    print("wrote", res_path)


if __name__ == "__main__":
    main()
