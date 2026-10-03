"""EXP-0059 R0 task 1: extended eval harness for ANY model on DS-0009 --
occupancy-in/out models (EXP-0053's OCC_ADAPTERS) AND particle-in/out
retrieval predictors, through ONE comparable set of metrics. Reuses
eval_narrow.py's `acc`/`swept_region`/`D` and eval_retrieval.py's particle
scoring path verbatim -- does not reimplement either.

Adds, on top of EXP-0053's own accuracy_1/rollout/slateN:
  * slateN on the 8-goal "tough" set (EXP-0055/0057: letter_O, letter_T,
    letter_S, letter_X, letter_L, letter_I, two_squares, quadrant_0),
    alongside the original 13-goal EXP-0053 set -- both lyapunov.
  * "cleanliness" diagnostics, applied IDENTICALLY to every model: blurred
    accuracy_1 at Gaussian sigma 1 and 2 px (blur is applied to pred, truth
    AND prev before scoring -- a symmetric smoothing of the same quantity
    `acc()` already computes, not a new metric).
  * for particle-in/out models only: moved-cube position error in mm
    (direct per-cube correspondence -- the sim never reorders particles --
    restricted to cubes whose TRUE push-frame-agnostic displacement exceeds
    1 mm, i.e. actually moved).

Usage:
    python -u eval_extended.py --models persistence nfd_3ch_narrow_l20 \\
        nfd_3ch_narrow_l20_wide linear_narrow_l20_res64 \\
        nfd_residual_worldframe_noaug_ep43 --particle-models retrieval_1nn
"""
import argparse, glob, json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fnn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
import eval_narrow as EN  # noqa: E402  (acc, swept_region, GOALS, D, goal_mask)
from simple_mpc.adapters import make_occ_adapter, occ_for_scoring, occ_from_particles  # noqa: E402
from simple_mpc.learned_mpc import lyap  # noqa: E402
from Baselines.common.goals import dist_field_from_mask, mass_in_region  # noqa: E402
from simple_mpc.value_functions import SlicedEMD  # noqa: E402

D = EN.D
GOALS_13 = EN.GOALS
GOALS_TOUGH = ["letter_O", "letter_T", "letter_S", "letter_X", "letter_L", "letter_I",
              "two_squares", "quadrant_0"]
RES = Path(__file__).resolve().parents[1] / "results" / "offline_eval_extended.json"


def _gauss_kernel1d(sigma, device):
    r = max(1, int(np.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum())[None, None, :]


def gaussian_blur_occ(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """(B,H,W) -> (B,H,W), separable Gaussian blur, same-size, reflect pad."""
    if sigma <= 0:
        return x
    x = x.float()
    k = _gauss_kernel1d(sigma, x.device)
    r = k.shape[-1] // 2
    x4 = x.unsqueeze(1)  # (B,1,H,W)
    x4 = Fnn.pad(x4, (r, r, 0, 0), mode="reflect")
    x4 = Fnn.conv2d(x4, k.unsqueeze(2))          # blur along W
    x4 = Fnn.pad(x4, (0, 0, r, r), mode="reflect")
    x4 = Fnn.conv2d(x4, k.unsqueeze(3))          # blur along H
    return x4[:, 0]


def blurred_acc(pred, truth, prev, region, sigma):
    return EN.acc(gaussian_blur_occ(pred, sigma), gaussian_blur_occ(truth, sigma),
                 gaussian_blur_occ(prev, sigma), region)


def slate_capture(po, o0, truth_occ, t0, Dist, blur_sigma=0.0):
    """One pool's slateN capture per goal, optionally blurring the MODEL'S OWN
    prediction (po, o0) before ranking -- truth_occ/t0 (which define what
    IS best) are never blurred, so this isolates whether blur changes the
    RANKING DECISION, not whether it changes the scored ground truth.
    po, o0: (n_candidates, 64, 64) / (1, 64, 64). Returns {goal: capture or
    None} (None -> degenerate pool, den<=1e-9, matching the existing
    skip-if-degenerate convention elsewhere in this file)."""
    if blur_sigma > 0:
        po = gaussian_blur_occ(po, blur_sigma)
        o0 = gaussian_blur_occ(o0, blur_sigma)
    out = {}
    for g in Dist:
        vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
        vp = (lyap(po, Dist[g]) - lyap(o0.cpu() if o0.is_cuda else o0, Dist[g])).numpy()
        den = vt.mean() - vt.min()
        out[g] = float((vt.mean() - vt[vp.argmin()]) / den) if den > 1e-9 else None
    return out


def _dist_fields(goals, dev="cpu"):
    return {g: torch.from_numpy(dist_field_from_mask(EN.goal_mask(g))).float().to(dev) for g in goals}


def _masks(goals):
    return {g: torch.from_numpy(EN.goal_mask(g)).float() for g in goals}


# --------------------------------------------------------------------------
# Correlate-with-slateN diagnostics (coordinator addition, 2026-09-28): a
# per-model scalar that (unlike blurred `accuracy`) should NOT reward a
# blurry/hedged prediction over a sharp-but-slightly-off one, so that later
# work can test which offline image metric actually tracks slateN across
# models. Reuses `simple_mpc.value_functions.SlicedEMD`'s fixed pixel-order
# precompute (its sort is over PIXEL POSITION projected onto each direction,
# independent of image content) for a genuine sliced-W1 distance between two
# arbitrary (pred, truth) mass fields, not just one field vs a fixed target
# -- `SlicedEMD.__call__` only supports the latter, so this reimplements its
# one-line integral with `a`/`b` both batched instead of `a` vs `self.t_sorted`.
_EMD_HELPER = None


def _emd_helper(dev):
    global _EMD_HELPER
    if _EMD_HELPER is None:
        _EMD_HELPER = SlicedEMD(torch.zeros(64, 64), res=32, n_dirs=16, device=dev)
    return _EMD_HELPER


def sliced_w1_pair(pred: torch.Tensor, truth: torch.Tensor, dev="cpu") -> torch.Tensor:
    """(B,64,64), (B,64,64) unnormalised mass fields -> (B,) sliced-W1 distance in
    64-px pixel units. Zero rows (no mass) are treated as such by `_normalise`'s
    clamp; callers should mask to the swept region beforehand if that is the
    intended comparison scope (this function itself performs no masking)."""
    e = _emd_helper(dev)
    a = e._down(pred.to(dev)).reshape(pred.shape[0], -1)
    b = e._down(truth.to(dev)).reshape(truth.shape[0], -1)
    ga, gb = a[:, e.order], b[:, e.order]
    g = ga - gb
    return (torch.cumsum(g, 2)[:, :, :-1].abs() * e.deltas[None]).sum(2).mean(1).cpu()


def _slate_n(pools_iter_fn, goals, Dist):
    """pools_iter_fn(pool_idx) -> yields nothing; caller supplies vt/vp per pool below.
    Kept as a thin helper so the two model kinds share the exact same formula."""
    pass  # formula inlined below (kept identical in both code paths, not abstracted
          # further to avoid a data-shape-hiding indirection layer)


# --------------------------------------------------------------------------
# Occupancy-in/out models (EXP-0053 OCC_ADAPTERS)
# --------------------------------------------------------------------------

def _predict_step_dispatch(ad, wants_particles, occ, act, states0=None):
    """Particle-aware dispatch (coordinator direction, 2026-09-28): a model
    whose underlying predictor declares `predict_step_particles` gets the
    TRUE particle state when the caller has one (`states0`, else None --
    e.g. a rollout step after the first, where only a predicted occupancy
    exists) instead of the occupancy-only `predict_step` contract. This is
    the ONLY change to this function's control flow; every other model's
    `predict_step(occ, act)` call is untouched."""
    if wants_particles:
        return ad.predictor.predict_step_particles(occ, act, states0)
    return ad.predict_step(occ, act)


def _encode(ad, occ):
    """Raster occupancy -> the model's input representation (identity except
    for soft-occupancy models, EXP-0063: `OccupancyGradientAdapter.encode_state`).
    Metric-side baselines (`prev`, rollout `start`, the pool's `o0` in vp) stay
    the raster: vp's `- lyap(o0)` is one constant per pool, so the ranking is
    unaffected either way."""
    enc = getattr(ad, "encode_state", None)
    return enc(occ) if enc is not None else occ


def eval_occ_model(model_id, dev, ch, pools, Dist13, DistTough, Masks13, MasksTough):
    if model_id == "persistence":
        class _Persist:
            def predict_step(self, occ, act):
                return occ.clone()
        ad = _Persist()
    else:
        ad = make_occ_adapter(model_id, dev, "corner")
    wants_particles = hasattr(ad, "predictor") and hasattr(ad.predictor, "predict_step_particles")
    r = {}
    preds, truths, prevs, regs, kinds = [], [], [], [], []
    for d in ch:
        act = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
        o0 = occ_from_particles(d["states"].float(), dev)
        o1 = occ_from_particles(d["states_"].float(), dev)
        with torch.no_grad():
            o0_in = _encode(ad, o0)
            p = torch.cat([_predict_step_dispatch(
                              ad, wants_particles, o0_in[i:i + 128], act[i:i + 128].to(dev),
                              d["states"][i:i + 128].to(dev) if wants_particles else None)
                          for i in range(0, len(act), 128)])
        preds.append(p.float().cpu()); truths.append(o1.float().cpu()); prevs.append(o0.float().cpu())
        regs.append(EN.swept_region(act, "cpu").cpu()); kinds += list(d["start_kind"])
    if ch:
        P, T, O, R = map(torch.cat, (preds, truths, prevs, regs))
        kinds = np.array(kinds)
        r["accuracy_1"] = EN.acc(P, T, O, R)
        for k in ("scatter", "clump"):
            ix = torch.from_numpy(np.nonzero(kinds == k)[0])
            r[f"accuracy_1_{k}"] = EN.acc(P[ix], T[ix], O[ix], R[ix])
        for sigma in (1, 2):
            r[f"accuracy_1_blur{sigma}"] = blurred_acc(P, T, O, R, sigma)
        # correlate-with-slateN diagnostic: sliced-W1 occupancy distance, swept region only
        r["occ_emd_swept"] = float(sliced_w1_pair(P * R.float(), T * R.float()).mean())
    else:
        # no chains in this corpus (e.g. DS-0011 val_pools, item 3, 2026-09-28): accuracy_1/
        # rollout/occ_emd_swept need a chain-shaped test set and are left absent, not
        # fabricated as 0 or nan-silently -- callers must check `is not None`.
        for k_ in ("accuracy_1", "accuracy_1_scatter", "accuracy_1_clump",
                  "accuracy_1_blur1", "accuracy_1_blur2", "occ_emd_swept"):
            r[k_] = None

    roll = {kk: [] for kk in range(1, 5)}
    for d in ch:
        E = int(d["chain_env"].max()) + 1
        for e in range(E):
            rows = [int(i) for i in torch.nonzero(d["chain_env"] == e)[:, 0]]
            rows = sorted(rows, key=lambda i: int(d["chain_step"][i]))[:4]
            if not rows:
                # Whole-sequence-clean corpora (EXP-0059 clean-data pass, 2026-09-28:
                # datasets/DS-0016-*/test_chains_v2_clean, split_clean_archive.py) remove an
                # entire bad chain_env at once, so chain_env indices are no longer contiguous
                # within a file -- `range(E)` still walks the original 0..max range and hits
                # the gaps. Skip rather than IndexError on `rows[0]`.
                continue
            cur = occ_from_particles(d["states"][rows[0]][None].float(), dev)
            start = cur.clone(); reg = torch.zeros(1, 64, 64, dtype=torch.bool)
            cur = _encode(ad, cur)  # model input only; `start` (metric baseline) stays the raster
            for kk, i in enumerate(rows, 1):
                act = torch.cat([d["p_starts"][i, :2], d["p_stops"][i, :2]])[None].float()
                # true particles only exist for the FIRST rollout step (`cur` after that is a
                # PREDICTED occupancy, not particles) -- matches the design doc's own rollout
                # spec (retrieve on the predicted occupancy after step 1).
                states0_step = d["states"][rows[0]][None].float().to(dev) if (wants_particles and kk == 1) else None
                with torch.no_grad():
                    cur = _predict_step_dispatch(ad, wants_particles, cur, act.to(dev), states0_step).float()
                reg = reg | EN.swept_region(act, "cpu").cpu()
                truth = occ_from_particles(d["states_"][i][None].float(), dev)
                roll[kk].append((cur.cpu(), truth.cpu(), start.cpu(), reg.clone()))
    for kk, L_ in roll.items():
        if L_:
            Pk, Tk, Ok, Rk = (torch.cat([x[j] for x in L_]) for j in range(4))
            r[f"rollout_accuracy_{kk}"] = EN.acc(Pk, Tk, Ok, Rk)

    caps13 = {g: [] for g in GOALS_13}
    capsT = {g: [] for g in GOALS_TOUGH}
    mass_err13, mass_errT = {g: [] for g in GOALS_13}, {g: [] for g in GOALS_TOUGH}
    raw13, rawT = {g: [] for g in GOALS_13}, {g: [] for g in GOALS_TOUGH}  # per-pool (pool, vt, vp)
    for d in pools:
        act_all = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
        for pi in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            s0 = d["states"][ix[0]][None].float()
            o0 = occ_from_particles(s0, dev)
            truth_occ = occ_for_scoring(d["states_"][ix, :, :3].float()); t0 = occ_for_scoring(s0[:, :, :3])
            states0_pool = s0.expand(len(ix), -1, -1).contiguous().to(dev) if wants_particles else None
            with torch.no_grad():
                po = _predict_step_dispatch(ad, wants_particles, _encode(ad, o0).expand(len(ix), -1, -1).contiguous(),
                                             act_all[ix].to(dev), states0_pool).float().cpu()
            for caps, mass_err, raw, Dist, Masks in ((caps13, mass_err13, raw13, Dist13, Masks13),
                                                     (capsT, mass_errT, rawT, DistTough, MasksTough)):
                for g in Dist:
                    vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
                    vp = (lyap(po, Dist[g]) - lyap(o0.cpu(), Dist[g])).numpy()
                    den = vt.mean() - vt.min()
                    if den > 1e-9:
                        caps[g].append(float((vt.mean() - vt[vp.argmin()]) / den))
                    raw[g].append({"pool": int(pi), "vt": vt.tolist(), "vp": vp.tolist()})
                    m_true = mass_in_region(truth_occ, Masks[g]); m_pred = mass_in_region(po, Masks[g])
                    mass_err[g].append(float((m_pred - m_true).abs().mean()))
    r["slateN_per_goal"] = {g: float(np.mean(v)) for g, v in caps13.items() if v}
    r["slateN"] = float(np.mean([np.mean(v) for v in caps13.values() if v]))
    r["slateN_tough_per_goal"] = {g: float(np.mean(v)) for g, v in capsT.items() if v}
    r["slateN_tough"] = float(np.mean([np.mean(v) for v in capsT.values() if v]))
    r["mass_in_goal_mae"] = float(np.mean([np.mean(v) for v in mass_err13.values() if v]))
    r["mass_in_goal_mae_tough"] = float(np.mean([np.mean(v) for v in mass_errT.values() if v]))
    r["n_test_rows"] = int(len(P)) if ch else 0; r["n_pools"] = int(sum(len(torch.unique(d["pool_idx"])) for d in pools))
    r["kind"] = "occ"
    return r, {"slateN_raw": raw13, "slateN_tough_raw": rawT}


# --------------------------------------------------------------------------
# Particle-in/out models (EXP-0059 retrieval predictors)
# --------------------------------------------------------------------------

def eval_particle_model(name, bank, ch, pools, Dist13, DistTough, Masks13, MasksTough,
                        moved_thresh_m=0.001, chunk=128, predictor_obj=None):
    """`predictor_obj`, if given, is used directly (any object exposing
    `predict_particles(states0, p_start, p_stop)`) instead of building one
    from `name` -- added (2026-09-28, data-scaling task) so this same harness
    can score an arbitrary `RetrievalPredictor(bank, cfg=..., k=..., ...)`
    config/bank combination without adding a new name to the hardcoded dict
    below. `name` is still used as the JSON/report key either way."""
    if predictor_obj is not None:
        predictor = predictor_obj
    else:
        from model.retrieval.predictor import PersistencePredictor, NearestTransitionPredictor
        predictor = {"persistence": lambda: PersistencePredictor(),
                    "retrieval_1nn": lambda: NearestTransitionPredictor(bank)}[name]()
    r = {}
    preds, truths, prevs, regs, kinds = [], [], [], [], []
    mm_errs = []
    for d in ch:
        act = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
        o0 = occ_from_particles(d["states"].float())
        o1 = occ_from_particles(d["states_"].float())
        pred_states = []
        for i in range(0, len(act), chunk):
            pred_states.append(predictor.predict_particles(
                d["states"][i:i + chunk].float(), d["p_starts"][i:i + chunk].float(),
                d["p_stops"][i:i + chunk].float()))
        pred_states = torch.cat(pred_states)
        p = occ_from_particles(pred_states)
        preds.append(p.float().cpu()); truths.append(o1.float().cpu()); prevs.append(o0.float().cpu())
        regs.append(EN.swept_region(act, "cpu").cpu()); kinds += list(d["start_kind"])
        # moved-cube mm error: direct per-cube correspondence (sim never reorders particles)
        true_disp = (d["states_"][:, :, :2] - d["states"][:, :, :2]).norm(dim=-1)  # (B,n) metres
        moved_mask = true_disp > moved_thresh_m
        err_mm = (pred_states[:, :, :2] - d["states_"][:, :, :2]).norm(dim=-1) * 1000.0  # (B,n) mm
        if bool(moved_mask.any()):
            mm_errs.append(err_mm[moved_mask])
    if ch:
        P, T, O, R = map(torch.cat, (preds, truths, prevs, regs))
        kinds = np.array(kinds)
        r["accuracy_1"] = EN.acc(P, T, O, R)
        for k in ("scatter", "clump"):
            ix = torch.from_numpy(np.nonzero(kinds == k)[0])
            r[f"accuracy_1_{k}"] = EN.acc(P[ix], T[ix], O[ix], R[ix])
        for sigma in (1, 2):
            r[f"accuracy_1_blur{sigma}"] = blurred_acc(P, T, O, R, sigma)
        r["occ_emd_swept"] = float(sliced_w1_pair(P * R.float(), T * R.float()).mean())
    else:
        for k_ in ("accuracy_1", "accuracy_1_scatter", "accuracy_1_clump",
                  "accuracy_1_blur1", "accuracy_1_blur2", "occ_emd_swept"):
            r[k_] = None
    if mm_errs:
        mm_all = torch.cat(mm_errs)
        r["moved_cube_mm_mean"] = float(mm_all.mean())
        r["moved_cube_mm_rms"] = float((mm_all ** 2).mean().sqrt())
        r["moved_cube_mm_n"] = int(len(mm_all))

    roll = {kk: [] for kk in range(1, 5)}
    for d in ch:
        E = int(d["chain_env"].max()) + 1
        for e in range(E):
            rows = [int(i) for i in torch.nonzero(d["chain_env"] == e)[:, 0]]
            rows = sorted(rows, key=lambda i: int(d["chain_step"][i]))[:4]
            if not rows:
                continue  # see the matching guard/comment in eval_occ_model's rollout loop
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

    caps13 = {g: [] for g in GOALS_13}
    capsT = {g: [] for g in GOALS_TOUGH}
    mass_err13, mass_errT = {g: [] for g in GOALS_13}, {g: [] for g in GOALS_TOUGH}
    raw13, rawT = {g: [] for g in GOALS_13}, {g: [] for g in GOALS_TOUGH}
    for d in pools:
        act_all = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
        for pi in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            s0 = d["states"][ix[0]][None].float()
            o0 = occ_from_particles(s0)
            truth_occ = occ_for_scoring(d["states_"][ix, :, :3].float()); t0 = occ_for_scoring(s0[:, :, :3])
            states0_batch = s0.expand(len(ix), -1, -1).contiguous()
            pred_states = []
            for i in range(0, len(ix), chunk):
                pred_states.append(predictor.predict_particles(
                    states0_batch[i:i + chunk], d["p_starts"][ix][i:i + chunk].float(),
                    d["p_stops"][ix][i:i + chunk].float()))
            pred_states = torch.cat(pred_states)
            po = occ_from_particles(pred_states).float().cpu()
            for caps, mass_err, raw, Dist, Masks in ((caps13, mass_err13, raw13, Dist13, Masks13),
                                                     (capsT, mass_errT, rawT, DistTough, MasksTough)):
                for g in Dist:
                    vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
                    vp = (lyap(po, Dist[g]) - lyap(o0.cpu(), Dist[g])).numpy()
                    den = vt.mean() - vt.min()
                    if den > 1e-9:
                        caps[g].append(float((vt.mean() - vt[vp.argmin()]) / den))
                    raw[g].append({"pool": int(pi), "vt": vt.tolist(), "vp": vp.tolist()})
                    m_true = mass_in_region(truth_occ, Masks[g]); m_pred = mass_in_region(po, Masks[g])
                    mass_err[g].append(float((m_pred - m_true).abs().mean()))
    r["slateN_per_goal"] = {g: float(np.mean(v)) for g, v in caps13.items() if v}
    r["slateN"] = float(np.mean([np.mean(v) for v in caps13.values() if v]))
    r["slateN_tough_per_goal"] = {g: float(np.mean(v)) for g, v in capsT.items() if v}
    r["slateN_tough"] = float(np.mean([np.mean(v) for v in capsT.values() if v]))
    r["mass_in_goal_mae"] = float(np.mean([np.mean(v) for v in mass_err13.values() if v]))
    r["mass_in_goal_mae_tough"] = float(np.mean([np.mean(v) for v in mass_errT.values() if v]))
    r["n_test_rows"] = int(len(P)) if ch else 0; r["n_pools"] = int(sum(len(torch.unique(d["pool_idx"])) for d in pools))
    r["kind"] = "particle"
    return r, {"slateN_raw": raw13, "slateN_tough_raw": rawT}


def _filter_flagged(d: dict, data_file: str) -> dict:
    """Drop gap_out_of_window / invalid / illegal (legality sidecar, older
    pre-ISS-010-fix corpora) / null (<1mm max per-cube xy displacement) rows
    from one loaded chain/pool dict -- the SAME criteria and threshold as
    ``Genesis/training/dataset.py::PileSweepData``'s ``exclude_flagged``
    (EXP-0059 clean-data re-collection, 2026-09-28), just applied to a raw
    per-file dict instead of through the Dataset/index-map machinery, since
    this harness reads chain/pool files directly rather than via
    PileSweepData. Every per-row tensor/list field is subset consistently by
    the same keep-mask; scalar/non-row fields (e.g. ``start_kind`` is a
    per-row list, handled) pass through unchanged."""
    n = d["states"].shape[0]
    bad = torch.zeros(n, dtype=torch.bool)
    if "gap_out_of_window" in d:
        bad |= d["gap_out_of_window"].bool()
    if "valid" in d:
        bad |= ~d["valid"].bool()
    legality_path = Path(data_file).with_name(Path(data_file).stem + "_legality.pt")
    if legality_path.exists():
        leg = torch.load(legality_path, map_location="cpu", weights_only=False)
        bad |= leg["illegal_0mm"].bool()
    disp_mm = (d["states_"][:, :, :2] - d["states"][:, :, :2]).float().norm(dim=-1).max(dim=1).values * 1000.0
    bad |= disp_mm < 1.0
    keep = (~bad).nonzero(as_tuple=True)[0]
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.shape[:1] == (n,):
            out[k] = v[keep]
        elif isinstance(v, list) and len(v) == n:
            keep_list = keep.tolist()
            out[k] = [v[i] for i in keep_list]
        else:
            out[k] = v
    return out


def _load_corpus(pools_dir=None, chains_dir=None, exclude_flagged=False):
    """pools_dir/chains_dir override DS-0009's own test_pools/test_chains -- e.g. pass
    DS-0011's val_pools to score the SAME interface there (item 3, coordinator 2026-09-28).
    DS-0011 has no chains equivalent, so `chains_dir` may resolve to nothing: callers get
    `ch=[]` and accuracy_1/rollout are skipped (guarded in eval_occ_model/eval_particle_model),
    leaving only the pool-based slateN/slateN_tough/mass_in_goal_mae -- still the SAME metric
    keys, just absent where the corpus cannot support them.

    exclude_flagged=True drops gap_out_of_window/invalid/illegal/null rows
    from every loaded file BEFORE scoring, via `_filter_flagged` (EXP-0059
    clean-data re-collection, 2026-09-28) -- default False so this remains a
    byte-identical no-op against every existing DS-0009/DS-0011 register
    number.

    **Bug found and fixed in THIS record's own earlier code, 2026-09-28**:
    the pool glob used to be the bare `"pools_*.pt"`, which (once
    `audit_tool_placement.py`/`flag_null_transitions.py` started writing
    `pools_<k>_legality.pt`/`pools_<k>_nullflag.pt` sidecars next to every
    `pools_<k>.pt`, both under `test_pools/` AND `val_pools/`) ALSO matches
    those sidecar files (`pools_0_legality.pt` starts with `pools_` and ends
    `.pt`, same as the data file) -- loading one as if it were a pool dict
    then KeyErrors downstream on `d["states"]`/`d["pool_idx"]`. Not
    triggered by any run that already landed a register number (checked:
    every prior `_load_corpus()` pools call in this experiment's history --
    `data_scaling_pools_raw.py` -- ran BEFORE the sidecar files existed,
    confirmed by LOG.md's own chronology), but it is live now that the
    sidecars are on disk, and would otherwise hit this record's own v2 test
    scoring the moment the data agent's full-scale audit adds the matching
    sidecars for `test_pools_v2`/`val_pools`. Fixed by anchoring the glob to
    the exact `pools_<digits>.pt` filename."""
    cpu = lambda d: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
    chains_dir = chains_dir or (D / "test_chains")
    pools_dir = pools_dir or (D / "test_pools")
    chain_files = sorted(glob.glob(str(Path(chains_dir) / "_*_data.pt")))
    pool_files = sorted(
        f for f in glob.glob(str(Path(pools_dir) / "pools_*.pt"))
        if re.fullmatch(r"pools_\d+\.pt", Path(f).name)
    )
    ch = [cpu(torch.load(f, map_location="cpu", weights_only=False)) for f in chain_files]
    pools = [cpu(torch.load(f, map_location="cpu", weights_only=False)) for f in pool_files]
    if exclude_flagged:
        ch = [_filter_flagged(d, f) for d, f in zip(ch, chain_files)]
        pools = [_filter_flagged(d, f) for d, f in zip(pools, pool_files)]
    return ch, pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--particle-models", nargs="*", default=[])
    ap.add_argument("--bank", default=str(Path(__file__).resolve().parents[1] / "artifacts" / "bank.pt"))
    ap.add_argument("--pools-dir", default=None,
                   help="override DS-0009's test_pools, e.g. Genesis/data/narrow_l20_n20/"
                        "val_pools to score on DS-0011 instead (item 3, 2026-09-28)")
    ap.add_argument("--chains-dir", default=None,
                   help="override DS-0009's test_chains; pass a nonexistent path (or leave a "
                        "corpus with no chains) to skip accuracy_1/rollout cleanly")
    ap.add_argument("--out", default=None, help="override the results JSON path (default: "
                                                 "results/offline_eval_extended.json)")
    ap.add_argument("--exclude-flagged", action="store_true",
                   help="drop gap_out_of_window/invalid/illegal/null rows before scoring "
                        "(EXP-0059 clean-data re-collection, 2026-09-28); default off, so "
                        "existing DS-0009/DS-0011 numbers stay reproducible byte-for-byte")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res_path = Path(a.out) if a.out else RES
    raw_path = res_path.with_name(res_path.stem + "_raw.json")
    out = json.loads(res_path.read_text()) if res_path.exists() else {}
    raw_out = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    res_path.parent.mkdir(parents=True, exist_ok=True)

    ch, pools = _load_corpus(a.pools_dir, a.chains_dir, exclude_flagged=a.exclude_flagged)
    Dist13 = _dist_fields(GOALS_13); DistTough = _dist_fields(GOALS_TOUGH)
    Masks13 = _masks(GOALS_13); MasksTough = _masks(GOALS_TOUGH)

    def _save():
        tmp = str(res_path) + ".tmp"; Path(tmp).write_text(json.dumps(out, indent=1)); os.replace(tmp, res_path)
        tmp2 = str(raw_path) + ".tmp"; Path(tmp2).write_text(json.dumps(raw_out, indent=1)); os.replace(tmp2, raw_path)

    for m in a.models:
        if m in out:
            print("skip (already scored):", m); continue
        t0 = time.time()
        try:
            r, raw = eval_occ_model(m, dev, ch, pools, Dist13, DistTough, Masks13, MasksTough)
        except KeyError:
            print("skip (not registered):", m); continue
        r["wall_s"] = time.time() - t0
        out[m] = r; raw_out[m] = raw
        _save()
        acc1 = r.get('accuracy_1')
        print(f"{m:36s} " + (f"acc1 {acc1:+.3f} blur1 {r['accuracy_1_blur1']:+.3f} blur2 {r['accuracy_1_blur2']:+.3f} "
                             if acc1 is not None else "acc1 n/a (no chains) ") +
             f"slateN {r['slateN']:.3f} slateN_tough {r['slateN_tough']:.3f} ({r['wall_s']:.1f}s)", flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()

    bank = None
    if a.particle_models and any(m != "persistence" for m in a.particle_models):
        from model.retrieval.bank import TransitionBank
        bank = TransitionBank.load(a.bank)
    for m in a.particle_models:
        if m in out:
            print("skip (already scored):", m); continue
        t0 = time.time()
        r, raw = eval_particle_model(m, bank, ch, pools, Dist13, DistTough, Masks13, MasksTough)
        r["wall_s"] = time.time() - t0
        out[m] = r; raw_out[m] = raw
        _save()
        acc1 = r.get('accuracy_1')
        print(f"{m:36s} " + (f"acc1 {acc1:+.3f} blur1 {r['accuracy_1_blur1']:+.3f} blur2 {r['accuracy_1_blur2']:+.3f} "
                             if acc1 is not None else "acc1 n/a (no chains) ") +
             f"slateN {r['slateN']:.3f} slateN_tough {r['slateN_tough']:.3f} "
             f"mm_mean {r.get('moved_cube_mm_mean', float('nan')):.2f} ({r['wall_s']:.1f}s)", flush=True)
    print("wrote", res_path, "and", raw_path)


if __name__ == "__main__":
    main()
