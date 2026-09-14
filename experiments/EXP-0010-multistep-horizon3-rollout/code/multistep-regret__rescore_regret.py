"""Re-score terminal (horizon-3) slateN regret under mass_in_region and
signed_mass_in_region, on the SAME held-out 15-slate test split used to
train the multistep NFD/linear models (seed 0, 35/15 of the 50 slates
per slates_multistep dataset). lyapunov is carried forward for reference
(it is where the original rollout numbers were measured and are known to
be saturated).

Why this exists: experiments/temp/multistep-rollout/rollout.py's terminal
numbers pooled ALL slates (not the held-out 15), which is not comparable
to the train/test split used to fit the multistep models -- the untrained
baselines there were scored on slates some models had trained on. Here
every model (including untrained NFD, persistence, random) is scored on
identically the same 15 held-out slates per dataset, for both datasets.

Reuses (does not redefine):
  - rollout.py: load_dataset (raw .pt files, unfiltered row alignment),
    occ_of/px_of, make_linear_step (model0001_switched/global, hybrid94),
    make_nfd_step (untrained NFD inference step), RawStub, geometry
    constants, MODEL-0002 descriptor-only readout pattern.
  - train_nfd_multistep.py: make_split (seed 0, 35/15), NFD fine-tuned
    checkpoints nfd_lam{0.3,0.5,0.7,0.9}.pth, nfd_step_diff/build_nfd_geometry
    (used here in eval/no-grad mode only).
  - Baselines/common/goals.py: mass_in_region, signed_mass_in_region,
    slate_n_capture -- verbatim, per METRICS.md's "generalised to an
    arbitrary VALUE function" section.
  - control_utility_test.py: lyapunov, lyapunov_weights.

Wins/losses/ties: METRICS.md's cross-model-agreement definition (see
EXP-0010's recompute_wlt.py::wlt_fixed, the CORRECTED version -- tie iff
both models pick the same candidate row; otherwise compare the TRUE
terminal value with the correct higher/lower-is-better sense). The task
brief explicitly warns off the strict argmax-vs-true-argmax definition
that reads 0 everywhere; we do not use that. Reported vs `persistence`
(does closed-loop chaining beat not moving at all) and, for the NFD
family specifically, vs `nfd_untrained` (does multistep fine-tuning help).
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/multistep-rollout")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/multistep-nfd")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/stage3-slaten")

import numpy as np
import torch

from rollout import (
    load_dataset, DATASETS, BOUNDS, GRID, PLATE_PX, WS_MIN, WS_MAX, DEVICE, REPO,
    occ_of, px_of, make_linear_step, make_nfd_step, RawStub, SD,
)
from fit_linear_foresight import actions_to_pixels, canonicalise, swept_region_mask, metrics
from Baselines.LinearForesight.model import bin_index as switched_bin_index
from control_utility_test import lyapunov, lyapunov_weights
from Baselines.common.goals import mass_in_region, signed_mass_in_region, slate_n_capture
from descriptors_d import push_frame_full_descriptors, slices_d
from descriptors_b import world_to_pushframe_px
from eval_slaten_latent import com_world_pixel, bilinear_sample
from utils import git_provenance
from transforms.functional import draw_plate_soft
from Baselines.NFD.predictor import NFDPredictor, _plate_geometry_px

from train_nfd_multistep import make_split, LAMBDAS as NFD_LAMBDAS, build_nfd_geometry, nfd_step_diff

OUT_DIR = "/home/alon/Code/pile_manipulation/experiments/temp/multistep-regret"
NFD_CKPT = f"{REPO}/Baselines/NFD/runs/nfd_3ch/unet_best.pth"
NFD_FT_DIR = f"{REPO}/experiments/temp/multistep-nfd"
SEED = 0
K_FIXED = 32
K_REPS = 50

VALUE_FNS = ["lyapunov", "mass_in_region", "signed_mass_in_region"]


def build_test_data():
    """Load both datasets, restrict to the held-out 15-slate test split
    (seed 0, 35/15 of 50), asserting the split is identical in shape for
    both datasets (same index-based split; consistency of the underlying
    slate_idx values is asserted too)."""
    train_slates, test_slates = make_split(seed=SEED)
    print(f"split (seed={SEED}): {len(train_slates)} train / {len(test_slates)} test slates")
    print(f"  test slates: {sorted(test_slates)}")
    assert train_slates.isdisjoint(test_slates)

    ds = {}
    for tag, root in DATASETS.items():
        data, _ = load_dataset(root)
        n_slates = data["n_slates"]
        assert n_slates == 50, f"{tag}: expected 50 slates, got {n_slates}"
        slate_id = data["slate_id"].numpy()
        uniq = np.unique(slate_id)
        assert set(uniq.tolist()) == set(range(50)), (
            f"{tag}: slate_idx values are not exactly 0..49, split indices would not "
            f"line up with train_nfd_multistep.py's make_split -- got {sorted(uniq.tolist())[:5]}...")
        is_test = np.isin(slate_id, list(test_slates))
        occ0 = occ_of(data["S0"]); occ1 = occ_of(data["S1"])
        occ2 = occ_of(data["S2"]); occ3 = occ_of(data["S3"])
        A = [data["A0"].to(DEVICE), data["A1"].to(DEVICE), data["A2"].to(DEVICE)]
        ang = data["ANG"].to(DEVICE)
        length_m = [(A[k][:, 2:4] - A[k][:, 0:2]).norm(dim=-1) for k in range(3)]
        s_px, e_px = zip(*[px_of(A[k]) for k in range(3)])
        s_px = [s.to(DEVICE) for s in s_px]; e_px = [e.to(DEVICE) for e in e_px]

        idx = torch.from_numpy(is_test)
        ds[tag] = dict(
            occ_true=[occ0[idx], occ1[idx], occ2[idx], occ3[idx]],
            A=[a[idx] for a in A], ang=ang[idx],
            s_px=[s[idx] for s in s_px], e_px=[e[idx] for e in e_px],
            length_m=[l[idx] for l in length_m],
            slate_id=slate_id[is_test],
        )
        n_test_rows = int(is_test.sum())
        print(f"  [{tag}] {n_test_rows} test rows ({len(test_slates)} slates x 128 envs)")
    return ds, sorted(train_slates), sorted(test_slates)


def build_models(ckpt0001, ckpt_hy):
    """Return dict: name -> step_fn(occ_in, k, sl_idx_tensor) for CLOSED-LOOP-
    capable image models (all but MODEL-0002 descriptor-only)."""
    step_fn_sw, _ = make_linear_step(ckpt0001, desc_dim=0, switched=True)
    step_fn_gl, _ = make_linear_step(ckpt0001, desc_dim=0, switched=False)
    step_fn_hy, _ = make_linear_step(ckpt_hy, desc_dim=94, switched=True)

    nfd_geom = build_nfd_geometry()
    base_pred = NFDPredictor(NFD_CKPT, channels=3, name="nfd_untrained")
    base_pred.model.to(DEVICE).eval()

    nfd_models = {"nfd_untrained": base_pred.model}
    for lam in NFD_LAMBDAS:
        m = NFDPredictor(NFD_CKPT, channels=3, name=f"nfd_lam{lam}").model
        sd = torch.load(f"{NFD_FT_DIR}/nfd_lam{lam}.pth", map_location=DEVICE)
        m.load_state_dict(sd)
        m.to(DEVICE).eval()
        nfd_models[f"nfd_lam{lam}"] = m

    def make_nfd_fn(model):
        def fn(occ_in, k, A_k_start, A_k_stop, ang_k):
            with torch.no_grad():
                return nfd_step_diff(model, nfd_geom, occ_in, A_k_start, A_k_stop, ang_k)
        return fn

    return dict(
        model0001_switched=("linear", step_fn_sw),
        model0001_global=("linear", step_fn_gl),
        hybrid94=("linear", step_fn_hy),
        **{name: ("nfd", make_nfd_fn(m)) for name, m in nfd_models.items()},
    )


def value_of(name, occ, dw_corner, mask_corner):
    if name == "lyapunov":
        return lyapunov(occ, dw_corner)
    elif name == "mass_in_region":
        return mass_in_region(occ, mask_corner)
    elif name == "signed_mass_in_region":
        return signed_mass_in_region(occ, mask_corner)
    raise ValueError(name)


HIGHER_IS_BETTER = {"lyapunov": False, "mass_in_region": True, "signed_mass_in_region": True}


def slate_capture(vp, vt, higher, slate_id, uniq):
    if not higher:
        vp = -vp; vt = -vt
    caps = []
    for sl in uniq:
        m = slate_id == sl
        p, t = vp[m], vt[m]
        bi = int(np.argmax(p))
        chosen = t[bi]; best = t.max(); mean = t.mean()
        denom = best - mean
        if abs(denom) > 1e-9:
            caps.append((chosen - mean) / denom)
    return np.array(caps)


def slate_capture_k(vp, vt, higher, slate_id, uniq, k=K_FIXED, reps=K_REPS, seed=0):
    if not higher:
        vp = -vp; vt = -vt
    rng = np.random.default_rng(seed)
    caps = []
    for sl in uniq:
        m = slate_id == sl
        p, t = vp[m], vt[m]
        n = len(p)
        kk = min(k, n)
        for _ in range(reps):
            idx = rng.choice(n, size=kk, replace=False)
            ps, ts = p[idx], t[idx]
            bi = int(np.argmax(ps))
            chosen = ts[bi]; best = ts.max(); mean = ts.mean()
            denom = best - mean
            if abs(denom) > 1e-9:
                caps.append((chosen - mean) / denom)
    return np.array(caps)


def wlt_cross_model_agreement(vp_a, vp_b, vt, higher, slate_id, uniq):
    """Cross-model-agreement wins/losses/ties (METRICS.md; the CORRECTED
    sense per EXP-0010/recompute_wlt.py::wlt_fixed): tie iff both models'
    argmax/argmin pick the SAME row; else compare the true terminal value
    with the metric's own higher/lower-is-better sense."""
    w = l = t = 0
    for sl in uniq:
        m = slate_id == sl
        pa, pb, tt = vp_a[m], vp_b[m], vt[m]
        ia = int(np.argmax(pa)) if higher else int(np.argmin(pa))
        ib = int(np.argmax(pb)) if higher else int(np.argmin(pb))
        if ia == ib:
            t += 1
            continue
        va, vb = tt[ia], tt[ib]
        if higher:
            if va > vb: w += 1
            elif va < vb: l += 1
            else: t += 1
        else:
            if va < vb: w += 1
            elif va > vb: l += 1
            else: t += 1
    return w, l, t


def paired_sem(x):
    x = np.asarray(x)
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def run_dataset(tag, d, ckpt0001, ckpt_hy, ckpt0002, dw_corner, mask_corner):
    print(f"\n=== {tag} (held-out test only) ===")
    occ0, occ1, occ2, occ3 = d["occ_true"]
    A = d["A"]; ang = d["ang"]; s_px = d["s_px"]; e_px = d["e_px"]; length_m = d["length_m"]
    slate_id = d["slate_id"]; uniq = np.unique(slate_id)
    N = occ0.shape[0]
    print(f"  N={N} rows, {len(uniq)} slates")

    models = build_models(ckpt0001, ckpt_hy)
    BATCH = 512

    def chunked_linear(fn, occ_in, k):
        outs = []
        for i in range(0, occ_in.shape[0], BATCH):
            sl = slice(i, i + BATCH)
            outs.append(fn(occ_in[sl], s_px[k][sl], e_px[k][sl], length_m[k][sl]))
        return torch.cat(outs, dim=0)

    def chunked_nfd(fn, occ_in, k):
        outs = []
        for i in range(0, occ_in.shape[0], BATCH):
            sl = slice(i, i + BATCH)
            outs.append(fn(occ_in[sl], k, A[k][sl, 0:2], A[k][sl, 2:4], ang[sl, k]))
        return torch.cat(outs, dim=0)

    occ_true = [occ0, occ1, occ2, occ3]
    preds_terminal = {}   # name -> mode -> occ3_hat
    acc_by_model = {}

    for name, (kind, fn) in models.items():
        for mode in ("teacher_forced", "closed_loop"):
            cur = occ0
            preds = []
            for k in range(3):
                inp = occ_true[k] if mode == "teacher_forced" else cur
                if kind == "linear":
                    cur = chunked_linear(fn, inp, k)
                else:
                    cur = chunked_nfd(fn, inp, k)
                preds.append(cur)
            acc_steps = {}
            for st in range(3):
                region = swept_region_mask(s_px[st], e_px[st], (GRID, GRID),
                                            0.5 * PLATE_PX + 2.0, 0.5 * PLATE_PX)
                m = metrics(preds[st], occ_true[st + 1], occ_true[st], region=region)
                acc_steps[st] = float(m["accuracy"])
            acc_by_model.setdefault(name, {})[mode] = acc_steps
            preds_terminal.setdefault(name, {})[mode] = preds[2]
        print(f"  {name:16s} acc(step3) TF={acc_by_model[name]['teacher_forced'][2]:+.4f} "
              f"CL={acc_by_model[name]['closed_loop'][2]:+.4f}")

    # persistence + random (closed_loop == teacher_forced structurally)
    preds_terminal["persistence"] = {"teacher_forced": occ0, "closed_loop": occ0}
    acc_by_model["persistence"] = {"teacher_forced": {}, "closed_loop": {}}
    rng_perm = torch.Generator(device=DEVICE).manual_seed(1)
    perm = torch.randperm(N, device=DEVICE, generator=rng_perm)
    preds_terminal["random"] = {"teacher_forced": occ3[perm], "closed_loop": occ3[perm]}

    all_model_names = list(models) + ["persistence", "random", "oracle"]

    # ---- MODEL-0002 descriptor-only: TEACHER-FORCED terminal ranking only ----
    ops0002 = [o.to(DEVICE) for o in ckpt0002["ops"]]
    bin_edges0002 = ckpt0002["bin_edges"].to(DEVICE)

    def desc0002_predict(occ_in, s_px_k, e_px_k, length_m_k):
        local0 = push_frame_full_descriptors(occ_in, s_px_k, e_px_k)
        gm0 = occ_in.sum(dim=(-2, -1)) / (occ_in.shape[-2] * occ_in.shape[-1])
        desc0 = torch.cat([gm0[:, None], length_m_k[:, None], local0], dim=1)
        bins = switched_bin_index(length_m_k, bin_edges0002)
        pred = torch.empty_like(desc0)
        for b in range(len(ops0002)):
            m = bins == b
            if bool(m.any()):
                pred[m] = (ops0002[b] @ desc0[m].T).T
        return pred

    sp_d2 = world_to_pushframe_px(A[2][:, 0:2])
    ep_d2 = world_to_pushframe_px(A[2][:, 2:4])
    pred_desc3 = desc0002_predict(occ2, s_px[2], e_px[2], length_m[2])
    com_row3 = pred_desc3[:, SD["com"].start]
    com_col3 = pred_desc3[:, SD["com"].start + 1]
    mass3_hat = pred_desc3[:, SD["global_mass"].start] * (GRID * GRID)
    wr3, wc3 = com_world_pixel(com_row3, com_col3, sp_d2, ep_d2, H=GRID, W=GRID)

    def desc0002_value(value_fn_name):
        if value_fn_name == "lyapunov":
            return bilinear_sample(dw_corner, wr3, wc3).cpu().numpy()
        mask_f = mask_corner.float()
        m_at_com = bilinear_sample(mask_f, wr3, wc3)
        if value_fn_name == "mass_in_region":
            return (mass3_hat * m_at_com).cpu().numpy()
        else:
            return (mass3_hat * (2 * m_at_com - 1)).cpu().numpy()

    # ---- true terminal values + degeneracy check ----
    v_true = {vf: value_of(vf, occ3, dw_corner, mask_corner).cpu().numpy() for vf in VALUE_FNS}
    v_start = {vf: value_of(vf, occ0, dw_corner, mask_corner).cpu().numpy() for vf in VALUE_FNS}
    degeneracy = {}
    for vf in VALUE_FNS:
        dv = v_true[vf] - v_start[vf]
        degeneracy[vf] = float(np.mean(np.abs(dv) < 1e-9))

    # ---- terminal slateN per (value_fn, mode, model) ----
    terminal = {}
    for vf in VALUE_FNS:
        higher = HIGHER_IS_BETTER[vf]
        vt = v_true[vf]
        for mode in ("teacher_forced", "closed_loop"):
            rows = {}
            for name in all_model_names:
                if name == "oracle":
                    vp = vt
                elif name == "random":
                    vp = value_of(vf, preds_terminal["random"][mode], dw_corner, mask_corner).cpu().numpy()
                elif name == "persistence":
                    vp = value_of(vf, preds_terminal["persistence"][mode], dw_corner, mask_corner).cpu().numpy()
                else:
                    vp = value_of(vf, preds_terminal[name][mode], dw_corner, mask_corner).cpu().numpy()
                caps = slate_capture(vp, vt, higher, slate_id, uniq)
                caps_k = slate_capture_k(vp, vt, higher, slate_id, uniq)
                wins = int((caps > 1e-6).sum()); losses = int((caps < -1e-6).sum())
                ties = len(caps) - wins - losses
                row = dict(mean_capture=float(caps.mean()) if len(caps) else float("nan"),
                           sem=paired_sem(caps), n_eff=len(caps),
                           wins=wins, losses=losses, ties=ties,
                           k32_mean=float(caps_k.mean()) if len(caps_k) else float("nan"),
                           k32_sem=paired_sem(caps_k))
                # cross-model-agreement wlt vs persistence
                if name not in ("persistence",):
                    vp_pers = value_of(vf, preds_terminal["persistence"][mode], dw_corner, mask_corner).cpu().numpy() \
                        if name != "oracle" else value_of(vf, occ0, dw_corner, mask_corner).cpu().numpy()
                    w2, l2, t2 = wlt_cross_model_agreement(vp, vp_pers, vt, higher, slate_id, uniq)
                    row["wlt_vs_persistence"] = {"wins": w2, "losses": l2, "ties": t2}
                rows[name] = row
            # MODEL-0002 descriptor: teacher-forced only
            if mode == "teacher_forced":
                vp = desc0002_value(vf)
                caps = slate_capture(vp, vt, higher, slate_id, uniq)
                caps_k = slate_capture_k(vp, vt, higher, slate_id, uniq)
                wins = int((caps > 1e-6).sum()); losses = int((caps < -1e-6).sum())
                ties = len(caps) - wins - losses
                rows["model0002_descriptor_only"] = dict(
                    mean_capture=float(caps.mean()) if len(caps) else float("nan"),
                    sem=paired_sem(caps), n_eff=len(caps), wins=wins, losses=losses, ties=ties,
                    k32_mean=float(caps_k.mean()) if len(caps_k) else float("nan"),
                    k32_sem=paired_sem(caps_k))
            terminal[f"{vf}__{mode}"] = rows

    # ---- specific NFD-finetuned-vs-untrained cross-model-agreement wlt,
    # PLUS a direct paired slateN-capture-difference test (per-slate paired
    # diff of capture fraction; a more sensitive complement to the wlt
    # cross-model-agreement counts, which collapse to small integer counts
    # at n=15 slates). ----
    nfd_vs_untrained = {}
    for vf in VALUE_FNS:
        higher = HIGHER_IS_BETTER[vf]
        vt = v_true[vf]
        nfd_vs_untrained[vf] = {}
        vp_base = value_of(vf, preds_terminal["nfd_untrained"]["closed_loop"], dw_corner, mask_corner).cpu().numpy()
        caps_base = slate_capture(vp_base, vt, higher, slate_id, uniq)
        for lam in NFD_LAMBDAS:
            vp_lam = value_of(vf, preds_terminal[f"nfd_lam{lam}"]["closed_loop"], dw_corner, mask_corner).cpu().numpy()
            w, l, t = wlt_cross_model_agreement(vp_lam, vp_base, vt, higher, slate_id, uniq)
            caps_lam = slate_capture(vp_lam, vt, higher, slate_id, uniq)
            diff = caps_lam - caps_base  # per-slate paired (n=15, same slate order both sides)
            nfd_vs_untrained[vf][lam] = {
                "wins": w, "losses": l, "ties": t, "n": len(uniq),
                "paired_capture_diff_mean": float(diff.mean()),
                "paired_capture_diff_sem": paired_sem(diff),
                "paired_capture_diff_n": len(diff),
            }

    # raw per-slate paired diff arrays (nfd_lam - nfd_untrained), for pooling
    # across datasets in main() -- not JSON-serialised here (arrays), kept
    # in a side-channel return value.
    raw_diffs = {}
    for vf in VALUE_FNS:
        higher = HIGHER_IS_BETTER[vf]
        vt = v_true[vf]
        vp_base = value_of(vf, preds_terminal["nfd_untrained"]["closed_loop"], dw_corner, mask_corner).cpu().numpy()
        caps_base = slate_capture(vp_base, vt, higher, slate_id, uniq)
        raw_diffs[vf] = {}
        for lam in NFD_LAMBDAS:
            vp_lam = value_of(vf, preds_terminal[f"nfd_lam{lam}"]["closed_loop"], dw_corner, mask_corner).cpu().numpy()
            caps_lam = slate_capture(vp_lam, vt, higher, slate_id, uniq)
            raw_diffs[vf][lam] = caps_lam - caps_base

    return dict(tag=tag, n_rows=N, n_slates=len(uniq),
                accuracy=acc_by_model, terminal=terminal,
                degeneracy_frac_dv_true_zero=degeneracy,
                nfd_finetuned_vs_untrained_closed_loop=nfd_vs_untrained), raw_diffs


def main():
    t0 = time.time()
    prov = git_provenance()
    print(f"provenance: {prov}")

    dw_corner = lyapunov_weights((GRID, GRID), "corner", DEVICE)
    mask_corner = torch.zeros((GRID, GRID), dtype=torch.bool, device=DEVICE)
    mask_corner[: GRID // 2, : GRID // 2] = True

    ckpt0001 = torch.load(f"{REPO}/weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
                           map_location=DEVICE, weights_only=False)
    ckpt_hy = torch.load(f"{REPO}/experiments/temp/stage2-slaten/operators/hybrid94_lam1.0.pt",
                          map_location=DEVICE, weights_only=False)
    ckpt0002 = torch.load(f"{REPO}/weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt",
                           map_location=DEVICE, weights_only=False)

    ds, train_slates, test_slates = build_test_data()

    all_results = {"provenance": prov, "seed": SEED,
                   "train_slates": train_slates, "test_slates": test_slates}
    pooled_diffs = {vf: {lam: [] for lam in NFD_LAMBDAS} for vf in VALUE_FNS}
    for tag, d in ds.items():
        res, raw_diffs = run_dataset(tag, d, ckpt0001, ckpt_hy, ckpt0002, dw_corner, mask_corner)
        all_results[tag] = res
        for vf in VALUE_FNS:
            for lam in NFD_LAMBDAS:
                pooled_diffs[vf][lam].extend(raw_diffs[vf][lam].tolist())

    # pooled (both datasets, n=30) paired capture-diff for the fine-tuning
    # question -- more powered than either dataset alone.
    all_results["nfd_finetuned_vs_untrained_pooled_paired_diff"] = {}
    for vf in VALUE_FNS:
        all_results["nfd_finetuned_vs_untrained_pooled_paired_diff"][vf] = {}
        for lam in NFD_LAMBDAS:
            diff = np.array(pooled_diffs[vf][lam])
            all_results["nfd_finetuned_vs_untrained_pooled_paired_diff"][vf][lam] = {
                "mean": float(diff.mean()), "sem": paired_sem(diff), "n": len(diff),
            }
            print(f"  [pooled n={len(diff)}] {vf:22s} lam={lam}: "
                  f"diff={diff.mean():+.4f} sem={paired_sem(diff):.4f}")

    with open(f"{OUT_DIR}/results_regret_massgoals.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nwrote {OUT_DIR}/results_regret_massgoals.json  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
