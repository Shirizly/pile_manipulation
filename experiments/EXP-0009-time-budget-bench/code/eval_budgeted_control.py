"""experiments/EXP-0009-time-budget-bench/code/eval_budgeted_control.py

RUN-0003: the CONTROL half of the time-budget experiment. RUN-0001/RUN-0002
(this same EXP dir) measured inference time and derived per-model candidate
budgets N_i(T) (results/budgets.json). This script measures slateN control
performance under those budgets.

Design (fixed by the user, see task brief -- not re-derived here):
  - Candidate pools: the EXISTING N=128-per-slate pools from
    `experiments/temp/slaten-broad/eval_slaten_broad.py` (slates_multistep,
    n20_L10mm excluded, step-0 rows grouped by slate_idx -- verified 128
    rows/slate below). Each model may rank only N_i(T) candidates, drawn
    WITHOUT replacement from the 128.
  - Normalisation is against the FULL 128-candidate pool (oracle max +
    random-pick mean) for EVERY model regardless of how many candidates it
    ranked -- `budgeted_slate_capture()` below, a variant of
    `Baselines/common/goals.py::slate_n_capture` that decouples "which
    indices the model is allowed to rank" from "which indices the true-best/
    mean-floor is computed over".
  - For N_i < 128: 5 independent subsets -> top/bottom/mean headline, PLUS
    a 50-draw distribution (mean/p5/p95) as a secondary sanity check on the
    5-draw spread (draws 0-4 of the 50 ARE the 5-draw headline, not a
    separate resample -- avoids discarding cheap-to-compute draws).

Unification of scoring path (task's explicit "single scoring path"
requirement): model FORWARD PASSES still go through model-specific code
(NFD/Schenck/GNN via `Baselines/common/eval_baseline.py`-equivalent
predictor classes, reused here via EXP-0009's own `bench.py::make_*`
wrappers -- these already fixed the device bug and are the SAME code this
EXP's own timing numbers came from, so N_i(T) and the control numbers are
at least self-consistent). Everything downstream of "model produced an
occupancy prediction (or, for descriptor-only, a predicted descriptor
vector)" -- goal mask/value-field construction, the true/predicted VALUE
computation (lyapunov / mass_in_region), slate_n_capture-style scoring,
paired sem, wins/losses/ties -- is the ONE path, reused verbatim from
`eval_slaten_broad.py` (`build_goal`, `value_true_and_pred`,
`desc_pointmass_value`, `paired_sem`) so the linear family and the
NFD/GNN/Schenck family are compared through identical value/scoring code,
not two independently-written scorers. What remains split (PROVENANCE
DOWNGRADE, documented in EXPERIMENT.md): the per-candidate PREDICTION step
itself (image reconstruction vs. particle rollout vs. descriptor readout)
is necessarily model-specific and was not further unified within budget.

Scope reductions taken under the 60-minute budget (documented, not hidden):
  - Goal shape: CORNER only (the task's own "lead with" shape). The other
    4 shapes from EXP-0008 (stripe/random_quadrant/ring_O/T) are NOT
    re-run here -- would multiply runtime and report size ~5x for a
    control-comparison whose primary question is model ranking under a
    budget, not shape sensitivity (already characterised in EXP-0008).
  - Value functions: lyapunov (lead) + mass_in_region (secondary), per the
    task brief. signed_mass_in_region (present in EXP-0008) is dropped.
  - Datasets: n20_L20mm + n20_L40mm (both, per task -- n20_L10mm excluded).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

REPO = "/home/alon/Code/pile_manipulation"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/experiments/temp/hybrid-vis-desc")
sys.path.insert(0, f"{REPO}/experiments/temp/dmdc-lenbins")
sys.path.insert(0, f"{REPO}/experiments/temp/stage3-slaten")
sys.path.insert(0, f"{REPO}/experiments/temp/slaten-broad")
sys.path.insert(0, f"{REPO}/experiments/EXP-0009-time-budget-bench/code")

from Baselines.common.data import load_cell                      # noqa: E402
from Baselines.common.goals import mass_in_region                # noqa: E402
from Baselines.GNN.perception import rasterize_nodes_as_cubes_batch  # noqa: E402
from utils import git_provenance                                 # noqa: E402
from descriptors_d import slices_d                                # noqa: E402
from descriptors_b import world_to_pushframe_px                   # noqa: E402
from eval_slaten_latent import com_world_pixel                    # noqa: E402
from eval_slaten_broad import build_goal, value_true_and_pred, desc_pointmass_value, paired_sem  # noqa: E402

import bench  # EXP-0009's own timing bench -- reused for model pre/fwd    # noqa: E402

DEVICE = bench.DEVICE
SD = slices_d()
REPO_ROOT = REPO
EXP_DIR = f"{REPO}/experiments/EXP-0009-time-budget-bench"
BUDGETS_JSON = f"{EXP_DIR}/results/budgets.json"

MULTISTEP_DATASETS = {
    # n20_L10mm DELIBERATELY EXCLUDED (task instruction: ruled problematic).
    "n20_L20mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json"),
    "n20_L40mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L40mm/manifest.json"),
}

GOAL_SHAPE = "corner"       # lead shape per task brief
VALUE_FNS = ["lyapunov", "mass_in_region"]
BUDGETS_T_MS = [2.5, 5, 10]
K_FIXED = 32                # mandatory fixed reference point (METRICS.md)
N_DRAWS_HEADLINE = 5
N_DRAWS_TOTAL = 50           # draws 0..4 ARE the headline 5, 0..49 is the distribution
POOL_N = 128

MODEL_NAMES = ["nfd", "schenck", "gnn", "model0001_switched", "model0001_global",
               "model0002_descriptor_only", "hybrid14", "hybrid94",
               "persistence", "random"]
BUDGETED_MODEL_NAMES = ["nfd", "schenck", "gnn", "model0001_switched", "model0001_global",
                         "model0002_descriptor_only", "hybrid14", "hybrid94"]


def build_model_specs():
    specs = {}
    specs["nfd"] = bench.make_nfd()
    specs["schenck"] = bench.make_schenck()
    specs["gnn"] = bench.make_gnn()
    specs["model0001_switched"] = bench.make_switched_linear(
        "model0001_switched", "weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
        desc_dim=0, switched=True)
    specs["model0001_global"] = bench.make_switched_linear(
        "model0001_global", "weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
        desc_dim=0, switched=False)
    specs["model0002_descriptor_only"] = bench.make_desc_only(
        "model0002_descriptor_only", "weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt")
    specs["hybrid14"] = bench.make_switched_linear(
        "hybrid14", "experiments/temp/stage2-slaten/operators/hybrid14_lam1.0.pt",
        desc_dim=14, switched=True)
    specs["hybrid94"] = bench.make_switched_linear(
        "hybrid94", "experiments/temp/stage2-slaten/operators/hybrid94_lam1.0.pt",
        desc_dim=94, switched=True)
    return specs


def build_batch_from_slate(slates, rows, device):
    """rows: LongTensor of indices into `slates` (all sharing the same
    occ0 start state -- the same-state-pool contract build_cache_multistep
    /build_cache_holdout already rely on)."""
    occ0_one = slates.occ0[rows[0]]
    occ0 = occ0_one.unsqueeze(0).expand(len(rows), -1, -1).contiguous().to(device)
    return dict(
        occ0=occ0,
        actions=slates.actions[rows].to(device),
        p_start=slates.p_start[rows].to(device),
        p_stop=slates.p_stop[rows].to(device),
        angle=slates.angle[rows].to(device),
        raw=slates.raw,
        workspace_min=slates.workspace_min, workspace_max=slates.workspace_max,
        H=slates.H, W=slates.W,
    )


def predict_occ_for_model(name, spec, batch):
    """Returns EITHER an (N,H,W) occupancy image tensor (image-producing
    models: nfd/schenck/gnn/model0001_*/hybrid*), OR, for
    model0002_descriptor_only, a dict of the point-mass readout ingredients
    (com pixel coords + predicted mass) so the caller can route through
    `desc_pointmass_value` (the SAME function eval_slaten_broad.py uses)."""
    H, W = batch["H"], batch["W"]
    if name == "persistence":
        return batch["occ0"].clone()
    if name == "model0002_descriptor_only":
        pre_out = spec["pre"](batch)
        pred_desc = spec["fwd"](pre_out)
        com_pred = pred_desc[:, SD["com"]]
        mass_hat = pred_desc[:, SD["global_mass"]][:, 0] * (H * W)
        wr, wc = com_world_pixel(com_pred[:, 0], com_pred[:, 1], pre_out["sp_d"], pre_out["ep_d"], H=H, W=W)
        return dict(desc_readout=True, wr=wr.cpu(), wc=wc.cpu(), mass_hat=mass_hat.cpu())
    if name == "gnn":
        pre_out = spec["pre"](batch)
        fwd_out = spec["fwd"](pre_out)
        node_xy = fwd_out[:, :, :2].detach().cpu().numpy()
        raw = pre_out["raw"]
        occ_np = rasterize_nodes_as_cubes_batch(node_xy, raw.to_pxl, raw.ctr_in_PXL, pre_out["H"], pre_out["W"])
        return torch.from_numpy(occ_np).float().cpu()
    # nfd / schenck / model0001_switched / model0001_global / hybrid14 / hybrid94
    occ_pred = spec["fwd"](spec["pre"](batch))
    if occ_pred.dim() == 4:
        occ_pred = occ_pred[:, 0]
    return occ_pred.detach().float().cpu()


def budgeted_slate_capture(value_pred_full, subset_idx, value_true_full, higher_is_better):
    """Variant of Baselines/common/goals.py::slate_n_capture: the model
    only RANKS within `subset_idx` (its own predicted value restricted to
    the drawn subset), but the oracle-best / random-floor normalisation is
    computed over `value_true_full` (the WHOLE 128-candidate pool) --
    the user's explicit design: a model that sees fewer actions must be
    penalised for possibly missing the best one.
    """
    vp = value_pred_full if higher_is_better else -value_pred_full
    vt = value_true_full if higher_is_better else -value_true_full
    sub_pred = vp[subset_idx]
    best_idx_in_subset = int(torch.argmax(sub_pred))
    chosen_global_idx = int(subset_idx[best_idx_in_subset])
    chosen = float(vt[chosen_global_idx])
    true_best = float(vt.max())
    mean_true = float(vt.mean())
    denom = true_best - mean_true
    if abs(denom) < 1e-9:
        return float("nan"), chosen if higher_is_better else -chosen
    cap = (chosen - mean_true) / denom
    goodness = chosen if higher_is_better else -chosen
    return cap, goodness


def draw_subset(n_pool, n_i, seed):
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    if n_i >= n_pool:
        return torch.arange(n_pool)
    idx = rng.choice(n_pool, size=n_i, replace=False)
    return torch.from_numpy(idx.astype(np.int64))


def slate_seed(model, ds_name, slate_id, draw_idx):
    key = f"{model}|{ds_name}|{slate_id}|{draw_idx}"
    return abs(hash(key)) % (2 ** 31)


def score_model_budgeted(model_name, v_pred_by_slate, v_true_by_slate, higher_is_better, n_i, ds_name):
    """Returns dict with headline (top/bottom/mean over draws 0-4), the
    50-draw distribution (mean/p5/p95), the per-slate goodness lists for
    draw 0 (for wins/losses/ties + paired sem), and n_i / n_pool used."""
    n_i_int = int(np.floor(n_i)) if n_i is not None else None
    if n_i_int is not None:
        n_i_int = max(1, min(POOL_N, n_i_int))
    draw_means = []
    draw_goodness_lists = []  # per draw: list over slates
    slate_ids = sorted(v_pred_by_slate.keys())
    for d in range(N_DRAWS_TOTAL):
        caps = []
        goods = []
        for sid in slate_ids:
            vp = v_pred_by_slate[sid]
            vt = v_true_by_slate[sid]
            n_pool = vt.shape[0]
            n_use = n_pool if n_i_int is None else n_i_int
            subset = draw_subset(n_pool, n_use, slate_seed(model_name, ds_name, sid, d))
            cap, good = budgeted_slate_capture(vp, subset, vt, higher_is_better)
            if cap == cap:
                caps.append(cap)
            goods.append(good)
        draw_means.append(float(np.mean(caps)) if caps else float("nan"))
        draw_goodness_lists.append(goods)
    headline = draw_means[:N_DRAWS_HEADLINE]
    dist = draw_means[:N_DRAWS_TOTAL]
    return dict(
        n_i_requested=n_i, n_i_used=n_i_int, n_pool=POOL_N, infeasible=(n_i is None),
        headline_top=float(np.max(headline)), headline_bottom=float(np.min(headline)),
        headline_mean=float(np.mean(headline)),
        dist_mean=float(np.mean(dist)), dist_p5=float(np.percentile(dist, 5)),
        dist_p95=float(np.percentile(dist, 95)),
        draw0_goodness=draw_goodness_lists[0],
        n_slates=len(slate_ids),
    )


def score_model_fixedK(model_name, v_pred_by_slate, v_true_by_slate, higher_is_better, k, ds_name, n_draws=20):
    """K=32 fixed reference point, independent of any T-budget -- same
    budgeted_slate_capture machinery with n_i=k."""
    return score_model_budgeted(f"{model_name}_K{k}", v_pred_by_slate, v_true_by_slate,
                                 higher_is_better, k, ds_name)


def wlt(good_a, good_b, tol=1e-9):
    w = l = t = 0
    diffs = []
    for a, b in zip(good_a, good_b):
        diffs.append(a - b)
        if abs(a - b) <= tol:
            t += 1
        elif a > b:
            w += 1
        else:
            l += 1
    return w, l, t, paired_sem(np.array(diffs))


def main():
    t0 = time.time()
    prov = git_provenance()
    print(f"[eval_budgeted_control] provenance: {prov}")
    print(f"[eval_budgeted_control] DEVICE={DEVICE}")

    with open(BUDGETS_JSON) as f:
        budgets = json.load(f)

    print("[1/4] loading model specs (bench.py's own pre/fwd wrappers)...")
    specs = build_model_specs()

    print("[2/4] loading datasets + computing per-model predictions over the full 128-pool...")
    # per dataset -> per (value_fn) -> per model -> per slate -> (v_pred, v_true, higher)
    per_ds_predictions = {}
    for ds_name, cfg in MULTISTEP_DATASETS.items():
        slates = load_cell(cfg["eval_cfg"], "train", manifest_path=cfg["manifest"], tag=f"budget_{ds_name}")
        step0 = slates.step_idx == 0
        idx_all = step0.nonzero(as_tuple=True)[0]
        slate_ids_all = slates.slate_idx[idx_all]
        occ1_all = slates.occ1[idx_all].float()
        H, W = occ1_all.shape[-2:]
        uniq = slate_ids_all.unique().tolist()
        print(f"  {ds_name}: {len(uniq)} slates")

        occ_pred_per_model = {n: {} for n in MODEL_NAMES if n != "random"}
        for sid in uniq:
            local_mask = (slate_ids_all == sid)
            rows_local = local_mask.nonzero(as_tuple=True)[0]
            rows_global = idx_all[rows_local]
            assert rows_global.numel() == POOL_N, f"{ds_name} slate {sid}: expected {POOL_N} rows, got {rows_global.numel()}"
            batch = build_batch_from_slate(slates, rows_global, DEVICE)
            for name in MODEL_NAMES:
                if name in ("random", "persistence"):
                    continue
                occ_pred_per_model[name][sid] = predict_occ_for_model(name, specs[name], batch)
            occ_pred_per_model["persistence"][sid] = predict_occ_for_model("persistence", None, batch)

        g_rand = torch.Generator().manual_seed(hash(ds_name) % (2 ** 31))

        per_ds_predictions[ds_name] = dict(
            occ1_all=occ1_all, idx_all=idx_all, slate_ids_all=slate_ids_all,
            uniq=uniq, H=H, W=W, occ_pred_per_model=occ_pred_per_model,
            rand_seed=hash(ds_name) % (2 ** 31),
        )
        print(f"  {ds_name}: predictions ready ({time.time()-t0:.1f}s elapsed)")

    print("[3/4] scoring under budgets + K=32 fixed reference...")
    results = {
        "note": ("n20_L10mm excluded. Goal shape=corner only, value_fns=[lyapunov,mass_in_region] "
                 "(scope reductions documented in the script docstring / EXPERIMENT.md). "
                 "N_i(T) taken from results/budgets.json's with_value sweep (RUN-0002 corrected) "
                 "-- with_value chosen because a real MPC step pays for value computation too."),
        "budgets_t_ms": BUDGETS_T_MS, "k_fixed": K_FIXED,
        "n_draws_headline": N_DRAWS_HEADLINE, "n_draws_total": N_DRAWS_TOTAL,
        "pool_n": POOL_N, "provenance": prov,
        "datasets": {},
    }

    for ds_name, dsd in per_ds_predictions.items():
        occ1_all, idx_all, slate_ids_all = dsd["occ1_all"], dsd["idx_all"], dsd["slate_ids_all"]
        H, W = dsd["H"], dsd["W"]
        occ_pred_per_model = dsd["occ_pred_per_model"]
        uniq = dsd["uniq"]
        results["datasets"][ds_name] = {"n_slates": len(uniq), "value_fns": {}}

        mask, dw = build_goal(GOAL_SHAPE, H, W, slate_seed=hash(ds_name) % (2 ** 31))
        g_rand = torch.Generator().manual_seed(dsd["rand_seed"])
        rand_score_all = torch.rand(occ1_all.shape[0], generator=g_rand) * 2 - 1

        for vfn in VALUE_FNS:
            v_true_full, _, higher = value_true_and_pred(vfn, occ1_all, {}, dw, mask)
            v_pred_by_slate_by_model = {n: {} for n in MODEL_NAMES}
            v_true_by_slate = {}
            for sid in uniq:
                rows_local = (slate_ids_all == sid).nonzero(as_tuple=True)[0]
                v_true_by_slate[sid] = v_true_full[rows_local]
                for name in MODEL_NAMES:
                    if name == "random":
                        rscore = rand_score_all[rows_local]
                        v_pred_by_slate_by_model[name][sid] = rscore if higher else -rscore
                        continue
                    pred = occ_pred_per_model[name][sid]
                    if isinstance(pred, dict) and pred.get("desc_readout"):
                        v_pred_by_slate_by_model[name][sid] = desc_pointmass_value(
                            vfn, pred["wr"], pred["wc"], pred["mass_hat"], dw, mask, H, W)
                    else:
                        _, out_pred, _ = value_true_and_pred(vfn, occ1_all[:1], {name: pred}, dw, mask)
                        v_pred_by_slate_by_model[name][sid] = out_pred[name]

            vfn_out = {"budgets": {}, "k_fixed_32": {}, "head_to_head": {}}
            for name in MODEL_NAMES:
                # K=32 fixed reference (mandatory)
                vfn_out["k_fixed_32"][name] = score_model_fixedK(
                    name, v_pred_by_slate_by_model[name], v_true_by_slate, higher, K_FIXED, ds_name)

            for T in BUDGETS_T_MS:
                key = f"{T}ms"
                per_model_T = {}
                for name in MODEL_NAMES:
                    if name in ("persistence", "random"):
                        n_i = POOL_N
                    else:
                        bkey = f"{T}ms"
                        n_i = budgets["per_model"][name]["with_value"].get(bkey)
                    per_model_T[name] = score_model_budgeted(
                        name, v_pred_by_slate_by_model[name], v_true_by_slate, higher, n_i, ds_name)
                vfn_out["budgets"][key] = per_model_T

                # head-to-head at draw 0, for interesting pairs
                pairs = [("model0001_global", "persistence"), ("model0001_global", "random"),
                         ("nfd", "model0001_global"), ("gnn", "model0001_global"),
                         ("schenck", "model0001_global"), ("gnn", "schenck")]
                h2h = {}
                for a, b in pairs:
                    w, l, tt, sem = wlt(per_model_T[a]["draw0_goodness"], per_model_T[b]["draw0_goodness"])
                    h2h[f"{a}_vs_{b}"] = {"wins": w, "losses": l, "ties": tt, "paired_sem_of_diff": sem,
                                          "n": len(per_model_T[a]["draw0_goodness"])}
                vfn_out["head_to_head"][key] = h2h

            results["datasets"][ds_name]["value_fns"][vfn] = vfn_out
        print(f"  {ds_name} scored ({time.time()-t0:.1f}s elapsed)")

    # -----------------------------------------------------------------
    # Pooled across datasets (n20_L20mm + n20_L40mm)
    # -----------------------------------------------------------------
    print("[4/4] pooling across datasets + writing artifacts...")
    results["pooled"] = {}
    for vfn in VALUE_FNS:
        pooled = {"budgets": {}, "k_fixed_32": {}}
        for T in BUDGETS_T_MS:
            key = f"{T}ms"
            pooled["budgets"][key] = {}
            for name in MODEL_NAMES:
                all_draw_means_headline = []
                all_draw_means_dist = []
                n_i_used = None
                infeasible = False
                for ds_name in MULTISTEP_DATASETS:
                    cell = results["datasets"][ds_name]["value_fns"][vfn]["budgets"][key][name]
                    n_i_used = cell["n_i_used"]
                    infeasible = infeasible or cell["infeasible"]
                # simple mean-of-means pooling (each dataset ~20 slates, comparable weight)
                tops, bots, means = [], [], []
                for ds_name in MULTISTEP_DATASETS:
                    cell = results["datasets"][ds_name]["value_fns"][vfn]["budgets"][key][name]
                    tops.append(cell["headline_top"]); bots.append(cell["headline_bottom"]); means.append(cell["headline_mean"])
                pooled["budgets"][key][name] = dict(
                    headline_top=float(np.mean(tops)), headline_bottom=float(np.mean(bots)),
                    headline_mean=float(np.mean(means)), n_i_used=n_i_used, infeasible=infeasible)
        for name in MODEL_NAMES:
            means = [results["datasets"][ds]["value_fns"][vfn]["k_fixed_32"][name]["headline_mean"]
                     for ds in MULTISTEP_DATASETS]
            pooled["k_fixed_32"][name] = dict(headline_mean=float(np.mean(means)))
        results["pooled"][vfn] = pooled

    out_dir = f"{EXP_DIR}/artifacts/RUN-0003-budgeted-control"
    os.makedirs(out_dir, exist_ok=True)
    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    results["elapsed_s"] = time.time() - t0
    out_path = f"{out_dir}/results_control.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval_budgeted_control] wrote {out_path} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
