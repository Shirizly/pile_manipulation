"""EXP-0022 A2 -- warped-GOAL control scoring, tested as a hypothesis.

Mirrors `ranking_robustness_check_n40.py` (>=40 step-0 same-state pools
across the 3 RUN-0009 corpora, canon_res=64) but replaces "warp the STATE,
keep the goal fixed" with "warp the GOAL into each candidate's OWN push
frame, keep the state at its raw truth". This is the risk the task brief
names explicitly: `slateN` ranks candidates FROM THE SAME STATE, but every
candidate has a DIFFERENT push frame, so warping the goal per candidate
means each candidate's value is computed in its own rotated, resampled
frame -- the resampling bias no longer cancels ACROSS the comparison, it
VARIES across it. This script measures whether that hurts the induced
ranking, using the exact same truth ranking, corpora, pools-per-corpus and
value function (`lyapunov` against a per-slate random-quadrant goal) as the
existing world-frame check, so the two flip rates are directly comparable.

For each pool:
  - v_true_raw   : lyapunov(true_pool, q_dist)                    -- world frame, UNCHANGED goal (the baseline ranking every other check in this
                    experiment is compared against).
  - v_warped_goal: for candidate i, warp q_dist's SOURCE MASK into
                    candidate i's own canonical push frame (canon_res=64),
                    recompute the distance field THERE (nonlinear function
                    of a mask does not commute with resampling the mask's
                    own precomputed field, so this thresholds the warped
                    mask fresh each time rather than warping the float
                    field directly), and score the candidate's OWN raw
                    true occupancy warped into that SAME per-candidate
                    frame against that per-candidate goal field. Batched
                    over the whole pool in one `to_push_frame` call (each
                    row gets its own start_px/end_px), no Python loop.

Reports mean/median Spearman and the top-1 flip rate against v_true_raw,
compared explicitly to the world-frame check's 2/42 = 4.76%.
"""
import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.ndimage import distance_transform_edt

from Baselines.common.data import load_cell
from Baselines.common.randlen_data import load_randlen_cell
from Baselines.common.goals import random_quadrant_mask
from fit_linear_foresight import actions_to_pixels
from transforms.functional import to_push_frame

CANON_RES = 64
N_POOLS_PER_CORPUS = 14  # matches ranking_robustness_check_n40.py exactly

CELLS = {
    "L20mm": dict(
        kind="slate",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json",
    ),
    "L40mm": dict(
        kind="slate",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L40mm/manifest.json",
    ),
    "randlen_test": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_all.yaml",
    ),
}


def _dist_field_batched(mask_batch: np.ndarray) -> np.ndarray:
    """Per-row `distance_transform_edt(~mask)` / its own max, batched -- same
    formula as `Baselines.common.goals.dist_field_from_mask`, applied
    independently to each (H, W) slice since `distance_transform_edt` has no
    native batch axis."""
    out = np.empty_like(mask_batch, dtype=np.float32)
    for i in range(mask_batch.shape[0]):
        d = distance_transform_edt(~mask_batch[i]).astype(np.float32)
        out[i] = d / max(float(d.max()), 1e-6)
    return out


def _lyapunov_percandidate(occ: torch.Tensor, d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Same formula as `control_utility_test.lyapunov`, but `d` is (N,H,W) --
    a DIFFERENT distance field per row -- instead of one shared (H,W) field."""
    n = occ.shape[0]
    flat = occ.reshape(n, -1)
    dflat = d.reshape(n, -1)
    return (flat * dflat).sum(1) / flat.sum(1).clamp_min(eps)


rng = np.random.default_rng(0)
all_results = []

for name, spec in CELLS.items():
    if spec["kind"] == "slate":
        cell = load_cell(spec["eval_cfg"], "train", manifest_path=spec["manifest"], tag=name)
    else:
        cell = load_randlen_cell(spec["cfg"], "train", tag=name)
    H, W = cell.occ0.shape[-2:]
    step0 = (cell.step_idx == 0)
    slate_ids = cell.slate_idx[step0]
    occ1_s0 = cell.occ1[step0].to(torch.float32)
    unique_slates = slate_ids.unique().tolist()

    s_px_all, e_px_all = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    s_px_s0, e_px_s0 = s_px_all[step0], e_px_all[step0]

    n_pick = min(N_POOLS_PER_CORPUS, len(unique_slates))
    chosen_slates = rng.choice(unique_slates, size=n_pick, replace=False)
    print(f"=== {name}: {len(unique_slates)} step-0 pools available, sampling {n_pick} ===")

    for sid in chosen_slates:
        rows = (slate_ids == int(sid)).nonzero(as_tuple=True)[0]
        true_pool = occ1_s0[rows]
        s_px = s_px_s0[rows]
        e_px = e_px_s0[rows]
        n_pool = true_pool.shape[0]
        if n_pool < 2:
            continue

        q_mask_np, _ = random_quadrant_mask(H, W, seed=int(sid))
        q_dist = torch.from_numpy(
            distance_transform_edt(~q_mask_np).astype(np.float32)
        )
        q_dist = q_dist / q_dist.max().clamp_min(1e-6)

        # ---- world-frame truth ranking (fixed goal, UNCHANGED from the
        # existing world-frame check -- this is the comparison target for
        # BOTH the identity-round-trip check and this warped-goal check). ----
        v_true_raw = _lyapunov_percandidate(true_pool, q_dist.unsqueeze(0).expand(n_pool, -1, -1))

        # ---- warped-goal ranking: warp the SAME source mask into EACH
        # candidate's own canonical push frame (batched: one to_push_frame
        # call, n_pool rows, each with its own start_px/end_px), and warp
        # the candidate's own true occupancy into that SAME frame. ----
        mask_batch = torch.from_numpy(q_mask_np.astype(np.float32)).unsqueeze(0).expand(n_pool, -1, -1)
        canon_mask = to_push_frame(mask_batch, s_px, e_px, (CANON_RES, CANON_RES), 1.0)
        canon_mask_bin = (canon_mask.numpy() > 0.5)
        canon_dist = torch.from_numpy(_dist_field_batched(canon_mask_bin))
        canon_true = to_push_frame(true_pool, s_px, e_px, (CANON_RES, CANON_RES), 1.0)
        v_warped_goal = _lyapunov_percandidate(canon_true, canon_dist)

        rho, pval = spearmanr(v_true_raw.numpy(), v_warped_goal.numpy())
        top1_raw = int(torch.argmin(v_true_raw))
        top1_wg = int(torch.argmin(v_warped_goal))
        top1_agree = (top1_raw == top1_wg)

        all_results.append(dict(corpus=name, slate=int(sid), n_pool=n_pool,
                                 spearman=float(rho), top1_agree=bool(top1_agree)))
        print(f"  [{name}] slate {sid}: n_pool={n_pool}  spearman={rho:.4f}  "
              f"top1_agree={top1_agree}")

spearmans = np.array([r["spearman"] for r in all_results])
flips = np.array([not r["top1_agree"] for r in all_results])
n = len(all_results)
print(f"\n=== Summary over N={n} pools across {len(CELLS)} corpora (WARPED-GOAL) ===")
print(f"mean spearman   = {spearmans.mean():.4f}")
print(f"median spearman = {np.median(spearmans):.4f}")
print(f"min spearman    = {spearmans.min():.4f}")
print(f"max spearman    = {spearmans.max():.4f}")
print(f"std spearman    = {spearmans.std():.4f}")
print(f"spearman deciles: {np.percentile(spearmans, [0,10,25,50,75,90,100]).round(4)}")
print(f"top-1 flip count = {flips.sum()} / {n}  (rate = {flips.mean():.4f})")
print(f"world-frame baseline (identity round-trip, RUN-0009): 2/42 = 0.0476")
for name in CELLS:
    sub = [r for r in all_results if r["corpus"] == name]
    sub_flips = sum(1 for r in sub if not r["top1_agree"])
    print(f"  [{name}] n={len(sub)}  mean_spearman={np.mean([r['spearman'] for r in sub]):.4f}  "
          f"flips={sub_flips}/{len(sub)}")
