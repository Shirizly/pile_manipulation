"""Scale up `ranking_robustness_check.py` (3 pools, L20mm only) to >=40
step-0 same-state pools spread across the 3 RUN-0009 corpora (L20mm, L40mm,
randlen_test), at canon_res=64 (matches nfd_warped_randlen's training-time
default). For every pool: push the TRUE occ1 of every candidate through
push_frame_roundtrip(identity) and compare the induced ranking (one value
function, lyapunov-to-a-per-slate-random-quadrant-goal) against the
un-round-tripped truth's ranking. Reports mean/median Spearman, the
distribution, and the top-1 flip RATE with its sample size -- this is what
decides whether slateN is trustworthy for warped arms at all.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from Baselines.common.data import load_cell
from Baselines.common.randlen_data import load_randlen_cell
from Baselines.common.goals import dist_field_from_mask, random_quadrant_mask
from control_utility_test import lyapunov
from fit_linear_foresight import actions_to_pixels
from transforms.functional import push_frame_roundtrip

CANON_RES = 64
N_POOLS_PER_CORPUS = 14  # 3 corpora x 14 = 42 >= 40
ident = lambda c: c

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
    actions_s0 = cell.actions[step0]
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
        q_dist = torch.from_numpy(dist_field_from_mask(q_mask_np)).float()

        v_true_raw = lyapunov(true_pool, q_dist)
        rt_true = push_frame_roundtrip(ident, true_pool, s_px, e_px, CANON_RES, 1.0).clamp(0, 1)
        v_true_rt = lyapunov(rt_true, q_dist)

        rho, pval = spearmanr(v_true_raw.numpy(), v_true_rt.numpy())
        top1_raw = int(torch.argmin(v_true_raw))
        top1_rt = int(torch.argmin(v_true_rt))
        top1_agree = (top1_raw == top1_rt)

        all_results.append(dict(corpus=name, slate=int(sid), n_pool=n_pool,
                                 spearman=float(rho), top1_agree=bool(top1_agree)))
        print(f"  [{name}] slate {sid}: n_pool={n_pool}  spearman={rho:.4f}  "
              f"top1_agree={top1_agree}")

spearmans = np.array([r["spearman"] for r in all_results])
flips = np.array([not r["top1_agree"] for r in all_results])
n = len(all_results)
print(f"\n=== Summary over N={n} pools across {len(CELLS)} corpora ===")
print(f"mean spearman   = {spearmans.mean():.4f}")
print(f"median spearman = {np.median(spearmans):.4f}")
print(f"min spearman    = {spearmans.min():.4f}")
print(f"max spearman    = {spearmans.max():.4f}")
print(f"std spearman    = {spearmans.std():.4f}")
print(f"spearman deciles: {np.percentile(spearmans, [0,10,25,50,75,90,100]).round(4)}")
print(f"top-1 flip count = {flips.sum()} / {n}  (rate = {flips.mean():.4f})")
for name in CELLS:
    sub = [r for r in all_results if r["corpus"] == name]
    sub_flips = sum(1 for r in sub if not r["top1_agree"])
    print(f"  [{name}] n={len(sub)}  mean_spearman={np.mean([r['spearman'] for r in sub]):.4f}  "
          f"flips={sub_flips}/{len(sub)}")
