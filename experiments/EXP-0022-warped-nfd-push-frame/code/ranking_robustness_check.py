"""Check EXP-0022's warp_accuracy_ceiling.md claim: does the warp round-trip's
resampling loss cancel in a ranking metric (slateN) across a same-state pool?

Method: for several step-0 candidate pools, push the TRUE occ1 of every
candidate through push_frame_roundtrip(identity) at canon_res=64 (matching
the warped pilot checkpoints and the ceiling measurement), score one value
function (lyapunov, to a per-slate random_quadrant goal) on both the raw
truth and the round-tripped truth, and compare the induced RANKINGS via
Spearman correlation and top-1 agreement.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from Baselines.common.data import load_cell
from Baselines.common.goals import dist_field_from_mask, random_quadrant_mask
from control_utility_test import lyapunov
from fit_linear_foresight import actions_to_pixels
from transforms.functional import push_frame_roundtrip

CFG = "configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml"
MANIFEST = "Genesis/data/slates_multistep/n20_L20mm/manifest.json"
CANON_RES = 64

cell = load_cell(CFG, "train", manifest_path=MANIFEST, tag="L20mm")
H, W = cell.occ0.shape[-2:]
step0 = (cell.step_idx == 0)
slate_ids = cell.slate_idx[step0]
occ1_s0 = cell.occ1[step0].to(torch.float32)
actions_s0 = cell.actions[step0]
unique_slates = slate_ids.unique().tolist()

s_px_all, e_px_all = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
s_px_s0, e_px_s0 = s_px_all[step0], e_px_all[step0]

ident = lambda c: c

rng = np.random.default_rng(0)
chosen_slates = rng.choice(unique_slates, size=min(3, len(unique_slates)), replace=False)

results = []
for sid in chosen_slates:
    rows = (slate_ids == int(sid)).nonzero(as_tuple=True)[0]
    true_pool = occ1_s0[rows]
    s_px = s_px_s0[rows]
    e_px = e_px_s0[rows]
    n_pool = true_pool.shape[0]

    q_mask_np, _ = random_quadrant_mask(H, W, seed=int(sid))
    q_dist = torch.from_numpy(dist_field_from_mask(q_mask_np)).float()

    v_true_raw = lyapunov(true_pool, q_dist)  # (n_pool,)

    rt_true = push_frame_roundtrip(ident, true_pool, s_px, e_px, CANON_RES, 1.0).clamp(0, 1)
    v_true_rt = lyapunov(rt_true, q_dist)

    rho, pval = spearmanr(v_true_raw.numpy(), v_true_rt.numpy())
    # lower_is_better goal (distance-to-goal); "best" = min
    top1_raw = int(torch.argmin(v_true_raw))
    top1_rt = int(torch.argmin(v_true_rt))
    top1_agree = (top1_raw == top1_rt)

    print(f"slate {sid}: n_pool={n_pool}  spearman={rho:.4f} (p={pval:.2e})  "
          f"top1_agree={top1_agree}  (raw_idx={top1_raw}, rt_idx={top1_rt})")
    results.append(dict(slate=int(sid), n_pool=n_pool, spearman=float(rho), top1_agree=bool(top1_agree)))

print()
print("mean spearman:", np.mean([r["spearman"] for r in results]))
print("top1 agreement rate:", np.mean([r["top1_agree"] for r in results]))
