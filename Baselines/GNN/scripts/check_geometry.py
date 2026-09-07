"""Cheap geometry sanity check, BEFORE training (SPEC.md sec 8/9 item 2).

Loads a few raw transitions from n20_L20mm and n20_L40mm, computes s_delta
and the resulting adjacency edge count per node at a few candidate
adj_thresh values, and prints:
  - how many of the 20 cubes get a non-trivial (>1e-4 m) s_delta magnitude
    (should be a handful in the swept lane, not 0 and not all 20);
  - the min/mean/max node degree of the adjacency graph built on
    s_cur + s_delta (should not be ~0 for every node, and not saturate at
    the hard cap of min(10, N-1)=10 for every node/adj_thresh pair, or the
    threshold clause is a no-op and top-10-NN is doing all the work).

Run: PYTHONPATH=. python Baselines/GNN/scripts/check_geometry.py
"""
import sys
sys.path.insert(0, ".")

import torch

from Baselines.GNN.geometry import PUSHER_W, SOFTNESS, compute_s_delta
from model.gnn_dyn import PropNetDiffDenModel

CANDIDATES = [0.008, 0.010, 0.012, 0.015, 0.020]


def degree_stats(s_cur, s_delta, adj_thresh):
    cfg = {"train": {"particle": {"adj_thresh": adj_thresh, "nf_effect": 64, "add_delta": True}}}
    model = PropNetDiffDenModel(cfg)
    B, N, _ = s_cur.shape
    a_cur = torch.zeros(B, N)
    dens = torch.full((B,), 1000.0)
    # Reimplement just the adjacency-construction half of predict_one_step,
    # to get Rr/Rs without running the full network (cheap, and avoids
    # depending on internal method names beyond what's already public).
    s_receiv = (s_cur + s_delta)[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = (s_cur + s_delta)[:, None, :, :].repeat(1, N, 1, 1)
    threshold = adj_thresh * adj_thresh
    dis = torch.sum((s_sender - s_receiv) ** 2, -1)
    max_rel = min(10, N)
    topk_idx = torch.topk(dis, k=max_rel, dim=2, largest=False).indices
    topk_bin_mat = torch.zeros_like(dis)
    topk_bin_mat.scatter_(2, topk_idx, 1)
    adj_matrix = ((dis - threshold) < 0).float() * topk_bin_mat
    degree = adj_matrix.sum(-1)   # (B, N) in-degree per receiver
    return degree


def main():
    for cell in ["n20_L20mm", "n20_L40mm"]:
        print(f"\n=== {cell} ===")
        d = torch.load(f"Genesis/data/slates_multistep/{cell}/_0_data.pt", weights_only=False)
        states = d["states"][:8, :, :3]      # (8,20,3), first 8 candidates
        p_start = d["p_starts"][:8]
        p_stop = d["p_stops"][:8]

        s_delta = compute_s_delta(states, p_start, p_stop)
        mag = s_delta.norm(dim=-1)   # (8,20)
        n_active = (mag > 1e-4).sum(-1).float()
        print(f"  push length (mean): {(p_stop[:, :2]-p_start[:, :2]).norm(dim=-1).mean():.4f} m")
        print(f"  s_delta active-node count per candidate (of 20): "
              f"min={n_active.min():.0f} mean={n_active.mean():.1f} max={n_active.max():.0f}")
        print(f"  s_delta magnitude (active nodes only): "
              f"mean={mag[mag>1e-4].mean() if (mag>1e-4).any() else float('nan'):.4f} m")

        for adj_thresh in CANDIDATES:
            degree = degree_stats(states, s_delta, adj_thresh)
            print(f"  adj_thresh={adj_thresh:.3f}: degree min={degree.min():.0f} "
                  f"mean={degree.mean():.2f} max={degree.max():.0f} "
                  f"(cap=10; frac_at_cap={(degree==10).float().mean():.2f})")

        # cross-check: print the raw x_loc (along-push)/y_loc (lateral)
        # coordinates and both mask components directly, to see exactly why
        # each cube is flagged active or not (rather than a lumped distance
        # metric that conflates the hard longitudinal gate and the soft
        # lateral one).
        p0 = p_start[0, :2]
        p1 = p_stop[0, :2]
        seg = p1 - p0
        seg_len = seg.norm()
        push_dir0 = seg / seg_len
        push_ortho0 = torch.stack([-push_dir0[1], push_dir0[0]])
        pos_xy = states[0, :, :2]
        rel = pos_xy - p0[None]
        x_loc = rel @ push_dir0
        y_loc = rel @ push_ortho0
        l_mask = ((x_loc > 0) & (x_loc < seg_len)).float()
        w_mask = torch.exp(-(y_loc.abs() - PUSHER_W).clamp(min=0) / SOFTNESS)
        print(f"  [candidate 0] push_len={seg_len:.4f}  per-cube (x_loc, y_loc, l_mask, w_mask, |s_delta|):")
        for i in range(20):
            print(f"    cube {i:2d}: x={float(x_loc[i]):+.4f} y={float(y_loc[i]):+.4f} "
                  f"l={float(l_mask[i]):.0f} w={float(w_mask[i]):.3f} "
                  f"|sd|={float(mag[0,i]):.4f}")


if __name__ == "__main__":
    main()
