"""Baselines/SchenckCNN/check_axes.py -- one-off axis-convention sanity
check (step 2 of the task brief): confirm the action channels this module
builds (action_encoding.build_action_channels) agree with occ0/occ1 about
which grid axis is world x, BEFORE spending any GPU time on training.

Method: pick a few pooled-train transitions with a push that is strongly
dominated by one world axis (|dx|>>|dy| or vice versa), and check that the
pixel-space displacement of the rendered action channels' mass (argmax of
r_stop - r_start) is dominated by the SAME grid axis as the world-dominant
one, per `actions_to_pixels`' own documented convention (dim0 <- world_x,
dim1 <- world_y). Also dumps a few printable channel summaries for manual
inspection.

Run: PYTHONPATH=. python Baselines/SchenckCNN/check_axes.py
"""
from __future__ import annotations

import torch

from Baselines.common.data import load_cell
from Baselines.SchenckCNN.action_encoding import build_action_channels

TRAIN_CFG = "configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml"


def main():
    cell = load_cell(TRAIN_CFG, "train")
    H, W = cell.H, cell.W
    dxy = (cell.p_stop[:, :2] - cell.p_start[:, :2])
    dx, dy = dxy[:, 0], dxy[:, 1]

    # pick the most x-dominant and most y-dominant pushes in the pool
    x_dom_idx = torch.argmax(dx.abs() - dy.abs()).item()
    y_dom_idx = torch.argmax(dy.abs() - dx.abs()).item()

    for label, idx in (("x-dominant push", x_dom_idx), ("y-dominant push", y_dom_idx)):
        i = idx
        p_start = cell.p_start[i:i + 1]
        p_stop = cell.p_stop[i:i + 1]
        angle = cell.angle[i:i + 1]
        world_dx, world_dy = float(dx[i]), float(dy[i])

        action = build_action_channels(p_start, p_stop, angle, cell.raw, H, W)
        r_start, r_stop = action[0, 0], action[0, 1]

        row0, col0 = divmod(int(torch.argmax(r_start)), W)
        row1, col1 = divmod(int(torch.argmax(r_stop)), W)
        drow, dcol = row1 - row0, col1 - col0

        # ground-truth occupancy delta (occ1-occ0), swept region only, as an
        # independent cross-check of where mass actually moved
        occ_delta = (cell.occ1[i] - cell.occ0[i])
        occ_row_marg = occ_delta.abs().sum(dim=1)   # collapse cols -> row profile
        occ_col_marg = occ_delta.abs().sum(dim=0)   # collapse rows -> col profile

        print(f"--- {label} (sample idx={i}) ---")
        print(f"  world displacement: dx={world_dx:+.4f} m, dy={world_dy:+.4f} m "
              f"(dominant axis: {'x' if abs(world_dx) > abs(world_dy) else 'y'})")
        print(f"  action-channel argmax displacement: drow={drow:+d}, dcol={dcol:+d} px "
              f"(dominant grid axis: {'row(dim0)' if abs(drow) > abs(dcol) else 'col(dim1)'})")
        print(f"  occ0/occ1 delta marginal spread: row-profile std={float(occ_row_marg.std()):.4f}, "
              f"col-profile std={float(occ_col_marg.std()):.4f}")
        print(f"  expected convention (actions_to_pixels): dim0<-world_x, dim1<-world_y")
        expect_axis = "row(dim0)" if abs(world_dx) > abs(world_dy) else "col(dim1)"
        got_axis = "row(dim0)" if abs(drow) > abs(dcol) else "col(dim1)"
        print(f"  PASS: {expect_axis == got_axis}  (expected {expect_axis}, got {got_axis})")
        print()


if __name__ == "__main__":
    main()
