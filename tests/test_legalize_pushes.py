"""Genesis-free test for Genesis/action_sampling.py::legalize_pushes (ISS-013)."""
import math

import numpy as np
import torch

from Baselines.common.cube_overlap import overlaps_rect_pairs
from Genesis.action_sampling import legalize_pushes

HL, HW, CH = 0.02, 0.001, 0.0025


def _illegal(act, cxy, cyaw):
    s, e = act[:2].numpy(), act[2:].numpy()
    yaw = math.atan2(e[1] - s[1], e[0] - s[0]) + math.pi / 2
    n = len(cxy)
    return overlaps_rect_pairs(np.repeat(s[None], n, 0), np.full(n, yaw), np.array([HL, HW]),
                               cxy.numpy(), cyaw.numpy(), np.array([CH, CH])).any()


def test_illegal_push_is_slid_back_and_legal_push_untouched():
    cubes = torch.tensor([[[0.0, 0.0], [0.03, 0.03]]])           # one cube under the start
    yaw = torch.zeros(1, 2)
    bad = torch.tensor([[0.0, 0.0, 0.02, 0.0]])                   # start ON cube 0, pushing +x
    out, shift, ok = legalize_pushes(bad, cubes, yaw, CH, HL, HW)
    assert ok[0] and shift[0] > 0
    assert not _illegal(out[0], cubes[0], yaw[0])
    assert torch.allclose(out[0, 2:] - out[0, :2], bad[0, 2:] - bad[0, :2])   # heading and length kept
    assert out[0, 0] < 0                                          # moved BACK along -x
    good = torch.tensor([[-0.02, 0.0, 0.0, 0.0]])                 # blade 20 mm behind the cube centre
    out2, shift2, ok2 = legalize_pushes(good, cubes, yaw, CH, HL, HW)
    assert ok2[0] and shift2[0] == 0 and torch.equal(out2, good)
