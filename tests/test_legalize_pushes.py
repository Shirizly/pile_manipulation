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


def test_torch_legalizer_matches_cpu_on_random_scatter():
    """legalize_pushes_torch (batched) reproduces legalize_pushes push for push (shared-state broadcast)."""
    from Genesis.action_sampling import legalize_pushes_torch
    g = torch.Generator().manual_seed(0)
    box = (-0.064 + 0.005, 0.064 - 0.005)
    agree = total = 0
    for _ in range(4):
        cubes = (torch.rand(1, 20, 2, generator=g) - 0.5) * 0.08
        yaw = torch.rand(1, 20, generator=g) * math.pi
        s = (torch.rand(200, 2, generator=g) - 0.5) * 0.09
        ang = torch.rand(200, generator=g) * 2 * math.pi; L = 0.02 + 0.05 * torch.rand(200, generator=g)
        acts = torch.cat([s, s + L[:, None] * torch.stack([ang.cos(), ang.sin()], -1)], -1)
        a1, s1, o1 = legalize_pushes(acts, cubes.expand(200, -1, -1), yaw.expand(200, -1), CH, HL, HW, box=box)
        a2, s2, o2 = legalize_pushes_torch(acts, cubes, yaw, CH, HL, HW, box=box)
        same = (o1 == o2) & ((a1 - a2).abs().max(1).values < 1e-6) & ((s1 - s2).abs() < 1e-6)
        agree += int(same.sum()); total += len(same)
        for k in torch.nonzero(o2).flatten().tolist():           # every push it calls legal is legal (exact SAT)
            assert not _illegal(a2[k], cubes[0], yaw[0])
    assert agree / total >= 0.99, agree / total
