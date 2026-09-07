"""Baselines/SchenckCNN/model.py -- the single-tower ablation of Schenck et
al.'s "Learning Robotic Manipulation of Granular Media" (CoRL 2017) network.

Per SPEC.md sec 5 (and ORCHESTRATION_LOG.md's wave-A scope note): reproduce
ONLY the single-net ablation -- one fully-convolutional tower, no
pooling/upsampling anywhere (every layer stays at the input's full spatial
resolution, matching the paper's "comprised of only convolution and ReLU
layers" description), residual output, L2 loss. The paper's headline
two-tower scoop&dump-net (with its inter-tower mass-conservation channel) is
OUT OF SCOPE: it exists to separate a lift-and-carry phase (scoop, removes
mass from one place) from a pour phase (dump, adds mass elsewhere); our task
is a single continuous plate push -- nothing is picked up, carried, or
poured, so there is no scoop/dump split to build an architecture for.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SchenckCNN(nn.Module):
    """~16x [Conv 32@3x3, ReLU] at constant 64x64 resolution (no pooling),
    ending in Conv 1@1x1, then a residual add of the input's channel-0
    field (occ0) -- matching the paper's Fig.3 "+" symbol after the final
    1x1 conv (the one explicit residual drawn in either reference paper's
    diagram; see SPEC.md sec 1).

    forward(x) with x: (B, in_channels, H, W), channel 0 MUST be occ0 (the
    field the residual is added onto). Returns (B, H, W) -- the predicted
    next occupancy field, un-clamped (plain linear regression head, no
    sigmoid: the paper's loss is plain L2 on a continuous height-map, not a
    BCE-style logit convention).
    """

    def __init__(self, in_channels: int = 4, width: int = 32, n_layers: int = 16):
        super().__init__()
        assert n_layers >= 1
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(n_layers):
            layers.append(nn.Conv2d(c, width, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c = width
        self.trunk = nn.Sequential(*layers)
        self.final = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        occ0 = x[:, 0:1]
        delta = self.final(self.trunk(x))
        return (occ0 + delta).squeeze(1)
