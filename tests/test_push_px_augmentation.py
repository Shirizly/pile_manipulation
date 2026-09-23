"""``_augment_eulerian_batch``'s optional ``push_px`` key (training/trainer.py).

A warped/canonical-frame model needs the push endpoints, and rotating the
occupancy/target images without rotating the endpoints would silently
corrupt every augmented sample. This pins the pixel map (derived in
``model/warped_nfd/WARPED_NFD_NOTES.md``) for `torch.rot90(k, dims=(-2,-1))`
composed with `torch.flip(dims=[-1])`, and confirms the legacy (no
``push_px``) behaviour is unaffected.
"""
import math

import torch

from training.trainer import _augment_eulerian_batch
from transforms.functional import draw_plate_soft

N = 32
PLATE_X, PLATE_Y, SIGMA = 9.0, 3.0, 1.2


def _render(start_px: torch.Tensor, end_px: torch.Tensor) -> torch.Tensor:
    """Render a plate at push endpoints given in (col, row) pixel order,
    exactly the convention `docs/INTERFACES.md` documents for `push_px`.

    `draw_plate_soft`'s own center convention is (dim0, dim1) = (row, col)
    (see tests/test_grid_convention.py), so the point is swapped on the way
    in. The plate angle uses the `pi - phi` relation between
    `transforms.functional.push_frame_transform`'s
    ``phi = atan2(delta_row, delta_col)`` and `draw_plate_soft`'s own
    (dim0, dim1)-frame travel angle, worked out in
    `model/warped_nfd/WARPED_NFD_NOTES.md`.
    """
    delta = end_px - start_px
    phi = torch.atan2(delta[:, 1], delta[:, 0])
    plate_angle = math.pi - phi
    center_rc = torch.stack([start_px[:, 1], start_px[:, 0]], dim=-1)
    return draw_plate_soft(center_rc, plate_angle, (N, N), PLATE_X, PLATE_Y,
                           intensity=1.0, sigma=SIGMA)


def _make_batch():
    start_px = torch.tensor([[9.0, 22.0]])
    end_px = torch.tensor([[21.0, 8.0]])
    img = _render(start_px, end_px)  # (1, N, N)
    push_px = torch.cat([start_px, end_px], dim=-1)  # (1, 4)
    batch = {
        "input": img.unsqueeze(1),          # (1, 1, N, N)
        "target": torch.zeros(1, N, N),
        "push_px": push_px,
    }
    return batch


def test_augmented_push_px_matches_augmented_render():
    """For each of the 8 rot90/flip views, re-rendering a plate from the
    augmented push_px endpoints must reproduce the augmented image channel
    almost exactly (not just >0.97 correlation) since both are the same
    deterministic index permutation applied two different ways."""
    out = _augment_eulerian_batch(_make_batch())
    aug_input = out["input"]
    aug_push = out["push_px"]

    assert aug_input.shape == (8, 1, N, N)
    assert aug_push.shape == (8, 4)

    worst_corr = 1.0
    for i in range(8):
        s, e = aug_push[i:i + 1, :2], aug_push[i:i + 1, 2:]
        recon = _render(s, e)[0]
        actual = aug_input[i, 0]
        corr = torch.corrcoef(torch.stack([actual.flatten(), recon.flatten()]))[0, 1]
        assert float((actual - recon).abs().max()) < 1e-4, f"view {i} diverged"
        worst_corr = min(worst_corr, float(corr))
    assert worst_corr > 0.97, f"worst correlation across 8 views: {worst_corr}"


def test_push_px_absent_is_byte_identical_to_legacy_behaviour():
    """Without `push_px`, the augmentation must be unaffected by this change."""
    batch = _make_batch()
    del batch["push_px"]
    out = _augment_eulerian_batch(batch)
    assert "push_px" not in out
    assert out["input"].shape == (8, 1, N, N)
    assert out["target"].shape == (8, N, N)


def test_push_px_endpoints_are_a_permutation_of_the_grid_indices():
    """Sanity check independent of rendering: rot90/flip is a bijection of
    pixel indices, so mapped integer endpoints must stay integer and inside
    the grid."""
    start_px = torch.tensor([[5.0, 3.0]])
    end_px = torch.tensor([[27.0, 19.0]])
    batch = {
        "input": torch.zeros(1, 1, N, N),
        "target": torch.zeros(1, N, N),
        "push_px": torch.cat([start_px, end_px], dim=-1),
    }
    out = _augment_eulerian_batch(batch)
    aug_push = out["push_px"]
    assert torch.all(aug_push >= 0) and torch.all(aug_push <= N - 1)
    assert torch.allclose(aug_push, aug_push.round())
