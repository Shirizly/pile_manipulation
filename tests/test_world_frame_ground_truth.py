"""The `world-frame-alignment` invariant.

EXP-0001 made the dataset's occupancy channel agree with its plate channel and
with `particles_to_occupancy`, all on `dim0 = world_x`. But "they agree" is not
"they are right": if all three had been standardised onto the *wrong* axis the
system would still be self-consistent, one-step prediction would be unaffected,
and only the MPC would suffer -- it would render and reason about pushes in a
frame mirrored relative to the world it commands actions in.

So this checks the convention against physics rather than against other code.
The chain is:

  1. world ground truth, no grid at all: material moves in the direction the
     blade travels, in world metres, straight from the recorded states.
  2. the grid agrees with (1): the centroid of where occupancy ARRIVES minus
     where it LEAVES, read back out of the grid under the assumed convention,
     points the same way as the world-frame push.

If dim0 and dim1 were swapped, (2) fails while (1) still passes -- which is
exactly the failure mode no amount of internal consistency can detect.
"""
import glob

import pytest
import torch

from transforms.functional import particles_to_occupancy

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
N = 64
GLOB = "Genesis/data/cube_spectrum/n20/*_data.pt"


@pytest.fixture(scope="module")
def batch():
    fs = sorted(glob.glob(GLOB))
    if not fs:
        pytest.skip("cube_spectrum/n20 not present")
    d = torch.load(fs[0], map_location="cpu", weights_only=False)
    s0, s1 = d["states"][..., :3], d["states_"][..., :3]
    push = (d["p_stops"][:, :2] - d["p_starts"][:, :2])
    keep = push.norm(dim=-1) > 1e-3
    return s0[keep], s1[keep], push[keep]


def test_material_moves_the_way_the_blade_travels_in_world_metres(batch):
    """Ground truth, grid-free. If this fails nothing else here is meaningful."""
    s0, s1, push = batch
    disp = (s1 - s0)[..., :2].mean(dim=1)                   # (B,2) world metres
    u = push / push.norm(dim=-1, keepdim=True)
    along = (disp * u).sum(-1)
    assert float(along.mean()) > 0, "material does not move along the push"
    assert float((along > 0).float().mean()) > 0.8, (
        f"only {float((along > 0).float().mean()):.0%} of pushes move material "
        f"forward in world coordinates")


def test_grid_delta_points_the_same_way_as_the_world_push(batch):
    """The convention check. Read the direction of transport back OUT of the
    grid under `dim0 = world_x`, and compare with the world-frame push."""
    s0, s1, push = batch
    o0 = particles_to_occupancy(s0, BOUNDS, (N, N), sigma=0.0)
    o1 = particles_to_occupancy(s1, BOUNDS, (N, N), sigma=0.0)
    delta = o1 - o0

    idx = torch.stack(torch.meshgrid(torch.arange(N, dtype=torch.float32),
                                     torch.arange(N, dtype=torch.float32),
                                     indexing="ij"), -1)     # (N,N,2) = (dim0, dim1)
    pos, neg = delta.clamp_min(0), (-delta).clamp_min(0)
    c_to = (pos.unsqueeze(-1) * idx).sum((1, 2)) / pos.sum((1, 2)).clamp_min(1e-6).unsqueeze(-1)
    c_from = (neg.unsqueeze(-1) * idx).sum((1, 2)) / neg.sum((1, 2)).clamp_min(1e-6).unsqueeze(-1)
    grid_dir = c_to - c_from                                 # (B,2) in (dim0, dim1)

    # Under dim0 = world_x, dim1 = world_y, this IS a world-frame vector.
    u = push / push.norm(dim=-1, keepdim=True)
    g = grid_dir / grid_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cos_assumed = (g * u).sum(-1)
    # The competing hypothesis: dim0 = world_y (the pre-fix convention).
    cos_swapped = (g.flip(-1) * u).sum(-1)

    assert float(cos_assumed.mean()) > 0.5, (
        f"grid transport direction does not match the world push "
        f"(mean cos {float(cos_assumed.mean()):.3f})")
    assert float(cos_assumed.mean()) > float(cos_swapped.mean()) + 0.3, (
        f"dim0=world_x (cos {float(cos_assumed.mean()):.3f}) does not clearly "
        f"beat dim0=world_y (cos {float(cos_swapped.mean()):.3f}) — the "
        f"convention is not pinned by the data")
