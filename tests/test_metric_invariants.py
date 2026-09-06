"""The `swept-region-metric` and `episode-split` invariants.

Both were `unchecked` from 2026-09-03, and between them they pinned every
record in docs/experiments/ to at least one `untested-dependency` downgrade --
not because anyone doubted them, but because nobody had written this file. That
is the grade scale saturating on missing tests rather than on weak evidence.

These are empirical invariants: they are properties of the metric applied to
real data, so they are checked against a real dataset and skipped when it is
absent.
"""
import glob

import pytest
import torch

from fit_linear_foresight import actions_to_pixels, swept_region_mask
from occupancy_foresight import BOUNDS, load_transition_fields

GLOB = "Genesis/data/cube_spectrum/n20/*_data.pt"
H = W = 64


@pytest.fixture(scope="module")
def data():
    if not glob.glob(GLOB):
        pytest.skip("cube_spectrum/n20 not present")
    o0, o1, act, ep, _, _ = load_transition_fields(
        GLOB, 64, 0.0, "mean", 19.9, "cpu", view="mask",
        min_grains=1.0, cube_size=0.005)
    s, e = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                             (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    plate = 0.04 / 0.128 * W
    region = swept_region_mask(s, e, (H, W), 0.5 * plate + 2.0, 0.5 * plate)
    return o0, o1, region


def test_swept_region_contains_almost_all_of_the_change(data):
    """The metric's whole justification: a whole-image error is ~95% pixels
    nothing could have changed, so models are scored inside this mask. That is
    only honest if the mask actually contains what changed."""
    o0, o1, region = data
    delta = (o1 - o0).abs()
    inside = (region * delta).sum()
    share = float(inside / delta.sum().clamp_min(1e-9))
    assert share > 0.90, (
        f"only {share:.1%} of the change falls inside the swept region — the "
        f"metric would be scoring models on a mask that misses the physics")


def test_swept_region_is_a_small_part_of_the_grid(data):
    """The complement of the above: if the mask covered most of the grid it
    would not be isolating anything, and the whole-image metric would do."""
    _, _, region = data
    frac = float(region.float().mean())
    assert 0.05 < frac < 0.45, f"swept region covers {frac:.1%} of the grid"


def test_change_outside_the_region_is_negligible_per_transition(data):
    """Per-transition, not just pooled: a pooled share can hide a minority of
    transitions whose change is mostly outside the mask."""
    o0, o1, region = data
    delta = (o1 - o0).abs()
    tot = delta.sum(dim=(1, 2)).clamp_min(1e-9)
    inside = (region * delta).sum(dim=(1, 2))
    share = inside / tot
    assert float((share > 0.75).float().mean()) > 0.95, (
        f"only {float((share > 0.75).float().mean()):.1%} of transitions have "
        f">75% of their change inside the mask")


def test_episode_split_does_not_leak():
    """`episode-split`: no episode may appear on both sides of a split."""
    from dmdc_baseline import split_by_episode, TransitionArrays
    ep = torch.arange(20).repeat_interleave(50)
    data = TransitionArrays(
        occ_t=torch.zeros(len(ep), 4, 4), occ_t1=torch.zeros(len(ep), 4, 4),
        actions=torch.zeros(len(ep), 4), episode_ids=ep,
        workspace_min=(-1.0, -1.0), workspace_max=(1.0, 1.0))
    torch.manual_seed(0)
    tr, te = split_by_episode(data, holdout_frac=0.25)
    assert not (set(ep[tr].tolist()) & set(ep[te].tolist())), "episodes leak across the split"
    assert tr.sum() + te.sum() == len(ep)
    assert te.sum() > 0 and tr.sum() > 0


def test_episode_split_matters_because_within_episode_states_are_similar(data):
    """Why the split rule exists, measured rather than asserted: consecutive
    transitions inside one episode share a pile, so a transition-level split
    would put near-duplicates on both sides."""
    o0, _, _ = data
    n = min(400, o0.shape[0])
    x = o0[:n].reshape(n, -1)
    # Files hold 320 transitions each; within-file neighbours share a pile.
    near = (x[:-1] - x[1:]).pow(2).mean()
    perm = torch.randperm(n - 1, generator=torch.Generator().manual_seed(0))
    far = (x[:-1] - x[1:][perm]).pow(2).mean()
    assert near < far, (
        f"neighbouring transitions ({near:.5f}) are not more similar than "
        f"random pairs ({far:.5f}) — the episode-split rule would be pointless")


def test_saved_configs_carry_git_provenance():
    """`dataset-provenance`: every config written by the collection path must
    carry the code state, so a dataset can be tied to the simulator that made
    it. Datasets collected before 2026-09-05 do not have this and never will --
    this test guards the path, not the history.
    """
    import inspect
    from Genesis import sandbox_manipulation_clean as smc
    src = inspect.getsource(smc.SandboxManipulation._save_config)
    assert "git_provenance" in src, (
        "_save_config no longer stamps utils.git_provenance() -- new datasets "
        "would be unreconstructable, which is how every pre-2026-09-05 dataset "
        "ended up with its lineage guessable only from file mtimes")


def test_git_provenance_reports_dirty_honestly():
    """A sha does not reconstruct a run if the tree was modified, so `dirty`
    has to be right. Two bugs were found here: tracked .pyc made it always
    true, and a fixed-index parse of stripped porcelain output clipped a path.
    """
    from utils import git_provenance
    p = git_provenance()
    assert set(("commit", "dirty", "branch")) <= set(p)
    assert isinstance(p["dirty"], bool) or p["dirty"] == "unknown"
    for f in p.get("dirty_files", []):
        assert not f.startswith(("enesis/", "ests/", "cripts/")), (
            f"path {f!r} is clipped -- porcelain parsed by fixed index again")
        assert "__pycache__" not in f and not f.endswith(".pyc")
