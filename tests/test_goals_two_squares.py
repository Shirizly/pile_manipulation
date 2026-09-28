"""EXP-0046: the opt-in `two_squares` goal -- two disconnected squares,
inside the tray, each large enough for ~10 cubes, not in the default sets."""
import numpy as np
from scipy.ndimage import label

from Baselines.common.goals import two_squares_mask, dist_field_from_mask


def test_two_components_sizes_inside_tray():
    m = two_squares_mask(64, 64)
    assert m.shape == (64, 64) and m.dtype == bool
    lab, n = label(m)
    assert n == 2
    for k in (1, 2):
        rr, cc = np.nonzero(lab == k)
        assert rr.max() - rr.min() + 1 == 12 and cc.max() - cc.min() + 1 == 12
        assert (lab == k).sum() == 144
        # strictly inside the tray, off the walls
        assert rr.min() > 0 and cc.min() > 0 and rr.max() < 63 and cc.max() < 63
    # well separated along world x (rows), mirror-symmetric under x -> -x
    assert np.array_equal(m, m[::-1, :])
    rows = np.nonzero(m.any(1))[0]
    gap = np.diff(rows).max() - 1
    assert gap >= 16
    d = dist_field_from_mask(m)
    assert d[m].max() == 0 and d.max() == 1.0


def test_opt_in_only():
    from Baselines.common.eval_report import goal_names, _fixed_goal
    assert "two_squares" not in goal_names("many")
    assert len(goal_names("many")) == 30
    assert goal_names("many_plus")[-1] == "two_squares"
    mask, dist = _fixed_goal("two_squares", 64, 64)
    assert int(mask.sum()) == 288
