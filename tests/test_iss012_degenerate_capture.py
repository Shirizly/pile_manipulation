"""ISS-012: slateN capture must drop near-degenerate (pool, goal) cells by a RELATIVE threshold
(5 % of the split's median true spread), not only an absolute 1e-9 floor (EXP-0059 eval_extended.caps_from_raw)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/EXP-0059-retrieval-transition-model/code"))
EE = pytest.importorskip("eval_extended")


def test_near_tie_cell_dropped_normal_cells_kept():
    rng = np.random.default_rng(0)
    rows = [{"vt": rng.normal(0, 1, 16).tolist(), "vp": rng.normal(0, 1, 16).tolist()} for _ in range(20)]
    tie_vt = (1e-4 * rng.normal(0, 1, 16)).tolist()         # spread ~1e-4 of the others, well above 1e-9
    rows.append({"vt": tie_vt, "vp": rng.normal(0, 1, 16).tolist()})
    caps, thr = EE.caps_from_raw({"g": rows})
    assert len(caps["g"]) == 20 and thr > 1e-3
    old, _ = EE.caps_from_raw({"g": rows}, frac=0.0)         # the pre-fix absolute guard keeps it
    assert len(old["g"]) == 21


def test_capture_value_unchanged_for_kept_cells():
    vt = np.array([0.0, 1.0, 2.0, 3.0]); vp = np.array([3.0, 0.0, 1.0, 2.0])   # model picks index 1
    caps, _ = EE.caps_from_raw({"g": [{"vt": vt.tolist(), "vp": vp.tolist()}]})
    assert caps["g"] == [pytest.approx((vt.mean() - vt[1]) / (vt.mean() - vt.min()))]
