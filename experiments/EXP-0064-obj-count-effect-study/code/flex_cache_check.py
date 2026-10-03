"""EXP-0064 RUN-0013 -- pytest-free check that DS-0021 / DS-0022 read correctly through FlexData.

Reuses EXP-0061's code/image_mask_cache_check.py::check_frame_and_sweep VERBATIM (module globals
re-pointed): for 500 random rows of each corpus via FlexPileData(occ_source="image_mask" vs
"particles"): (1) mask-vs-particle-raster IoU over integer shifts -3..3 px (peak must be (0,0));
(2) fraction of REMOVED mask pixels (occ0 & !occ1) inside the swept region for actions as stored
(Y = -z) vs the 2nd/4th components negated; (3) a 4-row figure of the rendered tensors.
    python -u experiments/EXP-0064-obj-count-effect-study/code/flex_cache_check.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(next(REPO.glob("experiments/EXP-0061-*/code"))))
import image_mask_cache_check as chk  # noqa: E402

E = REPO / "experiments/EXP-0064-obj-count-effect-study"
chk.OUT = E / "results/figures/data_check"
chk.CFGS = {"DS-0021": str(next(REPO.glob("datasets/DS-0021-*/config.yaml"))),
            "DS-0022": str(next(REPO.glob("datasets/DS-0022-*/config.yaml")))}
if __name__ == "__main__":
    chk.OUT.mkdir(parents=True, exist_ok=True)
    res = chk.check_frame_and_sweep(np.random.default_rng(0))
    out = E / "results/flex_cache_check.json"
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)
