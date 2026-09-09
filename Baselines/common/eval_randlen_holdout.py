"""EXP-0030: held-out `accuracy` (swept-region, vs persistence, per
docs/experiments/METRICS.md) on the overnight_randlen corpus's OWN held-out
test split (`configs/dataset/genesis_overnight_randlen_test_n20.yaml`,
n20 groups only -- Baselines/common/data.py::load_cell hard-assumes N=20).

This is a narrower cousin of Baselines/common/eval_baseline.py's accuracy
section: no manifest (this corpus has no slate/step structure), no step-0
control-utility scoring (that is done on the register's own
slates_multistep cells via eval_baseline.py, unchanged). Persistence is the
only reference row -- the point is "how well does the trained model
reconstruct ITS OWN held-out data", not a cross-model comparison.

Usage
-----
    PYTHONPATH=. python Baselines/common/eval_randlen_holdout.py \\
        --predictor Baselines.NFD.predictor:build_predictor \\
        --tag nfd_randlen --out Baselines/NFD/runs/nfd_randlen_holdout_accuracy.json
"""
from __future__ import annotations

import argparse
import importlib
import json

from fit_linear_foresight import actions_to_pixels, metrics, swept_region_mask
from utils import git_provenance

from Baselines.common.data import load_cell
from Baselines.common.eval_baseline import _predictor_batch

TEST_CFG = "configs/dataset/genesis_overnight_randlen_test_n20.yaml"
R = 64


def _load_predictor(spec: str):
    mod_name, _, fn_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", required=True, help="module.path:factory_callable")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-cfg", default=TEST_CFG)
    args = ap.parse_args()

    prov = git_provenance()
    cell = load_cell(args.test_cfg, "train", tag=args.tag)
    n = cell.occ0.shape[0]
    H, W = cell.H, cell.W
    s_px, e_px = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

    predictor = _load_predictor(args.predictor)
    batch = _predictor_batch(cell)
    pred = predictor.predict_occ(batch).detach().to("cpu").to(cell.occ0.dtype)
    assert pred.shape == cell.occ0.shape

    m_model = metrics(pred, cell.occ1, cell.occ0, region=region)
    m_persist = metrics(cell.occ0, cell.occ1, cell.occ0, region=region)
    print(f"n={n} transitions ({cell.run_idx.unique().numel()} held-out files)")
    print(f"  persistence accuracy: {m_persist['accuracy']:+.4f}")
    print(f"  {predictor.name:20s} accuracy: {m_model['accuracy']:+.4f}")

    out = {"tag": args.tag, "provenance": prov, "n": n,
           "n_files": int(cell.run_idx.unique().numel()),
           "test_cfg": args.test_cfg,
           "accuracy": {"persistence": m_persist["accuracy"], predictor.name: m_model["accuracy"]}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
