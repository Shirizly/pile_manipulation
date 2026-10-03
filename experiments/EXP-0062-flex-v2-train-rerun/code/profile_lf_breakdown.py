"""EXP-0062 RUN-0002 -- where the switched linear-foresight call spends its time.

Re-implements `Baselines/LinearForesight/model.py::predict_switched` stage by stage
(the same functions it calls: actions_to_pixels, bin_index, per-bin
to_push_frame -> A @ x -> from_push_frame -> push_frame_validity_mask +
blend_push_prediction -> clamp -> masked write), with a device sync after every
stage, on every DS-0019 slate's full pool as one batch. Asserts the staged output
equals `predict_occ` exactly. Also times the predict_occ body with the operators left on
the CPU (the registered behaviour before RUN-0002's predictor.py fix: `A.to(occ.device)`
per bin per call) vs pre-placed on the device (= predict_occ after the fix), and counts host<->device syncs from the
per-bin `bool(m.any())`.

    PYTHONPATH=. python -u experiments/EXP-0062-flex-v2-train-rerun/code/profile_lf_breakdown.py \
        --device cuda --out experiments/EXP-0062-flex-v2-train-rerun/results/timing_lf_parts/breakdown_cuda.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Baselines.common import eval_report as er  # noqa: E402
from fit_linear_foresight import actions_to_pixels  # noqa: E402
from Baselines.LinearForesight.model import bin_index, push_length_m  # noqa: E402
from Baselines.LinearForesight.predictor import SwitchedLinearForesightPredictor  # noqa: E402
from transforms.functional import (  # noqa: E402
    to_push_frame, from_push_frame, push_frame_validity_mask, blend_push_prediction,
)

sys.path.insert(0, str(Path(__file__).parent))
from time_inference import sub_batch  # noqa: E402


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def staged(ops, edges, b, res, crop, dev, T):
    """predict_switched, stage-timed (accumulates seconds into dict T)."""
    def tick(k, t0):
        sync(dev); t1 = time.perf_counter(); T[k] = T.get(k, 0.0) + t1 - t0; return t1
    sync(dev); t = time.perf_counter()
    H, W = b.H, b.W
    occ = b.occ0
    s, e = actions_to_pixels(b.actions, b.workspace_min, b.workspace_max, (H, W))
    L = push_length_m(b.actions)
    out = occ.clone()
    bins = bin_index(L, edges)
    t = tick("pixels_bins", t)
    for k, A in enumerate(ops):
        m = bins == k
        if not bool(m.any()):
            t = tick("mask_any_sync", t); continue
        o, ss, ee = occ[m], s[m], e[m]
        t = tick("mask_any_sync", t)
        for i in range(0, o.shape[0], 256):              # predict_world's batch loop
            sl = slice(i, i + 256)
            c = to_push_frame(o[sl], ss[sl], ee[sl], (res, res), crop); t = tick("warp", t)
            p = (A @ c.reshape(c.shape[0], -1).T).T.reshape(-1, res, res); t = tick("operator_matmul", t)
            back = from_push_frame(p, ss[sl], ee[sl], (H, W), crop); t = tick("unwarp", t)
            vm = push_frame_validity_mask(ss[sl], ee[sl], (H, W), (res, res), crop)
            r = blend_push_prediction(back, o[sl], vm, threshold=0.5).clamp(0.0, 1.0); t = tick("validity_blend", t)
            idx = m.nonzero(as_tuple=True)[0][sl]
            out[idx] = r; t = tick("scatter", t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/MODEL-0009-linear-foresight-flex-mask-v2/checkpoint.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    dev = a.device
    cell = er._load_cell(er.CORPORA["flex_ds0019_mask"], tag="flex_ds0019_mask")
    sids = cell.slate_idx.unique().tolist()
    rows = {s: (cell.slate_idx == s).nonzero(as_tuple=True)[0].tolist() for s in sids}
    on = {s: sub_batch(cell, r, dev) for s, r in rows.items()}
    pr = SwitchedLinearForesightPredictor(a.ckpt)
    ops_cpu = [A.cpu() for A in pr.operators]
    ops_dev = [A.to(dev) for A in pr.operators]
    edges = pr.bin_edges

    from Baselines.LinearForesight.model import predict_switched

    def call(ops, b):        # == predict_occ's body, with the operator list passed explicitly
        s_, e_ = actions_to_pixels(b.actions, b.workspace_min, b.workspace_max, (b.H, b.W))
        return predict_switched(edges, ops, b.occ0, s_, e_, push_length_m(b.actions), pr.res, (b.H, b.W), pr.crop)

    def run_predict(ops):
        ts = []
        for s in sids:
            sync(dev); t0 = time.perf_counter()
            with torch.no_grad():
                call(ops, on[s])
            sync(dev); ts.append(time.perf_counter() - t0)
        return ts

    res = dict(device=dev, ckpt=a.ckpt, n_slates=len(sids), n_candidates=sum(len(r) for r in rows.values()),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               operator_bytes_each=int(ops_cpu[0].numel() * ops_cpu[0].element_size()), n_operators=len(ops_cpu))
    for lab, ops in [("operators_on_cpu", ops_cpu), ("operators_resident", ops_dev)]:
        run_predict(ops)                                            # warm-up
        per = np.median([run_predict(ops) for _ in range(a.passes)], axis=0)
        res[f"predict_occ_{lab}"] = dict(per_slate_ms_median=float(1e3 * np.median(per)),
                                        total_all_slates_s=float(per.sum()),
                                        us_per_candidate=float(1e6 * per.sum() / res["n_candidates"]))
        print(lab, res[f"predict_occ_{lab}"], flush=True)
    # staged, resident operators; exactness vs predict_occ
    for s in sids[:5]:
        with torch.no_grad():
            ref = pr.predict_occ(on[s]); got = staged(ops_dev, edges, on[s], pr.res, pr.crop, dev, {})
        assert torch.equal(ref, got), "staged pipeline != predict_occ"
    T = {}
    with torch.no_grad():
        for _ in range(a.passes):
            for s in sids:
                staged(ops_dev, edges, on[s], pr.res, pr.crop, dev, T)
    tot = sum(T.values())
    res["staged_resident"] = dict(total_s_per_pass=tot / a.passes,
                                  per_slate_ms=1e3 * tot / a.passes / len(sids),
                                  share={k: v / tot for k, v in sorted(T.items(), key=lambda kv: -kv[1])},
                                  ms_per_slate={k: 1e3 * v / a.passes / len(sids) for k, v in T.items()},
                                  note="sync after every stage inflates the total vs predict_occ; read shares")
    nb = [int(torch.unique(bin_index(push_length_m(cell.actions[rows[s]]), edges)).numel()) for s in sids]
    res["syncs_per_call"] = dict(bool_m_any=len(ops_cpu), occupied_bins_per_slate_median=float(np.median(nb)),
                                 note="predict_switched: one bool(m.any()) per bin -> 6 host syncs per call, "
                                      "plus boolean-mask indexing (occ[m]) which also syncs to size the output")
    print(json.dumps(res["staged_resident"], indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp"; Path(tmp).write_text(json.dumps(res, indent=1)); os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
