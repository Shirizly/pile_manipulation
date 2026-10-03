"""EXP-0062 RUN-0002 -- merge the per-configuration LF timing files
(results/timing_lf_parts/*.json, each a time_inference.py / profile_lf_breakdown.py output)
into results/timing_lf.json (atomic).

    python -u experiments/EXP-0062-flex-v2-train-rerun/code/aggregate_timing_lf.py
"""
import json
import os
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
P = R / "timing_lf_parts"
out = dict(headline="switched_cuda (operators resident on the GPU, the fixed predictor.py); "
                    "switched_cuda_asis = before the fix (operators copied host->device every call)",
           model="MODEL-0009 (lf_flex_switched / lf_flex_single, --ckpt weights/MODEL-0009-*/checkpoint.pt)",
           configs={}, breakdown={})
for f in sorted(P.glob("*.json")):
    if ".worker" in f.name:
        continue
    d = json.load(open(f))
    if f.name.startswith("breakdown"):
        out["breakdown"][f.stem.replace("breakdown_", "")] = d
        continue
    a = d.get("aggregate", {})
    out["configs"][f.stem] = dict(
        status=d.get("status"), command=d.get("command"),
        device_requested=d["workers"][0]["device_requested"] if d.get("workers") else None,
        output_device=sorted({w.get("output_device") for w in d.get("workers", [])} - {None}),
        contended=a.get("any_contended"),
        per_slate_ms_median=a.get("per_slate_ms_median"), per_slate_ms_p90=a.get("per_slate_ms_p90"),
        per_slate_ms_with_h2d_median=a.get("per_slate_ms_with_h2d_median"),
        us_per_candidate=a.get("us_per_candidate"), total_all_slates_s=a.get("total_all_slates_s"),
        batch128_ms_per_batch=a.get("batch128_ms_per_batch"), batch128_us_per_candidate=a.get("batch128_us_per_candidate"),
        loop_check_ratio=a.get("loop_check_ratio"), per_sample_loop_flag=a.get("per_sample_loop_suspected"),
        loop_flag_note="time_inference's flag fires on a count ratio > 2 at B=16 vs 128; for the SWITCHED model "
                       "the extra ops are the per-BIN loop (B=16 occupies fewer of the 6 bins), not a per-sample "
                       "loop -- single (no bins) has ratio 1.00",
        pool_size=d["workers"][0].get("pool_size") if d.get("workers") else None,
        gpu=d["workers"][0].get("gpu") if d.get("workers") else None)
tmp = R / "timing_lf.json.tmp"
tmp.write_text(json.dumps(out, indent=1))
os.replace(tmp, R / "timing_lf.json")
for k, v in out["configs"].items():
    print(k, v["per_slate_ms_median"]["median"] if v["per_slate_ms_median"] else None,
          v["us_per_candidate"]["median"] if v["us_per_candidate"] else None, v["output_device"])
