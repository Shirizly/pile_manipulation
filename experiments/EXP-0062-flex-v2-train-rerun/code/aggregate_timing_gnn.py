"""EXP-0062 RUN-0003 -- merge the GNN timing parts (results/timing_gnn_parts/) into results/timing_gnn.json.
    python -u experiments/EXP-0062-flex-v2-train-rerun/code/aggregate_timing_gnn.py"""
import json, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; os.chdir(REPO)
R = Path("experiments/EXP-0062-flex-v2-train-rerun/results")
P = R / "timing_gnn_parts"
out = dict(
    note=("CUDA rows (gnn_v2_n30_cuda_*) were measured 2026-10-02 ~19:30 after the GPU was reset, with the same "
          "time_inference.py settings as RUN-0001/0002 (device check on one extra untimed call with hooks on every submodule, "
          "because the GNN bypasses its top-level __call__; timed passes carry the same hooks as the other models). "
          "ORIGINAL CPU NOTE: GPU UNAVAILABLE for the first run (nvidia-smi: 'GPU requires reset' from ~16:50 CEST 2026-10-02; torch.cuda.is_available() False), "
          "so every number is CPU (torch 4 threads, OMP_NUM_THREADS=4, machine otherwise idle). NOT comparable with the CUDA rows "
          "of RUN-0001/0002; same-device references: NFD v2 CPU (measured here) and LF v2 CPU (RUN-0002 timing_lf.json). The forward-hook "
          "device probe sees no call for the GNN (it calls predict_one_step / model.forward, not __call__), so the device is the "
          "predictor's (cpu) by construction -- no GPU exists in the process."),
    model="gnn_flex_v2_n30 (MODEL-0010)",
    configs={})
for f, lab in [("gnn_cuda_with_perception.json", "gnn_v2_n30_cuda_INCLUDING_perception (cache_states=False)"),
               ("gnn_cuda_cached.json", "gnn_v2_n30_cuda_model_and_render_only (perception cached after warm-up)"),
               ("gnn_cpu_with_perception.json", "gnn_v2_n30_cpu_INCLUDING_perception (cache_states=False: PNG imread + segment + back-project + voxel + FPS + recenter every call)"),
               ("gnn_cpu_cached.json", "gnn_v2_n30_cpu_model_and_render_only (perception cached after warm-up, = the other models' 'model path')"),
               ("nfd_cpu_reference.json", "nfd_v2_s0_cpu_reference (MODEL-0008 on the same CPU)")]:
    if (P / f).exists():
        d = json.load(open(P / f))
        out["configs"][lab] = dict(status=d.get("status"), aggregate=d.get("aggregate"), command=d.get("command"))
for f in ("breakdown_cpu.json", "breakdown_cpu_fast.json"):
    if (P / f).exists():
        out["stage_breakdown_" + f.split(".")[0]] = json.load(open(P / f))
lf = R / "timing_lf.json"
if lf.exists():
    out["lf_v2_cpu_reference_from_RUN-0002"] = "see results/timing_lf.json (switched CPU 55.7 ms/slate, single 39.7)"
json.dump(out, open(R / "timing_gnn.json.tmp", "w"), indent=1); os.replace(R / "timing_gnn.json.tmp", R / "timing_gnn.json")
for k, v in out["configs"].items():
    a = v["aggregate"]
    print(k[:40], "ms/slate", a["per_slate_ms_median"]["values"], "us/cand", round(a["us_per_candidate"]["median"], 1),
          "b128 us/cand", round(a["batch128_us_per_candidate"]["median"], 1), "loop", a["per_sample_loop_suspected"], a["loop_check_ratio"][0])
