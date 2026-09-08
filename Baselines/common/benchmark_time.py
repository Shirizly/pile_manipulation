"""Baselines/common/benchmark_time.py -- MPC-relevant wall-clock timing harness.

Motivation (verbatim from the user): "Will need to benchmark compute clock
time for all of these anyways, and compare MPC utility to that." An MPC step
scores EVERY candidate in a pool -- 128 in these slates, hundreds-to-thousands
in a real deployment -- so a model's value is not `slateK_exact` alone, it is
`slateK_exact` PER UNIT OF WALL-CLOCK spent getting it. This harness measures
that unit: the wall-clock cost of predicting a batch of K candidate actions
from ONE current state, for K in {1, 32, 128, 1024}, for every predictor this
project's scorer (`Baselines/common/eval_baseline.py`) can plug in.

Contract reuse (per task brief -- do not invent a second batch format)
-----------------------------------------------------------------------
Uses the SAME `BaselinePredictor` contract as the scorer
(`.name` / `.predict_occ(batch) -> (B,H,W)`) and the SAME `PredictorBatch`
dataclass, imported directly from `eval_baseline.py`. Real transitions come
from `Baselines.common.data.load_cell` (the same loader the scorer uses); a
K-candidate batch is built by taking ONE row's `occ0`/`states` (the "current
state") and repeating it K times, paired with K actions cycled from the
loaded eval cell's own `actions`/`p_start`/`p_stop`/`angle` -- i.e. exactly
the "same state, many candidate actions" shape an MPC inner loop repeats.
The reference `mean-delta`/`linear` predictors are fit with the EXACT same
calls `eval_baseline.py` makes (`fit_operator`/`canonicalise`/`predict_world`/
`predict_meandelta`, same `R`/`CR`/`RIDGE` constants, imported not
redefined) -- only the fitting CELL differs (single-cell `L20mm_train`, not
the pooled L20+L40 set) because fitting speed is irrelevant to this harness
(it is a one-time setup cost, not part of any timed region) and the smaller
cell is enough to produce a working operator to time against.

GPU timing methodology (see task brief -- this is the part that is easy to
get wrong)
-----------------------------------------------------------------------
1. `torch.cuda.synchronize()` immediately before AND after each timed region.
   CUDA is asynchronous; a naive `time.time()` around a forward pass measures
   queueing, not compute.
2. >= 3 warm-up calls discarded before timing (lazy CUDA context, cuDNN
   autotune, first-call allocator costs all land on iteration 0 otherwise).
3. >= 10 timed repeats; report MEDIAN and inter-quartile range, not mean or
   a single sample -- wall-clock timing on a shared machine is heavy-tailed
   (scheduler jitter, thermal throttling, and -- especially tonight -- other
   processes on the same GPU).
4. Record, per model, whether the timed region actually ran on GPU or CPU
   (introspected from the model's own parameter device AFTER the call, not
   assumed) and its parameter count.

A note on device, because it is a real (not hypothetical) methodological
trap in THIS repo: `eval_baseline.py` builds `PredictorBatch` straight from
`Baselines.common.data.load_cell`'s CPU tensors and never moves them to
CUDA. `NFDPredictor`/`SchenckPredictor` both derive their compute device
from `batch.occ0.device` (see their `predict_occ`), so AS ACTUALLY INVOKED
BY THE SCORER TODAY, both run on CPU regardless of where their checkpoint
was trained -- only `GNNPredictor` forces its own fixed (cuda-if-available)
device internally. This harness's `--device` flag controls the device of
the CONSTRUCTED CANDIDATE BATCH, which changes NFD/Schenck's effective
device (matching what the scorer would do if a future change moved batches
to CUDA) but has no effect on GNN (whose device is fixed at construction)
or on mean-delta/linear (deliberately kept CPU-only here, matching how they
are actually used -- see `Baselines/common/LOG.md`'s note that fitting them
is "NOT GPU work, all CPU tensor ops"). Both the requested device and the
device actually measured are recorded per model; a mismatch (e.g. `--device
cuda` but a model reports `cpu`) is exactly the kind of silent bug this
docstring exists to keep from being silently miscounted as a GPU number.

GPU-contention gate
-----------------------------------------------------------------------
Refuses (exits non-zero, writes nothing) if the GPU looks busy -- checked via
`torch.cuda.mem_get_info()` (more than ~10% of the 8 GB card already used is
suspicious for a card whose baselines are all sub-200k-parameter models,
see `gpu_lock.sh`'s own measurement) AND a process scan for this repo's own
training entry points (`train_nfd.py`, `train_schenck.py`, `training/train.py`,
`gpu_lock.sh`) -- unless `--force` is passed, in which case it proceeds but
marks `"contended": true` in the JSON and the run must be reported as
provisional (see `Baselines/common/TIMING.md`).

Usage
-----
    conda activate pme
    PYTHONPATH=. python Baselines/common/benchmark_time.py
    # add --force to record provisional numbers while the GPU is contended;
    # omit it (the default) to just validate the harness without recording.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch

from Baselines.common.data import load_cell
from Baselines.common.eval_baseline import PredictorBatch, R, CR, RIDGE
from fit_linear_foresight import actions_to_pixels, canonicalise, fit_operator, predict_world
from scripts.probes.exp0009_rerun import predict_meandelta
from utils import git_provenance

DEFAULT_TRAIN_CFG = "configs/dataset/genesis_slates_multistep_n20_L20mm_train.yaml"
DEFAULT_EVAL_CFG = "configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml"

# (display name, module, factory, default checkpoint path for size reporting)
EXTRA_PREDICTOR_SPECS = [
    ("gnn", "Baselines.GNN.predictor", "build_predictor",
     os.environ.get("GNN_CKPT", "Baselines/GNN/runs/ckpt_best.pth")),
    ("nfd_unet3ch", "Baselines.NFD.predictor", "build_predictor",
     "Baselines/NFD/runs/nfd_3ch/unet_best.pth"),
    ("schenck_singlenet", "Baselines.SchenckCNN.predictor", "build_predictor",
     "Baselines/SchenckCNN/runs/schenck.pth"),
]

TRAIN_PROC_NEEDLES = ("train_nfd.py", "train_schenck.py", "training/train.py",
                      "training.train", "gpu_lock.sh", "train_res_rgr.py")


# =============================================================================
# GPU contention gate
# =============================================================================

def check_gpu_contention() -> tuple[bool, dict]:
    if not torch.cuda.is_available():
        return False, {"cuda_available": False}
    free_b, total_b = torch.cuda.mem_get_info()
    used_frac = 1.0 - free_b / total_b
    try:
        ps_out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True,
                                 text=True, timeout=5).stdout
    except Exception as e:  # pragma: no cover -- best-effort only
        ps_out = ""
        print(f"[benchmark_time] WARNING: process scan failed ({e}); "
              f"contention check relies on memory only")
    my_pid = str(os.getpid())
    other_lines = [ln for ln in ps_out.splitlines()
                   if any(n in ln for n in TRAIN_PROC_NEEDLES) and f" {my_pid} " not in ln]
    details = {
        "cuda_available": True,
        "free_bytes": int(free_b), "total_bytes": int(total_b),
        "used_frac": round(used_frac, 4),
        "other_training_processes": other_lines,
    }
    contended = used_frac > 0.10 or len(other_lines) > 0
    return contended, details


# =============================================================================
# Reference predictors (mean-delta / linear) -- same calls eval_baseline.py
# makes, fit on a single (small) cell rather than the pooled train set.
# =============================================================================

class _RefPredictor:
    def __init__(self, name, fn, param_tensor):
        self.name = name
        self._fn = fn
        self._param_tensor = param_tensor
        self.model = None  # no nn.Module; timed on CPU, see module docstring

    def predict_occ(self, batch):
        return self._fn(batch)


def build_reference_predictors(train_cfg: str):
    print(f"[benchmark_time] fitting mean-delta/linear on {train_cfg} "
          f"(one-time setup, not timed) ...")
    t0 = time.time()
    data_tr = load_cell(train_cfg, "train")
    H, W = data_tr.H, data_tr.W
    s_tr, e_tr = actions_to_pixels(data_tr.actions, data_tr.workspace_min,
                                    data_tr.workspace_max, (H, W))
    n_tr = data_tr.occ0.shape[0]
    Y0 = canonicalise(data_tr.occ0, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    Y1 = canonicalise(data_tr.occ1, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    A = fit_operator(Y0, Y1, ridge=RIDGE, toward_identity=True)
    bmd = (Y1 - Y0).mean(dim=1)
    ws_min, ws_max = data_tr.workspace_min, data_tr.workspace_max
    print(f"[benchmark_time] fit done in {time.time() - t0:.1f}s ({n_tr} transitions)")

    # `A`/`bmd` are fit on CPU (see module docstring -- deliberately not
    # ported to GPU); the candidate batch may have been built on CUDA for
    # the neural predictors (`--device`), so pull just the two tensors these
    # closures need back to CPU rather than moving `A`/`bmd` to the batch's
    # device on every call.
    def _md(batch):
        occ0, actions = batch.occ0.cpu(), batch.actions.cpu()
        s_px, e_px = actions_to_pixels(actions, ws_min, ws_max, (batch.H, batch.W))
        return predict_meandelta(bmd, occ0, s_px, e_px, R, (batch.H, batch.W), CR)

    def _lin(batch):
        occ0, actions = batch.occ0.cpu(), batch.actions.cpu()
        s_px, e_px = actions_to_pixels(actions, ws_min, ws_max, (batch.H, batch.W))
        return predict_world(A, occ0, s_px, e_px, R, (batch.H, batch.W), CR)

    return [_RefPredictor("mean-delta", _md, bmd), _RefPredictor("linear", _lin, A)]


def load_extra_predictors():
    """GNN / NFD / SchenckCNN, via the exact `--predictor module:factory`
    spec the scorer uses. Skips gracefully (printed note) if a module or
    checkpoint is missing -- NFD/Schenck may still be training tonight."""
    out = []
    for name, mod_name, fn_name, ckpt_path in EXTRA_PREDICTOR_SPECS:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            factory = getattr(mod, fn_name)
            predictor = factory()
            out.append((predictor, ckpt_path, None))
            print(f"[benchmark_time] loaded predictor {predictor.name!r} from {mod_name}")
        except Exception as e:
            print(f"[benchmark_time] SKIPPING {name}: {type(e).__name__}: {e}")
            out.append((None, ckpt_path, f"{type(e).__name__}: {e}"))
    return out


# =============================================================================
# Candidate-pool batch construction -- "same state, K candidate actions"
# =============================================================================

def build_candidate_batch(cell, k: int, device: str) -> PredictorBatch:
    n = cell.occ0.shape[0]
    idx0 = 0
    cand_idx = torch.tensor([(idx0 + j) % n for j in range(k)], dtype=torch.long)
    occ0 = cell.occ0[idx0].unsqueeze(0).expand(k, -1, -1).contiguous().to(device)
    states = cell.states[idx0].unsqueeze(0).expand(k, -1, -1).contiguous().to(device)
    run_idx = cell.run_idx[idx0].expand(k).contiguous()  # same current state -> same run
    return PredictorBatch(
        occ0=occ0,
        actions=cell.actions[cand_idx].to(device),
        states=states,
        p_start=cell.p_start[cand_idx].to(device),
        p_stop=cell.p_stop[cand_idx].to(device),
        angle=cell.angle[cand_idx].to(device),
        run_idx=run_idx,
        raw=cell.raw,
        workspace_min=cell.workspace_min, workspace_max=cell.workspace_max,
        H=cell.H, W=cell.W,
    )


# =============================================================================
# Timing core
# =============================================================================

def _median_iqr(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    med = statistics.median(xs)

    def pct(p):
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    return {"median": med, "q1": pct(0.25), "q3": pct(0.75), "n": n}


def time_predictor_at_k(predictor, batch, k: int, warmup: int, repeats: int,
                          cuda_sync: bool) -> dict:
    for _ in range(warmup):
        _ = predictor.predict_occ(batch)
    if cuda_sync:
        torch.cuda.synchronize()

    total_s = []
    for _ in range(repeats):
        if cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = predictor.predict_occ(batch)
        if cuda_sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_s.append(t1 - t0)

    total_ms = [t * 1e3 for t in total_s]
    per_cand_us = [t * 1e6 / k for t in total_s]
    return {"total_ms": _median_iqr(total_ms), "per_candidate_us": _median_iqr(per_cand_us)}


def actual_device_of(predictor) -> str:
    model = getattr(predictor, "model", None)
    if model is not None:
        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            pass
    dev = getattr(predictor, "device", None)
    if dev is not None:
        return str(dev)
    return "cpu"


def param_count_of(predictor) -> int | None:
    model = getattr(predictor, "model", None)
    if model is not None:
        return sum(p.numel() for p in model.parameters())
    pt = getattr(predictor, "_param_tensor", None)
    if pt is not None:
        return int(pt.numel())
    return None


def ckpt_bytes_of(ckpt_path: str | None) -> int | None:
    if ckpt_path and os.path.exists(ckpt_path):
        return os.path.getsize(ckpt_path)
    return None


# =============================================================================
# Report writers
# =============================================================================

def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


RERUN_CMD = ("conda activate pme && PYTHONPATH=. python "
             "Baselines/common/benchmark_time.py --force")


def write_markdown(path, payload):
    lines = []
    lines.append("# MPC candidate-pool timing\n")
    lines.append(
        "Measures the wall-clock cost of predicting a batch of K candidate\n"
        "actions from one current state -- the operation an MPC inner loop\n"
        "repeats every step (K=128 in these slates; hundreds-to-thousands in a\n"
        "real deployment). See `Baselines/common/benchmark_time.py`'s module\n"
        "docstring for exact methodology (warm-up, CUDA sync, median/IQR,\n"
        "device provenance).\n")
    lines.append("\n## Re-run on an idle GPU\n")
    lines.append(f"```\n{RERUN_CMD}\n```\n")
    lines.append(
        "(Drop `--force` first to confirm the contention gate reports\n"
        "`contended: false` before trusting the numbers -- it refuses to write\n"
        "any output at all otherwise.)\n")

    if payload["contended"]:
        lines.append(
            "\n> **PROVISIONAL -- taken while the GPU was contended.** "
            "NFD and Schenck were both training concurrently on this card "
            "when these numbers were recorded (see `contention_details` in "
            "`timing_results.json`). Absolute latencies below are almost "
            "certainly inflated versus an idle card, and the inflation is not "
            "uniform across models (queueing depends on the other jobs' own "
            "kernel sizes) -- **do not use these to rank models by "
            "cost/candidate**, only to sanity-check the harness's methodology "
            "and units. Re-run with the command above once training finishes.\n")
    else:
        lines.append("\nGPU was idle (contention gate passed) when these numbers were recorded.\n")

    lines.append(f"\nRecorded: {payload['timestamp']} UTC. "
                 f"Commit: `{payload['provenance'].get('commit', '?')}`"
                 f"{' (dirty)' if payload['provenance'].get('dirty') else ''}.\n")

    lines.append("\n## Model size\n")
    lines.append("| model | device (requested) | device (actual) | param count | checkpoint bytes | note |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in payload["results"].items():
        if r.get("error"):
            lines.append(f"| {name} | - | - | - | - | SKIPPED: {r['error']} |")
            continue
        pc = f"{r['param_count']:,}" if r["param_count"] is not None else "n/a"
        cb = f"{r['ckpt_bytes']:,}" if r["ckpt_bytes"] is not None else "n/a (fit in-memory)"
        lines.append(f"| {name} | {payload['device_requested']} | {r['device_actual']} | "
                     f"{pc} | {cb} | |")

    lines.append("\n## Per-candidate cost vs. pool size K\n")
    lines.append("Median [IQR], over "
                 f"{payload['repeats']} repeats ({payload['warmup']} warm-up discarded).\n")
    ks = payload["ks"]
    header = "| model | " + " | ".join(f"K={k} (us/candidate)" for k in ks) + " |"
    sep = "|---|" + "---|" * len(ks)
    lines.append(header)
    lines.append(sep)
    for name, r in payload["results"].items():
        if r.get("error"):
            continue
        cells = []
        for k in ks:
            bk = r["by_k"].get(str(k))
            if bk is None:
                cells.append("n/a")
                continue
            pc = bk["per_candidate_us"]
            cells.append(f"{pc['median']:.1f} [{pc['q1']:.1f}, {pc['q3']:.1f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines.append("\n## Total batch latency vs. pool size K\n")
    header = "| model | " + " | ".join(f"K={k} (ms)" for k in ks) + " |"
    lines.append(header)
    lines.append(sep)
    for name, r in payload["results"].items():
        if r.get("error"):
            continue
        cells = []
        for k in ks:
            bk = r["by_k"].get(str(k))
            if bk is None:
                cells.append("n/a")
                continue
            tm = bk["total_ms"]
            cells.append(f"{tm['median']:.2f} [{tm['q1']:.2f}, {tm['q3']:.2f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines.append(
        "\n## Reading this table\n"
        "- **Per-candidate us** is the number that matters for choosing a pool\n"
        "  size K under a wall-clock budget -- it is what one extra candidate\n"
        "  costs.\n"
        "- **Total batch latency** is what matters for a single MPC step's\n"
        "  deadline -- fixed per-call overhead (Python/CUDA launch, any\n"
        "  per-row Python loop such as the GNN's particle rasterisation) does\n"
        "  not shrink with K, and can dominate at small K even for a model\n"
        "  that is cheap per-candidate at large K.\n"
        "- `mean-delta`/`linear` are timed on CPU throughout (matching how\n"
        "  they are actually invoked in this repo -- see the module\n"
        "  docstring); a GPU port was not attempted since the task is fitting\n"
        "  a small dense operator, not a neural forward pass.\n"
        "- The GNN predictor's `predict_occ` loops in Python over the batch\n"
        "  to call `rasterize_particles` once per row (OpenCV, not batched;\n"
        "  see `Baselines/common/data.py`'s docstring) -- expect its\n"
        "  per-candidate cost to fall much less steeply with K than a model\n"
        "  whose entire forward pass is one batched tensor op.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-cfg", default=DEFAULT_TRAIN_CFG)
    ap.add_argument("--eval-cfg", default=DEFAULT_EVAL_CFG)
    ap.add_argument("--ks", default="1,32,128,1024")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                     help="device for the CONSTRUCTED CANDIDATE BATCH (affects NFD/Schenck, "
                          "which derive their compute device from the batch; has no effect "
                          "on GNN, whose device is fixed at construction, or on mean-delta/"
                          "linear, kept CPU-only -- see module docstring)")
    ap.add_argument("--out-json", default="Baselines/common/timing_results.json")
    ap.add_argument("--out-md", default="Baselines/common/TIMING.md")
    ap.add_argument("--force", action="store_true",
                     help="record numbers even if the GPU is contended (marks "
                          "\"contended\": true; label the result provisional).")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    contended, contention_details = check_gpu_contention()
    if contended:
        print("\n" + "=" * 78)
        print("[benchmark_time] WARNING: GPU LOOKS BUSY -- timings would be contaminated.")
        print(f"  used_frac={contention_details.get('used_frac')}  "
              f"other training processes: {len(contention_details.get('other_training_processes', []))}")
        for ln in contention_details.get("other_training_processes", []):
            print(f"    {ln.strip()}")
        print("=" * 78 + "\n")
        if not args.force:
            print("[benchmark_time] refusing to record numbers (no --force). "
                  "Re-run once the GPU is idle, or pass --force for a PROVISIONAL "
                  "reading (will be marked contended:true).")
            sys.exit(3)
        print("[benchmark_time] --force given: proceeding, results will be marked "
              "contended:true and must be reported as provisional.\n")

    prov = git_provenance()
    print(f"provenance: {prov}")

    print(f"\n[benchmark_time] loading eval cell {args.eval_cfg} for candidate rows ...")
    cell = load_cell(args.eval_cfg, "train")
    print(f"[benchmark_time] {cell.occ0.shape[0]} transitions available to cycle as candidates")

    predictors = []  # list of (name, predictor_or_None, ckpt_path, error_or_None)
    for p in build_reference_predictors(args.train_cfg):
        predictors.append((p.name, p, None, None))
    for predictor, ckpt_path, err in load_extra_predictors():
        name = predictor.name if predictor is not None else ckpt_path
        predictors.append((getattr(predictor, "name", None) or os.path.basename(ckpt_path),
                            predictor, ckpt_path, err))

    cuda_sync = torch.cuda.is_available()
    results = {}
    for name, predictor, ckpt_path, err in predictors:
        if predictor is None:
            results[name] = {"error": err, "ckpt_path": ckpt_path}
            continue
        print(f"\n--- timing {name} ---")
        by_k = {}
        model_err = None
        for k in ks:
            try:
                batch = build_candidate_batch(cell, k, args.device)
                r = time_predictor_at_k(predictor, batch, k, args.warmup, args.repeats, cuda_sync)
                by_k[str(k)] = r
                print(f"  K={k:5d}  total_ms median={r['total_ms']['median']:.3f} "
                      f"[{r['total_ms']['q1']:.3f},{r['total_ms']['q3']:.3f}]   "
                      f"per_candidate_us median={r['per_candidate_us']['median']:.2f} "
                      f"[{r['per_candidate_us']['q1']:.2f},{r['per_candidate_us']['q3']:.2f}]")
            except Exception as e:
                print(f"  K={k:5d}  FAILED: {type(e).__name__}: {e}")
                model_err = f"{type(e).__name__}: {e} (at K={k})"
                break
        results[name] = {
            "error": model_err,
            "ckpt_path": ckpt_path,
            "device_actual": actual_device_of(predictor) if model_err is None or by_k else None,
            "param_count": param_count_of(predictor),
            "ckpt_bytes": ckpt_bytes_of(ckpt_path),
            "by_k": by_k,
        }

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": prov,
        "contended": contended,
        "contention_details": contention_details,
        "device_requested": args.device,
        "ks": ks,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    write_json(args.out_json, payload)
    write_markdown(args.out_md, payload)


if __name__ == "__main__":
    main()
