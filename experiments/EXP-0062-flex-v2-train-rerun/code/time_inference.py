"""EXP-0062 -- REUSABLE inference-timing harness for any eval_report model on the
FleX DS-0019 slates (written for NFD; the visual-foresight and GNN agents reuse it).

What is timed (per process):

(a) MODEL path, per slate, the slate's WHOLE candidate pool as ONE batch:
    ``predictor.predict_occ(batch)`` on a batch whose inputs (cached binary
    image mask occ0, actions) are ALREADY on ``--device`` -- i.e. everything the
    model does from the cached mask: model-specific preprocessing (e.g. NFD's
    two ``draw_plate_soft`` plate channels, the GNN's own perception), forward,
    postprocess to the (B, 64, 64) mask, and the predictor's own return
    (NFD's ``.cpu()``). ``torch.cuda.synchronize()`` immediately before and after.
    Also reported: the same with the host->device copy of the inputs INCLUDED
    (``with_h2d``), and a fixed batch of 128 candidates (all DS-0019 rows,
    chunked into 128-row batches, partial tail dropped) for comparability.
(b) SHARED perception cost, per image: colour PNG -> binary 64x64 image mask
    (``FlexData/image_mask.py``): ``cv2.imread`` and ``segment + mask_to_frac
    + threshold`` timed separately, on the 100 DS-0019 initial-state PNGs.
    Every image-mask model pays this once per MPC step (not per candidate).

Rules implemented: warm-up passes excluded; synchronize around every timed
region; the device of the tensors ACTUALLY fed to every nn.Module forward is
captured by forward-pre-hooks and asserted == ``--device``; the measurement
repeats in ``--procs`` (default 3) SEPARATE processes for a noise floor; a
per-sample Python-loop check (count of aten CPU ops and CUDA kernel launches
per call at batch 16 vs 128 via torch.profiler: a batch-independent count =
vectorised, a count scaling with B = a per-sample loop -- the CODEMAP trap).

    PYTHONPATH=. python -u experiments/EXP-0062-flex-v2-train-rerun/code/time_inference.py \\
        --model nfd_randlen --ckpt Baselines/NFD/runs/nfd_3ch_flex_mask_v2_seed0/unet_best.pth \\
        --name nfd_v2_seed0 --out experiments/EXP-0062-flex-v2-train-rerun/results/timing_nfd.json

    # another registered model (its own MODELS default ckpt): --model gnn_flex_drp (no --ckpt)
    # extra factory kwargs: --kwargs '{"particle_num": 50}'

Output: ``--out`` JSON (atomic), rewritten after every worker process; per-worker
raw JSON beside it (``<out>.worker{i}.json``). Run on an otherwise idle GPU:
the contention check (``benchmark_time.check_gpu_contention``) is recorded per
worker and the run refuses to start if contended unless ``--allow-contended``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Baselines.common import eval_report as er  # noqa: E402

TRUTH_FIELDS = ("occ1", "states_")


def _write_atomic(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def _stats(xs):
    a = np.asarray(xs, dtype=float)
    return dict(n=int(a.size), median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                p90=float(np.percentile(a, 90)), mean=float(a.mean()), min=float(a.min()),
                max=float(a.max()), total=float(a.sum()))


def sub_batch(batch, idx, device=None):
    """Rows `idx` of a predictor batch (every per-row tensor field sliced; truth
    fields never moved), inputs moved to `device` if given -- eval_report._predict's
    slicing, for an arbitrary index set."""
    N = batch.occ0.shape[0]
    t = torch.as_tensor(idx, dtype=torch.long)
    upd = {}
    for f in dataclasses.fields(batch):
        v = getattr(batch, f.name)
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N:
            v = v[t]
            if device is not None and f.name not in TRUTH_FIELDS:
                v = v.to(device)
            upd[f.name] = v
    return dataclasses.replace(batch, **upd)


def modules_of(predictor):
    """Every torch.nn.Module reachable as a direct attribute of the predictor
    (one level of nesting into plain objects)."""
    found = []
    for v in vars(predictor).values():
        if isinstance(v, torch.nn.Module):
            found.append(v)
        elif hasattr(v, "__dict__") and not isinstance(v, (type, torch.Tensor)):
            found += [w for w in vars(v).values() if isinstance(w, torch.nn.Module)]
    return found


class DeviceProbe:
    """forward-pre-hooks recording the device of every tensor fed to forward."""

    def __init__(self, mods):
        self.seen = set()
        self.calls = 0
        self.handles = [m.register_forward_pre_hook(self._hook) for m in mods]

    def _hook(self, mod, inputs):
        self.calls += 1
        for x in inputs:
            if torch.is_tensor(x):
                self.seen.add(str(x.device))
            elif isinstance(x, (list, tuple)):
                self.seen.update(str(y.device) for y in x if torch.is_tensor(y))

    def remove(self):
        for h in self.handles:
            h.remove()


def timed_call(predictor, b, dev):
    _sync(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = predictor.predict_occ(b)
    if torch.is_tensor(out) and out.is_cuda:
        _sync(dev)
    _sync(dev)
    return time.perf_counter() - t0, out


def loop_check(predictor, batch, rows, dev):
    """aten CPU-op and CUDA-kernel counts per predict_occ call at B=16 and B=128."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if str(dev).startswith("cuda") else [])
    out = {}
    for B in (16, 128):
        b = sub_batch(batch, rows[:B], dev)
        timed_call(predictor, b, dev)
        with profile(activities=acts) as prof:
            timed_call(predictor, b, dev)
        ev = prof.events()
        out[B] = dict(aten_ops=sum(1 for e in ev if e.name.startswith("aten::")),
                      cuda_kernels=sum(1 for e in ev if getattr(e, "device_type", None) is not None
                                       and "CUDA" in str(e.device_type)))
        if B == 128:   # where the time goes: top aten ops by self device (else CPU) time
            ka = prof.key_averages()
            key = "self_device_time_total" if hasattr(next(iter(ka)), "self_device_time_total") else "self_cuda_time_total"
            if not str(dev).startswith("cuda"):
                key = "self_cpu_time_total"
            top = sorted((k for k in ka if k.key.startswith("aten::")), key=lambda k: -getattr(k, key, 0))[:8]
            out["top_ops_b128_us"] = {k.key: float(getattr(k, key, 0)) for k in top}
            out["top_ops_metric"] = key
    r = {k: (out[128][k] / max(1, out[16][k])) for k in ("aten_ops", "cuda_kernels")}
    out["ratio_128_over_16"] = r
    out["per_sample_loop_suspected"] = bool(r["aten_ops"] > 2.0 or r["cuda_kernels"] > 2.0)
    return out


def png_mask_timing(n_images, warmup=3):
    from FlexData.dataset import load_instance_config
    from FlexData import image_mask as im
    import cv2
    ds_dir = REPO / "datasets/DS-0019-slates-flex-pile-varN"
    paths = json.load(open(ds_dir / "cache/image_paths.json"))["states"]
    cfg = load_instance_config(ds_dir / "config.yaml")
    grid = im.GridSpec.from_instance(cfg)
    cam = im.Camera.load()
    files = [ds_dir / paths[k]["initial_color"] for k in sorted(paths, key=int)][:n_images]
    for f in files[:warmup]:
        im.image_to_occupancy(str(f), grid, cam)          # builds the cached plane LUT
    t_read, t_mask = [], []
    for f in files:
        t0 = time.perf_counter()
        col = cv2.imread(str(f), cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        m = (im.mask_to_frac(im.segment(col), cam, grid) > im.THRESHOLD).astype(np.float32)
        t2 = time.perf_counter()
        assert m.shape == (grid.resolution, grid.resolution)
        t_read.append(t1 - t0); t_mask.append(t2 - t1)
    return dict(n_images=len(files), device="cpu (numpy/cv2; no GPU path exists)",
                imread_ms=_stats([1e3 * t for t in t_read]),
                segment_warp_threshold_ms=_stats([1e3 * t for t in t_mask]),
                total_per_image_ms=_stats([1e3 * (a + b) for a, b in zip(t_read, t_mask)]))


def worker(args):
    from Baselines.common.benchmark_time import check_gpu_contention
    dev = args.device
    contended, cdet = check_gpu_contention() if dev.startswith("cuda") else (False, {})
    spec = dict(er.MODELS[args.model])
    if args.ckpt:
        spec["ckpt"] = args.ckpt
    if args.kwargs:
        spec["kwargs"] = dict(spec.get("kwargs", {}), **json.loads(args.kwargs))
    t0 = time.time()
    cell = er._load_cell(er.CORPORA[args.corpus], tag=args.corpus)
    batch = er._predictor_batch(cell) if hasattr(cell, "states") else cell
    predictor = er._load_predictor(spec)
    load_s = time.time() - t0
    sids = cell.slate_idx.unique().tolist()
    slate_rows = {s: (cell.slate_idx == s).nonzero(as_tuple=True)[0].tolist() for s in sids}
    on_dev = {s: sub_batch(batch, r, dev) for s, r in slate_rows.items()}
    probe = DeviceProbe(modules_of(predictor))

    # warm-up passes (excluded)
    for _ in range(args.warmup):
        for s in sids:
            timed_call(predictor, on_dev[s], dev)
    probe.seen.clear()

    per_slate = {s: [] for s in sids}
    per_slate_h2d = {s: [] for s in sids}
    H, W = cell.H, cell.W
    for _ in range(args.passes):
        for s in sids:
            dt, out = timed_call(predictor, on_dev[s], dev)
            assert tuple(out.shape) == (len(slate_rows[s]), H, W), out.shape
            per_slate[s].append(dt)
        for s in sids:                                  # inputs start on the host
            _sync(dev)
            t0 = time.perf_counter()
            b = sub_batch(batch, slate_rows[s], dev)
            dt, _ = timed_call(predictor, b, dev)
            per_slate_h2d[s].append(time.perf_counter() - t0)
    fed = sorted(probe.seen)
    probe_calls = probe.calls
    if probe.handles and not fed:
        # the top-level module's forward was never called (e.g. the GNN calls a
        # method like predict_one_step that bypasses __call__): check the device on
        # ONE extra untimed call with hooks on every submodule, so the timed passes
        # above carry exactly the same hook overhead as the other models' runs
        deep = DeviceProbe([s for m in modules_of(predictor) for s in m.modules()])
        timed_call(predictor, on_dev[sids[0]], dev)
        fed, probe_calls = sorted(deep.seen), deep.calls
        deep.remove()

    # fixed batch 128 over all rows
    allrows = [r for s in sids for r in slate_rows[s]]
    chunks = [allrows[i:i + 128] for i in range(0, len(allrows) - 127, 128)]
    ch_dev = [sub_batch(batch, c, dev) for c in chunks]
    for b in ch_dev[:3]:
        timed_call(predictor, b, dev)
    t128 = []
    for _ in range(args.passes):
        t128.append([timed_call(predictor, b, dev)[0] for b in ch_dev])
    t128_med = np.median(np.asarray(t128), axis=0)
    probe.remove()

    loops = loop_check(predictor, batch, slate_rows[sids[0]] if len(slate_rows[sids[0]]) >= 128 else allrows, dev)
    med = {s: float(np.median(v)) for s, v in per_slate.items()}
    med_h2d = {s: float(np.median(v)) for s, v in per_slate_h2d.items()}
    pool = {s: len(r) for s, r in slate_rows.items()}
    n_tot = sum(pool.values())
    res = dict(
        pid=os.getpid(), model=args.model, name=args.name, ckpt=spec.get("ckpt"), kwargs=spec.get("kwargs", {}),
        corpus=args.corpus, device_requested=dev,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        torch=torch.__version__, contended=contended, contention=cdet,
        device_fed_to_forward=fed, n_forward_hook_calls=probe_calls,
        n_modules_hooked=len(probe.handles), load_s=load_s,
        warmup_passes=args.warmup, timed_passes=args.passes, n_slates=len(sids), n_candidates=n_tot,
        pool_size=_stats(list(pool.values())),
        per_slate_ms=_stats([1e3 * v for v in med.values()]),
        per_slate_ms_with_h2d=_stats([1e3 * v for v in med_h2d.values()]),
        us_per_candidate_slate_batches=1e6 * sum(med.values()) / n_tot,
        us_per_candidate_slate_batches_with_h2d=1e6 * sum(med_h2d.values()) / n_tot,
        total_all_slates_s=sum(med.values()),
        total_all_slates_s_with_h2d=sum(med_h2d.values()),
        batch128=dict(n_batches=len(chunks), ms_per_batch=_stats(1e3 * t128_med),
                      us_per_candidate=float(1e6 * t128_med.sum() / (128 * len(chunks)))),
        loop_check=loops,
        per_slate_raw_ms={str(s): [1e3 * x for x in v] for s, v in per_slate.items()},
    )
    if dev.startswith("cuda"):
        assert fed and all(d.startswith("cuda") for d in fed) or not probe.handles, \
            f"forward inputs on {fed}, requested {dev}"
    # predictors with no nn.Module (e.g. linear foresight): the hooks see nothing, so also
    # record + assert the device of the returned prediction (added for RUN-0002 LF)
    _, out_chk = timed_call(predictor, on_dev[sids[0]], dev)
    res["output_device"] = str(out_chk.device)
    if not probe.handles:
        assert res["output_device"].split(":")[0] == dev.split(":")[0], \
            f"no nn.Module to hook and output on {res['output_device']}, requested {dev}"
    if args.png:
        res["png_to_mask"] = png_mask_timing(args.png)
    return res


def aggregate(workers):
    def across(key, sub="median"):
        v = [w[key][sub] if isinstance(w[key], dict) else w[key] for w in workers]
        return dict(values=v, median=float(np.median(v)),
                    rel_spread=float((max(v) - min(v)) / np.median(v)) if np.median(v) else None)
    agg = dict(
        n_processes=len(workers),
        per_slate_ms_median=across("per_slate_ms", "median"),
        per_slate_ms_p90=across("per_slate_ms", "p90"),
        per_slate_ms_with_h2d_median=across("per_slate_ms_with_h2d", "median"),
        us_per_candidate=across("us_per_candidate_slate_batches"),
        us_per_candidate_with_h2d=across("us_per_candidate_slate_batches_with_h2d"),
        total_all_slates_s=across("total_all_slates_s"),
        batch128_ms_per_batch=dict(values=[w["batch128"]["ms_per_batch"]["median"] for w in workers]),
        batch128_us_per_candidate=dict(values=[w["batch128"]["us_per_candidate"] for w in workers]),
        device_fed_to_forward=sorted({d for w in workers for d in w["device_fed_to_forward"]}),
        any_contended=any(w["contended"] for w in workers),
        per_sample_loop_suspected=any(w["loop_check"]["per_sample_loop_suspected"] for w in workers),
        loop_check_ratio=[w["loop_check"]["ratio_128_over_16"] for w in workers],
    )
    for k in ("batch128_ms_per_batch", "batch128_us_per_candidate"):
        v = agg[k]["values"]
        agg[k].update(median=float(np.median(v)), rel_spread=float((max(v) - min(v)) / np.median(v)))
    pngs = [w["png_to_mask"] for w in workers if "png_to_mask" in w]
    if pngs:
        agg["png_to_mask_ms_per_image"] = dict(
            imread=[p["imread_ms"]["median"] for p in pngs],
            segment_warp_threshold=[p["segment_warp_threshold_ms"]["median"] for p in pngs],
            total=[p["total_per_image_ms"]["median"] for p in pngs])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="eval_report MODELS key (spec: module/factory/ckpt)")
    ap.add_argument("--ckpt", default=None, help="checkpoint override (default: the spec's ckpt)")
    ap.add_argument("--kwargs", default=None, help="JSON dict merged into the spec's factory kwargs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--corpus", default="flex_ds0019_mask")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=2, help="warm-up passes over all slates (excluded)")
    ap.add_argument("--passes", type=int, default=3, help="timed passes; per-slate time = median")
    ap.add_argument("--procs", type=int, default=3, help="separate processes (noise floor)")
    ap.add_argument("--png", type=int, default=100, help="#DS-0019 initial PNGs for the PNG->mask timing (0 = skip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-contended", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    if args.worker:
        res = worker(args)
        if res["contended"] and not args.allow_contended:
            res["WARNING"] = "GPU contended during this worker"
        _write_atomic(args.out, res)
        return
    argv = [a for a in sys.argv[1:]]
    o = argv.index("--out")
    workers = []
    summary = dict(command=" ".join([sys.executable, "-u"] + sys.argv), started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   model=args.model, name=args.name, ckpt=args.ckpt, workers=[], status="running")
    for i in range(args.procs):
        wout = f"{args.out}.worker{i}.json"
        wargv = argv[:o] + argv[o + 2:] + ["--out", wout, "--worker"]
        print(f"[time_inference] worker {i}: {' '.join(wargv)}", flush=True)
        subprocess.run([sys.executable, "-u", __file__] + wargv, check=True)
        w = json.load(open(wout))
        if w["contended"] and not args.allow_contended:
            raise SystemExit(f"GPU contended in worker {i}: {w['contention']} (use --allow-contended)")
        workers.append(w)
        summary["workers"].append({k: v for k, v in w.items() if k != "per_slate_raw_ms"})
        summary["aggregate"] = aggregate(workers)
        _write_atomic(args.out, summary)
        print(f"[time_inference] worker {i}: per-slate median {w['per_slate_ms']['median']:.3f} ms, "
              f"{w['us_per_candidate_slate_batches']:.2f} us/cand, b128 {w['batch128']['us_per_candidate']:.2f} "
              f"us/cand, fed {w['device_fed_to_forward']}, loop {w['loop_check']['ratio_128_over_16']}", flush=True)
    summary["status"] = "complete"
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_atomic(args.out, summary)
    print(json.dumps(summary["aggregate"], indent=1))


if __name__ == "__main__":
    main()
