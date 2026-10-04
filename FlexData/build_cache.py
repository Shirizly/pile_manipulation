"""FlexData/build_cache.py -- compact, checkpointed caches for the FleX corpora.

    python -u -m FlexData.build_cache ds0020 [--workers 12]   # DS-0020 v2 (trajectory-manifest payload)
    python -u -m FlexData.build_cache splits                  # DS-0020 v2 splits.json (trajectories 0-99 excluded)
    python -u -m FlexData.build_cache paths                   # DS-0020 v2 cache/image_paths.json
    python -u -m FlexData.build_cache ds0019 [--workers 12]   # DS-0019 (do not rebuild casually: EXP-0061 test corpus)
    python -u -m FlexData.build_cache splits_ds0019 | paths_ds0019
    python -u -m FlexData.build_cache ds0020_v1               # ARCHIVED DS-0020 v1 (old_data/), chunk format
    python -u -m FlexData.build_cache ds0021|splits_ds0021|paths_ds0021   # EXP-0064 count-group train (DS-0020 v2 layout)
    python -u -m FlexData.build_cache ds0022|splits_ds0022|paths_ds0022   # EXP-0064 count-group test slates (DS-0019 layout)

Reads the raw ported layout (see each dataset's DATASET.md) ONCE and writes:

DS-0020 v2 (2026-10-02; raw ``data/true_action_transitions_carrots/``: ``manifest.jsonl``
with ``state_init`` / ``transition`` / ``state_failed`` records, per-trajectory dirs
``<t>/initial_particles.npy, <k>_after_particles.npy``; state k+1 = after-state of push k,
state 0 = initial) -> ``cache/v2_chunk_{c:03d}.npz`` (100 state_idx slots each; only
trajectories with a ``state_init`` and all ``n_steps`` VALID transitions are stored).
Particle count VARIES per trajectory, so particles are concatenated along axis 1:
    traj_ids  (T,)          int32    p_off (T+1,) int64  -- traj j owns xz[:, p_off[j]:p_off[j+1]]
    xz        (11, sumP, 2) float16  raw flex (x, z); state 0 = initial, k = after push k-1
    y_max     (T, 11) f32,  bbox (T, 11, 4) f32 [x_min, x_max, z_min, z_max],  n_nan (T, 11) i32
    actions   (T, 10, 4)    float32  [x0, z0, x1, z1] as stored (table frame, see dataset.py)
    push_length / bin / retries (T, 10)   from the manifest
    max_disp / p99_disp / n_moved_010 / n_moved_025 (T, 10)
    n_particles / n_rigids (T,) int32 (rigids = initial_state.npz rigid_rotations),
    init_pos  (T,) int8  0 = rand_blob, 1 = rand_spread
DS-0020 v1 (ARCHIVED, ``old_data/<t>/``) -> ``old_data/_ported_v1/cache/chunk_{k:03d}.npz``
    (100 trajectories each, fixed P = 19513):
    xz        (T, 11, P, 2) float16   particle (x, z) per state, FleX units
    y_max     (T, 11)       float32   max particle height per state
    actions   (T, 10, 4)    float32   [x0, z0, x1, z1] per push (push k: state k -> k+1)
    traj_ids  (T,)          int32
    max_disp  (T, 10)       float32   max per-particle XZ displacement (float32 source)
    p99_disp  (T, 10)       float32
    n_moved_010 / n_moved_025 (T, 10) int32  particles with XZ disp > 0.10 / 0.25
    bbox      (T, 11, 4)    float32   [x_min, x_max, z_min, z_max] per state
    n_nan     (T, 11)       int32
DS-0019 -> ``datasets/DS-0019-*/cache/state_{s:03d}.npz`` (one same-state slate each)
    init_xz (P, 2) f16, after_xz (A, P, 2) f16, action_idx (A,), actions (A, 4),
    push_length (A,), bin (A,), plus the same per-row displacement stats and
    n_particles / n_rigids / init_pos.

Every unit is written atomically (``.tmp`` + ``os.replace``) and the
``cache/manifest.json`` is rewritten after every unit (project standing rule:
every job must survive being cut off). Existing complete units are skipped,
so a re-run resumes.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DS0020 = REPO / "datasets" / "DS-0020-training-data-flex-N864"
DS0020_RAW = DS0020 / "data" / "true_action_transitions_carrots"     # v2 payload
DS0020_V1_RAW = DS0020 / "old_data"                                  # archived v1 payload
DS0020_V1 = DS0020 / "old_data" / "_ported_v1"                       # v1 cache/config/splits
V2_N_STEPS = 10
V2_EXCLUDE = range(0, 100)    # near-copies of DS-0019's 100 test piles (same collector seed)
DS0019 = REPO / "datasets" / "DS-0019-slates-flex-pile-varN"
# EXP-0064 count-group corpora (raw true_action_* format, same layout as DS-0020 v2 / DS-0019,
# but the slate manifest lives in data/). Same -z action frame (flex-action-frame-neg-z).
DS0021 = REPO / "datasets" / "DS-0021-flex-carrots-countgroups-train"
DS0021_RAW = DS0021 / "data"
DS0022 = REPO / "datasets" / "DS-0022-flex-carrots-countgroups-test-slates"
DS0021_GROUP_SIZE = 500       # count_group = state_idx // 500 (DS-0021), // 50 (DS-0022)
CHUNK = 100
HIST_EDGES = np.arange(-10.0, 10.0 + 1e-9, 0.1)


def _atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _disp_stats(a: np.ndarray, b: np.ndarray):
    """(max, p99, n>0.10, n>0.25) of per-particle XZ displacement, float32."""
    d = np.hypot(b[:, 0] - a[:, 0], b[:, 2] - a[:, 2])
    return float(d.max()), float(np.percentile(d, 99)), int((d > 0.10).sum()), int((d > 0.25).sum())


def _read_particles(path: Path) -> np.ndarray:
    return np.load(path).reshape(-1, 4)


# ----------------------------------------------------------------- DS-0020 v1 (archived)
def _v1_n_traj() -> int:
    return len([p for p in DS0020_V1_RAW.iterdir() if p.is_dir() and p.name.isdigit()])


def _load_traj(t: int):
    d = DS0020_V1_RAW / str(t)
    acts = np.asarray(pickle.load(open(d / "actions.p", "rb")), dtype=np.float32)
    states = [_read_particles(d / f"{k}_particles.npy") for k in range(11)]
    P = states[0].shape[0]
    assert all(s.shape[0] == P for s in states), f"traj {t}: particle count varies"
    S = np.stack(states)                                     # (11, P, 4)
    xz32 = S[:, :, [0, 2]]
    xz16 = xz32.astype(np.float16)
    stats = [_disp_stats(S[k], S[k + 1]) for k in range(10)]
    bbox = np.stack([xz32[..., 0].min(1), xz32[..., 0].max(1),
                     xz32[..., 1].min(1), xz32[..., 1].max(1)], axis=1)
    hx = np.histogram(xz32[..., 0], HIST_EDGES)[0]
    hz = np.histogram(xz32[..., 1], HIST_EDGES)[0]
    return dict(
        t=t, xz=xz16, y_max=S[:, :, 1].max(1).astype(np.float32), actions=acts,
        stats=np.asarray(stats, dtype=np.float64), bbox=bbox.astype(np.float32),
        n_nan=np.isnan(S).reshape(11, -1).sum(1).astype(np.int32),
        f16_err=float(np.abs(xz16.astype(np.float32) - xz32).max()),
        inv_mass=np.unique(S[..., 3]).tolist(), hx=hx, hz=hz, P=P,
    )


def build_ds0020_v1(workers: int) -> None:
    """ARCHIVED v1 payload -> old_data/_ported_v1/cache (already built; resumable no-op)."""
    out = DS0020_V1 / "cache"
    out.mkdir(exist_ok=True)
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {
        "format": "FlexData/build_cache.py ds0020", "chunk_size": CHUNK, "chunks": {},
        "hist_edges": [float(HIST_EDGES[0]), float(HIST_EDGES[-1]), 0.1],
        "hist_x": [0] * (len(HIST_EDGES) - 1), "hist_z": [0] * (len(HIST_EDGES) - 1),
    }
    n_traj = _v1_n_traj()
    n_chunks = (n_traj + CHUNK - 1) // CHUNK
    with ProcessPoolExecutor(workers) as ex:
        for c in range(n_chunks):
            name = f"chunk_{c:03d}.npz"
            if name in man["chunks"] and (out / name).exists():
                continue
            t0 = time.time()
            ids = list(range(c * CHUNK, min((c + 1) * CHUNK, n_traj)))
            rs = list(ex.map(_load_traj, ids))
            st = np.stack([r["stats"] for r in rs])          # (T, 10, 4)
            _atomic_npz(
                out / name,
                xz=np.stack([r["xz"] for r in rs]), y_max=np.stack([r["y_max"] for r in rs]),
                actions=np.stack([r["actions"] for r in rs]),
                traj_ids=np.asarray(ids, dtype=np.int32),
                max_disp=st[..., 0].astype(np.float32), p99_disp=st[..., 1].astype(np.float32),
                n_moved_010=st[..., 2].astype(np.int32), n_moved_025=st[..., 3].astype(np.int32),
                bbox=np.stack([r["bbox"] for r in rs]), n_nan=np.stack([r["n_nan"] for r in rs]),
            )
            man["hist_x"] = (np.asarray(man["hist_x"]) + sum(r["hx"] for r in rs)).tolist()
            man["hist_z"] = (np.asarray(man["hist_z"]) + sum(r["hz"] for r in rs)).tolist()
            man["chunks"][name] = dict(
                traj=[ids[0], ids[-1]], n_particles=sorted({r["P"] for r in rs}),
                f16_max_abs_err=max(r["f16_err"] for r in rs),
                n_nan=int(sum(r["n_nan"].sum() for r in rs)),
                inv_mass=sorted({v for r in rs for v in r["inv_mass"]}),
                seconds=round(time.time() - t0, 1),
            )
            man["complete"] = len(man["chunks"]) == n_chunks
            _atomic_json(mpath, man)
            print(f"[ds0020] {name} traj {ids[0]}-{ids[-1]} {time.time() - t0:.1f}s", flush=True)


# ----------------------------------------------------------------- DS-0019
def _load_state(args):
    s, recs, init_pos, data = args
    data = Path(data)
    d = data / str(s)
    init = _read_particles(d / "initial_particles.npy")
    st = np.load(d / "initial_state.npz")
    assert np.array_equal(st["positions"], init), f"state {s}: npz/npy mismatch"
    recs = sorted(recs, key=lambda r: r["action_idx"])
    afters = [_read_particles(data / r["after_positions_path"]) for r in recs]
    assert all(a.shape == init.shape for a in afters), f"state {s}: particle count varies"
    A = np.stack(afters) if afters else np.zeros((0,) + init.shape, np.float32)
    stats = np.asarray([_disp_stats(init, a) for a in afters], dtype=np.float64).reshape(-1, 4)
    allx = np.concatenate([init[None], A])[..., [0, 2]]
    return dict(
        s=s,
        init_xz=init[:, [0, 2]].astype(np.float16),
        after_xz=A[:, :, [0, 2]].astype(np.float16),
        init_y_max=np.float32(init[:, 1].max()),
        action_idx=np.asarray([r["action_idx"] for r in recs], dtype=np.int32),
        actions=np.asarray([r["action"] for r in recs], dtype=np.float32).reshape(-1, 4),
        push_length=np.asarray([r["push_length"] for r in recs], dtype=np.float32),
        bin=np.asarray([r["bin"] for r in recs], dtype=np.int32),
        stats=stats, n_rigids=int(st["rigid_rotations"].shape[0]), n_particles=int(init.shape[0]),
        init_pos=init_pos, n_nan=int(np.isnan(A).sum() + np.isnan(init).sum()),
        f16_err=float(np.abs(allx.astype(np.float16).astype(np.float32) - allx).max()) if allx.size else 0.0,
        bbox=[float(allx[..., 0].min()), float(allx[..., 0].max()),
              float(allx[..., 1].min()), float(allx[..., 1].max())],
        hx=np.histogram(allx[..., 0], HIST_EDGES)[0], hz=np.histogram(allx[..., 1], HIST_EDGES)[0],
    )


def build_ds0019(workers: int, ds: Path = DS0019, manifest: Path | None = None, tag: str = "ds0019") -> None:
    """Slate corpus -> ``<ds>/cache/state_*.npz``. ``manifest`` defaults to ``<ds>/manifest.jsonl``
    (DS-0022: ``<ds>/data/manifest.jsonl``); paths in it are relative to ``<ds>/data``."""
    manifest = manifest or ds / "manifest.jsonl"
    out = ds / "cache"
    out.mkdir(exist_ok=True)
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {
        "format": f"FlexData/build_cache.py {tag}", "states": {},
        "hist_edges": [float(HIST_EDGES[0]), float(HIST_EDGES[-1]), 0.1],
        "hist_x": [0] * (len(HIST_EDGES) - 1), "hist_z": [0] * (len(HIST_EDGES) - 1),
    }
    recs = [json.loads(l) for l in open(manifest)]
    inits = {r["state_idx"]: r for r in recs if r["type"] == "state_init"}
    by_state: dict[int, list] = {s: [] for s in inits}
    n_invalid = {s: 0 for s in inits}
    for r in recs:
        if r["type"] == "action":
            if r["valid"]:
                by_state[r["state_idx"]].append(r)
            else:
                n_invalid[r["state_idx"]] += 1
    todo = [(s, by_state[s], inits[s].get("init_pos"), str(ds / "data")) for s in sorted(inits)
            if f"state_{s:03d}.npz" not in man["states"]]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_load_state, todo):
            s = r["s"]
            name = f"state_{s:03d}.npz"
            _atomic_npz(out / name, init_xz=r["init_xz"], after_xz=r["after_xz"],
                        action_idx=r["action_idx"], actions=r["actions"],
                        push_length=r["push_length"], bin=r["bin"],
                        max_disp=r["stats"][:, 0].astype(np.float32),
                        p99_disp=r["stats"][:, 1].astype(np.float32),
                        n_moved_010=r["stats"][:, 2].astype(np.int32),
                        n_moved_025=r["stats"][:, 3].astype(np.int32),
                        n_particles=np.int32(r["n_particles"]), n_rigids=np.int32(r["n_rigids"]),
                        init_y_max=r["init_y_max"])
            man["hist_x"] = (np.asarray(man["hist_x"]) + r["hx"]).tolist()
            man["hist_z"] = (np.asarray(man["hist_z"]) + r["hz"]).tolist()
            man["states"][name] = dict(
                state_idx=s, n_valid=int(len(r["action_idx"])), n_invalid=n_invalid[s],
                n_particles=r["n_particles"], n_rigids=r["n_rigids"], init_pos=r["init_pos"],
                n_nan=r["n_nan"], f16_max_abs_err=r["f16_err"], bbox=r["bbox"],
            )
            man["complete"] = len(man["states"]) == len(inits)
            _atomic_json(mpath, man)
            print(f"[{tag}] {name} valid={len(r['action_idx'])} P={r['n_particles']} "
                  f"rigids={r['n_rigids']}", flush=True)


# ----------------------------------------------------------------- DS-0020 v2
def read_v2_manifest(raw: Path = DS0020_RAW) -> dict:
    """Parse the v2 ``manifest.jsonl`` -> {inits {t: rec}, steps {t: {k: rec}},
    failed {t: rec}, n_invalid, n_dup}. Duplicate records (a trajectory re-run after
    a reset crash, e.g. 821) keep the LAST one, which is what is on disk."""
    inits, steps, failed, n_inv, n_dup = {}, {}, {}, 0, 0
    for line in open(raw / "manifest.jsonl"):
        r = json.loads(line)
        t = int(r["state_idx"])
        if r["type"] == "state_init":
            n_dup += t in inits
            inits[t] = r
        elif r["type"] == "transition":
            if not r["valid"]:
                n_inv += 1
                continue
            n_dup += int(r["step_idx"]) in steps.get(t, {})
            steps.setdefault(t, {})[int(r["step_idx"])] = r
        elif r["type"] == "state_failed":
            failed[t] = r
    return dict(inits=inits, steps=steps, failed=failed, n_invalid=n_inv, n_dup=n_dup)


def v2_complete_trajectories(man: dict) -> list[int]:
    """state_idx with an init record and every one of the V2_N_STEPS valid transitions."""
    return sorted(t for t in man["inits"]
                  if sorted(man["steps"].get(t, {})) == list(range(V2_N_STEPS)) and t not in man["failed"])


def _load_traj_v2(args):
    t, init, steps, raw = args
    raw = Path(raw)
    d = raw / str(t)
    init_p = _read_particles(raw / init["positions_path"])
    st = np.load(d / "initial_state.npz")
    init_match = bool(np.array_equal(st["positions"], init_p))
    states = [init_p] + [_read_particles(raw / steps[k]["after_positions_path"]) for k in range(V2_N_STEPS)]
    P = init_p.shape[0]
    assert P == int(init["n_particles"]), f"traj {t}: n_particles {P} != manifest {init['n_particles']}"
    assert all(s.shape[0] == P for s in states), f"traj {t}: particle count varies"
    S = np.stack(states)                                     # (11, P, 4)
    xz32 = S[:, :, [0, 2]]
    xz16 = xz32.astype(np.float16)
    stats = [_disp_stats(S[k], S[k + 1]) for k in range(V2_N_STEPS)]
    fin = np.where(np.isfinite(xz32), xz32, np.nan)
    bbox = np.stack([np.nanmin(fin[..., 0], 1), np.nanmax(fin[..., 0], 1),
                     np.nanmin(fin[..., 1], 1), np.nanmax(fin[..., 1], 1)], axis=1)
    rec = [steps[k] for k in range(V2_N_STEPS)]
    return dict(
        t=t, xz=xz16, P=P, y_max=np.nanmax(S[:, :, 1], 1).astype(np.float32),
        actions=np.asarray([r["action"] for r in rec], np.float32),
        push_length=np.asarray([r["push_length"] for r in rec], np.float32),
        bin=np.asarray([r["bin"] for r in rec], np.int32),
        retries=np.asarray([r["retries"] for r in rec], np.int32),
        stats=np.asarray(stats, dtype=np.float64), bbox=bbox.astype(np.float32),
        n_nan=np.isnan(S).reshape(11, -1).sum(1).astype(np.int32),
        n_rigids=int(st["rigid_rotations"].shape[0]), init_match=init_match,
        init_pos=0 if init["init_pos"] == "rand_blob" else 1,
        f16_err=float(np.nanmax(np.abs(xz16.astype(np.float32) - xz32)[np.abs(xz32) < 10])) if np.isfinite(xz32).any() else 0.0,
        hx=np.histogram(xz32[..., 0][np.isfinite(xz32[..., 0])], HIST_EDGES)[0],
        hz=np.histogram(xz32[..., 1][np.isfinite(xz32[..., 1])], HIST_EDGES)[0],
    )


def build_ds0020(workers: int, ds: Path = DS0020, raw: Path = DS0020_RAW, tag: str = "ds0020 v2") -> None:
    """DS-0020 v2 (or any trajectory corpus in its raw layout, e.g. DS-0021) ->
    <ds>/cache/v2_chunk_{c:03d}.npz + cache/manifest.json (atomic per chunk, resumable)."""
    out = ds / "cache"
    out.mkdir(exist_ok=True)
    mpath = out / "manifest.json"
    vm = read_v2_manifest(raw)
    keep = v2_complete_trajectories(vm)
    man = json.loads(mpath.read_text()) if mpath.exists() else {
        "format": f"FlexData/build_cache.py {tag} (v2 trajectory-manifest payload)", "chunk_size": CHUNK,
        "raw": str(raw.relative_to(REPO)), "chunks": {},
        "n_state_init": len(vm["inits"]), "n_failed": len(vm["failed"]), "failed": sorted(vm["failed"]),
        "n_invalid_transitions": vm["n_invalid"], "n_duplicate_records": vm["n_dup"],
        "n_complete_trajectories": len(keep),
        "incomplete": sorted(set(vm["inits"]) - set(keep)),
        "hist_edges": [float(HIST_EDGES[0]), float(HIST_EDGES[-1]), 0.1],
        "hist_x": [0] * (len(HIST_EDGES) - 1), "hist_z": [0] * (len(HIST_EDGES) - 1),
    }
    n_slots = max(vm["inits"]) + 1
    n_chunks = (n_slots + CHUNK - 1) // CHUNK
    with ProcessPoolExecutor(workers) as ex:
        for c in range(n_chunks):
            name = f"v2_chunk_{c:03d}.npz"
            if name in man["chunks"] and (out / name).exists():
                continue
            t0 = time.time()
            ids = [t for t in keep if c * CHUNK <= t < (c + 1) * CHUNK]
            rs = list(ex.map(_load_traj_v2, [(t, vm["inits"][t], vm["steps"][t], str(raw)) for t in ids]))
            st = np.stack([r["stats"] for r in rs])          # (T, 10, 4)
            p_off = np.concatenate([[0], np.cumsum([r["P"] for r in rs])]).astype(np.int64)
            _atomic_npz(
                out / name, traj_ids=np.asarray(ids, np.int32), p_off=p_off,
                xz=np.concatenate([r["xz"] for r in rs], axis=1),
                y_max=np.stack([r["y_max"] for r in rs]), actions=np.stack([r["actions"] for r in rs]),
                push_length=np.stack([r["push_length"] for r in rs]), bin=np.stack([r["bin"] for r in rs]),
                retries=np.stack([r["retries"] for r in rs]),
                max_disp=st[..., 0].astype(np.float32), p99_disp=st[..., 1].astype(np.float32),
                n_moved_010=st[..., 2].astype(np.int32), n_moved_025=st[..., 3].astype(np.int32),
                bbox=np.stack([r["bbox"] for r in rs]), n_nan=np.stack([r["n_nan"] for r in rs]),
                n_particles=np.asarray([r["P"] for r in rs], np.int32),
                n_rigids=np.asarray([r["n_rigids"] for r in rs], np.int32),
                init_pos=np.asarray([r["init_pos"] for r in rs], np.int8),
            )
            man["hist_x"] = (np.asarray(man["hist_x"]) + sum(r["hx"] for r in rs)).tolist()
            man["hist_z"] = (np.asarray(man["hist_z"]) + sum(r["hz"] for r in rs)).tolist()
            man["chunks"][name] = dict(
                state_idx_range=[c * CHUNK, (c + 1) * CHUNK - 1], n_traj=len(ids),
                n_particles=[int(min(r["P"] for r in rs)), int(max(r["P"] for r in rs))],
                f16_max_abs_err_inside_10=max(r["f16_err"] for r in rs),
                n_nan=int(sum(r["n_nan"].sum() for r in rs)),
                init_npz_mismatch=[r["t"] for r in rs if not r["init_match"]],
                seconds=round(time.time() - t0, 1),
            )
            man["complete"] = len(man["chunks"]) == n_chunks
            _atomic_json(mpath, man)
            print(f"[{tag}] {name} {len(ids)} traj, P {man['chunks'][name]['n_particles']} "
                  f"{time.time() - t0:.1f}s", flush=True)


def write_image_paths(ds: Path = DS0020, raw: Path = DS0020_RAW) -> None:
    """DS-0020 v2 ``cache/image_paths.json``: per (trajectory, state k) colour PNG path
    relative to the dataset dir (the cache dir's parent, as ``FlexGNNPredictor`` and
    ``FlexData.image_mask`` resolve it); state 0 = ``initial_color.png``, state k =
    ``<k-1>_after_color.png``. No depth PNGs in v2 (depth slot always null)."""
    vm = read_v2_manifest(raw)
    keep = v2_complete_trajectories(vm)
    rel = raw.relative_to(ds)
    trajs, missing = {}, 0
    for t in keep:
        cs = [vm["inits"][t]["color_path"]] + [vm["steps"][t][k]["after_color_path"] for k in range(V2_N_STEPS)]
        st = []
        for c in cs:
            ok = (raw / c).exists(); missing += not ok
            st.append([str(rel / c) if ok else None, None])
        trajs[str(t)] = st
    _atomic_json(ds / "cache" / "image_paths.json", {
        "layout": "trajectories[traj_id][state k] = [color_png, depth_png] (relative to dataset dir; "
                  "state 0 = initial, k = after push k-1; depth always null in v2; null = missing)",
        "n_missing": missing, "trajectories": trajs})
    print(f"image paths: {ds.name} {len(trajs)} traj, {missing} missing")


def write_image_paths_v1() -> None:
    """ARCHIVED v1 index (old_data/_ported_v1/cache/image_paths.json, paths ``../<t>/...``)."""
    n = _v1_n_traj()
    trajs, missing = {}, 0
    for t in range(n):
        st = []
        for k in range(11):
            c, d = f"../{t}/{k}_color.png", f"../{t}/{k}_depth.png"
            ok_c, ok_d = (DS0020_V1 / c).exists(), (DS0020_V1 / d).exists()
            missing += (not ok_c) + (not ok_d)
            st.append([c if ok_c else None, d if ok_d else None])
        trajs[str(t)] = st
    _atomic_json(DS0020_V1 / "cache" / "image_paths.json", {
        "layout": "trajectories[traj_id][state k] = [color_png, depth_png] (relative to old_data/_ported_v1; null = missing)",
        "n_missing": missing, "trajectories": trajs})


def write_image_paths_ds0019(ds: Path = DS0019, manifest: Path | None = None) -> None:
    recs = [json.loads(l) for l in open(manifest or ds / "manifest.jsonl")]
    states, miss19 = {}, 0
    for r in recs:
        if r["type"] == "state_init":
            c = r["color_path"]; ok = (ds / "data" / c).exists(); miss19 += not ok
            states.setdefault(str(r["state_idx"]), {"actions": {}})["initial_color"] = f"data/{c}" if ok else None
    for r in recs:
        if r["type"] == "action" and r["valid"]:
            c = r["after_color_path"]; ok = (ds / "data" / c).exists(); miss19 += not ok
            states[str(r["state_idx"])]["actions"][str(r["action_idx"])] = f"data/{c}" if ok else None
    _atomic_json(ds / "cache" / "image_paths.json", {
        "layout": "states[state_idx] = {initial_color, actions{action_idx: after_color}} (valid actions only; no depth PNGs in this corpus)",
        "n_missing": miss19, "states": states})
    print(f"image paths: {ds.name} {len(states)} states, {miss19} missing")


def write_splits(seed: int = 0, val_frac: float = 0.1) -> None:
    """DS-0020 v2: train/val split BY TRAJECTORY (rows inside one trajectory are a
    chain, hence correlated) over the complete trajectories with state_idx >= 100,
    fixed seed, explicit id lists. Trajectories 0-99 are EXCLUDED from every split
    (leakage: same collector seed as DS-0019 -> near-copies of its 100 test piles)."""
    vm = read_v2_manifest()
    keep = v2_complete_trajectories(vm)
    excl = sorted(t for t in keep if t in V2_EXCLUDE)
    cand = np.asarray(sorted(t for t in keep if t not in V2_EXCLUDE))
    perm = np.random.default_rng(seed).permutation(len(cand))
    n_val = int(round(val_frac * len(cand)))
    val, train = sorted(cand[perm[:n_val]].tolist()), sorted(cand[perm[n_val:]].tolist())
    assert not set(val) & set(train) and not (set(val) | set(train)) & set(V2_EXCLUDE)
    _atomic_json(DS0020 / "splits.json", {
        "unit": "trajectory id = manifest state_idx (datasets/DS-0020-*/data/true_action_transitions_carrots/<id>/)",
        "seed": seed,
        "method": "candidates = complete trajectories (state_init + 10 valid transitions, not state_failed) with "
                  "state_idx >= 100, sorted; perm = np.random.default_rng(seed).permutation(n_cand); "
                  "first round(0.1 n_cand) of perm -> val",
        "made_by": "python -u -m FlexData.build_cache splits",
        "excluded": {"ids": excl,
                     "reason": "leakage: DS-0020 v2 was collected with the same seed as DS-0019, so trajectories 0-99 "
                               "start from near-copies of DS-0019's 100 test piles (same piece count per index; initial "
                               "image-mask IoU median 0.906, min 0.72 vs 0.425 unrelated). User decision 2026-10-02: drop "
                               "ALL steps of trajectories 0-99 from train and val."},
        "absent": {"ids": sorted(set(range(max(vm["inits"]) + 1)) - set(keep)),
                   "reason": "state_failed (repeated_reset_crash) -- no data"},
        "splits": {"train": train, "val": val, "test": []},
    })
    print(f"splits: DS-0020 v2 train {len(train)} / val {len(val)} trajectories; excluded {len(excl)}")


def write_splits_ds0021(val_frac: float = 0.1) -> None:
    """DS-0021: train/val BY STATE (= trajectory), with EXP-0064's GroupedParticleDataset rule
    EXACTLY (experiments/EXP-0064-*/code/dataset_grouped_particles.py): per count_group (from the
    state_init record), the group's complete states sorted, first round(0.9 n) -> train, rest -> val.
    So the val trajectories are the ones the GNN (MODEL-0011/0012) was validated on."""
    vm = read_v2_manifest(DS0021_RAW)
    keep = v2_complete_trajectories(vm)
    by_g: dict[int, list] = {}
    for t in keep:
        by_g.setdefault(int(vm["inits"][t]["count_group"]), []).append(t)
    train, val = [], []
    for g in sorted(by_g):
        st = sorted(by_g[g]); n_tr = int(round(len(st) * (1 - val_frac)))
        train += st[:n_tr]; val += st[n_tr:]
    _atomic_json(DS0021 / "splits.json", {
        "unit": "trajectory id = manifest state_idx (datasets/DS-0021-*/data/<id>/)",
        "method": "per count_group (state_init record): sorted complete states, first round(0.9 n) train, rest val "
                  "(= EXP-0064 GroupedParticleDataset, train_valid_ratio 0.9)",
        "made_by": "python -u -m FlexData.build_cache splits_ds0021",
        "groups": {str(g): [min(v), max(v), len(v)] for g, v in sorted(by_g.items())},
        "splits": {"train": sorted(train), "val": sorted(val), "test": []},
    })
    print(f"splits: DS-0021 train {len(train)} / val {len(val)} trajectories")


def write_splits_ds0022() -> None:
    """DS-0022: test-only, explicit slates (state_idx -> valid action_idx), like DS-0019."""
    recs = [json.loads(l) for l in open(DS0022 / "data" / "manifest.jsonl")]
    slates: dict[int, list] = {}
    for r in recs:
        if r["type"] == "action" and r["valid"]:
            slates.setdefault(r["state_idx"], []).append(r["action_idx"])
    inits = {r["state_idx"]: r for r in recs if r["type"] == "state_init"}
    states = sorted(inits)
    _atomic_json(DS0022 / "splits.json", {
        "unit": "state_idx (one same-state slate per state)", "made_by": "python -u -m FlexData.build_cache splits_ds0022",
        "splits": {"test": states, "train": [], "val": []},
        "count_group": {str(s): int(inits[s]["count_group"]) for s in states},
        "slates": {str(s): sorted(slates.get(s, [])) for s in states},
        "note": "slates = valid action_idx per state BEFORE the loader's own flag filter "
                "(nan/escaped/out_of_grid/null); FlexPileData.flags reports which of these it drops.",
    })
    print(f"splits: DS-0022 {len(states)} slates, {sum(len(v) for v in slates.values())} valid rows")


def write_splits_ds0019() -> None:
    """DS-0019: test-only; records the explicit slate structure
    (state_idx -> sorted valid action_idx list) so slateN pools are explicit."""
    recs = [json.loads(l) for l in open(DS0019 / "manifest.jsonl")]
    slates: dict[int, list] = {}
    for r in recs:
        if r["type"] == "action" and r["valid"]:
            slates.setdefault(r["state_idx"], []).append(r["action_idx"])
    states = sorted(r["state_idx"] for r in recs if r["type"] == "state_init")
    _atomic_json(DS0019 / "splits.json", {
        "unit": "state_idx (one same-state slate per state)", "made_by": "python -u -m FlexData.build_cache splits_ds0019",
        "splits": {"test": states, "train": [], "val": []},
        "slates": {str(s): sorted(slates.get(s, [])) for s in states},
        "note": "slates = valid action_idx per state BEFORE the loader's own flag filter "
                "(nan/escaped/out_of_grid/null); FlexPileData.flags reports which of these it drops.",
    })
    print(f"splits: DS-0019 {len(states)} slates, {sum(len(v) for v in slates.values())} valid rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    cmds = {"ds0020": None, "ds0019": None, "ds0020_v1": None, "splits": write_splits,
            "paths": write_image_paths, "splits_ds0019": write_splits_ds0019,
            "paths_ds0019": write_image_paths_ds0019, "paths_v1": write_image_paths_v1,
            "ds0021": None, "ds0022": None, "splits_ds0021": write_splits_ds0021,
            "splits_ds0022": write_splits_ds0022,
            "paths_ds0021": lambda: write_image_paths(DS0021, DS0021_RAW),
            "paths_ds0022": lambda: write_image_paths_ds0019(DS0022, DS0022 / "data" / "manifest.jsonl")}
    ap.add_argument("which", choices=sorted(cmds))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    builders = {"ds0020": build_ds0020, "ds0019": build_ds0019, "ds0020_v1": build_ds0020_v1,
                "ds0021": lambda w: build_ds0020(w, DS0021, DS0021_RAW, "ds0021"),
                "ds0022": lambda w: build_ds0019(w, DS0022, DS0022 / "data" / "manifest.jsonl", "ds0022")}
    if a.which in builders:
        builders[a.which](a.workers)
    else:
        cmds[a.which]()


if __name__ == "__main__":
    main()
