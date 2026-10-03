"""Baselines/GNN/flex_train.py -- train the dyn-res-pile-manip GNN architecture
(`model/gnn_dyn.py::PropNetDiffDenModel`, Wang et al. RSS 2023) FROM SCRATCH on the
FleX DS-0020 v2 corpus with a CONSTANT node count, input from the COLOUR IMAGE only
(EXP-0062 RUN-0003).

Source recipe (/home/alon/Code/dyn-res-pile-manip, read-only; cited per item):
  * `train/train_gnn_dyn.py`: Adam(lr 1e-3, betas (0.9, 0.999)), batch 4, n_history 1,
    n_rollout 5 (chained: s_cur <- s_pred), loss = sum over rows/steps of
    F.mse_loss(s_pred[j], s_nxt[j]) / (n_rollout * B) on 3-D node positions (camera
    frame / global_scale 24), StepLR(step 1000 epochs, gamma 0.1), best-val checkpoint;
    nominal n_epoch 2000 (`config/train/gnn_dyn.yaml`).
  * `dataset/dataset_gnn_dyn.py`: nodes = fps_rad of the FIRST frame's depth point cloud at a
    RANDOM density (particle_den ~ U(15, 6500), so the node count varies), recentered; each
    node is tied to its NEAREST simulator particle, and the targets at every rollout step are
    those particles' positions (particle correspondence); the step-0 INPUT is also the
    particles' positions. s_delta from the TRUE tracked positions with pusher half-width
    0.8/24; attrs 0.
What is different here (user rules: model input ALWAYS from the visual information; constant
30 nodes; one seed):
  * Input nodes = `flex_predictor.perceive` on the colour PNG (fg != white -> constant plane
    y = 0.24 -> voxel 0.01 -> FPS `particle_num` from a per-state seeded start -> recenter;
    particle_den = 1/r_fps^2) -- the SAME function and seed (crc32("DS-0020/<traj>/<k>")) the
    test-time predictor uses, so train and test perception are identical. No particle positions
    ever enter the input.
  * Rollout steps j > 0: s_delta from the PREDICTED node positions (the true ones are not
    visually available); particle_den fixed per window (as the source: one den per sample).
  * Targets (`--target`):
      - `chamfer_carry` (FULLY VISUAL): every voxel of the input point cloud is carried by its
        nearest input node's predicted (cumulative) XY displacement -- exactly what the mask
        renderer does -- and compared with the voxel point cloud of the TRUE next frame's
        colour image by a symmetric squared Chamfer distance; + MSE pinning node z to the plane.
      - `particle` (particle-SUPERVISED, visually-input): each input node is tied to the nearest
        simulator particle (table XY) of its state; target = node + that particle's displacement
        (cumulative over the rollout); MSE on 3-D positions (z target = the plane) as the source.
  * batch size / epochs: `--batch` (source 4) and a val-loss plateau stop (patience, minimum
    relative improvement) instead of a fixed 2000 epochs; reported in the RUN record.
Graph construction uses `flex_predictor.predict_one_step_vec` (bit-identical to the source's
`predict_one_step`, without its per-sample Python loop; `check-vec` asserts it).

Subcommands:
    python -u -m Baselines.GNN.flex_train cache [--workers 12]        # node cache (atomic, chunked, manifest)
    python -u -m Baselines.GNN.flex_train check-vec
    python -u -m Baselines.GNN.flex_train train --out DIR --target chamfer_carry [--resume] ...
Interruptible: DIR/last_state.pt (model + optimizer + scheduler + epoch + position in the epoch's
permutation + every RNG + best + history), written atomically after every epoch and every
`--save-min` minutes mid-epoch; `--resume` continues from it (mid-epoch included).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Baselines.GNN.flex_predictor import (GLOBAL_SCALE, MODEL_CFG, perceive, predict_one_step_vec,  # noqa: E402
                                          s_delta_train)
from model.gnn_dyn import PropNetDiffDenModel  # noqa: E402

D20 = REPO / "datasets/DS-0020-training-data-flex-N864"
CACHE_VERSION = 1
PLANE_Y = 0.24


# ------------------------------------------------------------------ io
def _atomic_npz(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _atomic_torch(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ node cache
def cache_dir(n: int) -> Path:
    return D20 / "cache" / f"gnn_nodes_n{n}_plane{PLANE_Y:g}"


def _perceive_traj(args):
    """One trajectory: perceive all 11 states from the colour PNGs (test-time perception)."""
    t, paths, n = args
    import cv2
    out = dict(nodes=[], den=[], fallback=[], vox=[])
    for k, (cpath, _) in enumerate(paths):
        color = cv2.imread(str(D20 / cpath), cv2.IMREAD_COLOR)
        seed = zlib.crc32(f"DS-0020/{t}/{k}".encode())      # == FlexGNNPredictor._state's seed
        st = perceive(color, None, PLANE_Y, n, seed)
        # the voxel cloud perceive() sampled from (recomputed: perceive does not return it)
        from Baselines.GNN.flex_predictor import _voxel_down, image_pointcloud, VOXEL
        _, _, pts, _ = image_pointcloud(color, None, PLANE_Y)
        vox = _voxel_down(pts, VOXEL)[:, :2].astype(np.float32)
        out["nodes"].append(st["nodes"]); out["den"].append(st["den"])
        out["fallback"].append(st["fallback"]); out["vox"].append(vox)
    return t, out


def build_cache(n: int, workers: int, chunk: int = 100):
    """Per kept trajectory (train + val of splits.json): nodes (11, n, 3), den (11,), fallback
    (11,), voxel point cloud xy per state (variable; offsets) + nearest-particle index of every
    node (11, n) into that trajectory's particle array. Chunks of `chunk` trajectories, atomic,
    manifest; complete chunks skipped on rerun."""
    from FlexData.dataset import _TrajManifestSource, load_instance_config
    cfg = load_instance_config(D20 / "config.yaml")
    sp = json.loads((D20 / "splits.json").read_text())["splits"]
    tids = sorted(int(t) for t in sp["train"] + sp["val"])
    paths = json.loads((D20 / "cache/image_paths.json").read_text())["trajectories"]
    src = _TrajManifestSource(cfg, tids)
    pos_of = {int(t): i for i, t in enumerate(src.traj_ids)}
    od = cache_dir(n)
    od.mkdir(parents=True, exist_ok=True)
    man_p = od / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else dict(
        version=CACHE_VERSION, particle_num=n, plane_y=PLANE_Y, chunks={}, created=time.strftime("%F %T"),
        perception="Baselines/GNN/flex_predictor.py::perceive (colour PNG, constant plane), seed crc32('DS-0020/<t>/<k>')")
    chunks = [tids[i:i + chunk] for i in range(0, len(tids), chunk)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for ci, ct in enumerate(chunks):
            f = od / f"chunk_{ci:03d}.npz"
            if str(ci) in man["chunks"] and f.exists():
                continue
            res = dict(ex.map(_perceive_traj, [(t, paths[str(t)], n) for t in ct]))
            nodes = np.stack([np.stack(res[t]["nodes"]) for t in ct]).astype(np.float32)   # (T,11,n,3)
            den = np.array([res[t]["den"] for t in ct], np.float32)
            fb = np.array([res[t]["fallback"] for t in ct], bool)
            vox, voff = [], [0]
            for t in ct:
                for v in res[t]["vox"]:
                    vox.append(v); voff.append(voff[-1] + len(v))
            # nearest particle (table XY) of every node; node table XY = (24 cam_x, -24 cam_y)
            from scipy.spatial import cKDTree
            nidx = np.zeros((len(ct), 11, n), np.int64)
            for j, t in enumerate(ct):
                for k in range(11):
                    p = src.state(pos_of[t], k).astype(np.float32)
                    pt = np.stack([p[:, 0], -p[:, 1]], 1)                       # table frame
                    nd = nodes[j, k, :, :2] * GLOBAL_SCALE
                    q = np.stack([nd[:, 0], -nd[:, 1]], 1)
                    ok = np.isfinite(pt).all(1)
                    tree = cKDTree(pt[ok])
                    nidx[j, k] = np.nonzero(ok)[0][tree.query(q)[1]]
            _atomic_npz(f, traj_ids=np.array(ct), nodes=nodes, den=den, fallback=fb,
                        vox=np.concatenate(vox), vox_off=np.array(voff, np.int64), nearest_particle=nidx)
            man["chunks"][str(ci)] = dict(n_traj=len(ct), first=ct[0], last=ct[-1], n_vox=int(voff[-1]),
                                          fallback_states=int(fb.sum()))
            _atomic_json(man_p, man)
            print(f"[cache] chunk {ci + 1}/{len(chunks)} ({time.time() - t0:.0f}s) fallback {int(fb.sum())}", flush=True)
    man["complete"] = True
    _atomic_json(man_p, man)
    (od / "DONE").write_text(time.strftime("%F %T"))


def load_cache(n: int):
    od = cache_dir(n)
    assert (od / "DONE").exists(), f"node cache incomplete: python -u -m Baselines.GNN.flex_train cache"
    parts = []
    for f in sorted(od.glob("chunk_*.npz")):
        with np.load(f) as z:                      # materialise once (NpzFile re-reads per key access)
            parts.append({k: z[k] for k in z.files})
    tids = np.concatenate([p["traj_ids"] for p in parts])
    nodes = np.concatenate([p["nodes"] for p in parts])
    den = np.concatenate([p["den"] for p in parts])
    fb = np.concatenate([p["fallback"] for p in parts])
    nidx = np.concatenate([p["nearest_particle"] for p in parts])
    vox, cnt = [], []
    for p in parts:
        off, pv = p["vox_off"], p["vox"]
        for i in range(len(off) - 1):
            vox.append(pv[off[i]:off[i + 1]]); cnt.append(off[i + 1] - off[i])
    return dict(traj_ids=tids, nodes=nodes, den=den, fallback=fb, nearest_particle=nidx, vox=vox,
                vox_count=np.array(cnt).reshape(len(tids), 11))


# ------------------------------------------------------------------ training data (GPU-resident)
class Windows:
    """All (trajectory, k0) windows of `R` consecutive kept transitions of one split, GPU-resident."""

    def __init__(self, cache: dict, split: str, R: int, device: str):
        from FlexData.dataset import FlexPileData
        ds = FlexPileData(D20 / "config.yaml", split=split, channels=3, exclude_flagged=True,
                          occ_source="image_mask", verbose=True)
        src = ds.src
        kept = set(int(r) for r in ds._index_map)
        pos = {int(t): i for i, t in enumerate(cache["traj_ids"])}
        n_k = src.actions.shape[1]
        win, acts = [], []
        for ti, t in enumerate(src.traj_ids):
            for k0 in range(n_k - R + 1):
                rows = [ti * n_k + k0 + j for j in range(R)]
                if all(r in kept for r in rows):
                    win.append((pos[int(t)], ti, k0))
        self.n = len(win)
        self.R = R
        w = np.array(win)
        ci, si, k0 = w[:, 0], w[:, 1], w[:, 2]
        N = cache["nodes"].shape[2]
        self.nodes0 = torch.from_numpy(cache["nodes"][ci, k0]).to(device)                 # (W,N,3)
        self.den = torch.from_numpy(cache["den"][ci, k0]).to(device)
        self.actions = torch.from_numpy(np.stack([src.actions[si, k0 + j] for j in range(R)], 1)
                                        .astype(np.float32)).to(device)                    # (W,R,4)
        # particle-correspondence targets: cumulative XY displacement of each node's particle
        disp = np.zeros((self.n, R, N, 2), np.float32)
        for i in range(self.n):
            p0 = src.state(si[i], k0[i]).astype(np.float32)
            idx = cache["nearest_particle"][ci[i], k0[i]]
            for j in range(R):
                p1 = src.state(si[i], k0[i] + j + 1).astype(np.float32)
                d = (p1[idx] - p0[idx])                     # raw flex (dx, dz)
                # table (dX, dY) = (dx, -dz); cam-scaled (x, y) = (X, -Y)/24 = (dx, dz)/24
                disp[i, j] = d / GLOBAL_SCALE
        self.pdisp = torch.from_numpy(disp).to(device)
        # voxel clouds: input (k0) and targets (k0 + j + 1), padded
        need = sorted({(c, k) for c, k in zip(ci, k0)} | {(c, k + j + 1) for c, k in zip(ci, k0) for j in range(R)})
        key = {ck: i for i, ck in enumerate(need)}
        vmax = max(int(cache["vox_count"][c, k]) for c, k in need)
        V = np.zeros((len(need), vmax, 2), np.float32)
        M = np.zeros((len(need), vmax), bool)
        for i, (c, k) in enumerate(need):
            v = cache["vox"][c * 11 + k]
            V[i, :len(v)] = v; M[i, :len(v)] = True
        self.vox = torch.from_numpy(V).to(device)
        self.vmask = torch.from_numpy(M).to(device)
        self.vin = torch.tensor([key[(c, k)] for c, k in zip(ci, k0)], device=device)
        self.vtgt = torch.tensor([[key[(c, k + j + 1)] for j in range(R)] for c, k in zip(ci, k0)], device=device)
        self.traj = torch.from_numpy(src.traj_ids[si].astype(np.int64))
        self.fallback = torch.from_numpy(cache["fallback"][ci, k0])
        print(f"[Windows {split} R={R}] {self.n} windows, vmax {vmax}, fallback inputs {int(self.fallback.sum())}",
              flush=True)


def chamfer_sq(P, Pm, Q, Qm):
    """symmetric squared Chamfer between padded clouds P (B,n,2) / Q (B,m,2) with masks; per-row mean."""
    d = torch.cdist(P, Q).pow(2)
    big = torch.finfo(d.dtype).max / 4
    d = d.masked_fill(~Qm[:, None, :], big).masked_fill(~Pm[:, :, None], big)
    a = (d.min(2).values * Pm).sum(1) / Pm.sum(1).clamp_min(1)
    b = (d.min(1).values * Qm).sum(1) / Qm.sum(1).clamp_min(1)
    return a + b


def _subsample(X, Xm, k):
    """random subset of <= k VALID points per row of a padded cloud (training-time Chamfer estimate)."""
    if k <= 0 or X.shape[1] <= k:
        return X, Xm
    sc = torch.rand(Xm.shape, device=X.device).masked_fill(~Xm, -1.0)
    j = sc.topk(k, dim=1).indices
    return torch.gather(X, 1, j[..., None].expand(-1, -1, X.shape[2])), torch.gather(Xm, 1, j)


def window_loss(model, W: Windows, idx: torch.Tensor, target: str, fast: bool = True, sub: int = 0):
    """Source-style rollout loss for windows `idx`: sum over steps of the per-row loss, / (R * B).
    sub > 0: Chamfer clouds randomly subsampled to <= sub points per row (training speed)."""
    s0 = W.nodes0[idx]
    s_cur = s0
    den = W.den[idx]
    B, N, _ = s0.shape
    a_cur = torch.zeros(B, N, device=s0.device)
    loss = 0.0
    if target == "chamfer_carry":
        V, Vm = _subsample(W.vox[W.vin[idx]], W.vmask[W.vin[idx]], sub)
        assign = torch.cdist(V, s0[..., :2]).argmin(2)                    # nearest input node (fixed)
    for j in range(W.R):
        sd = s_delta_train(s_cur, W.actions[idx, j])
        if fast:
            s_pred = predict_one_step_vec(model, a_cur, s_cur, sd, den)
        else:
            s_pred = model.predict_one_step(a_cur, s_cur, sd, den)
        if target == "particle":
            tgt = s0.clone()
            tgt[..., :2] = s0[..., :2] + W.pdisp[idx, j]
            loss = loss + ((s_pred - tgt) ** 2).mean(dim=(1, 2)).sum()          # = sum_j F.mse_loss per row
        else:
            disp = (s_pred - s0)[..., :2]
            P = V + torch.gather(disp, 1, assign[..., None].expand(-1, -1, 2))
            Q, Qm = _subsample(W.vox[W.vtgt[idx, j]], W.vmask[W.vtgt[idx, j]], sub)
            cd = chamfer_sq(P, Vm, Q, Qm)
            zpin = ((s_pred[..., 2] - s0[..., 2]) ** 2).mean(1)
            loss = loss + (cd + zpin).sum()
        s_cur = s_pred
    return loss / (W.R * B)


# ------------------------------------------------------------------ train
def _rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _set_rng(st):
    random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["torch"])
    if st["cuda"] is not None:
        torch.cuda.set_rng_state_all(st["cuda"])


@torch.no_grad()
def eval_loss(model, W: Windows, target: str, bs: int = 64):
    model.eval()
    tot = 0.0
    for i in range(0, W.n, bs):
        idx = torch.arange(i, min(i + bs, W.n), device=W.nodes0.device)
        tot += float(window_loss(model, W, idx, target)) * len(idx)
    model.train()
    return tot / W.n


def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def P(*x):
        s = " ".join(str(y) for y in x)
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    dev = a.device
    cfg = vars(a).copy()
    state_p = out / "last_state.pt"
    resume = a.resume and state_p.exists()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cache = load_cache(a.particle_num)
    Wt = Windows(cache, "train", a.rollout, dev)
    Wv = Windows(cache, "val", a.rollout, dev)
    del cache
    model = PropNetDiffDenModel(MODEL_CFG, use_gpu=False).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.1)   # source; never fires here
    st = dict(epoch=0, pos=0, perm=None, best=float("inf"), best_epoch=-1, ref=float("inf"), bad=0, hist=[],
              iters=0, wall=0.0)
    if resume:
        ck = torch.load(state_p, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        st = ck["st"]; _set_rng(ck["rng"])
        assert ck["cfg"]["target"] == a.target and ck["cfg"]["rollout"] == a.rollout
        P(f"[resume] epoch {st['epoch']} pos {st['pos']} best {st['best']:.6g}@{st['best_epoch']}")
    else:
        _atomic_json(out / "config.json", dict(cfg, n_train_windows=Wt.n, n_val_windows=Wv.n,
                                               n_params=sum(p.numel() for p in model.parameters()),
                                               model_cfg=MODEL_CFG, started=time.strftime("%F %T")))
        P(f"[train] {cfg} train windows {Wt.n} val {Wv.n}")

    def save(tag=""):
        _atomic_torch(state_p, dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                    st=st, rng=_rng_state(), cfg=cfg))
        P(f"[ckpt] {tag} epoch {st['epoch']} pos {st['pos']}")

    last_save = time.time()
    while st["epoch"] < a.max_epochs:
        ep = st["epoch"]
        t_ep = time.time()
        if st["perm"] is None:
            st["perm"] = torch.randperm(Wt.n).tolist()
            st["pos"] = 0
            st["run_loss"] = 0.0
            st["run_n"] = 0
        perm = st["perm"]
        while st["pos"] < len(perm):
            idx = torch.tensor(perm[st["pos"]:st["pos"] + a.batch], device=dev)
            loss = window_loss(model, Wt, idx, a.target, sub=a.chamfer_sub)
            opt.zero_grad()
            loss.backward()
            opt.step()
            st["pos"] += len(idx); st["iters"] += 1
            st["run_loss"] += float(loss.detach()) * len(idx) if st["iters"] % 20 == 0 else 0.0
            st["run_n"] += len(idx) if st["iters"] % 20 == 0 else 0
            if time.time() - last_save > 60 * a.save_min:
                st["wall"] += time.time() - t_ep; t_ep = time.time()
                save("mid-epoch"); last_save = time.time()
            if a.kill_after_iters and st["iters"] >= a.kill_after_iters:   # resume test hook
                P(f"[kill-test] stopping hard at iter {st['iters']}")
                os._exit(9)
        sched.step()
        tr = st["run_loss"] / max(1, st["run_n"])
        va = eval_loss(model, Wv, a.target)
        st["wall"] += time.time() - t_ep
        improved = va < st["best"]
        if improved:
            st["best"], st["best_epoch"] = va, ep
            _atomic_torch(out / "net_best.pth", model.state_dict())
        if va < st["ref"] * (1 - a.min_rel_delta):
            st["ref"], st["bad"] = va, 0
        else:
            st["bad"] += 1
        st["hist"].append(dict(epoch=ep, train=tr, val=va, best=st["best"], bad=st["bad"],
                               wall=st["wall"], lr=opt.param_groups[0]["lr"]))
        P(f"epoch {ep} train~{tr:.6g} val {va:.6g} best {st['best']:.6g}@{st['best_epoch']} bad {st['bad']} "
          f"({time.time() - t_ep:.1f}s, wall {st['wall'] / 60:.1f} min)")
        st["epoch"] = ep + 1
        st["perm"] = None
        _atomic_json(out / "history.json", st["hist"])
        save("epoch"); last_save = time.time()
        if st["bad"] >= a.patience:
            P(f"[stop] plateau: {a.patience} epochs without a {a.min_rel_delta:.3%} relative val improvement")
            break
        if a.max_wall_min and st["wall"] / 60 > a.max_wall_min:
            P(f"[stop] wall budget {a.max_wall_min} min")
            break
    st["finished"] = time.strftime("%F %T")
    _atomic_json(out / "history.json", st["hist"])
    _atomic_json(out / "final.json", dict(best=st["best"], best_epoch=st["best_epoch"], epochs_run=st["epoch"],
                                          wall_min=st["wall"] / 60, iters=st["iters"]))
    P("[done]", st["best"], st["best_epoch"])


def check_vec(a):
    """predict_one_step_vec == the source predict_one_step (bit-identical) on cached val nodes."""
    cache = load_cache(a.particle_num)
    dev = a.device
    W = Windows(cache, "val", 1, dev)
    torch.manual_seed(0)
    model = PropNetDiffDenModel(MODEL_CFG, use_gpu=False).to(dev).eval()
    worst = 0.0
    with torch.no_grad():
        for i in range(0, min(W.n, 1024), 128):
            idx = torch.arange(i, i + 128, device=dev)
            s = W.nodes0[idx]
            sd = s_delta_train(s, W.actions[idx, 0])
            a0 = torch.zeros(s.shape[:2], device=dev)
            p1 = model.predict_one_step(a0, s, sd, W.den[idx])
            p2 = predict_one_step_vec(model, a0, s, sd, W.den[idx])
            worst = max(worst, float((p1 - p2).abs().max()))
    print("max |source - vec| =", worst, flush=True)
    assert worst == 0.0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--particle-num", type=int, default=30)
    c.add_argument("--workers", type=int, default=12)
    v = sub.add_parser("check-vec"); v.add_argument("--particle-num", type=int, default=30)
    v.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--target", choices=["chamfer_carry", "particle"], required=True)
    t.add_argument("--particle-num", type=int, default=30)
    t.add_argument("--rollout", type=int, default=5)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-epochs", type=int, default=200)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--min-rel-delta", type=float, default=0.005)
    t.add_argument("--max-wall-min", type=float, default=0)
    t.add_argument("--save-min", type=float, default=10)
    t.add_argument("--kill-after-iters", type=int, default=0)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--chamfer-sub", type=int, default=384, help="training-time Chamfer subsample (0 = full clouds; val always full)")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    if a.cmd == "cache":
        build_cache(a.particle_num, a.workers)
    elif a.cmd == "check-vec":
        check_vec(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
