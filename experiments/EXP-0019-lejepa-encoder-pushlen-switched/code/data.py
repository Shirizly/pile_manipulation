"""Corpus -> padded particle-state tensors, fast path only.

CODEMAP LOADER TRAP: `load_randlen_cell()` takes >10 min per call.  Glob
`*_data.pt` and `torch.load` directly; exclude `*_failed.pt` and `.zip`.
"""
from __future__ import annotations
import glob, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]


def randlen_files(split):
    fs = sorted(glob.glob(str(REPO / f"Genesis/data/overnight_randlen_{split}/*/*_data.pt")))
    return [f for f in fs if not f.endswith("_failed.pt")]


def sean_files():
    fs = sorted(glob.glob(str(REPO / "Genesis/data/Sean/**/*_data.pt"), recursive=True))
    return [f for f in fs if not f.endswith("_failed.pt")]


def pad_to(x, N):
    """(B,n,2) -> (B,N,2), padded by REPEATING existing particles.  The splat
    is a boolean 'any', so duplicate grains are exactly a no-op; the real count
    is returned separately so dropout never sees the padding."""
    B, n, _ = x.shape
    if n == N:
        return x
    idx = torch.arange(N) % n
    return x[:, idx]


def load_states(files, max_n=None, states_key=("states", "states_"), verbose=True):
    """Returns (pts (M,N,2) float32 cpu, n_real (M,) long)."""
    chunks, counts = [], []
    t0 = time.time()
    Ns = []
    raw = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        for k in states_key:
            s = d[k][..., :2].float()
            raw.append(s); Ns.append(s.shape[1])
    N = max(Ns)
    for s in raw:
        counts.append(torch.full((s.shape[0],), s.shape[1], dtype=torch.long))
        chunks.append(pad_to(s, N))
    pts = torch.cat(chunks); nr = torch.cat(counts)
    if verbose:
        print(f"load_states: {len(files)} files -> {tuple(pts.shape)} "
              f"(N_pad={N}, counts {sorted(set(nr.tolist()))}) {time.time()-t0:.1f}s", flush=True)
    return pts, nr


def load_transitions(files, verbose=True):
    """Returns dict with pts0, pts1 (M,N,2), n_real, p_start(M,2), p_stop(M,2), angle(M,)."""
    P0, P1, NR, PS, PE, AN = [], [], [], [], [], []
    Ns = []
    raw = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        s0 = d["states"][..., :2].float(); s1 = d["states_"][..., :2].float()
        raw.append((s0, s1, d["p_starts"][:, :2].float(), d["p_stops"][:, :2].float(),
                    d["angles"].float()))
        Ns.append(s0.shape[1])
    N = max(Ns)
    for s0, s1, ps, pe, an in raw:
        P0.append(pad_to(s0, N)); P1.append(pad_to(s1, N))
        NR.append(torch.full((s0.shape[0],), s0.shape[1], dtype=torch.long))
        PS.append(ps); PE.append(pe); AN.append(an)
    out = dict(pts0=torch.cat(P0), pts1=torch.cat(P1), n_real=torch.cat(NR),
               p_start=torch.cat(PS), p_stop=torch.cat(PE), angle=torch.cat(AN))
    if verbose:
        print(f"load_transitions: {len(files)} files -> {out['pts0'].shape[0]} rows", flush=True)
    return out
