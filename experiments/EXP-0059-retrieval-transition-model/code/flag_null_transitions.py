"""Null-transition flagging (EXP-0059 task 2, 2026-09-28).

A transition is NULL when s' == s in every practical sense: max per-cube xy
displacement < 1 mm -- the SAME threshold the retrieval bank's own `moved`
flag uses (`model/retrieval/bank.py`), so "null" here is directly comparable
to that flag. Max yaw change (degrees) is also recorded per row so the user
can see it, but is NOT part of the null criterion (a push that only rotates a
cube in place without translating it more than 1mm is unusual but the
criterion the user asked for is displacement-only).

Scope (coordinator, 2026-09-28): DS-0008 (train), DS-0009 (test_chains +
test_pools), DS-0011 (val_pools) only -- DS-0010/0012/0013 are out of scope
for this round (fresh recollection covers 0008/0009/0011 instead of a
replacement-and-archive pass on all six).

Writes one `_k_data_nullflag.pt` (or `pools_k_nullflag.pt`) per source file,
next to the existing `_legality.pt` files the tool-placement audit already
wrote (originals untouched), with keys:
  is_null            (N,) bool, max per-cube xy displacement < NULL_MM
  max_disp_mm        (N,) float
  max_dyaw_deg       (N,) float
  source_row         (N,) int64 (row index within this file)

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/flag_null_transitions.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from model.retrieval.frame import yaw_from_quat  # noqa: E402

D = REPO / "Genesis/data/narrow_l20_n20"
RES = REPO / "experiments/EXP-0059-retrieval-transition-model/results/null_transition_audit.json"
NULL_MM = 1.0  # same threshold as the retrieval bank's `moved` flag

DATASETS = [
    ("DS-0008_train",       str(D / "train/_*_data.pt"),       "chains"),
    ("DS-0009_test_chains", str(D / "test_chains/_*_data.pt"), "chains"),
    ("DS-0009_test_pools",  str(D / "test_pools/pools_*.pt"),  "pools"),
    ("DS-0011_val_pools",   str(D / "val_pools/pools_*.pt"),   "pools"),
]


def _exclude(f):
    return "_legality" in f or "_nullflag" in f


def flag_file(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    s0 = d["states"].float()
    s1 = d["states_"].float()
    N = s0.shape[0]

    disp_mm = (s1[:, :, :2] - s0[:, :, :2]).norm(dim=-1) * 1000.0     # (N, n)
    max_disp_mm = disp_mm.max(dim=1).values

    yaw0 = yaw_from_quat(s0[:, :, 3:7])
    yaw1 = yaw_from_quat(s1[:, :, 3:7])
    dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
    max_dyaw_deg = dyaw.abs().max(dim=1).values * 180.0 / np.pi

    is_null = max_disp_mm < NULL_MM

    legality_path = Path(path).with_name(Path(path).stem + "_legality.pt")
    illegal = None
    if legality_path.exists():
        leg = torch.load(legality_path, map_location="cpu", weights_only=False)
        illegal = leg["illegal_0mm"].bool()

    flags = dict(is_null=is_null, max_disp_mm=max_disp_mm,
                max_dyaw_deg=max_dyaw_deg, source_row=torch.arange(N))
    out_path = Path(path).with_name(Path(path).stem + "_nullflag.pt")
    tmp = str(out_path) + ".tmp"
    torch.save(flags, tmp)
    os.replace(tmp, out_path)

    valid = d["valid"].bool().numpy() if "valid" in d else np.ones(N, dtype=bool)
    return dict(N=N, is_null=is_null.numpy(), max_disp_mm=max_disp_mm.numpy(),
               max_dyaw_deg=max_dyaw_deg.numpy(), valid=valid,
               illegal=(illegal.numpy() if illegal is not None else None),
               out_path=str(out_path))


def _rate(mask, denom=None):
    if denom is not None:
        mask = mask[denom]
    n = len(mask)
    return dict(n=int(n), n_true=int(mask.sum()), frac=float(mask.mean()) if n else float("nan"))


def audit_dataset(name, pattern):
    files = sorted(f for f in glob.glob(pattern) if not _exclude(f))
    if not files:
        print(f"  {name}: NO FILES matched {pattern}")
        return None
    all_null, all_valid, all_illegal, all_disp = [], [], [], []
    have_illegal = True
    for f in files:
        r = flag_file(f)
        all_null.append(r["is_null"]); all_valid.append(r["valid"])
        all_disp.append(r["max_disp_mm"])
        if r["illegal"] is None:
            have_illegal = False
        else:
            all_illegal.append(r["illegal"])
        print(f"    wrote {r['out_path']}  N={r['N']}  null={int(r['is_null'].sum())}")
    is_null = np.concatenate(all_null)
    valid = np.concatenate(all_valid)
    disp = np.concatenate(all_disp)
    summary = dict(n_files=len(files),
                  all_rows=_rate(is_null), valid_rows=_rate(is_null, valid),
                  max_disp_mm_percentiles={
                      p: float(np.percentile(disp, p)) for p in (5, 25, 50, 75, 95)})
    if have_illegal:
        illegal = np.concatenate(all_illegal)
        both = is_null & illegal
        summary["illegal_and_null"] = dict(
            n=int(len(both)), n_both=int(both.sum()), frac_of_all=float(both.mean()),
            frac_of_illegal=float(both.sum() / max(1, illegal.sum())),
            frac_of_null=float(both.sum() / max(1, is_null.sum())))
    return summary


def main():
    out = {}
    for name, pattern, kind in DATASETS:
        print(f"flagging nulls: {name} ...")
        s = audit_dataset(name, pattern)
        if s is not None:
            out[name] = s
            print(f"  {name}: null {s['valid_rows']['n_true']}/{s['valid_rows']['n']} "
                 f"valid rows = {s['valid_rows']['frac']:.4f}")
            if "illegal_and_null" in s:
                b = s["illegal_and_null"]
                print(f"    illegal AND null: {b['n_both']}/{b['n']} "
                     f"({b['frac_of_all']:.4f} of all rows)")
    RES.parent.mkdir(parents=True, exist_ok=True)
    RES.write_text(json.dumps(out, indent=1))
    print("wrote", RES)


if __name__ == "__main__":
    main()
