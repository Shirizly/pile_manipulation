"""DS-0014: curated retrieval bank -- DS-0008 + DS-0010, push-frame
canonicalised (`model.retrieval.bank.TransitionBank.from_states`) PLUS the
geometric interaction-set mask (`model.retrieval.interaction.interaction_set`,
tuned tau=12mm/angle_max_deg=60 -- see `interaction.py`'s own docstring and
`experiments/EXP-0059-*/results/interaction_set_tuning.json`) and the ISS-010
touchdown-legality flag (`experiments/EXP-0059-*/code/audit_tool_placement.py`),
plus per-row provenance (source dataset/file/row).

Why a new dataset rather than another `model/retrieval/bank.py` knob: the
point (coordinator follow-up B, 2026-09-28) is that retrieval must never load
the FULL, uncurated state -- so the reduction (interaction set) and the
exclusion (illegal touchdown rows) are baked into a stored artifact ONCE,
not recomputed by every predictor/eval script that touches the bank.

Schema (all torch tensors, T = number of ORIGINAL DS-0008+DS-0010 valid rows,
n = 20 cubes; NOTHING is dropped from this file -- illegal rows are kept and
FLAGGED, per the coordinator's instruction, so `TransitionBank.load_curated`
can filter or a diagnostic can look at them):
  uv0, uv1     (T, n, 2)  push-frame cube centres pre/post push (own frame)
  yaw0, yaw1   (T, n)     push-frame cube yaws pre/post push
  duv, dyaw    (T, n, ..) uv1-uv0 / wrapped yaw1-yaw0
  moved        (T, n) bool   truth: duv.norm(-1) > 1mm
  in_set       (T, n) bool   geometric interaction set (truth-free, computable
                             identically at query time)
  legal        (T,) bool     True iff NOT flagged `illegal_0mm` by the ISS-010
                             audit (tool NOT overlapping any cube at touchdown)
  p_starts, p_stops (T, 3) world metres; push_len (T,)
  source          list[str], "DS-0008"/"DS-0010" per row
  source_file     list[str], the ORIGINAL `_k_data.pt` this row came from
  source_row      (T,) long, this row's index within `source_file` BEFORE any
                  valid-filtering (matches the `_legality.pt` companion files'
                  row order, which covers every row, valid or not)

Usage:
    python -u datasets/DS-0014-retrieval-curated-interaction-sets/build_curated_bank.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model.retrieval.bank import TransitionBank, DS0008_DIR, DS0010_DIR  # noqa: E402
from model.retrieval.interaction import interaction_set  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "curated_bank.pt"


def _load_with_provenance(pattern: str, has_valid: bool, source_tag: str):
    states0, states1, p_starts, p_stops = [], [], [], []
    legal, source_file, source_row = [], [], []
    files = sorted(glob.glob(pattern))
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        n = len(d["states"])
        v = d["valid"] if (has_valid and "valid" in d) else torch.ones(n, dtype=torch.bool)
        legality_path = Path(f).with_name(Path(f).stem + "_legality.pt")
        illegal_0mm = torch.load(legality_path, map_location="cpu", weights_only=False)["illegal_0mm"]
        idx = v.nonzero(as_tuple=True)[0]
        states0.append(d["states"][idx].float()); states1.append(d["states_"][idx].float())
        p_starts.append(d["p_starts"][idx].float()); p_stops.append(d["p_stops"][idx].float())
        legal.append(~illegal_0mm[idx])
        source_file += [os.path.relpath(f, REPO)] * len(idx)
        source_row.append(idx)
    if not files:
        raise FileNotFoundError(pattern)
    n_total = sum(len(x) for x in states0)
    return (torch.cat(states0), torch.cat(states1), torch.cat(p_starts), torch.cat(p_stops),
           torch.cat(legal), source_file, torch.cat(source_row), [source_tag] * n_total)


def main():
    s0a, s1a, psa, pea, lega, sfa, sra, srca = _load_with_provenance(
        str(DS0008_DIR / "_*_data.pt"), has_valid=True, source_tag="DS-0008")
    s0b, s1b, psb, peb, legb, sfb, srb, srcb = _load_with_provenance(
        str(DS0010_DIR / "*_data.pt"), has_valid=False, source_tag="DS-0010")

    states0 = torch.cat([s0a, s0b]); states1 = torch.cat([s1a, s1b])
    p_starts = torch.cat([psa, psb]); p_stops = torch.cat([pea, peb])
    legal = torch.cat([lega, legb])
    source_file = sfa + sfb
    source_row = torch.cat([sra, srb])
    source = srca + srcb
    print(f"loaded {len(states0)} rows (DS-0008 {len(s0a)}, DS-0010 {len(s0b)}); "
         f"legal {int(legal.sum())} ({100 * legal.float().mean():.1f}%)")

    bank = TransitionBank.from_states(states0, states1, p_starts, p_stops, source=source)
    in_set = interaction_set(bank.uv0, bank.push_len)
    print(f"interaction set: mean {in_set.sum(1).float().mean():.2f} / median "
         f"{in_set.sum(1).float().median():.1f} / max {int(in_set.sum(1).max())} cubes per row")

    payload = dict(
        uv0=bank.uv0, uv1=bank.uv1, yaw0=bank.yaw0, yaw1=bank.yaw1,
        duv=bank.duv, dyaw=bank.dyaw, moved=bank.moved,
        in_set=in_set, legal=legal,
        p_starts=bank.p_starts, p_stops=bank.p_stops, push_len=bank.push_len,
        source=bank.source, source_file=source_file, source_row=source_row,
        moved_threshold=bank.moved_threshold,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(OUT) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, OUT)
    print("saved", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
