"""Build a curated retrieval bank from the FRESH, ISS-010-fixed narrow_l20_n20
train corpus (EXP-0059 clean-data re-collection, 2026-09-28), replacing
DS-0008+DS-0010 as the retrieval bank's source data.

Unlike `datasets/DS-0014-*/build_curated_bank.py` (hardcoded to DS-0008/
DS-0010's directories and their older flagging conventions), this script
takes an arbitrary chain-shaped data directory and applies the SAME
exclusion criteria `Genesis/training/dataset.py::PileSweepData`'s
`exclude_flagged=True` and `eval_extended.py`'s `--exclude-flagged` use:
  * `gap_out_of_window` (sampler's own last-resort-accept flag)
  * `valid == False` (push length/perpendicularity redraw exhausted)
  * illegal touchdown, ONLY if a `<file>_legality.pt` sidecar exists next to
    a chunk (the ISS-010-fix sampler redraws until legal by construction, so
    a freshly-collected v2 corpus needs no such sidecar to already be clean;
    this is a defensive check in case a later full-scale audit adds one)
  * null transition (max per-cube xy displacement < 1mm, matching the bank's
    own `moved` flag threshold)

After filtering, every remaining row is legal by definition, so the saved
`legal` field is all-True (this is what `TransitionBank.load_curated`
expects: `legal` means "eligible", not "raw-illegal-flag-negated" -- the
exclusion already happened upstream, at load time here, rather than at
bank-load time as DS-0014 does it).

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/build_bank_v2.py \
        --data-dir Genesis/data/narrow_l20_n20/train_v2 \
        --out experiments/EXP-0059-retrieval-transition-model/artifacts/bank_train_v2_curated.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.interaction import interaction_set  # noqa: E402
from eval_extended import _filter_flagged  # noqa: E402  -- shared exclusion logic

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _load_dir_filtered(pattern: str):
    states0, states1, p_starts, p_stops = [], [], [], []
    source_file, source_row = [], []
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    n_before = n_after = 0
    for f in files:
        d_raw = torch.load(f, map_location="cpu", weights_only=False)
        n_before += d_raw["states"].shape[0]
        d = _filter_flagged(d_raw, f)
        n_after += d["states"].shape[0]
        states0.append(d["states"].float()); states1.append(d["states_"].float())
        p_starts.append(d["p_starts"].float()); p_stops.append(d["p_stops"].float())
        rel = os.path.relpath(f, REPO)
        source_file += [rel] * d["states"].shape[0]
        # source_row: index within the ORIGINAL (unfiltered) file, recovered
        # from which absolute rows survived `_filter_flagged`'s keep-mask --
        # `_filter_flagged` doesn't expose the mask directly, so recompute it
        # the same way it does (cheap, same formula) purely to get indices.
        n = d_raw["states"].shape[0]
        bad = torch.zeros(n, dtype=torch.bool)
        if "gap_out_of_window" in d_raw:
            bad |= d_raw["gap_out_of_window"].bool()
        if "valid" in d_raw:
            bad |= ~d_raw["valid"].bool()
        legality_path = Path(f).with_name(Path(f).stem + "_legality.pt")
        if legality_path.exists():
            leg = torch.load(legality_path, map_location="cpu", weights_only=False)
            bad |= leg["illegal_0mm"].bool()
        disp_mm = (d_raw["states_"][:, :, :2] - d_raw["states"][:, :, :2]).float().norm(dim=-1).max(dim=1).values * 1000.0
        bad |= disp_mm < 1.0
        idx = (~bad).nonzero(as_tuple=True)[0]
        source_row.append(idx)
    print(f"loaded {n_after}/{n_before} rows after exclusion "
         f"({100 * n_after / max(1, n_before):.1f}% kept) from {len(files)} files")
    return (torch.cat(states0), torch.cat(states1), torch.cat(p_starts), torch.cat(p_stops),
           source_file, torch.cat(source_row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="Genesis/data/narrow_l20_n20/train_v2",
                    help="chain-shaped directory (relative to repo root or absolute), "
                         "matching `_*_data.pt` files with paired `_*_config.yaml`")
    ap.add_argument("--source-tag", default="DS-train_v2")
    ap.add_argument("--out", default=str(ARTIFACTS / "bank_train_v2_curated.pt"))
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO / data_dir
    states0, states1, p_starts, p_stops, source_file, source_row = _load_dir_filtered(
        str(data_dir / "_*_data.pt"))
    source = [a.source_tag] * len(states0)

    bank = TransitionBank.from_states(states0, states1, p_starts, p_stops, source=source)
    in_set = interaction_set(bank.uv0, bank.push_len)
    print(f"interaction set: mean {in_set.sum(1).float().mean():.2f} / median "
         f"{in_set.sum(1).float().median():.1f} / max {int(in_set.sum(1).max())} cubes per row")
    legal = torch.ones(len(bank), dtype=torch.bool)  # already excluded upstream

    payload = dict(
        uv0=bank.uv0, uv1=bank.uv1, yaw0=bank.yaw0, yaw1=bank.yaw1,
        duv=bank.duv, dyaw=bank.dyaw, moved=bank.moved,
        in_set=in_set, legal=legal,
        p_starts=bank.p_starts, p_stops=bank.p_stops, push_len=bank.push_len,
        source=bank.source, source_file=source_file, source_row=source_row,
        moved_threshold=bank.moved_threshold,
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out)
    print("saved", out, f"({out.stat().st_size / 1e6:.1f} MB)")

    hist = bank.moved_count_histogram()
    total = len(bank)
    print("moved-cube-count histogram:")
    for k in sorted(hist):
        print(f"  {k:2d} moved: {hist[k]:5d}  ({100 * hist[k] / total:5.1f}%)")


if __name__ == "__main__":
    main()
