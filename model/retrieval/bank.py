"""Transition bank for the retrieval model (EXP-0059).

Loads DS-0008 (narrow-domain train chains) + DS-0010 (extra matched-physics
18-22mm rows) into one flat, push-frame-canonicalised, PER-OBJECT transition
bank -- the "database" of the design doc's Section 1
(`D = {(s_i, a_i, s'_i, Delta_i)}`). DS-0009 is the query/test set and is
loaded separately by the eval harness (`experiments/EXP-0059-*/code/
eval_retrieval.py`), never baked into the bank.

Fast-load path per docs/CODEMAP.md's "LOADER TRAP": glob `*_data.pt` and
`torch.load` directly (never `load_randlen_cell`, >10 min/call). Respects the
`valid` flag the way EXP-0053 did: DS-0008's rows carry `valid` (exact 20mm
+-0.1mm, perpendicular within 0.1deg -- see its DATASET.md) and rows failing
it are dropped; DS-0010 has no `valid` column because `extract.py` already
filtered to its 18-22mm/perpendicular/below-z criteria before writing, so its
rows are treated as valid unconditionally.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .frame import world_to_push_frame, push_angle, yaw_from_quat, wrap_angle

REPO = Path(__file__).resolve().parents[2]
DS0008_DIR = REPO / "Genesis/data/narrow_l20_n20/train"
DS0010_DIR = REPO / "Genesis/data/narrow_l20_n20/extra_18_22"

# Same +/-64mm slate workspace every OCC_ADAPTERS model is scored on
# (`simple_mpc.adapters.OCC_BOUNDS`) -- duplicated here as a plain dict
# (not imported) so this package stays independent of `simple_mpc`, which
# pulls in a much heavier import chain (control_utility_test, model cards).
OCC_BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}

DEFAULT_MOVED_THRESHOLD = 0.001  # metres; matches the designer's probe convention

# 2026-09-28 user guidance: design the retrieval SET for this scale (more data
# may be collected, but a subset is chosen to build the actual bank from) --
# `distance.py`'s exhaustive GPU-batched chamfer search is O(bank size), so
# this is a real cost/latency budget, not an arbitrary number.
MAX_RECOMMENDED_BANK_SIZE = 50_000


def _load_dir(pattern: str, has_valid: bool):
    states0, states1, p_starts, p_stops = [], [], [], []
    files = sorted(glob.glob(pattern))
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        n = len(d["states"])
        v = d["valid"] if (has_valid and "valid" in d) else torch.ones(n, dtype=torch.bool)
        states0.append(d["states"][v].float())
        states1.append(d["states_"][v].float())
        p_starts.append(d["p_starts"][v].float())
        p_stops.append(d["p_stops"][v].float())
    if not files:
        raise FileNotFoundError(f"no files matched {pattern}")
    return (torch.cat(states0), torch.cat(states1),
            torch.cat(p_starts), torch.cat(p_stops))


class TransitionBank:
    """Per-object, push-frame-canonicalised transition bank.

    All per-object fields are (T, n, ...); all per-transition fields are
    (T, ...). `n` is fixed at 20 for every source corpus here (n20
    single-layer). Kept as PLAIN TENSORS (not a torch.utils.data.Dataset) so
    a predictor can slice/broadcast against the whole bank in one shot.

    Attributes
    ----------
    uv0, uv1 : (T, n, 2)   cube centres before/after, in EACH transition's
                           OWN push frame (see `frame.world_to_push_frame`).
    yaw0, yaw1 : (T, n)    cube yaws before/after, push-frame (world yaw
                           minus that transition's push heading).
    duv : (T, n, 2)        uv1 - uv0 -- the per-object push-frame displacement.
    dyaw : (T, n)          wrapped yaw1 - yaw0.
    moved : (T, n) bool    duv.norm(dim=-1) > moved_threshold.
    p_starts, p_stops : (T, 3)  world metres, the action that produced row T.
    push_len : (T,)        ||p_stop - p_start|| (xy only).
    source : list[str], len T   provenance tag ("DS-0008" / "DS-0010" / ...).
    """

    def __init__(self, uv0, uv1, yaw0, yaw1, duv, dyaw, moved,
                p_starts, p_stops, push_len, source, moved_threshold,
                in_set=None, legal=None, source_file=None, source_row=None):
        self.uv0, self.uv1 = uv0, uv1
        self.yaw0, self.yaw1 = yaw0, yaw1
        self.duv, self.dyaw = duv, dyaw
        self.moved = moved
        self.p_starts, self.p_stops = p_starts, p_stops
        self.push_len = push_len
        self.source = list(source)
        self.moved_threshold = moved_threshold
        # Curated-bank-only fields (DS-0014, EXP-0059 coordinator follow-up B, 2026-09-28) --
        # None on a bank built the old way (`.build()`/`.from_states()`), so every OTHER consumer
        # of this class (e.g. `model/retrieval_nfd/donors.py`) is completely unaffected.
        self.in_set = in_set              # (T, n) bool: geometric interaction set (interaction.py)
        self.legal = legal                # (T,) bool: NOT flagged by the ISS-010 touchdown audit
        self.source_file = source_file    # list[str] or None
        self.source_row = source_row      # (T,) long or None

    def __len__(self):
        return self.uv0.shape[0]

    @classmethod
    def build(cls, moved_threshold: float = DEFAULT_MOVED_THRESHOLD,
             include: Sequence[str] = ("ds0008", "ds0010")):
        """Load DS-0008 and/or DS-0010 off disk and canonicalise. This is the
        entry point `experiments/EXP-0059-*/code/build_bank.py` calls."""
        chunks = []
        if "ds0008" in include:
            s0, s1, ps, pe = _load_dir(str(DS0008_DIR / "_*_data.pt"), has_valid=True)
            chunks.append((s0, s1, ps, pe, "DS-0008"))
        if "ds0010" in include:
            s0, s1, ps, pe = _load_dir(str(DS0010_DIR / "*_data.pt"), has_valid=False)
            chunks.append((s0, s1, ps, pe, "DS-0010"))
        if not chunks:
            raise ValueError(f"include={include!r} selected nothing")
        states0 = torch.cat([c[0] for c in chunks])
        states1 = torch.cat([c[1] for c in chunks])
        p_starts = torch.cat([c[2] for c in chunks])
        p_stops = torch.cat([c[3] for c in chunks])
        source = sum(([c[4]] * len(c[0]) for c in chunks), [])
        return cls.from_states(states0, states1, p_starts, p_stops, source,
                               moved_threshold=moved_threshold)

    @classmethod
    def from_states(cls, states0: torch.Tensor, states1: torch.Tensor,
                    p_starts: torch.Tensor, p_stops: torch.Tensor,
                    source: Iterable[str] = None,
                    moved_threshold: float = DEFAULT_MOVED_THRESHOLD):
        """Canonicalise already-in-memory (states0, states1, p_starts, p_stops)
        -- the path the eval harness / tests use to build a small bank
        without touching disk."""
        xy0, xy1 = states0[..., :2], states1[..., :2]
        uv0 = world_to_push_frame(xy0, p_starts, p_stops)
        uv1 = world_to_push_frame(xy1, p_starts, p_stops)
        phi = push_angle(p_starts, p_stops)
        yaw0 = wrap_angle(yaw_from_quat(states0[..., 3:7]) - phi.unsqueeze(-1))
        yaw1 = wrap_angle(yaw_from_quat(states1[..., 3:7]) - phi.unsqueeze(-1))
        duv = uv1 - uv0
        dyaw = wrap_angle(yaw1 - yaw0)
        moved = duv.norm(dim=-1) > moved_threshold
        push_len = (p_stops[:, :2] - p_starts[:, :2]).norm(dim=-1)
        if source is None:
            source = ["?"] * len(states0)
        if len(states0) > MAX_RECOMMENDED_BANK_SIZE:
            print(f"[TransitionBank] warning: {len(states0)} transitions exceeds the "
                 f"recommended {MAX_RECOMMENDED_BANK_SIZE}-transition design target "
                 "(distance.py's exhaustive search cost is O(bank size)); consider "
                 "subsetting before building the final retrieval set.")
        return cls(uv0, uv1, yaw0, yaw1, duv, dyaw, moved,
                   p_starts, p_stops, push_len, source, moved_threshold)

    def moved_count_histogram(self) -> dict:
        """{n_moved_cubes: n_transitions_with_that_count} over the whole bank."""
        counts = self.moved.sum(dim=1)
        vals, freq = torch.unique(counts, return_counts=True)
        return {int(v): int(f) for v, f in zip(vals.tolist(), freq.tolist())}

    # --- persistence (rule: "persist every fitted object") -----------------
    def save(self, path: str):
        path = str(path)
        payload = dict(
            uv0=self.uv0, uv1=self.uv1, yaw0=self.yaw0, yaw1=self.yaw1,
            duv=self.duv, dyaw=self.dyaw, moved=self.moved,
            p_starts=self.p_starts, p_stops=self.p_stops, push_len=self.push_len,
            source=self.source, moved_threshold=self.moved_threshold,
        )
        if self.in_set is not None:
            payload.update(in_set=self.in_set, legal=self.legal,
                          source_file=self.source_file, source_row=self.source_row)
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str):
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(d["uv0"], d["uv1"], d["yaw0"], d["yaw1"], d["duv"], d["dyaw"],
                   d["moved"], d["p_starts"], d["p_stops"], d["push_len"],
                   d["source"], d["moved_threshold"],
                   in_set=d.get("in_set"), legal=d.get("legal"),
                   source_file=d.get("source_file"), source_row=d.get("source_row"))

    @classmethod
    def load_curated(cls, path=None, include_illegal: bool = False):
        """Load DS-0014 (`datasets/DS-0014-retrieval-curated-interaction-sets/
        build_curated_bank.py`'s payload) -- the ONLY supported way to build
        the bank going forward for `RetrievalPredictor` (coordinator follow-up
        B, 2026-09-28: retrieval must never load the full, uncurated state,
        and must never train against a tool-on-cube touchdown). By default
        (`include_illegal=False`) drops every row the ISS-010 touchdown audit
        flagged `illegal_0mm` (kept in the file and flagged, never deleted --
        see that dataset's DATASET.md) -- excluded from the BANK, not from
        the stored evidence.
        """
        if path is None:
            path = (REPO / "datasets/DS-0014-retrieval-curated-interaction-sets"
                    / "data/curated_bank.pt")
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        legal = d["legal"]
        keep = torch.ones_like(legal) if include_illegal else legal
        idx = keep.nonzero(as_tuple=True)[0]
        source_file = [d["source_file"][i] for i in idx.tolist()]
        return cls(d["uv0"][idx], d["uv1"][idx], d["yaw0"][idx], d["yaw1"][idx],
                   d["duv"][idx], d["dyaw"][idx], d["moved"][idx],
                   d["p_starts"][idx], d["p_stops"][idx], d["push_len"][idx],
                   [d["source"][i] for i in idx.tolist()], d["moved_threshold"],
                   in_set=d["in_set"][idx], legal=legal[idx],
                   source_file=source_file, source_row=d["source_row"][idx])
