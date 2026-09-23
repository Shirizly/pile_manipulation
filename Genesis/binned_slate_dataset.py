"""
Genesis/binned_slate_dataset.py — reading a binned slate corpus.

The on-disk side of ``Genesis/binned_slate_collection.py``: a corpus of
``n_states`` start states x ``n_actions`` action chains x ``n_steps`` pushes,
with each push's length drawn from one of five bins. Genesis-free, so analysis
imports it without a GPU.

    from Genesis.binned_slate_dataset import BinnedSlateCorpus

    c = BinnedSlateCorpus.load("Genesis/data/slates_binned/n20_b5_L20-70mm")
    c.verify()                       # chain continuity + same-state slates
    c.trajectories(slate=0)          # every sequence from state 0
    c.select(slate=0, bin=3)         # that slate's 50-60 mm pushes, any step
    c.select(bin=3)                  # every 50-60 mm push in the corpus
    c.by_bin(step=0)                 # {bin: Transitions} for the first push
    c.select(bin=UNDERFLOW_BIN)      # pushes the tray wall cut below the lowest edge

Why a class rather than a loader function
-----------------------------------------
Every question asked of a slate corpus is a *grouping* question — by start
state, by length bin, by step, or by chain — and the answers have to agree with
each other. Doing that inline is where the ad-hoc versions went wrong before:
`docs/CODEMAP.md` records that the `*_eval` configs' row filtering broke
row-to-env alignment across steps, which is exactly the kind of bug that only
shows up as a silently mismatched pairing later. Here row identity is carried
explicitly (`chain_idx`, `step_idx`, `slate_idx`, `action_idx` travel with every
selection), so a selection can always be joined back or re-grouped, and
``verify`` asserts the two structural invariants the corpus is supposed to have.

This is deliberately NOT a `torch.utils.data.Dataset`. Training loaders for
push data already exist (`Genesis/training/dataset.py::PileSweepData`), but they
split at FILE granularity and know nothing about slates; a slate corpus is an
evaluation artefact whose whole point is the grouping this module exposes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

# Per-row tensors saved by binned_slate_collection.py, plus the three index
# columns this module derives. Both directions (load, select) walk this list, so
# adding a field to the collector means adding it here and nowhere else.
_SAVED_FIELDS = ("states", "states_", "p_starts", "p_stops", "angles",
                 "len_target", "len_realized", "bin_requested", "bin_realized",
                 "slate_idx", "reached_goal")

#: ``bin_realized`` for a push shorter than the lowest edge. Kept as its own
#: label rather than folded into bin 0: these are pushes the tray wall cut short
#: even after every redraw, so they are not samples of the 20-30 mm operator and
#: must not be fitted as though they were. They stay in the corpus because
#: *which* pile configurations produce them is itself information — filtering
#: them out at collection time would bias the corpus toward piles that happen to
#: sit where long pushes fit.
UNDERFLOW_BIN = -1


def realized_bin(length, edges) -> "np.ndarray":
    """Bin index for each realized push length; ``UNDERFLOW_BIN`` below the
    lowest edge.

    Overflow is not a case: a push is drawn at a target inside its bin and can
    only come out shorter (``min(target, wall limit)``), so anything at or above
    the top edge is numerical slack on the top bin and is clamped there.

    This is the single definition of the bin vocabulary — the collector imports
    it, so a corpus is binned on disk by exactly the rule readers apply.
    """
    idx = np.digitize(np.asarray(length), np.asarray(edges)[1:-1])
    idx = np.clip(idx, 0, len(edges) - 2)
    return np.where(np.asarray(length) < edges[0], UNDERFLOW_BIN, idx)


@dataclass(frozen=True)
class Transitions:
    """A flat table of push records, one row per (chain, step) pair.

    Every selection returns one of these, and every row carries the indices it
    came from (``chain_idx``, ``step_idx``, ``slate_idx``, ``action_idx``), so a
    subset is never anonymous: it can be re-grouped, joined against another
    selection, or traced back to its trajectory in the corpus.

    Row order is the corpus's own — chain-major within a step, steps ascending —
    and selection preserves it.
    """

    states: torch.Tensor          # (rows, n_particles, 7) pre-push, xyz + wxyz
    states_: torch.Tensor         # (rows, n_particles, 7) post-push, settled
    p_starts: torch.Tensor        # (R, 3) blade start
    p_stops: torch.Tensor         # (R, 3) blade stop as executed
    angles: torch.Tensor          # (R,) blade yaw, radians
    len_target: torch.Tensor      # (R,) length asked of the sampler, m
    len_realized: torch.Tensor    # (R,) sampled push length, m — what binning used
    bin_requested: torch.Tensor   # (R,) bin the schedule asked for
    bin_realized: torch.Tensor    # (R,) bin len_realized actually fell in
    slate_idx: torch.Tensor       # (R,) which start state
    reached_goal: torch.Tensor    # (R,) execute_action's own success flag
    chain_idx: torch.Tensor       # (R,) flat chain id, slate_idx * n_actions + action_idx
    step_idx: torch.Tensor        # (R,) which push of the chain
    action_idx: torch.Tensor      # (R,) which chain within its slate

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __repr__(self) -> str:
        if not len(self):
            return "Transitions(empty)"
        return (f"Transitions({len(self)} rows, "
                f"{len(set(self.slate_idx.tolist()))} slate(s), "
                f"steps {sorted(set(self.step_idx.tolist()))}, "
                f"bins {sorted(set(self.bin_realized.tolist()))}, "
                f"len {1e3 * float(self.len_realized.min()):.0f}-"
                f"{1e3 * float(self.len_realized.max()):.0f} mm)")

    def __getitem__(self, index) -> "Transitions":
        """Row subset by mask, index array, slice or int. Always a Transitions."""
        if isinstance(index, (int, np.integer)):
            index = [int(index)]
        if isinstance(index, np.ndarray):
            index = torch.from_numpy(index)
        return replace(self, **{f.name: getattr(self, f.name)[index]
                                for f in fields(self)})

    @staticmethod
    def concat(parts) -> "Transitions":
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("nothing to concatenate")
        return replace(parts[0], **{
            f.name: torch.cat([getattr(p, f.name) for p in parts])
            for f in fields(parts[0])})

    def group_by(self, field: str) -> dict[int, "Transitions"]:
        """Split into ``{value: rows}`` on one index column, keys ascending.

        Only the columns that name a group are sensible here — ``bin_realized``,
        ``bin_requested``, ``slate_idx``, ``step_idx``, ``action_idx``,
        ``chain_idx``. Empty groups are absent rather than empty, so
        ``len(d)`` is the number of groups that actually have data.
        """
        column = getattr(self, field)
        if column.dtype.is_floating_point:
            raise ValueError(f"{field} is continuous; group by a bin or index column")
        return {int(v): self[column == v] for v in column.unique(sorted=True)}

    def push_vectors(self) -> torch.Tensor:
        """(R, 2) planar start→stop displacement, as executed."""
        return self.p_stops[:, :2] - self.p_starts[:, :2]


class BinnedSlateCorpus:
    """A directory written by ``Genesis/binned_slate_collection.py``.

    Holds one ``Transitions`` table per step, all of them in the same row order
    (row ``c`` is chain ``c`` in every step file), which is what makes
    trajectory reconstruction a stack rather than a join.
    """

    def __init__(self, manifest: dict, steps: list[Transitions], path: Path | None = None):
        self.manifest = manifest
        self.path = path
        self._steps = steps

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "BinnedSlateCorpus":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        n_actions = int(manifest["n_actions_per_state"])
        steps = []
        for k in range(int(manifest["n_steps"])):
            blob = torch.load(path / f"step{k}.pt", weights_only=False)
            chain = torch.arange(blob["slate_idx"].shape[0])
            steps.append(Transitions(
                **{f: blob[f] for f in _SAVED_FIELDS},
                chain_idx=chain,
                step_idx=torch.full_like(chain, k),
                action_idx=chain % n_actions))
        return cls(manifest, steps, path)

    # -- shape ------------------------------------------------------------

    @property
    def n_slates(self) -> int:
        return int(self.manifest["n_states"])

    @property
    def n_actions(self) -> int:
        """X — chains per start state."""
        return int(self.manifest["n_actions_per_state"])

    @property
    def n_steps(self) -> int:
        return int(self.manifest["n_steps"])

    @property
    def n_chains(self) -> int:
        return self.n_slates * self.n_actions

    @property
    def bin_edges(self) -> np.ndarray:
        return np.asarray(self.manifest["bin_edges"], dtype=float)

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges) - 1

    def bin_range_m(self, b: int) -> tuple[float, float]:
        """Metre range of bin ``b``; ``UNDERFLOW_BIN`` spans 0 to the lowest edge."""
        if b == UNDERFLOW_BIN:
            return 0.0, float(self.bin_edges[0])
        return float(self.bin_edges[b]), float(self.bin_edges[b + 1])

    def bin_label(self, b: int) -> str:
        lo, hi = self.bin_range_m(b)
        return (f"underflow (<{hi * 1e3:.0f} mm)" if b == UNDERFLOW_BIN
                else f"{lo * 1e3:.0f}-{hi * 1e3:.0f} mm")

    @property
    def spawn_mode(self) -> str:
        """The raw simulator spawn mode: ``heap``, ``pyramid`` or ``drop``."""
        return str(self.manifest["spawn_mode"])

    @property
    def spawn_style(self) -> str:
        """``"pile"`` or ``"scatter"`` — the vocabulary docs and reports use.

        ``heap`` and ``pyramid`` place particles in multiple layers (a *pile*);
        ``drop`` cannot, because the cubes bounce outward on landing and 90-94 %
        settle in layer 0 (a *scatter*). The distinction changes what a push
        does, so it belongs in any comparison between corpora.
        """
        return "scatter" if self.spawn_mode == "drop" else "pile"

    def chain_id(self, slate: int, action: int) -> int:
        return slate * self.n_actions + action

    # -- selection --------------------------------------------------------

    def step(self, k: int) -> Transitions:
        """All ``n_chains`` pushes taken at step ``k``, in chain order."""
        return self._steps[k]

    def transitions(self) -> Transitions:
        """Every push in the corpus, steps ascending."""
        return Transitions.concat(self._steps)

    def select(self, *, slate=None, action=None, chain=None, step=None,
               bin=None, requested_bin=None, reached=None) -> Transitions:
        """Rows matching every constraint given. Each accepts an int or a sequence.

        ``bin`` accepts ``UNDERFLOW_BIN`` (-1) for pushes shorter than the
        lowest edge; those are excluded from every real bin, so summing the bins
        does not give the corpus unless you add it back.

        ``bin`` filters on the length actually achieved (``bin_realized``) —
        the honest one, and the one a length-conditioned operator must be fitted
        and scored on. ``requested_bin`` filters on what the schedule asked for,
        which is only what you want when auditing the sampler itself: the two
        differ wherever the tray wall cut a push short (see
        ``binned_slate_collection``'s retry policy).
        """
        rows = self.transitions() if step is None else \
            Transitions.concat([self._steps[k] for k in _as_list(step)])
        for column, wanted in (("slate_idx", slate), ("action_idx", action),
                               ("chain_idx", chain), ("bin_realized", bin),
                               ("bin_requested", requested_bin)):
            if wanted is not None:
                col = getattr(rows, column)
                # device/dtype pinned to the column: Genesis sets torch's
                # default device to cuda, so a bare as_tensor here would build
                # a cuda index for a cpu column once this is imported
                # alongside the simulator.
                keep = torch.isin(col, torch.as_tensor(_as_list(wanted),
                                                       dtype=col.dtype,
                                                       device=col.device))
                rows = rows[keep]
        if reached is not None:
            rows = rows[rows.reached_goal.bool() == bool(reached)]
        return rows

    def slate(self, i: int) -> Transitions:
        """Every push of every chain that started from state ``i``."""
        return self.select(slate=i)

    def by_bin(self, **filters) -> dict[int, Transitions]:
        """``{realized bin: rows}`` — 'all the sweeps of each bin', optionally
        including ``UNDERFLOW_BIN`` (-1, first key) when the corpus has any,
        narrowed by any ``select`` keyword (e.g. ``by_bin(slate=0)`` for one
        start state, ``by_bin(step=0)`` for the first push only)."""
        return self.select(**filters).group_by("bin_realized")

    def by_slate(self, **filters) -> dict[int, Transitions]:
        return self.select(**filters).group_by("slate_idx")

    # -- trajectory reconstruction ----------------------------------------

    def trajectory(self, chain: int) -> torch.Tensor:
        """``(n_steps + 1, n_particles, 7)`` — the states one chain passed through.

        Element ``k`` is the state *before* push ``k``; the last element is the
        state after the final push. So ``trajectory(c)[j]`` is "the state after
        the first ``j`` pushes of chain ``c``".
        """
        return torch.cat([s.states[chain:chain + 1] for s in self._steps]
                         + [self._steps[-1].states_[chain:chain + 1]])

    def trajectories(self, slate: int | None = None) -> torch.Tensor:
        """``(n_chains, n_steps + 1, n_particles, 7)``, or one slate's
        ``(n_actions, n_steps + 1, n_particles, 7)``.

        'All the sequences for a given initial state' — every chain of a slate
        shares element 0 exactly (that is the same-state property ``verify``
        checks) and diverges from there.
        """
        rows = slice(None) if slate is None else \
            slice(slate * self.n_actions, (slate + 1) * self.n_actions)
        return torch.stack([s.states[rows] for s in self._steps]
                           + [self._steps[-1].states_[rows]], dim=1)

    def state_after(self, chain: int, n_pushes: int) -> torch.Tensor:
        """``(n_particles, 7)`` — that chain's state after its first ``n_pushes``."""
        return self.trajectory(chain)[n_pushes]

    def initial_states(self) -> torch.Tensor:
        """``(n_slates, n_particles, 7)`` — the state each slate's chains share."""
        return self._steps[0].states[::self.n_actions]

    def final_states(self) -> torch.Tensor:
        """``(n_chains, n_particles, 7)`` — where each chain ended."""
        return self._steps[-1].states_

    def bin_sequence(self, chain: int) -> list[int]:
        """The realized length bin of each of a chain's pushes, in order."""
        return [int(s.bin_realized[chain]) for s in self._steps]

    # -- integrity --------------------------------------------------------

    def verify(self, verbose: bool = True) -> dict:
        """Assert the two structural invariants, and report the sampler's drift.

        1. **Chain continuity** — step ``k+1``'s pre-state is step ``k``'s
           post-state, bit for bit. If this fails the row order diverged between
           step files and every trajectory is a splice of different chains.
        2. **Same-state slates** — all ``X`` chains of a slate share one start
           state exactly. This is what makes the slate a fair comparison of
           actions rather than of starting piles.

        Also returns (does not assert) requested-vs-realized bin agreement:
        pushes cut short by the tray wall are expected and are labelled by the
        bin they reached, so a gap here is information, not a failure.
        """
        for k in range(self.n_steps - 1):
            if not torch.equal(self._steps[k].states_, self._steps[k + 1].states):
                raise AssertionError(
                    f"chain continuity broken between step {k} and {k + 1}: "
                    f"row order differs, so trajectories would splice chains")

        starts = self._steps[0].states.reshape(self.n_slates, self.n_actions, -1)
        spread = float((starts.amax(1) - starts.amin(1)).abs().max())
        if spread > 1e-9:
            raise AssertionError(
                f"slate start states differ by up to {spread:.3e} m within a "
                f"slate; candidates are not comparable")

        rows = self.transitions()
        report = {
            "n_chains": self.n_chains, "n_steps": self.n_steps,
            "n_pushes": len(rows),
            "worst_slate_spread_m": spread,
            "bin_counts_realized": [
                int((rows.bin_realized == b).sum()) for b in range(self.n_bins)],
            "n_underflow": int((rows.bin_realized == UNDERFLOW_BIN).sum()),
            "bin_counts_requested": np.bincount(rows.bin_requested.numpy(),
                                                minlength=self.n_bins).tolist(),
            "frac_in_requested_bin": float(
                (rows.bin_realized == rows.bin_requested).float().mean()),
            "len_realized_mm": [float(1e3 * rows.len_realized.min()),
                                float(1e3 * rows.len_realized.mean()),
                                float(1e3 * rows.len_realized.max())],
            "frac_state_unchanged": float(
                (rows.states == rows.states_).all(-1).all(-1).float().mean()),
        }
        if verbose:
            print(f"{self.path}: {self.n_slates} slates x {self.n_actions} chains "
                  f"x {self.n_steps} steps = {report['n_pushes']} pushes")
            print(f"  chain continuity OK; slate start spread "
                  f"{spread:.1e} m (must be 0)")
            print(f"  bins requested {report['bin_counts_requested']} -> "
                  f"realized {report['bin_counts_realized']} "
                  f"({100 * report['frac_in_requested_bin']:.0f}% on target), "
                  f"{report['n_underflow']} underflow "
                  f"(<{self.bin_edges[0] * 1e3:.0f} mm, bin {UNDERFLOW_BIN})")
            print(f"  push length {report['len_realized_mm'][0]:.1f}/"
                  f"{report['len_realized_mm'][1]:.1f}/"
                  f"{report['len_realized_mm'][2]:.1f} mm (min/mean/max); "
                  f"{100 * report['frac_state_unchanged']:.1f}% no-op pushes")
        return report


def _as_list(value):
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(v) for v in value]


if __name__ == "__main__":
    import sys
    BinnedSlateCorpus.load(sys.argv[1]).verify()
