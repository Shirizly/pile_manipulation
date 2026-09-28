"""simple_mpc/gt_bank.py -- append-only bank of simulated push outcomes.

Ground truth in this repo is deterministic for n20 single pushes through the
snapshot-restore path (EXP-0024, invariant
`genesis-snapshot-restore-repeat-determinism`), so every simulated
(start state, action) pair is reusable. This bank stores the FINAL PARTICLE
STATE of each simulated push (pos + quat (N, 7) where the path provides it,
positions (N, 3) for `rollout_candidates`), not a `dv`: any goal / value function can then
be scored later against already-simulated actions without touching Genesis.

Keyed by
  * `sim_path`   -- WHICH execution path produced the outcome. Paths are NOT
                    interchangeable: EXP-0023 re-executed DS-0001 corpus actions
                    through `GenesisOracleEnv.rollout_candidates` (full fidelity)
                    and got r = 0.959 against the corpus's own `dv`, not identity.
                    A lookup never crosses paths. Register new paths in SIM_PATHS.
  * state key    -- sha1 of the start state's float32 bytes (pos + quat, (n, 7)).
  * action       -- exact float32 bytes of the action row as simulated (i.e.
                    AFTER any legality projection -- store what actually ran).
  * `fingerprint` -- a free-form dict (config sha, settle budgets, genesis
                    version, ...) fixed per (sim_path, state) file; an `add`
                    with a different fingerprint raises instead of mixing.

On disk: `<root>/<sim_path>/<state_key>.pt`, one file per start state, written
atomically (tmp + rename). Single writer assumed (one GPU, sequential jobs).
Genesis-free: `evaluate` takes a `simulate` callable, so this module never
imports the simulator. Every tensor it creates is pinned to CPU explicitly --
Genesis sets torch's DEFAULT device to CUDA, which otherwise puts index
tensors on the GPU and breaks CPU indexing.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import torch

SIM_PATHS = {
    "binned_collection": "Genesis/binned_slate_collection.py corpus collection "
                         "(SandboxManipulation.execute_action per env) -- DS-0001",
    "oracle_rollout_full": "GenesisOracleEnv.rollout_candidates(use_rollout_fidelity=False) "
                           "-- stores final POSITIONS only, (N, 3)",
    "oracle_rollout_reduced": "GenesisOracleEnv.rollout_candidates(use_rollout_fidelity=True) "
                              "-- halved settle/clearance budgets, planning only",
}


def state_key(state: torch.Tensor) -> str:
    """sha1 of an (n, 7) start state's float32 bytes."""
    s = state.detach().to("cpu", torch.float32).contiguous()
    assert s.ndim == 2 and s.shape[1] == 7, f"expected (n, 7) pos+quat, got {tuple(s.shape)}"
    return hashlib.sha1(s.numpy().tobytes()).hexdigest()[:20]


def _action_keys(actions: torch.Tensor) -> list[bytes]:
    a = actions.detach().to("cpu", torch.float32).contiguous()
    return [bytes(r.numpy().tobytes()) for r in a]


class GroundTruthBank:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self._cache: dict[tuple[str, str], dict] = {}

    # ── storage ──────────────────────────────────────────────────────────────
    def _path(self, sim_path: str, key: str) -> Path:
        if sim_path not in SIM_PATHS:
            raise KeyError(f"unregistered sim_path {sim_path!r}; known: {sorted(SIM_PATHS)}")
        return self.root / sim_path / f"{key}.pt"

    def _load(self, sim_path: str, key: str) -> dict | None:
        ck = (sim_path, key)
        if ck not in self._cache:
            p = self._path(sim_path, key)
            if not p.exists():
                return None
            rec = torch.load(p, map_location="cpu", weights_only=False)
            rec["_index"] = {k: i for i, k in enumerate(_action_keys(rec["actions"]))}
            self._cache[ck] = rec
        return self._cache[ck]

    def _save(self, sim_path: str, key: str, rec: dict) -> None:
        p = self._path(sim_path, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".pt.tmp")
        torch.save({k: v for k, v in rec.items() if k != "_index"}, tmp)
        os.replace(tmp, p)
        self._cache[(sim_path, key)] = rec

    # ── API ──────────────────────────────────────────────────────────────────
    def add(self, state: torch.Tensor, actions: torch.Tensor, final: torch.Tensor,
            sim_path: str, fingerprint: dict, source: str) -> int:
        """Store `final[i]` as the outcome of `actions[i]` from `state`.
        Rows already present are skipped (their stored outcome is kept).
        `source` names the run that produced them (e.g. "EXP-0023/RUN-0002").
        Returns the number of rows added."""
        key = state_key(state)
        assert actions.shape[0] == final.shape[0]
        rec = self._load(sim_path, key)
        if rec is None:
            rec = dict(state=state.detach().cpu().float().clone(), sim_path=sim_path,
                       fingerprint=dict(fingerprint), sources=[],
                       actions=torch.zeros(0, actions.shape[1], device="cpu"),
                       final=torch.zeros(0, *final.shape[1:], device="cpu"),
                       source_idx=torch.zeros(0, dtype=torch.long, device="cpu"), _index={})
        elif rec["fingerprint"] != dict(fingerprint):
            raise ValueError(f"fingerprint mismatch for {sim_path}/{key}: stored "
                             f"{rec['fingerprint']} vs new {fingerprint} -- refusing to mix")
        if actions.shape[1] != rec["actions"].shape[1] and rec["actions"].shape[0] > 0:
            raise ValueError("action dimensionality differs from stored rows")
        new = [i for i, k in enumerate(_action_keys(actions)) if k not in rec["_index"]]
        if not new:
            return 0
        if source not in rec["sources"]:
            rec["sources"].append(source)
        si = rec["sources"].index(source)
        # .clone(): never store a view -- torch.save writes a view's WHOLE
        # underlying storage (a row of a 20k-row corpus tensor saves all 20k)
        a_new = actions[new].detach().cpu().float().clone()
        f_new = final[new].detach().cpu().float().clone()
        base = rec["actions"].shape[0]
        rec["actions"] = torch.cat([rec["actions"], a_new]) if base else a_new
        rec["final"] = torch.cat([rec["final"], f_new]) if base else f_new
        rec["source_idx"] = torch.cat([rec["source_idx"],
                                       torch.full((len(new),), si, dtype=torch.long, device="cpu")])
        for j, k in enumerate(_action_keys(a_new)):
            rec["_index"][k] = base + j
        self._save(sim_path, key, rec)
        return len(new)

    def lookup(self, state: torch.Tensor, actions: torch.Tensor, sim_path: str):
        """-> (hit (B,) bool, final (B, ...) with NaN rows where missing)."""
        rec = self._load(sim_path, state_key(state))
        B = actions.shape[0]
        hit = torch.zeros(B, dtype=torch.bool, device="cpu")
        if rec is None:
            return hit, None
        final = torch.full((B, *rec["final"].shape[1:]), float("nan"), device="cpu")
        for i, k in enumerate(_action_keys(actions)):
            j = rec["_index"].get(k)
            if j is not None:
                hit[i] = True
                final[i] = rec["final"][j]
        return hit, final

    def evaluate(self, state: torch.Tensor, actions: torch.Tensor, sim_path: str,
                 fingerprint: dict, source: str,
                 simulate: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Final states for every action; only the misses are passed to
        `simulate(actions_miss) -> final_miss`, and their outcomes are stored."""
        hit, final = self.lookup(state, actions, sim_path)
        miss = (~hit).nonzero(as_tuple=True)[0]
        if len(miss):
            f_miss = simulate(actions[miss.to(actions.device)]).detach().cpu().float()
            self.add(state, actions[miss], f_miss, sim_path, fingerprint, source)
            if final is None:
                final = torch.full((actions.shape[0], *f_miss.shape[1:]), float("nan"), device="cpu")
            final[miss] = f_miss
        return final

    def states(self, sim_path: str) -> list[str]:
        d = self.root / sim_path
        return sorted(p.stem for p in d.glob("*.pt")) if d.exists() else []

    def record(self, sim_path: str, key: str) -> dict | None:
        """The stored record for one start state (state, actions, final, ...)."""
        return self._load(sim_path, key)
