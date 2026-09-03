"""
Genesis/sand_state_library.py — varied sand start states, from states we already have.

Why
---
MPM samples its pile once, at entity creation, so every episode of the first sand
dataset restarted from the *identical* pile. Diversity came only from the five
pushes within an episode. That is good for fitting an operator (many actions from
one state) and bad for state coverage, and it showed up as a hard measurement:
the canonical-frame input states spanned only ~14 dimensions for 90% of their
variance, which caps the rank any operator can need
(docs/sand_manipulation.md §7). A low-rank result on a low-rank dataset says as
much about the data as about granular transport.

The fix does not require new physics. States collected *mid-episode* are already
valid, settled, pushed-around piles — asymmetric, off-centre, spread to varying
degrees, which is exactly the variety missing from the start distribution. So:

    1. draw states from anywhere in an existing dataset EXCEPT the first push of
       an episode (those are all the same untouched pile, which is the thing we
       are trying to get away from),
    2. expand each through the tray's symmetry group — a square tray admits D4,
       so one state yields 8 physically distinct ones for free,
    3. jitter the grains slightly, so repeated draws of the same base state are
       not exact duplicates.

Steps 2 and 3 are why a modest source dataset yields a large library. The
symmetry expansion is exact rather than approximate: a rotated or mirrored
settled pile in a square tray is still a settled pile.

Reuses `box_symmetries` / `apply_symmetry` from `state_library.py` rather than
reimplementing D4 — sand grains carry identity quaternions, which those functions
handle unchanged.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path

import torch

from .state_library import apply_symmetry, box_symmetries


def build_sand_state_library(pattern: str,
                             box_vol=(0.128, 0.128, 0.04),
                             n_states: int | None = None,
                             skip_first_push: int = 8,
                             jitter_frac: float = 0.15,
                             particle_size: float = 0.002,
                             seed: int = 0,
                             device: str = "cpu") -> torch.Tensor:
    """Build a bank of varied sand start states from an existing dataset.

    Parameters
    ----------
    pattern : glob for `*_data.pt` files of a previous sand collection.
    skip_first_push : how many leading transitions of each file to discard. The
        collection writes (push, env) flattened with env fastest, so the first
        ``n_envs`` entries are all push 1 — every one of them the same untouched
        pile. Set this to ``n_envs`` (default 8) to drop exactly those.
    jitter_frac : grain jitter as a fraction of a grain diameter. Small on
        purpose: enough that two draws of the same base state are not identical,
        far too small to invalidate the settle. MPM tolerates this easily, and
        the pile is re-settled after loading anyway.
    n_states : cap on the library size (after symmetry expansion). ``None``
        keeps everything.

    Returns ``(K, N, 3)`` positions on ``device``.
    """
    files = sorted(_glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"no sand data matches {pattern}")

    base = []
    for f in files:
        d = torch.load(f, weights_only=False)
        # states_ is the POST-push, post-settle state: a valid resting pile.
        st = d["states_"][..., :3]
        if st.shape[0] > skip_first_push:
            base.append(st[skip_first_push:])
    if not base:
        raise SystemExit("every file was shorter than skip_first_push")
    base = torch.cat(base).to(device)

    # Explicit CPU device on every generator-backed draw: Genesis sets torch's
    # DEFAULT device to cuda, so `torch.rand(shape, generator=cpu_gen)` raises
    # "Expected a 'cuda' device type for generator but found 'cpu'".
    g = torch.Generator(device="cpu").manual_seed(seed)
    syms = box_symmetries(box_vol)

    # Expand by symmetry. One draw per (state, symmetry) pair would be
    # 8x the source, which is usually more than needed, so states are drawn
    # first and a symmetry assigned to each.
    n_out = n_states if n_states is not None else base.shape[0] * len(syms)
    idx = torch.randint(0, base.shape[0], (n_out,), generator=g, device="cpu")
    sym_idx = torch.randint(0, len(syms), (n_out,), generator=g, device="cpu")

    quat = torch.zeros((1, base.shape[1], 4), device=device)
    quat[..., 0] = 1.0

    out = torch.empty((n_out, base.shape[1], 3), device=device)
    for s_i, (yaw, mirror) in enumerate(syms):
        sel = (sym_idx == s_i).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        pos = base[idx[sel]]
        p, _ = apply_symmetry(pos, quat.expand(pos.shape[0], -1, -1), yaw, mirror)
        out[sel] = p

    jitter = (torch.rand(out.shape, generator=g, device="cpu").to(device) - 0.5) \
        * (jitter_frac * particle_size)
    out[..., :2] += jitter[..., :2]          # xy only; z is set by the settle

    print(f"sand state library: {base.shape[0]} source states from {len(files)} "
          f"files -> {n_out} varied starts "
          f"({len(syms)} symmetries, {1000 * jitter_frac * particle_size:.2f} mm jitter)")
    return out


def save(states: torch.Tensor, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": states.cpu()}, path)
    return path


def load(path: str | Path, device: str = "cpu") -> torch.Tensor:
    return torch.load(Path(path), weights_only=False)["states"].to(device)
