"""Baselines/NFD/train_nfd.py -- thin driver: register NFD, then run training.trainer.Trainer.

Mirrors ``training.train``'s CLI (config path, --override, --no-resume) but
imports ``Baselines.NFD.nfd_lib`` FIRST so the ``nfd-genesis-3ch`` dataset
type and ``nfd-unet3ch`` model type are registered in the shared registries
before ``Trainer.from_config`` looks them up. ``training/train.py`` itself is
read-only and knows nothing about this baseline's registrations, so it is not
reused as the entry point -- this script duplicates its handful of lines of
CLI/override glue instead.

Usage
-----
    PYTHONPATH=. python Baselines/NFD/train_nfd.py \\
        Baselines/NFD/configs/nfd_train_3ch.yaml

    PYTHONPATH=. python Baselines/NFD/train_nfd.py \\
        Baselines/NFD/configs/nfd_train_3ch.yaml --override training.epochs=2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import Baselines.NFD.nfd_lib  # noqa: F401  -- registers nfd-genesis-3ch / nfd-unet3ch
import model.warped_nfd.lib  # noqa: F401  -- registers nfd-genesis-3ch-warped / nfd-unet-warped
import model.residual_nfd.lib  # noqa: F401  -- EXP-0022 R1/R2: registers nfd-unet3ch-residual / nfd-unet-warped-residual
import model.flow_nfd.lib  # noqa: F401  -- EXP-0025: registers nfd-flow-warp


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got: {override!r}")
        key, _, raw_value = override.partition("=")
        value = yaml.safe_load(raw_value)
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return cfg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the NFD (non-FiLM) baseline.")
    ap.add_argument("config", type=Path, help="Path to a Baselines/NFD/configs/*.yaml training config.")
    ap.add_argument("--no-resume", action="store_true",
                     help="Start from scratch even if a checkpoint already exists in log_dir.")
    ap.add_argument("--resume", action="store_true",
                     help="Require and continue from <log_dir>/last_state.pt (training.save_full_state): "
                          "same epoch counter, batch position, LR schedule, optimizer and RNG state.")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                     help="e.g. --override training.epochs=2")
    ap.add_argument("--seed", type=int, default=None,
                     help="seed torch / numpy / random BEFORE the model and loaders are built "
                          "(weight init, augmentation, shuffling). Default: unseeded -- the "
                          "historical behaviour, which made runs irreproducible (TODO H1).")
    args = ap.parse_args(argv)

    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    from training.trainer import Trainer, DEVICE
    print(f"Config: {args.config}")
    print(f"Device: {DEVICE}")

    # Overrides must land BEFORE the resume lookup (it reads output.log_dir and
    # training.save_full_state), so build without resuming, override, then
    # resume. Dataset/model overrides still cannot take effect (already built).
    trainer = Trainer.from_config(args.config, resume=False)
    _apply_overrides(trainer.cfg, args.override or [])
    if args.resume or not args.no_resume:
        trainer._resume_full = True
        trainer._try_resume()
    if args.resume:
        st = Path(trainer.cfg.get("output", {}).get("log_dir", "")) / "last_state.pt"
        if not st.exists():
            raise SystemExit(f"--resume: no full state at {st}")
    trainer.run()


if __name__ == "__main__":
    main()
