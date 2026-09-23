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
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                     help="e.g. --override training.epochs=2")
    args = ap.parse_args(argv)

    from training.trainer import Trainer, DEVICE
    print(f"Config: {args.config}")
    print(f"Device: {DEVICE}")

    trainer = Trainer.from_config(args.config, resume=not args.no_resume)
    _apply_overrides(trainer.cfg, args.override or [])
    trainer.run()


if __name__ == "__main__":
    main()
