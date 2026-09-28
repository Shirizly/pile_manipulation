"""model/retrieval_nfd/train_retrieval_nfd.py -- thin driver, mirrors
`Baselines/NFD/train_nfd.py`: register the retrieval-ref dataset type (and
reuse the existing generic `nfd-unet3ch` model factory) before
`training.trainer.Trainer.from_config` looks them up.

Usage:
    PYTHONPATH=. python -u model/retrieval_nfd/train_retrieval_nfd.py \\
        model/retrieval_nfd/configs/retrieval_nfd_ref.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import Baselines.NFD.nfd_lib  # noqa: F401 -- registers the generic nfd-unet3ch model factory
import model.retrieval_nfd.lib  # noqa: F401 -- registers nfd-genesis-retrieval-ref


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the NFD-with-retrieved-reference model.")
    ap.add_argument("config", type=Path)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    import yaml

    def _apply_overrides(cfg, overrides):
        for o in overrides:
            key, _, raw_value = o.partition("=")
            value = yaml.safe_load(raw_value)
            parts = key.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value
        return cfg

    from training.trainer import Trainer, DEVICE
    print(f"Config: {args.config}")
    print(f"Device: {DEVICE}")
    trainer = Trainer.from_config(args.config, resume=not args.no_resume)
    _apply_overrides(trainer.cfg, args.override or [])
    trainer.run()


if __name__ == "__main__":
    main()
