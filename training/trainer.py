"""
Model-agnostic training loop.

The Trainer is responsible for:
    dataset/loader construction, model instantiation, optimizer + scheduler
    + AMP GradScaler, augmentation dispatch, TensorBoard logging, early
    stopping, checkpoint saving, and model-card writing.

The Trainer is NOT responsible for:
    loss arithmetic, metric computation, model architecture — those are
    delegated to ``loss_fn``, ``metrics``, and ``model_wrapper``.

Config file path references
---------------------------
If a top-level key's value is a string ending in ``.yaml``, the referenced
file is loaded and its content replaces the string.  Paths are resolved
against the project root first, then against the config file's directory::

    model:   configs/model/unetfilm.yaml    ← loaded and inlined
    dataset: configs/dataset/genesis_cube.yaml

This allows composable configs without YAML extension libraries.

Checkpoint format
-----------------
Each checkpoint is a plain ``state_dict`` saved with ``torch.save``.
A ``model_card.yaml`` sidecar is always written alongside the best
checkpoint.  Weight checkpoints are written atomically (``.tmp`` +
``os.replace``).

Full resumable state (``training.save_full_state: true``, opt-in; added
2026-10-01 for EXP-0061): ``<log_dir>/last_state.pt`` holds model +
optimizer + scheduler + GradScaler + epoch + batches-done-in-epoch + the
partial epoch sums + best-val bookkeeping + RNG states + a fingerprint of
the train/val rows. It is rewritten atomically at the end of every epoch
(together with ``unet_last.pth``) and mid-epoch every
``training.full_state_every_min`` minutes (default 10). The train loader
then draws its order from ``_EpochPermSampler`` (permutation seeded by
``training.shuffle_seed`` + epoch, independent of the global RNG), so a
resume continues the SAME epoch from the SAME batch with the same LR
schedule. On resume the stored row fingerprints must match (same split) or
the run refuses to continue. ``resume=False`` (``--no-resume``) ignores it.

Plateau stop (``training.patience_min_rel_delta``, opt-in, default 0 = the
old behaviour; added 2026-10-02 for EXP-0062): an epoch only resets the
early-stopping counter if its val loss is below ``(1 - delta) x`` the val
loss at the last such reset (``plateau_ref``), so training stops after
``patience`` epochs without a > delta relative improvement. ``unet_best.pth``
is still written on ANY new best, independent of the counter.

Deduplication
-------------
``_get_log_dir()`` returns ``output.log_dir`` unchanged unless a completed
run (one that has written ``unet.pth``) already exists there — in that case
it appends ``_2``, ``_3``, etc.  An in-progress run (checkpoint exists but
no ``unet.pth``) is always resumed in-place.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from registry.model_registry import build_model, ModelTrainingWrapper
from registry.dataset_registry import build_dataset
from training.losses import build_loss
from training.metrics import build_metrics
from model.model_card import ModelCard
from training.types import TrainingBatch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """
    Load a YAML config file and resolve any path-referenced sub-configs.

    Parameters
    ----------
    path : str | Path — path to a training config YAML.

    Returns
    -------
    dict — fully resolved config (sub-configs inlined under their keys).
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    return _resolve_path_includes(cfg, path.parent)


def _resolve_path_includes(cfg: dict, base_dir: Path) -> dict:
    """
    Replace any top-level string value ending in ``.yaml`` with the loaded
    sub-config.  Paths are resolved first relative to the working directory
    (project root), then relative to the config file's directory as a fallback.
    Only one level of resolution is applied.
    """
    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, str) and value.endswith(".yaml"):
            # Try project root first (most natural for configs/ paths)
            from_root = Path(value)
            from_dir  = base_dir / value
            if from_root.exists():
                sub = from_root
            elif from_dir.exists():
                sub = from_dir
            else:
                raise FileNotFoundError(
                    f"Sub-config not found: tried '{from_root}' and '{from_dir}'"
                )
            resolved[key] = yaml.safe_load(sub.read_text())
        else:
            resolved[key] = value
    return resolved


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

def _to_device(batch: TrainingBatch, device: str) -> TrainingBatch:
    """Move all tensor values in a batch dict to ``device``."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _augment_push_endpoints(push_px: torch.Tensor, n: int, k: int,
                            flipped: bool) -> torch.Tensor:
    """Map push endpoints through the SAME rot90(k, dims=(-2,-1)) [+ hflip on
    dims=[-1]] pixel transform ``_augment_eulerian_batch`` applies to the
    image tensors, so the augmented occupancy/target and the augmented
    ``push_px`` stay geometrically consistent.

    ``push_px`` is (B, 4) = ``[start_col, start_row, end_col, end_row]`` in
    pixels, (col, row) order (see docs/INTERFACES.md). The square grid side
    is ``n``. The per-point map below is derived and empirically verified in
    ``model/warped_nfd/WARPED_NFD_NOTES.md``; it is an exact affine reflection
    (not a rounding/nearest-pixel approximation), so it is exact for
    sub-pixel (float) endpoint coordinates too, not just integer pixel
    centers.
    """
    sc, sr, ec, er = push_px[:, 0], push_px[:, 1], push_px[:, 2], push_px[:, 3]

    def _map(c, r):
        if k == 0:
            co, ro = c, r
        elif k == 1:
            co, ro = r, n - 1 - c
        elif k == 2:
            co, ro = n - 1 - c, n - 1 - r
        elif k == 3:
            co, ro = n - 1 - r, c
        else:
            raise ValueError(k)
        if flipped:
            co = n - 1 - co
        return co, ro

    sco, sro = _map(sc, sr)
    eco, ero = _map(ec, er)
    return torch.stack([sco, sro, eco, ero], dim=-1)


AUGMENT_FACTOR = {"full": 8, "flip": 2}


def _augment_eulerian_batch(batch: TrainingBatch,
                            mode: str = "full") -> TrainingBatch:
    """
    Spatial augmentation for Eulerian batches.

    ``mode="full"`` (the default, and what ``augmentation: true`` selects) is
    the ×8 group: 4 rotations × 2 horizontal flips.

    ``mode="flip"`` is the ×2 subgroup: identity and one horizontal flip, no
    rotations. This exists for models that are ROTATION-INVARIANT BY
    CONSTRUCTION, for which the 4 rotations are not augmentation at all. A
    push-frame model is the case in point: it warps its input into a frame
    defined by the push direction, so all 4 rotated views of a sample map to
    the SAME canonical input, and the ×8 group yields only 2 distinct inputs.
    Feeding it the full group spends 4× the compute per gradient step on
    exact duplicates, which is why ``mode="flip"`` is the correct setting
    there rather than a weaker one — see
    ``experiments/EXP-0022-warped-nfd-push-frame/PLAN.md``.

    Operates on:
        "input":   Tensor[B, C, H, W]  — spatial dims are the last two
        "target":  Tensor[B, H, W]
        "physics": Tensor[B, P]        — replicated ×8, not spatially modified
        "push_px": Tensor[B, 4]        — OPTIONAL, [start_col, start_row,
                   end_col, end_row] pixels; carried through the SAME
                   rot90/flip transform as the images (see
                   ``_augment_push_endpoints``) so a push-frame model's
                   endpoints stay consistent with the augmented occupancy.
                   If absent, behaviour is byte-identical to before this key
                   existed.

    Returns a new batch dict with batch dimension ×``AUGMENT_FACTOR[mode]``.

    Note: this augmentation assumes spatial symmetry of the occupancy grid.
    It is only appropriate for Eulerian (grid-based) representations.
    For GNN/Lagrangian datasets, set ``augmentation: false`` in the training
    config.
    """
    x       = batch["input"]    # (B, C, H, W)
    targets = batch["target"]   # (B, H, W)
    push_px = batch.get("push_px")   # optional (B, 4)
    if push_px is not None:
        H, W = x.shape[-2], x.shape[-1]
        if H != W:
            raise ValueError(
                "push_px augmentation requires a square grid, got "
                f"(H, W) = {(H, W)}")

    if mode not in AUGMENT_FACTOR:
        raise ValueError(
            f"Unknown augmentation mode {mode!r}; expected one of "
            f"{sorted(AUGMENT_FACTOR)}")
    ks = range(4) if mode == "full" else range(1)

    xs, ts, ps = [], [], []
    for k in ks:
        xr = torch.rot90(x,       k, dims=(-2, -1))
        xm = torch.flip(xr, dims=[-1])
        tr = torch.rot90(targets, k, dims=(-2, -1))
        tm = torch.flip(tr, dims=[-1])
        xs.extend([xr, xm])
        ts.extend([tr, tm])
        if push_px is not None:
            ps.extend([_augment_push_endpoints(push_px, x.shape[-1], k, False),
                       _augment_push_endpoints(push_px, x.shape[-1], k, True)])

    new_batch: TrainingBatch = {
        "input":  torch.cat(xs, dim=0),
        "target": torch.cat(ts, dim=0),
    }
    if "physics" in batch:
        new_batch["physics"] = batch["physics"].repeat(AUGMENT_FACTOR[mode], 1)
    if push_px is not None:
        new_batch["push_px"] = torch.cat(ps, dim=0)
    return new_batch


def _is_eulerian_batch(batch: TrainingBatch) -> bool:
    return ("input" in batch) and ("target" in batch)


def _batch_size(batch: TrainingBatch) -> int:
    """Infer batch size from the first present tensor key."""
    for key in (
        "input",
        "target",
        "s_cur",
        "target_particles",
        "particles",
        "a_cur",
        "particle_nums",
    ):
        tensor = batch.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
            return int(tensor.shape[0])
    raise KeyError("Cannot infer batch size from batch keys.")


class _EpochPermSampler(torch.utils.data.Sampler):
    """Resumable shuffling: epoch e's order is randperm(n) under seed
    (seed + e), independent of the global RNG; ``set_epoch(e, skip)`` drops
    the first ``skip`` indices (the already-trained part of a resumed epoch)."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.skip = n, int(seed), 0, 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch, self.skip = int(epoch), int(skip)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.skip:].tolist())

    def __len__(self):
        return self.n - self.skip


def _rows_fingerprint(ds) -> str:
    """sha1 of a dataset's row identity (the raw dataset's index map and
    group ids when available, else just its length)."""
    raw = getattr(ds, "raw_dataset", ds)
    h = hashlib.sha1(str(len(ds)).encode())
    im = getattr(raw, "_index_map", None)
    if im is not None:
        h.update(np.asarray(im, dtype=np.int64).tobytes())
        if hasattr(raw, "get_run_index"):
            h.update(np.asarray([raw.get_run_index(i) for i in range(len(raw))], np.int64).tobytes())
    return h.hexdigest()


def _atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Model-agnostic training loop for push-dynamics models.

    Parameters
    ----------
    model_wrapper : ModelTrainingWrapper
        Wraps the underlying nn.Module; forward(batch) → prediction tensor.
    train_ds / val_ds / test_ds : Dataset
        Each returns a standard batch dict (see registry/dataset_registry.py).
    loss_fn  : LossFn callable
        (prediction, batch) → (total_loss: Tensor, components: dict[str, float])
    metrics  : EulerianMetrics or compatible
        update / compute / reset interface.
    cfg : dict
        Fully resolved training config dict (has "model", "dataset",
        "training", "inference", "output" sections).
    """

    def __init__(
        self,
        model_wrapper: ModelTrainingWrapper,
        train_ds,
        val_ds,
        test_ds,
        loss_fn,
        metrics,
        cfg: dict,
    ):
        self.model_wrapper = model_wrapper.to(DEVICE)
        self.train_ds  = train_ds
        self.val_ds    = val_ds
        self.test_ds   = test_ds
        self.loss_fn   = loss_fn
        self.metrics   = metrics
        self.cfg       = cfg
        self._tcfg     = cfg.get("training", {})
        self._resumed  = False
        self._resume_full = True

    @classmethod
    def from_config(cls, config_path: str | Path, resume: bool = True) -> "Trainer":
        """
        Build a Trainer from a training config YAML file.

        Parameters
        ----------
        config_path : str | Path — path to training config YAML.
        resume      : bool       — if True, load checkpoint from log_dir before
                                   training (weights only; epoch counter resets).

        Required config sections: model, dataset, training
        Optional config sections: inference, output
        """
        cfg = load_config(config_path)
        cfg["_config_path"] = str(Path(config_path).resolve())

        model_wrapper = build_model(cfg["model"])
        default_loss_type = "lagrangian_mse" if cfg["model"]["type"] == "gnn-propnet" else "eulerian_combined"
        loss_cfg      = cfg["training"].get("loss", {"type": default_loss_type})
        loss_fn       = build_loss(loss_cfg)
        metrics       = build_metrics(cfg["model"]["type"])

        train_ds = build_dataset(cfg["dataset"], "train")
        val_ds   = build_dataset(cfg["dataset"], "val")
        test_ds  = build_dataset(cfg["dataset"], "test")

        trainer = cls(model_wrapper, train_ds, val_ds, test_ds, loss_fn, metrics, cfg)
        trainer._resume_full = bool(resume)
        if resume:
            trainer._try_resume()
        return trainer

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full training loop, then evaluate on the test set."""
        tcfg    = self._tcfg
        log_dir = self._get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

        # Write config for reproducibility
        with open(log_dir / "run_config.yaml", "w") as f:
            yaml.dump(self.cfg, f, sort_keys=False)

        # Prepare model card (written after each best checkpoint)
        self._model_card = ModelCard.from_training_config(
            self.cfg, log_dir, "unet_best.pth"
        )

        writer     = SummaryWriter(log_dir=log_dir)
        optimizer, scheduler, scaler = self._build_optimizer()

        batch_size  = int(tcfg.get("batch_size", 64))
        # `augmentation` is true / false / "flip" (the ×2 rotation-free
        # subgroup, for models that are rotation-invariant by construction --
        # see `_augment_eulerian_batch`). true and false behave exactly as
        # they always have.
        aug_cfg     = tcfg.get("augmentation", True)
        if isinstance(aug_cfg, str):
            aug_mode = aug_cfg.lower()
            if aug_mode not in AUGMENT_FACTOR:
                raise ValueError(
                    f"training.augmentation={aug_cfg!r} not understood; use "
                    f"true, false, or one of {sorted(AUGMENT_FACTOR)}")
            augment = True
        else:
            augment  = bool(aug_cfg)
            aug_mode = "full"
        num_workers = int(tcfg.get("num_workers", 4))

        # Reduce the loader batch so the AUGMENTED batch is `batch_size`,
        # by whatever factor this augmentation mode multiplies by.
        loader_bs = (max(1, batch_size // AUGMENT_FACTOR[aug_mode])
                     if augment else batch_size)

        full_state = bool(tcfg.get("save_full_state", False))
        sampler = None
        if full_state:
            sampler = _EpochPermSampler(len(self.train_ds), int(tcfg.get("shuffle_seed", 0)))
            train_loader = DataLoader(
                self.train_ds, batch_size=loader_bs, sampler=sampler, num_workers=num_workers,
                pin_memory=(DEVICE == "cuda"), collate_fn=getattr(self.train_ds, "collate_fn", None))
        else:
            train_loader = self._make_loader(self.train_ds, loader_bs, shuffle=True,  num_workers=num_workers)
        val_loader   = self._make_loader(self.val_ds,   loader_bs, shuffle=False, num_workers=num_workers)

        epochs     = int(tcfg.get("epochs",               100))
        patience   = int(tcfg.get("patience",             100))
        min_rel    = float(tcfg.get("patience_min_rel_delta", 0.0))
        save_every = int(tcfg.get("save_every_n_epochs",   10))

        best_val_loss = float("inf")
        best_epoch    = 0
        no_improve    = 0
        plateau_ref   = float("inf")
        start_epoch, skip_batches = 0, 0
        resume_sums = None
        state_path = log_dir / "last_state.pt"
        every_s = 60.0 * float(tcfg.get("full_state_every_min", 10))
        fps = None
        if full_state:
            fps = {"train": _rows_fingerprint(self.train_ds), "val": _rows_fingerprint(self.val_ds)}
        if full_state and self._resume_full and state_path.exists():
            st = torch.load(state_path, map_location=DEVICE, weights_only=False)
            if st["fingerprints"] != fps:
                raise RuntimeError(f"{state_path}: train/val rows differ from the checkpoint "
                                   f"({st['fingerprints']} vs {fps}); refusing to resume")
            self.model_wrapper.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            scaler.load_state_dict(st["scaler"])
            best_val_loss, best_epoch, no_improve = st["best_val_loss"], st["best_epoch"], st["no_improve"]
            plateau_ref = st.get("plateau_ref", best_val_loss)
            start_epoch, skip_batches = st["epoch"], st["batches_done"]
            resume_sums = st["sums"]
            torch.set_rng_state(st["rng"]["torch"].cpu()); np.random.set_state(st["rng"]["numpy"])
            random.setstate(st["rng"]["python"])
            if DEVICE == "cuda" and st["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all([t.cpu() for t in st["rng"]["cuda"]])
            print(f"Resumed FULL state from {state_path}: epoch {start_epoch} "
                  f"(+{skip_batches} batches), best={best_epoch} ({best_val_loss:.6f}), "
                  f"lr={scheduler.get_last_lr()[0]:.3g}", flush=True)

        def _save_full(epoch_, batches_done_, sums_):
            _atomic_torch_save({
                "model": self.model_wrapper.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch_, "batches_done": batches_done_, "sums": sums_,
                "best_val_loss": best_val_loss, "best_epoch": best_epoch, "no_improve": no_improve,
                "plateau_ref": plateau_ref,
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                        "python": random.getstate(),
                        "cuda": torch.cuda.get_rng_state_all() if DEVICE == "cuda" else None},
                "fingerprints": fps, "loader_bs": loader_bs, "time": time.time(),
            }, state_path)

        print(
            f"Training on {DEVICE}, epochs 0→{epochs}, log_dir={log_dir}\n"
            f"Train: {len(self.train_ds)}  Val: {len(self.val_ds)}  Test: {len(self.test_ds)}"
        )

        with trange(start_epoch, epochs, desc="Epochs") as tbar:
            for epoch in tbar:

                # ── Train ────────────────────────────────────────────────────
                self.model_wrapper.train()
                train_loss_sum = 0.0
                train_comp_sum: dict[str, float] = {}
                train_n = 0
                b_done = 0
                if sampler is not None:
                    sampler.set_epoch(epoch, skip_batches * loader_bs)
                    if resume_sums is not None and skip_batches > 0:
                        train_loss_sum, train_comp_sum, train_n = (
                            resume_sums["loss"], dict(resume_sums["comp"]), resume_sums["n"])
                    b_done, skip_batches, resume_sums = skip_batches, 0, None
                t_last = time.time()

                for batch in train_loader:
                    batch = _to_device(batch, DEVICE)
                    if augment and _is_eulerian_batch(batch):
                        batch = _augment_eulerian_batch(batch, aug_mode)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                        prediction = self.model_wrapper(batch)
                        loss, components = self.loss_fn(prediction, batch)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model_wrapper.model.parameters(), grad_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()

                    bsz = _batch_size(batch)
                    train_loss_sum += loss.item() * bsz
                    for k, v in components.items():
                        train_comp_sum[k] = train_comp_sum.get(k, 0.0) + v * bsz
                    train_n += bsz
                    b_done += 1
                    if full_state and time.time() - t_last > every_s:
                        _save_full(epoch, b_done, {"loss": train_loss_sum, "comp": train_comp_sum,
                                                   "n": train_n})
                        t_last = time.time()

                scheduler.step()
                train_loss  = train_loss_sum / max(1, train_n)
                train_comps = {k: v / max(1, train_n) for k, v in train_comp_sum.items()}

                # ── Validate ─────────────────────────────────────────────────
                val_loss, val_comps, val_metrics = self._evaluate(val_loader)

                # ── Checkpoint / early stopping ───────────────────────────────
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch    = epoch + 1
                    self._save_checkpoint(log_dir / "unet_best.pth")
                    self._model_card.save()
                if val_loss < plateau_ref * (1.0 - min_rel):
                    plateau_ref = val_loss
                    no_improve  = 0
                else:
                    no_improve += 1

                if (epoch + 1) % save_every == 0:
                    self._save_checkpoint(log_dir / f"unet_epoch_{epoch + 1}.pth")
                if full_state:
                    self._save_checkpoint(log_dir / "unet_last.pth")
                    _save_full(epoch + 1, 0, None)

                # ── Logging ───────────────────────────────────────────────────
                self._log(
                    writer, epoch,
                    train_loss, train_comps,
                    val_loss, val_comps, val_metrics,
                    scheduler.get_last_lr()[0], best_epoch, no_improve,
                )
                tbar.set_postfix({
                    "trn":  f"{train_loss:.4f}",
                    "val":  f"{val_loss:.4f}",
                    "iou":  f"{val_metrics.get('hard_iou', 0):.3f}",
                    "best": best_epoch,
                    "ni":   no_improve,
                })
                print(
                    f"Epoch {epoch+1:4d}: trn={train_loss:.6f}  val={val_loss:.6f}  "
                    f"iou={val_metrics.get('hard_iou', 0):.4f}  "
                    f"chg_mse={val_metrics.get('changed_mse', 0):.6f}  "
                    f"best={best_epoch}", flush=True
                )

                if no_improve >= patience:
                    print(f"Early stopping after {patience} epochs without improvement.")
                    break

        # Write the final "completed" checkpoint; dedup uses this as sentinel
        self._save_checkpoint(log_dir / "unet.pth")
        writer.close()

        # ── Test ─────────────────────────────────────────────────────────────
        test_loader = self._make_loader(
            self.test_ds, loader_bs, shuffle=False, num_workers=num_workers
        )
        _, _, test_metrics = self._evaluate(test_loader)
        print("\n=== Test metrics ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")
        print(
            f"\n=== Training complete ===\n"
            f"  Best val loss : {best_val_loss:.6f}  (epoch {best_epoch})\n"
            f"  Model card    : {self._model_card.path}\n"
            f"  Log dir       : {log_dir}"
        )

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _evaluate(self, loader) -> tuple[float, dict, dict]:
        """Run one evaluation pass. Returns (mean_loss, loss_components, metrics)."""
        self.model_wrapper.eval()
        self.metrics.reset()
        loss_sum: float = 0.0
        comp_sum: dict[str, float] = {}
        n = 0

        with torch.no_grad():
            for batch in loader:
                batch      = _to_device(batch, DEVICE)
                prediction = self.model_wrapper(batch)
                loss, comps = self.loss_fn(prediction, batch)
                bsz = _batch_size(batch)
                loss_sum += loss.item() * bsz
                for k, v in comps.items():
                    comp_sum[k] = comp_sum.get(k, 0.0) + v * bsz
                n += bsz
                self.metrics.update(prediction, batch)

        mean_loss  = loss_sum / max(1, n)
        mean_comps = {k: v / max(1, n) for k, v in comp_sum.items()}
        return mean_loss, mean_comps, self.metrics.compute()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_log_dir(self) -> Path:
        """
        Return the log directory, deduplicating if a completed run already lives there.
        Completed = ``unet.pth`` present (written at the very end of run()).
        An in-progress run (has checkpoints but no unet.pth) is resumed in-place.
        """
        base = Path(self.cfg.get("output", {}).get("log_dir", "runs/training"))
        if self._resumed:
            return base   # resume in the same directory
        if (base / "unet.pth").exists():
            counter = 2
            while True:
                cand = base.with_name(f"{base.name}_{counter}")
                if not (cand / "unet.pth").exists():
                    return cand
                counter += 1
        return base

    def _build_optimizer(self):
        tcfg = self._tcfg
        lr   = float(tcfg.get("lr", 1e-4))

        optimizer = torch.optim.Adam(
            self.model_wrapper.model.parameters(), lr=lr
        )
        sched_cfg  = tcfg.get("lr_scheduler", {})
        step_size  = int(sched_cfg.get("step_size", 100))
        gamma      = float(sched_cfg.get("gamma", 0.75))
        scheduler  = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        use_amp = bool(tcfg.get("mixed_precision", True)) and (DEVICE == "cuda")
        scaler  = torch.amp.GradScaler(enabled=use_amp)
        return optimizer, scheduler, scaler

    def _make_loader(self, ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        collate_fn = getattr(ds, "collate_fn", None)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(DEVICE == "cuda"),
            collate_fn=collate_fn,
        )

    def _save_checkpoint(self, path: Path) -> None:
        _atomic_torch_save(self.model_wrapper.state_dict(), path)

    def _try_resume(self) -> None:
        """
        If a checkpoint exists in output.log_dir, load it (weights only).
        Sets self._resumed = True so _get_log_dir() keeps the same directory.
        """
        log_dir = Path(self.cfg.get("output", {}).get("log_dir", "runs/training"))
        if not log_dir.exists():
            return
        if bool(self._tcfg.get("save_full_state", False)) and (log_dir / "last_state.pt").exists():
            self._resumed = True     # run() restores the full state itself
            print(f"Found full resumable state {log_dir / 'last_state.pt'}")
            return

        # Prefer the most recent epoch checkpoint, fall back to unet_best.pth
        candidates = []
        for p in log_dir.glob("unet_epoch_*.pth"):
            try:
                candidates.append((int(p.stem.split("_")[-1]), p))
            except ValueError:
                pass
        if candidates:
            _, latest = max(candidates)
        elif (log_dir / "unet_best.pth").exists():
            latest = log_dir / "unet_best.pth"
        else:
            return

        state = torch.load(latest, map_location=DEVICE, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model_wrapper.load_state_dict(state["model_state_dict"])
        else:
            self.model_wrapper.load_state_dict(state)

        self._resumed = True
        print(f"Resumed weights from {latest}")

    def _log(
        self, writer, epoch,
        train_loss, train_comps,
        val_loss, val_comps, val_metrics,
        lr, best_epoch, no_improve,
    ) -> None:
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val",   val_loss,   epoch)
        writer.add_scalar("LR",         lr,         epoch)
        writer.add_scalar("Convergence/BestEpoch",            best_epoch, epoch)
        writer.add_scalar("Convergence/EpochsNoImprovement",  no_improve, epoch)
        for k, v in train_comps.items():
            writer.add_scalar(f"TrainComponent/{k}", v, epoch)
        for k, v in val_comps.items():
            writer.add_scalar(f"ValComponent/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"ValMetric/{k}", v, epoch)
