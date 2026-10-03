"""Composable representation transforms for dataset and adapter pipelines."""

from __future__ import annotations

from typing import Any, Callable

from transforms.functional import particles_to_occupancy


class Compose:
    """Simple callable pipeline similar to torchvision Compose."""

    def __init__(self, transforms: list[Callable[[dict[str, Any]], dict[str, Any]]]):
        self.transforms = transforms

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            batch = transform(batch)
        return batch


class EnsureRepresentation:
    def __init__(self, representation: str):
        self.representation = representation

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch.setdefault("representation", self.representation)
        return batch


class EulerianOccupancyAliases:
    """Populate explicit occupancy keys for representation-agnostic losses/metrics."""

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "input" in batch and "current_occupancy" not in batch:
            batch["current_occupancy"] = batch["input"][:, 0] if batch["input"].ndim == 4 else batch["input"][0]
        if "target" in batch and "target_occupancy" not in batch:
            batch["target_occupancy"] = batch["target"]
        return batch


class OccupancyBlur:
    """Soft-occupancy representation (EXP-0063): Gaussian-blur the occupancy
    channels of ``input`` (default channel 0 only -- the plate channels are
    already soft renders) and, if ``target``, the target, with
    ``transforms.functional.gaussian_blur_occ``.

    Writes NEW tensors (the raw dataset may hand back its own reused grid
    buffers) and re-points the ``current_occupancy``/``target_occupancy``
    aliases, which the wrapper's default transforms set BEFORE config
    transforms run -- a stale alias would silently train on the hard target
    (``EulerianCombinedLoss`` reads ``target_occupancy`` first).
    """

    def __init__(self, sigma: float, input_channels=(0,), target: bool = True):
        self.sigma = float(sigma)
        self.input_channels = tuple(int(c) for c in input_channels)
        self.target = bool(target)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        from transforms.functional import gaussian_blur_occ
        if "input" in batch:
            x = batch["input"].clone()
            for c in self.input_channels:
                x[..., c, :, :] = gaussian_blur_occ(x[..., c, :, :], self.sigma)
            batch["input"] = x
            if "current_occupancy" in batch:
                batch["current_occupancy"] = x[..., 0, :, :]
        if self.target and "target" in batch:
            batch["target"] = gaussian_blur_occ(batch["target"], self.sigma)
            if "target_occupancy" in batch:
                batch["target_occupancy"] = batch["target"]
        return batch


class LagrangianAliases:
    """Add generic aliases for particle-based samples."""

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "s_cur" in batch:
            batch.setdefault("particles", batch["s_cur"])
        if "target_particles" in batch:
            batch.setdefault("target", batch["target_particles"])
        return batch


class LagrangianToEulerian:
    """Convert particles to occupancy and optionally set as input channel 0."""

    def __init__(
        self,
        bounds: dict[str, float],
        resolution: tuple[int, int],
        sigma: float = 0.0,
        write_input: bool = False,
    ):
        self.bounds = bounds
        self.resolution = resolution
        self.sigma = sigma
        self.write_input = write_input

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        particles = batch.get("particles", batch.get("s_cur"))
        if particles is None:
            return batch

        if particles.ndim == 2:
            particles = particles.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        occ = particles_to_occupancy(
            particles,
            self.bounds,
            self.resolution,
            sigma=self.sigma,
        )

        if squeeze:
            occ = occ[0]
        batch["occupancy"] = occ
        if self.write_input:
            batch["input"] = occ
        return batch


def build_transforms(
    configs: list[dict[str, Any]] | None,
    *,
    defaults: list[Callable[[dict[str, Any]], dict[str, Any]] | Any] | None = None,
) -> Compose:
    """Build a transform pipeline from config dicts + optional default callables."""
    tx: list[Callable[[dict[str, Any]], dict[str, Any]] | Any] = []
    if defaults:
        tx.extend(defaults)

    for cfg in configs or []:
        ttype = cfg.get("type", "").strip()
        if ttype == "ensure_representation":
            tx.append(EnsureRepresentation(cfg["representation"]))
        elif ttype == "eulerian_aliases":
            tx.append(EulerianOccupancyAliases())
        elif ttype == "occupancy_blur":
            tx.append(OccupancyBlur(
                sigma=float(cfg["sigma"]),
                input_channels=cfg.get("input_channels", (0,)),
                target=bool(cfg.get("target", True)),
            ))
        elif ttype == "lagrangian_aliases":
            tx.append(LagrangianAliases())
        elif ttype == "lagrangian_to_eulerian":
            tx.append(
                LagrangianToEulerian(
                    bounds=cfg["bounds"],
                    resolution=tuple(cfg["resolution"]),
                    sigma=float(cfg.get("sigma", 0.0)),
                    write_input=bool(cfg.get("write_input", False)),
                )
            )
        else:
            raise ValueError(f"Unknown transform type: {ttype!r}")

    return Compose([t for t in tx if callable(t)])
