"""EXP-0022 R1/R2 -- dead-gradient check for `clamp(occ0 + delta, 0, 1)`.

`clamp` has zero local gradient wherever the pre-clamp sum is <=0 or >=1 --
most of an occupancy image, since `occ0` is close to 0 or 1 almost
everywhere except thin blurred edges. This script measures, on a real
trained checkpoint and a real batch:

  - the fraction of pixels where the clamp is ACTIVE (dead gradient there)
  - of those dead pixels, the fraction that actually needed to change
    (|occ1 - occ0| > 0.05) -- i.e. whether the dead zone overlaps the
    pixels that matter for the loss, or only the truly-static background.

Usage: python residual_dead_gradient.py <config.yaml> <checkpoint.pth> unwarped|warped
"""
import sys

import torch
import yaml

import Baselines.NFD.nfd_lib  # noqa: F401
import model.warped_nfd.lib  # noqa: F401
import model.residual_nfd.lib as rlib  # noqa: F401
from registry.dataset_registry import build_dataset
from registry.model_registry import build_model


def main():
    cfg_path, ckpt_path, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = yaml.safe_load(open(cfg_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = build_dataset(cfg["dataset"], split="test")
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    batch = next(iter(loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    wrapper = build_model(cfg["model"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    wrapper.model.load_state_dict(state)
    wrapper.model.to(device).eval()

    x = batch["input"]
    occ0 = x[:, 0]
    target = batch["target"] if "target" in batch else batch.get("target_occupancy")

    with torch.no_grad():
        if kind == "unwarped":
            raw = wrapper.model(x)
            if raw.dim() == 4:
                raw = raw.squeeze(1)
            delta = torch.tanh(raw)
            pre_clamp = occ0 + delta
        elif kind == "warped":
            from model.warped_nfd.predictor import build_canonical_stack
            from transforms.functional import from_push_frame, to_push_frame

            action_world = x[:, 1:3]
            push_px = batch["push_px"]
            start_px, end_px = push_px[:, :2], push_px[:, 2:]
            H = occ0.shape[-1]
            canon_res = wrapper.canon_res or H
            canon_occ0 = to_push_frame(occ0, start_px, end_px, (canon_res, canon_res), wrapper.scale)
            stack = build_canonical_stack(
                canon_occ0, start_px, end_px, action_world, occ0,
                wrapper.plate_mode, wrapper.wall_channel, canon_res, wrapper.scale,
                wrapper.plate_dim_x_px, wrapper.plate_dim_y_px, wrapper.plate_sigma_px, H,
            )
            raw = wrapper.model(stack)
            if raw.dim() == 4:
                raw = raw.squeeze(1)
            delta_canon = torch.tanh(raw)
            delta = from_push_frame(delta_canon, start_px, end_px, (H, H), wrapper.scale)
            pre_clamp = occ0 + delta
        else:
            raise ValueError(kind)

    dead = (pre_clamp <= 0.0) | (pre_clamp >= 1.0)
    dead_frac = dead.float().mean().item()

    if target is not None:
        needs_change = (target - occ0).abs() > 0.05
        dead_and_needed = (dead & needs_change).float().sum().item()
        needed_total = needs_change.float().sum().item()
        frac_needed_pixels_dead = dead_and_needed / max(needed_total, 1.0)
    else:
        frac_needed_pixels_dead = float("nan")

    print(f"kind={kind}")
    print(f"dead-gradient fraction (clamp active): {dead_frac:.4f}")
    print(f"of pixels needing change (|occ1-occ0|>0.05), fraction with dead gradient: "
          f"{frac_needed_pixels_dead:.4f}")
    print(f"batch size: {occ0.shape[0]}, image {occ0.shape[-2]}x{occ0.shape[-1]}")


if __name__ == "__main__":
    main()
