"""Soft-occupancy NFD plumbing (EXP-0063): the blur, the `occupancy_blur`
dataset transform, and the eval-side sigma lookup / adapter encoding."""
import torch
import yaml

from transforms.functional import gaussian_blur_occ
from transforms.representation import build_transforms, EnsureRepresentation, EulerianOccupancyAliases


def test_blur_identity_at_zero_and_shape():
    x = torch.rand(2, 3, 16, 16)
    assert gaussian_blur_occ(x, 0.0) is x
    assert gaussian_blur_occ(x, 1.0).shape == x.shape


def test_blur_conserves_interior_mass_and_spreads():
    x = torch.zeros(1, 32, 32); x[0, 16, 16] = 1.0
    y = gaussian_blur_occ(x, 1.0)
    assert abs(float(y.sum()) - 1.0) < 1e-5
    assert float(y.max()) < 0.2 and float(y[0, 16, 17]) > 0


def test_blur_symmetric_commutes_with_augmentation():
    # the trainer's rot90/flip augmentation runs AFTER the dataset transform
    x = torch.rand(1, 20, 20)
    a = torch.rot90(gaussian_blur_occ(x, 1.5), 1, dims=(-2, -1))
    b = gaussian_blur_occ(torch.rot90(x, 1, dims=(-2, -1)), 1.5)
    assert torch.allclose(a, b, atol=1e-6)


def test_occupancy_blur_transform_updates_aliases():
    tx = build_transforms([{"type": "occupancy_blur", "sigma": 1.0}],
                          defaults=[EnsureRepresentation("eulerian"), EulerianOccupancyAliases()])
    inp = torch.zeros(3, 16, 16); inp[0, 8, 8] = 1.0; inp[1, 4, 4] = 1.0
    tgt = torch.zeros(16, 16); tgt[9, 9] = 1.0
    inp0, tgt0 = inp.clone(), tgt.clone()
    out = tx({"input": inp, "target": tgt})
    assert torch.allclose(out["input"][0], gaussian_blur_occ(inp0[0], 1.0))
    assert torch.equal(out["input"][1:], inp0[1:])           # plate channels untouched
    assert torch.allclose(out["target"], gaussian_blur_occ(tgt0, 1.0))
    # the aliases the loss reads first must be the blurred tensors, not the stale hard ones
    assert out["target_occupancy"] is out["target"]
    assert torch.equal(out["current_occupancy"], out["input"][0])
    assert torch.equal(inp, inp0) and torch.equal(tgt, tgt0)  # raw buffers not mutated


def test_sigma_read_from_run_config(tmp_path):
    from Baselines.NFD.predictor import input_blur_sigma_from_run
    ck = tmp_path / "unet_best.pth"
    assert input_blur_sigma_from_run(str(ck)) == 0.0
    (tmp_path / "run_config.yaml").write_text(yaml.safe_dump(
        {"dataset": {"transforms": [{"type": "eulerian_aliases"},
                                    {"type": "occupancy_blur", "sigma": 2.0}]}}))
    assert input_blur_sigma_from_run(str(ck)) == 2.0


def test_adapter_encode_state_identity_by_default():
    from simple_mpc.adapters import OccupancyGradientAdapter
    ad = OccupancyGradientAdapter("x", device="cpu")
    occ = torch.rand(1, 64, 64)
    assert ad.encode_state(occ) is occ
    ad.input_blur_sigma = 1.0
    assert torch.allclose(ad.encode_state(occ), gaussian_blur_occ(occ, 1.0))
