"""Assert the fast rasteriser matches transforms.functional.particles_to_occupancy."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transforms.functional import particles_to_occupancy
from raster import rasterise, BOUNDS, GRID, BASE_RADIUS
from data import randlen_files, load_transitions

DEV = "cuda" if torch.cuda.is_available() else "cpu"
d = load_transitions(randlen_files("train")[:2])
p = d["pts0"][:64].to(DEV)
p3 = torch.cat([p, torch.zeros_like(p[..., :1])], -1)
assert p.device.type == DEV, p.device
t0 = time.time(); ref = particles_to_occupancy(p3, BOUNDS, (GRID, GRID), footprint_radius=BASE_RADIUS); t_ref = time.time()-t0
t0 = time.time(); new = rasterise(p, BASE_RADIUS); t_new = time.time()-t0
print("ref device", ref.device, "new device", new.device)
print(f"max abs diff {(ref-new).abs().max():.3g}  mismatched cells {(ref!=new).sum().item()}")
print(f"project rasteriser {t_ref*1000:.0f} ms vs fast {t_new*1000:.0f} ms for B=64")
assert torch.equal(ref, new), "fast rasteriser is NOT equivalent"
print("EQUIVALENT")
