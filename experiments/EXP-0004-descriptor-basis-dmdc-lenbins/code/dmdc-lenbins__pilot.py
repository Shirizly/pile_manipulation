"""Cost pilot: load N files, rasterise + compute descriptors, time it, extrapolate."""
import sys, time, glob
sys.path.insert(0, "/home/alon/Code/pile_manipulation")
import torch
from transforms.functional import particles_to_occupancy
from dmdc_baseline import occupancy_descriptors

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH

files = sorted(glob.glob("Genesis/data/overnight_randlen/*/cube/**/_*_data.pt", recursive=True))
print(f"total files found: {len(files)}")

t0 = time.time()
for f in files[:2]:
    d = torch.load(f, map_location="cpu")
    states = d["states"][:, :, :3]   # [512,P,3]
    states_ = d["states_"][:, :, :3]
    occ0 = particles_to_occupancy(states, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
    occ1 = particles_to_occupancy(states_, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
    phi0 = occupancy_descriptors(occ0)
    phi1 = occupancy_descriptors(occ1)
dt = time.time() - t0
print(f"2 files: {dt:.2f}s -> per file {dt/2:.3f}s -> extrapolated 213 files: {dt/2*213:.1f}s ({dt/2*213/60:.2f} min)")
