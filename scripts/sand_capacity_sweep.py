"""Crop/resolution sweep for both observation views."""
from __future__ import annotations
import argparse, os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--glob", required=True)
a = ap.parse_args()
print(f"{'view':>8s} {'crop':>5s} {'res':>4s} {'M/D':>7s} | "
      f"{'linear':>8s} {'identity':>9s} {'heur':>7s}", flush=True)
print("-" * 62, flush=True)
for view, blur in (("density", "0"), ("mask", "1")):
    for crop, res in ((1.0, 64), (0.5, 32), (0.375, 24), (0.25, 16)):
        out = subprocess.run(
            [sys.executable, "sand_foresight.py", "--glob", a.glob, "--view", view,
             "--blur", blur, "--min-grains", "2", "--res", str(res),
             "--crop", str(crop), "--seed", "0"],
            capture_output=True, text=True, cwd=ROOT).stdout
        md = (re.search(r"M/D=([\d.]+)", out) or [None, "?"])[1]
        blk = out.split("SWEPT REGION")[-1]
        g = lambda n: (re.search(rf"^{re.escape(n)}\s+[\d.]+\s+([\d.]+)%", blk, re.M)
                       or [None, "?"])[1]
        print(f"{view:>8s} {crop:5.3g} {res:4d} {md:>7s} | {g('linear-nonneg'):>7s}% "
              f"{g('identity (warp only)'):>8s}% {g('heur-cumulative'):>6s}%", flush=True)
