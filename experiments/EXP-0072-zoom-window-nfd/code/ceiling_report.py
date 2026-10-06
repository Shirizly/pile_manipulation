"""Render markdown tables from results/ceiling/*.json (raster_perceptibility, ceiling_scores, metrics_vs_slate) -> results/ceiling/tables.md"""
import json, numpy as np, sys
sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code")
from ceiling_common import OUT
L = []; w = L.append
rp = json.load(open(OUT + "raster_perceptibility.json")); cs = json.load(open(OUT + "ceiling_scores.json")); mv = json.load(open(OUT + "metrics_vs_slate.json"))
w("## T1 raster perceptibility (per-cube iid xy gaussian + yaw 2 deg/mm; 200 rows x 2 draws)")
w("Changed-pixel fraction is linear in sigma (boundary pixels flip at a rate proportional to perimeter x shift), so there is no threshold: sigma_1% = sigma where the hard raster changes by 1% of its occupied pixels; sigma_95 = sigma where >=95% of states give a bit-identical raster, extrapolated from the 0.01 mm point with P(identical)=exp(-lam sigma).\n")
w("| model input raster | px/mm | changed px @0.01mm | frac occupied changed @0.01mm | P(identical) @0.01mm | sigma_1% (mm) | sigma_95 identical (mm) |\n|---|---|---|---|---|---|---|")
for k, v in rp.items():
    a = v["0.01"]; lam = -np.log(max(a["frac_identical"], 1e-3)) / 0.01; s95 = -np.log(.95) / lam
    k1 = np.polyfit([float(s) for s in v if float(s) <= 0.1], [x["changed_frac_of_occupied"] for s, x in v.items() if float(s) <= 0.1], 1)[0]
    w(f"| {k} | | {a['mean_changed_px']:.1f} | {a['changed_frac_of_occupied']:.4f} | {a['frac_identical']:.2f} | {0.01 / k1:.3f} | {s95:.4f} |")
w("\n## T2 ceiling: perfect-physics re-simulation scored as a model prediction of the recorded DS-0016 outcome (accuracy_1, pooled; n=896 rows)")
tags = sorted([t for t in cs if not t.startswith("_")], key=lambda t: (float(t.split("mm")[0]), t))
fr = ["w64_native", "w128_native", "window64", "window128", "paste_zoom64", "paste_zoom128", "paste_world128"]
w("| level / rep | " + " | ".join(fr) + " | mm_rms | frac cubes >1mm |\n|" + "---|" * (len(fr) + 3))
tl = cs["_true_label_cap"]["onestep"]; w("| TRUE label rendered in frame (raster-style cap) | " + " | ".join(f"{tl[f]:.3f}" for f in fr) + " | 0 | 0 |")
for t in tags:
    o = cs[t]["onestep"]; w(f"| {t} | " + " | ".join(f"{o['all'][f]:.3f}" for f in fr) + f" | {o['mm']['mm_rms']:.2f} | {o['mm']['frac_cubes_gt1mm']:.3f} |")
w("\nScatter / clump split (w64_native | paste_zoom64):")
for t in tags: o = cs[t]["onestep"]; w(f"- {t}: scatter {o['scatter']['w64_native']:.3f} | {o['scatter']['paste_zoom64']:.3f}; clump {o['clump']['w64_native']:.3f} | {o['clump']['paste_zoom64']:.3f}")
w("\n## T3 ceiling slateN (32 pools x 64 pushes; one perturbed start per pool; 13 goals)")
pf = ["w64_native", "paste_zoom64", "paste_zoom128"]; tc = cs["_true_label_cap"]["pools_slateN"]
w("| level / rep | " + " | ".join(pf) + " |\n|---|---|---|---|"); w("| TRUE label in frame (no noise) | " + " | ".join(f"{tc[f]:.3f}" for f in pf) + " |")
for t in tags: w(f"| {t} | " + " | ".join(f"{cs[t]['pools_slateN'][f]:.3f}" for f in pf) + " |")
w("\n## T4 multi-push divergence (chains, perturbed ONCE at chain start, pushes re-run in sequence; accuracy of push k's swept region, truth = recorded)")
w("| level | metric | " + " | ".join(f"push {k}" for k in range(1, 9)) + " |\n|" + "---|" * 10)
for t in tags:
    c = cs[t]["chain"]
    w(f"| {t} | acc w64_native | " + " | ".join(f"{c[str(k)]['acc']['w64_native']:.3f}" for k in range(1, 9)) + " |")
    w(f"| {t} | acc paste_zoom64 | " + " | ".join(f"{c[str(k)]['acc']['paste_zoom64']:.3f}" for k in range(1, 9)) + " |")
    w(f"| {t} | cube mm_rms | " + " | ".join(f"{c[str(k)]['mm']['mm_rms']:.2f}" for k in range(1, 9)) + " |")
    w(f"| {t} | frac cubes >2.5mm | " + " | ".join(f"{c[str(k)]['mm']['frac_cubes_gt2.5mm']:.3f}" for k in range(1, 9)) + " |")
open(OUT + "tables.md", "w").write("\n".join(L)); print("\n".join(L))
