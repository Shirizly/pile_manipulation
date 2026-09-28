"""Flatten EXP-0059's `perturbed_sim_zoo.json` (nested {levels_mm: {level: {blurN: {...}}},
model_control: {model: {blurN: {...}}}}) into the flat {model_name: {metric: value}} shape
`correlate.py --sources` expects, so the perturbed-sim zoo members and the blurred-NFD control
become ordinary rows in the correlation table -- no change to `correlate.py` itself.

One flattened row per (perturbation level, blur sigma) cell, named e.g. `sim_0.5mm_blur1`, and
one per (model, blur sigma) control cell, named e.g. `nfd_3ch_narrow_l20_blur1_control`. Only
`slateN`/`slateN_tough` exist for these rows (perturbed_sim_zoo.py does not compute
accuracy_1/occ_emd_swept/mass_in_goal_mae -- it is a slateN-only extension by design), so
`correlate.py`'s other candidate metrics are simply absent (nan) for these rows, exactly like
any model missing a key.

Usage:
    python -u ingest_perturbed_sim_zoo.py \\
        --zoo ../../EXP-0059-retrieval-transition-model/results/perturbed_sim_zoo.json \\
        --out results/perturbed_sim_zoo_flat.json
Then:
    python -u correlate.py --sources \\
        ../../EXP-0059-retrieval-transition-model/results/offline_eval_extended.json \\
        results/perturbed_sim_zoo_flat.json
"""
import argparse
import json
from pathlib import Path


def flatten(zoo: dict) -> dict:
    out = {}
    for model, by_sigma in zoo.get("model_control", {}).items():
        for sigma_key, cell in by_sigma.items():
            out[f"{model}_{sigma_key}_control"] = dict(slateN=cell.get("slateN"),
                                                        slateN_tough=cell.get("slateN_tough"))
    for level_key, by_sigma in zoo.get("levels_mm", {}).items():
        for sigma_key, cell in by_sigma.items():
            if sigma_key == "wall_s":
                continue
            out[f"sim_{level_key}_{sigma_key}"] = dict(slateN=cell.get("slateN"),
                                                        slateN_tough=cell.get("slateN_tough"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zoo = json.loads(Path(a.zoo).read_text())
    flat = flatten(zoo)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(flat, indent=1))
    print(f"{len(flat)} rows -> {a.out}")
    for k, v in flat.items():
        print(f"  {k:32s} slateN={v['slateN']}  slateN_tough={v['slateN_tough']}")


if __name__ == "__main__":
    main()
