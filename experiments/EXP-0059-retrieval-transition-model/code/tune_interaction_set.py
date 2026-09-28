"""Tune `model/retrieval/interaction.py`'s contact-chain slack `tau` on
TRAINING data only (DS-0008 + DS-0010, the bank source -- never DS-0009),
against the bank's own truth "moved" set (coordinator follow-up B,
2026-09-28). Reports recall/precision of the GEOMETRIC interaction set
against `moved` (displacement > 1mm) for tau in {1, 2, 3} mm, on ALL rows and
on LEGAL-ONLY rows (ISS-010: ~44-56% of DS-0008 rows have the tool on a cube
at touchdown -- a corrupted row's "moved" cubes include non-physical ejection
artifacts a geometric sweep rule cannot and should not be tuned to predict).
`DEFAULT_TAU` in interaction.py is then whichever tau reaches ~95% recall on
the LEGAL-ONLY population (the curated bank, built next, uses legal rows
only).

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/tune_interaction_set.py
"""
import glob
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from model.retrieval.bank import DS0008_DIR, DS0010_DIR  # noqa: E402
from model.retrieval.frame import world_to_push_frame  # noqa: E402
from model.retrieval.interaction import interaction_set, recall_precision  # noqa: E402

RES = REPO / "experiments/EXP-0059-retrieval-transition-model/results/interaction_set_tuning.json"


def _load(pattern, has_valid):
    states0, states1, p_starts, p_stops, legal = [], [], [], [], []
    for f in sorted(glob.glob(pattern)):
        d = torch.load(f, map_location="cpu", weights_only=False)
        n = len(d["states"])
        v = d["valid"] if (has_valid and "valid" in d) else torch.ones(n, dtype=torch.bool)
        legality_path = Path(f).with_name(Path(f).stem + "_legality.pt")
        legal_row = ~torch.load(legality_path, map_location="cpu", weights_only=False)["illegal_0mm"]
        states0.append(d["states"][v].float()); states1.append(d["states_"][v].float())
        p_starts.append(d["p_starts"][v].float()); p_stops.append(d["p_stops"][v].float())
        legal.append(legal_row[v])
    return (torch.cat(states0), torch.cat(states1), torch.cat(p_starts), torch.cat(p_stops),
           torch.cat(legal))


def main():
    s0a, s1a, psa, pea, lega = _load(str(DS0008_DIR / "_*_data.pt"), has_valid=True)
    s0b, s1b, psb, peb, legb = _load(str(DS0010_DIR / "*_data.pt"), has_valid=False)
    states0 = torch.cat([s0a, s0b]); states1 = torch.cat([s1a, s1b])
    p_starts = torch.cat([psa, psb]); p_stops = torch.cat([pea, peb])
    legal = torch.cat([lega, legb])
    print(f"training rows: {len(states0)} (DS-0008 {len(s0a)} + DS-0010 {len(s0b)}); "
         f"legal {int(legal.sum())} ({100*legal.float().mean():.1f}%)")

    uv0 = world_to_push_frame(states0[..., :2], p_starts, p_stops)
    uv1 = world_to_push_frame(states1[..., :2], p_starts, p_stops)
    duv = uv1 - uv0
    moved = duv.norm(dim=-1) > 0.001
    push_len = (p_stops[:, :2] - p_starts[:, :2]).norm(dim=-1)

    out = {}
    # design doc's own tau in {1,2,3}mm range first (default angle_max_deg=60), then a wider
    # sweep -- none of {1,2,3}mm reaches ~95% recall on legal-only rows (measured 0.823-0.898),
    # so the sweep continues until it does.
    for tau_mm in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0):
        for angle_max_deg in (60.0, 90.0):
            geo = interaction_set(uv0, push_len, tau=tau_mm / 1000.0, angle_max_deg=angle_max_deg)
            rp_all = recall_precision(geo, moved)
            rp_legal = recall_precision(geo[legal], moved[legal])
            key = f"tau_{tau_mm:g}mm_ang{angle_max_deg:g}"
            out[key] = dict(all_rows=rp_all, legal_only=rp_legal)
            print(f"tau={tau_mm:g}mm ang={angle_max_deg:g}  ALL: recall={rp_all['recall']:.4f} "
                 f"precision={rp_all['precision']:.4f}  LEGAL-ONLY: recall={rp_legal['recall']:.4f} "
                 f"precision={rp_legal['precision']:.4f}")

    out["chosen_default"] = dict(tau_mm=12.0, angle_max_deg=60.0, m_v_mm=1.0,
                                 note="smallest tau (at angle_max_deg=60) reaching ~95% recall "
                                      "on legal-only training rows; see interaction.py::DEFAULT_TAU")
    RES.parent.mkdir(parents=True, exist_ok=True)
    RES.write_text(json.dumps(out, indent=1))
    print("wrote", RES)


if __name__ == "__main__":
    main()
