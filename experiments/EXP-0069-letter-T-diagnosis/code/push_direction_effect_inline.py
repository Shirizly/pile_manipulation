"""EXP-0069 RUN-0001 step 3 (RECONSTRUCTED from the inline heredoc that produced results/push_direction_effect.json;
identical code, saved so the output is regenerable). Per-push gain in in-goal mass (and, for letter_T, crossbar / stem
mass) by push-direction class (along x / along y / diagonal, 22.5 deg bands) per family and letter, EXP-0066 cells.
Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -u experiments/EXP-0069-letter-T-diagnosis/code/push_direction_effect_inline.py
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_T import load_episodes, R66, CELLS66, LETTERS4, push_geom, R  # noqa: E402

out = {}
for fam in CELLS66:
    E = []
    for t in CELLS66[fam]:
        f = R66 / f"{t}.json"
        if f.exists():
            E += load_episodes(f)
    for g in LETTERS4:
        Eg = [e for e in E if e["goal"] == g]; rows = []
        for e in Eg:
            ang, sw, ew, L, a = push_geom(e["actions"]); mf = np.array(e["mf"]); d = np.diff(mf)
            cls = np.where(np.abs(np.cos(np.radians(ang))) > np.cos(np.radians(22.5)), "x", np.where(np.abs(np.sin(np.radians(ang))) > np.cos(np.radians(22.5)), "y", "diag"))
            for k in range(16):
                r = dict(cls=cls[k], d=d[k])
                if g == "letter_T":
                    r["dbar"] = e["mf_bar"][k + 1] - e["mf_bar"][k]; r["dstem"] = e["mf_stem"][k + 1] - e["mf_stem"][k]
                rows.append(r)
        res = {}
        for c in ("x", "y", "diag"):
            rr = [r for r in rows if r["cls"] == c]
            if rr:
                res[c] = dict(n=len(rr), frac=len(rr) / len(rows), d_ingoal=float(np.mean([r["d"] for r in rr])))
                if g == "letter_T":
                    res[c]["d_bar"] = float(np.mean([r["dbar"] for r in rr])); res[c]["d_stem"] = float(np.mean([r["dstem"] for r in rr]))
        out[f"{fam}|{g}"] = res
        print(fam, g, " | ".join(f"{c}: n {v['n']} ({v['frac']:.2f}) d_ingoal {v['d_ingoal'] * 1000:+.1f}e-3" + (f" d_bar {v['d_bar'] * 1000:+.1f}e-3 d_stem {v['d_stem'] * 1000:+.1f}e-3" if "d_bar" in v else "") for c, v in res.items()))
json.dump(out, open(R / "push_direction_effect.json", "w"), indent=1)
