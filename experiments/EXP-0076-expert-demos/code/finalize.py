"""EXP-0076 finalisation: datasets/DS-0024-expert-demos/data/INDEX.json = per generated scenario the FASTEST solved demo across pipelines
(expert / feasible) with pipeline, file, pushes, time, final mass, rho, and replay flags (results/replay_*_n*.json: open-loop replay still
>= 0.95), plus the fastest FEASIBLE demo separately; mined demos indexed from data/mined/MANIFEST.json best_per_task. Rewrites atomically."""
import json, glob, os, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; EXP = HERE.parent; REPO = EXP.parents[1]; DATA = REPO / "datasets/DS-0024-expert-demos/data"


def main():
    rep = {}
    for f in sorted(glob.glob(str(EXP / "results/replay_*_n*.json"))):
        pipe = Path(f).stem.split("_")[1]
        for r in json.load(open(f)):
            rep.setdefault((pipe, r["id"]), []).append(dict(rep_mass=r["rep_final_mass"], rep_solved=r["rep_solved"], err_mm=r["final_pos_err_mm_mean"], src=Path(f).name, pushes=r["pushes"], rec_mass=r["rec_final_mass"]))
    idx = {}
    for pipe in ("expert", "feasible"):
        for f in sorted(glob.glob(str(DATA / pipe / "*.json"))):
            d = json.load(open(f)); e = dict(pipeline=pipe, file=os.path.relpath(f, REPO), pushes=d["pushes"], time_s=round(d["time_s"], 3), final_mass=round(d["final_mass"], 4),
                                             n=d["n"], init=d["init"], kind=d["kind"], rho=round(d["goal"].get("rho", 0), 3), tag=d["method"]["tag"],
                                             replays=[r for r in rep.get((pipe, d["id"]), []) if r["pushes"] == d["pushes"] and abs(r["rec_mass"] - d["final_mass"]) < 1e-6])
            cur = idx.setdefault(d["id"], {})
            if "best" not in cur or e["time_s"] < cur["best"]["time_s"]:
                cur["best"] = e
            if pipe == "feasible":
                cur["best_feasible"] = e
    for f in sorted(glob.glob(str(DATA / "expert_robust" / "*.json"))):     # margin demos (accepted only at mass >= 0.97), listed separately
        d = json.load(open(f)); idx.setdefault(d["id"], {})["robust"] = dict(file=os.path.relpath(f, REPO), pushes=d["pushes"], time_s=round(d["time_s"], 3),
                                                                             final_mass=round(d["final_mass"], 4), tag=d["method"]["tag"])
    idx = {k: v for k, v in idx.items() if "best" in v or "robust" in v}
    for v in idx.values():
        if "best" not in v:
            continue
        rs = v["best"]["replays"]; v["best"]["replay_ok"] = (all(r["rep_solved"] for r in rs) if rs else None)
    mined = json.load(open(DATA / "mined/MANIFEST.json")).get("best_per_task", []) if (DATA / "mined/MANIFEST.json").exists() else []
    out = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), success="in-goal mass >= 0.95 (occ_for_scoring)", time="EXP-0070 action_time physical seconds",
               scenarios=idx, mined_best_per_task=mined,
               counts=dict(scenarios_with_demo=sum("best" in v for v in idx.values()), best_from_expert=sum(v.get("best", {}).get("pipeline") == "expert" for v in idx.values()),
                           robust_margin097=sum("robust" in v for v in idx.values()),
                           with_feasible_demo=sum("best_feasible" in v for v in idx.values()),
                           replay_checked=sum(v.get("best", {}).get("replay_ok") is not None for v in idx.values()),
                           replay_ok=sum(bool(v.get("best", {}).get("replay_ok")) for v in idx.values()), mined_tasks=len(mined)))
    tmp = DATA / "INDEX.json.tmp"; tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, DATA / "INDEX.json"); print(json.dumps(out["counts"]))


if __name__ == "__main__":
    main()
