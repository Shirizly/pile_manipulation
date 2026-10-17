"""EXP-0076 demonstration store (DS-0024): the best (fastest physical time) SOLVED trajectory per (scenario, pipeline),
data/<pipeline>/<scenario id>.json, replaced only by a strictly faster one; every attempt (solved or not) is appended to data/attempts.jsonl."""
import json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "datasets/DS-0024-expert-demos/data"


def _atomic(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True); tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(obj)); os.replace(tmp, p)


def offer(rec, method):
    """rec: beam.Scenario.record(); method: dict incl. 'pipeline' (expert | feasible) and 'tag'. Returns True if stored as the new best."""
    ROOT.mkdir(parents=True, exist_ok=True); stored = False
    if rec["solved"]:
        p = ROOT / method["pipeline"] / f"{rec['id']}.json"
        old = json.loads(p.read_text())["time_s"] if p.exists() else float("inf")
        if rec["time_s"] < old - 1e-9:
            _atomic(p, dict(rec, method=method, stored_at=time.strftime("%Y-%m-%dT%H:%M:%S"))); stored = True
    line = dict(id=rec["id"], n=rec["n"], init=rec["init"], kind=rec["kind"], rho=rec["goal"].get("rho"), pipeline=method["pipeline"], tag=method["tag"],
                solved=rec["solved"], pushes=rec["pushes"], time_s=rec["time_s"], final_mass=rec["final_mass"], wall_s=rec["wall_s"], n_sims=rec["n_sims"],
                plan_s_mean=(sum(rec["plan_s"]) / len(rec["plan_s"])) if rec.get("plan_s") else None, stored=stored, method=method,
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(ROOT / "attempts.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    return stored
