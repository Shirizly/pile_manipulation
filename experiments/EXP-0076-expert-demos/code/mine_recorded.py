"""EXP-0076: mine already-recorded closed-loop episodes that qualify as expert demonstrations (CPU only, no Genesis).

Qualifying episode: (1) per-push true states + actions + a resolvable goal name; (2) area rule n*25e-6 <= 0.6*goal area;
(3) solved: first k >= 1 with in-goal mass fraction >= 0.95, truncated there; (4) legal: every kept push legal.
Legality comes from the record (`legal_ok`, or `not legal_fail`); with no legality field the kept pushes are re-checked with
Genesis/action_sampling.py::legalize_pushes when quats exist, otherwise the episode is legality_unverified (listed, not mined).
Records that claim legality are also re-checked when quats exist (sanity: a stored push must have shift 0 on recomputation);
failures are excluded as legal_recheck_fail and listed in the manifest.

Writes datasets/DS-0024-expert-demos/data/mined/<EXP>__<file stem>__<cell>__<goal>__<start>.json, unverified ones to
mined/unverified/, and mined/MANIFEST.json (rewritten after every source file). Full recompute each run; stale files removed at end.
"""
import argparse, collections, datetime, hashlib, importlib.util, json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
EXP_DIR = REPO / "experiments/EXP-0076-expert-demos"
sys.path.insert(0, str(EXP_DIR / "code"))
import tasks as T                                                    # noqa: E402  (Genesis-free helpers)
from Baselines.common.goals import named_goal_mask                   # noqa: E402

OUT = REPO / "datasets/DS-0024-expert-demos/data/mined"
UNV = OUT / "unverified"
COMMANDS = REPO / "experiments/COMMANDS.jsonl"
ROOTS = ["experiments/EXP-0039-closed-loop-metric-validity", "experiments/EXP-0051-task-success-completion-time",
         "experiments/EXP-0055-capacity-aware-value", "experiments/EXP-0057-oracle-budget-ablation",
         "experiments/EXP-0066-closed-loop-power-study", "experiments/EXP-0067-letter-stroke-width-sweep",
         "experiments/EXP-0070-time-optimal-mpc", "experiments/EXP-0071-gated-ensemble",
         "experiments/EXP-0075-closed-loop-benchmark"]
CUBE_HALF, BLADE_HL, BLADE_HW, BOX = 0.0025, 0.02, 0.001, (-0.06, 0.06)
INIT_DEFAULT = "ds0006-scatter"


def load_module(name, path):
    """Load a single file without running its package __init__ (avoids any Genesis import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


AS = load_module("action_sampling_standalone", REPO / "Genesis/action_sampling.py")
FR = load_module("frame_standalone", REPO / "model/retrieval/frame.py")


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log_command(run_id, event, **kw):
    rec = dict(ts=now(), run_id=run_id, event=event, **kw)
    with open(COMMANDS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def walk_episodes(o, out):
    if isinstance(o, dict):
        if "actions" in o and "states" in o:
            out.append(o); return
        for v in o.values(): walk_episodes(v, out)
    elif isinstance(o, list):
        for v in o: walk_episodes(v, out)


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.\-]+", "-", str(s)).strip("-")


def recheck_legal(states, quats, actions, k):
    """Recompute legality of the kept pushes with legalize_pushes. Returns (all_legal, max_shift_m) or None if no quats."""
    if quats is None:
        return None
    q = torch.as_tensor(np.asarray(quats[:k], dtype=np.float64))                  # (k, n, 4)
    yaw = FR.yaw_from_quat(q).double()                                             # (k, n)
    cube_xy = torch.as_tensor(states[:k, :, :2], dtype=torch.float64)
    acts = torch.as_tensor(actions[:k], dtype=torch.float64)
    _, shift, ok = AS.legalize_pushes(acts, cube_xy, yaw, CUBE_HALF, BLADE_HL, BLADE_HW, box=BOX)
    shift = np.asarray(shift, dtype=np.float64); ok = np.asarray(ok, dtype=bool)
    return bool(np.all((shift == 0) & ok)), float(shift.max()) if len(shift) else 0.0


def mine_episode(e):
    """Returns (status, payload). status in {'mined','unverified','rejected:<reason>'}."""
    goal = e.get("goal")
    if goal is None:
        return "rejected:no_goal", None
    try:
        mask = np.asarray(named_goal_mask(goal, 64, 64)).astype(np.uint8)
    except Exception:
        return "rejected:goal_unresolved", None
    area = T.mask_area(mask)
    raw_a = e["actions"]
    L = next((i for i, a in enumerate(raw_a) if a is None), len(raw_a))     # None = push never planned (oracle records): stop there
    states = np.asarray(e["states"][:L + 1], dtype=np.float64)
    actions = np.asarray(raw_a[:L], dtype=np.float64).reshape(-1, 4)
    if states.ndim != 3 or states.shape[-1] < 3 or actions.ndim != 2 or actions.shape[1] != 4 or len(states) != len(actions) + 1:
        return "rejected:bad_shapes", None
    n = states.shape[1]
    rho = n * T.CUBE_AREA / area if area > 0 else float("inf")
    if rho > T.RHO_MAX + 1e-12:
        return "rejected:area_rule", None
    masses = T.mass_frac(states[:, :, :3], mask)
    hits = np.nonzero(masses[1:] >= T.SUCCESS)[0]
    if not len(hits):
        return "rejected:unsolved", None
    k = int(hits[0]) + 1
    quats = e.get("quats")
    if quats is not None and len(quats) != len(states):
        quats = None
    # legality on the kept pushes
    legal_check, recheck = None, None
    fld = "legal_ok" if "legal_ok" in e else ("legal_fail" if "legal_fail" in e else None)
    if fld is not None and len(e[fld]) >= k:
        flags = np.asarray(e[fld], dtype=bool)[:k]
        if fld == "legal_fail":
            flags = ~flags
        if not flags.all():
            return "rejected:record_illegal", None
        legal_check = "record"
        rc = recheck_legal(states, quats, actions, k)
        if rc is not None:
            recheck = rc[1]
            if not rc[0]:
                return "rejected:legal_recheck_fail", {"max_shift_m": rc[1]}
    else:
        rc = recheck_legal(states, quats, actions, k)
        if rc is None:
            legal_check = "unverified"
        else:
            legal_check = "recomputed"; recheck = rc[1]
            if not rc[0]:
                return "rejected:recomputed_illegal", None
    pay = dict(k=k, n=n, rho=rho, area=area, masses=masses[:k + 1], states=states[:k + 1], actions=actions[:k],
               quats=None if quats is None else [list(np.asarray(q).round(8).tolist()) for q in quats[:k + 1]],
               legal_check=legal_check, legal_recheck_max_shift_m=recheck, goal=goal)
    pay["fallback_pushes"] = int(np.count_nonzero(np.asarray(e["legal_fallback"])[:k])) if "legal_fallback" in e else None
    return ("unverified" if legal_check == "unverified" else "mined"), pay


def build_record(e, pay, src_rel, exp, stem, cell, start, idx):
    k = pay["k"]
    acts_list = pay["actions"].tolist()
    times = [float(x) for x in T.push_times(acts_list)]
    assert len(times) == k, (len(times), k)
    seq = hashlib.sha1(json.dumps([pay["goal"], str(start), np.round(pay["actions"], 6).tolist()]).encode()).hexdigest()[:12]
    rec = dict(source_file=src_rel, exp=exp, cell=str(cell), goal_name=pay["goal"], start=start, n=pay["n"],
               init=INIT_DEFAULT, rho=pay["rho"], goal_area_m2=pay["area"], solved=True, pushes=k, time_s=float(sum(times)),
               push_times=times, masses=[float(x) for x in pay["masses"]], actions=acts_list,
               states=pay["states"].tolist(), legal_check=pay["legal_check"], legal_recheck_max_shift_m=pay["legal_recheck_max_shift_m"],
               legal_fallback_pushes=pay["fallback_pushes"], seq_hash=seq, pipeline="mined", source_episode_index=idx)
    if pay["quats"] is not None:
        rec["quats"] = pay["quats"]
    for key in ("model", "planner", "H", "push_len"):
        if key in e:
            rec[key] = e[key]
    return rec


def manifest(state, scanned, examined_total, unv_files):
    mined = state["mined"]
    pushes = [m["pushes"] for m in mined]; times = [m["time_s"] for m in mined]
    best = {}
    for m in mined:
        key = (m["goal_name"], str(m["start"]))
        if key not in best or m["time_s"] < best[key]["time_s"]:
            best[key] = m
    by_goal = collections.Counter(m["goal_name"] for m in mined)
    by_exp = collections.Counter(m["exp"] for m in mined)
    by_n = collections.Counter(m["n"] for m in mined)
    return dict(
        dataset="DS-0024-expert-demos", pipeline="mined", experiment="EXP-0076", generated=now(),
        rules=dict(area="n*25e-6 <= 0.6*goal area", success="in-goal mass fraction >= 0.95 at first k>=1 (truncated)",
                   legal="all kept pushes legal (record legal_ok / not legal_fail; recheck with legalize_pushes when quats exist)"),
        files_scanned=scanned["files"], files_with_episodes=scanned["with_eps"], episodes_examined=examined_total,
        qualifying=len(mined), unique_tasks=len(best), unique_action_sequences=len({m["seq_hash"] for m in mined}),
        median_pushes=float(np.median(pushes)) if pushes else None, median_time_s=float(np.median(times)) if times else None,
        counts_by_goal=dict(sorted(by_goal.items())), counts_by_source_exp=dict(sorted(by_exp.items())),
        counts_by_n=dict(sorted((str(k), v) for k, v in by_n.items())),
        rejections=dict(sorted(state["rej"].items())), legality_unverified=state["unverified"],
        legal_recheck_fail=state["recheck_fail"],
        sources=state["sources"],
        episodes=[dict(id=m["id"], pushes=m["pushes"], time_s=m["time_s"]) for m in sorted(mined, key=lambda m: m["id"])],
        best_per_task=[dict(id=m["id"], goal_name=m["goal_name"], start=m["start"], pushes=m["pushes"], time_s=m["time_s"])
                       for m in sorted(best.values(), key=lambda m: (m["goal_name"], str(m["start"])))],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-files", type=int, default=None, help="debug: only the first N source files")
    a = ap.parse_args()
    run_id = f"EXP-0076/mine_recorded-{int(time.time()) % 100000000}"
    cmd = "CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/alon/anaconda3/envs/pme/bin/python -u experiments/EXP-0076-expert-demos/code/mine_recorded.py"
    t0 = time.time(); log_command(run_id, "start", cmd=cmd, status="running")
    status, rc = "failed", 1
    try:
        files = []
        for r in ROOTS:
            for dp, _, fn in os.walk(REPO / r):
                files += [os.path.join(dp, f) for f in fn if f.endswith(".json") and not f.endswith("MANIFEST.json")]
        files = sorted(files)
        if a.limit_files:
            files = files[:a.limit_files]
        state = dict(mined=[], rej=collections.Counter(), unverified=[], recheck_fail=[], sources={})
        scanned = dict(files=len(files), with_eps=0)
        examined_total = 0
        written = set()
        names = set()
        UNV.mkdir(parents=True, exist_ok=True)
        for fpath in files:
            rel = os.path.relpath(fpath, REPO)
            try:
                data = json.load(open(fpath))
            except Exception as ex:
                print("SKIP unreadable", rel, ex, flush=True); state["sources"][rel] = dict(error=str(ex)); continue
            eps = []; walk_episodes(data, eps)
            if not eps:
                continue
            scanned["with_eps"] += 1
            m_exp = re.search(r"EXP-\d{4}", rel); exp = m_exp.group(0) if m_exp else "unknown"
            stem = Path(rel).stem
            src_stats = collections.Counter()
            for idx, e in enumerate(eps):
                examined_total += 1
                cell = e.get("cell", "nocell"); start = e.get("start", "nostart")
                if isinstance(start, float) and start.is_integer(): start = int(start)
                try:
                    st, pay = mine_episode(e)
                except Exception as ex:
                    st, pay = f"rejected:error:{type(ex).__name__}", None
                    print("ERROR", rel, idx, ex, flush=True)
                src_stats[st] += 1
                if st.startswith("rejected"):
                    state["rej"][st.split(":", 1)[1]] += 1
                    if st == "rejected:legal_recheck_fail":
                        state["recheck_fail"].append(dict(source_file=rel, index=idx, cell=str(cell), goal=e.get("goal"), start=start, max_shift_m=pay["max_shift_m"]))
                    continue
                rec = build_record(e, pay, rel, exp, stem, cell, start, idx)
                base = f"{exp}__{safe(stem)}__{safe(cell)}__{safe(rec['goal_name'])}__{safe(start)}"
                name = base
                if name in names:
                    name = f"{base}__e{idx}"
                names.add(name)
                rec["id"] = name
                if st == "mined":
                    write_json(OUT / f"{name}.json", rec); written.add(OUT / f"{name}.json")
                    state["mined"].append(dict(id=name, pushes=rec["pushes"], time_s=rec["time_s"], goal_name=rec["goal_name"],
                                               start=start, exp=exp, seq_hash=rec["seq_hash"], n=rec["n"]))
                else:
                    write_json(UNV / f"{name}.json", rec); written.add(UNV / f"{name}.json")
                    state["unverified"].append(dict(id=name, source_file=rel, cell=str(cell), goal_name=rec["goal_name"], start=start,
                                                    pushes=rec["pushes"], time_s=rec["time_s"]))
            state["sources"][rel] = dict(exp=exp, episodes=len(eps), **dict(src_stats))
            write_json(OUT / "MANIFEST.json", manifest(state, scanned, examined_total, None))
            print(f"{rel}: eps={len(eps)} {dict(src_stats)} mined_total={len(state['mined'])}", flush=True)
        # remove stale files from earlier runs
        removed = 0
        for d in ((OUT, UNV) if not a.limit_files else ()):          # debug runs (--limit-files) must not delete full outputs
            for p in d.glob("*.json"):
                if p.name != "MANIFEST.json" and p not in written:
                    p.unlink(); removed += 1
        man = manifest(state, scanned, examined_total, None)
        write_json(OUT / "MANIFEST.json", man)
        print(f"DONE files={scanned} examined={examined_total} mined={len(state['mined'])} unverified={len(state['unverified'])} "
              f"tasks={man['unique_tasks']} stale_removed={removed}", flush=True)
        print("rejections", dict(state["rej"]), flush=True)
        status, rc = "ok", 0
    finally:
        log_command(run_id, "end", cmd=cmd, status=status, returncode=rc, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
