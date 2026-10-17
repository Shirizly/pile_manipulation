"""EXP-0075 T0a: CLEAN-PC per-candidate planning-cost table (RAL run plan 2026-10-15). One process = one (row-group, repetition); run each group in 3 separate processes for the noise floor.
  python -u code/time_clean.py --group <lf|vanilla64|zoom64|vanilla128|zoom128|ens128|legal> --rep <r>   -> results/timing_clean_raw/<group>_p<r>.json (atomic)
  python -u code/time_clean.py --aggregate                                                             -> results/timing_clean.json / .md (atomic)
Rows per model (same objects as the harness: bs_lib.get_model / wide_planner.SeqObjective, letter_T_w20 on DS slate 40, mass balance on except LF):
  fwd_harness_H{1,2,4}: obj.cost on a 256-candidate batch (the plan_seq n_pool / pop batch) with SeqObjective's default chunk 128 (closed_loop harness), ms per candidate.
  fwd_bs_H{1,2,4}: obj.cost on 2,048 candidates with chunk 256 (the 2026-10-08 timing_units.json / bs_lib.unit convention), ms per candidate.
  gd_step_H{1,2,4}_n8: one Adam step through obj.cost_grad on gd_k = 8 sequences (plan_seq's GD path; bs_lib uses the n8 step), ms per step.
Legaliser group: Genesis/action_sampling.legalize_pushes_torch on 1,280 pile-aware pushes with one shared state (wide_planner.make_legal_scorer path), and the CPU legalize_pushes on 200 pushes.
Each row: warm-up, then R timed repeats between torch.cuda.synchronize(); the in-process value is the median. The device of every tensor fed to a model forward is recorded by a hook and asserted == cuda."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"; RAW = OUT / "timing_clean_raw"
GROUPS = ["lf", "vanilla64", "zoom64", "vanilla128", "zoom128", "ens128"]
OLD = {"lf": "LF (switched linear 64)", "vanilla64": "vanilla64 (f4)", "zoom64": "zoom64 (f8)", "vanilla128": "vanilla128", "zoom128": "zoom128", "ens128": "ens128 (zoom+vanilla)"}
BUDGETS = [0.1, 0.3, 0.5, 1.0, 3.0]


def dump(obj, path):
    tmp = Path(str(path) + ".tmp"); json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)


def sync():
    torch.cuda.synchronize()


def timed(fn, R, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(R):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(t) for t in ts]


def run_model(key, rep, gd_only=False):
    sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code"))
    import bs_lib as bl
    from intensive import proj
    torch.manual_seed(1000 + rep); g, s = bl.TASKS[0]; S0 = bl.START(s); model = bl.get_model(key); obj = bl.make_obj(model, S0, g)
    devs = set()
    if key == "lf":
        m = model.members[0]; ps = m._ps
        def rec(edges, ops, cur, *a, **k):
            devs.add(str(cur.device)); return ps(edges, ops, cur, *a, **k)
        m._ps = rec
    else:
        for m in model.members:
            m.net.register_forward_pre_hook(lambda mod, inp: devs.update(str(t.device) for t in inp if torch.is_tensor(t)))
    first = bl.propose_from_occ(model.occ0[0], 20000); r = dict(model=key, rep=rep, pid=os.getpid(), loadavg_start=os.getloadavg(), first_pool=len(first), rows={})
    def put(name, ms, raw, **kw):
        assert devs and all(d.startswith("cuda") for d in devs), (name, devs)
        r["rows"][name] = dict(ms=ms, raw_s=raw, device=sorted(devs), **kw); devs.clear(); print(key, rep, name, round(ms, 4), flush=True)
    for H in (1, 2, 4):
        if gd_only:      # noise check of the GD step: 10 warm-up + 100 timed steps (vs 3 + 20 in the main rows), row prefix gdlong_
            x = bl.pool_seqs(first, 8, H).clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=2e-3)
            def step():
                with torch.enable_grad():
                    opt.zero_grad(); obj.cost_grad(proj(x.reshape(-1, 4)).reshape(-1, H, 4)).sum().backward(); opt.step()
            med, raw = timed(step, R=100, warm=10); put(f"gdlong_step_H{H}_n8", med * 1000, raw, batch=8, unit="ms per step of 8 sequences", input_device=str(x.device)); continue
        seqs = bl.pool_seqs(first, 256, H); obj.chunk = 128
        med, raw = timed(lambda: obj.cost(seqs), R=10); put(f"fwd_harness_H{H}", med / 256 * 1000, raw, batch=256, chunk=128, unit="ms per candidate", input_device=str(seqs.device))
        seqs = bl.pool_seqs(first, 2048, H); obj.chunk = 256
        med, raw = timed(lambda: obj.cost(seqs), R=3, warm=1); put(f"fwd_bs_H{H}", med / 2048 * 1000, raw, batch=2048, chunk=256, unit="ms per candidate", input_device=str(seqs.device))
        x = bl.pool_seqs(first, 8, H).clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=2e-3)
        def step():
            with torch.enable_grad():
                opt.zero_grad(); obj.cost_grad(proj(x.reshape(-1, 4)).reshape(-1, H, 4)).sum().backward(); opt.step()
        med, raw = timed(step, R=20, warm=3); put(f"gd_step_H{H}_n8", med * 1000, raw, batch=8, unit="ms per step of 8 sequences", input_device=str(x.device))
    r["loadavg_end"] = os.getloadavg(); r["gpu"] = torch.cuda.get_device_name(0); return r


def run_legal(rep):
    sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code"))
    import bs_lib as bl
    import wide_planner as wp
    from Genesis.action_sampling import legalize_pushes
    from model.retrieval.frame import yaw_from_quat
    from simple_mpc.adapters import OCC_BOUNDS
    from simple_mpc.learned_mpc import XY_MARGIN
    torch.manual_seed(2000 + rep); g, s = bl.TASKS[0]; S0 = bl.START(s); model = bl.get_model("vanilla64"); model.begin(S0)
    acts = bl.propose_from_occ(model.occ0[0], 20000); acts = acts[torch.randint(0, len(acts), (1280,), device=acts.device)]
    xy, yaw = S0[:, :2], yaw_from_quat(S0[:, 3:7]); scorer = wp.make_legal_scorer(xy, yaw); r = dict(model="legal", rep=rep, pid=os.getpid(), loadavg_start=os.getloadavg(), rows={})
    for n in (256, 1280):
        a = acts[:n].to(wp.DEV)
        out = scorer(a); dev = str(out[0].device); assert dev.startswith("cuda"), dev
        med, raw = timed(lambda: scorer(a), R=10); r["rows"][f"legal_gpu_n{n}"] = dict(ms=med / n * 1000, raw_s=raw, batch=n, device=[dev], unit="ms per push", frac_ok=float(out[2].float().mean()), frac_shifted=float((out[1] > 0).float().mean()))
        print("legal", rep, n, round(med / n * 1000, 4), flush=True)
    n = 200; a = acts[:n].cpu().float(); box = (OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN)
    cxy, cyaw = xy[None].expand(n, -1, -1).cpu().float(), yaw[None].expand(n, -1).cpu().float()
    t0 = time.perf_counter(); out = legalize_pushes(a, cxy, cyaw, 0.0025, 0.02, 0.001, box=box); t = time.perf_counter() - t0
    r["rows"]["legal_cpu_n200"] = dict(ms=t / n * 1000, raw_s=[t], batch=n, device=[str(out[0].device)], unit="ms per push"); print("legal cpu", rep, round(t / n * 1000, 3), flush=True)
    r["loadavg_end"] = os.getloadavg(); return r


def run_overhead(rep):
    """Per-decision overheads of the best-planner parts at default sizes (coordinator scope addition 2026-10-15): fetch_candidates, ot_candidates (uniform_layout slots),
    make_legal_scorer + legalising the 1,280 first pushes of a 256 + 4 x 256 CEM decision, and end-to-end plan_seq (cem, 1,280 evals, gd 0) in modes base / fl (fetch n_extra 64 + legal) /
    ot (ot + fetch + legal, pool_mode ot) for vanilla64 and ens128 at H = 1, 2. Bank = 4,096 propose_from_occ rows (the harness uses the Genesis pile-aware bank; same size)."""
    sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code"))
    import bs_lib as bl
    import wide_planner as wp
    from batched_closed_loop import goal_mask
    from simple_mpc.value_functions import uniform_layout
    from model.retrieval.frame import yaw_from_quat
    torch.manual_seed(3000 + rep); g, s = bl.TASKS[0]; S0 = bl.START(s); mask = torch.from_numpy(goal_mask(g)); slots = torch.from_numpy(uniform_layout(mask.numpy(), 20)).float()
    xy = S0[:, :2].to(wp.DEV); yaw = yaw_from_quat(S0[:, 3:7]); r = dict(model="overhead", rep=rep, pid=os.getpid(), loadavg_start=os.getloadavg(), rows={})
    def put(name, med, raw, **kw):
        r["rows"][name] = dict(ms=med * 1000, raw_s=raw, device=["cuda:0"], unit="ms per decision", **kw); print("overhead", rep, name, round(med * 1000, 3), flush=True)
    med, raw = timed(lambda: wp.fetch_candidates(xy, mask), R=10); put("fetch_candidates", med, raw, n_rows=len(wp.fetch_candidates(xy, mask)))
    med, raw = timed(lambda: wp.ot_candidates(xy, mask, slots), R=10); put("ot_candidates", med, raw, n_rows=len(wp.ot_candidates(xy, mask, slots)[0]))
    model = bl.get_model("vanilla64"); model.begin(S0); bank = bl.propose_from_occ(model.occ0[0], 20000); bank = bank[torch.randint(0, len(bank), (4096,), device=bank.device)]
    def leg():
        sc = wp.make_legal_scorer(S0[:, :2], yaw)
        for i in range(5): sc(bank[i * 256:(i + 1) * 256])
    med, raw = timed(leg, R=10); put("legal_scorer_1280", med, raw)
    for key in ("vanilla64", "ens128"):
        model = bl.get_model(key); obj = bl.make_obj(model, S0, g); obj.chunk = 128
        for H in (1, 2):
            for mode in ("base", "fl", "ot"):
                def dec():
                    kw = {}
                    if mode in ("fl", "ot"):
                        kw["extra"] = wp.fetch_candidates(xy, mask); kw["n_extra"] = 64; kw["legal"] = wp.make_legal_scorer(S0[:, :2], yaw)
                    if mode == "ot":
                        kw["ot"] = wp.ot_candidates(xy, mask, slots); kw["cube_xy"] = xy; kw["pool_mode"] = "ot"
                    return wp.plan_seq(obj, bank, "cem", H, **kw)
                out = dec(); med, raw = timed(dec, R=5); put(f"plan_seq_{key}_H{H}_{mode}", med, raw, n_evals=int(out["n_evals"]))
    r["loadavg_end"] = os.getloadavg(); return r


def counts(c_s, g_s, B):
    """bs_lib.counts convention: cem n_eval = max(40, int(B / c)); cem_gd n_eval = max(40, int(0.5 B / c)), gd_steps = max(1, int(0.5 B / g)) (g = n8 step)."""
    return dict(cem=dict(n_eval=max(40, int(B / c_s)), gd_steps=0), cem_gd=dict(n_eval=max(40, int(0.5 * B / c_s)), gd_steps=max(1, int(0.5 * B / g_s))))


HARNESS_TIMING_KEY = {"ens2bal": "ens128", "zoom128bal": "zoom128", "vanilla128bal": "vanilla128", "vanilla64bal": "vanilla64", "zoom64": "zoom64", "lf": "lf",
                      "vanilla128_s1": "vanilla128", "zoom128_s1": "zoom128", "ens128_s1": "ens128"}   # closed_loop.MODELS key -> timing_clean row group (s1 seeds: same architecture as s0)


def budget_counts(model, H, budget_s, planner="cem", legal=False, iters=4, table=None):
    """Closed-loop cell key `budget_s` (T0b, 2026-10-15) -> deterministic evaluation counts from the CLEAN table (results/timing_clean.json):
    c = per-candidate cost of the harness forward row H{H}_harness (H{H}_harness_legal when legal_score is on: + legal_scorer_1280 / 1280 per candidate),
    g = GD step (gdlong n8 row); n_eval / gd_steps by `counts` (bs_lib.counts convention: cem -> all of B on forward passes, cem_gd -> half on forward, half on GD).
    plan_seq split: n_pool = pop = n_eval // (iters + 1) (>= 8), so n_pool + iters * pop <= n_eval; rank -> rank_n = n_eval. model: closed_loop MODELS key or a row group.
    Returns dict(plan_kw for plan_seq, and the provenance record: table created stamp + sha256[:12], row, c_ms, gd_ms, n_eval, gd_steps)."""
    import hashlib
    p = Path(table) if table else OUT / "timing_clean.json"; raw = p.read_bytes(); T = json.loads(raw)
    grp = HARNESS_TIMING_KEY.get(model, model); row = f"H{H}_harness" + ("_legal" if legal else ""); r = T["n_eval"][grp][row]
    pl = {"cem+gd": "cem_gd"}.get(planner, planner); cn = counts(r["c_ms"] / 1000, r["gd_ms"] / 1000, float(budget_s))["cem_gd" if pl == "cem_gd" else "cem"]
    if pl == "rank":
        kw = dict(rank_n=cn["n_eval"])
    else:
        n0 = max(8, cn["n_eval"] // (iters + 1)); kw = dict(n_pool=n0, pop=n0, iters=iters)
        if pl == "cem_gd":
            kw["gd_steps"] = cn["gd_steps"]
    return dict(plan_kw=kw, budget_s=float(budget_s), planner=pl, timing_group=grp, timing_row=row, c_ms=r["c_ms"], gd_ms=r["gd_ms"], n_eval=cn["n_eval"], gd_steps=cn["gd_steps"],
                timing_table=str(p.name), timing_created=T["created"], timing_sha256=hashlib.sha256(raw).hexdigest()[:12])


def aggregate():
    old1 = json.load(open(OUT / "timing_units.json")); old2 = json.load(open(OUT / "timing_units2.json"))
    T = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), protocol=__doc__, gpu=None, rows={}, n_eval={}, flags=[], loadavg=[])
    for f in sorted(RAW.glob("*_p*.json")):
        d = json.load(open(f)); d["model"] = d["model"] if not f.name.startswith("gd_") else d["model"]; T["gpu"] = d.get("gpu", T["gpu"]); T["loadavg"].append(dict(file=f.name, start=d["loadavg_start"], end=d.get("loadavg_end")))
        for name, row in d["rows"].items():
            e = T["rows"].setdefault(d["model"], {}).setdefault(name, dict(per_process_ms=[], device=set(), unit=row["unit"]))
            e["per_process_ms"].append(row["ms"]); e["device"].update(row["device"])
    for m, rows in T["rows"].items():
        for name, e in rows.items():
            v = np.array(e["per_process_ms"]); e["device"] = sorted(e["device"]); e["n_processes"] = len(v); e["mean_ms"] = float(v.mean()); e["spread_rel"] = float((v.max() - v.min()) / v.mean()) if len(v) > 1 else None
            e["cv"] = float(v.std(ddof=1) / v.mean()) if len(v) > 1 else None
            if e["spread_rel"] is not None and e["spread_rel"] > 0.30: T["flags"].append(f"SPREAD>30%: {m} {name} {e['spread_rel']:.0%}")
            old = None
            if m in OLD:
                ok = OLD[m]
                if name.startswith("fwd_bs_H"): old = old1.get(ok, {}).get(f"fwd_H{name[-1]}_chunk256_ms_per_seq")
                if name.startswith("gd_step_H") and name[9] in "14": old = old2.get(ok, {}).get(f"gd_step_H{name[9]}_n8_ms")
            if m == "legal" and name == "legal_cpu_n200": old = 2.2
            if old:
                e["old_2026_10_08_ms"] = old; e["delta_rel"] = e["mean_ms"] / old - 1
                if abs(e["delta_rel"]) > 0.20: T["flags"].append(f"DELTA>20% vs 2026-10-08: {m} {name} {old:.4g} -> {e['mean_ms']:.4g} ms ({e['delta_rel']:+.0%})")
    for m in GROUPS:
        rows = T["rows"].get(m, {})
        for H in (1, 2, 4):
            gst = rows.get(f"gdlong_step_H{H}_n8") or rows.get(f"gd_step_H{H}_n8")      # GD step: the 100-step rows (lower spread) when present
            leg = T["rows"].get("overhead", {}).get("legal_scorer_1280")
            for conv in ("harness", "bs", "harness_legal"):
                c = rows.get(f"fwd_{conv.replace('_legal', '')}_H{H}")
                if c and gst and (conv != "harness_legal" or leg):
                    cms = c["mean_ms"] + (leg["mean_ms"] / 1280 if conv == "harness_legal" else 0.0)   # harness_legal: + legalising every candidate's first push (legal_score / fl / ot cells)
                    T["n_eval"].setdefault(m, {}).setdefault(f"H{H}_{conv}", dict(c_ms=cms, gd_ms=gst["mean_ms"], gd_row="gdlong" if f"gdlong_step_H{H}_n8" in rows else "gd", **{str(B): counts(cms / 1000, gst["mean_ms"] / 1000, B) for B in BUDGETS}))
    dump(T, OUT / "timing_clean.json")
    L = [f"# EXP-0075 CLEAN timing table ({T['created']}, {T['gpu']}; code/time_clean.py; each row = 3 separate processes, value = mean of per-process medians, spread = (max-min)/mean)", "",
         "Forward = ms per candidate through SeqObjective.cost (rollout + paste + mass balance + objective). harness = 256-candidate batch, chunk 128 (plan_seq / closed_loop); bs = 2,048, chunk 256 (2026-10-08 convention). GD = ms per Adam step on 8 sequences.", "",
         "| model | row | mean ms | spread | device | 2026-10-08 | delta |", "|---|---|---|---|---|---|---|"]
    for m, rows in T["rows"].items():
        for name, e in rows.items():
            L.append(f"| {m} | {name} | {e['mean_ms']:.4g} | {e['spread_rel']:.0%} | {','.join(e['device'])} | {e.get('old_2026_10_08_ms', '')} | {'%+.0f%%' % (100 * e['delta_rel']) if 'delta_rel' in e else ''} |" if e["spread_rel"] is not None else f"| {m} | {name} | {e['mean_ms']:.4g} | n=1 | {','.join(e['device'])} | | |")
    L += ["", "## Implied evaluation counts (bs_lib.counts convention, harness forward row [+legal = + legal_scorer_1280/1280 per candidate], GD = gdlong 100-step row): CEM n_eval; CEM+GD n_eval / gd_steps", "", "| model | H | " + " | ".join(f"{B} s" for B in BUDGETS) + " |", "|---|---|" + "---|" * len(BUDGETS)]
    for m, d in T["n_eval"].items():
        for H, k in [(H, f"H{H}_{v}") for v in ("harness", "harness_legal") for H in (1, 2, 4)]:
            if k in d: L.append(f"| {m} | {H}{' +legal' if 'legal' in k else ''} | " + " | ".join(f"{d[k][str(B)]['cem']['n_eval']} ; {d[k][str(B)]['cem_gd']['n_eval']}/{d[k][str(B)]['cem_gd']['gd_steps']}" for B in BUDGETS) + " |")
    L += ["", "## Flags", ""] + [f"- {f}" for f in T["flags"]] + ["", "## Load average per process (1/5/15 min, start -> end)", ""] + [f"- {x['file']}: {x['start']} -> {x['end']}" for x in T["loadavg"]]
    tmp = OUT / "timing_clean.md.tmp"; open(tmp, "w").write("\n".join(L) + "\n"); os.replace(tmp, OUT / "timing_clean.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--group"); ap.add_argument("--rep", type=int, default=0); ap.add_argument("--aggregate", action="store_true"); a = ap.parse_args()
    if a.aggregate: aggregate(); sys.exit(0)
    RAW.mkdir(exist_ok=True); r = run_legal(a.rep) if a.group == "legal" else run_overhead(a.rep) if a.group == "overhead" else run_model(a.group[3:], a.rep, gd_only=True) if a.group.startswith("gd_") else run_model(a.group, a.rep); dump(r, RAW / f"{a.group}_p{a.rep}.json")
