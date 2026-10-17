"""EXP-0076 summaries.
  python summarize.py demos            -> results/demo_counts.md: solved demos per (n, init, goal kind) x pipeline (expert / feasible / mined), best-overall
  python summarize.py arms R1 R2 ...   -> paired arm comparison over the scenarios every run finished (results/runs/<R>.json)"""
import json, sys, glob, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; EXP = HERE.parent; REPO = EXP.parents[1]
DATA = REPO / "datasets/DS-0024-expert-demos/data"


def load_dir(p):
    out = {}
    for f in sorted(glob.glob(str(p / "*.json"))):
        if f.endswith("MANIFEST.json"):
            continue
        d = json.load(open(f)); out[d["id"]] = d
    return out


def demos():
    ex, fe, mi = load_dir(DATA / "expert"), load_dir(DATA / "feasible"), load_dir(DATA / "mined")
    sc = {s["id"]: s for s in json.load(open(EXP / "results/scenarios.json"))["scenarios"]}
    best = {}
    for src, D in (("expert", ex), ("feasible", fe)):
        for i, d in D.items():
            if i not in best or d["time_s"] < best[i]["time_s"]:
                best[i] = dict(d, src=src)
    L = ["# Demonstration counts (DS-0024)", "", f"generated scenarios with a solved demo: {len(best)} of {len(sc)} generated "
         f"(expert {len(ex)}, feasible {len(fe)}); mined from earlier records (separate, n20 DS-0006 scatter starts): {len(mi)}", "",
         "| n | init | goal | scenarios with demo | from expert | from feasible | best pushes median [min-max] | best time s median | <10 pushes |", "|---|---|---|---|---|---|---|---|---|"]
    strata = collections.defaultdict(list)
    for i, d in best.items():
        strata[(d["n"], d["init"], d["kind"])].append(d)
    for key in sorted(set((s["n"], s["init"], s["kind"]) for s in sc.values())):
        v = strata.get(key, [])
        if v:
            P = np.array([d["pushes"] for d in v]); Ts = np.array([d["time_s"] for d in v])
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(v)} | {sum(d['src'] == 'expert' for d in v)} | {sum(d['src'] == 'feasible' for d in v)} | "
                     f"{np.median(P):.0f} [{P.min()}-{P.max()}] | {np.median(Ts):.1f} | {int((P < 10).sum())} |")
        else:
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | 0 | 0 | 0 | - | - | - |")
    if mi:
        L += ["", "## Mined (earlier EXP records; goal = named repo goal; init ds0006-scatter unless stated)", "", "| goal | n demos | unique tasks | pushes median | time s median |", "|---|---|---|---|---|"]
        g = collections.defaultdict(list)
        for d in mi.values():
            g[d.get("goal_name")].append(d)
        for k in sorted(g, key=str):
            v = g[k]; L.append(f"| {k} | {len(v)} | {len(set(d.get('start') for d in v))} | {np.median([d['pushes'] for d in v]):.0f} | {np.median([d['time_s'] for d in v]):.1f} |")
    att = [json.loads(l) for l in open(DATA / "attempts.jsonl")] if (DATA / "attempts.jsonl").exists() else []
    L += ["", f"attempts logged: {len(att)} (solved {sum(a['solved'] for a in att)})"]
    (EXP / "results/demo_counts.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


def arms(tags):
    R = {t: json.load(open(EXP / f"results/runs/{t}.json"))["scenarios"] for t in tags}
    ids = sorted(set.intersection(*[set(i for i, r in v.items() if r.get("complete")) for v in R.values()]))
    L = [f"# arms over {len(ids)} common scenarios", "", "| arm | solved | pushes (solved) mean | time s (solved) mean | final mass mean | sims / scenario | wall s / scenario | plan s / decision |", "|---|---|---|---|---|---|---|---|"]
    for t, v in R.items():
        rs = [v[i] for i in ids]; so = [r for r in rs if r["solved"]]
        ps = [x for r in rs for x in r.get("plan_s", [])]
        L.append(f"| {t} | {len(so)}/{len(rs)} | {np.mean([r['pushes'] for r in so]) if so else float('nan'):.2f} | {np.mean([r['time_s'] for r in so]) if so else float('nan'):.2f} | "
                 f"{np.mean([r['final_mass'] for r in rs]):.3f} | {np.mean([r['n_sims'] for r in rs]):.0f} | {np.mean([r['wall_s'] for r in rs]):.0f} | {np.mean(ps) if ps else float('nan'):.2f} |")
    if len(tags) >= 2:
        L += ["", "paired on scenarios solved by both (b - a): pushes, time s"]
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                both = [s for s in ids if R[a][s]["solved"] and R[b][s]["solved"]]
                if both:
                    dp = np.array([R[b][s]["pushes"] - R[a][s]["pushes"] for s in both]); dt = np.array([R[b][s]["time_s"] - R[a][s]["time_s"] for s in both])
                    rng = np.random.default_rng(0); bs = [dt[rng.integers(0, len(dt), len(dt))].mean() for _ in range(2000)]
                    L.append(f"- {b} - {a}: n {len(both)}, pushes {dp.mean():+.2f}, time {dt.mean():+.2f} s [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}], "
                             f"only-a solved {sum(R[a][s]['solved'] and not R[b][s]['solved'] for s in ids)}, only-b solved {sum(R[b][s]['solved'] and not R[a][s]['solved'] for s in ids)}")
    L += ["", "| scenario | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for s in ids:
        L.append(f"| {s} (rho {R[tags[0]][s]['goal'].get('rho', 0):.2f}) | " + " | ".join(
            (f"{R[t][s]['pushes']}p {R[t][s]['time_s']:.1f}s" if R[t][s]["solved"] else f"x {R[t][s]['final_mass']:.2f}") for t in tags) + " |")
    out = EXP / f"results/arms_{'_'.join(tags)}.md"; out.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    demos() if sys.argv[1] == "demos" else arms(sys.argv[2:])
