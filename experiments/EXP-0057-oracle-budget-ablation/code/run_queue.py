"""EXP-0057 resumable queue runner. Each job = one ablation cell (all tasks), run via
scripts/run_probe.py -> code/oracle.py (which itself checkpoints per push and resumes).
`skip` if the cell's results JSON already has step >= --steps for the configured episode
count. Appends run_probe's runs/COMMANDS.jsonl lines into experiments/COMMANDS.jsonl too.
Usage: python code/queue.py main   (or: extra)
"""
import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ORACLE = REPO / "experiments/EXP-0050-oracle-ceiling/code/oracle.py"
RESULTS = EXP / "results"
GOALS = ["letter_O", "letter_T", "letter_S", "letter_X", "letter_L", "letter_I", "two_squares", "quadrant_0"]
STARTS = [40, 41]
STEPS = 20
N_ENVS = 128
PY = "/home/alon/anaconda3/envs/pme/bin/python"


def cell(tag, cem_pop, cem_iters, extra=None, goals=GOALS, starts=STARTS, steps=STEPS):
    elite = max(1, round(0.25 * cem_pop))
    args = ["--mode", "cem", "--tag", tag, "--n-envs", str(N_ENVS), "--cem-pop", str(cem_pop),
            "--cem-iters", str(cem_iters), "--cem-elite", str(elite), "--steps", str(steps),
            "--goals", *goals, "--starts", *[str(s) for s in starts], "--out-dir", str(RESULTS),
            # identical random streams per (task, push, iteration) at every cell; stop once solved
            "--seed-base", "0", "--stop-solved"]
    if extra:
        args += extra
    return dict(tag=tag, args=args, n_tasks=len(goals) * len(starts), steps=steps)


# split ablation (A): (tag_suffix, cem_pop, cem_iters), 256 sims/push at every cell -- the split
# ONLY varies P vs I, so every cell costs the SAME wall time (measured: 16 episodes x 20 pushes
# x 256 sims/episode / 5.989 sims/s at n_envs=128 = ~3.8h/cell, identical across splits since
# total simulated actions/push is fixed at n_episodes x budget regardless of P/I). DEFAULT=64x4.
SPLITS_MAIN = [("default", 64, 4), ("128x2", 128, 2), ("32x8", 32, 8)]
SPLITS_EXTRA = [("256x1", 256, 1), ("16x16", 16, 16)]
SPLITS = SPLITS_MAIN + SPLITS_EXTRA

# --- MAIN queue: priority 1, three of the five (A) splits (default centred, one on each side),
# objective=lyapunov, budget=256 sims/push (fallback from the pre-registered 512: at ~6.0
# sims/s, 5 full cells would be ~19h -- over the 12h target even before objective cells or the
# 256-fallback's own headroom. DEVIATION from DESIGN.md's "priority 1 = all of (A)": only 3 of
# 5 splits fit the 12h window; the other two extremes (256x1 pure sampling, 16x16 many small
# iterations) move to the FRONT of the EXTRA queue so the full (A) ablation still completes,
# just after ~12h instead of within it. See DESIGN.md "Deviations".)
MAIN = [cell(f"oracle_A_{suf}", pop, it) for suf, pop, it in SPLITS_MAIN]  # default cell FIRST

# --- EXTRA queue: the two remaining (A) splits first (completes priority 1), then priority 2
# (objective ablation, default split), then the remaining-starts extension (paired, same goals,
# new starts 44/45) for every (A) split, then the push-length-range ablation, then (if time)
# mass-weight 1x at the best (A) split -- left as a placeholder cell pointing at the default
# split until the main queue's analysis names the winner (see TODO.md).
EXTRA = [
    *[cell(f"oracle_A_{suf}", pop, it) for suf, pop, it in SPLITS_EXTRA],
    cell("oracle_B_mass1", 64, 4, extra=["--mass-weight", "1.0"]),
    cell("oracle_B_mass3", 64, 4, extra=["--mass-weight", "3.0"]),
    cell("oracle_B_crowd", 64, 4, extra=["--value", "crowd_floor"]),
    # remaining starts (44, 45) for every (A) split cell, same goals -- paired extension
    *[cell(f"oracle_A_{suf}_starts4445", pop, it, starts=[44, 45]) for suf, pop, it in SPLITS],
    cell("oracle_len_20_40", 64, 4, extra=["--l-min", "0.020", "--l-max", "0.040"]),
    cell("oracle_len_fixed20", 64, 4, extra=["--l-min", "0.020", "--l-max", "0.0201"]),
]


def is_complete(c):
    p = RESULTS / f"{c['tag']}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    return d.get("step", 0) >= c["steps"] and len(d.get("episodes", [])) == c["n_tasks"]


def ledger_line_count():
    src = REPO / "runs/COMMANDS.jsonl"
    if not src.exists():
        return 0
    return len(src.read_text().splitlines())


def copy_ledger_lines(before_n):
    """Append runs/COMMANDS.jsonl lines beyond line `before_n` (captured before the job
    started) into experiments/COMMANDS.jsonl. A line-count offset, not a timestamp
    comparison, so two jobs finishing within the same one-second tick never duplicate or
    drop a line."""
    src = REPO / "runs/COMMANDS.jsonl"
    dst = REPO / "experiments/COMMANDS.jsonl"
    if not src.exists():
        return
    lines = src.read_text().splitlines()
    with open(dst, "a") as fh:
        for line in lines[before_n:]:
            fh.write(line + "\n")


def run_queue(cells, exp_tag):
    for c in cells:
        if is_complete(c):
            print(f"[queue] {c['tag']}: already complete, skipping", flush=True)
            continue
        print(f"[queue] {c['tag']}: launching ({c['n_tasks']} tasks x {c['steps']} pushes)", flush=True)
        before_n = ledger_line_count()
        cmd = [PY, "-u", str(ORACLE), *c["args"]]
        rc = subprocess.run(
            [PY, str(REPO / "scripts/run_probe.py"), "--tag", c["tag"], "--out-dir",
             str(EXP / "runs"), "--timeout", "0", "--exp", "EXP-0057", "--artifact-dir",
             str(EXP / f"artifacts/{c['tag']}"), "--", *cmd]).returncode
        copy_ledger_lines(before_n)
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"[queue] {c['tag']}: {status}", flush=True)
        if not is_complete(c):
            print(f"[queue] {c['tag']}: still not complete after run (rc={rc}) -- will retry on next queue invocation", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    if which == "main":
        run_queue(MAIN, "main")
        print("[queue] MAIN queue finished; launching EXTRA queue", flush=True)
        run_queue(EXTRA, "extra")
    elif which == "extra":
        run_queue(EXTRA, "extra")
    else:
        print("usage: queue.py [main|extra]"); sys.exit(1)
    print("[queue] all done", flush=True)


if __name__ == "__main__":
    main()
