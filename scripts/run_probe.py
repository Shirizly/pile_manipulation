#!/usr/bin/env python3
"""Run a long job so it cannot fail in the three ways it kept failing.

    python scripts/run_probe.py --tag blur_sweep -- python scripts/probes/view_blur.py --res 32

On 2026-09-05 the same three mechanical hazards cost more time than several of
the experiments they were supporting, and prose warnings in the skill did not
stop them -- the author of the warning hit two of them afterwards. So they are
made unavailable here rather than discouraged:

1. **Buffered output.** Python fully buffers stdout to a file, so a running job
   looks identical to a hung one. `-u` is forced into the command.
2. **Piping through `tail`/`head`.** It buffers, and can leave the shell hung on
   a pipe whose child has already exited (that one wasted ~15 min). Output is
   redirected straight to a file; there is no pipe to get wrong.
3. **`pkill -f <pattern>`.** A pattern broad enough to match the job is also
   broad enough to match the shell doing the killing, which is exactly what
   happened -- the command killed itself before starting the run. The PID is
   recorded to a file so a job is stopped by PID, never by pattern.

Also stamps `utils.git_provenance()` beside the log, so the code state a run
happened under is recorded at run time rather than reconstructed afterwards
(12 of 15 existing records name a script that did not exist at the sha they
record).

4. **A command recorded only on success.** `runs/<tag>.json` used to be written
   after the process exited, so every killed, OOMed or suspended run left a log
   with no invocation beside it -- `runs/n50_L20mm.log` is exactly that, a
   `.pid` and a log from a collection that was stopped, and no record of the
   command that produced it. Those are precisely the runs whose command you
   need. Now the meta is written **before** `Popen` with `status: running` and
   rewritten on exit, and every run also appends to an append-only ledger
   `runs/COMMANDS.jsonl` (start and end events, joined on `run_id`). A ledger
   entry with a start and no end means the job was interrupted, which is
   information rather than an absence.

Pass `--exp EXP-0024_v1` to file the run against a record, and
`--artifact-dir <dir>` to drop a `COMMAND.txt` in the run's own output
directory, so the invocation travels with the data when that directory is
copied or shared.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "runs" / "COMMANDS.jsonl"


def ledger_append(entry: dict) -> None:
    """Append one event to the command ledger. Append-only, never rewritten.

    Two runs finishing at the same moment would lose an update under
    read-modify-write, so completion is a second `event: end` record joined to
    its start on `run_id` rather than an edit of the first line.
    """
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="short name; log is runs/<tag>.log")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--threads", type=int, default=4,
                    help="OMP/MKL threads. This is a shared box; four concurrent "
                         "BLAS jobs took the load average to 37 on 20 cores and "
                         "starved an agent's entire budget.")
    ap.add_argument("--timeout", type=int, default=0, help="seconds; 0 = none")
    ap.add_argument("--exp", default="unfiled",
                    help="record this run belongs to, e.g. EXP-0024_v1. "
                         "'unfiled' is accepted -- an unfiled command is worth "
                         "more than no command -- but name one when you can.")
    ap.add_argument("--artifact-dir", default=None,
                    help="the run's own output directory. A COMMAND.txt is "
                         "written there so the invocation travels with the data; "
                         "a central ledger is useless once an output dir is "
                         "copied somewhere else.")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="the command, after a literal --")
    a = ap.parse_args()

    cmd = [c for c in a.cmd if c != "--"]
    if not cmd:
        ap.error("no command given (put it after `--`)")
    # Force unbuffered python.
    if pathlib.Path(cmd[0]).name.startswith("python") and "-u" not in cmd:
        cmd.insert(1, "-u")
    if any(c in ("|", "tail", "head") for c in cmd):
        ap.error("do not pipe: output is redirected to a file already")

    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log, pidf, meta = out / f"{a.tag}.log", out / f"{a.tag}.pid", out / f"{a.tag}.json"

    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT),
                "OMP_NUM_THREADS": str(a.threads),
                "MKL_NUM_THREADS": str(a.threads)})

    sys.path.insert(0, str(ROOT))
    try:
        from utils import git_provenance
        prov = git_provenance(str(ROOT))
    except Exception as exc:
        prov = {"commit": "unknown", "error": repr(exc)}

    t0 = time.time()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_id = f"{a.tag}-{ts}-{os.getpid()}"

    # --- write the command BEFORE the run, not after ------------------------
    # A command recorded only on success is missing from exactly the runs whose
    # command you need: the killed, the OOMed, the suspended.
    started = {"run_id": run_id, "event": "start", "ts": ts, "tag": a.tag,
               "exp": a.exp, "cmd": cmd, "out": a.artifact_dir,
               "log": str(log.relative_to(ROOT)),
               "commit": prov.get("commit"), "dirty": prov.get("dirty"),
               "status": "running"}
    ledger_append(started)
    meta.write_text(json.dumps({**started, "provenance": prov}, indent=2))

    if a.artifact_dir:
        art = (ROOT / a.artifact_dir)
        art.mkdir(parents=True, exist_ok=True)
        (art / "COMMAND.txt").write_text(
            "# written by scripts/run_probe.py before the run started\n"
            f"# run_id {run_id}\n"
            f"# exp     {a.exp}\n"
            f"# commit  {prov.get('commit')} dirty={prov.get('dirty')}\n"
            f"# log     {log.relative_to(ROOT)}\n"
            # shlex.join, not " ".join: an argument containing spaces or quotes
            # (a --goals list, a python -c body) must paste back into a shell
            # unchanged, or the copy is not the command that ran.
            + shlex.join(cmd) + "\n")

    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             env=env, cwd=ROOT)
        pidf.write_text(str(p.pid))
        print(f"[run_probe] pid {p.pid} -> {log.relative_to(ROOT)}")
        print(f"[run_probe] commit {prov.get('commit')} dirty={prov.get('dirty')}")
        if prov.get("dirty"):
            print(f"[run_probe] WARNING dirty tree: {prov.get('dirty_files')} — "
                  f"this sha will not reconstruct the run")
        try:
            rc = p.wait(timeout=a.timeout or None)
        except subprocess.TimeoutExpired:
            p.kill(); rc = -9
            print(f"[run_probe] TIMEOUT after {a.timeout}s, killed by pid")

    secs = round(time.time() - t0, 1)
    status = "ok" if rc == 0 else ("killed" if rc < 0 else "failed")
    meta.write_text(json.dumps(
        {**started, "status": status, "returncode": rc,
         "seconds": secs, "provenance": prov}, indent=2))
    ledger_append({"run_id": run_id, "event": "end", "tag": a.tag,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "status": status, "returncode": rc, "seconds": secs})
    pidf.unlink(missing_ok=True)
    print(f"[run_probe] exit {rc} in {time.time() - t0:.0f}s; meta {meta.relative_to(ROOT)}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
