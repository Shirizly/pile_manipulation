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
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent


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

    meta.write_text(json.dumps(
        {"tag": a.tag, "cmd": cmd, "returncode": rc,
         "seconds": round(time.time() - t0, 1), "provenance": prov}, indent=2))
    pidf.unlink(missing_ok=True)
    print(f"[run_probe] exit {rc} in {time.time() - t0:.0f}s; meta {meta.relative_to(ROOT)}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
