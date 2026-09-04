"""The experiment register validates.

A register nobody checks goes stale; this repo's doc map already pointed at a
skill that did not exist. Running the validator in the suite is what keeps
docs/experiments/ honest.
"""
import subprocess
import sys


def test_check_register_passes():
    r = subprocess.run([sys.executable, "scripts/check_register.py"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
