"""repeat_check.py -- defect 3 (precision never characterised). Runs the
SAME config (one model, one K, REPEATS timed reps) in a freshly-launched
process (no shared state with any other invocation) and prints its own
end2end median. Invoked 3x as SEPARATE `python -u` processes (see RUN.md);
the between-run spread of these 3 medians is the harness's actual
repeatability floor -- differences smaller than this floor are not
resolvable, which is exactly what explains RUN-0001's fwd_ms@128 (48.67ms)
apparently exceeding its own end2end_ms@128 (46.27ms) for schenck: two
DIFFERENT timed loops, each with its own measurement noise, compared
against each other as if noise-free.
"""
import sys
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/EXP-0009-time-budget-bench/code")
import bench  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "schenck"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 128

cell = bench.load_state_and_actions()
maker = {"schenck": bench.make_schenck, "model0001_global": lambda: bench.make_switched_linear(
    "model0001_global", "weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
    desc_dim=0, switched=False)}[MODEL]
spec = maker()
batch = bench.build_batch(cell, K, bench.DEVICE)
r = bench.time_region(lambda: spec["fwd"](spec["pre"](batch)), bench.WARMUP, bench.REPEATS)
print(f"REPEAT_RESULT model={MODEL} K={K} median_ms={r['median']:.4f} q1={r['q1']:.4f} q3={r['q3']:.4f}")
