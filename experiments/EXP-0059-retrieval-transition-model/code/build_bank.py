"""Build and persist the EXP-0059 transition bank (DS-0008 + DS-0010).

    python -u experiments/EXP-0059-retrieval-transition-model/code/build_bank.py           # legacy, uncurated
    python -u experiments/EXP-0059-retrieval-transition-model/code/build_bank.py --curated  # DS-0014, legal rows only

Writes `experiments/EXP-0059-retrieval-transition-model/artifacts/bank.pt` (legacy) or
`bank_curated.pt` (`--curated`; atomic, `TransitionBank.save`) and prints the moved-cube-count
sanity histogram (task requirement 2).

`--curated` (2026-09-28, coordinator follow-up B/ISS-010): loads DS-0014
(`TransitionBank.load_curated`, `datasets/DS-0014-retrieval-curated-interaction-sets/
build_curated_bank.py`'s payload) instead of re-deriving frames from the raw DS-0008/DS-0010
files -- excludes the ~23% of rows the ISS-010 touchdown-legality audit flagged, and carries a
geometric interaction-set mask (`in_set`) that `RetrievalPredictor` uses to restrict both search
and Hungarian transfer to the cubes that actually interact with the push. The legacy path
(`TransitionBank.build()`) is UNCHANGED -- still what `model/retrieval_nfd/donors.py` asserts
its own bank matches row-for-row.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from model.retrieval.bank import TransitionBank, DEFAULT_MOVED_THRESHOLD

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", action="store_true",
                    help="load DS-0014 (legal rows only, with interaction sets) instead of "
                         "re-deriving frames from raw DS-0008/DS-0010")
    ap.add_argument("--include-illegal", action="store_true",
                    help="(--curated only) keep ISS-010-flagged illegal rows in the bank too, "
                         "for a diagnostic/ablation comparison -- never the default")
    a = ap.parse_args()

    t0 = time.time()
    if a.curated:
        bank = TransitionBank.load_curated(include_illegal=a.include_illegal)
        print(f"loaded CURATED bank (DS-0014, include_illegal={a.include_illegal}): "
             f"{len(bank)} transitions in {time.time() - t0:.1f}s "
             f"(moved_threshold={bank.moved_threshold*1000:.1f} mm)")
        in_set_counts = bank.in_set.sum(dim=1).float()
        print(f"interaction set size: mean {in_set_counts.mean():.2f} / "
             f"median {in_set_counts.median():.1f} / max {int(in_set_counts.max())} cubes/row")
    else:
        bank = TransitionBank.build(moved_threshold=DEFAULT_MOVED_THRESHOLD,
                                    include=("ds0008", "ds0010"))
        print(f"built bank: {len(bank)} transitions in {time.time() - t0:.1f}s "
             f"(moved_threshold={bank.moved_threshold*1000:.1f} mm)")
    for src in sorted(set(bank.source)):
        print(f"  {src}: {bank.source.count(src)} rows")

    hist = bank.moved_count_histogram()
    total = len(bank)
    print("moved-cube-count histogram (n cubes with push-frame displacement "
         f"> {bank.moved_threshold*1000:.1f} mm, out of 20 per transition):")
    for k in sorted(hist):
        print(f"  {k:2d} moved: {hist[k]:5d}  ({100 * hist[k] / total:5.1f}%)")
    mean_moved = float(bank.moved.float().sum(dim=1).mean())
    print(f"mean moved cubes/transition: {mean_moved:.2f}")

    out = ARTIFACTS / ("bank_curated.pt" if a.curated else "bank.pt")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    bank.save(str(out))
    print("saved", out)


if __name__ == "__main__":
    main()
