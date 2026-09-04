"""Collapse the per-dataset model-zoo logs into one table across the spectrum.

Reads the logs `scripts/cube_spectrum_analysis.sh` writes and prints, for each
model family, its score at each cube count. The whole reason the spectrum
exists is the TREND, and reading it off several separate 8-row tables is how the
earlier comparison stayed muddled for as long as it did.

Reported number is "% of change": 100% = no better than predicting nothing
moved, lower is better. The number to beat is not persistence but mean-delta.

    python scripts/cube_spectrum_summary.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROW = re.compile(r"^(?P<name>\S.*?)\s{2,}(?P<rms>[\d.]+)\s+(?P<pct>[\d.]+)%\s+"
                 r"(?P<exp>-?[\d.]+)")


def parse(path: Path) -> dict[str, tuple[float, float]]:
    """(% of change, explained) per model, from the SWEPT REGION table."""
    if not path.exists():
        return {}
    text = path.read_text()
    if "SWEPT REGION" not in text:
        return {}
    out = {}
    for line in text.split("SWEPT REGION", 1)[1].splitlines():
        m = ROW.match(line)
        if m and m.group("name") not in ("model",):
            out[m.group("name").strip()] = (float(m.group("pct")),
                                            float(m.group("exp")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/cube_spectrum")
    ap.add_argument("--cols", nargs="+",
                    default=["n20", "n30", "n40", "n50", "n80"],
                    help="columns to show, left to right; missing logs are "
                         "skipped. Counts above n=20 are expensive (Newton's "
                         "dense per-island Hessian goes as island_size^2.64, so "
                         "cost runs 0.36 -> 9.70 -> ~28 s/transition at "
                         "n=20/30/50), so a partial spectrum is the normal case "
                         "rather than an error.")
    args = ap.parse_args()

    d = Path(args.dir)
    data = {c: parse(d / f"zoo_{c}.log") for c in args.cols}
    have = [c for c in args.cols if data[c]]
    if not have:
        raise SystemExit(f"no parsable zoo logs in {d}")

    names = []
    for c in have:
        for k in data[c]:
            if k not in names:
                names.append(k)

    print(f"% of change (lower is better; 100% = nothing moved)\n")
    print(f"{'model':24s} " + " ".join(f"{c:>9s}" for c in have))
    print("-" * (25 + 10 * len(have)))
    for k in names:
        cells = " ".join(f"{data[c][k][0]:8.1f}%" if k in data[c] else f"{'--':>9s}"
                         for c in have)
        print(f"{k:24s} {cells}")

    # The comparison that actually settles the granularity question: does the
    # operator beat a constant displacement, and does that margin grow as the
    # material gets finer?
    print("\nmargin of the linear operator over mean-delta "
          "(positive = state-dependence is being used):")
    for c in have:
        r = data[c]
        lin = next((v for k, v in r.items() if k.startswith("linear")), None)
        md = next((v for k, v in r.items() if k.startswith("mean-delta")), None)
        if lin and md:
            print(f"  {c:6s} explained {md[1]:.3f} -> {lin[1]:.3f}  "
                  f"margin {lin[1] - md[1]:+.3f}")


if __name__ == "__main__":
    main()
