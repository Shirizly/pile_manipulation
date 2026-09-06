#!/usr/bin/env python3
"""Generate the state-of-play summary from the register and records.

Hand-copying numbers out of tables is how EXP-0013's headline came to be wrong
(a column shift put a partial-correlation value in a slate-4 column, and the
conclusion rested on it). So this is generated, never typed.

    python scripts/summarise_register.py            # markdown to stdout
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "docs" / "experiments"
ORDER = {"supported": 0, "refuted": 1, "contested": 2, "open": 3,
         "inconclusive": 4, "invalidated": 5, "superseded": 6}


def records():
    out = {}
    for f in sorted(EXP.glob("EXP-*.md")):
        m = re.match(r"^---\n(.*?)\n---\n", f.read_text(), re.S)
        if not m:
            continue
        try:
            r = yaml.safe_load(m.group(1))
        except yaml.YAMLError:
            continue
        if isinstance(r, dict) and r.get("id"):
            r["_file"] = f.name
            out[r["id"]] = r
    return out


def claims():
    rows = []
    for line in (EXP / "REGISTER.md").read_text().splitlines():
        if not line.startswith("| C-"):
            continue
        c = [x.strip() for x in line.split("|")[1:-1]]
        if len(c) < 4:
            continue
        rows.append({"id": c[0], "claim": c[1], "status": c[2].split()[0].lower()
                     if c[2] else "?", "grade": c[3],
                     "evidence": " ".join(c[4:])})
    return rows


def main() -> int:
    recs, cl = records(), claims()
    sup = [r for r in recs.values() if r.get("superseded_by")]
    live = [r for r in recs.values() if not r.get("superseded_by")]

    print("# State of play — generated, do not hand-edit\n")
    print(f"`python scripts/summarise_register.py` · "
          f"{len(cl)} claims · {len(live)} live records · {len(sup)} superseded\n")

    print("## Claims by status\n")
    by = {}
    for c in cl:
        by.setdefault(c["status"], []).append(c)
    for st in sorted(by, key=lambda s: ORDER.get(s, 9)):
        print(f"### {st} ({len(by[st])})\n")
        print("| id | grade | claim | evidence |")
        print("|---|---|---|---|")
        for c in sorted(by[st], key=lambda x: x["id"]):
            claim = re.sub(r"\*\*(.*?)\*\*", r"\1", c["claim"])[:150]
            print(f"| {c['id']} | {c['grade'][:12]} | {claim} | {c['evidence'][:60]} |")
        print()

    print("## Records\n")
    print("| id | verdict | grade | downgrades | title |")
    print("|---|---|---|---|---|")
    for r in sorted(live, key=lambda x: x["id"]):
        dg = ",".join(sorted(set(r.get("downgrades") or []))) or "—"
        print(f"| {r['id']} | {r.get('verdict','?')} | {r.get('grade','?')} | "
              f"{dg[:46]} | {str(r.get('title',''))[:70]} |")
    if sup:
        print("\n### Superseded\n")
        for r in sorted(sup, key=lambda x: x["id"]):
            print(f"- **{r['id']}** → {r['superseded_by']} — {str(r.get('title',''))[:80]}")

    print("\n## Evidence health\n")
    g = {}
    for r in live:
        g.setdefault(r.get("grade", "?"), []).append(
            (r["id"], len(set(r.get("downgrades") or []))))
    print("| grade | n | records (downgrade-domain count) |")
    print("|---|---|---|")
    for k in ("high", "moderate", "low", "very-low"):
        if k in g:
            items = ", ".join(f"{i}({d})" for i, d in sorted(g[k]))
            print(f"| {k} | {len(g[k])} | {items} |")
    dom = {}
    for r in live:
        for d in set(r.get("downgrades") or []):
            dom[d] = dom.get(d, 0) + 1
    print("\n| downgrade domain | records |")
    print("|---|---|")
    for d, n in sorted(dom.items(), key=lambda x: -x[1]):
        print(f"| {d} | {n} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
