#!/usr/bin/env python3
"""Validate docs/experiments/ — the experiment records, the claim register and
the invariant registry.

A register nobody checks goes stale within a month; this repo's own doc map
already points at a skill that does not exist. So the rules in
.claude/skills/experiment-log/SKILL.md are enforced here rather than trusted.

    python scripts/check_register.py            # check
    python scripts/check_register.py --fix-grades   # rewrite computed grades

Exits non-zero on any error. Warnings do not fail the run.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPDIR = ROOT / "docs" / "experiments"

GRADES = ["high", "moderate", "low", "very-low"]
VERDICTS = {"supported", "refuted", "inconclusive", "invalidated"}
TIERS = {"T0", "T1", "T2"}
MODES = {"exploratory", "confirmatory"}
DOWNGRADES = {"provenance", "imprecision", "indirectness", "inconsistency",
              "selection", "untested-dependency", "incomplete-design"}

# Fields every tier needs, and the extra ones each tier adds.
BASE = ["id", "title", "tier", "mode", "date", "claim", "provenance", "result",
        "verdict", "downgrades", "grade"]
T1_EXTRA = ["design", "noise_floor", "depends_on"]
PROVENANCE = ["commit", "script", "data", "code_path", "seed", "split"]
DESIGN = ["varied", "held_fixed", "baselines", "metric"]


def load_records():
    """Parse the YAML frontmatter of every EXP-*.md."""
    out = []
    for path in sorted(EXPDIR.glob("EXP-*.md")):
        text = path.read_text()
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        if not m:
            out.append((path, None, "no YAML frontmatter"))
            continue
        try:
            out.append((path, yaml.safe_load(m.group(1)), None))
        except yaml.YAMLError as exc:
            out.append((path, None, f"unparseable frontmatter: {exc}"))
    return out


def load_invariants():
    """Tag -> status, from the table in INVARIANTS.md."""
    path = EXPDIR / "INVARIANTS.md"
    if not path.exists():
        return {}
    tags = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*`([a-z0-9-]+)`\s*\|.*?\|\s*[*`]*([a-z]+)[*`]*", line)
        if m:
            tags[m.group(1)] = m.group(2)
    return tags


def computed_grade(downgrades):
    """Grade is derived, never chosen: start high, drop one level per domain."""
    return GRADES[min(len(set(downgrades or [])), len(GRADES) - 1)]


def check_record(path, rec, invariants, errors, warnings):
    name = path.name
    def err(msg): errors.append(f"{name}: {msg}")
    def warn(msg): warnings.append(f"{name}: {msg}")

    tier = rec.get("tier")
    if tier not in TIERS:
        err(f"tier {tier!r} not in {sorted(TIERS)}")
        return
    mode = rec.get("mode")
    if mode not in MODES:
        err(f"mode {mode!r} not in {sorted(MODES)}")

    required = list(BASE)
    if tier in ("T1", "T2"):
        required += T1_EXTRA
    for f in required:
        if f not in rec or rec[f] in (None, "", []):
            if f == "downgrades" and rec.get(f) == []:
                continue          # an empty downgrade list is a real answer
            err(f"missing required field for {tier}: {f}")

    if rec.get("id") != path.stem.split("-")[0] + "-" + path.stem.split("-")[1]:
        err(f"id {rec.get('id')!r} does not match filename")

    if rec.get("verdict") not in VERDICTS:
        err(f"verdict {rec.get('verdict')!r} not in {sorted(VERDICTS)}")

    bad = set(rec.get("downgrades") or []) - DOWNGRADES
    if bad:
        err(f"unknown downgrade domain(s): {sorted(bad)}")

    want = computed_grade(rec.get("downgrades"))
    if rec.get("grade") != want:
        err(f"grade is {rec.get('grade')!r} but {len(set(rec.get('downgrades') or []))} "
            f"downgrade(s) compute to {want!r}")

    prov = rec.get("provenance") or {}
    sha = str(prov.get("commit", ""))
    if sha and sha not in ("unknown",) and not re.match(r"^[0-9a-f]{7,40}\b", sha):
        warn(f"provenance.commit {sha!r} does not look like a git sha")
    if "dirty" not in prov:
        warn("no provenance.dirty — a sha does not reconstruct a run if the "
             "tree was modified; get it from utils.git_provenance()")
    elif prov.get("dirty") is True and "dirty" not in str(rec.get("downgrades", "")):
        warn("ran with a dirty tree: say in the body what was uncommitted")
    # A sha that predates the script it names does not reconstruct the run.
    # 11 of 14 records were in this state on 2026-09-05, all written in the
    # same session as their script and committed afterwards.
    script = str(prov.get("script", "")).split()[0].split("(")[0].strip()
    if sha and script.endswith(".py") and re.match(r"^[0-9a-f]{7,40}$", sha):
        import subprocess as _sp
        if _sp.run(["git", "cat-file", "-e", f"{sha}:{script}"],
                   capture_output=True).returncode != 0:
            (err if prov.get("dirty") is not True else warn)(
                f"provenance.commit {sha} predates its own script {script} — "
                f"the sha does not reconstruct this run"
                + ("" if prov.get("dirty") is True else "; set dirty: true and say so"))
    if "data_commit" not in prov:
        warn("no provenance.data_commit — the simulator lineage of the dataset "
             "is a separate question from the analysis code's")
    for f in PROVENANCE:
        if f not in prov or prov[f] in (None, ""):
            err(f"provenance.{f} is missing — comparisons need it")

    if tier in ("T1", "T2"):
        # An "Unrelated findings" section is required at T1+, and must say
        # something -- an empty heading is ambiguous, "none" is an answer.
        body = path.read_text().split("\n---\n", 2)[-1]
        if "## Unrelated findings" not in body:
            err("missing the '## Unrelated findings' section required at T1+")
        b = rec.get("budget") or {}
        if not b.get("declared"):
            warn("no budget.declared — the task-giver should set one before the run")
        elif b.get("outcome") not in (None, "within", "exceeded", "stopped-early"):
            err(f"budget.outcome {b.get('outcome')!r} not in "
                f"within|exceeded|stopped-early")
        design = rec.get("design") or {}
        for f in DESIGN:
            if f not in design or design[f] in (None, "", [], {}):
                err(f"design.{f} is missing")
        met = str(design.get("metric", ""))
        keys = ("pct_persistence", "pct_persistence_wholeimage",
                "explained_over_meandelta", "explained", "soft_iou",
                "l1_per_mass", "frobenius", "spearman", "slate4", "partial",
                "FSS", "canonical_delta_profile", "r2_grouped_cv",
                "world_alignment_cosine")
        if met and not any(k in met for k in keys):
            warn(f"design.metric does not name a key from "
                 f"docs/experiments/METRICS.md: {met[:60]!r}")
        if not (design.get("baselines") or []):
            err("design.baselines must be non-empty and include a do-nothing baseline")

    # depends_on / establishes must name known tags
    cited = list(rec.get("depends_on") or [])
    establishes = list(rec.get("establishes") or [])
    for tag in cited + establishes:
        if tag not in invariants:
            err(f"depends_on/establishes tag {tag!r} is not in INVARIANTS.md")

    # A record that ESTABLISHES a tag's status is not penalised for it.
    weak = [t for t in cited if invariants.get(t) not in ("holds", "fixed")]
    if weak and "untested-dependency" not in (rec.get("downgrades") or []):
        err(f"cites non-holding tag(s) {weak} but does not take the "
            f"'untested-dependency' downgrade")

    if tier == "T2":
        if mode != "confirmatory":
            err("T2 records must be confirmatory")
        pred = rec.get("prediction")
        if not isinstance(pred, dict) or not pred.get("supports") or not pred.get("refutes"):
            err("T2 requires a prediction block with supports and refutes")
        elif pred.get("discriminating") is not True:
            err("T2 requires discriminating: true — redesign rather than run")
        if weak:
            err(f"T2 may not depend on non-holding tag(s): {weak}")

    pred = rec.get("prediction")
    if isinstance(pred, dict) and pred.get("discriminating") is False:
        warn("prediction.discriminating is false — this design cannot separate "
             "its branches; it should have been redesigned, not run")

    if mode == "exploratory" and rec.get("grade") == "high":
        err("an exploratory record cannot grade 'high' — re-run as T2 to promote")


def check_register(records, invariants, errors, warnings):
    path = EXPDIR / "REGISTER.md"
    if not path.exists():
        errors.append("REGISTER.md is missing")
        return
    text = path.read_text()
    known = {r["id"] for _, r, e in records if r and not e}

    for exp in sorted(set(re.findall(r"EXP-\d{4}", text))):
        if exp not in known:
            errors.append(f"REGISTER.md cites {exp}, which has no record file")

    # Only the "depends on" column names tags; the rest of a row is prose.
    for line in text.splitlines():
        if not line.startswith("|") or line.count("|") < 4:
            continue
        for tag in re.findall(r"`([a-z][a-z0-9-]{3,})`", line.rsplit("|", 2)[-2]):
            if tag not in invariants:
                warnings.append(f"REGISTER.md depends-on column names `{tag}`, "
                                f"which is not an INVARIANTS.md tag")

    # Duplicate claim ids: two records allocating the same C-#### silently
    # overwrite each other's rows. Caught the hard way on 2026-09-05.
    ids = re.findall(r"^\| (C-\d{3,4}) \|", text, re.M)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        errors.append(f"REGISTER.md has duplicate claim id(s): {dupes} — two "
                      f"experiments allocated the same number")

    cited_in_register = set(re.findall(r"EXP-\d{4}", text))
    for exp in sorted(known - cited_in_register):
        warnings.append(f"{exp} is not referenced from REGISTER.md — a record "
                        f"nobody cites is a number nobody can find")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix-grades", action="store_true",
                    help="rewrite each record's grade to the computed value")
    args = ap.parse_args()

    if not EXPDIR.exists():
        print(f"no {EXPDIR.relative_to(ROOT)} — nothing to check")
        return 0

    invariants = load_invariants()
    records = load_records()
    errors, warnings = [], []

    for path, rec, parse_err in records:
        if parse_err:
            errors.append(f"{path.name}: {parse_err}")
            continue
        if args.fix_grades:
            want = computed_grade(rec.get("downgrades"))
            if rec.get("grade") != want:
                text = path.read_text()
                path.write_text(re.sub(r"^grade: .*$", f"grade: {want}", text,
                                       count=1, flags=re.M))
                print(f"  fixed {path.name}: grade -> {want}")
                rec["grade"] = want
        check_record(path, rec, invariants, errors, warnings)

    check_register(records, invariants, errors, warnings)

    n = len([r for _, r, e in records if r and not e])
    print(f"checked {n} record(s), {len(invariants)} invariant tag(s)")
    # The letter grade saturates at 3 downgrade domains, so print the counts:
    # a claim going 4 -> 3 domains is real progress the letter cannot show.
    tally = {}
    for _, r, e in records:
        if r and not e:
            tally.setdefault(r.get("grade", "?"), []).append(
                (r.get("id", "?"), len(set(r.get("downgrades") or []))))
    for g in ("high", "moderate", "low", "very-low"):
        if g in tally:
            items = ", ".join(f"{i}({d})" for i, d in sorted(tally[g]))
            print(f"  {g:9s} {len(tally[g]):2d}  {items}")
    for w in warnings:
        print(f"  warn:  {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print(f"\n{len(errors)} error(s)")
        return 1
    print("ok" + (f" ({len(warnings)} warning(s))" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
