---
id: DS-0014
title: Curated retrieval bank -- DS-0008+DS-0010 with geometric interaction sets + touchdown-legality flags
status: active
date: 2026-09-28
path: datasets/DS-0014-retrieval-curated-interaction-sets/data/curated_bank.pt
producer: datasets/DS-0014-retrieval-curated-interaction-sets/build_curated_bank.py
---

Coordinator follow-up B (2026-09-28), after ISS-010 (see `experiments/OPEN_ISSUES.md`): the
retrieval model must never see uninvolved parts of the state, and must never train/retrieve
against a transition where the tool started ON a cube. This dataset makes both reductions ONE
TIME, stored, rather than recomputed by every consumer.

**Source:** DS-0008 (train chains, `valid`-filtered) + DS-0010 (extra 18-22mm rows) -- the exact
same source `TransitionBank.build()` already used, push-frame canonicalised identically
(`model.retrieval.bank.TransitionBank.from_states`).

**What is added, per row:**
- `in_set` (T, 20) bool -- the GEOMETRIC interaction set (`model.retrieval.interaction
  .interaction_set`, truth-free: blade-swept cubes + a forward contact-chain closure). Tuned on
  this same training data (tau=12mm, angle_max_deg=60mm -- the smallest slack reaching ~95%
  recall against the truth `moved` set on LEGAL rows): **recall 0.960, precision 0.899** pooled
  over cubes (`experiments/EXP-0059-*/results/interaction_set_tuning.json`; design doc's own
  tau in {1,2,3}mm range only reaches 0.82-0.90 recall, hence the wider sweep).
- `legal` (T,) bool -- **NOT** flagged `illegal_0mm` by the ISS-010 touchdown-overlap audit
  (`experiments/EXP-0059-*/code/audit_tool_placement.py`, exact SAT test of the blade footprint
  vs every cube's rotated-square footprint at `p_start`). 9,198 / 11,921 rows (77.2%) are legal;
  the other 22.8% are **kept in this file and flagged, not deleted** (per the coordinator's
  instruction), so a diagnostic can still look at them, but `TransitionBank.load_curated`
  excludes them from the bank BY DEFAULT (`include_illegal=False`).
- `source`, `source_file`, `source_row` -- per-row provenance: which of DS-0008/DS-0010, which
  original `_k_data.pt`, and that row's index within it (BEFORE any valid-filtering, so it lines
  up with that file's own `_k_data_legality.pt` companion).

**Everything else** (`uv0, uv1, yaw0, yaw1, duv, dyaw, moved, p_starts, p_stops, push_len,
moved_threshold`) is `TransitionBank`'s own existing per-object/per-transition schema,
byte-identical to what `TransitionBank.build()` would compute from the same source rows.

**Consumer:** `model/retrieval/bank.py::TransitionBank.load_curated(path, include_illegal=False)`
is the ONLY supported way to build the bank going forward for `RetrievalPredictor`
(`experiments/EXP-0059-*/code/build_bank.py --curated`); `TransitionBank.build()`/`.from_states()`
are UNCHANGED (still used by `model/retrieval_nfd/donors.py`, which asserts its own bank matches
`TransitionBank.build()` row-for-row -- that assertion must keep working, so this dataset is
additive, not a replacement of the underlying class).

**Regenerate:** `python -u datasets/DS-0014-retrieval-curated-interaction-sets/build_curated_bank.py`
(requires `experiments/EXP-0059-*/code/audit_tool_placement.py` to have already written every
source file's `_legality.pt` companion -- see ISS-010).
