# Claim register

One row per claim this project has made or is testing. **The `depends on`
column is the point of the file:** when an invariant breaks, `grep` it here and
every affected claim is found in one step, instead of being reconstructed by
hand.

**Status:** `supported` · `refuted` · `open` (stated, under test) ·
**`narrowed`** (the measurement stands and reproduces; a newly tested condition
changes the result, so the generality is smaller than the claim asserted —
distinct from `contested`) · `contested` (evidence genuinely conflicts **under
matched conditions**) · `invalidated` (an input is now known broken; the
measurement stands, the conclusion does not) · `superseded` ·
`withdrawn` (the whole line of work was abandoned).

**Grade** is the best grade among supporting records, capped at `low` when
support is exploratory-only. It is recomputed by `scripts/check_register.py`.

## Reset 2026-09-10

Every prior claim (C-###) and its supporting `EXP-####` records were archived
wholesale to `archive/2026-09-10_pre-reset/docs/experiments/` — not because
any claim was found wrong, but to start the ledger clean rather than carry
ids, dependency chains, and partially-superseded rows forward from a
superseded work program. A withdrawn or invalidated claim from before this
date is not restated here; if it matters again, re-test it and open a new row
— the old measurement and its full history are intact in the archive.

## Open and contested

| ID | Claim | Status | Grade | Supported by | Contradicted by | Depends on |
|---|---|---|---|---|---|---|
| _(none yet)_ | | | | | | |

## Supported

| ID | Claim | Status | Grade | Supported by | Depends on |
|---|---|---|---|---|---|
| C-001 | NFD_randlen's swept-region image accuracy exceeds both tested GNN variants' (gnn_l20l40, gnn_randlen_n30) on L20mm, L40mm, and the overnight_randlen held-out test; slateN capture is positive (better than random) across all 3 goals x 3 value functions tested, for all 3 models, on all 3 corpora | supported | very-low | EXP-0001 | randlen-train-test-file-disjoint, randlen-step0-pool-size-128, occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x |
| C-002 | For the two overnight_randlen-trained models, accuracy varies <0.04 across the piled/scattered/mixed spawn-mode split, and nfd_randlen beats gnn_randlen_n30 in every mode; slateN capture is positive in all 54 (model x mode x goal x value-fn) cells | supported | very-low | EXP-0002 | randlen-train-test-file-disjoint, randlen-step0-pool-size-128, occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x | randlen-train-test-file-disjoint, randlen-step0-pool-size-128, occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x |
| C-003 | Switching the pixel-space linear-foresight operator by push-length bin (6 bins) beats the single global operator, freshly retrained on the same data, on overnight_randlen held-out image `accuracy` AND step-0 `slateN`/capture control-utility, at both 64x64 and 32x32 | supported | very-low | EXP-0003 | randlen-train-test-file-disjoint, push-frame-warp-roundtrip, randlen-step0-pool-size-128, goal-mask-axis-convention-row-y-col-x |
