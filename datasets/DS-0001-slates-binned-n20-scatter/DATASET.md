# DS-0001 — `slates_binned` n20 scatter, 20 slates x 1000 candidates, 5 length bins

**Status:** active
**Payload:** `Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm` (23 MB, gitignored)

## What this is

20 distinct settled start states, each with **1000 candidate single pushes**
simulated from that identical state — a same-state candidate slate corpus, the
structure an action ranker is scored against, because the true outcome of every
candidate is known.

The difference from `Genesis/data/slates_multistep/*` is the action-length
distribution. Those corpora fix ONE push length each, which left four of six
length bins with no data (`slates-multistep-single-pushlen-bin-starvation`).
Here the push length is drawn uniformly inside one of **five bins spanning
20–70 mm**, so a single corpus supports fitting and scoring length-conditioned
operators over the whole range.

Material and spawn: **20 cubes, 5 mm edge, density 1000, friction 0.3,
`drop` spawn = a SCATTER (single layer)** — not a pile. `drop` cannot build more
than one layer because the cubes bounce outward on landing (90–94 % settle in
layer 0); `heap`/`pyramid` are the pile spawns. This matters for comparability:
a push through one layer is not the same operator as a push through two, and
every other same-state corpus in this repo (`slates`, `slates_multistep`) is
`heap`, i.e. a pile.

`n_steps = 1`, so this corpus contains **no action sequences** — it is a
one-step ranking corpus only. The collector supports N-step chains; that was
not what was collected here.

## Bin fidelity

**19,999 of 20,000 pushes landed in their requested length bin.** One push came
in at 18.2 mm, below the lowest edge, and carries `bin_realized = -1`
(`UNDERFLOW_BIN`) rather than being folded into bin 0 — a wall-shortened push is
not a sample of the 20–30 mm operator and must not be fitted as one. Realised
lengths span 18.2 / 45.0 / 70.0 mm (min/mean/max); 0.0 % of pushes left the
state unchanged.

Scatter spawn fills the bins far more cleanly than a pile does: heap-spawn smoke
runs at the same settings reached only ~87 % on-target, with the 60–70 mm bin
leaking heavily into 50–60 mm because the tray wall cuts long pushes short more
often when the blade must start against a two-layer heap. Raising
`--max-length-tries` from 5 to 10 also contributed; the two effects are not
separated by this collection, since only the scatter config was run at 10 tries.

## Provenance

- **commit:** `6ea03278` (dirty — the collector and reader were new, uncommitted files at run time)
- **script:** `Genesis/binned_slate_collection.py`
- **command:** recorded in `experiments/COMMANDS.jsonl` and reproduced in `config.yaml`; the invocation was
  `python -u -m Genesis.binned_slate_collection --n-states 20 --n-actions 1000 --n-steps 1 --n-envs 128 --n-cubes 20 --spawn-mode drop --seed 0 --output-root data/slates_binned --tag n20_scatter_s20a1000_L20-70mm`
- **seed:** 0
- **runtime:** 5027 s (84 min) on 128 concurrent envs, 161 simulated batches
- **resolved config:** `config.yaml` in this directory (copied from the corpus's own `manifest.json`)

## Reading it

`Genesis/binned_slate_dataset.py::BinnedSlateCorpus` — do not read the files by
hand. `verify()` asserts the two structural invariants (chain continuity across
step files; bit-identical start states within a slate, measured 0.0 m) and
reports bin fidelity. `select(bin=...)` filters on the length ACHIEVED;
`requested_bin=` on what the schedule asked for — they differ wherever the tray
wall cut a push short, and the achieved one is what a length-conditioned
operator must be fitted on.

On-disk: `step0.pt`, every tensor indexed by a flat chain id
`c = slate_idx * n_actions + action_idx`.

## Records citing this dataset

EXP-0011, EXP-0012 and EXP-0013 all used this corpus and cite it **by path**,
not by this id — they predate the id's allocation. EXP-0014 and EXP-0015 cite
`DS-0001`. The path and the id refer to the same bytes; nothing was
regenerated.

## Known limitations

- **One spawn style only** (scatter). Any result on this corpus is a
  scatter result until replicated on a pile corpus.
- **Single step.** No sequence-level scoring is possible here.
- Per-slate `dv_true` spread is uniformly small (std 0.041, against 0.134 on
  `slates_multistep`) — this corpus contains no "easy" slates, which is what
  `slates-binned-uniform-difficulty` records and what makes it a harder ranking
  benchmark than its predecessors (EXP-0012).
