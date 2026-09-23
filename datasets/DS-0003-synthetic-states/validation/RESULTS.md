# DS-0003 validation results — generator rewrite (2026-09-17: scattered+clump -> compaction)

Both runs: `datasets/DS-0003-synthetic-states/validate.py`, ~4000-state
matched sample (1/3 each of n_objects=20/50/100) from each of DS-0002 (real)
and DS-0003 (synthetic), fast bit-exact rasteriser, `dmdc_baseline.py`
descriptors. "frac_covered" = fraction of the block's dims where synthetic's
[min,max] covers real's [min,max] with a 5% pad. "before" numbers are the
post-region-fix (2026-09-15) generator, carried over from the previous
`DATASET.md`/this same script; "after" is this rewrite
(`pile_compaction.py::sample_compacted_state` + exact SAT legality).

## Descriptor block coverage, before vs. after

| block | dim | before | after | verdict |
|---|---|---|---|---|
| const | 1 | 1.00 | 1.00 | unchanged (trivially covered) |
| mass | 1 | 0.00 | 0.00 | still OOD, unchanged |
| com | 2 | 1.00 | 1.00 | unchanged (fixed by the 2026-09-15 region fix, retained) |
| moments2 | 3 | 0.67 | **0.00** | **regressed** |
| dft_real | 40 | 0.40 | **0.00** | **regressed** |
| dft_imag | 40 | 0.38 | **0.10** | **regressed** |

**This is a real regression, reported honestly rather than hidden.** Cause:
the new generator's dispersed end (`compaction=0.05`) is still a grid seed
sized from cube geometry, not from a large placement region — unlike the
retired generator, which explicitly drew per-state placement regions up to
`R=0.085` m and so could put mass anywhere across most of the workspace.
`sample_compacted_state`'s `compaction` parameter only controls relaxation
*sweep count*; it does not control seed-lattice extent. So across its whole
range this generator produces tight-to-very-tight clusters, never the
loosely scattered configurations that drive real states' higher `moments2`/
`dft` values.

**Confirmed directly on the corpus** (RMS-from-own-COM spread and mean
touching-neighbour count at threshold 0.00813 m, this run's full 30,000-state
corpus vs. DS-0002):

| n_objects | real spread range | synth spread range | real mean neighbours | synth mean neighbours |
|---|---|---|---|---|
| 20  | [0.0079, 0.0629] | [0.0102, 0.0133] | 1.04 | 3.64 |
| 50  | [0.0153, 0.0608] | [0.0165, 0.0206] | 1.79 | 4.02 |
| 100 | [0.0230, 0.0637] | [0.0237, 0.0288] | 2.06 | 4.20 |

Synthetic spread never reaches even a third of real's upper range at any
`n_objects`, and mean touching-neighbour count is now roughly 2x real at
every count — the corpus is uniformly denser than real, not diverse across
the dispersed<->compact axis the way DS-0002 actually is. COM range is
unaffected and still matches real well (both use the same per-state
random-offset placement idea).

**This is the generator's structural trade: it closes the "old generator
never compacted" gap (see `DATASET.md`'s packing-floor table — new minima
beat the old ones at n=50/100 and come within ~15% of real at n=100) but,
as built here, does not preserve the earlier generator's ability to scatter
material loosely across the workspace. Extending `compaction`'s low end to
draw a genuinely large seed lattice (not just a slower relaxation) would be
required to recover the dispersed tail — not attempted here per the "do not
redesign the geometry modules" instruction; flagged for whoever owns the
next iteration.**

## Pixel mass (unchanged shape, matched n_objects mix)

| n_objects | real | synthetic |
|---|---|---|
| 20  | 97.0 +/- 4.8   | 98.2 +/- 3.9  |
| 50  | 241.8 +/- 7.7  | 245.4 +/- 6.5 |
| 100 | 481.9 +/- 13.2 | 490.8 +/- 8.9 |

Full numbers: `results.json`; figure: `real_vs_synthetic.png`.
