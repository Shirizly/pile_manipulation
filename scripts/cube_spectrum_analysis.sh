#!/usr/bin/env bash
# Model comparison as a function of cube count.
#
# There used to be a sand row here, as the continuum limit. The MPM sand path
# was abandoned (docs/rejected_mpm_sand.md) and its datasets are void, so the
# spectrum is cube counts alone -- which is sufficient: run through one code
# path the operator's margin over mean-delta is flat across regimes anyway
# (EXP-0002, EXP-0006).
#
# The point is a TREND, so every run below differs only in the dataset and the
# cube footprint -- same views, same resolution, same crop, same splits, same
# model families. Anything else drifting between rows would land inside the
# comparison rather than beside it.
#
#   bash scripts/cube_spectrum_analysis.sh
#
set -u
ROOT="${1:-Genesis/data/cube_spectrum}"
SIZE="${3:-0.005}"
OUT=outputs/cube_spectrum
mkdir -p "$OUT"
say() { echo "=== $(date '+%H:%M:%S')  $*" | tee -a "$OUT/00_timeline.log"; }

# Mask view only, and blurred. Both choices are measurements, not taste: the
# SE(2) warp destroys pixel-scale features, so a SHARP field needs smoothing
# before it survives the round trip, and a cube silhouette is as sharp as they
# come. The mask is also the honest camera model -- it is what an overhead
# camera sees.
for N in 20 30 40 50 80; do
  G="$ROOT/n$N/**/*_data.pt"
  ls $ROOT/n$N >/dev/null 2>&1 || { say "n=$N: no data, skipping"; continue; }
  # Characterise BEFORE fitting. A Genesis solver swap produced an 8.6x
  # "speedup" whose data had cubes 2896 mm outside a 128 mm tray, and the model
  # numbers from it would have looked unremarkable rather than wrong -- so
  # containment and displacement get checked first, every time.
  say "n=$N cubes: physics check"
  python -u scripts/describe_dataset.py --glob "$G" > "$OUT/describe_n$N.log" 2>&1
  grep -E "mass in tray|z range|displacement|transitions" "$OUT/describe_n$N.log" \
      | tee -a "$OUT/00_timeline.log"
  say "n=$N cubes"
  python -u model_zoo.py --glob "$G" --view mask --cube-size "$SIZE" \
      --blur 1 --res 32 --crop 0.5 --ranks 4 16 64 256 \
      > "$OUT/zoo_n$N.log" 2>&1
  sed -n '/SWEPT REGION/,$p' "$OUT/zoo_n$N.log" | tee -a "$OUT/00_timeline.log"
done

say "trend across the spectrum"
python -u scripts/cube_spectrum_summary.py --dir "$OUT" \
    --cols n20 n30 n40 n50 n80 \
    | tee -a "$OUT/00_timeline.log"

say "done -> $OUT"
