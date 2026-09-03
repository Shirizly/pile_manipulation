#!/usr/bin/env bash
# Model comparison as a function of particle count: cubes at n=20/50/80, sand
# as the continuum limit.
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
SAND="${2:-Genesis/data/sand/varied/**/*_data.pt}"
SIZE="${3:-0.003}"
OUT=outputs/cube_spectrum
mkdir -p "$OUT"
say() { echo "=== $(date '+%H:%M:%S')  $*" | tee -a "$OUT/00_timeline.log"; }

# Mask view only, and blurred. Both choices are measurements, not taste: the
# SE(2) warp destroys pixel-scale features, so SHARP fields need smoothing
# (cube silhouettes and the sand mask both improved a lot; smooth sand density
# got worse). A binary mask is also the only view the whole spectrum shares --
# it is what an overhead camera sees of cubes and of sand alike.
for N in 20 50 80; do
  G="$ROOT/n$N/**/*_data.pt"
  ls $ROOT/n$N >/dev/null 2>&1 || { say "n=$N: no data, skipping"; continue; }
  say "n=$N cubes"
  python -u sand_model_zoo.py --glob "$G" --view mask --cube-size "$SIZE" \
      --blur 1 --res 32 --crop 0.5 --ranks 4 16 64 256 \
      > "$OUT/zoo_n$N.log" 2>&1
  sed -n '/SWEPT REGION/,$p' "$OUT/zoo_n$N.log" | tee -a "$OUT/00_timeline.log"
done

say "sand (continuum limit)"
python -u sand_model_zoo.py --glob "$SAND" --view mask --min-grains 2 \
    --blur 1 --res 32 --crop 0.5 --ranks 4 16 64 256 \
    > "$OUT/zoo_sand.log" 2>&1
sed -n '/SWEPT REGION/,$p' "$OUT/zoo_sand.log" | tee -a "$OUT/00_timeline.log"

say "trend across the spectrum"
python -u scripts/cube_spectrum_summary.py --dir "$OUT" \
    | tee -a "$OUT/00_timeline.log"

say "done -> $OUT"
