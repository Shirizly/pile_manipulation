#!/usr/bin/env bash
# Re-run the whole sand analysis on a dataset, in one command.
#
# Every number in docs/sand_manipulation.md comes from these calls, so a new
# dataset can be characterised without reconstructing the argument lists by
# hand -- which is how the earlier cube work ended up comparing configurations
# that had silently drifted apart.
#
#   bash scripts/sand_full_analysis.sh 'Genesis/data/sand/varied/**/*_data.pt' varied
#
set -u
GLOB="${1:-Genesis/data/sand/pile20*/**/*_data.pt}"
TAG="${2:-sand}"
OUT="outputs/sand_analysis/$TAG"
mkdir -p "$OUT"
say() { echo "=== $(date '+%H:%M:%S')  $*" | tee -a "$OUT/00_timeline.log"; }

say "dataset: $GLOB"

# --- 1. dataset characterisation: is the physics sane, is the data diverse? --
say "1. dataset characterisation"
python -u scripts/sand_describe.py --glob "$GLOB" > "$OUT/10_describe.log" 2>&1
tail -30 "$OUT/10_describe.log" | tee -a "$OUT/00_timeline.log"

# --- 2. the two observation models, across validation splits -----------------
# density = depth-sensing; mask = what an overhead camera sees. Blur is a
# precondition for SHARP fields only, so the two views want different settings:
# mask needs blur, density is hurt by it.
say "2. view comparison and split robustness"
for cfg in "density 0" "mask 1"; do
  set -- $cfg; VIEW=$1; BLUR=$2
  for SEED in 0 1 2; do
    python -u sand_foresight.py --glob "$GLOB" --view "$VIEW" --blur "$BLUR" \
        --min-grains 2 --res 32 --crop 0.5 --seed "$SEED" \
        >> "$OUT/20_views.log" 2>&1
  done
done
grep -E "^(view|linear-nonneg|identity|persistence|heur|affine)" "$OUT/20_views.log" \
    | tail -40 | tee -a "$OUT/00_timeline.log"

# --- 3. model families on both views ----------------------------------------
say "3. model zoo"
python -u sand_model_zoo.py --glob "$GLOB" --view density --blur 0 \
    --res 32 --crop 0.5 --ranks 4 16 64 256 > "$OUT/30_zoo_density.log" 2>&1
python -u sand_model_zoo.py --glob "$GLOB" --view mask --min-grains 2 --blur 1 \
    --res 32 --crop 0.5 --ranks 4 16 64 256 > "$OUT/31_zoo_mask.log" 2>&1
sed -n '/SWEPT REGION/,$p' "$OUT/30_zoo_density.log" | tee -a "$OUT/00_timeline.log"
sed -n '/SWEPT REGION/,$p' "$OUT/31_zoo_mask.log" | tee -a "$OUT/00_timeline.log"

# --- 4. effective dimensionality --------------------------------------------
# The rank results are only meaningful next to the dimensionality of the INPUT
# states: an operator cannot need more rank than its inputs supply, and the
# single-pile dataset supplied only ~14 dims.
say "4. effective dimensionality"
python -u scripts/sand_dimensionality.py --glob "$GLOB" > "$OUT/40_rank.log" 2>&1
cat "$OUT/40_rank.log" | tee -a "$OUT/00_timeline.log"

# --- 5. capacity sweep -------------------------------------------------------
say "5. crop/resolution sweep"
python -u scripts/sand_capacity_sweep.py --glob "$GLOB" > "$OUT/50_sweep.log" 2>&1
cat "$OUT/50_sweep.log" | tee -a "$OUT/00_timeline.log"

say "done -> $OUT"
