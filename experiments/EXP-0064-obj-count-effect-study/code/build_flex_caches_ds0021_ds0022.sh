#!/bin/bash
# EXP-0064 RUN-0013 step 1: make DS-0021 / DS-0022 readable by FlexData (resumable; every unit atomic).
set -e
cd /home/alon/Code/pile_manipulation
W=${W:-8}
mkdir -p datasets/DS-0021-flex-carrots-countgroups-train/cache datasets/DS-0022-flex-carrots-countgroups-test-slates/cache
python -u -m FlexData.build_cache splits_ds0021
python -u -m FlexData.build_cache splits_ds0022
python -u -m FlexData.build_cache paths_ds0021
python -u -m FlexData.build_cache paths_ds0022
python -u -m FlexData.build_cache ds0021 --workers $W
python -u -m FlexData.build_cache ds0022 --workers $W
python -u -m FlexData.image_mask ds0021 --workers $W
python -u -m FlexData.image_mask ds0022 --workers $W
echo CACHES DONE
