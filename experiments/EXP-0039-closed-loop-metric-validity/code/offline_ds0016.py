"""EXP-0039 RUN-0002 offline metrics on DS-0016 (legal by construction) for zoo members EXP-0059 did not score,
by calling EXP-0059 code/test_v2.py's main() with its module-level model list and output path overridden
(that script has no CLI). Usage: python -u offline_ds0016.py --out <json> --models m1 m2 ..."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments/EXP-0059-retrieval-transition-model/code'))
import test_v2
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--models', nargs='+', required=True)
a = ap.parse_args()
test_v2.OCC_MODELS = list(a.models); test_v2.PARTICLE_MODELS = []; test_v2.RES = Path(a.out).resolve()
test_v2.main()
