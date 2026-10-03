"""EXP-0062 RUN-0003 -- DS-0020 v2 val swept-region mask accuracy of a GNN checkpoint through the
test-time predictor (eval_report spec gnn_flex_v2_n30 with a ckpt override; same perception,
renderer and scorer as the DS-0019 test). Used to choose the training target and to track training.
    python -u experiments/EXP-0062-flex-v2-train-rerun/code/gnn_val_acc.py CKPT [CKPT ...] --out results/gnn_target_choice.json
"""
import argparse, json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); os.chdir(REPO)
import torch
from Baselines.common import eval_report as er
from FlexData.dataset import load_flex_cell

ap = argparse.ArgumentParser(); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
vcell = load_flex_cell("datasets/DS-0020-training-data-flex-N864/config.yaml", "val", tag="ds0020v2_val_mask", occ_source="image_mask")
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
for ck in a.ckpts:
    t = time.time()
    spec = dict(er.MODELS["gnn_flex_v2_n30"], ckpt=ck)
    spec["kwargs"] = dict(spec["kwargs"], device=a.device, fast_graph=True, vector_render=True)
    acc, _ = er._accuracy(spec, er._load_predictor(spec), vcell, a.device)
    res[ck] = dict(val_accuracy=acc, n_rows=len(vcell.occ0), device=a.device, seconds=time.time() - t,
                   mtime=time.ctime(os.path.getmtime(ck)))
    print(ck, f"val acc {acc:.4f} ({time.time() - t:.0f}s)", flush=True)
    json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
