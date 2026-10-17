"""Adapter-vs-cache parity: (a) batched swept mask == EXP-0053 swept_region; (b) gated_ens_narrow_v2 offline accuracy_1 on DS-0016 == cached-gate pipeline."""
import sys; from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"experiments/temp/ensemble-proto"))
import torch
from ens import TV, acc
from simple_mpc.adapters import make_occ_adapter, GatedEnsembleAdapter, occ_from_particles
d = TV["test"]; act = d["act"]
a = GatedEnsembleAdapter.swept_mask(act[:200].cuda(), "cuda").cpu(); b = d["region"][:200]
print("swept mask pixel mismatches:", int((a != b).sum()), "of", a.numel())
ad = make_occ_adapter("gated_ens_narrow_v2", "cuda")
P = []
with torch.no_grad():
    for i in range(0, len(act), 128): P.append(ad.predict_step(d["prev"][i:i+128].float().cuda(), act[i:i+128].cuda()).cpu())
P = torch.cat(P); print("adapter test acc1:", round(acc(P, d["truth"].float(), d["prev"].float(), d["region"]), 4))
