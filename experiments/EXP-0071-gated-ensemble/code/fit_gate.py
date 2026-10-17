"""Fit the MODEL-0015 gate: members = _GATE_MEMBERS (occupancy-only), training rows = DS-0017 val_pools_v2 (flagged rows
dropped), loss = swept-region(+0.05 floor) MSE of the blend vs the soft... hard-raster truth, 60 epochs Adam 2e-3, seed 0.
Reads the member-prediction cache built by experiments/temp/ensemble-proto/collect.py (regenerable)."""
import sys, os, torch, torch.nn as nn
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"experiments/temp/ensemble-proto"))
from ens import TV, M
from simple_mpc.adapters import _GATE_MEMBERS as ms
SEED = 0; HID = 24; EPOCHS = 60; dv = "cuda"
torch.manual_seed(SEED); d = TV["val"]
Xv = torch.cat([d["prev"].float()[:,None], torch.stack([M[m]["val"].float() for m in ms],1), d["region"].float()[:,None]],1)
Mv = Xv[:,1:1+len(ms)]; T = d["truth"].float(); R = d["region"].float()
net = nn.Sequential(nn.Conv2d(Xv.shape[1],HID,5,padding=2),nn.ReLU(),nn.Conv2d(HID,HID,5,padding=2),nn.ReLU(),nn.Conv2d(HID,len(ms),3,padding=1)).to(dv)
opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
for ep in range(EPOCHS):
    perm = torch.randperm(len(Xv))
    for i in range(0, len(Xv), 64):
        ix = perm[i:i+64]; p = (torch.softmax(net(Xv[ix].to(dv)),1)*Mv[ix].to(dv)).sum(1); r = R[ix].to(dv)+0.05
        loss = (((p-T[ix].to(dv))**2)*r).sum()/r.sum(); opt.zero_grad(); loss.backward(); opt.step()
out = REPO/"weights/MODEL-0015-gated-ensemble-narrow-v2/checkpoint.pt"
torch.save(dict(members=ms, hid=HID, state_dict={k: v.cpu() for k, v in net.state_dict().items()}, seed=SEED, epochs=EPOCHS), str(out)+".tmp"); os.replace(str(out)+".tmp", out)
print("saved", out, "final loss", float(loss))
