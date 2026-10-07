"""Cache per-candidate error statistics of each member model (raw and mass-balanced predictions) on the VAL and TEST sets so that hard-gating rules can be scored offline.
Per shard and model: rows -> (num, den) [swept-region squared errors of the prediction / of persistence] one-step; chains (4 pushes) -> (num_k, den_k) per chain; pools -> predicted dv vp (K,13 goals)
and true dv vt. acc = 1 - sqrt(sum num / sum den) (pooled ratio of means, as everywhere); slateN from vp/vt as in eval_wide. usage: gate_cache.py SPLIT(val|test) MODEL(z|w|lf)"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as ew
from eval_wide import *
from posthoc_fix import val_sets, fix_delta, VCF
R0 = "experiments/EXP-0074-wide-domain-zoom-nfd/"
split, mk = sys.argv[1], sys.argv[2]
torch.set_num_threads(3)
if split == "val":
    ew.CV = torch.load(VCF, weights_only=False)
    ew.CVF = VCF
sets = val_sets() if split == "val" else test_sets()
if mk == "z":
    model = Zoom(make_net(R0 + "runs/z128_f8_ms4/unet_best.pth", (8, 16, 32)), 128)
elif mk == "w":
    model = World(make_net(R0 + "runs/w128_f8_ms4/unet_best.pth", (8, 16, 32)), 128)
else:
    from eval_lf_wide import LF
    model = LF()
Dist = {g: torch.from_numpy(dist_field_from_mask(goal_mask(g))).float() for g in GOALS}


def bal(raw, prev):
    return (prev + fix_delta(raw - prev, 0., "balance")).clamp(0, 1)


def lv(x, y):
    return np.stack([(lyap(x, Dist[g]) - lyap(y, Dist[g])).numpy() for g in GOALS], 1)


out = {}
for sh in SHARDS:
    s = sets[sh]
    t = s["t"]
    o = {}
    rows = torch.from_numpy(s["rows"])
    NUM, NUMB, DEN, LS = [], [], [], []
    for i in range(0, len(rows), 128):
        ix = rows[i:i + 128]
        S0, S1, P0, P1 = t["S"][ix], t["S_"][ix], t["P0"][ix], t["P1"][ix]
        raw = model.rollout(S0, P0[:, None], P1[:, None], canv(sh, t, ix))[0]
        O0 = occ_from_particles(S0)
        T = occ_from_particles(S1)
        Rg = swept_region(torch.cat([P0, P1], 1), "cpu").cpu().float()
        b = bal(raw, O0)
        NUM.append((((raw - T) ** 2) * Rg).sum((1, 2)))
        NUMB.append((((b - T) ** 2) * Rg).sum((1, 2)))
        DEN.append((((O0 - T) ** 2) * Rg).sum((1, 2)))
        LS.append((P1 - P0).norm(dim=1) * 1000)
    o["rows"] = dict(num=torch.cat(NUM).numpy(), numb=torch.cat(NUMB).numpy(), den=torch.cat(DEN).numpy(), L=torch.cat(LS).numpy())
    ch = torch.tensor(s["chains"])
    NC, NCB, DC, LC = [], [], [], []
    for i in range(0, len(ch), 128):
        c = ch[i:i + 128]
        rawl = model.rollout(t["S"][c[:, 0]], t["P0"][c], t["P1"][c], canv(sh, t, c[:, 0]))
        O0 = occ_from_particles(t["S"][c[:, 0]])
        reg = torch.zeros(len(c), 64, 64, dtype=torch.bool)
        prev_raw = prev = O0
        nn, nb, dd = [], [], []
        for k in range(4):
            reg = reg | swept_region(torch.cat([t["P0"][c[:, k]], t["P1"][c[:, k]]], 1), "cpu").cpu()
            Tk = occ_from_particles(t["S_"][c[:, k]])
            r = reg.float()
            prev = (prev + fix_delta(rawl[k] - prev_raw, 0., "balance")).clamp(0, 1)
            prev_raw = rawl[k]
            nn.append((((rawl[k] - Tk) ** 2) * r).sum((1, 2)))
            nb.append((((prev - Tk) ** 2) * r).sum((1, 2)))
            dd.append((((O0 - Tk) ** 2) * r).sum((1, 2)))
        NC.append(torch.stack(nn, 1))
        NCB.append(torch.stack(nb, 1))
        DC.append(torch.stack(dd, 1))
        LC.append((t["P1"][c] - t["P0"][c]).norm(dim=-1) * 1000)
    o["chains"] = dict(num=torch.cat(NC).numpy(), numb=torch.cat(NCB).numpy(), den=torch.cat(DC).numpy(), L=torch.cat(LC).numpy())
    pools = []
    for pidx in s["pools"]:
        ix = torch.from_numpy(pidx)
        S0 = t["S"][ix]
        o0 = occ_from_particles(S0[:1])
        raw = model.rollout(S0, t["P0"][ix][:, None], t["P1"][ix][:, None], canv(sh, t, ix[:1]).expand(len(ix), -1, -1))[0]
        b = bal(raw, o0.expand(len(ix), -1, -1))
        tr = occ_for_scoring(t["S_"][ix][:, :, :3])
        t0 = occ_for_scoring(S0[:1][:, :, :3])
        pools.append(dict(vp=lv(raw, o0.expand(len(ix), -1, -1)), vpb=lv(b, o0.expand(len(ix), -1, -1)), vt=lv(tr, t0.expand(len(ix), -1, -1)),
                          L=((t["P1"][ix] - t["P0"][ix]).norm(dim=1) * 1000).numpy()))
    o["pools"] = pools
    out[sh] = o
    print(split, mk, sh, "done", flush=True)
torch.save(out, R0 + f"artifacts/gate_{split}_{mk}.pt")
