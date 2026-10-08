"""EXP-0074 addendum: LF-zoom (zoom-window switched-linear foresight, lf_zoom.py / runs/lf_zoom) on the wide test sets (eval_wide harness, pasted 64x64 frame), and the ensemble
   ens2  = z128_f8_ms4 + w128_f8_ms4 (mean) + mass balance  (the top NFD ensemble)
   ens3g = ens2 + LF-zoom as a third member, HARD-GATED by push-length bin: LF-zoom enters the mean only for the bins where adding it improved val one-step acc1 (gate chosen on the VAL rows
           of lf_split, 600/shard; then mass balance on the mean), outside the gated bins the prediction is ens2.
   also reports the per-bin comparison (acc1 by LF bin), the 128-px output sanity check.   usage: eval_lf_zoom.py"""
import sys, json, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as ew
from eval_variants import Ens
from posthoc_fix import Fixed, fix_delta
from lf_split import load_split
from sean_data import rows_table
from lf_zoom import LFZoom, EDGES, delta_to_world_cc
from simple_mpc.adapters import occ_from_particles
R4 = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/"; RES = "experiments/EXP-0074-wide-domain-zoom-nfd/results/"
TAG = sys.argv[1] if len(sys.argv) > 1 else "lf_zoom"; lf = LFZoom(R4 + f"{TAG}/operators.pt"); z = ew.Zoom(ew.make_net(R4 + "z128_f8_ms4/unet_best.pth", (8, 16, 32)), 128); w = ew.World(ew.make_net(R4 + "w128_f8_ms4/unet_best.pth", (8, 16, 32)), 128)
ens2 = Fixed(Ens([z, w]), 0., "balance")


class Gated3:
    """mean of (z, w, lf-if-gated) per sample and step, then mass balance."""
    def __init__(self, gate): self.gate = torch.tensor(gate, dtype=torch.float32)
    def rollout(self, S0, P0s, P1s, canvas0=None):
        oz, ow, ol = z.rollout(S0, P0s, P1s, canvas0), w.rollout(S0, P0s, P1s, canvas0), lf.rollout(S0, P0s, P1s, canvas0); raw = []
        for k in range(len(oz)):
            g = self.gate[torch.bucketize((P1s[:, k] - P0s[:, k]).norm(dim=-1), EDGES[1:-1])][:, None, None]; raw.append((oz[k] + ow[k] + g * ol[k]) / (2 + g))
        prev_raw = prev = occ_from_particles(S0); out = []
        for r in raw:
            prev = (prev + fix_delta(r - prev_raw, 0., "balance")).clamp(0, 1); prev_raw = r; out.append(prev)
        return out


# ---- val rows (lf_split val), per-LF-bin one-step acc1 of ens2 vs ens3 (all bins) vs LF alone
rng = np.random.default_rng(1); V = load_split("val", ew.SHARDS); P = {k: [] for k in ("ens2", "ens3", "lf")}; T, O, Rg, Ls = [], [], [], []
for sh in ew.SHARDS:
    t = rows_table(V[sh]); r = np.sort(rng.permutation(len(t["S"]))[:600])
    for i in range(0, 600, 150):
        ix = torch.from_numpy(r[i:i + 150]); S0, S1, P0, P1 = t["S"][ix], t["S_"][ix], t["P0"][ix], t["P1"][ix]; a, b = P0[:, None], P1[:, None]
        P["ens2"].append(ens2.rollout(S0, a, b)[0]); P["ens3"].append(Gated3([1.0] * 6).rollout(S0, a, b)[0]); P["lf"].append(lf.rollout(S0, a, b)[0]); T.append(occ_from_particles(S1)); O.append(occ_from_particles(S0))
        Rg.append(ew.swept_region(torch.cat([P0, P1], 1), "cpu").cpu()); Ls.append((P1 - P0).norm(dim=1))
P = {k: torch.cat(v) for k, v in P.items()}; T, O, Rg, Ls = map(torch.cat, (T, O, Rg, Ls)); bi = torch.bucketize(Ls, EDGES[1:-1]); gate = []; perbin = {}
for k in range(6):
    m = bi == k; perbin[k] = {n: ew.acc(P[n][m], T[m], O[m], Rg[m]) for n in P} | {"n": int(m.sum())}; gate.append(1.0 if perbin[k]["ens3"] > perbin[k]["ens2"] else 0.0)
print("VAL per LF bin (acc1):", {k: {n: round(v, 3) if n != "n" else v for n, v in d.items()} for k, d in perbin.items()}, "gate", gate, flush=True)
json.dump(dict(val_per_bin=perbin, gate=gate), open(RES + f"{TAG}_gate_val.json", "w"), indent=1)
sets = ew.test_sets(); out = {}
for name, model in ((TAG, lf), ("ens2_balance", ens2), (f"ens3_{TAG}_gated", Gated3(gate))):
    r = ew.run(model, sets); json.dump(r, open(RES + f"eval_{name}.json", "w"), indent=1); out[name] = r["MEAN"]; print(name, "MEAN", {k: round(v, 3) for k, v in r["MEAN"].items()}, flush=True)
# 128-px output check: pasted 128 frame vs the 64 pasted frame (area-downsampled), one-step on a test shard
t = sets["scattered_n50"]["t"]; ix = torch.from_numpy(sets["scattered_n50"]["rows"][:64]); S0, P0, P1 = t["S"][ix], t["P0"][ix][:, None], t["P1"][ix][:, None]
o128 = lf.rollout128(S0, P0, P1)[0]; o64 = lf.rollout(S0, P0, P1)[0]; print("128 output", tuple(o128.shape), "mean mass change 128:", float((o128.sum((1, 2)) - 0).mean()), "64:", float(o64.sum((1, 2)).mean()))
json.dump(out, open(RES + f"{TAG}_summary.json", "w"), indent=1)
