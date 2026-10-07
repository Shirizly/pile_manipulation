"""Ensembles and length-routed specialists, scored by the eval_wide harness (pasted 64x64 frame).
  ens    : average of the members' pasted predictions at every step (each member keeps its own internal state/canvas)
  routed : each candidate push is predicted by the member for its push-length bin (edges in mm); one-step only (no rollouts)
spec JSON: {"type": "ens"|"routed", "edges": [35, 55], "members": [["zoom"|"world", RES, CKPT, "8,16,32"], ...]}   usage: eval_variants.py --name X --spec '{...}' [--shards a,b] [--rolls 0]"""
import sys, json, argparse, os, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as ew


def build(kind, res, ck, feats, thr=0.2):
    n = ew.make_net(ck, tuple(int(x) for x in feats.split(","))); return ew.Zoom(n, res, thr=thr) if kind == "zoom" else ew.World(n, res)


class Ens:
    def __init__(self, ms, w=None): self.ms = ms; self.w = torch.tensor(w if w else [1.0] * len(ms)).float(); self.w = self.w / self.w.sum()
    def rollout(self, S0, P0s, P1s, canvas0=None):
        outs = [m.rollout(S0, P0s, P1s, canvas0) for m in self.ms]; return [sum(self.w[i] * outs[i][k] for i in range(len(outs))) for k in range(len(outs[0]))]


class Routed:
    def __init__(self, ms, edges): self.ms, self.edges = ms, [e / 1000 for e in edges]
    def rollout(self, S0, P0s, P1s, canvas0=None):
        assert P0s.shape[1] == 1, "routed models are one-step only"
        L = (P1s[:, 0] - P0s[:, 0]).norm(dim=-1); b = torch.bucketize(L, torch.tensor(self.edges), right=True); out = torch.zeros(len(S0), 64, 64)
        for i, m in enumerate(self.ms):
            ix = torch.nonzero(b == i)[:, 0]
            if len(ix): out[ix] = m.rollout(S0[ix], P0s[ix], P1s[ix], None if canvas0 is None else canvas0[ix])[0]
        return [out]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--shards", default=""); ap.add_argument("--rolls", type=int, default=1); a = ap.parse_args()
    sp = json.loads(a.spec); ms = [build(*m) for m in sp["members"]]; model = Ens(ms, sp.get("weights")) if sp["type"] == "ens" else Routed(ms, sp["edges"])
    if sp.get("fix"):
        from posthoc_fix import Fixed; model = Fixed(model, 0., sp["fix"])   # val-selected plain mass balance of each step's predicted change, applied to the (ensemble) output
    sets = ew.test_sets(); r = ew.run(model, sets, shards=a.shards.split(",") if a.shards else None, rolls=bool(a.rolls))
    json.dump(r, open(f"experiments/EXP-0074-wide-domain-zoom-nfd/results/eval_{a.name}.json", "w"), indent=1); print("MEAN", {k: round(v, 3) for k, v in r["MEAN"].items()})
