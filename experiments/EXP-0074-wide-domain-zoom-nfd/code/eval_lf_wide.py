"""Switched-linear foresight (Suh & Tedrake; Baselines/LinearForesight) on the WIDE Sean corpus, EXP-0074 harness.
 LF            wide-fit switched operators (runs/lf_wide/operators.pt, from code/fit_lf_wide.py), hard length gate (the fitted model)
 NarrowAdapter  zero-shot `linear_narrow_l20_v2_res64` via simple_mpc.adapters.make_occ_adapter(...).predict_step
 Persist        persistence (acc1 = 0 by construction)
Both expose rollout(S0,P0s,P1s,canvas0=None) -> list of T (B,64,64) CPU tensors (own prediction fed back, no clamp: as the fitted model).
Usage: python eval_lf_wide.py [--sanity] [--which wide|narrow|persist|all]"""
import sys, os, json, argparse, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as ew
from simple_mpc.adapters import occ_from_particles, OCC_BOUNDS
from Baselines.LinearForesight.model import predict_switched, push_length_m
from fit_linear_foresight import actions_to_pixels
dev = "cuda" if torch.cuda.is_available() else "cpu"
WS = (torch.tensor([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]]), torch.tensor([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]]))
CK = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/lf_wide/operators.pt"


class LF:
    def __init__(self, ck=CK, device=dev):
        c = torch.load(ck, map_location="cpu", weights_only=False) if isinstance(ck, str) else ck
        self.dev = device; self.res = c["res"]; self.edges = c["bin_edges"].to(device); self.ops = [o.to(device).float() for o in c["operators"]]
    @torch.no_grad()
    def step(self, occ, act):
        s, e = actions_to_pixels(act, WS[0], WS[1], tuple(occ.shape[-2:]))
        return predict_switched(self.edges, self.ops, occ, s, e, push_length_m(act), self.res, tuple(occ.shape[-2:]))
    @torch.no_grad()
    def rollout(self, S0, P0s, P1s, canvas0=None):
        cur = occ_from_particles(S0).to(self.dev); out = []
        for k in range(P0s.shape[1]):
            act = torch.cat([P0s[:, k], P1s[:, k]], 1).float().to(self.dev); cur = self.step(cur, act); out.append(cur.cpu())
        return out


class NarrowAdapter:
    def __init__(self, mid="linear_narrow_l20_v2_res64", device=dev):
        from simple_mpc.adapters import make_occ_adapter
        self.ad = make_occ_adapter(mid, device); self.dev = device
    @torch.no_grad()
    def rollout(self, S0, P0s, P1s, canvas0=None):
        cur = occ_from_particles(S0).to(self.dev); out = []
        for k in range(P0s.shape[1]):
            act = torch.cat([P0s[:, k], P1s[:, k]], 1).float().to(self.dev); cur = self.ad.predict_step(cur, act); out.append(cur.cpu())
        return out


class Persist:
    def rollout(self, S0, P0s, P1s, canvas0=None):
        o = occ_from_particles(S0).cpu(); return [o.clone() for _ in range(P0s.shape[1])]


def save(r, name):
    json.dump(r, open(f"experiments/EXP-0074-wide-domain-zoom-nfd/results/eval_{name}.json", "w"), indent=1)
    print(name, "MEAN", {k: round(v, 3) for k, v in r["MEAN"].items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="all"); ap.add_argument("--sanity", action="store_true"); a = ap.parse_args()
    sets = ew.test_sets()
    if a.sanity:   # LF.rollout vs the narrow adapter's formulas (same ops file loaded through SwitchedLinearGradientAdapter, hard gate)
        from simple_mpc.adapters import SwitchedLinearGradientAdapter
        c = torch.load(CK, map_location="cpu", weights_only=False); ad = SwitchedLinearGradientAdapter("x", c, dev, "corner", res=c["res"], gate="hard"); m = LF()
        t = sets["scattered_n50"]["t"]; ix = torch.arange(24)
        o1 = m.rollout(t["S"][ix], t["P0"][ix][:, None], t["P1"][ix][:, None])[0]
        o2 = ad.predict_step(occ_from_particles(t["S"][ix]).to(dev), torch.cat([t["P0"][ix], t["P1"][ix]], 1).float().to(dev)).cpu()
        print("max |LF - adapter.predict_step| =", float((o1 - o2).abs().max()), " vs persistence |d|", float((o1 - occ_from_particles(t["S"][ix])).abs().max()))
        # 2-step rollout self-consistency vs two manual predict_step calls
        ch = sets["scattered_n50"]["chains"][:8]; ch = torch.tensor(ch)
        r = m.rollout(t["S"][ch[:, 0]], t["P0"][ch], t["P1"][ch])
        cur = occ_from_particles(t["S"][ch[:, 0]]).to(dev)
        for k in range(2): cur = ad.predict_step(cur, torch.cat([t["P0"][ch[:, k]], t["P1"][ch[:, k]]], 1).float().to(dev))
        print("2-step max diff", float((r[1] - cur.cpu()).abs().max())); sys.exit()
    if a.which in ("persist", "all"): save(ew.run(Persist(), sets), "persistence_wide")
    if a.which in ("wide", "all"): save(ew.run(LF(), sets), "lf_wide")
    if a.which in ("narrow", "all"): save(ew.run(NarrowAdapter(), sets), "lf_narrow_zeroshot")
