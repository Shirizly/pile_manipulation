"""EXP-0068 state selection (CPU, deterministic; rule pre-registered in DESIGN.md).

Reads the RUN-0001 closed-loop results (driver JSON with `states`, `quats`, `topk_actions`) and writes
datasets/DS-0023-visited-state-pools/data/starts.pt: 48 visited states = 24 cells (8 letters x k in {4, 8, 12}) x 2,
cell c = 3 * goal_index + k_index; picks = models (c mod 8) and ((c + 4) mod 8) in ZOO order, at starts 40 / 48 (c even)
or 48 / 40 (c odd). Each model contributes 6 states, each start 24, no episode is used twice.
Usage: python -u select_states.py --episodes <driver json> [--n-per-cell 2]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.cube_overlap import state_is_legal  # noqa: E402
from model.retrieval.frame import yaw_from_quat  # noqa: E402

ZOO = ["nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2",
       "nfd_3ch_narrow_l20_v2_soft_s2", "nfd_3ch_narrow_l20_v2_epoch10", "linear_narrow_l20_v2_res64",
       "linear_switched_soft", "nfd_3ch_randlen"]
LETTERS = ["letter_O", "letter_T", "letter_S", "letter_L", "letter_X", "letter_Z", "letter_C", "letter_H"]
KS = [4, 8, 12]
STARTS = [40, 48]
OUT = REPO / "datasets/DS-0023-visited-state-pools/data/starts.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--n-per-cell", type=int, default=2, help="2 = full design; 1 = the (c mod 8) pick only (fallback)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--starts", type=int, nargs="+", default=STARTS, help="the closed-loop starts available (DESIGN: 40 48; cut run: 40)")
    a = ap.parse_args()
    ST = list(a.starts)
    eps = {}
    for e in json.loads(Path(a.episodes).read_text())["episodes"]:
        if e.get("complete"):
            eps[(e["model"], e["goal"], e["start"])] = e
    states, quats, topk, topk_dv, meta = [], [], [], [], []
    for gi, g in enumerate(LETTERS):
        for ki, k in enumerate(KS):
            c = 3 * gi + ki
            picks = [(ZOO[c % 8], ST[c % len(ST)]), (ZOO[(c + 4) % 8], ST[(c + 1) % len(ST)])][:a.n_per_cell]
            for pi, (m, s) in enumerate(picks):
                e = eps[(m, g, s)]
                assert len(e["states"]) > k and len(e["quats"]) > k and len(e["topk_actions"]) > k
                st = torch.tensor(e["states"][k], dtype=torch.float32); q = torch.tensor(e["quats"][k], dtype=torch.float32)
                states.append(torch.cat([st, q], 1)); quats.append(q)
                topk.append(torch.tensor(e["topk_actions"][k], dtype=torch.float32)); topk_dv.append(torch.tensor(e["topk_dv"][k]))
                xy = st[:, :2].numpy(); yaw = yaw_from_quat(q).numpy()
                meta.append(dict(pool=len(meta), cell=c, pick=pi, model=m, goal=g, start=s, k=k,
                                 n_layer1=int((st[:, 2] > 0.015).sum()), legal_state=bool(state_is_legal(xy, yaw, 0.005, 0.0)),
                                 n_outside=int((xy.__abs__() > 0.0635).any(1).sum()),
                                 in_goal_before=None, pred_dv_executed=float(e["pred_dv"][k]) if k < len(e["pred_dv"]) else None,
                                 true_dv_executed=float(e["true_dv"][k]) if k < len(e["true_dv"]) else None))
    S = torch.stack(states); T = torch.stack(topk); TD = torch.stack(topk_dv)
    out = dict(states=S, topk_actions=T, topk_dv=TD, meta=meta, zoo=ZOO, letters=LETTERS, ks=KS, starts=ST, n_per_cell=a.n_per_cell,
               source=str(a.episodes), rule=__doc__.strip())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); torch.save(out, a.out)
    from collections import Counter
    print("states", tuple(S.shape), "topk", tuple(T.shape))
    print("per model", Counter(m["model"] for m in meta)); print("per start", Counter(m["start"] for m in meta))
    print("per goal", Counter(m["goal"] for m in meta)); print("per k", Counter(m["k"] for m in meta))
    print("layer-1 cubes per state:", [m["n_layer1"] for m in meta])
    print("illegal (overlapping) states:", sum(not m["legal_state"] for m in meta), "; states with cubes outside +-63.5 mm:",
          sum(m["n_outside"] > 0 for m in meta))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
